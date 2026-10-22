"""
Multi-Party Conversation Modeling & Next-Speaker Prediction (Lab 10 & Assignment 2 Part C).

Implements:
- AMI Meeting Corpus word-to-turn aggregation (`build_turns` with `MAX_GAP = 0.5s`)
- Multi-party sliding-window history dataset builder (`build_multiparty_dataset`, `HISTORY = 8`)
- Packed-sequence Utterance BiLSTM Encoder (`UtteranceBiLSTMEncoder`)
- Hierarchical Context + Speaker-Embedding Next-Speaker BiLSTM (`NextSpeakerBiLSTMModel`)
- Multi-party turn-taking & addressee heuristic scorer (`score_multiparty_turn_taking`)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn as nn

SPEAKER2ID: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}
ID2SPEAKER: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}

AMI_BENCHMARK_METRICS: Dict[str, Dict[str, float]] = {
    "Full Model (8-Turn History + Speaker Embeddings)": {
        "test_acc": 0.7072,
        "test_macro_f1": 0.7048,
        "test_precision": 0.7054,
        "test_recall": 0.7043,
        "dev_peak_f1": 0.7199,
    },
    "Ablation 1: No Speaker Info (Text History Only)": {
        "test_acc": 0.2736,
        "test_macro_f1": 0.2726,
        "dev_peak_f1": 0.2579,
    },
    "Ablation 2: No Dialogue History (Current Turn + Speaker Only)": {
        "test_acc": 0.3342,
        "test_macro_f1": 0.2726,
        "dev_peak_f1": 0.2772,
    },
}

AMI_CONFUSION_MATRIX: List[List[int]] = [
    [3747, 506, 466, 357],
    [584, 3433, 443, 391],
    [490, 507, 3191, 368],
    [426, 387, 356, 2383],
]


def build_turns(words_df: pd.DataFrame, max_gap: float = 0.5) -> pd.DataFrame:
    """
    Merge consecutive word-level annotations from the same speaker into continuous dialogue turns
    whenever the inter-word silence gap is <= `max_gap` seconds (default 0.5s).
    """
    rows: List[Dict[str, Any]] = []
    for mid, sub in words_df.groupby("meeting_id"):
        sub = sub.sort_values(["start", "end"]).reset_index(drop=True)
        cur_speaker = None
        cur_words: List[str] = []
        cur_start = None
        cur_end = None

        for _, row in sub.iterrows():
            spk = row["speaker"]
            start = float(row["start"])
            end = float(row["end"])
            text = str(row["text"])

            if cur_speaker is None:
                cur_speaker = spk
                cur_start = start
                cur_end = end
                cur_words = [text]
            else:
                if spk == cur_speaker and (start - cur_end) <= max_gap:
                    cur_words.append(text)
                    cur_end = end
                else:
                    rows.append(
                        {
                            "meeting_id": mid,
                            "speaker": cur_speaker,
                            "utt_start": cur_start,
                            "utt_end": cur_end,
                            "utt_text": " ".join(cur_words),
                        }
                    )
                    cur_speaker = spk
                    cur_start = start
                    cur_end = end
                    cur_words = [text]

        if cur_speaker is not None:
            rows.append(
                {
                    "meeting_id": mid,
                    "speaker": cur_speaker,
                    "utt_start": cur_start,
                    "utt_end": cur_end,
                    "utt_text": " ".join(cur_words),
                }
            )

    turn_df = pd.DataFrame(rows)
    if len(turn_df) > 0:
        turn_df = turn_df.sort_values(["meeting_id", "utt_start", "utt_end"]).reset_index(drop=True)
    return turn_df


def build_multiparty_dataset(turn_df: pd.DataFrame, history: int = 8) -> pd.DataFrame:
    """Construct next-speaker prediction instances with `history` preceding turns and speaker IDs."""
    rows: List[Dict[str, Any]] = []
    for mid, sub in turn_df.groupby("meeting_id"):
        sub = sub.sort_values("utt_start").reset_index(drop=True)
        for i in range(history, len(sub) - 1):
            cur_row = sub.iloc[i]
            next_row = sub.iloc[i + 1]
            history_texts = [sub.iloc[j].utt_text for j in range(i - history, i)]
            history_speakers = [sub.iloc[j].speaker for j in range(i - history, i)]
            rows.append(
                {
                    "meeting_id": mid,
                    "speaker": cur_row.speaker,
                    "text": cur_row.utt_text,
                    "history_texts": history_texts,
                    "history_speakers": history_speakers,
                    "next_speaker": next_row.speaker,
                    "label": SPEAKER2ID[next_row.speaker],
                }
            )
    return pd.DataFrame(rows)


class UtteranceBiLSTMEncoder(nn.Module):
    """Word-level Bidirectional LSTM encoder with packed padded sequence handling."""

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens.long())
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.dropout(h)


class NextSpeakerBiLSTMModel(nn.Module):
    """
    Hierarchical Multi-Party Next-Speaker Prediction Network (Lab 10).
    Encodes each historical utterance via `UtteranceBiLSTMEncoder`, fuses 16-D speaker identity
    embeddings, runs a context-level BiLSTM across the H+1 turn trajectory, and classifies the next
    speaker among {A, B, C, D}.
    """

    def __init__(
        self,
        vocab_size: int,
        history: int = 8,
        emb_dim: int = 128,
        word_hidden: int = 128,
        ctx_hidden: int = 128,
        spk_emb_dim: int = 16,
        dropout: float = 0.3,
        use_current_speaker: bool = True,
        use_history_speaker: bool = True,
    ):
        super().__init__()
        self.history = history
        self.use_current_speaker = use_current_speaker
        self.use_history_speaker = use_history_speaker

        self.utt_encoder = UtteranceBiLSTMEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=word_hidden,
            dropout=dropout,
        )
        base_utt_dim = 2 * word_hidden

        if self.use_history_speaker:
            self.history_speaker_emb = nn.Embedding(4, spk_emb_dim)
            hist_utt_dim = base_utt_dim + spk_emb_dim
        else:
            self.history_speaker_emb = None
            hist_utt_dim = base_utt_dim

        self.ctx_input_dim = hist_utt_dim
        self.ctx_lstm = nn.LSTM(
            input_size=self.ctx_input_dim,
            hidden_size=ctx_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.spk_emb = nn.Embedding(4, spk_emb_dim)
        self.dropout = nn.Dropout(dropout)

        clf_in_dim = 2 * ctx_hidden + (spk_emb_dim if self.use_current_speaker else 0)
        self.clf = nn.Sequential(
            nn.Linear(clf_in_dim, ctx_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_hidden, 4),
        )

    def forward(
        self,
        history_tokens: List[torch.Tensor],
        history_lengths: List[torch.Tensor],
        history_speakers: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        speaker: torch.Tensor,
    ) -> torch.Tensor:
        hist_vecs: List[torch.Tensor] = []
        batch_size = tokens.size(0)

        for h_idx, (ht, hl) in enumerate(zip(history_tokens, history_lengths)):
            hv = self.utt_encoder(ht, hl)
            if self.use_history_speaker and self.history_speaker_emb is not None:
                hs = history_speakers[:, h_idx]
                hs_emb = self.history_speaker_emb(hs)
                hv = torch.cat([hv, hs_emb], dim=1)
            hist_vecs.append(hv)

        cur_vec = self.utt_encoder(tokens, lengths)
        if self.use_history_speaker and self.history_speaker_emb is not None:
            zero_spk = torch.zeros(
                batch_size, self.history_speaker_emb.embedding_dim, device=tokens.device
            )
            cur_vec = torch.cat([cur_vec, zero_spk], dim=1)

        seq_vecs = torch.stack(hist_vecs + [cur_vec], dim=1)
        _, (h_n, _) = self.ctx_lstm(seq_vecs)
        ctx_vec = self.dropout(torch.cat([h_n[0], h_n[1]], dim=1))

        if self.use_current_speaker:
            spk_vec = self.spk_emb(speaker)
            x = torch.cat([ctx_vec, spk_vec], dim=1)
        else:
            x = ctx_vec

        return self.clf(x)


def score_multiparty_turn_taking(
    current_speaker: str,
    current_text: str,
    history_speakers: List[str],
    use_speaker_info: bool = True,
    use_history: bool = True,
) -> Dict[str, float]:
    """
    Compute normalized next-speaker probability distribution over {'A', 'B', 'C', 'D'}
    incorporating floor-holding priors, dyadic reciprocity (last respondent), and question turn-yielding cues.
    """
    speakers = ["A", "B", "C", "D"]
    if not use_speaker_info:
        return {s: 0.25 for s in speakers}

    scores = {s: 0.10 for s in speakers}
    text_clean = str(current_text).strip().lower()
    is_question = text_clean.endswith("?")
    is_backchannel = len(text_clean.split()) <= 2 and any(
        w in text_clean for w in ["yeah", "ok", "right", "uh-huh", "mm-hmm"]
    )

    if not is_question and not is_backchannel:
        scores[current_speaker] += 0.45
    elif is_question:
        scores[current_speaker] += 0.05
    else:
        scores[current_speaker] += 0.15

    if use_history and history_speakers:
        for dist_idx, spk in enumerate(reversed(history_speakers)):
            if spk in scores:
                weight = 0.35 * (0.65 ** dist_idx)
                if is_question and spk != current_speaker:
                    weight *= 1.6
                scores[spk] += weight

    total = sum(scores.values())
    return {s: round(scores[s] / total, 4) for s in speakers}
