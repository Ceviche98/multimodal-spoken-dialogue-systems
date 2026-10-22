"""
Dialogue Act Tagging & Utterance Intent Classification (Lab 2).

Implements:
- Switchboard DAMSL 43-tag canonical simplifier (`damsl_act_tag`)
- Utterance sequence padding (`pad_sequences`)
- Stacked Bidirectional LSTM Utterance Classifier (`BiLSTMModel`) with mean-pooling over time
- Parallel Spatial-Temporal Hierarchical Classifier (`CNN_BiLSTM`) combining multi-scale Conv2D (k=3,4,5)
  utterance n-gram features with stacked BiLSTM contextual representations
- Feature extraction and heuristic dialogue act classification for lightweight inference
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

SWDA_BENCHMARK_METRICS: Dict[str, Dict[str, float]] = {
    "Model 1: BiLSTM (Unweighted, 5 Epochs)": {
        "overall_acc": 0.6796,
        "test_loss": 1.0683,
        "br_acc": 0.0000,
        "bf_acc": 0.0000,
    },
    "Model 2: BiLSTM (Class-Balanced, 5 Epochs)": {
        "overall_acc": 0.3483,
        "test_loss": 2.2324,
        "br_acc": 0.5472,
        "bf_acc": 0.0280,
    },
    "Model 2b: BiLSTM (Class-Balanced, 20 Epochs)": {
        "overall_acc": 0.3701,
        "test_loss": 1.9764,
        "br_acc": 0.5472,
        "bf_acc": 0.1240,
    },
    "Model 3: Parallel CNN + BiLSTM Context (5 Epochs)": {
        "overall_acc": 0.7063,
        "test_loss": 0.9414,
        "br_acc": 0.3774,
        "bf_acc": 0.0000,
    },
}


def damsl_act_tag(input_tag: str) -> str:
    """
    Simplify Switchboard combination DAMSL tags into the canonical 43 generic Dialogue Act tags
    following Jurafsky et al. (1997) / Potts SWDA Coders' Manual.
    """
    d_tags: List[str] = []
    tags = re.split(r"\s*[,;]\s*", str(input_tag).strip())
    for tag in tags:
        if tag in ("qy^d", "qw^d", "b^m"):
            pass
        elif tag == "nn^e":
            tag = "ng"
        elif tag == "ny^e":
            tag = "na"
        else:
            tag = re.sub(r"(.)\^.*", r"\1", tag)
            tag = re.sub(r"[\(\)@*]", "", tag)
            if tag in ("qr", "qy"):
                tag = "qy"
            elif tag in ("fe", "ba"):
                tag = "ba"
            elif tag in ("oo", "co", "cc"):
                tag = "oo_co_cc"
            elif tag in ("fx", "sv"):
                tag = "sv"
            elif tag in ("aap", "am"):
                tag = "aap_am"
            elif tag in ("arp", "nd"):
                tag = "arp_nd"
            elif tag in ("fo", "o", "fw", '"', "by", "bc"):
                tag = 'fo_o_fw_"_by_bc'
        d_tags.append(tag)
    return d_tags[0]


def pad_sequences(sequences: List[List[int]], maxlen: int = 150) -> List[List[int]]:
    """Pad or truncate integer token sequences to fixed length `maxlen`."""
    return [seq[:maxlen] + [0] * max(0, maxlen - len(seq)) for seq in sequences]


class BiLSTMModel(nn.Module):
    """
    Two-layer Bidirectional LSTM Dialogue Act Tagger with temporal mean pooling (Lab 2 Model 1 & 2).
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, max_length: int = 150):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.blstm1 = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x.long())
        lstm_out1, _ = self.blstm1(embedded)
        lstm_out2, _ = self.blstm2(lstm_out1)
        output = torch.mean(lstm_out2, dim=1)
        logits = self.fc(output)
        return logits


class CNN_BiLSTM(nn.Module):
    """
    Hierarchical Dual-Path CNN + BiLSTM Dialogue Act Classifier (Lab 2 Model 3).
    Path A extracts spatial n-gram features via parallel Conv2d blocks (filter_sizes=[3, 4, 5]).
    Path B extracts sequential temporal features via stacked BiLSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=num_filters,
                        kernel_size=(fs, embed_size),
                        stride=1,
                        padding=0,
                    ),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),
                )
                for fs in filter_sizes
            ]
        )
        self.dense_1 = nn.Linear(num_filters * len(filter_sizes), embed_size)
        self.dropout_1 = nn.Dropout(dropout)

        self.blstm1 = nn.LSTM(
            input_size=embed_size,
            hidden_size=embed_size,
            batch_first=True,
            bidirectional=True,
        )
        self.blstm2 = nn.LSTM(
            input_size=2 * embed_size,
            hidden_size=embed_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dense_2 = nn.Linear(2 * embed_size, embed_size)
        self.dropout_2 = nn.Dropout(dropout)
        self.output = nn.Linear(embed_size + embed_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x.long())
        cnn_in = embedded.unsqueeze(1)

        pooled_outputs = []
        for conv in self.conv_blocks:
            conv_out = conv(cnn_in)
            pooled = F.max_pool2d(conv_out, kernel_size=(conv_out.shape[2], conv_out.shape[3]))
            pooled_outputs.append(pooled)

        cat_cnn = torch.cat(pooled_outputs, dim=1)
        flat_cnn = cat_cnn.view(cat_cnn.size(0), -1)
        utterance_features = self.dropout_1(F.relu(self.dense_1(flat_cnn)))

        lstm_out1, _ = self.blstm1(embedded)
        lstm_out2, _ = self.blstm2(lstm_out1)
        lstm_avg = torch.mean(lstm_out2, dim=1)
        context_features = self.dropout_2(F.relu(self.dense_2(lstm_avg)))

        combined = torch.cat((utterance_features, context_features), dim=1)
        logits = self.output(combined)
        return logits


def extract_dialogue_act_features(utterance: str, prev_utterance: str = "") -> Dict[str, Any]:
    """Extract surface, syntactic, and contextual features for Dialogue Act analysis."""
    text = str(utterance).strip()
    lower = text.lower()
    words = re.findall(r"\w+", lower)
    prev_lower = str(prev_utterance).strip().lower()

    wh_words = {"what", "where", "when", "who", "why", "how", "which"}
    aux_verbs = {"do", "does", "did", "is", "are", "was", "were", "can", "could", "would", "will", "should", "have", "has"}
    backchannels = {"yeah", "uh-huh", "uhhuh", "right", "okay", "ok", "yep", "mhm", "huh"}
    repair_markers = {"what?", "sorry?", "pardon?", "excuse me?", "huh?", "what did you say"}
    opinion_markers = {"think", "believe", "feel", "seem", "seems", "probably", "maybe", "suppose", "guess"}
    reformulate_markers = {"you mean", "so you're saying", "in other words", "so basically"}

    is_question = text.endswith("?")
    starts_wh = bool(words and words[0] in wh_words)
    starts_aux = bool(words and words[0] in aux_verbs)
    is_echo = bool(prev_lower and is_question and len(words) <= 5 and any(w in prev_lower for w in words))

    return {
        "token_count": len(words),
        "is_question": is_question,
        "starts_wh": starts_wh,
        "starts_aux": starts_aux,
        "is_backchannel": len(words) <= 2 and any(w in backchannels for w in words) and not is_question,
        "is_repair_signal": lower in repair_markers or is_echo,
        "has_opinion_marker": any(w in opinion_markers for w in words),
        "has_reformulate_marker": any(m in lower for m in reformulate_markers),
    }


def classify_utterance_heuristic(utterance: str, prev_utterance: str = "") -> str:
    """
    Lightweight rule-based DAMSL Dialogue Act classifier for rapid unit testing & pipeline routing.
    Returns one of key Switchboard tags: {'br', 'bf', 'qw', 'qy', 'b', 'sv', 'sd'}.
    """
    feats = extract_dialogue_act_features(utterance, prev_utterance)
    if feats["is_repair_signal"]:
        return "br"
    if feats["has_reformulate_marker"]:
        return "bf"
    if feats["is_backchannel"]:
        return "b"
    if feats["is_question"] or feats["starts_wh"] or feats["starts_aux"]:
        if feats["starts_wh"]:
            return "qw"
        return "qy"
    if feats["has_opinion_marker"]:
        return "sv"
    return "sd"
