"""Unit tests for Multi-Party Conversation Modeling and Multimodal Visual Grounding (Labs 10 & 11)."""

from __future__ import annotations

import pandas as pd
import torch

from dialogue_systems.multiparty import (
    AMI_BENCHMARK_METRICS,
    AMI_CONFUSION_MATRIX,
    ID2SPEAKER,
    SPEAKER2ID,
    NextSpeakerBiLSTMModel,
    UtteranceBiLSTMEncoder,
    build_multiparty_dataset,
    build_turns,
    score_multiparty_turn_taking,
)
from dialogue_systems.multimodal_grounding import (
    GUESSWHAT_BENCHMARK_METRICS,
    VALID_ANSWERS,
    build_candidate_description,
    evaluate_oracle_outputs,
    extract_selector_answer,
    normalize_oracle_answer,
    score_multimodal_referent_fusion,
    to_qwen_coord,
)


def test_multiparty_turn_builder() -> None:
    df = pd.DataFrame(
        {
            "meeting_id": ["m1", "m1", "m1", "m1", "m1", "m1"],
            "speaker": ["A", "A", "B", "B", "C", "C"],
            "start": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
            "end": [0.1, 0.4, 1.1, 1.3, 2.1, 2.4],
            "text": ["hi", "there", "how", "are", "you", "doing"],
        }
    )
    turns = build_turns(df, max_gap=0.5)
    assert len(turns) == 3
    assert turns.iloc[0]["speaker"] == "A"
    assert turns.iloc[0]["utt_text"] == "hi there"
    assert turns.iloc[1]["speaker"] == "B"
    assert turns.iloc[1]["utt_text"] == "how are"
    assert turns.iloc[2]["speaker"] == "C"
    assert turns.iloc[2]["utt_text"] == "you doing"

    ds = build_multiparty_dataset(turns, history=1)
    assert len(ds) == 1
    assert ds.iloc[0]["label"] == SPEAKER2ID["C"]


def test_multiparty_bilstm_architecture() -> None:
    history_len = 2
    model = NextSpeakerBiLSTMModel(
        vocab_size=100,
        history=history_len,
        emb_dim=16,
        word_hidden=16,
        ctx_hidden=32,
        spk_emb_dim=8,
    )
    batch_size = 2
    seq_len = 5
    history_tokens = [torch.randint(1, 100, (batch_size, seq_len)) for _ in range(history_len)]
    history_lengths = [torch.full((batch_size,), seq_len, dtype=torch.long) for _ in range(history_len)]
    history_speakers = torch.randint(0, 4, (batch_size, history_len))

    current_tokens = torch.randint(1, 100, (batch_size, seq_len))
    current_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    current_speaker = torch.randint(0, 4, (batch_size,))

    logits = model(
        history_tokens,
        history_lengths,
        history_speakers,
        current_tokens,
        current_lengths,
        current_speaker,
    )
    assert logits.shape == (2, 4)

    dist = score_multiparty_turn_taking("A", "What do you think?", ["B", "A", "B"])
    assert abs(sum(dist.values()) - 1.0) < 1e-3
    assert dist["B"] > dist["C"]


def test_multimodal_grounding_helpers() -> None:
    assert to_qwen_coord(320, 640) == 500
    cands = [
        {"id": 1, "category": "person", "bbox": [10.0, 20.0, 100.0, 200.0]},
        {"id": 2, "category": "car", "bbox": [400.0, 200.0, 150.0, 120.0]},
    ]
    desc = build_candidate_description(cands, image_size=(640, 480))
    assert "[0] person" in desc
    assert "[1] car" in desc

    assert normalize_oracle_answer("Yes, it is.") == "Yes"
    assert normalize_oracle_answer("No way") == "No"
    assert normalize_oracle_answer("I am not sure") == "None"

    stats = evaluate_oracle_outputs(["Yes", "No", "No, not that", "Unintelligible"])
    assert stats["strict_valid_rate"] == 0.50
    assert stats["recoverable_rate"] == 0.75

    assert extract_selector_answer("The target is [1].") == 1

    probs = score_multimodal_referent_fusion(
        cands,
        [("Q", "Is it a car on the right?"), ("A", "Yes")],
        image_size=(640, 480),
    )
    assert probs[1] > probs[0]


def test_benchmark_metrics_constants() -> None:
    assert "Full Model (8-Turn History + Speaker Embeddings)" in AMI_BENCHMARK_METRICS
    full_m = AMI_BENCHMARK_METRICS["Full Model (8-Turn History + Speaker Embeddings)"]
    assert full_m["test_acc"] == 0.7072
    assert full_m["test_macro_f1"] == 0.7048

    assert "Config B (Zero-Shot, Prefill 'Is it ', Temp=0.7, 0-Shot Oracle)" in GUESSWHAT_BENCHMARK_METRICS
    assert len(AMI_CONFUSION_MATRIX) == 4
