"""Multi-Party Conversation Modeling & Multimodal Visual Grounding (Labs 10 & 11)."""

from __future__ import annotations

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

__all__ = [
    "SPEAKER2ID",
    "ID2SPEAKER",
    "AMI_BENCHMARK_METRICS",
    "AMI_CONFUSION_MATRIX",
    "build_turns",
    "build_multiparty_dataset",
    "UtteranceBiLSTMEncoder",
    "NextSpeakerBiLSTMModel",
    "score_multiparty_turn_taking",
    "VALID_ANSWERS",
    "GUESSWHAT_BENCHMARK_METRICS",
    "to_qwen_coord",
    "build_candidate_description",
    "normalize_oracle_answer",
    "evaluate_oracle_outputs",
    "extract_selector_answer",
    "score_multimodal_referent_fusion",
]
