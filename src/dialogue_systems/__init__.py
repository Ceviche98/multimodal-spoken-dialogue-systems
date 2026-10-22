"""
Multimodal & Spoken Dialogue Systems Toolkit.

Provides production-grade modular implementations of:
1. Spoken Dialogue ASR & Acoustic Resilience Evaluation (`spoken_asr`)
2. Dialogue Act Tagging & Hierarchical Context Modeling (`dialogue_act`)
3. Attention-Based Seq2Seq Neural Response Generation (`seq2seq_generator`)
4. Uncertainty-Aware Clarification Dialogue Management (`clarification_dm`)
5. Multi-Party Turn-Taking & Next-Speaker Prediction (`multiparty`)
6. Multimodal Visual-Linguistic Reference Grounding (`multimodal_grounding`)
"""

from .spoken_asr import (
    compute_wer,
    compute_cer,
    compute_semantic_similarity,
    normalize_transcript,
    load_asr_evaluation_results,
    summarize_asr_models,
    analyze_asr_by_domain,
    FLEURS_BENCHMARK_WER,
)
from .dialogue_act import (
    damsl_act_tag,
    pad_sequences,
    BiLSTMModel,
    CNN_BiLSTM,
    extract_dialogue_act_features,
    classify_utterance_heuristic,
    SWDA_BENCHMARK_METRICS,
)
from .seq2seq_generator import (
    BahdanauAttention,
    Encoder,
    Decoder,
    DecoderAdvanced,
    preprocess_utterance,
    greedy_decode_Ids,
    SEQ2SEQ_ABLATION_METRICS,
)
from .clarification_dm import (
    AREA_MAP,
    FOOD_MAP,
    PRICE_MAP,
    RESTAURANTS,
    normalize_slot_verbose,
    parse_user_constraints,
    init_state,
    decide_clarification,
    build_clarification_response,
    update_state_from_parsed,
    dm_preprocess,
    evaluate_clarification_benchmarks,
    compute_expected_utility_policy,
)
from .multiparty import (
    SPEAKER2ID,
    ID2SPEAKER,
    build_turns,
    build_multiparty_dataset,
    UtteranceBiLSTMEncoder,
    NextSpeakerBiLSTMModel,
    score_multiparty_turn_taking,
    AMI_BENCHMARK_METRICS,
)
from .multimodal_grounding import (
    to_qwen_coord,
    build_candidate_description,
    build_questioner_messages,
    build_oracle_messages,
    build_selector_prompt,
    normalize_oracle_answer,
    extract_selector_answer,
    evaluate_oracle_outputs,
    score_multimodal_referent_fusion,
    GUESSWHAT_BENCHMARK_METRICS,
)

__all__ = [
    "compute_wer",
    "compute_cer",
    "compute_semantic_similarity",
    "normalize_transcript",
    "load_asr_evaluation_results",
    "summarize_asr_models",
    "analyze_asr_by_domain",
    "FLEURS_BENCHMARK_WER",
    "damsl_act_tag",
    "pad_sequences",
    "BiLSTMModel",
    "CNN_BiLSTM",
    "extract_dialogue_act_features",
    "classify_utterance_heuristic",
    "SWDA_BENCHMARK_METRICS",
    "BahdanauAttention",
    "Encoder",
    "Decoder",
    "DecoderAdvanced",
    "preprocess_utterance",
    "greedy_decode_Ids",
    "SEQ2SEQ_ABLATION_METRICS",
    "AREA_MAP",
    "FOOD_MAP",
    "PRICE_MAP",
    "RESTAURANTS",
    "normalize_slot_verbose",
    "parse_user_constraints",
    "init_state",
    "decide_clarification",
    "build_clarification_response",
    "update_state_from_parsed",
    "dm_preprocess",
    "evaluate_clarification_benchmarks",
    "compute_expected_utility_policy",
    "SPEAKER2ID",
    "ID2SPEAKER",
    "build_turns",
    "build_multiparty_dataset",
    "UtteranceBiLSTMEncoder",
    "NextSpeakerBiLSTMModel",
    "score_multiparty_turn_taking",
    "AMI_BENCHMARK_METRICS",
    "to_qwen_coord",
    "build_candidate_description",
    "build_questioner_messages",
    "build_oracle_messages",
    "build_selector_prompt",
    "normalize_oracle_answer",
    "extract_selector_answer",
    "evaluate_oracle_outputs",
    "score_multimodal_referent_fusion",
    "GUESSWHAT_BENCHMARK_METRICS",
]
