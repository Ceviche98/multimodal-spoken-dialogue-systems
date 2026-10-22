"""
Spoken Dialogue ASR Evaluation & Acoustic Resilience Pipeline (Lab 8).

Implements:
- Word Error Rate (WER) and Character Error Rate (CER) via dynamic programming Levenshtein alignment
- Character n-gram & token-level semantic similarity resilience to ASR transcription corruption
- Acoustic domain & latency/RTF analysis over `data/asr_evaluation_results.csv`
- FLEURS benchmark baselines across Whisper scales (Tiny, Base, Small, Medium, Distil-Large-v3, Large-v3-Turbo)
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

FLEURS_BENCHMARK_WER: Dict[str, float] = {
    "tiny": 0.1502,
    "base": 0.1066,
    "small": 0.0696,
    "medium": 0.0513,
}


def normalize_transcript(text: str) -> str:
    """Normalize transcript text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    cleaned = str(text).lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _edit_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis strings.
    WER = (S + D + I) / N_ref
    """
    ref_norm = normalize_transcript(reference) if normalize else reference.strip()
    hyp_norm = normalize_transcript(hypothesis) if normalize else hypothesis.strip()

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = _edit_distance(ref_words, hyp_words)
    return float(dist) / float(len(ref_words))


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Compute Character Error Rate (CER) between reference and hypothesis strings.
    """
    ref_norm = normalize_transcript(reference) if normalize else reference.strip()
    hyp_norm = normalize_transcript(hypothesis) if normalize else hypothesis.strip()

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = _edit_distance(ref_chars, hyp_chars)
    return float(dist) / float(len(ref_chars))


def compute_semantic_similarity(reference: str, hypothesis: str) -> float:
    """
    Compute hybrid character 3-gram + word token cosine similarity in [0, 1]
    to quantify downstream dense retrieval resilience to phonetic ASR errors.
    """
    ref_norm = normalize_transcript(reference)
    hyp_norm = normalize_transcript(hypothesis)
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0
    if ref_norm == hyp_norm:
        return 1.0

    def _features(s: str) -> Counter:
        feats: Counter = Counter()
        for w in s.split():
            feats[f"w:{w}"] += 2.0
        padded = f"#{s}#"
        for i in range(max(0, len(padded) - 2)):
            feats[f"c3:{padded[i:i+3]}"] += 1.0
        return feats

    v1 = _features(ref_norm)
    v2 = _features(hyp_norm)

    common = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(val * val for val in v1.values()))
    norm2 = math.sqrt(sum(val * val for val in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(min(1.0, max(0.0, dot / (norm1 * norm2))))


def load_asr_evaluation_results(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load `asr_evaluation_results.csv` and enrich with computed CER and Semantic Similarity
    for Base, Distil, and Turbo Whisper models.
    """
    if csv_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        csv_path = os.path.join(repo_root, "data", "asr_evaluation_results.csv")

    df = pd.read_csv(csv_path)
    df["Domain"] = df["Topic/Setting"].apply(lambda x: "_".join(str(x).split("_")[:-1]))

    for model in ["Base", "Distil", "Turbo"]:
        out_col = f"Output ({model})"
        cer_col = f"CER ({model})"
        sem_col = f"SemSim ({model})"
        df[cer_col] = [
            compute_cer(ref, hyp)
            for ref, hyp in zip(df["Normalized Reference"], df[out_col])
        ]
        df[sem_col] = [
            compute_semantic_similarity(ref, hyp)
            for ref, hyp in zip(df["Normalized Reference"], df[out_col])
        ]

    return df


def summarize_asr_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute aggregate WER, CER, Semantic Similarity, Latency, and RTF per ASR model."""
    summary: Dict[str, Dict[str, float]] = {}
    for model in ["Base", "Distil", "Turbo"]:
        summary[model] = {
            "mean_wer": float(df[f"WER ({model})"].mean()),
            "median_wer": float(df[f"WER ({model})"].median()),
            "max_wer": float(df[f"WER ({model})"].max()),
            "zero_error_rate": float((df[f"WER ({model})"] == 0).mean()),
            "mean_cer": float(df[f"CER ({model})"].mean()),
            "mean_semantic_similarity": float(df[f"SemSim ({model})"].mean()),
            "mean_latency_s": float(df[f"Lat ({model})"].mean()),
            "mean_rtf": float(df[f"RTF ({model})"].mean()),
        }
    return summary


def analyze_asr_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate WER, CER, Semantic Similarity, and Latency across acoustic/topic domains."""
    grouped = (
        df.groupby("Domain")
        .agg(
            count=("Topic/Setting", "count"),
            wer_base=("WER (Base)", "mean"),
            wer_distil=("WER (Distil)", "mean"),
            wer_turbo=("WER (Turbo)", "mean"),
            cer_base=("CER (Base)", "mean"),
            cer_distil=("CER (Distil)", "mean"),
            cer_turbo=("CER (Turbo)", "mean"),
            sem_base=("SemSim (Base)", "mean"),
            sem_distil=("SemSim (Distil)", "mean"),
            sem_turbo=("SemSim (Turbo)", "mean"),
            lat_base=("Lat (Base)", "mean"),
            lat_distil=("Lat (Distil)", "mean"),
            lat_turbo=("Lat (Turbo)", "mean"),
        )
        .reset_index()
    )
    return grouped
