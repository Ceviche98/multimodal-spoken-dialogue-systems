"""
CLI Script: Spoken Dialogue ASR Evaluation & Visualization Pipeline.

Evaluates `data/asr_evaluation_results.csv` across Whisper Base, Distil-Large-v3,
and Large-v3-Turbo models (plus FLEURS benchmark baselines) and generates
publication-quality figures in `docs/assets/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dialogue_systems.spoken_asr import (
    FLEURS_BENCHMARK_WER,
    analyze_asr_by_domain,
    load_asr_evaluation_results,
    summarize_asr_models,
)


def generate_asr_plots(df, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=180)

    # Panel 1: WER vs Semantic Similarity Scatter & Regression Trend
    colors = {"Base": "#d95f02", "Distil": "#1b9e77", "Turbo": "#7570b3"}
    markers = {"Base": "o", "Distil": "s", "Turbo": "^"}
    for model in ["Base", "Distil", "Turbo"]:
        wers = df[f"WER ({model})"] * 100
        sems = df[f"SemSim ({model})"] * 100
        axes[0].scatter(
            wers,
            sems,
            label=f"Whisper-{model} (Mean WER={wers.mean():.1f}%)",
            color=colors[model],
            marker=markers[model],
            s=55,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
        )

    all_w = np.concatenate([df[f"WER ({m})"].values * 100 for m in ["Base", "Distil", "Turbo"]])
    all_s = np.concatenate([df[f"SemSim ({m})"].values * 100 for m in ["Base", "Distil", "Turbo"]])
    if len(all_w) > 1:
        coef = np.polyfit(all_w, all_s, 1)
        x_line = np.linspace(0, 42, 100)
        axes[0].plot(x_line, np.polyval(coef, x_line), "--", color="#444444", linewidth=1.5, label="Linear Fit")

    axes[0].set_title("A. ASR WER vs. Downstream Semantic Similarity")
    axes[0].set_xlabel("Word Error Rate (WER %)")
    axes[0].set_ylabel("Hybrid Semantic Similarity (%)")
    axes[0].set_xlim(-2, 43)
    axes[0].set_ylim(75, 102)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower left", fontsize=8.5)

    # Panel 2: Domain-Level WER Comparison
    domain_df = analyze_asr_by_domain(df).sort_values("wer_base", ascending=False)
    x = np.arange(len(domain_df))
    w = 0.26
    axes[1].bar(x - w, domain_df["wer_base"] * 100, width=w, label="Base", color=colors["Base"], edgecolor="black", linewidth=0.5)
    axes[1].bar(x, domain_df["wer_distil"] * 100, width=w, label="Distil-v3", color=colors["Distil"], edgecolor="black", linewidth=0.5)
    axes[1].bar(x + w, domain_df["wer_turbo"] * 100, width=w, label="Large-v3-Turbo", color=colors["Turbo"], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domain_df["Domain"], rotation=35, ha="right", fontsize=8.5)
    axes[1].set_title("B. Mean WER (%) by Acoustic / Topic Domain")
    axes[1].set_ylabel("Mean WER (%)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8.5)

    # Panel 3: Latency vs WER Pareto Frontier + FLEURS Scaling
    summary = summarize_asr_models(df)
    for model in ["Base", "Distil", "Turbo"]:
        lat = summary[model]["mean_latency_s"] * 1000
        wer = summary[model]["mean_wer"] * 100
        rtf = summary[model]["mean_rtf"] * 100
        axes[2].scatter(lat, wer, s=180, color=colors[model], edgecolors="black", linewidth=1.2, zorder=5)
        axes[2].annotate(
            f"Whisper-{model}\n(WER={wer:.1f}%, Lat={lat:.0f}ms, RTF={rtf:.1f}%)",
            (lat, wer),
            xytext=(8, 6 if model != "Base" else -18),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[2].set_title("C. Latency vs. WER Pareto Frontier (20 Accented Utterances)")
    axes[2].set_xlabel("Mean Inference Latency (ms)")
    axes[2].set_ylabel("Mean Word Error Rate (WER %)")
    axes[2].set_xlim(110, 320)
    axes[2].set_ylim(-1, 19)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "asr_wer_vs_semantic_similarity.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Spoken Dialogue ASR Pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to asr_evaluation_results.csv")
    parser.add_argument("--assets-dir", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assets_dir = args.assets_dir or os.path.join(repo_root, "docs", "assets")

    df = load_asr_evaluation_results(args.csv)
    summary = summarize_asr_models(df)
    domain_df = analyze_asr_by_domain(df)

    print("=== FLEURS (en_us) Benchmark Baselines ===")
    for model_size, wer_val in FLEURS_BENCHMARK_WER.items():
        print(f"  Whisper-{model_size:<8}: WER = {wer_val * 100:.2f}%")

    print("\n=== Custom 20-Utterance Multi-Domain Spoken Benchmark ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Domain Breakdown ===")
    print(domain_df.to_string(index=False))

    generate_asr_plots(df, assets_dir)


if __name__ == "__main__":
    main()
