"""
CLI Script: Comprehensive Dialogue Systems Benchmark Runner & Plot Generator.

Executes benchmarks and generates publication-grade visualizations across:
1. Uncertainty-Aware Clarification Dialogue Manager (`clarification_dm_policy_tradeoff.png`)
2. Multimodal Referring Expression Comprehension & Visual Grounding (`multimodal_grounding_accuracy.png`)
3. Dialogue Act Tagging, Seq2Seq Add & Norm Ablation, and Multi-Party Turn-Taking (`dialogue_models_ablation_summary.png`)
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dialogue_systems.clarification_dm import (
    compute_expected_utility_policy,
    evaluate_clarification_benchmarks,
)
from dialogue_systems.dialogue_act import SWDA_BENCHMARK_METRICS
from dialogue_systems.multimodal_grounding import GUESSWHAT_BENCHMARK_METRICS
from dialogue_systems.multiparty import AMI_BENCHMARK_METRICS
from dialogue_systems.seq2seq_generator import SEQ2SEQ_ABLATION_METRICS


def plot_clarification_policy(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)

    # Panel A: Expected Utility across Slot Confidence c in [0, 1]
    confs = np.linspace(0.0, 1.0, 200)
    eu_accept = [compute_expected_utility_policy(c)["utilities"]["accept"] for c in confs]
    eu_implicit = [compute_expected_utility_policy(c)["utilities"]["implicit_clarification"] for c in confs]
    eu_explicit = [compute_expected_utility_policy(c)["utilities"]["explicit_clarification"] for c in confs]
    eu_reject = [compute_expected_utility_policy(c)["utilities"]["reject_reask"] for c in confs]

    axes[0].plot(confs, eu_accept, label="Direct Commit / Accept", color="#1b9e77", linewidth=2.2)
    axes[0].plot(confs, eu_implicit, label="Implicit Clarification", color="#d95f02", linewidth=2.2)
    axes[0].plot(confs, eu_explicit, label="Explicit Confirmation", color="#7570b3", linewidth=2.2, linestyle="--")
    axes[0].plot(confs, eu_reject, label="Reject / OOV Re-prompt", color="#e7298a", linewidth=1.8, linestyle=":")

    axes[0].axvline(0.84, color="#333333", linestyle="-.", alpha=0.75, label="Low Conf Threshold (0.84)")
    axes[0].axvline(0.94, color="#666666", linestyle="--", alpha=0.75, label="Med Conf Threshold (0.94)")
    axes[0].axvspan(0.72, 0.84, color="#7570b3", alpha=0.10)
    axes[0].axvspan(0.84, 0.94, color="#d95f02", alpha=0.10)
    axes[0].axvspan(0.94, 1.00, color="#1b9e77", alpha=0.10)

    axes[0].set_title("A. Decision-Theoretic Expected Utility vs. ASR Slot Confidence")
    axes[0].set_xlabel("Normalized Slot Confidence Score")
    axes[0].set_ylabel("Expected Dialogue Utility (EU)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=8.5)

    # Panel B: Benchmark Evaluation Results by Category & Threshold Contradiction (A2 vs N3)
    categories = ["Missing Slot\n(M1-M3)", "Ambiguous Typo\n(A1-A2)", "OOV Ontology\n(O1-O3)", "Noisy ASR\n(N1-N4)", "Overall\n(12 Cases)"]
    default_acc = [100.0, 50.0, 100.0, 100.0, 91.67]
    relaxed_acc = [100.0, 100.0, 100.0, 75.0, 91.67]

    x = np.arange(len(categories))
    w = 0.34
    b1 = axes[1].bar(x - w / 2, default_acc, width=w, label="Default Policy (low_conf=0.84)", color="#2b5c8f", edgecolor="black", linewidth=0.6)
    b2 = axes[1].bar(x + w / 2, relaxed_acc, width=w, label="Punct-Stripped + Relaxed (low_conf=0.83)", color="#e08214", edgecolor="black", linewidth=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=9)
    axes[1].set_ylim(0, 115)
    axes[1].set_title("B. Clarification Policy Accuracy & A2/N3 Threshold Trade-off")
    axes[1].set_ylabel("Clarification Move Accuracy (%)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "clarification_dm_policy_tradeoff.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_multimodal_grounding(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=180)

    # Panel A: Grid Search & LoRA Fine-Tuning Comparison
    configs = [
        "Config B\n(0-Shot Run 1)",
        "Config B\n(0-Shot Run 2)",
        "Config B\n(LoRA r=32)",
        "Config C\n(0-Shot Run 1)",
        "Config C\n(0-Shot Run 2)",
        "Config C\n(LoRA r=32)",
        "Config A\n(1-Shot T=0.7)",
        "Config D\n(1-Shot T=0.3)",
    ]
    accs = [45.0, 50.0, 45.0, 25.0, 35.0, 30.0, 20.0, 15.0]
    strict_valid = [91.0, 92.0, 100.0, 72.0, 76.0, 100.0, 68.0, 65.0]

    x = np.arange(len(configs))
    w = 0.36
    axes[0].bar(x - w / 2, accs, width=w, label="Target Selection Accuracy (%)", color="#1f78b4", edgecolor="black", linewidth=0.6)
    axes[0].bar(x + w / 2, strict_valid, width=w, label="Oracle Strict Format Validity (%)", color="#33a02c", edgecolor="black", linewidth=0.6)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=28, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 112)
    axes[0].set_title("A. GuessWhat?! Visual Grounding Accuracy vs. Oracle Format Compliance")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8.5)

    # Panel B: Oracle Error Mode Shift (Zero-Shot Sycophancy vs LoRA Negative Bias)
    regimes = ["Config B\nZero-Shot", "Config B\nLoRA Tuned", "Config C\nZero-Shot", "Config C\nLoRA Tuned"]
    yes_rate = [48.0, 14.0, 54.0, 26.0]
    no_rate = [43.0, 86.0, 22.0, 74.0]
    none_rate = [9.0, 0.0, 24.0, 0.0]

    rx = np.arange(len(regimes))
    axes[1].bar(rx, yes_rate, width=0.5, label="'Yes' (Affirmation / Sycophancy)", color="#66c2a5", edgecolor="black", linewidth=0.6)
    axes[1].bar(rx, no_rate, width=0.5, bottom=yes_rate, label="'No' (Rejection / Mode Collapse)", color="#fc8d62", edgecolor="black", linewidth=0.6)
    axes[1].bar(
        rx,
        none_rate,
        width=0.5,
        bottom=np.array(yes_rate) + np.array(no_rate),
        label="'None' / Format Breakdown",
        color="#8da0cb",
        edgecolor="black",
        linewidth=0.6,
    )

    axes[1].set_xticks(rx)
    axes[1].set_xticklabels(regimes, fontsize=9.5)
    axes[1].set_ylim(0, 108)
    axes[1].set_title("B. Oracle Response Distribution Shift (100 Dialogue Turns)")
    axes[1].set_ylabel("Proportion of Oracle Responses (%)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "multimodal_grounding_accuracy.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_dialogue_models_summary(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), dpi=180)

    # Panel 1: Lab 2 Switchboard Dialogue Act Tagging (Overall vs Minority Classes)
    da_models = ["M1: BiLSTM\n(Unweighted)", "M2: BiLSTM\n(Balanced 5ep)", "M2b: BiLSTM\n(Balanced 20ep)", "M3: CNN+BiLSTM\n(Hierarchical)"]
    overall = [67.96, 34.83, 37.01, 70.63]
    br_acc = [0.0, 54.72, 54.72, 37.74]
    bf_acc = [0.0, 2.80, 12.40, 0.0]

    x1 = np.arange(len(da_models))
    w1 = 0.25
    axes[0].bar(x1 - w1, overall, width=w1, label="Overall Acc (43 Tags)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    axes[0].bar(x1, br_acc, width=w1, label="'br' (Non-Understanding, 0.13%)", color="#d95f02", edgecolor="black", linewidth=0.5)
    axes[0].bar(x1 + w1, bf_acc, width=w1, label="'bf' (Reformulate, 0.42%)", color="#1b9e77", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(da_models, fontsize=8.5)
    axes[0].set_ylim(0, 82)
    axes[0].set_title("A. Switchboard DA Tagging: Overall vs. Minority Repair")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)

    # Panel 2: Lab 3 Seq2Seq HybridDialogue Architecture Ablation (Final Loss @ Ep 50)
    s2s_models = ["Baseline\n(L=1, d=0.2)", "High Drop\n(L=1, d=0.5)", "Deeper\n(L=2, d=0.2)", "Mix\n(L=2, d=0.5)", "Add & Norm\n(L=2, d=0.3)", "Add & Norm\n(L=2, d=0.2)"]
    losses = [1.2123, 1.3000, 2.0400, 2.0100, 0.1060, 0.0830]
    bar_colors = ["#4daf4a", "#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#1b9e77"]
    bars = axes[1].bar(np.arange(len(s2s_models)), losses, color=bar_colors, edgecolor="black", linewidth=0.6, width=0.55)
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.05, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(np.arange(len(s2s_models)))
    axes[1].set_xticklabels(s2s_models, fontsize=8)
    axes[1].set_ylim(0, 2.35)
    axes[1].set_title("B. Seq2Seq Attention Decoder Ablation (Epoch 50 Loss)")
    axes[1].set_ylabel("Cross-Entropy Training Loss")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Lab 10 AMI Multi-Party Next-Speaker Prediction Ablation
    mp_models = ["Full BiLSTM\n(History + Speaker)", "No History\n(Current Turn + Spk)", "No Speaker Info\n(History Text Only)"]
    mp_acc = [70.72, 33.42, 27.36]
    mp_f1 = [70.48, 27.26, 27.26]
    x3 = np.arange(len(mp_models))
    w3 = 0.32
    axes[2].bar(x3 - w3 / 2, mp_acc, width=w3, label="Test Accuracy (%)", color="#386cb0", edgecolor="black", linewidth=0.6)
    axes[2].bar(x3 + w3 / 2, mp_f1, width=w3, label="Test Macro-F1 (%)", color="#f0027f", edgecolor="black", linewidth=0.6)
    axes[2].axhline(25.0, color="#555555", linestyle="--", linewidth=1.2, label="4-Speaker Random Baseline (25%)")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(mp_models, fontsize=8.5)
    axes[2].set_ylim(0, 82)
    axes[2].set_title("C. AMI Multi-Party Next-Speaker Prediction Ablation")
    axes[2].set_ylabel("Metric (%)")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "dialogue_models_ablation_summary.png")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assets_dir = os.path.join(repo_root, "docs", "assets")

    print("=== 1. Clarification Dialogue Manager Benchmark (Lab 9) ===")
    res_default = evaluate_clarification_benchmarks(low_conf=0.84, strip_punctuation=False)
    print(f"Default Policy Accuracy: {res_default['accuracy'] * 100:.2f}% (11/12)")
    for row in res_default["rows"]:
        status = "PASS" if row["match"] else "FAIL"
        print(f"  [{status}] {row['id']} ({row['group']}): '{row['user']}' -> {row['predicted_type']} | {row['reply']}")

    print("\n=== 2. Dialogue Act Tagging Benchmarks (Lab 2) ===")
    for k, v in SWDA_BENCHMARK_METRICS.items():
        print(f"  {k}: Acc={v['overall_acc']*100:.2f}%, br={v['br_acc']*100:.2f}%, bf={v['bf_acc']*100:.2f}%")

    print("\n=== 3. Seq2Seq Attention Generator Benchmarks (Lab 3) ===")
    for k, v in SEQ2SEQ_ABLATION_METRICS.items():
        print(f"  {k}: Initial Loss={v['initial_loss']:.3f} -> Final Loss={v['final_loss']:.3f}")

    print("\n=== 4. Multi-Party Turn-Taking Benchmarks (Lab 10) ===")
    for k, v in AMI_BENCHMARK_METRICS.items():
        print(f"  {k}: Test Acc={v['test_acc']*100:.2f}%, Macro-F1={v['test_macro_f1']*100:.2f}%")

    print("\n=== 5. Multimodal Visual Grounding Benchmarks (Lab 11) ===")
    for k, v in GUESSWHAT_BENCHMARK_METRICS.items():
        print(f"  {k}: Run1={v['run1_acc']*100:.1f}%, Run2={v['run2_acc']*100:.1f}%, StrictValid={v['strict_valid_rate']*100:.1f}%")

    plot_clarification_policy(assets_dir)
    plot_multimodal_grounding(assets_dir)
    plot_dialogue_models_summary(assets_dir)


if __name__ == "__main__":
    main()
