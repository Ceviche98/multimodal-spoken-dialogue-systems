"""
Multimodal Referring Expression Comprehension & Visual Grounding (Lab 11 & Assignment 2 Part D).

Implements:
- Qwen3.5-VL [0, 999] normalized 2D bounding-box coordinate transformation (`to_qwen_coord`, `build_candidate_description`)
- Multi-agent GuessWhat?! prompt engineering for Questioner (Config B Prefill vs. Config C Open-Ended),
  Oracle (0-shot hyper-constrained), and Selector (greedy bracket prefill `[`)
- Oracle response normalizer and strict/recoverable validity evaluator (`normalize_oracle_answer`, `evaluate_oracle_outputs`)
- Multimodal referent fusion scorer combining visual spatial priors, category alignment, and multi-turn Oracle QA (`score_multimodal_referent_fusion`)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

VALID_ANSWERS = {"yes", "no", "n/a"}

GUESSWHAT_BENCHMARK_METRICS: Dict[str, Dict[str, Any]] = {
    "Config B (Zero-Shot, Prefill 'Is it ', Temp=0.7, 0-Shot Oracle)": {
        "run1_acc": 0.45,
        "run2_acc": 0.50,
        "strict_valid_rate": 0.91,
        "recoverable_rate": 1.00,
        "time_s": 619,
    },
    "Config B (LoRA Fine-Tuned Oracle r=32, alpha=64)": {
        "run1_acc": 0.45,
        "run2_acc": 0.45,
        "strict_valid_rate": 1.00,
        "recoverable_rate": 1.00,
        "time_s": 5865,
    },
    "Config C (Zero-Shot Open-Ended Semantic, Temp=0.5, 1-Shot Oracle)": {
        "run1_acc": 0.25,
        "run2_acc": 0.35,
        "strict_valid_rate": 0.76,
        "recoverable_rate": 1.00,
        "time_s": 784,
    },
    "Config C (LoRA Fine-Tuned Oracle r=32, alpha=64)": {
        "run1_acc": 0.30,
        "run2_acc": 0.30,
        "strict_valid_rate": 1.00,
        "recoverable_rate": 1.00,
        "time_s": 2562,
    },
    "Config A (Prefill 'Is it ', Temp=0.7, 1-Shot Oracle)": {
        "run1_acc": 0.20,
        "run2_acc": 0.20,
        "strict_valid_rate": 0.68,
        "recoverable_rate": 0.95,
        "time_s": 717,
    },
    "Config D (Prefill 'Is it ', Temp=0.3, 1-Shot Oracle)": {
        "run1_acc": 0.15,
        "run2_acc": 0.15,
        "strict_valid_rate": 0.65,
        "recoverable_rate": 0.94,
        "time_s": 684,
    },
}


def to_qwen_coord(v: float, size: int) -> int:
    """Normalize pixel coordinate `v` along image dimension `size` into Qwen-VL's [0, 999] grid."""
    if size <= 1:
        return 0
    return int(round((float(v) / float(size)) * 999))


def build_candidate_description(candidates: List[Dict[str, Any]], image_size: Tuple[int, int] = (640, 480)) -> str:
    """
    Convert COCO candidate objects (top-left x, y, width, height) into Qwen-VL [0, 999]
    bounding box strings: `[idx] category {"bbox_2d": [xmin, ymin, xmax, ymax]}`.
    """
    img_w, img_h = image_size
    lines: List[str] = []
    for idx, obj in enumerate(candidates):
        x, y, w, h = obj["bbox"]
        q_xmin = to_qwen_coord(x, img_w)
        q_ymin = to_qwen_coord(y, img_h)
        q_xmax = to_qwen_coord(x + w, img_w)
        q_ymax = to_qwen_coord(y + h, img_h)
        lines.append(
            f'[{idx}] {obj["category"]} {{"bbox_2d": [{q_xmin}, {q_ymin}, {q_xmax}, {q_ymax}]}}'
        )
    return "\n".join(lines)


def build_questioner_messages(
    candidates: List[Dict[str, Any]],
    dialogue_history: List[Tuple[str, str]],
    image_size: Tuple[int, int] = (640, 480),
    config: str = "B",
) -> List[Dict[str, Any]]:
    """Construct role-isolated chat messages for the GuessWhat?! Questioner agent."""
    cd = build_candidate_description(candidates, image_size=image_size)
    if config.upper() == "B":
        system_prompt = (
            "You are the Questioner in the GuessWhat?! game.\n"
            "Your goal is to identify a secret target object from the list of candidates.\n"
            f"Candidates:\n{cd}\n\n"
            "You MUST ask a YES/NO question to narrow down the candidates.\n"
            "DO NOT describe the image. DO NOT ask 'Who', 'What', or 'How many'.\n"
            "Your question MUST start with 'Is it'.\n"
            "Output ONLY the question text."
        )
    else:
        system_prompt = (
            "You are the Questioner in the GuessWhat?! game.\n"
            f"Identify the secret target from:\n{cd}\n\n"
            "Ask a short YES/NO question about ONE candidate to narrow down.\n"
            "Example: Is it a person? Is it on the left?\n"
            "Output ONLY the question."
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    for role, text in dialogue_history:
        if role == "Q":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        elif role == "A":
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    if not dialogue_history:
        messages.append({"role": "user", "content": [{"type": "text", "text": "Ask the first question."}]})
    else:
        messages[-1]["content"].append({"type": "text", "text": " Ask the next question."})

    return messages


def build_oracle_messages(
    question: str,
    candidates: List[Dict[str, Any]],
    target_idx: int,
    image_size: Tuple[int, int] = (640, 480),
) -> List[Dict[str, Any]]:
    """Construct 0-shot hyper-constrained messages for the GuessWhat?! Oracle agent."""
    cd = build_candidate_description(candidates, image_size=image_size)
    system_text = (
        f"You are the Oracle in the GuessWhat?! game.\n"
        f"The secret target object is candidate [{target_idx}].\n"
        f"Candidates:\n{cd}\n\n"
        f'Answer "Yes" or "No" about candidate [{target_idx}]. ONLY output one word.'
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": f"Question: {question}"}]},
    ]


def build_selector_prompt(
    dialogue_history: List[Tuple[str, str]],
    candidates: List[Dict[str, Any]],
    image_size: Tuple[int, int] = (640, 480),
) -> List[Dict[str, Any]]:
    """Construct Selector prompt to pick the target object index `[idx]` after multi-turn QA."""
    cd = build_candidate_description(candidates, image_size=image_size)
    system_text = (
        f"You are the Selector in the GuessWhat?! game.\n"
        f"Candidates:\n{cd}\n\n"
        "Based on the dialogue, select the target. Output ONLY the index in brackets like [0]."
    )
    history_str = "\n".join([f"{r}: {t}" for r, t in dialogue_history])
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Dialogue history:\n{history_str}\n\nSelect the target index."}],
        },
    ]


def normalize_oracle_answer(raw: str) -> str:
    """
    Normalize raw Oracle model generation into 'Yes', 'No', 'N/A', or 'None' (if unrecoverable).
    """
    cleaned = str(raw).strip().lower()
    if cleaned in VALID_ANSWERS:
        return cleaned.upper() if cleaned == "n/a" else cleaned.capitalize()

    for token in ("yes", "no", "n/a"):
        if cleaned.startswith(token):
            return token.upper() if token == "n/a" else token.capitalize()

    match = re.search(r"\b(yes|no|n/a)\b", cleaned)
    if match:
        token = match.group(1)
        return token.upper() if token == "n/a" else token.capitalize()

    return "None"


def is_strictly_valid(raw: str) -> bool:
    """Return True if raw output is strictly one of {'yes', 'no', 'n/a'} without extra tokens."""
    return str(raw).strip().lower() in VALID_ANSWERS


def evaluate_oracle_outputs(oracle_predictions: List[str]) -> Dict[str, Any]:
    """Evaluate strict formatting compliance and recoverability rate across Oracle responses."""
    if not oracle_predictions:
        return {"n": 0, "strict_valid_rate": 0.0, "recoverable_rate": 0.0}
    n = len(oracle_predictions)
    strict_count = sum(1 for raw in oracle_predictions if is_strictly_valid(raw))
    recoverable_count = sum(1 for raw in oracle_predictions if normalize_oracle_answer(raw) != "None")
    return {
        "n": n,
        "strict_valid_rate": strict_count / n,
        "recoverable_rate": recoverable_count / n,
    }


def extract_selector_answer(text: str) -> Optional[int]:
    """Extract candidate integer index from bracketed Selector output `[idx]`."""
    matches = re.findall(r"\[(\d+)\]", str(text))
    return int(matches[-1]) if matches else None


def score_multimodal_referent_fusion(
    candidates: List[Dict[str, Any]],
    dialogue_history: List[Tuple[str, str]],
    image_size: Tuple[int, int] = (640, 480),
) -> Dict[int, float]:
    """
    Multimodal Visual-Linguistic Grounding Scorer (Lab 11):
    Combines spatial region predicates (left/right/top/bottom/center), object category mentions,
    and candidate index hypotheses against Oracle Yes/No confirmations to compute a posterior
    probability distribution over visual candidates.
    """
    if not candidates:
        return {}

    img_w, img_h = image_size
    log_scores = {i: 0.0 for i in range(len(candidates))}

    for turn_idx in range(0, len(dialogue_history) - 1, 2):
        q_role, q_text = dialogue_history[turn_idx]
        a_role, a_text = dialogue_history[turn_idx + 1]
        if q_role != "Q" or a_role != "A":
            continue

        ans = normalize_oracle_answer(a_text)
        if ans not in ("Yes", "No"):
            continue
        polarity = 1.0 if ans == "Yes" else -1.0
        q_low = q_text.lower()

        # Check explicit candidate ID mentions (e.g. "1st", "candidate 2", "Is it 0?")
        id_matches = re.findall(r"\b(\d+)(?:st|nd|rd|th)?\b", q_low)
        for m in id_matches:
            cid = int(m)
            # Handle 1-based ordinal phrasing ("1st candidate") vs 0-based index
            matched_indices = []
            if f"{m}st" in q_low or f"{m}nd" in q_low or f"{m}rd" in q_low or f"{m}th" in q_low:
                if 0 <= cid - 1 < len(candidates):
                    matched_indices.append(cid - 1)
            if 0 <= cid < len(candidates) and cid not in matched_indices:
                matched_indices.append(cid)
            for idx in matched_indices:
                log_scores[idx] += 1.5 * polarity

        # Check category & spatial grounding attributes
        for idx, obj in enumerate(candidates):
            cat = str(obj.get("category", "")).lower()
            x, y, w, h = obj["bbox"]
            cx = (x + w / 2.0) / max(1.0, float(img_w))
            cy = (y + h / 2.0) / max(1.0, float(img_h))

            if cat and cat in q_low:
                log_scores[idx] += 1.2 * polarity
            if "person" in q_low or "human" in q_low:
                if cat == "person":
                    log_scores[idx] += 1.0 * polarity
                else:
                    log_scores[idx] -= 0.5 * polarity

            if "left" in q_low:
                log_scores[idx] += (0.8 if cx < 0.5 else -0.8) * polarity
            if "right" in q_low:
                log_scores[idx] += (0.8 if cx >= 0.5 else -0.8) * polarity
            if "top" in q_low or "upper" in q_low:
                log_scores[idx] += (0.8 if cy < 0.5 else -0.8) * polarity
            if "bottom" in q_low or "lower" in q_low:
                log_scores[idx] += (0.8 if cy >= 0.5 else -0.8) * polarity

    # Softmax normalization
    import math

    max_s = max(log_scores.values())
    exp_scores = {i: math.exp(s - max_s) for i, s in log_scores.items()}
    total = sum(exp_scores.values())
    return {i: round(exp_scores[i] / total, 4) for i in exp_scores}
