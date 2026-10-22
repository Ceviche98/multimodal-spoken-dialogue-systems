"""
Uncertainty-Aware Clarification Dialogue Manager (Lab 9 & Assignment 2 Part B).

Implements:
- Hierarchical slot normalization (`normalize_slot_verbose`): exact -> substring -> n-gram `difflib` fuzzy -> OOV/missing
- Slot mention detection with lexical hints (`parse_user_constraints`)
- Confidence-thresholded clarification policy (`decide_clarification`):
  - High confidence (>= 0.94): Direct execution / state commit
  - Medium confidence ([0.84, 0.94)): Implicit confirmation embedded in next slot query
  - Low confidence (< 0.84): Explicit binary confirmation ("I think you said X. Is that right?")
  - Mentioned OOV (confidence == 0.0, mentioned=True): Explicit alternative-listing clarification (`requires_options=True`)
  - Missing slot (mentioned=False): Follow-up wh-question
- Optimistic safe state update (`update_state_from_parsed`)
- Expected utility decision policy analysis (`compute_expected_utility_policy`)
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Any, Dict, List, Optional, Tuple

RESTAURANTS: List[Dict[str, str]] = [
    {"id": "r001", "name": "Centre Noodle House", "area": "centre", "food": "chinese", "pricerange": "cheap", "address": "11 King St", "phone": "01223 100001"},
    {"id": "r002", "name": "Centre La Pasta", "area": "centre", "food": "italian", "pricerange": "moderate", "address": "4 Market Sq", "phone": "01223 100002"},
    {"id": "r003", "name": "Centre Spice Garden", "area": "centre", "food": "indian", "pricerange": "expensive", "address": "25 Regent St", "phone": "01223 100003"},
    {"id": "r004", "name": "Centre Sakura", "area": "centre", "food": "japanese", "pricerange": "moderate", "address": "8 Trinity Ln", "phone": "01223 100004"},
    {"id": "r005", "name": "West Green Bowl", "area": "west", "food": "vegetarian", "pricerange": "cheap", "address": "3 Willow Rd", "phone": "01223 100005"},
    {"id": "r006", "name": "West Dragon", "area": "west", "food": "chinese", "pricerange": "moderate", "address": "18 Queen Rd", "phone": "01223 100006"},
    {"id": "r007", "name": "North Grill", "area": "north", "food": "british", "pricerange": "moderate", "address": "77 Milton Rd", "phone": "01223 100007"},
    {"id": "r008", "name": "Riverside Curry House", "area": "riverside", "food": "indian", "pricerange": "cheap", "address": "2 River Walk", "phone": "01223 100008"},
]

AREA_MAP: Dict[str, str] = {
    "centre": "centre",
    "center": "centre",
    "city centre": "centre",
    "city center": "centre",
    "downtown": "centre",
    "central": "centre",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "riverside": "riverside",
    "river side": "riverside",
    "riverside area": "riverside",
}

FOOD_MAP: Dict[str, str] = {
    "chinese": "chinese",
    "italian": "italian",
    "indian": "indian",
    "japanese": "japanese",
    "vegetarian": "vegetarian",
    "veggie": "vegetarian",
    "british": "british",
    "english": "british",
}

PRICE_MAP: Dict[str, str] = {
    "cheap": "cheap",
    "inexpensive": "cheap",
    "budget": "cheap",
    "low": "cheap",
    "low priced": "cheap",
    "moderate": "moderate",
    "medium": "moderate",
    "moderately priced": "moderate",
    "expensive": "expensive",
    "pricey": "expensive",
    "high": "expensive",
    "high end": "expensive",
}

AREA_HINTS: List[str] = [
    "area", "location", "part of town", "part", "side", "district",
    "centre", "center", "central", "north", "south", "east", "west",
    "riverside", "river side", "downtown", "seaside", "senter", "cent",
]

FOOD_HINTS: List[str] = [
    "food", "cuisine", "eat", "meal",
    "chinese", "italian", "indian", "japanese", "vegetarian", "veggie",
    "british", "english", "asian", "oriental", "chainese", "japanees",
]

PRICE_HINTS: List[str] = [
    "price", "priced", "cost", "cheap", "budget", "expensive", "pricey",
    "moderate", "mid", "medium", "midscale", "premium", "high end", "modrate",
]

TEST_BENCHMARKS: List[Dict[str, str]] = [
    {"group": "missing", "id": "M1", "user": "I want Chinese food.", "expected_type": "follow_up"},
    {"group": "missing", "id": "M2", "user": "I want a cheap place.", "expected_type": "follow_up"},
    {"group": "missing", "id": "M3", "user": "Find me somewhere in the centre.", "expected_type": "follow_up"},
    {"group": "ambiguous", "id": "A1", "user": "I want chainese food.", "expected_type": "implicit"},
    {"group": "ambiguous", "id": "A2", "user": "Find me something in the senter.", "expected_type": "implicit"},
    {"group": "oov", "id": "O1", "user": "I want asian food.", "expected_type": "explicit"},
    {"group": "oov", "id": "O2", "user": "Find me a midscale place.", "expected_type": "explicit"},
    {"group": "oov", "id": "O3", "user": "I'm looking for oriental food in the west.", "expected_type": "explicit"},
    {"group": "noisy_asr", "id": "N1", "user": "i want a cheap chainese place", "expected_type": "implicit"},
    {"group": "noisy_asr", "id": "N2", "user": "modrate japanees food", "expected_type": "implicit"},
    {"group": "noisy_asr", "id": "N3", "user": "i want a restarant in the senter", "expected_type": "explicit"},
    {"group": "noisy_asr", "id": "N4", "user": "book for to people at seven", "expected_type": "follow_up"},
]


def normalize_slot_verbose(
    value: Optional[str],
    mapping: Dict[str, str],
    fuzzy_cutoff: float = 0.72,
    strip_punctuation: bool = False,
) -> Dict[str, Any]:
    """
    Hierarchical slot normalizer:
    1. Empty/None -> missing (conf=0.0)
    2. Exact dictionary match -> mapped (conf=1.0)
    3. Substring inclusion -> substring (conf=0.95)
    4. Unigram & bigram `difflib.SequenceMatcher` >= fuzzy_cutoff -> fuzzy (conf=best_score)
    5. Otherwise -> unknown (conf=0.0)
    """
    if not value or str(value).strip() == "":
        return {"raw": value, "value": None, "status": "missing", "confidence": 0.0, "candidates": []}

    val_str = str(value).strip().lower()

    if val_str in mapping:
        return {
            "raw": value,
            "value": mapping[val_str],
            "status": "mapped",
            "confidence": 1.0,
            "candidates": [mapping[val_str]],
        }

    candidates: List[str] = []
    for key, canonical in mapping.items():
        if key in val_str and canonical not in candidates:
            candidates.append(canonical)

    if candidates:
        return {
            "raw": value,
            "value": candidates[0],
            "status": "substring",
            "confidence": 0.95,
            "candidates": candidates,
        }

    if strip_punctuation:
        words = [w.strip(string.punctuation) for w in val_str.split() if w.strip(string.punctuation)]
    else:
        words = val_str.split()

    ngrams = words.copy()
    for i in range(len(words) - 1):
        ngrams.append(f"{words[i]} {words[i+1]}")

    best_match: Optional[str] = None
    best_score: float = 0.0

    for ngram in ngrams:
        for key in mapping:
            score = difflib.SequenceMatcher(None, ngram, key).ratio()
            if score > best_score:
                best_score = score
                best_match = key

    if best_match is not None and best_score >= fuzzy_cutoff:
        return {
            "raw": value,
            "value": mapping[best_match],
            "status": "fuzzy",
            "confidence": float(best_score),
            "candidates": [mapping[best_match]],
        }

    return {"raw": value, "value": None, "status": "unknown", "confidence": 0.0, "candidates": []}


def detect_slot_mention(text: str, hints: List[str]) -> bool:
    """Return True if any domain hint token/phrase appears in text."""
    return any(h in text for h in hints)


def parse_slot_with_hints(
    text: str,
    mapping: Dict[str, str],
    hints: List[str],
    fuzzy_cutoff: float = 0.72,
    strip_punctuation: bool = False,
) -> Dict[str, Any]:
    """Parse a slot and distinguish `unknown` (mentioned OOV) from `missing` (not mentioned)."""
    info = normalize_slot_verbose(
        text, mapping, fuzzy_cutoff=fuzzy_cutoff, strip_punctuation=strip_punctuation
    )
    if info["status"] == "unknown":
        if detect_slot_mention(text, hints):
            info["mentioned"] = True
        else:
            info["status"] = "missing"
            info["mentioned"] = False
    else:
        info["mentioned"] = info["status"] != "missing"
    return info


def parse_user_constraints(
    user_text: str, fuzzy_cutoff: float = 0.72, strip_punctuation: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Parse area, food, and pricerange slot hypotheses from raw user utterance."""
    text = str(user_text).lower()
    return {
        "area": parse_slot_with_hints(text, AREA_MAP, AREA_HINTS, fuzzy_cutoff, strip_punctuation),
        "food": parse_slot_with_hints(text, FOOD_MAP, FOOD_HINTS, fuzzy_cutoff, strip_punctuation),
        "pricerange": parse_slot_with_hints(text, PRICE_MAP, PRICE_HINTS, fuzzy_cutoff, strip_punctuation),
    }


def init_state() -> Dict[str, Any]:
    """Initialize clean task-oriented dialogue state."""
    return {
        "constraints": {"area": None, "food": None, "pricerange": None, "name": None},
        "booking": {
            "restaurant_id": None,
            "day": None,
            "time": None,
            "people": None,
            "customer_name": None,
        },
        "pending_clarification": None,
        "last_results": [],
    }


def get_next_missing_slot(state: Dict[str, Any], exclude_slot: Optional[str] = None) -> Optional[str]:
    """Return the first uninstantiated search constraint slot in canonical order (area -> food -> pricerange)."""
    for slot in ["area", "food", "pricerange"]:
        if slot == exclude_slot:
            continue
        if state["constraints"].get(slot) is None:
            return slot
    return None


def decide_clarification(
    state: Dict[str, Any],
    parsed_slots: Dict[str, Dict[str, Any]],
    low_conf: float = 0.84,
    medium_conf: float = 0.94,
) -> Optional[Dict[str, Any]]:
    """
    Deterministic clarification policy evaluating confidence and OOV status:
    - unknown + mentioned -> explicit_clarification (requires_options=True)
    - conf < low_conf -> explicit_clarification (requires_options=False)
    - low_conf <= conf < medium_conf -> implicit_clarification
    - missing slot -> follow_up
    - all slots filled with high confidence -> None
    """
    for slot in ["area", "food", "pricerange"]:
        info = parsed_slots.get(slot, {})
        status = info.get("status", "missing")
        conf = info.get("confidence", 0.0)

        if status == "unknown" and info.get("mentioned", False):
            return {
                "type": "explicit_clarification",
                "slot": slot,
                "info": info,
                "requires_options": True,
            }

        if status in ["fuzzy", "substring", "mapped"]:
            if conf < low_conf:
                return {
                    "type": "explicit_clarification",
                    "slot": slot,
                    "info": info,
                    "requires_options": False,
                }
            elif conf < medium_conf:
                return {
                    "type": "implicit_clarification",
                    "slot": slot,
                    "info": info,
                }

    missing = get_next_missing_slot(state)
    if missing:
        return {"type": "follow_up", "slot": missing, "info": None}

    return None


def slot_values_text(slot: str) -> str:
    if slot == "area":
        return "centre, north, south, east, west, or riverside"
    if slot == "food":
        return "chinese, italian, indian, japanese, vegetarian, or british"
    if slot == "pricerange":
        return "cheap, moderate, or expensive"
    return ""


def follow_up_question(slot: str) -> str:
    if slot == "area":
        return "What area are you looking for?"
    if slot == "food":
        return "What kind of food would you like?"
    if slot == "pricerange":
        return "What price range would you like?"
    return "Could you tell me a bit more?"


def build_clarification_response(clf: Dict[str, Any]) -> Dict[str, str]:
    """Build natural language clarification or follow-up response from policy decision."""
    ctype = clf.get("type")
    slot = str(clf.get("slot", ""))
    info = clf.get("info") or {}

    if ctype == "follow_up":
        return {"action": "follow_up", "response": follow_up_question(slot)}

    if ctype == "explicit_clarification":
        if clf.get("requires_options"):
            options = slot_values_text(slot)
            return {
                "action": "explicit_clarification",
                "response": f"I didn't catch the exact {slot}. You can choose {options}.",
            }
        guess = info.get("value", "that")
        return {
            "action": "explicit_clarification",
            "response": f"I think you said {guess}. Is that right?",
        }

    if ctype == "implicit_clarification":
        guess = info.get("value", "")
        if slot == "food":
            response_text = f"OK, in what area are you looking for a {guess} restaurant?"
        elif slot == "area":
            response_text = f"OK, what kind of food would you like in the {guess} area?"
        else:
            response_text = f"OK, what kind of food would you like in the {guess} price range?"
        return {"action": "implicit_clarification", "response": response_text}

    return {"action": "follow_up", "response": "I couldn't understand. Could you please repeat it again?"}


def update_state_from_parsed(
    state: Dict[str, Any],
    parsed_slots: Dict[str, Dict[str, Any]],
    low_conf: float = 0.84,
) -> Dict[str, Any]:
    """
    Optimistic safe state update:
    Commits mapped and substring matches immediately, plus fuzzy matches with confidence >= low_conf.
    Blocks low-confidence fuzzy guesses and unknown/OOV values from polluting tracked state.
    """
    for slot, info in parsed_slots.items():
        status = info.get("status", "missing")
        conf = info.get("confidence", 0.0)
        if status in ["mapped", "substring"] and info.get("value"):
            state["constraints"][slot] = info["value"]
        elif status == "fuzzy" and conf >= low_conf and info.get("value"):
            state["constraints"][slot] = info["value"]
    return state


def dm_preprocess(
    user_text: str,
    state: Dict[str, Any],
    low_conf: float = 0.84,
    medium_conf: float = 0.94,
    strip_punctuation: bool = False,
) -> Tuple[Optional[Dict[str, str]], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Controller pipeline executing:
    1. Slot constraint parsing
    2. Safe optimistic state update
    3. Clarification policy evaluation
    4. Pending clarification state locking (only for explicit clarifications)
    """
    parsed = parse_user_constraints(user_text, strip_punctuation=strip_punctuation)
    state = update_state_from_parsed(state, parsed, low_conf=low_conf)
    clf = decide_clarification(state, parsed, low_conf=low_conf, medium_conf=medium_conf)

    if clf is not None:
        if clf.get("type") == "explicit_clarification":
            state["pending_clarification"] = clf
        else:
            state["pending_clarification"] = None
        response_obj = build_clarification_response(clf)
        return response_obj, state, parsed

    state["pending_clarification"] = None
    return None, state, parsed


def classify_reply(reply_text: str) -> str:
    """Classify generated DM response into {'implicit', 'follow_up', 'explicit', 'other'}."""
    lower = reply_text.lower().strip()
    if (
        lower.startswith("ok, in what area")
        or lower.startswith("ok, what kind of food")
        or lower.startswith("ok, what price range")
    ):
        return "implicit"
    if (
        lower.startswith("what area")
        or lower.startswith("what kind of food")
        or lower.startswith("what price range")
    ):
        return "follow_up"
    if (
        lower.startswith("i didn't catch")
        or lower.startswith("i think you said")
        or lower.startswith("could you confirm")
    ):
        return "explicit"
    return "other"


def evaluate_clarification_benchmarks(
    low_conf: float = 0.84,
    medium_conf: float = 0.94,
    strip_punctuation: bool = False,
) -> Dict[str, Any]:
    """Run all 12 benchmark test cases (M1-M3, A1-A2, O1-O3, N1-N4) from Lab 9."""
    rows: List[Dict[str, Any]] = []
    for item in TEST_BENCHMARKS:
        state = init_state()
        dm_obj, state, parsed = dm_preprocess(
            item["user"],
            state,
            low_conf=low_conf,
            medium_conf=medium_conf,
            strip_punctuation=strip_punctuation,
        )
        reply = dm_obj["response"] if dm_obj else "Proceed to search_restaurants"
        pred_type = classify_reply(reply)
        rows.append(
            {
                "group": item["group"],
                "id": item["id"],
                "user": item["user"],
                "expected_type": item["expected_type"],
                "predicted_type": pred_type,
                "match": pred_type == item["expected_type"],
                "reply": reply,
            }
        )
    accuracy = sum(1 for r in rows if r["match"]) / len(rows) if rows else 0.0
    return {"accuracy": accuracy, "rows": rows}


def compute_expected_utility_policy(
    confidence: float,
    reward_success: float = 10.0,
    cost_corrupt_state: float = -12.0,
    cost_implicit_turn: float = -1.0,
    cost_explicit_turn: float = -2.5,
    cost_reject: float = -4.0,
) -> Dict[str, Any]:
    """
    Decision-theoretic Expected Utility (EU) model for dialogue grounding under uncertainty:
    - EU(Accept)   = p * R_success + (1 - p) * C_corrupt
    - EU(Implicit) = p * (R_success + C_implicit) + (1 - p) * (0.5 * R_success + 0.5 * C_corrupt + C_implicit)
    - EU(Explicit) = R_success + C_explicit  (resolves uncertainty at cost of an extra binary turn)
    - EU(Reject)   = C_reject
    """
    p = max(0.0, min(1.0, float(confidence)))
    eu_accept = p * reward_success + (1.0 - p) * cost_corrupt_state
    eu_implicit = p * (reward_success + cost_implicit_turn) + (1.0 - p) * (
        0.5 * reward_success + 0.5 * cost_corrupt_state + cost_implicit_turn
    )
    eu_explicit = reward_success + cost_explicit_turn
    eu_reject = cost_reject

    utilities = {
        "accept": eu_accept,
        "implicit_clarification": eu_implicit,
        "explicit_clarification": eu_explicit,
        "reject_reask": eu_reject,
    }
    optimal_action = max(utilities, key=utilities.get)
    return {"confidence": p, "utilities": utilities, "optimal_action": optimal_action}
