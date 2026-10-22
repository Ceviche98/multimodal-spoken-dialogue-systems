"""Unit tests for the multimodal_spoken_dialogue_systems package."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

from dialogue_systems.spoken_asr import (
    compute_cer,
    compute_semantic_similarity,
    compute_wer,
    load_asr_evaluation_results,
    normalize_transcript,
    summarize_asr_models,
)
from dialogue_systems.clarification_dm import (
    AREA_MAP,
    FOOD_MAP,
    PRICE_MAP,
    compute_expected_utility_policy,
    dm_preprocess,
    evaluate_clarification_benchmarks,
    init_state,
    normalize_slot_verbose,
)
from dialogue_systems.multiparty import (
    NextSpeakerBiLSTMModel,
    build_multiparty_dataset,
    build_turns,
    score_multiparty_turn_taking,
)
from dialogue_systems.multimodal_grounding import (
    build_candidate_description,
    evaluate_oracle_outputs,
    extract_selector_answer,
    normalize_oracle_answer,
    score_multimodal_referent_fusion,
    to_qwen_coord,
)
from dialogue_systems.dialogue_act import (
    BiLSTMModel,
    CNN_BiLSTM,
    classify_utterance_heuristic,
    damsl_act_tag,
    pad_sequences,
)
from dialogue_systems.seq2seq_generator import (
    BahdanauAttention,
    Decoder,
    DecoderAdvanced,
    Encoder,
    greedy_decode_Ids,
    preprocess_utterance,
)


def test_spoken_asr_metrics_and_csv() -> None:
    assert normalize_transcript("Hello, World! 123") == "hello world 123"
    assert compute_wer("dense vectors", "dense vectors") == 0.0
    assert compute_wer("a b c d", "a b x d") == 0.25
    assert compute_cer("abcd", "abxd") == 0.25
    assert compute_semantic_similarity("dense vectors", "dense vectors") == 1.0

    csv_path = Path(__file__).resolve().parents[1] / "data" / "asr_evaluation_results.csv"
    df = load_asr_evaluation_results(str(csv_path))
    assert len(df) == 20
    summaries = summarize_asr_models(df)
    assert abs(summaries["Base"]["mean_wer"] * 100 - 15.24) < 0.05
    assert abs(summaries["Distil"]["mean_wer"] * 100 - 7.51) < 0.05
    assert abs(summaries["Turbo"]["mean_wer"] * 100 - 1.77) < 0.05


def test_clarification_dm_normalizer_and_benchmark() -> None:
    exact = normalize_slot_verbose("I want a restaurant in the centre", AREA_MAP)
    assert exact["value"] == "centre"
    assert exact["confidence"] == 0.95  # substring match inside full sentence
    assert exact["status"] == "substring"

    exact_word = normalize_slot_verbose("centre", AREA_MAP)
    assert exact_word["value"] == "centre"
    assert exact_word["confidence"] == 1.0
    assert exact_word["status"] == "mapped"

    fuzzy = normalize_slot_verbose("I want chainese food.", FOOD_MAP)
    assert fuzzy["value"] == "chinese"
    assert 0.84 <= fuzzy["confidence"] < 0.94
    assert fuzzy["status"] == "fuzzy"

    state = init_state()
    response_obj, state, parsed = dm_preprocess("I want chainese food.", state)
    assert response_obj is not None
    assert response_obj["action"] == "implicit_clarification"
    assert "chinese" in response_obj["response"].lower()

    eu = compute_expected_utility_policy(0.90)
    assert eu["optimal_action"] == "implicit_clarification"

    bench = evaluate_clarification_benchmarks(low_conf=0.84, strip_punctuation=False)
    assert len(bench["rows"]) == 12
    assert abs(bench["accuracy"] - 11 / 12) < 1e-6


def test_multiparty_merging_and_hierarchical_bilstm() -> None:
    words_df = pd.DataFrame(
        [
            {"meeting_id": "M1", "speaker": "A", "start": 0.0, "end": 1.0, "text": "Hello"},
            {"meeting_id": "M1", "speaker": "A", "start": 1.2, "end": 2.0, "text": "everyone"},
            {"meeting_id": "M1", "speaker": "B", "start": 2.1, "end": 3.0, "text": "Hi"},
            {"meeting_id": "M1", "speaker": "C", "start": 3.2, "end": 4.0, "text": "There"},
        ]
    )
    turns_df = build_turns(words_df, max_gap=0.5)
    assert len(turns_df) == 3
    assert turns_df.iloc[0]["utt_text"] == "Hello everyone"

    ds = build_multiparty_dataset(turns_df, history=1)
    assert len(ds) == 1

    model = NextSpeakerBiLSTMModel(
        vocab_size=100,
        history=2,
        emb_dim=16,
        word_hidden=16,
        ctx_hidden=16,
        spk_emb_dim=8,
    )
    history_tokens = [torch.randint(1, 100, (2, 6)), torch.randint(1, 100, (2, 6))]
    history_lengths = [torch.full((2,), 6, dtype=torch.long), torch.full((2,), 6, dtype=torch.long)]
    history_speakers = torch.randint(0, 4, (2, 2))
    current_tokens = torch.randint(1, 100, (2, 6))
    current_lengths = torch.full((2,), 6, dtype=torch.long)
    current_speaker = torch.randint(0, 4, (2,))

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


def test_dialogue_act_taggers() -> None:
    assert damsl_act_tag("qy^d") == "qy^d"
    assert damsl_act_tag("qr") == "qy"
    padded = pad_sequences([[1, 2, 3]], maxlen=5)
    assert padded == [[1, 2, 3, 0, 0]]

    x = torch.randint(1, 50, (3, 12))
    bilstm = BiLSTMModel(vocab_size=50, embed_size=16, hidden_size=10, max_length=12)
    out1 = bilstm(x)
    assert out1.shape == (3, 10)

    cnn_bilstm = CNN_BiLSTM(
        vocab_size=50,
        embed_size=16,
        hidden_size=10,
        filter_sizes=[3, 4, 5],
        num_filters=8,
    )
    out2 = cnn_bilstm(x)
    assert out2.shape == (3, 10)

    heur = classify_utterance_heuristic("Could you repeat that please?")
    assert heur in ("br", "qy", "qw")


def test_seq2seq_generator_models() -> None:
    assert preprocess_utterance("I'm won't can't!") == "i am will not cannot"
    vocab_size = 40
    enc = Encoder(vocab_size=vocab_size, embedding_dim=16, enc_units=16, batch_size=2)
    dec_van = Decoder(vocab_size=vocab_size, embedding_dim=16, dec_units=16, batch_size=2)
    dec_adv = DecoderAdvanced(vocab_size=vocab_size, embedding_dim=16, dec_units=16, batch_size=2)

    attn = BahdanauAttention(units=32)
    src = torch.randint(1, vocab_size, (2, 7))
    enc_out, h_f, h_b = enc(src)
    h_cat = torch.cat([h_f, h_b], dim=-1)
    assert enc_out.shape == (2, 7, 32)
    assert h_cat.shape == (2, 32)

    ctx, weights = attn(h_cat, enc_out)
    assert ctx.shape == (2, 32)
    assert weights.shape == (2, 7, 1)

    step_in = torch.randint(1, vocab_size, (2, 1))
    logits_van, _, _ = dec_van(step_in, h_cat, enc_out)
    logits_adv, _, _ = dec_adv(step_in, h_cat, enc_out)
    assert logits_van.shape == (2, vocab_size)
    assert logits_adv.shape == (2, vocab_size)

    gen_ids, attn_mat = greedy_decode_Ids(enc, dec_adv, src[:1], max_len=5)
    assert isinstance(gen_ids, list)
    assert attn_mat.dim() == 3


def test_spoken_rag_pipeline() -> None:
    from dialogue_systems import (
        ContrieverKnowledgeRetriever,
        SpokenRAGAgent,
        build_retrieval_query,
        format_rag_dialogue_prompt,
        passage_to_text,
    )

    doc1 = {"title": "Apollo 11", "text": "Apollo 11 landed on the Moon on July 20, 1969."}
    doc2 = {"title": "Mars Rover", "text": "Perseverance landed in Jezero crater on Mars."}
    p_text = passage_to_text(doc1)
    assert "Title: Apollo 11" in p_text
    assert "Moon" in p_text

    history = [("User", "Tell me about space missions"), ("Assistant", "We have Apollo and Mars records")]
    rq = build_retrieval_query(history, "When did Apollo land?")
    assert "Apollo" in rq

    retriever = ContrieverKnowledgeRetriever([doc1, doc2])
    results = retriever.retrieve("When did Apollo 11 land on the Moon?", top_k=1)
    assert len(results) == 1
    assert results[0]["title"] == "Apollo 11"

    agent = SpokenRAGAgent(retriever=retriever)
    step_out = agent.step("When did Apollo 11 land?", top_k=1)
    assert step_out["retrieved_count"] == 1
    assert "Apollo 11 landed on the Moon" in step_out["response_text"]
    assert step_out["tts_ready_payload"]["text"] == step_out["response_text"]

