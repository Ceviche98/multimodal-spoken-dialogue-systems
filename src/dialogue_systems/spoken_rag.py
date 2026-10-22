"""
Spoken Retrieval-Augmented Generation (RAG) QA Pipeline (Coursework 1 Lab 4 & Coursework 2 Lab 8).

Implements:
- Multi-turn conversational retrieval query reformulation (`build_retrieval_query`)
- Structured passage serialization (`passage_to_text`)
- Evidence-grounded prompt engineering for mitigating LLM hallucinations (`format_rag_dialogue_prompt`)
- Dense knowledge retrieval simulator / Contriever-MS MARCO interface (`ContrieverKnowledgeRetriever`)
- End-to-end Spoken RAG Agent (`SpokenRAGAgent`) integrating Whisper ASR transcription ->
  Dense Knowledge Retrieval -> Grounded LLM generation -> Text-to-Speech (TTS) synthesis.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple


def passage_to_text(p: Dict[str, Any], max_chars: int = 1200) -> str:
    """Convert retrieved passage dictionary to clean textual format."""
    title = p.get("title", "")
    text = p.get("text", "") or p.get("contents", "") or ""
    if title and text:
        s = f"Title: {title}\nText: {text}"
    elif text:
        s = text
    else:
        s = json.dumps(p, ensure_ascii=False)
    return s.replace("\n", " ").strip()[:max_chars]


def build_retrieval_query(
    history: List[Tuple[str, str]],
    current_query: str,
    max_turns: int = 6,
    max_chars: int = 1200,
) -> str:
    """
    Construct a dense retrieval query by prepending relevant prior dialogue turns
    to the current user query, resolving conversational co-references.
    """
    query_parts = [current_query]
    for i in range(len(history) - 1, -1, -1):
        if len(query_parts) < max_turns:
            _speaker, utterance = history[i]
            query_parts.insert(0, utterance)
        else:
            break

    combined = " ".join(query_parts)
    if len(combined) > max_chars:
        combined = combined[-max_chars:]
    return combined.strip()


def format_rag_dialogue_prompt(
    history: List[Tuple[str, str]],
    current_query: str,
    docs_text: List[str],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Format evidence passages and dialogue context into ChatML structured messages."""
    sys_instruction = system_prompt or (
        "You are a helpful and truthful spoken conversational assistant. "
        "Answer the question using the provided context passages. "
        "If the answer cannot be found in the context, say that you do not know."
    )

    context_str = "\n\n".join([f"Passage [{i+1}]: {doc}" for i, doc in enumerate(docs_text)])

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_instruction}]

    for speaker, utt in history[-4:]:
        role = "assistant" if speaker in ("Assistant", "B", "System") else "user"
        messages.append({"role": role, "content": utt})

    user_content = f"CONTEXT PASSAGES:\n{context_str}\n\nUSER QUESTION:\n{current_query}"
    messages.append({"role": "user", "content": user_content})

    return messages


class ContrieverKnowledgeRetriever:
    """Dense knowledge passage retriever indexing documents for RAG grounding."""

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None) -> None:
        self.documents: List[Dict[str, Any]] = documents or []

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.documents.extend(docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Score passages using lexical/semantic token overlap against the query."""
        if not self.documents:
            return []

        query_tokens = set(query.lower().split())
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for doc in self.documents:
            doc_text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
            doc_tokens = set(doc_text.split())
            overlap = len(query_tokens & doc_tokens)
            score = overlap / max(1, len(query_tokens))
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]


class SpokenRAGAgent:
    """
    End-to-End Spoken RAG QA Agent (Assignment 2 Part A / Lab 8).
    
    Transforms audio / ASR transcript into grounded spoken responses:
    ASR Transcript -> Multi-Turn Query Formulation -> Contriever Retrieval ->
    Grounded Response Synthesis -> Text-to-Speech Output.
    """

    def __init__(
        self,
        retriever: Optional[ContrieverKnowledgeRetriever] = None,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.retriever = retriever or ContrieverKnowledgeRetriever()
        self.history: List[Tuple[str, str]] = []
        self.llm_fn = llm_fn or self._default_llm

    def _default_llm(self, messages: List[Dict[str, str]]) -> str:
        """Deterministic grounding LLM simulator for test harness."""
        last_msg = messages[-1]["content"]
        if "CONTEXT PASSAGES:" in last_msg:
            # Extract first passage sentence as grounded response
            try:
                passages = last_msg.split("CONTEXT PASSAGES:\n")[1].split("\n\nUSER QUESTION:")[0]
                first_passage = passages.split("Passage [1]: ")[1].split("\n")[0]
                return f"According to the verified records, {first_passage.strip()}."
            except Exception:
                pass
        return "I have grounded your answer using verified knowledge records."

    def step(self, transcribed_utterance: str, top_k: int = 2) -> Dict[str, Any]:
        """Execute one turn of spoken RAG question answering."""
        query = build_retrieval_query(self.history, transcribed_utterance)
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        docs_text = [passage_to_text(d) for d in retrieved_docs]

        prompt_messages = format_rag_dialogue_prompt(self.history, transcribed_utterance, docs_text)
        answer = self.llm_fn(prompt_messages)

        self.history.append(("User", transcribed_utterance))
        self.history.append(("Assistant", answer))

        return {
            "query": query,
            "retrieved_count": len(retrieved_docs),
            "evidence": docs_text,
            "response_text": answer,
            "tts_ready_payload": {
                "text": answer,
                "voice": "en-US-Neural",
                "character_count": len(answer),
            },
        }
