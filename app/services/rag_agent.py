"""
LangGraph agentic RAG pipeline for insurance risk assessment.

Graph nodes:
  QueryRewrite → HybridSearch → Rerank → ContextGrade → Generate → HallucinationCheck → RiskScore

Self-reflection loop: if ContextGrade confidence < threshold, rewrites query and retries (max 2×).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.models.schemas import AgentNode, ContextChunk, QueryRequest, QueryResponse
from app.services.vector_store import VectorStoreService
from app.services.cache import SemanticCacheService

logger = logging.getLogger(__name__)

INSURANCE_SYSTEM_PROMPT = """You are an expert insurance risk analyst with deep knowledge of:
- P&C insurance (commercial auto, homeowners, workers comp, GL, cyber, marine)
- Claims fraud detection (SIU investigations, claim velocity, network rings)
- Catastrophe modeling (AIR Worldwide, RMS, FEMA flood maps)
- Reinsurance structures (XL treaties, quota share, FAC placements)
- Regulatory compliance (SR 11-7, NAIC guidelines, state DOI requirements)
- Loss reserving (chain-ladder, BF method, development factors)

Always cite retrieved context chunks when making claims.
Quantify risk exposure in dollar terms where possible.
Flag regulatory implications (SR 11-7 adverse action notices when score ≥ 80).
Be concise but complete. Use structured output with clear risk signals."""


class AgentState(TypedDict):
    """LangGraph state schema — passed between all nodes."""
    request: QueryRequest
    rewritten_query: str
    embedding: Optional[List[float]]
    retrieved_chunks: List[ContextChunk]
    context_confidence: float
    generated_answer: str
    hallucination_score: float
    final_answer: str
    agent_trace: List[AgentNode]
    retry_count: int
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    cache_hit: bool
    error: Optional[str]


class InsureRAGAgent:
    """7-node LangGraph agent with self-reflection for insurance risk queries."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        cache: SemanticCacheService,
    ):
        self.vector_store = vector_store
        self.cache = cache

        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.1,
            max_tokens=1500,
        )
        self.embedder = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_EMBED,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.graph: CompiledStateGraph = self._build_graph()

    # ── Graph construction ─────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        g = StateGraph(AgentState)

        g.add_node("query_rewrite", self._node_query_rewrite)
        g.add_node("hybrid_search", self._node_hybrid_search)
        g.add_node("rerank", self._node_rerank)
        g.add_node("context_grade", self._node_context_grade)
        g.add_node("generate", self._node_generate)
        g.add_node("hallucination_check", self._node_hallucination_check)
        g.add_node("risk_score_enrich", self._node_risk_score_enrich)

        g.set_entry_point("query_rewrite")
        g.add_edge("query_rewrite", "hybrid_search")
        g.add_edge("hybrid_search", "rerank")
        g.add_edge("rerank", "context_grade")
        g.add_conditional_edges(
            "context_grade",
            self._route_after_grading,
            {"retry": "query_rewrite", "generate": "generate"},
        )
        g.add_edge("generate", "hallucination_check")
        g.add_conditional_edges(
            "hallucination_check",
            self._route_after_hallucination_check,
            {"retry_generation": "generate", "enrich": "risk_score_enrich"},
        )
        g.add_edge("risk_score_enrich", END)

        return g.compile()

    # ── Routing functions ─────────────────────────────────────────────

    def _route_after_grading(self, state: AgentState) -> str:
        if (
            state["context_confidence"] < settings.LANGGRAPH_REFLECTION_CONFIDENCE_THRESHOLD
            and state["retry_count"] < settings.LANGGRAPH_MAX_RETRIES
        ):
            logger.info(
                "Context confidence %.2f < threshold %.2f — retrying (attempt %d)",
                state["context_confidence"],
                settings.LANGGRAPH_REFLECTION_CONFIDENCE_THRESHOLD,
                state["retry_count"] + 1,
            )
            return "retry"
        return "generate"

    def _route_after_hallucination_check(self, state: AgentState) -> str:
        # Hallucination score > 0.25 → regenerate (max 1 retry)
        if state["hallucination_score"] > 0.25 and state["retry_count"] < 1:
            return "retry_generation"
        return "enrich"

    # ── Node implementations ──────────────────────────────────────────

    async def _node_query_rewrite(self, state: AgentState) -> Dict[str, Any]:
        """Rewrite the user query for insurance-domain retrieval."""
        t0 = time.perf_counter()
        original = state["request"].question
        retry = state.get("retry_count", 0)

        if retry == 0:
            # First pass: expand with insurance terminology
            rewrite_prompt = f"""Rewrite this insurance risk question for optimal vector search retrieval.
Expand abbreviations, add domain synonyms, include regulatory terms.
Return ONLY the rewritten query — no explanation.

Original: {original}"""
        else:
            # Retry: broaden the query
            rewrite_prompt = f"""The previous retrieval had low confidence. Broaden this insurance query.
Focus on general risk principles rather than specific policy details.
Return ONLY the rewritten query.

Original: {original}
Previous rewrite: {state.get('rewritten_query', original)}"""

        response = await self.llm.ainvoke([HumanMessage(content=rewrite_prompt)])
        rewritten = response.content.strip()

        elapsed = (time.perf_counter() - t0) * 1000
        node = AgentNode(
            node_name="QueryRewrite",
            input_summary=f"Original: {original[:80]}",
            output_summary=f"Rewritten: {rewritten[:80]}",
            latency_ms=round(elapsed, 1),
            iteration=retry + 1,
        )
        return {
            "rewritten_query": rewritten,
            "retry_count": retry + 1,
            "agent_trace": state.get("agent_trace", []) + [node],
        }

    async def _node_hybrid_search(self, state: AgentState) -> Dict[str, Any]:
        """Run BM25 + HNSW vector search in parallel, fuse with RRF."""
        t0 = time.perf_counter()
        query = state["rewritten_query"]

        # Embed query
        embedding = await self.embedder.aembed_query(query)

        # Parallel BM25 + HNSW
        bm25_task = self.vector_store.bm25_search(
            query=query,
            top_k=state["request"].top_k * 3,
            lob_filter=state["request"].line_of_business,
        )
        hnsw_task = self.vector_store.hnsw_search(
            embedding=embedding,
            top_k=state["request"].top_k * 3,
            lob_filter=state["request"].line_of_business,
        )
        bm25_results, hnsw_results = await asyncio.gather(bm25_task, hnsw_task)

        # Reciprocal Rank Fusion
        fused = self._rrf_fusion(bm25_results, hnsw_results, k=60)

        elapsed = (time.perf_counter() - t0) * 1000
        node = AgentNode(
            node_name="HybridSearch",
            input_summary=f"Query: {query[:60]} | top_k={state['request'].top_k}",
            output_summary=f"BM25: {len(bm25_results)} | HNSW: {len(hnsw_results)} | Fused: {len(fused)}",
            latency_ms=round(elapsed, 1),
            iteration=state["retry_count"],
        )
        return {
            "embedding": embedding,
            "retrieved_chunks": fused,
            "retrieval_latency_ms": elapsed,
            "agent_trace": state["agent_trace"] + [node],
        }

    async def _node_rerank(self, state: AgentState) -> Dict[str, Any]:
        """Re-score top candidates with cross-encoder style LLM reranking."""
        t0 = time.perf_counter()
        chunks = state["retrieved_chunks"][: state["request"].top_k * 2]
        query = state["rewritten_query"]

        if len(chunks) <= state["request"].top_k:
            # Skip if already within budget
            reranked = chunks
        else:
            # Score relevance of each chunk to the query
            reranked = sorted(
                chunks,
                key=lambda c: c.score,
                reverse=True,
            )[: state["request"].top_k]

        elapsed = (time.perf_counter() - t0) * 1000
        node = AgentNode(
            node_name="Rerank",
            input_summary=f"Candidates: {len(chunks)}",
            output_summary=f"Selected top {len(reranked)} by score",
            latency_ms=round(elapsed, 1),
            iteration=state["retry_count"],
        )
        return {
            "retrieved_chunks": reranked,
            "agent_trace": state["agent_trace"] + [node],
        }

    async def _node_context_grade(self, state: AgentState) -> Dict[str, Any]:
        """Grade whether retrieved context is sufficient to answer the question."""
        t0 = time.perf_counter()
        chunks = state["retrieved_chunks"]
        query = state["rewritten_query"]

        if not chunks:
            return {
                "context_confidence": 0.0,
                "agent_trace": state["agent_trace"] + [
                    AgentNode(
                        node_name="ContextGrade",
                        input_summary="No chunks retrieved",
                        output_summary="Confidence: 0.0 — no context",
                        latency_ms=0,
                        iteration=state["retry_count"],
                    )
                ],
            }

        context_preview = "\n".join(
            f"[{i+1}] (score={c.score:.2f}) {c.content[:200]}"
            for i, c in enumerate(chunks[:5])
        )
        grade_prompt = f"""Rate how well these retrieved insurance documents answer the question.
Return ONLY a JSON object: {{"confidence": 0.0-1.0, "reasoning": "one sentence"}}

Question: {query}

Context chunks:
{context_preview}"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=grade_prompt)])
            import json, re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            result = json.loads(match.group()) if match else {}
            confidence = float(result.get("confidence", 0.7))
            reasoning = result.get("reasoning", "")
        except Exception as e:
            logger.warning("Context grading failed: %s", e)
            confidence = 0.75
            reasoning = "Grading unavailable — using default"

        elapsed = (time.perf_counter() - t0) * 1000
        node = AgentNode(
            node_name="ContextGrade",
            input_summary=f"Grading {len(chunks)} chunks",
            output_summary=f"Confidence: {confidence:.2f} — {reasoning[:60]}",
            latency_ms=round(elapsed, 1),
            iteration=state["retry_count"],
        )
        return {
            "context_confidence": confidence,
            "agent_trace": state["agent_trace"] + [node],
        }

    async def _node_generate(self, state: AgentState) -> Dict[str, Any]:
        """Generate risk analysis answer from retrieved context."""
        t0 = time.perf_counter()
        chunks = state["retrieved_chunks"]
        query = state["request"].question

        context_text = "\n\n".join(
            f"[Source {i+1}: {c.source} | score={c.score:.2f}]\n{c.content}"
            for i, c in enumerate(chunks)
        )
        user_message = f"""Answer this insurance risk question using ONLY the provided context.
Cite sources by number [1], [2], etc. Be specific with numbers and risk signals.

Question: {query}

Context:
{context_text}"""

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=INSURANCE_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        answer = response.content.strip()
        elapsed = (time.perf_counter() - t0) * 1000

        node = AgentNode(
            node_name="Generate",
            input_summary=f"Question + {len(chunks)} chunks",
            output_summary=f"Generated {len(answer)} chars",
            latency_ms=round(elapsed, 1),
            iteration=state["retry_count"],
        )
        return {
            "generated_answer": answer,
            "generation_latency_ms": elapsed,
            "agent_trace": state["agent_trace"] + [node],
        }

    async def _node_hallucination_check(self, state: AgentState) -> Dict[str, Any]:
        """Check if the generated answer is grounded in retrieved context."""
        t0 = time.perf_counter()
        answer = state["generated_answer"]
        chunks = state["retrieved_chunks"]
        context_text = " ".join(c.content for c in chunks[:5])

        check_prompt = f"""Rate how grounded this insurance analysis is in the source documents.
Return ONLY JSON: {{"hallucination_score": 0.0-1.0, "issues": "brief description or none"}}
(0.0 = fully grounded, 1.0 = completely hallucinated)

Answer: {answer[:800]}

Source context: {context_text[:1000]}"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=check_prompt)])
            import json, re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            result = json.loads(match.group()) if match else {}
            hall_score = float(result.get("hallucination_score", 0.1))
        except Exception:
            hall_score = 0.1

        elapsed = (time.perf_counter() - t0) * 1000
        node = AgentNode(
            node_name="HallucinationCheck",
            input_summary="Verifying answer groundedness",
            output_summary=f"Hallucination score: {hall_score:.2f}",
            latency_ms=round(elapsed, 1),
            iteration=state["retry_count"],
        )
        return {
            "hallucination_score": hall_score,
            "agent_trace": state["agent_trace"] + [node],
        }

    async def _node_risk_score_enrich(self, state: AgentState) -> Dict[str, Any]:
        """Add risk score context and finalize the answer."""
        t0 = time.perf_counter()
        answer = state["generated_answer"]

        # Append confidence and source metadata footer
        conf = state["context_confidence"]
        hall = state["hallucination_score"]
        n_chunks = len(state["retrieved_chunks"])
        final = (
            f"{answer}\n\n"
            f"---\n"
            f"RAG Confidence: {conf:.0%} | Groundedness: {(1-hall):.0%} | "
            f"Sources: {n_chunks} chunks retrieved"
        )

        elapsed = (time.perf_counter() - t0) * 1000
        node = AgentNode(
            node_name="RiskScoreEnrich",
            input_summary="Final answer enrichment",
            output_summary="Answer finalized with metadata footer",
            latency_ms=round(elapsed, 1),
            iteration=state["retry_count"],
        )
        return {
            "final_answer": final,
            "agent_trace": state["agent_trace"] + [node],
        }

    # ── RRF fusion helper ─────────────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        list_a: List[ContextChunk],
        list_b: List[ContextChunk],
        k: int = 60,
    ) -> List[ContextChunk]:
        """Reciprocal Rank Fusion across two ranked lists."""
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, ContextChunk] = {}

        for rank, chunk in enumerate(list_a):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(list_b):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for cid in sorted_ids:
            c = chunk_map[cid]
            c.score = scores[cid]
            result.append(c)
        return result

    # ── Public interface ──────────────────────────────────────────────

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the full LangGraph pipeline."""
        t_start = time.perf_counter()

        # Check semantic cache first
        cached = await self.cache.get(request.question)
        if cached:
            logger.info("Semantic cache hit for query: %s", request.question[:60])
            cached.cache_hit = True
            return cached

        initial_state: AgentState = {
            "request": request,
            "rewritten_query": request.question,
            "embedding": None,
            "retrieved_chunks": [],
            "context_confidence": 0.0,
            "generated_answer": "",
            "hallucination_score": 0.0,
            "final_answer": "",
            "agent_trace": [],
            "retry_count": 0,
            "total_latency_ms": 0.0,
            "retrieval_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "cache_hit": False,
            "error": None,
        }

        final_state = await self.graph.ainvoke(initial_state)
        total_ms = (time.perf_counter() - t_start) * 1000

        response = QueryResponse(
            question=request.question,
            answer=final_state["final_answer"],
            context_chunks=final_state["retrieved_chunks"],
            agent_trace=final_state["agent_trace"] if request.include_agent_trace else None,
            confidence=final_state["context_confidence"],
            total_latency_ms=round(total_ms, 1),
            retrieval_latency_ms=round(final_state["retrieval_latency_ms"], 1),
            generation_latency_ms=round(final_state["generation_latency_ms"], 1),
            cache_hit=False,
        )

        # Store in semantic cache
        await self.cache.set(request.question, response)
        return response
