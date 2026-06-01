"""RAG Query endpoint."""
from fastapi import APIRouter, Depends, Request
from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_agent import InsureRAGAgent
from app.services.cache import SemanticCacheService

router = APIRouter()


def get_agent(request: Request) -> InsureRAGAgent:
    vs = request.app.state.vector_store
    cache = SemanticCacheService()
    return InsureRAGAgent(vector_store=vs, cache=cache)


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Run a RAG query against the insurance knowledge base",
)
async def rag_query(body: QueryRequest, agent: InsureRAGAgent = Depends(get_agent)) -> QueryResponse:
    return await agent.run(body)
