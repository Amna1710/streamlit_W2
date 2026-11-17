from uuid import UUID

from fastapi import APIRouter, Header, HTTPException

from rag_system.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    EvaluationRequest,
    EvaluationResponse,
)

from rag_system.chains.conversational_rag import ConversationalRAG
from rag_system.core.logging import get_logger
from rag_system.core.config import settings
from rag_system.core.user_identity import normalize_user_id

from rag_system.services.evaluation_service import (
    RAGEvaluationService,
)

from rag_system.services.retrieval_service import RetrievalService


logger = get_logger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

retrieval_service = RetrievalService()
rag_service = ConversationalRAG()


def get_normalized_user_id(x_user_id: str) -> str:
    """Validate and normalize the user ID from the request header."""

    try:
        return normalize_user_id(x_user_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


def validate_api_key(
    provider: str | None,
    api_key: str | None,
) -> None:
    """Require a request-specific key for Gemini and OpenAI."""

    selected_provider = (
        provider or settings.llm_provider
    ).lower().strip()

    if selected_provider in {"gemini", "openai"}:

        if not api_key or not api_key.strip():

            provider_name = (
                "Gemini"
                if selected_provider == "gemini"
                else "OpenAI"
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{provider_name} API key is required. "
                    "Enter your personal API key in Streamlit."
                ),
            )


@router.post(
    "/answer",
    response_model=ChatResponse,
    summary="Generate a conversational RAG answer",
    description=(
        "Rewrites follow-up questions, retrieves relevant document "
        "chunks, generates an answer, and returns source citations."
    ),
)
async def generate_answer(
    request: ChatRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_llm_api_key: str | None = Header(
        None,
        alias="X-LLM-API-Key",
    ),
):

    user_id = get_normalized_user_id(x_user_id)

    validate_api_key(
        provider=request.provider,
        api_key=x_llm_api_key,
    )

    try:

        result = rag_service.answer(
            query=request.query,
            user_id=user_id,
            conversation_id=request.conversation_id,
            k=request.k,
            provider=request.provider,
            api_key=x_llm_api_key,
        )

        return ChatResponse(
            query=request.query,
            rewritten_query=result["rewritten_query"],
            answer=result["answer"],
            user_id=user_id,
            conversation_id=request.conversation_id,
            total_sources=result["total_sources"],
            sources=result["sources"],
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:

        # Do not log exception text because provider exceptions
        # may contain sensitive request details.
        logger.error(
            "RAG answer generation failed for user_id=%s",
            user_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer. Check your provider and API key.",
        )


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
async def retrieve_documents(
    request: RetrievalRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    user_id = get_normalized_user_id(x_user_id)

    try:

        documents = retrieval_service.retrieve(
            query=request.query,
            user_id=user_id,
            k=request.k,
        )

        results = [
            RetrievedChunk(
                content=document.page_content,
                score=document.metadata.get("retrieval_score"),
                score_type=document.metadata.get("score_type"),
                metadata={
                    key: str(value)
                    for key, value in document.metadata.items()
                    if key not in {
                        "retrieval_score",
                        "score_type",
                        "reranker_score",
                        "rrf_score",
                        "bm25_score",
                    }
                },
            )
            for document in documents
        ]

        return RetrievalResponse(
            query=request.query,
            user_id=user_id,
            total_results=len(results),
            results=results,
        )

    except Exception:

        logger.error(
            "Document retrieval failed for user_id=%s",
            user_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed.",
        )


@router.delete(
    "/conversation/{conversation_id}",
)
async def clear_conversation(
    conversation_id: UUID,
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    user_id = get_normalized_user_id(x_user_id)

    deleted = rag_service.clear_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )

    return {
        "conversation_id": str(conversation_id),
        "cleared": deleted,
        "message": (
            "Conversation history cleared."
            if deleted
            else "Conversation history was not found."
        ),
    }


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
)
def evaluate_answer(
    request: EvaluationRequest,
    provider: str | None = None,
    x_llm_api_key: str | None = Header(
        None,
        alias="X-LLM-API-Key",
    ),
):

    validate_api_key(
        provider=provider,
        api_key=x_llm_api_key,
    )

    evaluation_service = RAGEvaluationService()

    try:

        result = evaluation_service.evaluate(
            question=request.question,
            answer=request.answer,
            context=request.context,
            provider=provider,
            api_key=x_llm_api_key,
        )

        return EvaluationResponse(**result)

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    except Exception:

        logger.exception("Answer evaluation failed.")

        raise HTTPException(

            status_code=500,

            detail="Answer evaluation failed. Check FastAPI logs.",

        )