from fastapi import (
    APIRouter,
    File,
    Header,
    HTTPException,
    UploadFile,
)

from rag_system.schemas.document import DocumentUploadResponse
from rag_system.services.ingestion_service import (
    DocumentIngestionService,
)
from rag_system.core.user_identity import (
    generate_user_id,
    normalize_user_id,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

ingestion_service = DocumentIngestionService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload and process a document",
    description=(
        "Uploads a document, extracts its text, creates chunks, "
        "and stores the chunks in the configured vector database."
    ),
)
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-ID",
    ),
):

    if x_user_id is None or not x_user_id.strip():
        user_id = generate_user_id()

    else:
        try:
            user_id = normalize_user_id(x_user_id)

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    metadata, chunks_created = (
        await ingestion_service.ingest_document(
            file=file,
            user_id=user_id,
        )
    )

    return DocumentUploadResponse(
        document_id=metadata.document_id,
        user_id=user_id,
        filename=metadata.filename,
        file_type=metadata.file_type,
        file_size=metadata.file_size,
        chunks_created=chunks_created,
        uploaded_at=metadata.uploaded_at,
    )