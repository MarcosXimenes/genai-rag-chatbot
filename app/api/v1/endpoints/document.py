import asyncio
import logging
from io import BytesIO
from typing import Any, Dict, List

from fastapi import APIRouter, Body, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader

from crud.firestore import FirestoreCRUD
from crud.langchain import chunk_document_text
from crud.vertex_ai import VertexAICRUD

router = APIRouter()


async def generate_embeddings_in_batches(
    chunks: list[str], batch_size: int = 250
) -> list[list[float]]:
    if not chunks:
        return []

    tasks = [
        VertexAICRUD.generate_embedding(chunks[i : i + batch_size])
        for i in range(0, len(chunks), batch_size)
    ]

    all_batch_responses = await asyncio.gather(*tasks)

    final_vectors = [
        resp.values
        for batch_responses in all_batch_responses
        for resp in batch_responses
    ]
    return final_vectors


def _extract_and_chunk_text(pdf_content: bytes) -> List[str]:
    try:
        document_text = "".join(
            page.extract_text() or "" for page in PdfReader(BytesIO(pdf_content)).pages
        )

        if not document_text.strip():
            logging.warning("No text found in the PDF content.")
            return []

        return chunk_document_text(
            document_text=document_text,
            chunk_size=1000,
            chunk_overlap=150,
        )
    except Exception as e:
        logging.error(f"Failed to extract or chunk text: {e}", exc_info=True)
        return []


async def _process_and_index_file(
    user: str, session: str, file: UploadFile
) -> Dict[str, Any]:
    try:
        logging.info(f"Starting file processing: {file.filename}")

        pdf_content = await file.read()

        chunks = await asyncio.to_thread(_extract_and_chunk_text, pdf_content)

        if not chunks:
            logging.warning(f"Could not generate chunks for file: {file.filename}")
            return {
                "filename": file.filename,
                "status": "error",
                "detail": "No text found or failed to create chunks.",
            }

        chunk_vectors = await generate_embeddings_in_batches(chunks=chunks)
        if not chunk_vectors:
            return {
                "filename": file.filename,
                "status": "error",
                "detail": "Failed to generate embeddings for text chunks.",
            }

        chunks_to_index = [
            {"text": text, "vector": vector}
            for text, vector in zip(chunks, chunk_vectors)
        ]

        indexed_ids = await FirestoreCRUD.index_batch(
            user=user,
            session=session,
            filename=file.filename,
            chunks_with_vectors=chunks_to_index,
        )

        logging.info(
            f"File {file.filename} processed successfully. {len(indexed_ids)} chunks indexed."
        )
        return {
            "filename": file.filename,
            "status": "success",
            "indexed_chunks": len(indexed_ids),
            "document_ids": indexed_ids,
        }
    except Exception as e:
        logging.error(f"Failed to process file {file.filename}: {e}", exc_info=True)
        return {
            "filename": file.filename,
            "status": "error",
            "detail": str(e),
        }


@router.post("/index", status_code=status.HTTP_200_OK)
@router.post("/index/", include_in_schema=False)
async def index_documents(
    user: str,
    session: str,
    files: List[UploadFile] = File(
        ..., description="List of PDF documents to be indexed"
    ),
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded."
        )

    tasks = [_process_and_index_file(user, session, file) for file in files]

    results = await asyncio.gather(*tasks)

    successful_files = [r for r in results if r["status"] == "success"]
    failed_files = [r for r in results if r["status"] == "error"]

    if not successful_files and failed_files:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Failed to process all documents.",
                "results": failed_files,
            },
        )

    return JSONResponse(
        content={
            "message": "Document processing completed.",
            "results": {
                "successful": successful_files,
                "failed": failed_files,
            },
        }
    )


@router.delete("/delete", status_code=status.HTTP_200_OK)
@router.delete("/delete/", include_in_schema=False)
async def delete_document(
    user: str,
    session: str,
    filename: str = Body(
        ..., embed=True, description="Filename of the document to be removed"
    ),
):
    try:
        result = await FirestoreCRUD.delete(
            user=user, session=session, filename=filename
        )
        return JSONResponse(
            content={"message": result.get("message", "Document successfully deleted.")}
        )

    except Exception as e:
        logging.error(f"Error deleting document {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete the document: {str(e)}",
        )


@router.get("/list", status_code=status.HTTP_200_OK)
@router.get("/list/", include_in_schema=False)
async def list_documents(user: str):
    try:
        documents = await FirestoreCRUD.list(user=user)
        return JSONResponse(content={"documents": documents})
    except Exception as e:
        logging.error(f"Error listing documents for user {user}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )
