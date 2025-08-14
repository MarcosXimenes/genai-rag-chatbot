import logging

from fastapi import HTTPException
from google import genai
from google.genai.types import EmbedContentConfig

from utils.globals import (
    GEMINI_EMBEDDING_DIMENSIONALITY,
    GEMINI_EMBEDDING_MODEL,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_GENAI_USE_VERTEXAI,
)

client = genai.Client(
    vertexai=GOOGLE_GENAI_USE_VERTEXAI,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)


class VertexAICRUD:
    @staticmethod
    async def generate_embedding(
        texts: list[str],
        model: str = GEMINI_EMBEDDING_MODEL,
        output_dimensionality: int = GEMINI_EMBEDDING_DIMENSIONALITY,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        try:
            embed_config = EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            )

            response = client.models.embed_content(
                model=model,
                contents=texts,
                config=embed_config,
            )

            logging.info(f"Generated Embeddings: {response.embeddings}")
            return response.embeddings

        except Exception as e:
            logging.error(f"Error generating embeddings: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to generate embeddings: {str(e)}"
            )
