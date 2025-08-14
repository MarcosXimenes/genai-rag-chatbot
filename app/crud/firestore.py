import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from fastapi import HTTPException
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

from utils.globals import GOOGLE_CLOUD_PROJECT

db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)


class FirestoreCRUD:
    @staticmethod
    async def index(
        user: str, session: str, filename: str, vectors_list: list, document_text: str
    ):
        try:
            user_ref = db.collection("users").document(user)
            session_ref = user_ref.collection("sessions").document(session)

            user_ref.set(
                {
                    "updated": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

            session_ref.set(
                {
                    "updated": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )

            document_data = {
                "filename": filename,
                "created_at": firestore.SERVER_TIMESTAMP,
                "text": document_text,
                "embedding": Vector(vectors_list),
                "active": True,
            }

            _, document_ref = session_ref.collection("documents").add(document_data)

            return {
                "document_id": document_ref.id,
                "message": "Document indexed successfully",
            }

        except Exception as e:
            logging.error(f"Error indexing document: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to index document: {str(e)}"
            )

    @staticmethod
    async def index_batch(
        user: str,
        session: str,
        filename: str,
        chunks_with_vectors: List[Dict[str, any]],
    ) -> List[str]:
        try:
            batch = db.batch()

            user_ref = db.collection("users").document(user)
            session_ref = user_ref.collection("sessions").document(session)
            documents_collection_ref = session_ref.collection("documents")

            batch.set(user_ref, {"updated": firestore.SERVER_TIMESTAMP}, merge=True)
            batch.set(session_ref, {"updated": firestore.SERVER_TIMESTAMP}, merge=True)

            document_ids = []

            for chunk_data in chunks_with_vectors:
                new_document_ref = documents_collection_ref.document()
                document_ids.append(new_document_ref.id)

                document_payload = {
                    "filename": filename,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "text": chunk_data["text"],
                    "embedding": Vector(chunk_data["vector"]),
                    "active": True,
                }

                batch.set(new_document_ref, document_payload)
            await asyncio.to_thread(batch.commit)
            logging.info(
                f"{len(document_ids)} chunks para o arquivo '{filename}' indexados em lote."
            )
            return document_ids

        except Exception as e:
            logging.error(f"Erro ao indexar documentos em lote: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Falha ao indexar documentos em lote: {str(e)}"
            )

    @staticmethod
    async def delete(user: str, session: str, filename: str):
        try:
            documents_ref = (
                db.collection("users")
                .document(user)
                .collection("sessions")
                .document(session)
                .collection("documents")
            )

            docs_query = documents_ref.where("filename", "==", filename)
            docs_to_delete = await asyncio.to_thread(list, docs_query.stream())

            if not docs_to_delete:
                raise HTTPException(
                    status_code=404,
                    detail=f"No documents found with the filename '{filename}'.",
                )

            batch = db.batch()
            for doc in docs_to_delete:
                batch.delete(doc.reference)

            await asyncio.to_thread(batch.commit)

            deleted_count = len(docs_to_delete)
            logging.info(
                f"{deleted_count} documents with the name '{filename}' have been deleted."
            )

            return {
                "message": f"Successfully deleted {deleted_count} document(s) with the filename '{filename}'."
            }

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logging.error(f"Error deleting documents by filename: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to delete documents: {str(e)}"
            )

    @staticmethod
    async def list(user: str):
        """
        Lista todos os chunks e os agrupa por sessão e, em seguida,
        por filename para retornar um sumário aninhado.
        """
        try:
            sessions_ref = db.collection("users").document(user).collection("sessions")
            all_sessions = await asyncio.to_thread(list, sessions_ref.stream())

            all_documents_flat = []
            for session in all_sessions:
                documents_ref = session.reference.collection("documents")
                docs_stream = await asyncio.to_thread(list, documents_ref.stream())
                for doc in docs_stream:
                    doc_data = doc.to_dict()
                    all_documents_flat.append(
                        {
                            "session_id": session.id,
                            "filename": doc_data.get("filename"),
                        }
                    )

            sessions_summary = {}
            for doc in all_documents_flat:
                session_id = doc.get("session_id")
                filename = doc.get("filename")

                if not session_id or not filename:
                    continue

                if session_id not in sessions_summary:
                    sessions_summary[session_id] = {}

                if filename not in sessions_summary[session_id]:
                    sessions_summary[session_id][filename] = {
                        "filename": filename,
                        "chunk_count": 0,
                    }

                sessions_summary[session_id][filename]["chunk_count"] += 1

            final_result = []
            for session_id, files_dict in sessions_summary.items():
                final_result.append(
                    {"session_id": session_id, "files": list(files_dict.values())}
                )

            return final_result

        except Exception as e:
            logging.error(f"Error listing documents: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to list documents: {str(e)}"
            )

    @staticmethod
    async def vectors(user: str, session: str):
        try:
            session_ref = (
                db.collection("users")
                .document(user)
                .collection("sessions")
                .document(session)
            )
            documents_ref = session_ref.collection("documents")
            docs = documents_ref.stream()

            all_vectors = []
            all_texts = []

            for doc in docs:
                embedding = doc.to_dict().get("embedding")
                document_text = doc.to_dict().get("text")
                if embedding:
                    all_vectors.append(list(embedding.to_map_value().get("value")))
                    all_texts.append(document_text)

            return all_vectors, all_texts

        except Exception as e:
            logging.error(f"Error retrieving vectors: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve vectors: {str(e)}"
            )
