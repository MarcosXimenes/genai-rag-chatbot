from fastapi import APIRouter

from api.v1.endpoints import chat, document

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(document.router, prefix="/document", tags=["document"])
