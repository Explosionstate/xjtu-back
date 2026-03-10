from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.chat_logs import router as chat_logs_router
from app.api.routes.documents import router as doc_router
from app.api.routes.knowledge_bases import router as kb_router
from app.api.routes.rbac import router as rbac_router
from app.api.routes.retrieval_config import router as retrieval_config_router
from app.api.routes.system_config import router as system_config_router


api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(kb_router)
api_router.include_router(doc_router)
api_router.include_router(chat_router)
api_router.include_router(chat_logs_router)
api_router.include_router(rbac_router)
api_router.include_router(system_config_router)
api_router.include_router(retrieval_config_router)
