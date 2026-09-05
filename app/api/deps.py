from fastapi import Header, HTTPException

from app.core.config import settings


async def verify_internal_key(x_internal_api_key: str = Header(default="")) -> None:
    """This service is never reachable from the frontend/ingress — only from
    jarvis-backend, over the cluster-internal network. A shared secret header
    is enough to prove the caller is jarvis-backend; the end user's identity
    (user_id) travels as a plain field in the request body/query instead,
    since jarvis-backend already verified it via the user's JWT before
    calling here. Same pattern as jarvis-conversation-service."""
    if not settings.INTERNAL_API_KEY or x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")
