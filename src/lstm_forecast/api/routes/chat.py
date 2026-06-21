"""RAG chat endpoint grounded in docs and (optionally) a fresh forecast run."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from lstm_forecast.ai.assistant import ChatAssistant
from lstm_forecast.ai.doc_index import DocIndex
from lstm_forecast.api import service
from lstm_forecast.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

# Build the documentation index once at import time (cheap, embedding-free).
_DOC_INDEX = DocIndex()
for candidate in ("README.md", "docs"):
    if Path(candidate).exists():
        _DOC_INDEX.add_paths([candidate])
_DOC_INDEX.build()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question, grounded in project docs and (optionally) a forecast run."""
    result = None
    if req.forecast is not None:
        try:
            _, result = service.run_forecast(req.forecast)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    assistant = ChatAssistant(_DOC_INDEX, result=result)
    answer = assistant.ask(req.question)
    return ChatResponse(answer=answer, ai_enabled=assistant.client.available)
