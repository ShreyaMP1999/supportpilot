"""FastAPI service exposing the pipeline, plus a static dashboard at `/`."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import SupportPipeline
from .store import TicketStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).parent / "static"


class TicketIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, examples=["I was charged twice for order #A1234!!"])


class ClassifyOut(BaseModel):
    intent: str
    confidence: float
    alternatives: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pipeline = SupportPipeline.from_settings(settings)
    app.state.store = TicketStore(settings.db_path)
    yield
    app.state.store.close()


app = FastAPI(
    title="SupportPilot",
    description="AI triage and grounded reply drafting for customer-support tickets.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _pipeline(request: Request) -> SupportPipeline:
    return request.app.state.pipeline


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict:
    p = _pipeline(request)
    return {
        "status": "ok",
        "retrieval": p.retriever.mode,
        "generator": p.generator.name,
        "classifier": p.classifier.mode,
        "intents": len(p.classifier.labels),
    }


@app.post("/v1/tickets")
def create_ticket(ticket: TicketIn, request: Request) -> dict:
    result = _pipeline(request).process(ticket.text).to_dict()
    ticket_id = request.app.state.store.add(ticket.text, result)
    return {"id": ticket_id, **result}


@app.get("/v1/tickets")
def list_tickets(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    return request.app.state.store.recent(limit)


@app.get("/v1/tickets/{ticket_id}")
def get_ticket(ticket_id: int, request: Request) -> dict:
    ticket = request.app.state.store.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.post("/v1/classify", response_model=ClassifyOut)
def classify(ticket: TicketIn, request: Request) -> ClassifyOut:
    probs, _, _ = _pipeline(request).classify(ticket.text)
    (intent, confidence), *alternatives = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return ClassifyOut(
        intent=intent,
        confidence=round(float(confidence), 4),
        alternatives=[{"intent": i, "confidence": round(float(p), 4)} for i, p in alternatives],
    )


@app.get("/v1/search")
def search(request: Request, q: str = Query(..., min_length=2), k: int = Query(3, ge=1, le=10)) -> list[dict]:
    hits = _pipeline(request).retriever.search(q, top_k=k)
    return [
        {
            "id": h.article_id,
            "title": h.title,
            "score": round(h.score, 4),
            "passages": [{"heading": c.heading, "text": c.text} for c in h.chunks],
        }
        for h in hits
    ]


@app.get("/v1/stats")
def stats(request: Request) -> dict:
    return request.app.state.store.stats()
