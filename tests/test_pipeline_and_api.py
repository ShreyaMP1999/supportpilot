import pytest
from fastapi.testclient import TestClient

from supportpilot.generator import ClaudeGenerator, build_generator


def test_pipeline_end_to_end(pipeline):
    result = pipeline.process("How do I reset my password? The email never arrives.")
    assert result.intent == "recover_password"
    assert result.team == "Accounts & Security"
    assert result.articles[0]["id"] == "KB-015"
    assert "[KB-015]" in result.reply
    assert result.citations == ["KB-015"]
    assert result.reply_backend == "extractive"
    assert result.latency_ms["total"] > 0


def test_intent_prior_combines_multi_intent_articles(pipeline):
    prior = pipeline.article_prior({"cancel_order": 0.5, "check_cancellation_fee": 0.3, "track_order": 0.2})
    assert prior["KB-003"] == pytest.approx(0.8)


def test_generator_falls_back_without_credentials(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    assert build_generator(True, "claude-opus-5", "low").name == "extractive"


def test_claude_generator_redacts_and_validates_citations(monkeypatch, retriever):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured = {}

    class FakeBlock:
        type = "text"
        text = "Hi! You can reset it from the sign-in page [KB-015]. See also [KB-999].\nShopSphere Support"

    class FakeResponse:
        stop_reason = "end_turn"
        model = "claude-opus-5"
        content = [FakeBlock()]

    gen = ClaudeGenerator(model="claude-opus-5")

    def fake_create(**params):
        captured.update(params)
        return FakeResponse()

    monkeypatch.setattr(gen.client.beta.messages, "create", fake_create)
    hits = retriever.search("forgot password", top_k=2)
    draft = gen.generate("forgot password, my email is bob@example.com", hits, "Accounts & Security", False)

    prompt = captured["messages"][0]["content"]
    assert "bob@example.com" not in prompt and "[EMAIL]" in prompt
    assert captured["fallbacks"] == "default"
    assert draft.citations == ["KB-015"]
    assert not draft.grounded  # cited an article that was not retrieved
    assert any("KB-999" in n for n in draft.notes)


@pytest.fixture()
def client(model_path, tmp_path, monkeypatch):
    monkeypatch.setenv("SP_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SP_USE_DENSE", "false")
    monkeypatch.setenv("SP_USE_LLM", "false")
    from supportpilot.api import app

    with TestClient(app) as c:
        yield c


def test_api_health_and_ui(client):
    health = client.get("/health").json()
    assert health == {
        "status": "ok",
        "retrieval": "bm25",
        "generator": "extractive",
        "classifier": "tfidf",
        "intents": 27,
    }
    assert "SupportPilot" in client.get("/").text


def test_api_ticket_lifecycle(client):
    created = client.post("/v1/tickets", json={"text": "Cancel my order please, email me at a@b.com"})
    assert created.status_code == 200
    body = created.json()
    assert body["intent"] == "cancel_order" and body["team"] == "Fulfillment"

    stored = client.get(f"/v1/tickets/{body['id']}").json()
    assert "a@b.com" not in stored["text"]  # ticket log is PII-redacted

    assert client.get("/v1/tickets").json()[0]["id"] == body["id"]
    stats = client.get("/v1/stats").json()
    assert stats["total"] == 1 and stats["by_team"] == {"Fulfillment": 1}
    assert client.get("/v1/tickets/9999").status_code == 404


def test_api_classify_search_and_validation(client):
    assert client.post("/v1/classify", json={"text": "I want to track my order"}).json()["intent"] == "track_order"
    results = client.get("/v1/search", params={"q": "refund policy", "k": 2}).json()
    assert len(results) == 2 and results[0]["passages"]
    assert client.post("/v1/tickets", json={"text": ""}).status_code == 422
