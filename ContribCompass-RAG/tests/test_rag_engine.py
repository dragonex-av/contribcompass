from pathlib import Path

import pytest

from src.rag_engine import HybridRAG, chunk_document


@pytest.fixture
def rag():
    return HybridRAG(chunk_document("# Setup\n\nInstall dependencies with uv sync.\n\n# Tests\n\nRun pytest to test the project.", "GUIDE.md"), use_embeddings=False)


def test_chunking_preserves_source_and_heading():
    chunks = chunk_document("# Setup\n\nInstall dependencies with pip.", "CONTRIBUTING.md")
    assert chunks[0].source == "CONTRIBUTING.md"
    assert chunks[0].heading == "Setup"


def test_chunking_rejects_infinite_loop_parameters():
    with pytest.raises(ValueError):
        chunk_document("sample", "test.txt", size=20, overlap=20)


def test_empty_heading_is_not_retrieved():
    assert chunk_document("# Empty heading", "test.md") == []


def test_retrieval_finds_relevant_document_without_embeddings(rag):
    assert rag.retrieve("install dependencies", top_k=1)[0].chunk.heading == "Setup"


def test_off_topic_refusal(rag):
    result = rag.answer("banana astronaut banking", generate=False)
    assert result["refused"]
    assert not result["citations"]


def test_follow_up_uses_previous_user_context(rag):
    history = [{"role": "user", "content": "How do I install dependencies?"}]
    result = rag.answer("How do I do that?", history=history, generate=False, threshold=0)
    assert "install dependencies" in result["query"]


def test_valid_generated_citation_includes_filename(rag, monkeypatch):
    monkeypatch.setattr(rag, "_generate", lambda *args: "Run uv sync. [S1]")
    result = rag.answer("install dependencies", api_key="test", threshold=0)
    assert "Gemma" in result["mode"]
    assert "GUIDE.md" in result["answer"]
    assert result["citations"][0]["id"] == "S1"


def test_invalid_citation_falls_back_to_real_evidence(rag, monkeypatch):
    monkeypatch.setattr(rag, "_generate", lambda *args: "Invented command. [S999]")
    result = rag.answer("install dependencies", api_key="test", threshold=0)
    assert "fallback" in result["mode"]
    assert "Invented command" not in result["answer"]


def test_model_can_refuse_topically_similar_but_unsupported_question(rag, monkeypatch):
    monkeypatch.setattr(rag, "_generate", lambda *args: "INSUFFICIENT_EVIDENCE")
    assert rag.answer("install dependencies", api_key="test", threshold=0)["refused"]


def test_no_fake_baseline_without_api_key(rag, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    result = rag.answer_no_rag("book my flight")
    assert not result["available"]
    assert "No live baseline" in result["answer"]


def test_code_impact_is_structured_and_cited(rag, monkeypatch):
    import json
    report = {'supported':True, 'summary':'Install the documented dependencies [S1].',
              'code':'uv sync', 'language':'bash', 'code_kind':'exact source example',
              'effects':[{'area':'Environment', 'before':'Unknown', 'after':'Dependencies installed', 'evidence':'[S1]'}],
              'risks':['Dependency versions may change.'], 'checks':['Run the documented tests.'], 'assumptions':['Repository not inspected.']}
    monkeypatch.setattr(rag, '_generate', lambda *args: json.dumps(report))
    result = rag.code_impact('install dependencies', threshold=0, api_key='test')
    assert result['impact']['code'] == 'uv sync'
    assert result['citations'][0]['source'] == 'GUIDE.md'
    assert not result['refused']


def test_uncited_impact_falls_back(rag, monkeypatch):
    monkeypatch.setattr(rag, '_generate', lambda *args: '{"supported":true,"summary":"No evidence"}')
    result = rag.code_impact('install dependencies', threshold=0, api_key='test')
    assert result['impact'] is None
    assert 'fallback' in result['mode']


def test_code_impact_without_key_does_not_invent_code(rag, monkeypatch):
    monkeypatch.setenv('GOOGLE_API_KEY', '')
    result = rag.code_impact('install dependencies', threshold=0)
    assert result['impact'] is None
    assert 'No complete code example' in result['answer']
    assert 'Impact analysis pending' in result['answer']
