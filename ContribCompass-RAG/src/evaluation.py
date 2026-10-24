from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer: str
    required_terms: tuple[str, ...] = ()
    should_refuse: bool = False


EVALUATION_SET = [
    EvaluationCase("What is FastAPI used for?", "A Python framework for building APIs.", ("api", "python")),
    EvaluationCase("How do I install requirements for contributing?", "Install uv and run uv sync.", ("uv sync",)),
    EvaluationCase("How do I install the pre-commit hooks?", "Run uv run prek install.", ("uv run prek install",)),
    EvaluationCase("How do I run the test script?", "Run bash scripts/test.sh, or pytest manually.", ("scripts/test.sh",)),
    EvaluationCase("What should I do before sending a pull request?", "Start a GitHub Discussion and wait for a team member to request a PR.", ("discussion", "team member")),
    EvaluationCase("Can outside contributors modify pyproject.toml or uv.lock?", "Non-team PRs cannot modify these files; discuss new dependencies first.", ("not allowed", "pyproject.toml", "uv.lock")),
    EvaluationCase("How can I preview documentation locally?", "Run uv run ./scripts/docs.py live; the server uses port 8008.", ("scripts/docs.py live", "8008")),
    EvaluationCase("Where are the Markdown documentation files?", "Markdown docs are in ./docs; example Python sources are in ./docs_src/.", ("./docs", "./docs_src/")),
    EvaluationCase("Why do tests run against documentation examples?", "They keep examples runnable, documentation current, and features covered.", ("up-to-date", "coverage")),
    EvaluationCase("Where should I ask FastAPI questions?", "Use GitHub Discussions; chat is for general conversation.", ("github", "question")),
    EvaluationCase("Should I use the chat for technical questions?", "Prefer GitHub Discussions; chats are harder to search.", ("github", "chat")),
    EvaluationCase("How can I receive notifications of new releases?", "Watch the GitHub repository and select Releases only.", ("releases only",)),
    EvaluationCase("What command creates a new FastAPI project with uv?", "uv init awesome-project --bare, then cd and uv add fastapi[standard].", ("uv init", "uv add")),
    EvaluationCase("How do I run commands inside the uv environment?", "Use uv run, for example uv run fastapi dev.", ("uv run fastapi dev",)),
    EvaluationCase("Can this app book my flight?", "Decline: booking is outside the documents.", should_refuse=True),
    EvaluationCase("What is the recipe for chicken biryani?", "Decline: no recipe evidence.", should_refuse=True),
    EvaluationCase("What is my bank account balance?", "Decline: no personal banking data.", should_refuse=True),
    EvaluationCase("What is the exact guaranteed review deadline for my pull request?", "Decline: no guaranteed review deadline in the documents.", should_refuse=True),
]


def evaluate(rag, top_k: int, threshold: float, generate: bool = False, api_key: str | None = None) -> list[dict]:
    """Transparent rubric; generated answers still require human semantic review.

    Default mode measures retrieval coverage + refusal gating without API calls.
    Live mode applies the rubric only to answer text, never to unseen passages.
    """
    rows = []
    for case in EVALUATION_SET:
        response = rag.answer(case.question, top_k=top_k, threshold=threshold, generate=generate, api_key=api_key)
        text = (response["answer"] if generate else " ".join(r.chunk.text for r in response["results"])).lower()
        missing = [term for term in case.required_terms if term.lower() not in text]
        passed = response["refused"] if case.should_refuse else not response["refused"] and not missing
        rows.append({
            "Question": case.question,
            "Expected": "Refusal" if case.should_refuse else "Supported answer",
            "Expected answer": case.expected_answer,
            "Outcome": "Pass" if passed else "Review",
            "Reason": ("Expected refusal" if case.should_refuse else ("Evidence below relevance threshold" if response["refused"] else "Missing: " + ", ".join(missing))) if not passed else "Rubric satisfied",
            "Actual answer": response["answer"],
            "Mode": response["mode"], "Relevance": round(response["relevance"], 3),
            "Latency ms": response["latency_ms"],
            "Evaluation": "Answer rubric (human review required)" if generate else "Retrieval + gate only",
        })
    return rows
