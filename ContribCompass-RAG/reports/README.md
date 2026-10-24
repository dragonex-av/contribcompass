# Showcase evaluation — 2026-10-24

These are measured retrieval/gating results against 18 explicit cases, not generated-answer accuracy.

| Configuration | Passed | Rate |
| --- | ---: | ---: |
| MiniLM + TF-IDF, top-5, threshold 0.24 | 17/18 | 94.4% |
| TF-IDF only, top-5, threshold 0.24 | 15/18 | 83.3% |

Hybrid failure: a question about an exact guaranteed pull-request review deadline retrieves related PR guidance, so similarity gating alone does not refuse. There is no supported deadline in the documents. The live generator is instructed to refuse insufficient evidence, but this requires validation with an API key.

Keyword failures: paraphrased FastAPI purpose, pull-request workflow, and project creation questions fall below the threshold. Semantic embeddings recover these cases.

`experiments-hybrid.csv` and `experiments-keyword.csv` compare nine top-k/threshold combinations each. Detailed evaluation files contain expected answers, actual evidence output, reasons, timing, and retrieval mode. The seeded documents only were used; personal uploads were excluded.

No live Gemma quality or speed claim is made: the environment had no configured API key. The UI supports entering one for live generation and a real no-RAG baseline.

Run `python scripts/run_evaluation.py` to reproduce hybrid results or add `--keyword-only` for the lexical baseline. The current suite has 17 passing regression/smoke tests, including chat navigation, account persistence and owner isolation, source-grounded code/impact validation, and report rendering.
