# ContribCompass — your first contribution, backed by evidence

A FastAPI contributor assistant for problem statement **2B**. Ask about local setup, pull requests, code formatting, tests, or community help. Every supported response names source files and sections; the evidence drawer shows complete retrieved passages.

## Personal chat workspace and Code + impact

The website now has a ChatGPT-style chat surface, searchable sidebar history, multiple conversations, rename/delete/export actions, and a separate document library. Guest chats last for the browser session. Create a **local account in Settings → Account** to save future conversations and documents between visits. Passwords use salted PBKDF2; every saved chat and upload directory is scoped to its owner. This is a local application, not a deployed multi-tenant service.

The API key field is in **Settings → Model connection**, with a Test connection action. Keys stay in session memory and are not written to chat storage. Existing server configuration works automatically.

For the USP, upload project docs in **Documents**, select **My documents** in Chat, then choose **Code + impact**. With a working model connection, the response includes:

- A cited explanation and code block, labelled as a source example or proposed adaptation.
- A before/after table showing affected components and expected behavior, with evidence references.
- Risks, verification steps, and assumptions or missing context.

Effects are predictions based on uploaded documentation. The app does not execute generated code, inspect an unuploaded repository, or claim measured performance gains. Unsupported requests are declined. Without a working key, it shows exact code examples found in retrieved docs and clearly marks impact analysis as pending.

Private runtime data is stored in ignored `.data/`; keys never go there. The bundled FastAPI corpus and your documents are separate knowledge workspaces.

## GitHub sign-in setup

1. [Register a GitHub OAuth app](https://github.com/settings/applications/new) owned by you.
2. For this local project, set **Homepage URL** and **Authorization callback URL** to `http://localhost:8501/`.
3. Copy the Client ID and create a Client Secret. Put them in the project's ignored `.env` file:

```dotenv
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
GITHUB_REDIRECT_URI=http://localhost:8501/
```

4. Restart Streamlit. Open **Settings → Account → Continue with GitHub**. Click **Open GitHub to authorize**, approve there, return to the original tab and click **Finish sign-in**.

The GitHub button is disabled until credentials are configured. For deployment, replace the callback with the exact HTTPS app URL in both GitHub and `.env`; never commit the Client Secret. Standard OAuth authorization is used; enabling device flow is not required.

The implementation follows [GitHub's OAuth web flow](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#web-application-flow), using a random state, S256 PKCE, and a ten-minute single-use flow. The original browser session holds a separate claim secret so a fresh Streamlit callback session cannot claim another user's workspace. The callback verifies identity through `/user`; the OAuth access token is discarded after that request. Client secrets and access tokens are not stored in the workspace database.

GitHub accounts use GitHub's numeric user ID. Renaming a GitHub account retains the same workspace. Local accounts with matching usernames are kept separate; existing guest/local data is not silently linked or transferred. The implementation requests an empty scope and uses public profile identity only, with no repository operations. Sign-out ends the app session; users can separately revoke the OAuth app in GitHub Settings.

## Start the demo

```bash
cd ~/Desktop/ContribCompass-RAG
source .venv/bin/activate
streamlit run app.py
```

For a fresh installation, run `bash start.sh`. Open http://localhost:8501. Enter a Google AI Studio API key in the **Settings → Model connection** panel (session only), or set `GOOGLE_API_KEY` in `.env`. The configured model is `gemma-4-26b-a4b-it`; access depends on your account. No working key means exact source excerpts, explicitly labelled as extractive mode, rather than generated answers. A failed API call also falls back visibly.

## 90-second showcase

1. Ask **What coding style rules should I follow?** Show the maintainer's `prek` guidance and source file.
2. Ask **How do I install requirements for contributing?** Show `uv sync` and open the original source.
3. Ask **How do I do that?** Open the evidence panel to show the contextual search query.
4. Ask **Can this app book my flight?** Show a refusal.
5. Open **Evaluation → Onboarding**, generate the five-step checklist, and tick a completed action.
6. Open **Evaluation → Quality lab**. Run 18 labelled questions, inspect expected answers and failures, then run the experiment matrix.
7. With a working key, open **Evaluation → RAG comparison** for actual model outputs. The app never fabricates a baseline.

## Evidence and provenance

The corpus contains FastAPI README, virtual environment, deployment, community help, and the maintainer's contribution guide linked from FastAPI. URLs are in `documents/sources.json`; refresh notes are in `documents/provenance.json`. Contributor and community sources were refreshed on 2026-10-24. Other source files are existing downloaded snapshots.

Earlier synthetic docs contained unverified commands and contact details. They are preserved in `archived_samples/` and excluded from retrieval. Legacy uploads remain untouched in `.uploads/`; add them through the new Documents library to associate them with your personal workspace. Uploaded files support Markdown, UTF-8 text, and text-based PDFs. PDF citations retain page numbers. Scanned PDFs need OCR before uploading.

## Pipeline and design choices

Documents → heading-aware chunks → MiniLM + TF-IDF → reciprocal-rank shortlist → cosine reranking → relevance gate → evidence-only Gemma response → citation validation → full evidence drawer.

- Chunk size: 900 characters, 150-character overlap. Heading-only sections are excluded.
- Default retrieval: top 5, relevance threshold 0.24. Similarity is not answer confidence.
- Terminology expansion maps contributor phrases such as pull request/PR and coding style/linting.
- Short referential follow-ups use the previous user question in retrieval. This is lightweight context resolution, not unrestricted long-term memory.
- Generated answers require valid source IDs. Missing or invalid citations trigger extractive fallback. Citation existence does not prove every claim is entailed.
- The model can refuse even when related passages pass the retrieval gate.
- No-model mode returns exact excerpts and cannot reliably detect every unsupported detail in a topically related question.
- Imported engine changes invalidate the Streamlit resource cache, fixing stale method signatures.

## Reproduce validation

```bash
pip install pytest
python -m pytest -q
python scripts/run_evaluation.py
python scripts/run_evaluation.py --keyword-only
```

The 18-question benchmark includes explicit expected answers and four refusal cases. Default runs measure **retrieval coverage plus refusal gating**, not generated-answer accuracy. Live evaluation checks generated answer text against a transparent term rubric and still requires human semantic review. CSV outputs in `reports/` include actual outputs, failure reasons, timing, and mode. The matrix compares top-k 3/5/7 with thresholds 0.15/0.24/0.30 without API calls.

A difficult negative asks for a guaranteed PR review deadline. Related PR passages can pass the retrieval gate despite missing the requested deadline. This is an intentionally retained failure case for explaining the limits of relevance thresholds; live model refusal is a second check, not a guarantee.

## Project files

- `app.py`: personal chat UI, document library, account/settings, onboarding, comparison and evaluation
- `src/workspace.py`: local accounts, password hashing and owner-scoped saved conversations
- `static/app.css`: responsive visual design
- `src/rag_engine.py`: ingestion, retrieval, contextual queries, generation and citations
- `src/evaluation.py`: expected-answer rubric and benchmark
- `scripts/run_evaluation.py`: reproducible offline benchmark and matrix
- `tests/`: pipeline regression tests and all-tab Streamlit smoke test
- `02_rag_starter_kit_solution.ipynb`: earlier starter notebook; the validated submission entrypoint is `app.py`

The optional two-model Gemma comparison is not implemented. No deployment or public publishing is required for the local showcase.
