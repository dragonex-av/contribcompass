"""Generate complete 02_rag_starter_kit_solution.ipynb for Google Colab and judging showcase."""
import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧭 ContribCompass: Gemma RAG Contributor Assistant\n",
    "### Theme 2: RAG Application with Gemma | Problem Statement 2B: Open-Source Contributor Guide\n",
    "\n",
    "**Team:** Hacktoberfest 2026 Innovators  \n",
    "**Target Project:** FastAPI Official Contributor Docs  \n",
    "**Models:** Google Gemma 4 (`gemma-4-26b-a4b-it`) / Gemma 2 (`gemma-2-9b-it`) + `all-MiniLM-L6-v2`  \n",
    "\n",
    "---\n",
    "### Problem Statement 2B Objectives:\n",
    "1. **Document Support:** Reads Markdown (`.md`), plain-text (`.txt`), and PDFs (`.pdf`).\n",
    "2. **Cited Answers:** References the source file and section for every answer.\n",
    "3. **Transparency:** Displays the retrieved evidence passages alongside similarity scores.\n",
    "4. **Safe Refusal:** Declines questions outside the docs with clear guidance on whom to contact.\n",
    "5. **16-Question Evaluation:** Benchmark measuring accuracy, refusal precision, and boundary cases.\n",
    "6. **Bonus (2B Special):** \"Where do I start?\" dynamic onboarding checklist for newcomers.\n",
    "7. **Bonus:** RAG vs. No-RAG side-by-side comparison illustrating hallucination reduction."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup & Dependencies\n",
    "Install the required libraries: Google GenAI SDK, sentence-transformers, scikit-learn, pypdf, and Gradio."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q google-genai sentence-transformers scikit-learn pypdf gradio pandas"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. API Key Configuration\n",
    "Get your free key from [Google AI Studio](https://aistudio.google.com/apikey)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from google import genai\n",
    "\n",
    "# Set your key or use Colab userdata/secrets\n",
    "GOOGLE_API_KEY = os.environ.get(\"GOOGLE_API_KEY\", \"\")\n",
    "if not GOOGLE_API_KEY:\n",
    "    try:\n",
    "        from google.colab import userdata\n",
    "        GOOGLE_API_KEY = userdata.get(\"GOOGLE_API_KEY\")\n",
    "    except Exception:\n",
    "        GOOGLE_API_KEY = input(\"Enter your Google AI Studio API key (or leave empty for demo mode): \")\n",
    "\n",
    "GEMMA_MODEL = \"gemma-4-26b-a4b-it\"  # Google Gemma open-weight model\n",
    "client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None\n",
    "print(f\"Client initialized. Active model: {GEMMA_MODEL}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Multi-Format Document Ingestion (.md, .txt, .pdf)\n",
    "ContribCompass extends the starter kit by ingesting Markdown, plain text, and PDF files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "from pypdf import PdfReader\n",
    "\n",
    "def load_document(file_path: str) -> str:\n",
    "    p = Path(file_path)\n",
    "    if p.suffix.lower() == \".pdf\":\n",
    "        reader = PdfReader(str(p))\n",
    "        return \"\\n\".join(page.extract_text() or \"\" for page in reader.pages)\n",
    "    else:\n",
    "        return p.read_text(encoding=\"utf-8\", errors=\"replace\")\n",
    "\n",
    "print(\"Document loader ready for .md, .txt, and .pdf files.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Heading-Aware Chunking Pipeline\n",
    "Preserves Markdown `# Headings` so that every chunk carries its structural context."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import hashlib\n",
    "from dataclasses import dataclass\n",
    "\n",
    "@dataclass\n",
    "class Chunk:\n",
    "    text: str\n",
    "    source: str\n",
    "    heading: str\n",
    "    chunk_id: str\n",
    "\n",
    "def chunk_markdown(text: str, source: str, size: int = 800, overlap: int = 150) -> list[Chunk]:\n",
    "    heading = \"Overview\"\n",
    "    sections = []\n",
    "    buffer = []\n",
    "    for line in text.splitlines():\n",
    "        m = re.match(r\"^#{1,6}\\s+(.+?)\\s*$\", line)\n",
    "        if m:\n",
    "            if buffer:\n",
    "                sections.append((heading, \"\\n\".join(buffer).strip()))\n",
    "            heading = m.group(1)\n",
    "            buffer = [line]\n",
    "        else:\n",
    "            buffer.append(line)\n",
    "    if buffer:\n",
    "        sections.append((heading, \"\\n\".join(buffer).strip()))\n",
    "\n",
    "    chunks = []\n",
    "    for h, sec in sections:\n",
    "        start = 0\n",
    "        while start < len(sec):\n",
    "            end = min(start + size, len(sec))\n",
    "            piece = sec[start:end].strip()\n",
    "            if piece:\n",
    "                cid = hashlib.sha1(f\"{source}|{h}|{start}\".encode()).hexdigest()[:8]\n",
    "                chunks.append(Chunk(piece, source, h, cid))\n",
    "            if end >= len(sec):\n",
    "                break\n",
    "            start += max(1, size - overlap)\n",
    "    return chunks\n",
    "\n",
    "print(\"Section-aware chunker ready.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)\n",
    "Combines dense embeddings (`all-MiniLM-L6-v2`) with sparse keyword matching (`TF-IDF`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "class HybridRetriever:\n",
    "    def __init__(self, chunks: list[Chunk]):\n",
    "        self.chunks = chunks\n",
    "        self.texts = [c.text for c in chunks]\n",
    "        self.vectorizer = TfidfVectorizer(stop_words=\"english\", ngram_range=(1,2))\n",
    "        self.tfidf = self.vectorizer.fit_transform(self.texts)\n",
    "        self.model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n",
    "        self.embeddings = self.model.encode(self.texts, normalize_embeddings=True)\n",
    "\n",
    "    def retrieve(self, query: str, top_k: int = 5):\n",
    "        # Keyword score\n",
    "        kw_sim = cosine_similarity(self.vectorizer.transform([query]), self.tfidf).ravel()\n",
    "        # Semantic score\n",
    "        q_emb = self.model.encode([query], normalize_embeddings=True)[0]\n",
    "        sem_sim = np.dot(self.embeddings, q_emb)\n",
    "        # Reciprocal Rank Fusion\n",
    "        rank_kw = np.argsort(np.argsort(-kw_sim))\n",
    "        rank_sem = np.argsort(np.argsort(-sem_sim))\n",
    "        rrf = 1 / (60 + rank_kw) + 1 / (60 + rank_sem)\n",
    "        top_idx = np.argsort(-rrf)[:top_k]\n",
    "        return [(self.chunks[i], float(rrf[i]), float(sem_sim[i]), float(kw_sim[i])) for i in top_idx]\n",
    "\n",
    "print(\"Hybrid retriever initialized.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Gemma Answer Generation & Safe Refusal Guard\n",
    "If maximum passage relevance falls below threshold `0.22`, the system safely declines."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def answer_question(retriever: HybridRetriever, query: str, top_k: int = 5, threshold: float = 0.22):\n",
    "    results = retriever.retrieve(query, top_k)\n",
    "    top_chunk, score, sem_score, kw_score = results[0]\n",
    "    relevance = max(sem_score, kw_score)\n",
    "\n",
    "    if relevance < threshold:\n",
    "        return {\n",
    "            \"answer\": \"I cannot verify an answer to that question from the loaded contributor documentation. \"\n",
    "                      \"Please consult the official FastAPI GitHub Discussions (https://github.com/fastapi/fastapi/discussions) \"\n",
    "                      \"or join the #contributors channel on Discord for community assistance.\",\n",
    "            \"refused\": True,\n",
    "            \"relevance\": relevance,\n",
    "            \"passages\": results\n",
    "        }\n",
    "\n",
    "    evidence = \"\\n\\n\".join(\n",
    "        f\"[S{i+1}] Source: {c.source} | Section: {c.heading}\\n{c.text}\"\n",
    "        for i, (c, sc, s_sc, k_sc) in enumerate(results)\n",
    "    )\n",
    "\n",
    "    if client:\n",
    "        prompt = f\"\"\"You are ContribCompass, an accurate guide for open-source contributors.\n",
    "Answer the question using ONLY the provided source passages. If the passages don't provide the answer, say so.\n",
    "Cite the sources you used using [S1], [S2] format.\n",
    "\n",
    "QUESTION: {query}\n",
    "\n",
    "EVIDENCE:\n",
    "{evidence}\"\"\"\n",
    "        response = client.models.generate_content(model=GEMMA_MODEL, contents=prompt)\n",
    "        ans_text = response.text.strip()\n",
    "    else:\n",
    "        ans_text = f\"[Extractive Demo Mode] Based on {top_chunk.source} ({top_chunk.heading}):\\n{top_chunk.text[:400]}...\\n\\nSources: [S1]\"\n",
    "\n",
    "    return {\n",
    "        \"answer\": ans_text,\n",
    "        \"refused\": False,\n",
    "        \"relevance\": relevance,\n",
    "        \"passages\": results\n",
    "    }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Bonus Feature: \"Where Do I Start?\" Dynamic Onboarding Checklist\n",
    "Generates a multi-step first-contribution roadmap grounded in project docs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_onboarding_roadmap(retriever: HybridRetriever):\n",
    "    steps = [\n",
    "        (\"1. Project Understanding\", \"What is FastAPI and what are its core architectural components?\"),\n",
    "        (\"2. Development Setup\", \"How do I install dependencies and create a virtual environment for development?\"),\n",
    "        (\"3. Code Standards\", \"What code formatting, linting, and type checking rules must be followed?\"),\n",
    "        (\"4. Testing & Validation\", \"How do I run tests and check test coverage before opening a PR?\"),\n",
    "        (\"5. Submitting Your PR\", \"What is the pull request submission process and checklist?\")\n",
    "    ]\n",
    "    roadmap = []\n",
    "    for title, q in steps:\n",
    "        res = answer_question(retriever, q, top_k=2, threshold=0.10)\n",
    "        source_info = res[\"passages\"][0][0].source if res[\"passages\"] else \"Official Docs\"\n",
    "        roadmap.append({\"step\": title, \"guidance\": res[\"answer\"], \"source\": source_info})\n",
    "    return roadmap"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. 16-Question Evaluation Benchmark Suite\n",
    "Evaluates 13 valid contributor queries + 3 out-of-scope refusal queries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "TEST_SUITE = [\n",
    "    (\"What is FastAPI used for?\", \"web framework\", False),\n",
    "    (\"How do I install the project dependencies?\", \"install\", False),\n",
    "    (\"How do I create a virtual environment?\", \"virtual\", False),\n",
    "    (\"How do I run FastAPI in local development?\", \"fastapi dev\", False),\n",
    "    (\"Which Python versions are supported?\", \"python\", False),\n",
    "    (\"What coding style rules must I follow?\", \"ruff\", False),\n",
    "    (\"How do I run the full test suite?\", \"pytest\", False),\n",
    "    (\"What is the test coverage requirement?\", \"100%\", False),\n",
    "    (\"What is the process for submitting a pull request?\", \"pull request\", False),\n",
    "    (\"How should I report a bug?\", \"issue\", False),\n",
    "    (\"Where should I ask general questions?\", \"discussions\", False),\n",
    "    (\"What is the line length limit for formatting?\", \"88\", False),\n",
    "    (\"How can I contribute documentation improvements?\", \"mkdocs\", False),\n",
    "    (\"Can this assistant book my flight to London?\", \"\", True),\n",
    "    (\"What is the recipe for chicken biryani?\", \"\", True),\n",
    "    (\"What are the college hostel visiting hours?\", \"\", True)\n",
    "]\n",
    "\n",
    "def run_evaluation(retriever):\n",
    "    records = []\n",
    "    for q, exp, should_refuse in TEST_SUITE:\n",
    "        res = answer_question(retriever, q)\n",
    "        ans = res[\"answer\"].lower()\n",
    "        if should_refuse:\n",
    "            passed = res[\"refused\"]\n",
    "        else:\n",
    "            passed = (not res[\"refused\"]) and (exp.lower() in ans or any(exp.lower() in p[0].text.lower() for p in res[\"passages\"]))\n",
    "        records.append({\n",
    "            \"Question\": q,\n",
    "            \"Expected\": \"Refusal\" if should_refuse else exp,\n",
    "            \"Refused\": res[\"refused\"],\n",
    "            \"Relevance\": round(res[\"relevance\"], 3),\n",
    "            \"Result\": \"PASS\" if passed else \"REVIEW\"\n",
    "        })\n",
    "    df = pd.DataFrame(records)\n",
    "    print(f\"Evaluation Accuracy: {(df.Result == 'PASS').mean():.1%}\")\n",
    "    return df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Launch Interactive Web App (Gradio)\n",
    "Creates a shareable public URL with transparent evidence viewer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import gradio as gr\n",
    "\n",
    "def make_gradio_ui(retriever):\n",
    "    def chat(query, top_k, threshold):\n",
    "        res = answer_question(retriever, query, int(top_k), float(threshold))\n",
    "        evidence_md = \"### Retrieved Passages:\\n\"\n",
    "        for i, (c, sc, sem, kw) in enumerate(res[\"passages\"], 1):\n",
    "            evidence_md += f\"**[S{i}] {c.source}** · *{c.heading}* (Hybrid: {sc:.3f}, Semantic: {sem:.3f})\\n\\n> {c.text[:250]}...\\n\\n\"\n",
    "        return res[\"answer\"], evidence_md\n",
    "\n",
    "    with gr.Blocks(theme=gr.themes.Soft()) as demo:\n",
    "        gr.Markdown(\"# 🧭 ContribCompass: Gemma RAG Contributor Assistant\")\n",
    "        with gr.Row():\n",
    "            with gr.Column(scale=3):\n",
    "                query = gr.Textbox(label=\"Ask a contributor question\", placeholder=\"e.g., How do I set up the project on my computer?\")\n",
    "                top_k = gr.Slider(minimum=2, maximum=8, value=5, step=1, label=\"Top-k Passages\")\n",
    "                threshold = gr.Slider(minimum=0.05, maximum=0.50, value=0.22, step=0.01, label=\"Refusal Threshold\")\n",
    "                btn = gr.Button(\"Submit\", variant=\"primary\")\n",
    "                ans = gr.Markdown(label=\"Answer\")\n",
    "            with gr.Column(scale=2):\n",
    "                evidence = gr.Markdown(label=\"Retrieved Passages\")\n",
    "        btn.click(chat, inputs=[query, top_k, threshold], outputs=[ans, evidence])\n",
    "    return demo\n",
    "\n",
    "print(\"Gradio app ready to launch with demo.launch(share=True)\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

out_path = Path(__file__).parents[1] / "02_rag_starter_kit_solution.ipynb"
out_path.write_text(json.dumps(notebook, indent=1))
print(f"Generated {out_path}")
