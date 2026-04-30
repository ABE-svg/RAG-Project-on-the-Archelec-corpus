import os
import re
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

EXAMPLE_QUESTIONS = [
    "Que disent les manifestes sur l'Europe ?",
    "Comment les candidats abordent-ils le chômage ?",
    "Quelle place est faite à l'éducation dans les programmes ?",
    "Quels sont les thèmes principaux des professions de foi ?",
    "Comment la question de l'immigration est-elle traitée ?",
]

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

state: dict = {}


FAISS_INDEX_PATH = Path("faiss_index")


def build_faiss_index(embedding_model):
    print("Building the index for the first time (this will take a few minutes)...")

    base_path = Path("text_files")
    documents, file_names = [], []

    for path in sorted(base_path.rglob("*.txt")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text and "_PF_" in path.name:
            documents.append(text)
            file_names.append(path.name)

    docs = [
        Document(page_content=text, metadata={"source": name})
        for text, name in zip(documents, file_names)
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    faiss_db = FAISS.from_documents(chunks, embedding_model)
    faiss_db.save_local(str(FAISS_INDEX_PATH))
    print(f"Index built and saved: {len(chunks)} chunks from {len(documents)} documents.")
    return faiss_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if FAISS_INDEX_PATH.exists():
        print("Loading existing FAISS index from disk...")
        faiss_db = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("Index loaded.")
    else:
        faiss_db = build_faiss_index(embedding_model)

    state["faiss_db"] = faiss_db
    print("Ready.")

    yield
    state.clear()


app = FastAPI(title="Plateforme RAG", lifespan=lifespan)


# ---------------------------------------------------------------------------
# RAG logic
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = 5) -> list[dict]:
    results = state["faiss_db"].similarity_search(query, k=k)
    return [
        {"chunk": doc.page_content, "source": doc.metadata.get("source", "unknown")}
        for doc in results
    ]


def get_chat_model():
    """Retourne le modele existant ou en cree un nouveau si le token a change."""
    if "chat_model" not in state:
        token = state.get("hf_token") or HF_TOKEN
        if not token:
            raise ValueError("Aucun token HuggingFace disponible.")
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=token,
        )
        state["chat_model"] = ChatHuggingFace(llm=llm)
    return state["chat_model"]


def generate(query: str, context: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "Tu es un journaliste politique spécialisé dans les élections législatives françaises de 1988. "
                "Tu réponds de manière factuelle et rigoureuse uniquement à partir du contexte fourni. "
                "Tu n'inventes jamais d'information. "
                "Si la réponse ne se trouve pas dans le contexte, dis-le clairement."
            )
        ),
        HumanMessage(
            content=f"Question: {query}\n\nContexte:\n{context}\n\nRéponds de façon courte et factuelle."
        ),
    ]
    response = get_chat_model().invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class TokenRequest(BaseModel):
    token: str


@app.post("/set-token")
async def set_token(request: TokenRequest):
    token = request.token.strip()
    if not token.startswith("hf_"):
        return {"ok": False, "error": "Le token doit commencer par hf_"}
    # On stocke juste le token — le modele sera (re)cree au prochain appel
    state["hf_token"] = token
    state.pop("chat_model", None)
    return {"ok": True}


@app.post("/ask")
async def ask(request: QueryRequest):
    if not state.get("hf_token") and not HF_TOKEN:
        return {"answer": "Aucun token configure. Rends-toi dans les Parametres pour entrer ton token HuggingFace.", "sources": []}
    retrieved = retrieve(request.query)
    context = "\n\n".join(
        [f"Source: {r['source']}\n{r['chunk']}" for r in retrieved]
    )
    answer = generate(request.query, context)
    return {"answer": answer, "sources": retrieved}


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Archelec 1988</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #111827;
      --bg-deep:   #0d1321;
      --surface:   #1a2438;
      --card:      #1e2d45;
      --card-h:    #243252;
      --border:    rgba(148,163,184,0.1);
      --border-h:  rgba(148,163,184,0.2);
      --blue:      #3b82f6;
      --indigo:    #6366f1;
      --violet:    #8b5cf6;
      --grad:      linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%);
      --grad-soft: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15));
      --amber:     #f59e0b;
      --amber-d:   rgba(245,158,11,0.1);
      --teal:      #14b8a6;
      --text:      #e2e8f0;
      --text-soft: #94a3b8;
      --muted:     #64748b;
      --white:     #f8fafc;
    }

    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }

    /* Ambient background orbs */
    body::before {
      content: '';
      position: fixed;
      top: -200px; left: -150px;
      width: 600px; height: 600px;
      background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
      pointer-events: none; z-index: 0;
    }
    body::after {
      content: '';
      position: fixed;
      bottom: -150px; right: -100px;
      width: 500px; height: 500px;
      background: radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 70%);
      pointer-events: none; z-index: 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.25); }

    /* ── Header ─────────────────────────────── */
    header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 28px;
      height: 58px;
      background: rgba(17,24,39,0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      position: relative;
      z-index: 10;
    }

    .logo-wrap {
      display: flex; align-items: center; gap: 10px;
    }

    .logo-mark {
      width: 32px; height: 32px; border-radius: 8px;
      background: var(--grad);
      display: flex; align-items: center; justify-content: center;
      font-family: 'Playfair Display', serif;
      font-size: 16px; font-weight: 600; color: #fff;
      box-shadow: 0 0 16px rgba(99,102,241,0.35), 0 2px 8px rgba(0,0,0,0.3);
      flex-shrink: 0;
    }

    .logo-text {
      display: flex; flex-direction: column; line-height: 1;
    }
    .logo-name {
      font-size: 14px; font-weight: 600; color: var(--white);
      letter-spacing: -0.01em;
    }
    .logo-year {
      font-size: 10.5px; color: var(--muted); letter-spacing: 0.05em;
      margin-top: 1px;
    }

    .header-divider {
      width: 1px; height: 24px;
      background: var(--border);
      margin: 0 4px;
    }

    .header-tag {
      font-size: 11.5px; color: var(--text-soft);
      background: rgba(148,163,184,0.06);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 3px 10px;
      letter-spacing: 0.01em;
    }

    .header-right { margin-left: auto; display: flex; align-items: center; gap: 8px; }

    .stat-pill {
      display: flex; align-items: center; gap: 5px;
      font-size: 11px; font-weight: 500;
      color: var(--amber);
      background: var(--amber-d);
      border: 1px solid rgba(245,158,11,0.2);
      padding: 4px 10px; border-radius: 20px;
    }
    .stat-pill::before {
      content: '';
      width: 5px; height: 5px; border-radius: 50%;
      background: var(--amber);
      animation: blink 2.5s infinite;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.3; }
    }

    /* ── Layout ──────────────────────────────── */
    .layout {
      display: flex;
      flex: 1;
      overflow: hidden;
      position: relative;
      z-index: 1;
    }

    /* ── Sidebar ─────────────────────────────── */
    aside {
      width: 260px;
      min-width: 260px;
      background: var(--bg-deep);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-top {
      padding: 20px 16px 16px;
      border-bottom: 1px solid var(--border);
    }

    .sidebar-label {
      font-size: 9.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .12em;
      color: var(--muted);
      margin-bottom: 12px;
      display: flex; align-items: center; gap: 6px;
    }
    .sidebar-label::after {
      content: '';
      flex: 1; height: 1px;
      background: var(--border);
    }

    .example-btn {
      display: block;
      width: 100%;
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text-soft);
      font-size: 12.5px;
      font-family: inherit;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      margin-bottom: 6px;
      transition: all .2s;
      line-height: 1.5;
      position: relative;
      overflow: hidden;
    }
    .example-btn::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 2px;
      background: var(--grad);
      opacity: 0;
      transition: opacity .2s;
    }
    .example-btn:hover {
      border-color: rgba(99,102,241,0.3);
      background: rgba(99,102,241,0.06);
      color: var(--text);
      transform: translateX(2px);
    }
    .example-btn:hover::before { opacity: 1; }

    .sidebar-history {
      flex: 1;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      padding: 16px 16px 12px;
    }

    .history-wrap {
      flex: 1;
      overflow-y: auto;
      margin-top: 8px;
    }

    .history-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 7px;
      font-size: 12px;
      color: var(--muted);
      cursor: pointer;
      margin-bottom: 2px;
      transition: all .15s;
    }
    .history-item svg { flex-shrink: 0; opacity: 0.5; }
    .history-item span {
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .history-item:hover { background: rgba(148,163,184,0.05); color: var(--text-soft); }
    .history-item.active {
      color: #818cf8;
      background: rgba(99,102,241,0.08);
    }
    .history-item.active svg { opacity: 1; color: #6366f1; }

    .history-empty {
      font-size: 12px; color: var(--muted);
      text-align: center; padding: 24px 8px;
      line-height: 1.6;
    }

    /* ── Sidebar tabs ────────────────────────── */
    .sidebar-tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .sidebar-tab {
      flex: 1;
      padding: 10px 0;
      font-size: 11.5px; font-weight: 500;
      color: var(--muted);
      background: none; border: none;
      font-family: inherit;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all .18s;
      letter-spacing: 0.01em;
    }
    .sidebar-tab:hover { color: var(--text-soft); }
    .sidebar-tab.active {
      color: #818cf8;
      border-bottom-color: #6366f1;
    }

    .tab-panel { display: none; flex-direction: column; flex: 1; overflow: hidden; }
    .tab-panel.active { display: flex; }

    /* ── Settings panel ──────────────────────── */
    .settings-wrap {
      padding: 18px 16px;
      display: flex; flex-direction: column; gap: 18px;
      overflow-y: auto; flex: 1;
    }

    .settings-group { display: flex; flex-direction: column; gap: 6px; }

    .settings-label {
      font-size: 10px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .1em;
      color: var(--muted);
    }

    .settings-desc {
      font-size: 11.5px; color: var(--text-soft); line-height: 1.55;
    }

    .token-input-wrap {
      display: flex;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 9px;
      overflow: hidden;
      transition: border-color .2s;
    }
    .token-input-wrap:focus-within { border-color: rgba(99,102,241,0.5); }

    #token-input {
      flex: 1;
      background: none; border: none; outline: none;
      color: var(--text); font-size: 12.5px; font-family: inherit;
      padding: 9px 12px;
      min-width: 0;
    }
    #token-input::placeholder { color: var(--muted); }

    #token-eye {
      background: none; border: none; cursor: pointer;
      color: var(--muted); padding: 0 10px;
      font-size: 14px; transition: color .15s;
    }
    #token-eye:hover { color: var(--text-soft); }

    #token-save {
      width: 100%;
      padding: 9px;
      background: var(--grad);
      border: none; border-radius: 9px;
      color: #fff; font-size: 13px; font-weight: 500;
      font-family: inherit; cursor: pointer;
      transition: opacity .15s, transform .12s;
      box-shadow: 0 2px 10px rgba(99,102,241,0.3);
    }
    #token-save:hover { opacity: .9; transform: translateY(-1px); }
    #token-save:active { transform: translateY(0); }

    .token-status {
      font-size: 11.5px; padding: 7px 10px;
      border-radius: 7px; text-align: center;
      display: none;
    }
    .token-status.ok { background: rgba(20,184,166,0.1); color: var(--teal); border: 1px solid rgba(20,184,166,0.2); }
    .token-status.err { background: rgba(239,68,68,0.08); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

    .token-link {
      font-size: 11px; color: var(--indigo);
      text-decoration: none; text-align: center;
      display: block; margin-top: 2px;
    }
    .token-link:hover { text-decoration: underline; }

    /* ── Main ────────────────────────────────── */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: var(--bg);
    }

    /* ── Chat ────────────────────────────────── */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 32px 52px 16px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Empty state */
    .empty-state {
      margin: auto;
      text-align: center;
      max-width: 460px;
      padding: 20px;
    }
    .empty-icon-wrap {
      position: relative;
      width: 72px; height: 72px;
      margin: 0 auto 24px;
    }
    .empty-icon-bg {
      width: 72px; height: 72px; border-radius: 20px;
      background: var(--grad-soft);
      border: 1px solid rgba(99,102,241,0.2);
      display: flex; align-items: center; justify-content: center;
      font-size: 30px;
    }
    .empty-icon-ring {
      position: absolute;
      inset: -6px;
      border-radius: 26px;
      border: 1px solid rgba(99,102,241,0.1);
      animation: ringPulse 3s infinite;
    }
    @keyframes ringPulse {
      0%, 100% { opacity: 0.5; transform: scale(1); }
      50% { opacity: 0; transform: scale(1.08); }
    }
    .empty-state h2 {
      font-family: 'Playfair Display', serif;
      font-size: 22px; font-weight: 600;
      color: var(--white);
      margin-bottom: 10px;
      line-height: 1.3;
    }
    .empty-state p {
      font-size: 14px; color: var(--text-soft); line-height: 1.7;
      margin-bottom: 20px;
    }
    .empty-chips {
      display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;
    }
    .empty-chip {
      font-size: 11.5px;
      background: rgba(148,163,184,0.06);
      border: 1px solid var(--border);
      border-radius: 20px;
      color: var(--text-soft);
      padding: 4px 12px;
    }

    /* Messages */
    .message {
      max-width: 700px;
      width: 100%;
      animation: fadeUp .28s cubic-bezier(.22,1,.36,1);
    }
    .message.user { align-self: flex-end; }
    .message.assistant { align-self: flex-start; }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* Row with avatar */
    .msg-row {
      display: flex;
      align-items: flex-end;
      gap: 10px;
    }
    .message.user .msg-row { flex-direction: row-reverse; }

    .avatar {
      width: 28px; height: 28px; border-radius: 8px;
      flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 600;
    }
    .avatar.user-av {
      background: var(--grad);
      color: #fff;
      box-shadow: 0 2px 8px rgba(99,102,241,0.4);
    }
    .avatar.bot-av {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-soft);
      font-size: 14px;
    }

    /* User bubble */
    .message.user .bubble {
      background: var(--grad);
      color: #fff;
      border-radius: 16px 16px 4px 16px;
      padding: 12px 16px;
      font-size: 14px;
      line-height: 1.65;
      box-shadow: 0 4px 16px rgba(59,130,246,0.25);
    }

    /* Assistant bubble */
    .message.assistant .bubble {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 4px 16px 16px 16px;
      padding: 14px 18px;
      font-size: 14px;
      line-height: 1.75;
      color: var(--text);
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }

    /* Sources */
    .sources { margin-top: 10px; margin-left: 38px; }
    .sources-toggle {
      display: inline-flex; align-items: center; gap: 6px;
      background: rgba(148,163,184,0.05);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--muted); font-size: 11.5px; font-family: inherit;
      cursor: pointer; padding: 4px 10px;
      transition: all .15s;
    }
    .sources-toggle:hover {
      background: rgba(148,163,184,0.09);
      color: var(--text-soft);
      border-color: var(--border-h);
    }
    .chevron { transition: transform .25s; display: inline-block; font-size: 9px; }
    .sources-toggle.open .chevron { transform: rotate(90deg); }

    .sources-list {
      display: none;
      flex-direction: column;
      gap: 6px;
      margin-top: 8px;
    }
    .sources-list.open { display: flex; }

    .source-card {
      background: var(--bg-deep);
      border: 1px solid var(--border);
      border-left: 2px solid var(--amber);
      border-radius: 8px;
      padding: 10px 14px;
      transition: all .15s;
    }
    .source-card:hover {
      border-color: var(--border-h);
      border-left-color: var(--amber);
      background: var(--surface);
    }
    .source-name {
      font-size: 10px; font-weight: 600;
      color: var(--amber); margin-bottom: 5px;
      font-family: 'SF Mono', 'Fira Code', monospace;
      letter-spacing: .05em;
    }
    .source-text {
      font-size: 12px; color: var(--muted); line-height: 1.55;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
      overflow: hidden;
    }

    /* Loader */
    .loader-wrap {
      display: flex; align-items: center; gap: 12px;
      padding: 14px 16px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 4px 16px 16px 16px;
      width: fit-content;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }
    .loader-text { font-size: 12.5px; color: var(--muted); font-style: italic; }
    .dots { display: flex; gap: 4px; }
    .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--grad);
      animation: dotBounce 1.4s infinite ease-in-out;
    }
    .dot:nth-child(1) { animation-delay: 0s; }
    .dot:nth-child(2) { animation-delay: .2s; }
    .dot:nth-child(3) { animation-delay: .4s; }
    @keyframes dotBounce {
      0%, 80%, 100% { transform: scale(0.5); opacity: .3; }
      40%            { transform: scale(1);   opacity: 1; }
    }

    /* ── Input bar ───────────────────────────── */
    .input-bar {
      padding: 12px 52px 18px;
      background: linear-gradient(to top, var(--bg) 60%, transparent);
      flex-shrink: 0;
    }

    .input-outer {
      position: relative;
      border-radius: 14px;
      padding: 1.5px;
      background: var(--border);
      transition: background .3s;
    }
    .input-outer:focus-within {
      background: var(--grad);
    }

    .input-inner {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      background: var(--card);
      border-radius: 13px;
      padding: 10px 10px 10px 16px;
    }

    #query-input {
      flex: 1;
      background: none; border: none; outline: none;
      color: var(--text); font-size: 14px; font-family: inherit;
      resize: none; line-height: 1.55;
      max-height: 120px;
    }
    #query-input::placeholder { color: var(--muted); }

    #send-btn {
      width: 34px; height: 34px; min-width: 34px; border-radius: 9px;
      background: var(--grad);
      border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: opacity .15s, transform .12s, box-shadow .15s;
      box-shadow: 0 2px 10px rgba(99,102,241,0.35);
      align-self: flex-end;
    }
    #send-btn:hover { opacity: .9; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(99,102,241,0.5); }
    #send-btn:active { transform: translateY(0); }
    #send-btn:disabled { opacity: .25; cursor: not-allowed; transform: none; box-shadow: none; }

    .hint {
      text-align: center;
      font-size: 10.5px;
      color: var(--muted);
      margin-top: 7px;
      letter-spacing: 0.01em;
    }
    .hint span { opacity: 0.5; margin: 0 4px; }
  </style>
</head>
<body>

<header>
  <div class="logo-wrap">
    <div class="logo-mark">A</div>
    <div class="logo-text">
      <span class="logo-name">Archelec</span>
      <span class="logo-year">Legislatives 1988</span>
    </div>
  </div>
  <div class="header-divider"></div>
  <span class="header-tag">Professions de foi &middot; CEVIPOF</span>
  <div class="header-right">
    <div class="stat-pill">3&thinsp;544 documents indexes</div>
  </div>
</header>

<div class="layout">
  <aside>
    <div class="sidebar-tabs">
      <button class="sidebar-tab active" onclick="switchTab('chat', this)">Recherche</button>
      <button class="sidebar-tab" onclick="switchTab('settings', this)">Parametres</button>
    </div>

    <!-- Tab: Recherche -->
    <div class="tab-panel active" id="tab-chat">
      <div class="sidebar-top">
        <div class="sidebar-label">Suggestions</div>
        EXAMPLES_PLACEHOLDER
      </div>
      <div class="sidebar-history">
        <div class="sidebar-label">Historique</div>
        <div class="history-wrap" id="history-list">
          <div class="history-empty" id="history-empty">
            Vos questions<br>apparaitront ici
          </div>
        </div>
      </div>
    </div>

    <!-- Tab: Parametres -->
    <div class="tab-panel" id="tab-settings">
      <div class="settings-wrap">
        <div class="settings-group">
          <div class="settings-label">Token HuggingFace</div>
          <p class="settings-desc">
            Entrez votre token personnel pour utiliser le modele Llama. Il reste dans votre navigateur et n'est jamais partage.
          </p>
          <div class="token-input-wrap">
            <input id="token-input" type="password" placeholder="hf_xxxxxxxxxxxxxxxx" autocomplete="off"/>
            <button id="token-eye" onclick="toggleTokenVis()" title="Afficher/masquer">&#128065;</button>
          </div>
          <button id="token-save" onclick="saveToken()">Appliquer le token</button>
          <div class="token-status" id="token-status"></div>
          <a class="token-link" href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">
            Obtenir un token sur HuggingFace &rarr;
          </a>
        </div>
      </div>
    </div>
  </aside>

  <main>
    <div id="chat">
      <div class="empty-state">
        <div class="empty-icon-wrap">
          <div class="empty-icon-bg">&#128441;</div>
          <div class="empty-icon-ring"></div>
        </div>
        <h2>Explorez les archives de 1988</h2>
        <p>Interrogez les professions de foi des candidats aux elections legislatives du 5 et 12 juin 1988, numerisees et indexees par le CEVIPOF.</p>
        <div class="empty-chips">
          <span class="empty-chip">Europe</span>
          <span class="empty-chip">Chomage</span>
          <span class="empty-chip">Education</span>
          <span class="empty-chip">Immigration</span>
          <span class="empty-chip">Securite</span>
        </div>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-outer">
        <div class="input-inner">
          <textarea id="query-input" rows="1"
            placeholder="Que disent les candidats sur le chomage en 1988 ?"></textarea>
          <button id="send-btn" title="Envoyer">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14"
                 viewBox="0 0 24 24" fill="none" stroke="#fff"
                 stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
      <p class="hint">Entree pour envoyer <span>&bull;</span> Shift+Entree pour un saut de ligne</p>
    </div>
  </main>
</div>

<script>
  const chat      = document.getElementById('chat');
  const input     = document.getElementById('query-input');
  const sendBtn   = document.getElementById('send-btn');
  const histList  = document.getElementById('history-list');
  const histEmpty = document.getElementById('history-empty');
  let history = [];

  // ── Onglets sidebar ──
  function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-tab').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ── Token HuggingFace ──
  function toggleTokenVis() {
    const inp = document.getElementById('token-input');
    inp.type = inp.type === 'password' ? 'text' : 'password';
  }

  function showTokenStatus(msg, type) {
    const el = document.getElementById('token-status');
    el.textContent = msg;
    el.className = 'token-status ' + type;
    el.style.display = 'block';
    if (type === 'ok') setTimeout(() => { el.style.display = 'none'; }, 4000);
  }

  function saveToken() {
    const token = document.getElementById('token-input').value.trim();
    if (!token) { showTokenStatus('Veuillez entrer un token.', 'err'); return; }
    const btn = document.getElementById('token-save');
    btn.textContent = 'Application...';
    btn.disabled = true;
    fetch('/set-token', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token})
    })
    .then(r => r.json())
    .then(data => {
      btn.textContent = 'Appliquer le token';
      btn.disabled = false;
      if (data.ok) {
        localStorage.setItem('hf_token', token);
        showTokenStatus('Token applique avec succes.', 'ok');
      } else {
        showTokenStatus(data.error || 'Erreur inconnue.', 'err');
      }
    })
    .catch(() => {
      btn.textContent = 'Appliquer le token';
      btn.disabled = false;
      showTokenStatus('Impossible de joindre le serveur.', 'err');
    });
  }

  // Restaurer le token depuis localStorage au chargement
  (function restoreToken() {
    const saved = localStorage.getItem('hf_token');
    if (!saved) return;
    document.getElementById('token-input').value = saved;
    fetch('/set-token', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token: saved})
    }).catch(() => {});
  })();

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  });

  sendBtn.addEventListener('click', sendQuery);

  document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      input.value = btn.dataset.query;
      input.dispatchEvent(new Event('input'));
      sendQuery();
    });
  });

  function sendQuery() {
    const query = input.value.trim();
    if (!query) return;
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    const empty = chat.querySelector('.empty-state');
    if (empty) empty.remove();

    appendUser(query);

    const loader = document.createElement('div');
    loader.className = 'message assistant';
    loader.innerHTML = '<div class="msg-row"><div class="avatar bot-av">&#128270;</div><div class="loader-wrap"><div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div><span class="loader-text">Recherche dans les archives...</span></div></div>';
    chat.appendChild(loader);
    scrollBottom();

    fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    })
    .then(r => r.json())
    .then(data => {
      loader.remove();
      appendAnswer(data.answer, data.sources);
      addHistory(query);
      sendBtn.disabled = false;
    })
    .catch(() => {
      loader.remove();
      appendAnswer('Une erreur est survenue. Verifie que le serveur est bien lance.', []);
      sendBtn.disabled = false;
    });
  }

  function appendUser(text) {
    const el = document.createElement('div');
    el.className = 'message user';
    el.innerHTML = '<div class="msg-row"><div class="avatar user-av">V</div><div class="bubble">' + esc(text) + '</div></div>';
    chat.appendChild(el);
    scrollBottom();
  }

  function appendAnswer(answer, sources) {
    const el = document.createElement('div');
    el.className = 'message assistant';

    const cards = (sources || []).map(s =>
      '<div class="source-card">' +
      '<div class="source-name">' + esc(s.source) + '</div>' +
      '<div class="source-text">' + esc(s.chunk) + '</div>' +
      '</div>'
    ).join('');

    const srcBlock = sources && sources.length ? `
      <div class="sources">
        <button class="sources-toggle" onclick="toggleSrc(this)">
          <span class="chevron">&#9658;</span>
          ${sources.length} passage${sources.length > 1 ? 's' : ''} consulte${sources.length > 1 ? 's' : ''}
        </button>
        <div class="sources-list">${cards}</div>
      </div>` : '';

    el.innerHTML = '<div class="msg-row"><div class="avatar bot-av">&#128270;</div><div class="bubble">' + esc(answer) + '</div></div>' + srcBlock;
    chat.appendChild(el);
    scrollBottom();
  }

  function toggleSrc(btn) {
    btn.classList.toggle('open');
    btn.nextElementSibling.classList.toggle('open');
  }

  function addHistory(query) {
    if (histEmpty) histEmpty.style.display = 'none';
    history.unshift(query);
    histList.innerHTML = '';
    history.forEach((q, i) => {
      const el = document.createElement('div');
      el.className = 'history-item' + (i === 0 ? ' active' : '');
      el.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg><span>' + esc(q) + '</span>';
      el.title = q;
      el.onclick = () => {
        document.querySelectorAll('.history-item').forEach(h => h.classList.remove('active'));
        el.classList.add('active');
      };
      histList.appendChild(el);
    });
  }

  function scrollBottom() { chat.scrollTop = chat.scrollHeight; }

  function esc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
             .replace(/"/g,'&quot;').replace(/\\n/g,'<br>');
  }
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    examples_html = "\n".join(
        f'<button class="example-btn" data-query="{q}">{q}</button>'
        for q in EXAMPLE_QUESTIONS
    )
    return HTML.replace("EXAMPLES_PLACEHOLDER", examples_html)
