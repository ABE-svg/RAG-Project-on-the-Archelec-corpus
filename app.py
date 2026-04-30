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
        print(f"Index built and saved. {len(chunks)} chunks from {len(documents)} documents.")

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
    )
    chat_model = ChatHuggingFace(llm=llm)

    state["faiss_db"] = faiss_db
    state["chat_model"] = chat_model

    print(f"Ready. {len(chunks)} chunks indexed from {len(documents)} documents.")
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
    response = state["chat_model"].invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(request: QueryRequest):
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
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Archelec RAG • 1988</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d1117;
      --surface:   #161b22;
      --border:    #30363d;
      --primary:   #58a6ff;
      --primary-d: #1f6feb;
      --text:      #e6edf3;
      --muted:     #8b949e;
      --accent:    #3fb950;
      --danger:    #f85149;
      --radius:    10px;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Header */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 18px 40px;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    header .logo {
      width: 36px; height: 36px;
      background: var(--primary-d);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; font-weight: 700; color: #fff;
    }
    header h1 { font-size: 17px; font-weight: 600; }
    header span { font-size: 13px; color: var(--muted); margin-left: 4px; }

    /* Layout */
    .layout {
      display: flex;
      flex: 1;
      height: calc(100vh - 61px);
      overflow: hidden;
    }

    /* Sidebar */
    aside {
      width: 280px;
      min-width: 280px;
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    aside section { padding: 20px; border-bottom: 1px solid var(--border); }
    aside h2 { font-size: 11px; font-weight: 600; text-transform: uppercase;
               letter-spacing: .08em; color: var(--muted); margin-bottom: 12px; }

    .example-btn {
      display: block; width: 100%;
      background: transparent;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      color: var(--text);
      font-size: 13px;
      padding: 9px 12px;
      text-align: left;
      cursor: pointer;
      margin-bottom: 8px;
      transition: border-color .15s, background .15s;
      line-height: 1.4;
    }
    .example-btn:hover { border-color: var(--primary); background: rgba(88,166,255,.07); }

    /* History */
    #history-list { overflow-y: auto; flex: 1; padding: 20px; }
    .history-item {
      padding: 9px 12px;
      border-radius: var(--radius);
      font-size: 13px;
      color: var(--muted);
      cursor: pointer;
      margin-bottom: 6px;
      border: 1px solid transparent;
      transition: all .15s;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .history-item:hover { color: var(--text); border-color: var(--border); background: rgba(255,255,255,.03); }
    .history-item.active { color: var(--primary); border-color: var(--primary-d); background: rgba(88,166,255,.08); }

    /* Main */
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* Chat area */
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 32px 40px;
      display: flex;
      flex-direction: column;
      gap: 28px;
    }

    .empty-state {
      margin: auto;
      text-align: center;
      color: var(--muted);
    }
    .empty-state .icon { font-size: 48px; margin-bottom: 16px; opacity: .4; }
    .empty-state p { font-size: 15px; }

    /* Message bubbles */
    .message { max-width: 760px; width: 100%; }
    .message.user { align-self: flex-end; }
    .message.assistant { align-self: flex-start; }

    .bubble {
      padding: 14px 18px;
      border-radius: var(--radius);
      font-size: 15px;
      line-height: 1.65;
    }
    .message.user .bubble {
      background: var(--primary-d);
      color: #fff;
      border-bottom-right-radius: 3px;
    }
    .message.assistant .bubble {
      background: var(--surface);
      border: 1px solid var(--border);
      border-bottom-left-radius: 3px;
    }

    /* Sources */
    .sources { margin-top: 12px; }
    .sources-toggle {
      background: none; border: none;
      color: var(--muted); font-size: 12px;
      cursor: pointer; display: flex; align-items: center; gap: 6px;
      padding: 0; transition: color .15s;
    }
    .sources-toggle:hover { color: var(--text); }
    .sources-toggle svg { transition: transform .2s; }
    .sources-toggle.open svg { transform: rotate(90deg); }

    .sources-list { display: none; margin-top: 10px; }
    .sources-list.open { display: flex; flex-direction: column; gap: 8px; }

    .source-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: var(--radius);
      padding: 10px 14px;
    }
    .source-card .source-name {
      font-size: 11px; font-weight: 600;
      color: var(--accent); margin-bottom: 6px;
      font-family: monospace; letter-spacing: .03em;
    }
    .source-card .source-text {
      font-size: 13px; color: var(--muted); line-height: 1.5;
      max-height: 80px; overflow: hidden;
      display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical;
    }

    /* Loader */
    .loader {
      display: flex; gap: 6px; align-items: center; padding: 14px 18px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); border-bottom-left-radius: 3px;
      width: fit-content;
    }
    .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--muted);
      animation: bounce .9s infinite ease-in-out;
    }
    .dot:nth-child(2) { animation-delay: .15s; }
    .dot:nth-child(3) { animation-delay: .30s; }
    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0.7); opacity: .4; }
      40%            { transform: scale(1);   opacity: 1; }
    }

    /* Input bar */
    .input-bar {
      padding: 20px 40px;
      background: var(--surface);
      border-top: 1px solid var(--border);
    }
    .input-wrap {
      display: flex; gap: 10px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 14px;
      transition: border-color .2s;
    }
    .input-wrap:focus-within { border-color: var(--primary); }

    #query-input {
      flex: 1; background: none; border: none; outline: none;
      color: var(--text); font-size: 15px; resize: none;
      font-family: inherit; line-height: 1.5; max-height: 120px;
    }
    #query-input::placeholder { color: var(--muted); }

    #send-btn {
      background: var(--primary);
      border: none; border-radius: 8px;
      width: 36px; height: 36px; min-width: 36px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; transition: background .15s, opacity .15s;
      align-self: flex-end;
    }
    #send-btn:hover { background: #79b8ff; }
    #send-btn:disabled { opacity: .4; cursor: not-allowed; }
    #send-btn svg { color: #000; }
  </style>
</head>
<body>

<header>
  <div class="logo">A</div>
  <h1>Archelec 1988</h1>
  <span>Question-réponse sur les professions de foi des législatives de 1988</span>
</header>

<div class="layout">
  <aside>
    <section>
      <h2>Exemples de questions</h2>
      EXAMPLES_PLACEHOLDER
    </section>
    <section style="flex:1; display:flex; flex-direction:column; overflow:hidden; border-bottom:none;">
      <h2>Historique</h2>
    </section>
    <div id="history-list"></div>
  </aside>

  <main>
    <div id="chat">
      <div class="empty-state">
        <div class="icon">🗳</div>
        <p>Pose une question sur les manifestes électoraux de 1988.</p>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-wrap">
        <textarea id="query-input" rows="1"
          placeholder="Ex. : Que dit-on de l'Europe dans les professions de foi ?"></textarea>
        <button id="send-btn" title="Envoyer">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
               viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  </main>
</div>

<script>
  const chat      = document.getElementById('chat');
  const input     = document.getElementById('query-input');
  const sendBtn   = document.getElementById('send-btn');
  const histList  = document.getElementById('history-list');

  let history = [];

  // Auto-resize textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = input.scrollHeight + 'px';
  });

  // Send on Enter (Shift+Enter = newline)
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  });
  sendBtn.addEventListener('click', sendQuery);

  // Example buttons
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

    // Remove empty state
    const empty = chat.querySelector('.empty-state');
    if (empty) empty.remove();

    // User bubble
    appendBubble('user', query);

    // Loader
    const loader = document.createElement('div');
    loader.className = 'message assistant';
    loader.innerHTML = '<div class="loader"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
    chat.appendChild(loader);
    chat.scrollTop = chat.scrollHeight;

    fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query})
    })
    .then(r => r.json())
    .then(data => {
      loader.remove();
      appendAnswer(data.answer, data.sources);
      addToHistory(query, data);
      sendBtn.disabled = false;
    })
    .catch(() => {
      loader.remove();
      appendBubble('assistant', 'Une erreur est survenue. Vérifie que le serveur est bien lancé.');
      sendBtn.disabled = false;
    });
  }

  function appendBubble(role, text) {
    const el = document.createElement('div');
    el.className = 'message ' + role;
    el.innerHTML = '<div class="bubble">' + escHtml(text) + '</div>';
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
  }

  function appendAnswer(answer, sources) {
    const el = document.createElement('div');
    el.className = 'message assistant';

    const sourcesHtml = sources.map(s => `
      <div class="source-card">
        <div class="source-name">${escHtml(s.source)}</div>
        <div class="source-text">${escHtml(s.chunk)}</div>
      </div>
    `).join('');

    el.innerHTML = `
      <div class="bubble">${escHtml(answer)}</div>
      <div class="sources">
        <button class="sources-toggle" onclick="toggleSources(this)">
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"
               viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
          ${sources.length} passage${sources.length > 1 ? 's' : ''} consulté${sources.length > 1 ? 's' : ''}
        </button>
        <div class="sources-list">${sourcesHtml}</div>
      </div>
    `;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
  }

  function toggleSources(btn) {
    btn.classList.toggle('open');
    btn.nextElementSibling.classList.toggle('open');
  }

  function addToHistory(query, data) {
    history.unshift({query, data});
    renderHistory();
  }

  function renderHistory() {
    histList.innerHTML = '';
    history.forEach((item, i) => {
      const el = document.createElement('div');
      el.className = 'history-item' + (i === 0 ? ' active' : '');
      el.textContent = item.query;
      el.title = item.query;
      el.onclick = () => {
        document.querySelectorAll('.history-item').forEach(h => h.classList.remove('active'));
        el.classList.add('active');
      };
      histList.appendChild(el);
    });
  }

  function escHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
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
