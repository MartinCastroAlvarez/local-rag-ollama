// local-rag web UI. Everything that comes from documents or the model is inserted with
// textContent / createTextNode (never as HTML), so a hostile document cannot run script.
"use strict";

const $ = (id) => document.getElementById(id);
const turns = [];
let sources = [];

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const s = await res.json();
    $("status").textContent =
      `${s.documents} documents, ${s.chunks} chunks - embedder ${s.embedder || "none"} ` +
      `(${s.vector_index}) - chat ${s.chat_model}`;
    $("mode").value = s.mode;
    $("topk").value = s.top_k;
  } catch (err) {
    $("status").textContent = "index status unavailable";
  }
}

async function openChunk(n) {
  const src = sources[n - 1];
  if (!src) return;
  const res = await fetch(`/api/chunks/${encodeURIComponent(src.chunk_id)}`);
  if (!res.ok) return;
  const chunk = await res.json();
  $("viewer-title").textContent = `[${n}] ${chunk.source}`;
  $("viewer-location").textContent = chunk.location;
  $("viewer-text").textContent = chunk.text;
  $("viewer").hidden = false;
}

// Render answer text, turning valid [n] markers into buttons that open the source.
function renderAnswer(text, valid) {
  const box = $("answer");
  box.replaceChildren();
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    box.appendChild(document.createTextNode(text.slice(last, m.index)));
    const n = Number(m[1]);
    if (valid === null || valid.includes(n)) {
      const btn = el("button", `[${n}]`, "cite");
      btn.type = "button";
      btn.addEventListener("click", () => openChunk(n));
      box.appendChild(btn);
    }
    last = re.lastIndex;
  }
  box.appendChild(document.createTextNode(text.slice(last)));
}

function renderSources(list, cited) {
  const ol = $("sources");
  ol.replaceChildren();
  list.forEach((s) => {
    const li = el("li");
    if (cited.includes(s.n)) li.className = "cited";
    const btn = el("button", `[${s.n}] ${s.source}`, "source");
    btn.type = "button";
    btn.addEventListener("click", () => openChunk(s.n));
    li.append(btn, el("span", ` ${s.location}`, "muted"), el("p", s.preview, "preview"));
    ol.appendChild(li);
  });
}

async function ask(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("send").disabled = true;
  $("answer-box").hidden = false;
  $("viewer").hidden = true;
  $("meta").textContent = "retrieving...";
  sources = [];
  renderSources([], []);
  renderAnswer("", null);
  const body = {
    question,
    mode: $("mode").value,
    top_k: Number($("topk").value) || undefined,
    history: $("followup").checked ? turns.slice(-3) : [],
  };
  let text = "";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "sources") {
          sources = ev.sources;
          renderSources(sources, []);
          $("meta").textContent = `generating (${ev.mode})...`;
        } else if (ev.type === "token") {
          text += ev.text;
          renderAnswer(text, null);
        } else if (ev.type === "done") {
          renderAnswer(ev.text, ev.citations);
          renderSources(sources, ev.citations);
          const t = ev.timings || {};
          const ms = (k) => (t[k] !== undefined ? `${Math.round(t[k] * 1000)} ms` : "-");
          $("meta").textContent =
            (ev.refused ? "Not enough context to answer. " : "") +
            `retrieve ${ms("retrieve")}, first token ${ms("first_token")}, generate ${ms("generate")}`;
          turns.push([question, ev.text]);
        } else if (ev.type === "error") {
          $("meta").textContent = `Error: ${ev.message}`;
        }
      }
    }
  } catch (err) {
    $("meta").textContent = `Error: ${err.message}`;
  } finally {
    $("send").disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("ask").addEventListener("submit", ask);
  $("viewer-close").addEventListener("click", () => { $("viewer").hidden = true; });
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("ask").requestSubmit();
  });
  loadStatus();
});
