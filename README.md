# Practical-Agents

An **agentic RAG** system built **eval-first**: a tool-using LLM agent answers
questions over a technical documentation corpus, and every answer is scored for
**retrieval quality, correctness, faithfulness, cost, and latency**. The point of
this project is not that an agent can answer questions — it's that you can *prove
how well it does*, find where it breaks, and make principled tradeoffs.

> Most portfolio "agent" projects ship a demo and stop. This one ships the
> measurement: a reproducible evaluation harness with an LLM-as-judge, retrieval
> metrics, and ablation studies. The harness is the product.

## TL;DR

```bash
make setup      # create venv, install deps
make ingest     # chunk + embed the corpus into a vector index
make demo       # ask the agent a question interactively
make eval       # run the full evaluation harness -> eval/results/
```

`make eval` runs with **zero API keys** using a local embedding + FAISS backend,
so anyone can reproduce the numbers. Point it at **Azure OpenAI + Azure AI
Search** with a `.env` file to run the production path (see
[Configuration](#configuration)).

### Run it without `make` (Windows / any OS)

`make` is a convenience for Unix/CI. On any platform, from the project root:

```bash
pip install -r requirements.txt          # numpy is the only hard requirement
python -m src.practical_agents.ingest     # build the index
python -m eval.run_eval                    # -> eval/results/report.md
python -m eval.run_eval --ablate           # -> eval/results/ablation.md
python scripts/demo.py "How many retries by default?"
```

> Optional extras (`sentence-transformers`, `faiss-cpu`) are auto-detected and
> improve retrieval; without them the pipeline still runs on a built-in hashed
> embedding + NumPy search. If you have an `OPENAI_API_KEY`/Azure keys set in
> your shell, they take priority — unset them to force the reproducible local
> path.

## Why this design

| Decision | Rationale |
| --- | --- |
| **Eval-first** | The scarce skill in applied ML isn't building an agent — it's knowing whether it's good. The harness (`eval/`) is the core deliverable. |
| **Azure-first, provider-agnostic** | Runs on Azure OpenAI + Azure AI Search in production, but falls back to local FAISS + `sentence-transformers` so results are reproducible offline. |
| **Closed-world synthetic corpus** | The docs describe a fictional library ("Meridian"). Because the world is closed, every question has an *unambiguous* ground truth, which makes faithfulness/groundedness measurable rather than a judgment call. Swap in real docs by pointing `CORPUS_DIR` at any folder of `.md` files. |
| **Agent, not just a pipeline** | The model decides *whether* and *how many times* to search, and can say "not in the docs." That decision quality is itself evaluated (see over/under-retrieval metrics). |

## Architecture

```
                      ┌─────────────────────────────────────────┐
   question ─────────▶│                 Agent                    │
                      │  (plans, calls search_docs 0..N times,   │
                      │   decides when it has enough context)    │
                      └───────┬───────────────────────┬──────────┘
                              │ search_docs(query)     │ answer
                              ▼                        ▼
                     ┌─────────────────┐      grounded answer + citations
                     │    Retriever    │
                     │  embed → top-k  │
                     │  → (rerank)     │
                     └────────┬────────┘
                              ▼
                     ┌─────────────────┐     Backends (auto-selected):
                     │  Vector Store   │      • Azure AI Search  (prod)
                     │  FAISS / Azure  │      • FAISS + local ST (default)
                     └─────────────────┘

   Everything above is measured by  eval/  ─────────────────────────────┐
   • retrieval:  recall@k, MRR, context precision                       │
   • answer:     correctness (judge), faithfulness/groundedness (judge) │
   • behavior:   abstention accuracy, avg tool calls                    │
   • ops:        latency p50/p95, token cost per query                  │
```

## Evaluation

The harness scores four families of metrics on a golden set (`eval/golden.jsonl`):

1. **Retrieval** — did we fetch the right chunks?
   `recall@k`, `MRR`, `context precision`.
2. **Answer quality** — an LLM-judge grades each answer against the reference for
   `correctness` and against the *retrieved context* for `faithfulness`
   (did the answer invent anything not in the retrieved text?).
3. **Behavior** — for questions whose answer is deliberately *not* in the corpus,
   does the agent correctly **abstain** instead of hallucinating?
   (`abstention precision/recall`.)
4. **Operations** — `latency p50/p95` and estimated `$/query`.

### Results (local backend, reproducible with `make eval`)

Measured on the 20-question golden set with the **zero-key local backend**
(MiniLM embeddings + FAISS + extractive answerer). Anyone can reproduce these:

| Metric | Value | Reading |
| --- | --- | --- |
| recall@5 | **0.895** | retrieval finds the right doc ~90% of the time |
| MRR | **0.781** | the right doc is usually rank 1 |
| faithfulness | **1.000** | extractive answers never invent facts |
| correctness | **0.200** | the trivial local generator is weak — see below |
| abstention accuracy | **0.800** | out-of-scope handling, floored by the local stub |
| latency p95 | **0.04 s** | local, no network |

**How to read the low correctness/abstention.** The offline answerer is an
*extractive stub* on purpose: it exists so the harness runs with no API keys.
It returns sentences by lexical overlap, so it cannot reason that a fact is
**absent or negated** — e.g. "Does Meridian support Kafka?" retrieves the broker
docs, finds overlap, and answers instead of declining. That is a fundamental
limit of extractive QA, and it is exactly the gap a real LLM closes. Retrieval
(recall@5 = 0.90) and faithfulness (1.00) are already strong; **correctness and
abstention are the metrics that climb when you plug in Azure OpenAI**, because
the harness stays identical and only the generator/judge get smarter.

### Ablations (`make ablate`)

The sweep produces a real, interpretable signal — reranking and smaller chunks
each recover the last ~10% of retrieval recall:

| Config | recall@k | faithfulness | correctness | abstain acc | p95 latency (s) | $/1k q |
| --- | --- | --- | --- | --- | --- | --- |
| k=3, no rerank, chunk=220 | 0.895 | 1.000 | 0.200 | 0.800 | 0.04 | 0.10 |
| k=5, no rerank, chunk=220 | 0.895 | 1.000 | 0.200 | 0.800 | 0.04 | 0.15 |
| k=5, **+ rerank**, chunk=220 | **1.000** | 1.000 | 0.250 | 0.800 | 0.83 | 0.16 |
| k=5, no rerank, **chunk=120** | **1.000** | 1.000 | 0.300 | 0.800 | 0.04 | 0.10 |

**Takeaway:** on this corpus, halving the chunk size buys the same recall gain as
a cross-encoder reranker at ~**20× lower latency** (0.04 s vs 0.83 s p95) — the
kind of cost/quality tradeoff the harness exists to surface. Numbers regenerate
into `eval/results/`.

## Configuration

Copy `.env.example` to `.env`. With **no** variables set, the project runs the
local backend. To use Azure:

```dotenv
# --- Azure OpenAI (generation + embeddings) ---
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small

# --- Azure AI Search (vector store) — optional; else local FAISS ---
AZURE_SEARCH_ENDPOINT=https://<resource>.search.windows.net
AZURE_SEARCH_API_KEY=<key>
AZURE_SEARCH_INDEX=practical-agents
```

The provider layer (`src/practical_agents/config.py`) auto-selects a backend at
runtime, so the same code path is exercised locally and on Azure.

## Project layout

```
corpus/                     closed-world synthetic docs (the knowledge base)
src/practical_agents/
  config.py                 provider/config resolution (Azure-first, local fallback)
  llm.py                    chat + judge client (Azure OpenAI / OpenAI / local)
  embeddings.py             embedding client (Azure / OpenAI / sentence-transformers / hashing)
  vectorstore.py            FAISS (local) and Azure AI Search backends
  ingest.py                 chunk → embed → index
  retriever.py              top-k retrieval (+ optional cross-encoder rerank)
  tools.py                  the search_docs tool the agent calls
  agent.py                  the tool-using agent loop
eval/
  golden.jsonl              questions with reference answers + relevant doc ids
  metrics.py                retrieval + behavior metrics (pure functions, tested)
  judge.py                  LLM-as-judge for correctness + faithfulness
  run_eval.py               runs the harness → eval/results/report.md
scripts/demo.py             interactive CLI
tests/                      unit tests for metrics + chunking
```

## Roadmap / what I'd do next

- Add a **hard-negative** retrieval test set to stress the reranker.
- Swap the synthetic corpus for a real one (Azure SDK docs) and re-baseline.
- Add **online eval**: sample live queries and score them nightly.
- Trace every run with token-level cost attribution per tool call.

## License

MIT
