# first_crew — Yelp Review Prediction with CrewAI

> **AgentSociety Challenge · Track 1 — User Modeling / Recommendation**

A multi-agent pipeline that predicts **(a) star rating** and **(b) a plausible review** given a `(user_id, item_id)` pair from the Yelp dataset. Built with **CrewAI Flows**, **Nvidia Build API** (`minimaxai/minimax-m2.7`), and **ChromaDB** for RAG.

**50 rows · 100% success · MAE 0.71 stars · text-cosine 0.7974**

---

## Deliverables

| # | Deliverable | Status |
|---|---|:---:|
| 1 | Index-Reuse RAG mechanism | Done |
| 2 | `Process.sequential` crew — Pattern 2 Collaborative | Done |
| 3 | `Process.hierarchical` crew — Manager Agent | Done |
| 4 | New agents: Calibrator + Manager | Done |
| B1 | Bonus: EDA knowledge source | Done |
| B2 | Bonus: CrewAI Flow integration | Done |

---

## Installation

**Requirements:** Python ≥ 3.10, < 3.14 · [`uv`](https://docs.astral.sh/uv/) package manager

```bash
# Install uv (Windows PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Clone and install
git clone <this-repo-url>
cd assignment_1
uv sync
```

---

## Configuration

Create a `.env` at the **workspace root** (one level above `assignment_1/`):

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...          # get at build.nvidia.com
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1

RAG_ENABLED=1                     # Lab Task 1
EDA_KNOWLEDGE_ENABLED=1           # Bonus
CALIBRATOR_ENABLED=1              # Lab Task 4
```

**Install the ChromaDB index (one-time, 4.7 GB):**

```bash
# Windows
cp ../chroma_index/chroma.sqlite3 "$LOCALAPPDATA/CrewAI/assignment_1/chroma.sqlite3"

# Linux / macOS
cp ../chroma_index/chroma.sqlite3 ~/.local/share/CrewAI/assignment_1/chroma.sqlite3
```

---

## Running

All commands use `uv run` — never `pip install` or `activate`.

```bash
# Smoke test (2 rows, ~1 min)
uv run first_crew --limit 2

# Full 50-row run
rm -f results/predictions.jsonl
uv run first_crew --limit 52

# Alternative crew topologies
uv run python demo_pattern2_crew.py       # Process.sequential
uv run python demo_hierarchical_crew.py  # Process.hierarchical
```

> The runner sleeps 10s between rows to respect Nvidia free-tier limits. Override with `INTER_ROW_SLEEP_SEC=0`. Runs are fully **resumable** — completed rows are skipped on restart.

---

## Architecture

### Agents

| Agent | Role | Tools |
|---|---|---|
| `user_profiler` | Taste, writing style, rating tendencies | — |
| `item_analyst` | Business strengths & customer experience | RAG (`search_historical_reviews`) |
| `calibrator` | Attacks regression-to-mean with rating priors | — |
| `prediction_modeler` | Synthesises profiles → `{"stars", "review"}` | — |
| `review_prediction_manager` | *(Hierarchical only)* Delegates & validates | — |

### Default Flow

```
init_request
     │
     ├── profile_user ──→ calibrate_user ──┐
     │                                      and_() ──→ predict
     └── profile_item ─────────────────────┘
```

User and item profiling run **in parallel**, cutting wall-clock time ~50%.

### Alternative Topologies

| Lab | Factory | Pattern |
|---|---|---|
| 1 | `pattern2_collaborative_crew()` | Hub-and-spoke; Prediction Modeler owns one task, 3 peers answer delegated queries |
| 2 | `hierarchical_predict_crew()` | Manager above 4 workers; delegates and validates |

---

## Knowledge Sources

**1. Per-record context** — User record + 10 recent reviews + item record, injected directly into agent prompts via Pydantic field substitution. Covers ~95% of rows.

**2. Semantic RAG** — ChromaDB collection `benchmark_true_fresh_index_Filtered_Review_1` (100k+ reviews, `BAAI/bge-small-en-v1.5`). Used by the Item Analyst for sparse-item rows. Smart-cache pattern: sqlite3 probe runs before instantiation to prevent accidental 3-hour re-indexing.

**3. EDA Knowledge** — `docs/EDA_Knowledge.md` (generated from 382k reviews). Mounted on every Crew via `StringKnowledgeSource`. Key stats agents receive:
- 42.7% of reviews are 5★ · 10.7% are 1★
- User buckets: 18% Strict · 38% Balanced · 44% Lenient
- Global mean business star: 3.60

**4. Calibrator history** — Top-3 most extreme past reviews per user (deterministic sort, no embeddings) injected as direct context.

---

## Results

| Metric | Value |
|---|---|
| Rows | 50 |
| Success rate | **100%** |
| MAE (stars) | **0.71** |
| Text cosine (`bge-small`) | **0.7974** |
| Cold-start fallbacks | 0 |
| Empty reviews | 0 |

### Key Findings

**Phi-3 failed at scale.** `ollama/phi3` produced 73% empty reviews from 4K context overflow. Switching to `minimax-m2.7` eliminated the failure.

**RAG alone didn't move MAE; the Calibrator did.** The residual error was regression-to-mean on extreme users — not context starvation. The Calibrator directly attacked this as an explicit rating-prior layer.

**Flow parallelism is the right primitive.** Parallel profiling with `and_()` halved wall-clock time; Pydantic state gave an explicit contract between agents that made debugging tractable.

---

## Known Limitations

- **~45% cold-start users** — no record in `filtered_user.json` → forced `stars=3.75` + neutral review.
- **Minimax hedges to 3.75** — small-model regression-to-mean bias; stronger base models or OpenEvolve calibration (Milestone 2) will address this.
- **Nvidia free-tier pacing** — 10s inter-row sleep; set `INTER_ROW_SLEEP_SEC=0` on paid tiers.

---

## Project Layout

```
assignment_1/
├── src/first_crew/
│   ├── main.py              # CLI + batch runner + rate-limit pacing
│   ├── flow.py              # @start → and_(calibrate, profile_item) → predict
│   ├── crews.py             # Crew factories, LLM switch, RAG tool, knowledge source
│   ├── data_store.py        # In-memory index over filtered_*.json
│   └── config/
│       ├── agents.yaml      # 5 agents (role / goal / backstory)
│       └── tasks.yaml       # 6 task variants
├── data/                    # 8.6 MB Yelp subset (committed)
├── docs/
│   ├── EDA_Knowledge.md     # Generated EDA stats (StringKnowledgeSource)
│   └── Yelp_Data_Translation.md
├── results/
│   ├── predictions.jsonl    # 50-row append-only output
│   └── metrics.json         # MAE + cosine
├── demo_pattern2_crew.py
├── demo_hierarchical_crew.py
├── build_eda_knowledge.py
└── pyproject.toml · uv.lock
```

---

## Tech Stack

| Component | Choice |
|---|---|
| Agent framework | [CrewAI](https://crewai.com) — Flows + Crews + Knowledge |
| LLM | `minimaxai/minimax-m2.7` via [Nvidia Build](https://build.nvidia.com/) (free tier) |
| Vector DB | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (384-dim, CPU) |
| Package manager | [Astral uv](https://docs.astral.sh/uv/) |

---

