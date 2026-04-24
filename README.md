# first_crew — A CrewAI Multi-Agent System for Yelp Review Prediction

> **Track:** AgentSociety Challenge — Track 1 (User Modeling / Recommendation)

A multi-agent pipeline that, given a `(user_id, item_id)` pair from the Yelp
dataset, predicts **(a)** the star rating that user would give and **(b)** a
plausible review text in the user's voice. Built with **CrewAI Flows**, the
**Nvidia Build API** (`minimaxai/minimax-m2.7`, free tier), and **ChromaDB index** is used for retrieval-augmented context.


---

## 📋 Deliverables

| # | Deliverable | Status | Where it lives in this repo |
|---|---|:---:|---|
| **1** | **Index-Reuse mechanism integration** | Done | [`src/first_crew/crews.py::_create_rag_tool`](src/first_crew/crews.py) — sqlite3 probe + `FixedJSONSearchToolSchema` swap, mirroring the reference pattern. `else`-branch raises instead of silently re-indexing, defending against Pitfall 1. |
| **2** | **Crew with `Process.sequential` (Pattern 2: Collaborative Single Task)** | Done | [`crews.py::pattern2_collaborative_crew()`](src/first_crew/crews.py) · demo entry: [`demo_pattern2_crew.py`](demo_pattern2_crew.py). Prediction Modeler is the primary (`allow_delegation=True`); User Profiler + Item Analyst + Calibrator are peers. |
| **3** | **Crew with `Process.hierarchical` (Manager Agent)** | Done | [`crews.py::hierarchical_predict_crew()`](src/first_crew/crews.py) · demo entry: [`demo_hierarchical_crew.py`](demo_hierarchical_crew.py). Explicit `manager_agent=` (role `review_prediction_manager`) routes & validates work across the 4 workers. |
| **4** | **New agents to strengthen the crew** | Done | Calibrator (`calibrator` in [`agents.yaml`](src/first_crew/config/agents.yaml)) + Manager (`review_prediction_manager`). The Calibrator directly attacks the regression-to-mean failure (see Current findings). |
| **B1** | **Bonus: EDA knowledge source** | Done | [`build_eda_knowledge.py`](build_eda_knowledge.py) → [`docs/EDA_Knowledge.md`](docs/EDA_Knowledge.md) (4.9 KB, derived from 382k reviews / 26k users / 22k items). Mounted via `StringKnowledgeSource` on every Crew, with explicit `sentence-transformer/BAAI/bge-small-en-v1.5` embedder to defend against Pitfall 4. |
| **B2** | **Bonus: Crew integrated into a CrewAI Flow** | Done | [`src/first_crew/flow.py`](src/first_crew/flow.py) — the default pipeline. `@start → profile_user → calibrate_user ∧ profile_item → predict` with Pydantic state. |

Headline: **50 rows · 100% success · MAE 0.71 stars · text-cosine 0.7974**.

---

## ⚙️ Installation

Ensure you have **Python ≥ 3.10, < 3.14** installed.
This project uses **[Astral `uv`](https://docs.astral.sh/uv/)** exclusively
for dependency and environment management.

### 1. Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Clone and install dependencies

```bash
git clone <this-repo-url>
cd assignment_1
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, creates a local `.venv/`, and
pins every dependency to the exact version locked by the author.

---

## 🔑 Configuration

### Environment file

Create a `.env` file at the **workspace root** (one level above `assignment_1/`):

```dotenv
# Primary LLM — Nvidia Build free tier (course-recommended)
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...                                    # get one at build.nvidia.com
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1

# Feature flags (all three ON by default for the final submission)
RAG_ENABLED=1                 # Lab Task 1 — smart-cache RAG over chroma_index
EDA_KNOWLEDGE_ENABLED=1       # Bonus — inject EDA stats on every Crew
CALIBRATOR_ENABLED=1          # Lab Task 4 — run the Calibrator between profile_user and predict

# Local fallback (only used if LLM_PROVIDER=ollama)
MODEL=ollama/phi3
```

To fall back to a **local Ollama Phi-3** model (slower, less accurate, but
offline), set `LLM_PROVIDER=ollama` and start Ollama locally. The
`.env.example` template is a copy of the above with the API key redacted.

### Install the pre-built ChromaDB index (one-time)

A 4.7 GB `chroma_index/chroma.sqlite3` containing
100k+ indexed reviews. Copy it into the CrewAI storage path so the RAG
layer finds it:

```bash
# Windows
cp ../chroma_index/chroma.sqlite3 "$LOCALAPPDATA/CrewAI/assignment_1/chroma.sqlite3"

# Linux / macOS
cp ../chroma_index/chroma.sqlite3 ~/.local/share/CrewAI/assignment_1/chroma.sqlite3
```

The rationale (CrewAI hard-codes the persist path via pydantic-captured
constants; copying is more robust than monkey-patching) is in the design
notes in `crews.py`.

---

## ▶️ Running the Project

Every command is driven by **`uv run`**. Never `pip install` or
`source .venv/bin/activate`.

### Two-row smoke test (~1 minute)

```bash
uv run first_crew --limit 2
```

Expected output: both rows complete, `results/predictions.jsonl` has two
lines with `"ok": true`, and the terminal prints `=== Metrics ===` with MAE
and cosine.

### Full 50-row deliverable run

```bash
rm -f results/predictions.jsonl          # clean slate
uv run first_crew --limit 52             # test file has 2 duplicate pairs → 50 unique predictions
```

The runner **paces requests with a 10-second sleep between rows**
(`INTER_ROW_SLEEP_SEC=10`, override via env). This keeps us below the Nvidia
free-tier rate limit. Set to `0` to disable if your tier is paid.

### Resumability

The runner is **fully resumable**. Each completed row is appended to
`results/predictions.jsonl`; on the next invocation, the runner reads the
file, builds a set of completed `(user_id, item_id)` pairs, and skips them.
A crash at row 37 costs you only the in-flight row.

### Alternative crew topologies

```bash
# Pattern 2 — Collaborative Single Task (hub-and-spoke with delegation)
uv run python demo_pattern2_crew.py

# Process.hierarchical — Manager Agent coordinates 4 workers
uv run python demo_hierarchical_crew.py
```

Each runs **one** prediction (row 1 of the test file) through the alternative
topology and prints the crew's final output.

---

## 🏗️ Architecture

### The Three Pillars

1. **CrewAI** — orchestration of five specialist agents.
2. **RAG (ChromaDB + sentence-transformer)** — semantic retrieval over the
   historical-review corpus.
3. **OpenEvolve** — Future.

### Crew Members (5 Agents + 1 Manager)

Every agent is configured by a YAML role/goal/backstory block in
[`config/agents.yaml`](src/first_crew/config/agents.yaml) and a matching task
in [`config/tasks.yaml`](src/first_crew/config/tasks.yaml). **No prompt text
lives in Python** — `crews.py` only assembles.

| # | Agent | Responsibility | Tools | Output |
|---|---|---|---|---|
| 1 | **`user_profiler`** | Characterise the reviewer's taste, writing style, and rating tendencies | None (user record + 10 recent reviews pre-injected as JSON) | Markdown user profile |
| 2 | **`item_analyst`** | Extract the strengths, customer experience, and notable attributes of a business | `search_historical_reviews` (RAG over the review corpus) | Markdown business evaluation |
| 3 | **`calibrator`** | *New agent *. Attack regression-to-mean by deriving an `expected_range` + `most_likely_rating` + `confidence` from the user's most extreme past reviews | None (top-3 extremes are pre-computed deterministically and injected) | Markdown calibration report |
| 4 | **`prediction_modeler`** | Synthesise the three profile reports into a star + review-text prediction; **rule #1: stars MUST fall inside the Calibrator's expected_range** | None | Strict JSON `{"stars": …, "review": …}` |
| +1 | **`review_prediction_manager`** | *Hierarchical-variant only.* Sits above the team, delegates to workers, validates each result | None | Orchestration only |

### How They Collaborate (the default Flow)

```
              ┌─────────────┐
              │ init_request│
              └──────┬──────┘
                     │
         ┌───────────┴───────────┐
         │                       │
   ┌───────────┐           ┌───────────┐
   │profile_user│           │profile_item│   (run in parallel)
   └─────┬─────┘           └─────┬─────┘
         │                       │
   ┌────────────┐                │
   │calibrate_user│               │            (Lab 4 — after profile_user)
   └─────┬──────┘                │
         │                       │
         └──────── and_() ───────┘
                     │
               ┌──────────┐
               │  predict │
               └──────────┘
```

### Two Alternative Topologies

| Lab | Factory | Topology | Primary/Manager | Demo |
|---|---|---|---|---|
| **1** | `pattern2_collaborative_crew()` | `Process.sequential` with Pattern 2 — hub-and-spoke | Prediction Modeler (primary) owns ONE task; 3 peers wait for delegated questions | [`demo_pattern2_crew.py`](demo_pattern2_crew.py) |
| **2** | `hierarchical_predict_crew()` | `Process.hierarchical` with explicit Manager | `review_prediction_manager` above 4 workers; Manager delegates AND validates | [`demo_hierarchical_crew.py`](demo_hierarchical_crew.py) |

---

## 📚 Knowledge Sources

Four complementary sources feed the agents, on purpose:

### 1. Per-record context (deterministic, injected as JSON)

Primary-key lookup in [`data_store.py`](src/first_crew/data_store.py)
fetches user record + 10 most-recent past reviews + item record from the
`Complete Training Set/filtered_*.json` files. These are placed directly
into agent prompts via Pydantic field substitution
(`{user_record_json}`, `{item_record_json}`, `{past_reviews_json}`). This
is the dominant channel and covers ~95% of rows.

### 2. Semantic RAG (`search_historical_reviews`)

For the small fraction of rows where the item record is too sparse, the
Item Analyst can call a **single RAG query** against the pre-built
ChromaDB collection `benchmark_true_fresh_index_Filtered_Review_1`
(100k+ reviews, `BAAI/bge-small-en-v1.5` embeddings). The tool is
constructed with the reference's **smart-cache pattern**:

- sqlite3 probe confirms the collection exists **before** the tool is
  instantiated — no accidental 3-hour re-index loop.
- `args_schema = FixedJSONSearchToolSchema` hides the `json_path` parameter
  from the LLM, so the agent can't accidentally trigger a rebuild.
- The `else`-branch **raises `RuntimeError`** rather than silently falling
  back to indexing from scratch.

### 3. EDA knowledge source (Bonus)

[`docs/EDA_Knowledge.md`](docs/EDA_Knowledge.md) is generated by
[`build_eda_knowledge.py`](build_eda_knowledge.py) from a 382k-review scan.
Every Crew mounts it via `knowledge_sources=[...]`. Key stats the agents see:

- Rating distribution: **42.7% of reviews are 5★, 10.7% are 1★** — 53% at extremes.
- User-calibration buckets: **18% Strict, 38% Balanced, 44% Lenient**.
- Review length by star (1-2★ reviews average 139-144 words; 5★ reviews average 98).
- Global mean business star: **3.60**.

**Note:** the embedder is set explicitly to
`sentence-transformer/BAAI/bge-small-en-v1.5`; a dedicated
`collection_name="eda_knowledge_bge_small_v1"` avoids colliding with the
pre-existing `knowledge_crew` collection in `chroma.sqlite3`.

### 4. Extreme-history retrieval for the Calibrator (Lab 4)

`data_store.get_user_extreme_reviews(user_id, avg_stars, k=3)` returns the
user's top-3 most-negative + top-3 most-positive past reviews by
`|stars − user_avg|`. This is **not RAG** (no embeddings) — it's an O(n)
deterministic sort over the user's own history, chosen over RAG because of
an unresolvable CrewAI parsing issue (agents emitting tool-call JSON as a
Final Answer). See `crews.py::calibrator_crew` for the full design note.

---

## 📊 Results (Current findings)

### Final 50-row evaluation (RAG + EDA + Calibrator, full stack)

| Metric | Value |
|---|---|
| Rows total | 50 |
| Rows OK (valid prediction produced) | 50 |
| **Success rate** | **100%** |
| Empty review rate | 0% |
| Cold-start fallbacks | 0 |
| **MAE (stars)** | **0.71** |
| **Text cosine mean (`bge-small`)** | **0.7974** |

### Three concrete observations

**Observation 1 — The Phi-3 local model catastrophically failed at scale.**
Initial runs with `ollama/phi3` produced **73% empty reviews** due to a 4K
context overflow on heavy-tail users. Pivoting to `minimax-m2.7` via
Nvidia Build eliminated the failure
entirely. Phi-3 remains as an `else`-branch fallback in `crews.py` for
offline development.

**Observation 2 — RAG alone did not improve accuracy; Calibrator did.**
Adding semantic RAG on the Item Analyst moved MAE only marginally (noise
range). The residual error is not *context starvation* but
**regression-to-the-mean on extreme ratings** — the Prediction Modeler
defaults toward 3.5-4.0 even for Strict (avg ≤ 3.0) or Lenient (avg ≥ 4.0)
users. Building the Calibrator as an explicit rating-prior layer is what
actually attacked this failure mode.

**Observation 3 — The Flow architecture is the right primitive.**
Factoring user and item profiling into parallel nodes with an `and_()` join
cuts wall-clock time roughly in half vs. a naïve serial chain, while the
Pydantic state model gives us an explicit data contract between agents —
invaluable when debugging the Phi-3 and RAG failures above.

---

## 🗂️ Project Layout

```
assignment_1/
├── README.md                        ← this file
├── build_eda_knowledge.py           ← regenerates docs/EDA_Knowledge.md
├── demo_pattern2_crew.py            ← Lab 1 — Pattern-2 Collaborative Single Task
├── demo_hierarchical_crew.py        ← Lab 2 — Process.hierarchical
├── pyproject.toml  ·  uv.lock       ← uv-managed deps
├── data/                            ← 8.6 MB Yelp subset (committed for reproducibility)
│   ├── filtered_user.json           │    covers ~55% of test users
│   ├── filtered_item.json           │    covers ~100% of test items
│   └── test_review_subset.json      │    198-row test file
├── Complete Training Set/           ← 323 MB full corpus (gitignored)
├── docs/
│   ├── EDA_Knowledge.md             ← Bonus — generated EDA stats as a StringKnowledgeSource
│   └── Yelp_Data_Translation.md     ← schema glossary
├── results/
│   ├── predictions.jsonl            ← 50 rows, per-row JSON (resumable append-only)
│   └── metrics.json                 ← aggregate MAE + cosine
└── src/first_crew/
    ├── __init__.py
    ├── main.py                      ← CLI entry point + batch runner + rate-limit pacing
    ├── flow.py                      ← @start → and_(calibrate_user, profile_item) → predict
    ├── crews.py                     ← Crew factories + LLM switch + RAG tool + knowledge source
    ├── data_store.py                ← in-memory dict index over filtered_*.json
    └── config/
        ├── agents.yaml              ← 5 agents (role/goal/backstory)
        └── tasks.yaml               ← 6 task variants (description/expected_output)
```

---

## 🛡️ Compliance with `.agents/rules/`

The instructor's rules at
[`Rag_Crew_Profiler-main/.agents/rules/`](../Rag_Crew_Profiler-main/.agents/rules/)
apply to every contribution.

| Rule | Compliance | Notes |
|---|---|---|
| `uv-package-management.md` | ✅ Full | All installs via `uv add`; all scripts via `uv run`; `pyproject.toml` + `uv.lock` are the SSoT. No `pip`/`poetry`/`conda` used. |
| `crewai-strict-separation.md` (YAML-first) | ✅ Spirit | All agent role/goal/backstory in `agents.yaml`; all task description/expected_output in `tasks.yaml`; zero hardcoded prompts in Python. We use factory functions instead of `@CrewBase` because our orchestration is Flow-based (matching the professor's own PDF example for `[Lifecycle B]`). |
| `vudovn-antigravity-kit-clean-code.md` | ✅ Full | SRP, DRY (`_build_crew`, `_make_agent`, `_attach_knowledge` helpers), KISS, YAGNI. Every non-demo factory function is under 30 lines. |

### Four documented pitfalls — all defended against

| Pitfall (from [`Embedding_Index_Lessons_Learned.md`](../Rag_Crew_Profiler-main/docs/Embedding_Index_Lessons_Learned.md)) | How we handled it |
|---|---|
| **1 — `json_path` triggers 3-hour re-chunking** | `_create_rag_tool` raises `RuntimeError` in the else-branch (no silent rebuild); `FixedJSONSearchToolSchema` hides `json_path` from the LLM |
| **2 — ChromaDB singleton collision** | Production code touches ChromaDB ONLY via `JSONSearchTool` and CrewAI's internal knowledge store. No second `PersistentClient` is ever created |
| **3 — Vague tool descriptions** | `search_historical_reviews` description includes input contract, concrete example query, and when-not-to-use rule |
| **4 — `StringKnowledgeSource` silently uses OpenAI** | Explicit `embedder={"provider": "sentence-transformer", "config": {"model_name": "BAAI/bge-small-en-v1.5"}}` passed to every Crew |

---

## ⚠️ Known Limitations

- **Cold-start users (~45% of test rows).** When `filtered_user.json` has no
  record for the target user, the Flow forces `stars=3.75` and a neutral
  review. The Calibrator is correctly skipped for these rows. Addressing
  this is the strongest candidate lever for future work.
- **Minimax hedges to 3.75.** Even with calibration, the Prediction Modeler
  occasionally outputs 3.75 when it feels uncertain — a small-model
  regression-to-mean bias that only stronger base models or a validated
  post-hoc calibration step (OpenEvolve in Milestone 2) will fully fix.
- **Nvidia free-tier rate limit.** 10-second inter-row pacing is the
  current safety net; paid tiers should set `INTER_ROW_SLEEP_SEC=0`.

---

## 🚀 Future Work (Milestone 2 / TODO)

- [ ] Evaluate with a larger model (minimax-m2, Sonnet-4.5) and compare
      regression-to-mean rate.
- [ ] Add user-side RAG (top-k similar users for cold-start rows).
- [ ] Integrate the official AgentSocietyChallenge simulator + full training set.
- [ ] OpenEvolve prompt optimisation targeting the `calibrate_user_task_flow`
      and `predict_review_task_flow` prompts.
- [ ] Automated regression tests for the deterministic layer
      (`get_user_extreme_reviews`, `_extract_json`, metric computation).

---

## 🛠️ Tech Stack

- **Agent framework:** [CrewAI](https://crewai.com) (Flows + Crews + Knowledge)
- **LLM:** `minimaxai/minimax-m2.7` via [Nvidia Build API](https://build.nvidia.com/) (free tier)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/) with the instructor's pre-built `chroma.sqlite3`
- **Embeddings:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (384-dim, CPU-friendly)
- **Package manager:** [Astral `uv`](https://docs.astral.sh/uv/)

---

## 📎 Support

For questions or feedback on this implementation, contact
`atsbaha.teweldemedhn@mu.edu.et`. For CrewAI questions in general, visit
the [CrewAI documentation](https://docs.crewai.com/) or the
[official GitHub](https://github.com/crewAIInc/crewAI).
