# Self-Improving Multi-Agent Orchestration Harness

> A harness that **hardens a multi-agent team by re-orchestrating it.** It runs a team of
> agents on a task, observes their *inter-agent coordination* through **Weave**, diagnoses
> which coordination failure dominates, rewrites the team's orchestration **one lever at a
> time**, and re-runs — improving generation over generation.
>
> **The artifact is a better-configured team, not an answer.**

Built for a Multi-Agent Orchestration hackathon, targeting **Most Sophisticated Harness** and
**Best Use of Weave**.

---

## TL;DR

On a root-cause-analysis benchmark over a synthetic microservices wiki, the harness lifts a
team from a naive single-agent RAG baseline to a coordinated team **without a human touching
the config**:

| Gen | Team config | Correctness | Multi-hop chain | Avg tokens | Lever pulled |
|----:|-------------|:-----------:|:---------------:|:----------:|--------------|
| 0 | `single_agent` | 0.65 | 0.36 | 1.2k | — (baseline floor) |
| 1 | `supervisor_workers` | 0.73 | 0.64 | 3.9k | hops dropped → decompose |
| 2 | `supervisor_workers + gate` | 0.89 | 0.86 | 13k | hops still dropped → completeness gate |
| 3 | `debate_verify + gate` | **0.92** | **0.93** | 35k | evidence present but wrong → escalate |
| 4 | `debate_verify + gate + dedup` | 0.92 | 0.86 ↓ | 11k | **rejected by progress guard → halt** |

Each row is **one Weave Evaluation = one leaderboard row**, and each step changes **exactly one
knob**, so every delta is attributable. The progress guard reverts gen 4 (dedup regressed the
multi-hop chain for no correctness gain) and halts on the best config.

**It's adaptive, not hardcoded:** re-run on the *single-hop-only* slice and the harness lands on
`single_agent` instead — it tries escalating, sees it doesn't help, and reverts. Same harness,
different failure distribution, different winning team.

---

## The core mechanic (the loop)

One **generation** = run the whole Q&A bank through ONE team config, then diagnose and change
one thing.


1. The team runner returns a **Rollout** per question — *what it did*, not just the answer:
   `{answer, retrieved_docs, verifier_verdict, agent_steps, tool_calls}`.
2. **Weave scores** every run: an LLM-judge **outcome** scorer plus deterministic
   **coordination** scorers that parse the Rollout against `gold_docs` labels.
3. The **optimizer** reads the aggregate failure profile and pulls **one lever** (enable the
   completeness gate → enable retrieval dedup → cap decomposition → escalate topology).
4. The next generation runs the new config. **Loop guards:** step budget, monotonic-progress
   check (revert + halt if no gain), and a repeat-config halt.

---

## What gets scored, and what the optimizer acts on

The metric logic lives in plain helpers (one source of truth), wrapped two ways: as `@weave.op`
scorers for the leaderboard **and** called in-process to build the optimizer's failure profile —
so the optimizer acts on the **same numbers** Weave logs.

| Scorer | Family | Reads from Rollout | Signal |
|--------|--------|--------------------|--------|
| `correctness_scorer` | **outcome** | `answer` | LLM-judge correct vs gold |
| `gold_doc_recall_scorer` | coordination | `retrieved_docs` vs `gold_docs` | recall + complete-chain (**dropped hops**) |
| `redundant_retrieval_scorer` | coordination | `retrieved_docs` | duplicate fetches (**dedup**) |
| `verifier_override_scorer` | coordination | `verifier_verdict` | verifier "revise" rate (**verify**) |
| `cost_scorer` | coordination | `tool_calls`, `agent_steps` | tokens / calls / steps (**budget**) |

Metrics are split **by hop type** (single vs multi), so dropped-hop signals come from the
multi-hop slice where hops actually get dropped. The rule-based optimizer fires in priority
order on three thresholds: multi-hop chain `< 0.90` → fix dropped hops; redundancy `> 0.15` →
dedup/cap; correctness `< 0.85` with evidence present → escalate to a verifier.

---

## The team topologies (a discrete escalation ladder)

The optimizer **selects + tunes** from a fixed menu; it does not synthesize arbitrary graphs.
Each topology is a LangGraph graph that honors the config knobs (the contract):

1. **`single_agent`** — one retrieve → one answer. The deliberate floor; drops the low-overlap
   second hop on multi-hop questions.
2. **`supervisor_workers`** — decompose → workers retrieve per sub-question → **completeness
   gate** → aggregate.
3. **`debate_verify`** — two solvers + a verifier with a bounded revise loop.

**Config knobs (the contract):** `retrieval_k`, `max_subquestions`, `dedup_retrieval`,
`completeness_gate`. The **completeness gate** is the key mechanism — it audits retrieved
evidence for *dependency IDs* (e.g. `DS-GEO-3`) referenced but whose own doc is missing, and
fetches them. That's what recovers the dropped hop.

---

## The task: root-cause analysis over a synthetic ops wiki

The corpus (`documents/`) is **"Meridian"**, a fictional rideshare platform: 11 markdown docs —
service docs, datastore/third-party catalogs, conventions, and a system overview with the
dependency graph. The Q&A bank (`qa_bank.json`) has **26 questions (12 single-hop + 14
multi-hop)**, each with `gold_docs` (the ordered file chain that yields the answer).

Multi-hop questions are built so the **symptom** and the **root cause** live in *different*
files with low lexical overlap — naive retrieval drops a hop, a coordinating team catches it.
The corpus is fully synthetic, so no model can answer from pretraining; **retrieval is forced.**

---

## Setup & run

Requires Python 3.11+ and a [W&B](https://wandb.ai) account (for Weave + W&B Inference).

```bash
pip install langgraph rank_bm25 openai weave wandb matplotlib pandas

# Create .env in the repo root (KEY = value, spaces OK). Never commit this.
cat > .env <<'EOF'
WANDB_API_KEY = <your-wandb-api-key>
PROJECT       = <entity>/<project>      # used for W&B Inference attribution + Weave logging
EOF
```

> **Models:** LLM + judge calls go through **W&B Inference** (OpenAI-compatible), keyed on
> `WANDB_API_KEY`. Defaults are **instruct (non-reasoning)** models
> (`meta-llama/Llama-3.3-70B-Instruct`) — a reasoning model would burn the judge's tiny
> `max_tokens` budget on hidden reasoning and return nothing. Override with `RUNNER_MODEL` /
> `JUDGE_MODEL`. (An `OPENROUTER_API_KEY` path is supported as a fallback.)

```bash
python optimizer.py        # the whole system: full loop, logs each generation to Weave
python run_single_hop.py   # adaptivity check on the single-hop slice
python scorers.py          # offline smoke of the coordination scorers (no API, no Weave)
python runners.py          # one question through all three topologies (needs API)

# Visualize results (no API — pure replay of the saved trajectories):
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

---

## Weave is the spine

- **Tracing:** every agent step and LLM call is a `@weave.op`, traced regardless of provider.
- **Scoring:** the outcome + coordination scorers are Weave `Scorer`s.
- **Leaderboard:** each generation logs one `Evaluation` (named `gen{N}-{topology}-k{k}-sub{n}+gate…`)
  → one comparable leaderboard row; the climbing score is the demo.
- **Drill-down:** per-question traces are tagged with `{gen, topology, gate, dedup}`, so you can
  pull up a question that was wrong in gen N and right in gen N+1 and diff the agent behavior.

Weave runs on a free W&B account; the harness consumes no extra LLM credits to log.

---

## Repo layout

```
optimizer.py          # Config menu, rule-based + LLM optimizer, run_harness (the loop + guards)
scorers.py            # outcome + coordination scorers, run_generation (1 Evaluation / gen)
runners.py            # 3 LangGraph topologies + BM25 Corpus; returns Rollouts; env loading
qa_bank.json          # 26 Q&A rows (12 single + 14 multi), each with gold_docs
documents/            # the synthetic "Meridian" corpus 
```

---

## Design principles

- **One change per generation** — every leaderboard delta stays attributable.
- **The config knobs are the contract** — runners must honor every knob, or the matching lever
  no-ops and the loop stalls.
- **Topology is a discrete menu, not arbitrary graphs** — the optimizer selects + tunes.
- **Keep the loop guards** — monotonic-progress + repeat-config halts are cheap loop-control.
- **Deterministic by default** — temp 0, rule-based optimizer;
