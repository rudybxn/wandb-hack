# CLAUDE.md — Self-Improving Multi-Agent Orchestration Harness

## What this is

A hackathon project for a **Multi-Agent Orchestration** build day. We are building a
**harness that hardens a multi-agent team by re-orchestrating it.** It runs a team of
agents on a task, observes their *inter-agent coordination* through Weave, diagnoses
which coordination failure dominates, rewrites the team's orchestration, and re-runs —
improving generation over generation. The artifact is a *better-configured team*, not
an answer.

We are targeting two prizes: **Most Sophisticated Harness** (the harness, not an app,
is the star) and **Best Use of Weave** (Weave is the spine: tracing, scoring, and the
leaderboard, not a bolt-on).

## The core mechanic (the loop)

One **generation** = run the entire Q&A bank through ONE team configuration, then:

1. The team runner returns a structured **Rollout** per question (what it did, not just
   the answer): `{answer, retrieved_docs, verifier_verdict, agent_steps, tool_calls}`.
2. Weave scores every run — **outcome** scorers (LLM-judge correctness) and **coordination**
   scorers (gold-doc recall, redundant-retrieval rate, verifier override, cost) that parse
   the Rollout against `gold_docs` labels.
3. The **optimizer** reads the aggregate failure profile and pulls **exactly one lever**
   (enable a completeness gate → enable retrieval dedup → cap decomposition → escalate
   topology). One change per generation so every leaderboard delta is attributable.
4. The next generation runs the new config. **Loop guards:** a step budget, a
   monotonic-progress check (revert + halt if no gain), and a repeat-config halt.

Each generation is one row on the **Weave leaderboard**. The demo is the score climbing,
with a side-by-side trace drill-down showing a question that was wrong in gen N and right
in gen N+1 after the orchestration changed.

## Target task

Root-cause analysis over a **synthetic microservices operations wiki** ("Meridian", a
fictional rideshare platform) in `documents/`. Questions are single-hop (one file) or
multi-hop (chain facts across files). Multi-hop questions are built so the *symptom* and
the *root cause* live in DIFFERENT files with low lexical overlap — so naive retrieval
drops a hop and a coordinating team catches it. The corpus is fully synthetic so no model
can answer from pretraining; retrieval is forced.

## Files

- `documents/` — the corpus. Service docs `svc-*.md` (one uniform schema), reference
  catalogs `catalog-datastores.md` / `catalog-third-party.md`, `conventions-timeouts-retries.md`,
  and `00-system-overview.md` (the dependency graph + schema spec). Two planted chains:
  Chain A (gateway→booking→dispatch→driver-location, root cause in the datastore catalog)
  and Chain B (payments inbound budget vs. NorthPay timeout, across two files).
- `scorers.py` — Weave outcome + coordination scorers and `run_generation()` (one
  Evaluation per generation). Coordination scorers are deterministic and read the Rollout.
- `optimizer.py` — `Config`, the `Topology` menu, `build_runner()`, the rule-based
  optimizer, the LLM-optimizer seam, and `run_harness()` (the driver loop + guards).
- `runners.py` — the three team topologies (single-agent, supervisor/workers,
  debate-verify) as LangGraph graphs, plus a BM25 `Corpus`. Returns Rollouts. Wire these
  into the three stubs in `optimizer.py`.
- `qa_bank.json` — **TODO.** ~24 Q&A rows, each `{question, answer, gold_docs, hop_type}`.
  Generate from the corpus (see below). `gold_docs` is mandatory — the coordination
  scorers compare retrieval against it.

## Conventions / hard rules

- **The config knobs are the contract.** Runners MUST honor `retrieval_k`,
  `max_subquestions`, `dedup_retrieval`, `completeness_gate`. If a runner ignores a knob,
  the matching optimizer lever won't move the metric and the loop stalls. Thin agents that
  respect the knobs beat clever agents that don't.
- **Topology is a discrete menu, not arbitrary graphs.** The optimizer selects + tunes;
  it does not synthesize graphs. The LLM-optimizer is the "generalizes" pitch seam — demo
  off the rule-based one (reproducible).
- **One change per generation.** Keep deltas attributable.
- **Keep the loop guards.** Monotonic-progress + repeat-config halts are cheap and are
  exactly the loop-control competence judges check for.
- **LLM + judge calls go through OpenRouter; Weave traces them regardless of provider.**
  Weave runs on a free W&B account — no LLM credits consumed by Weave itself.
- **Eligibility:** all code built at the event; fresh work, not extensions of prior
  projects. The corpus is synthetic and authored here.

## Generating the Q&A bank

Feed `documents/*.md` to a model and require this JSON per question:
`id, question, answer (short/exact), hop_type ("single"|"multi"),
gold_docs (ORDERED filenames in hop order), reasoning (one line)`.
12 single-hop + 12 multi-hop. Multi-hop must place symptom and root cause in different
files. Vet: confirm each `gold_docs` chain actually yields the answer; spot-check leakage
by asking the multi-hop questions with no corpus attached.

## How to run

```
pip install langgraph rank_bm25 openai weave
export OPENROUTER_API_KEY=...        # LLM + judge
export DOCS_DIR=documents
python scorers.py                    # offline smoke test of coordination scorers
python optimizer.py                  # full loop (after qa_bank.json + runner wire-in)
```

## Status

- DONE: corpus seed (11 docs, 2 chains), scorers, optimizer + loop, three runners.
- TODO: generate + vet `qa_bank.json`; wire `runners` into `optimizer` stubs; add the
  remaining `documents/` service docs if more retrieval distractors are needed; build the
  rehearsed demo script (the win condition).