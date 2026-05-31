"""scorers.py - Weave outcome + coordination scorers, and run_generation().

One generation = run the whole Q&A bank through ONE team config, score every
rollout, and (optionally) log one Weave Evaluation = one leaderboard row.

Two families of scorer:
  OUTCOME      - correctness, via an LLM judge against the gold answer.
  COORDINATION - deterministic, read the Rollout the runner produced:
     gold_doc_recall      retrieved_docs vs gold_docs (the dropped-hop signal)
     redundant_retrieval  repeats in retrieved_docs    (the dedup signal)
     verifier_override    verifier_verdict == "revise" (the verify signal)
     cost                 tokens / llm_calls / steps    (the budget signal)

The metric logic lives in plain helpers (_recall, _redundant, ...) so there is one
source of truth, wrapped two ways: as @weave.op scorers (for the Evaluation/leaderboard)
and called directly in _profile() (for the optimizer's failure profile).

  python scorers.py     # offline smoke of the coordination scorers (no API, no Weave)
run_generation() needs OPENROUTER_API_KEY (judge) and, to log a leaderboard row, a W&B login.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, is_dataclass

from runners import RUNNERS, Config, Corpus, llm

try:
    import weave
    from weave import Evaluation
    op = weave.op
except Exception:  # pragma: no cover
    Evaluation = None
    def op(fn=None, **_):
        return fn if fn else (lambda f: f)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")

JUDGE_SYS = (
    "You grade whether a CANDIDATE answer matches the REFERENCE answer to a question. "
    "Reply with ONLY 'YES' or 'NO'. YES if the candidate conveys the same key fact/value as the "
    "reference (paraphrase, different units, rounding all OK). NO if it is wrong, missing the key "
    "fact, or declines to answer."
)


# ===================== metric helpers (single source of truth) =====================
def _recall(gold_docs, retrieved) -> dict:
    gold, got = set(gold_docs or []), set(retrieved or [])
    if not gold:
        return {"recall": 1.0, "complete_chain": 1.0}
    return {"recall": len(gold & got) / len(gold),
            "complete_chain": 1.0 if gold <= got else 0.0}


def _redundant(retrieved) -> dict:
    n = len(retrieved or [])
    return {"redundant_rate": (n - len(set(retrieved))) / n if n else 0.0}


def _verifier(verdict) -> dict:
    return {"verifier_override": 1.0 if verdict == "revise" else 0.0,
            "has_verifier": 1.0 if verdict is not None else 0.0}


def _cost(tool_calls, steps) -> dict:
    tc = tool_calls or []
    return {"tokens": sum(c.get("total_tokens", 0) for c in tc),
            "llm_calls": sum(1 for c in tc if c.get("type") == "llm"),
            "retrieve_calls": sum(1 for c in tc if c.get("type") == "retrieve"),
            "steps": len(steps or [])}


_judge_cache: dict = {}


def _judge_correct(question: str, gold: str, candidate: str, model: str = JUDGE_MODEL) -> float:
    candidate = candidate or ""
    if candidate.startswith("ERROR") or not candidate.strip():
        return 0.0
    key = (question, gold, candidate)
    if key in _judge_cache:
        return _judge_cache[key]
    text, _ = llm(model, JUDGE_SYS, f"QUESTION: {question}\nREFERENCE: {gold}\nCANDIDATE: {candidate}",
                  max_tokens=3)
    val = 1.0 if text.strip().upper().startswith("Y") else 0.0
    _judge_cache[key] = val
    return val


# ===================== Weave scorers (wrap the helpers) =====================
@op
def correctness_scorer(question: str, answer: str, output: dict) -> dict:
    return {"correct": _judge_correct(question, answer, output.get("answer", ""))}


@op
def gold_doc_recall_scorer(gold_docs: list, output: dict) -> dict:
    return _recall(gold_docs, output.get("retrieved_docs", []))


@op
def redundant_retrieval_scorer(output: dict) -> dict:
    return _redundant(output.get("retrieved_docs", []))


@op
def verifier_override_scorer(output: dict) -> dict:
    return _verifier(output.get("verifier_verdict"))


@op
def cost_scorer(output: dict) -> dict:
    return _cost(output.get("tool_calls", []), output.get("agent_steps", []))


SCORERS = [correctness_scorer, gold_doc_recall_scorer, redundant_retrieval_scorer,
           verifier_override_scorer, cost_scorer]


# ===================== aggregation -> failure profile =====================
def _profile(config: Config, bank: list, rollouts: dict, judge: bool = True) -> dict:
    per = []
    for row in bank:
        ro = rollouts[row["id"]]
        r = _recall(row["gold_docs"], ro.get("retrieved_docs", []))
        per.append({
            "id": row["id"], "hop": row["hop_type"],
            "correct": _judge_correct(row["question"], row["answer"], ro.get("answer", "")) if judge else 0.0,
            "recall": r["recall"], "complete_chain": r["complete_chain"],
            "redundant_rate": _redundant(ro.get("retrieved_docs", []))["redundant_rate"],
            "verifier_override": _verifier(ro.get("verifier_verdict"))["verifier_override"],
            **_cost(ro.get("tool_calls", []), ro.get("agent_steps", [])),
        })

    def agg(rows, key):
        return round(sum(x[key] for x in rows) / len(rows), 3) if rows else 0.0

    def metrics(rows):
        return {"n": len(rows),
                "correctness": agg(rows, "correct"),
                "gold_doc_recall": agg(rows, "recall"),
                "complete_chain_rate": agg(rows, "complete_chain"),
                "redundant_rate": agg(rows, "redundant_rate"),
                "verifier_override_rate": agg(rows, "verifier_override"),
                "avg_tokens": agg(rows, "tokens"),
                "avg_llm_calls": agg(rows, "llm_calls")}

    profile = metrics(per)
    profile["config"] = asdict(config) if is_dataclass(config) else dict(config)
    profile["by_hop"] = {h: metrics([x for x in per if x["hop"] == h]) for h in ("single", "multi")}
    profile["rows"] = per
    return profile


# ===================== one generation =====================
def run_generation(config: Config, bank: list, corpus: Corpus | None = None,
                   weave_project: str | None = None, gen: int = 0, judge: bool = True) -> dict:
    """Run the whole bank through one team config; return the failure profile.
    If weave_project is given (and W&B is logged in), also log one Evaluation = one
    leaderboard row, reusing the already-computed rollouts (no extra team calls)."""
    corpus = corpus or Corpus()
    runner = RUNNERS[config.topology](config, corpus)
    rollouts = {row["id"]: runner(row["question"]).to_dict() for row in bank}

    profile = _profile(config, bank, rollouts, judge=judge)

    if weave_project and Evaluation is not None:
        weave.init(weave_project)

        @op
        def predict(id: str) -> dict:           # cached: Evaluation does not re-run the team
            return rollouts[id]

        name = f"gen{gen}-{config.topology}-k{config.retrieval_k}"
        evaluation = Evaluation(name=name, dataset=bank, scorers=SCORERS)
        asyncio.run(evaluation.evaluate(predict))  # judge memoised -> cheap

    return profile


# ===================== offline smoke (coordination scorers) =====================
if __name__ == "__main__":
    # mock rollouts; no API, no Weave - exercises the deterministic coordination scorers
    dropped = {"retrieved_docs": ["a.md", "b.md", "a.md"], "verifier_verdict": None,
               "tool_calls": [{"type": "retrieve"}, {"type": "llm", "total_tokens": 100}],
               "agent_steps": [{}, {}], "answer": "partial"}
    complete = {"retrieved_docs": ["a.md", "b.md", "c.md"], "verifier_verdict": "revise",
                "tool_calls": [{"type": "retrieve"}, {"type": "retrieve"},
                               {"type": "llm", "total_tokens": 250}],
                "agent_steps": [{}, {}, {}, {}], "answer": "full"}
    gold = ["a.md", "b.md", "c.md"]

    r1 = _recall(gold, dropped["retrieved_docs"])
    assert abs(r1["recall"] - 2 / 3) < 1e-9 and r1["complete_chain"] == 0.0, r1
    assert abs(_redundant(dropped["retrieved_docs"])["redundant_rate"] - 1 / 3) < 1e-9
    assert _verifier(dropped["verifier_verdict"])["verifier_override"] == 0.0
    assert _cost(dropped["tool_calls"], dropped["agent_steps"]) == \
        {"tokens": 100, "llm_calls": 1, "retrieve_calls": 1, "steps": 2}

    r2 = _recall(gold, complete["retrieved_docs"])
    assert r2 == {"recall": 1.0, "complete_chain": 1.0}, r2
    assert _redundant(complete["retrieved_docs"])["redundant_rate"] == 0.0
    assert _verifier(complete["verifier_verdict"])["verifier_override"] == 1.0
    assert _cost(complete["tool_calls"], complete["agent_steps"])["tokens"] == 250

    # the @weave.op wrappers must return the same as the helpers
    assert gold_doc_recall_scorer(gold_docs=gold, output=dropped) == r1
    assert redundant_retrieval_scorer(output=dropped)["redundant_rate"] != 0.0
    assert cost_scorer(output=complete)["retrieve_calls"] == 2

    print("offline coordination-scorer smoke: PASS")
    print("  dropped-hop rollout :", {**r1, **_redundant(dropped["retrieved_docs"])})
    print("  complete rollout    :", {**r2, **_verifier(complete["verifier_verdict"])})
