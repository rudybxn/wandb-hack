"""main.py - entry point for the self-improving multi-agent orchestration harness.

Modes:
  python main.py                          # full harness loop (rule-based optimizer)
  python main.py --mode harness           # same
  python main.py --mode harness --llm-opt # use LLM optimizer instead
  python main.py --mode smoke             # offline scorer smoke test (no API)
  python main.py --mode runners           # single-question smoke across all three topologies
  python main.py --mode single-gen        # one generation only (no loop)

Environment:
  OPENROUTER_API_KEY   required for runner + judge LLM calls
  WEAVE_PROJECT        e.g. entity/project -- enables Weave leaderboard logging
  DOCS_DIR             default: documents
  RUNNER_MODEL         default: openai/gpt-4o-mini
  JUDGE_MODEL          default: openai/gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _require_key():
    import re
    def _env_file():
        for f in (".env", "env"):
            if os.path.exists(f):
                return dict(re.findall(r"^([A-Z_]+)=(.*)$", open(f).read(), re.M))
        return {}
    kv = _env_file()
    wandb = os.environ.get("WANDB_API_KEY") or kv.get("WANDB_API_KEY", "").strip()
    openrouter = os.environ.get("OPENROUTER_API_KEY") or kv.get("OPENROUTER_API_KEY", "").strip()
    if not wandb and not openrouter:
        sys.exit("ERROR: Set WANDB_API_KEY (W&B Inference) or OPENROUTER_API_KEY in env or .env")


def _load_bank(path: str = "qa_bank.json") -> list:
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Generate it first (see CLAUDE.md).")
    bank = json.load(open(path))
    if not bank:
        sys.exit(f"ERROR: {path} is empty.")
    return bank


# ------------------------------------------------------------------ #
# Modes
# ------------------------------------------------------------------ #

def mode_smoke():
    """Offline coordination-scorer smoke test. No API key needed."""
    import scorers  # noqa: F401 -- triggers __main__ block via direct import of helpers
    from scorers import (
        _recall, _redundant, _verifier, _cost,
        gold_doc_recall_scorer, redundant_retrieval_scorer, cost_scorer,
    )
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
    assert gold_doc_recall_scorer(gold_docs=gold, output=dropped) == r1
    assert redundant_retrieval_scorer(output=dropped)["redundant_rate"] != 0.0
    assert cost_scorer(output=complete)["retrieve_calls"] == 2

    print("offline coordination-scorer smoke: PASS")
    print("  dropped-hop:", {**r1, **_redundant(dropped["retrieved_docs"])})
    print("  complete:   ", {**r2, **_verifier(complete["verifier_verdict"])})


def mode_runners(question: str | None = None):
    """Run one question through all three topologies. Uses first qa_bank.json row if no question."""
    _require_key()
    bank = _load_bank()
    from runners import RUNNERS, Config, Corpus, DOCS_DIR

    corpus = Corpus(DOCS_DIR)
    row = bank[0] if question is None else next(
        (r for r in bank if r["question"] == question), bank[0])
    gold = set(row["gold_docs"])
    print(f"Q ({row['id']}): {row['question']}\nGOLD ({len(gold)}): {sorted(gold)}\n")

    configs = {
        "single_agent":      Config("single_agent", retrieval_k=3),
        "supervisor_workers": Config("supervisor_workers", retrieval_k=3, max_subquestions=4,
                                    dedup_retrieval=True, completeness_gate=True),
        "debate_verify":     Config("debate_verify", retrieval_k=3, max_subquestions=4,
                                    dedup_retrieval=True, completeness_gate=True),
    }
    for name, cfg in configs.items():
        r = RUNNERS[name](cfg, corpus)(row["question"])
        got = r.retrieved_docs
        recall = len(set(got) & gold) / len(gold) if gold else 1.0
        n_llm = sum(c["type"] == "llm" for c in r.tool_calls)
        n_ret = sum(c["type"] == "retrieve" for c in r.tool_calls)
        toks = sum(c.get("total_tokens", 0) for c in r.tool_calls)
        redundant = len(got) - len(set(got))
        print(f"--- {name} ---")
        print(f"  recall {recall:.0%} ({len(set(got) & gold)}/{len(gold)})  "
              f"llm={n_llm} retrieve={n_ret} tokens={toks} "
              f"redundant_docs={redundant} verdict={r.verifier_verdict}")
        print(f"  retrieved: {got}")
        print(f"  answer: {r.answer[:200]}\n")


def mode_single_gen(topology: str = "single_agent", weave_project: str | None = None):
    """One generation only — useful for checking a specific config without the full loop."""
    _require_key()
    bank = _load_bank()
    from runners import Config
    from scorers import run_generation

    cfg = Config(topology=topology, retrieval_k=3, max_subquestions=3)
    print(f"Running single generation: {topology} over {len(bank)} questions …")
    profile = run_generation(cfg, bank, weave_project=weave_project, gen=0)
    print(f"\nResults:")
    print(f"  correctness:       {profile['correctness']:.2f}")
    print(f"  gold_doc_recall:   {profile['gold_doc_recall']:.2f}")
    print(f"  complete_chain:    {profile['complete_chain_rate']:.2f}")
    print(f"  redundant_rate:    {profile['redundant_rate']:.2f}")
    print(f"  avg_tokens:        {profile['avg_tokens']:.0f}")
    by_hop = profile.get("by_hop", {})
    for h in ("single", "multi"):
        s = by_hop.get(h, {})
        if s.get("n", 0):
            print(f"  [{h}] n={s['n']} correct={s['correctness']:.2f} "
                  f"recall={s['gold_doc_recall']:.2f} chain={s['complete_chain_rate']:.2f}")


def mode_harness(budget: int = 6, llm_opt: bool = False, weave_project: str | None = None,
                 model: str | None = None):
    """Full self-improving loop. Logs one Weave leaderboard row per generation when
    WEAVE_PROJECT is set."""
    _require_key()
    bank = _load_bank()
    from optimizer import LLMOptimizer, RuleBasedOptimizer, run_harness

    optimizer = LLMOptimizer(model=model) if llm_opt else RuleBasedOptimizer()
    opt_name = "LLMOptimizer" if llm_opt else "RuleBasedOptimizer"
    print(f"Harness: {len(bank)} questions, budget={budget}, optimizer={opt_name}")
    if weave_project:
        print(f"  Weave leaderboard: {weave_project}")
    print()

    best, history = run_harness(
        bank, budget=budget, optimizer=optimizer,
        weave_project=weave_project, model=model,
    )
    print(f"\nDone. {len(history)} generation(s) run.")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent orchestration harness (Meridian RCA)")
    parser.add_argument("--mode", choices=["harness", "smoke", "runners", "single-gen"],
                        default="harness",
                        help="harness=full loop (default), smoke=offline scorer test, "
                             "runners=three-topology single-question test, "
                             "single-gen=one generation only")
    parser.add_argument("--budget", type=int, default=6,
                        help="generation budget for harness mode (default: 6)")
    parser.add_argument("--llm-opt", action="store_true",
                        help="use LLM optimizer instead of rule-based (harness mode)")
    parser.add_argument("--topology", default="single_agent",
                        choices=["single_agent", "supervisor_workers", "debate_verify"],
                        help="starting topology for single-gen mode")
    parser.add_argument("--question", default=None,
                        help="question string for runners mode (defaults to first qa_bank row)")
    parser.add_argument("--model", default=None,
                        help="override RUNNER_MODEL (e.g. openai/gpt-4o)")
    parser.add_argument("--weave-project", default=os.environ.get("WEAVE_PROJECT"),
                        help="W&B project for Weave leaderboard (entity/project). "
                             "Defaults to $WEAVE_PROJECT env var.")
    args = parser.parse_args()

    if args.mode == "smoke":
        mode_smoke()
    elif args.mode == "runners":
        mode_runners(question=args.question)
    elif args.mode == "single-gen":
        mode_single_gen(topology=args.topology, weave_project=args.weave_project)
    else:
        mode_harness(budget=args.budget, llm_opt=args.llm_opt,
                     weave_project=args.weave_project, model=args.model)


if __name__ == "__main__":
    main()
