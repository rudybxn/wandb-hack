"""optimizer.py - the harness brain + the closed loop.

Pairs with runners.py (Config, RUNNERS, the three LangGraph teams) and scorers.py
(run_generation -> failure profile + one Weave leaderboard row per generation).

One generation = run the whole Q&A bank through one Config, read the aggregate
failure profile, pull exactly ONE lever, emit the next Config. One change per
generation is deliberate: every leaderboard delta stays attributable to a single
decision, which is what makes the loop legible to a judge.

The optimizer does NOT synthesize arbitrary agent graphs. It selects from a discrete
topology menu and flips a few knobs (the contract in runners.Config). The intelligence
is in diagnosing *which* coordination failure dominates and matching it to the right
repair - not in inventing topologies.

  python optimizer.py            # full loop over qa_bank.json (rule-based, reproducible)
  WEAVE_PROJECT=entity/proj python optimizer.py   # also log one Eval per gen to the leaderboard
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

from runners import RUNNERS, Config, Corpus, DEFAULT_MODEL, _parse_json, llm
from scorers import run_generation

EPS = 1e-9

# Topology menu = a small, ordered escalation ladder (matches runners.RUNNERS keys).
TOPOLOGIES = ["single_agent", "supervisor_workers", "debate_verify"]


def _rank(topology: str) -> int:
    return TOPOLOGIES.index(topology)


def _next_topology(topology: str) -> str | None:
    i = _rank(topology)
    return TOPOLOGIES[i + 1] if i + 1 < len(TOPOLOGIES) else None


def _sig(cfg: Config) -> tuple:
    """A config's lever fingerprint - used for the repeat-config guard. (model and
    max_verify_rounds are not levers, so they're excluded.)"""
    return (cfg.topology, cfg.retrieval_k, cfg.max_subquestions,
            cfg.dedup_retrieval, cfg.completeness_gate)


# --------------------------------------------------------------------------- #
# Config -> runnable team. run_generation builds the team itself from the Config;
# this is the public seam for anyone who wants the raw runner (and the CLAUDE.md
# contract). Each RUNNERS[...] returns an @weave.op run(question) -> Rollout.
# --------------------------------------------------------------------------- #
def build_runner(config: Config, corpus: Corpus | None = None):
    return RUNNERS[config.topology](config, corpus or Corpus())


# --------------------------------------------------------------------------- #
# Failure profile - the few aggregate numbers the optimizer reasons over, pulled
# straight from the dict run_generation returns (no Weave-summary parsing needed).
# Dropped-hop signals come from the MULTI-hop slice, where hops actually get dropped.
# --------------------------------------------------------------------------- #
@dataclass
class FailureProfile:
    correctness: float
    multi_complete: float          # multi-hop complete-chain rate (the dropped-hop signal)
    multi_recall: float
    redundant_rate: float
    verifier_override_rate: float
    avg_tokens: float

    @classmethod
    def from_profile(cls, p: dict) -> "FailureProfile":
        multi = p["by_hop"]["multi"]
        if multi["n"] == 0:        # no multi-hop rows in this bank -> fall back to global
            multi = p
        return cls(
            correctness=p["correctness"],
            multi_complete=multi["complete_chain_rate"],
            multi_recall=multi["gold_doc_recall"],
            redundant_rate=p["redundant_rate"],
            verifier_override_rate=p["verifier_override_rate"],
            avg_tokens=p["avg_tokens"],
        )


# --------------------------------------------------------------------------- #
# The rule-based optimizer. Rules are checked in priority order; the FIRST
# unaddressed signal fires and returns (reason, next_config). None == converged.
# The order is TOPOLOGY-AWARE so a lever never no-ops (which would stall the loop):
# the completeness gate only exists on supervisor/debate, so on single_agent the
# fix for dropped hops is to escalate, not to flip a gate the team ignores.
# --------------------------------------------------------------------------- #
DROPPED_HOP_LIMIT = 0.90    # multi-hop complete-chain below this == hops being dropped
REDUNDANCY_LIMIT = 0.15     # repeated-fetch rate above this == duplicated worker effort
CORRECTNESS_TARGET = 0.85
MAX_K = 5


class RuleBasedOptimizer:
    def propose(self, p: FailureProfile, cfg: Config):
        # 1) Dropped hops: the most damaging failure. Repair in cheapest-first order,
        #    but only with a lever the current topology can actually act on.
        if p.multi_complete < DROPPED_HOP_LIMIT:
            if cfg.topology == "single_agent":
                return ("multi-hop hops dropped; a single agent can't decompose -> "
                        "escalate to supervisor_workers", replace(cfg, topology="supervisor_workers"))
            if not cfg.completeness_gate:
                return ("hops still dropped -> enable completeness_gate",
                        replace(cfg, completeness_gate=True))
            if cfg.topology != "debate_verify":
                return ("hops still dropped after gate -> escalate to debate_verify "
                        "(verify-driven extra retrieval)", replace(cfg, topology="debate_verify"))
            if cfg.retrieval_k < MAX_K:
                return ("hops still dropped at top topology -> widen retrieval_k",
                        replace(cfg, retrieval_k=min(MAX_K, cfg.retrieval_k + 2)))

        # 2) Wasted retrieval -> share a fetch cache before touching structure.
        if p.redundant_rate > REDUNDANCY_LIMIT and not cfg.dedup_retrieval:
            return ("redundant retrieval -> enable dedup_retrieval",
                    replace(cfg, dedup_retrieval=True))

        # 3) Still wasteful -> the team is over-decomposing; tighten the cap.
        if p.redundant_rate > REDUNDANCY_LIMIT and cfg.max_subquestions > 2:
            return ("still redundant -> cap decomposition",
                    replace(cfg, max_subquestions=cfg.max_subquestions - 1))

        # 4) Has the evidence but still wrong -> reasoning failure, not retrieval.
        #    Escalate so a verifier can catch bad synthesis.
        if (p.correctness < CORRECTNESS_TARGET and p.multi_complete >= DROPPED_HOP_LIMIT
                and cfg.topology != "debate_verify"):
            return ("evidence present but answers wrong -> escalate to debate_verify",
                    replace(cfg, topology="debate_verify"))

        return None  # converged


# --------------------------------------------------------------------------- #
# The same interface backed by an LLM instead of rules - the "this generalizes to
# an open topology space" seam for the pitch. Demo off RuleBasedOptimizer
# (reproducible); show this exists behind a flag. One validated lever per call.
# --------------------------------------------------------------------------- #
LEVER_SCHEMA = {
    "topology": TOPOLOGIES,
    "retrieval_k": "int 1..8",
    "max_subquestions": "int 1..6",
    "dedup_retrieval": "bool",
    "completeness_gate": "bool",
}


class LLMOptimizer:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def propose(self, p: FailureProfile, cfg: Config):
        sys = (
            "You tune a multi-agent retrieval team to fix coordination failures. Given the failure "
            "profile and current config, choose EXACTLY ONE knob change that best addresses the "
            "dominant failure. Topology may only move up the ladder "
            f"{TOPOLOGIES}. Knobs: {json.dumps(LEVER_SCHEMA)}. "
            'Respond with ONLY JSON: {"reason": "...", "knob": "<name>", "value": <new value>}.'
        )
        user = json.dumps({"profile": p.__dict__, "config": _sig_dict(cfg)})
        text, _ = llm(self.model, sys, user)
        j = _parse_json(text)
        knob, value = j.get("knob"), j.get("value")
        if knob not in LEVER_SCHEMA or not _valid_lever(knob, value, cfg):
            return None
        return (j.get("reason", f"llm: {knob}={value}"), replace(cfg, **{knob: value}))


def _sig_dict(cfg: Config) -> dict:
    return {k: getattr(cfg, k) for k in
            ("topology", "retrieval_k", "max_subquestions", "dedup_retrieval", "completeness_gate")}


def _valid_lever(knob: str, value, cfg: Config) -> bool:
    if knob == "topology":
        return value in TOPOLOGIES and _rank(value) > _rank(cfg.topology)  # up the ladder only
    if knob == "retrieval_k":
        return isinstance(value, int) and 1 <= value <= 8
    if knob == "max_subquestions":
        return isinstance(value, int) and 1 <= value <= 6
    if knob in ("dedup_retrieval", "completeness_gate"):
        return isinstance(value, bool)
    return False


# --------------------------------------------------------------------------- #
# The closed loop. Step budget + monotonic-progress guard + repeat-config guard.
# Progress = improvement in correctness OR in multi-hop complete-chain (so a lever
# that fixes retrieval before correctness catches up isn't punished as "no gain").
# The artifact is the best-by-correctness config.
# --------------------------------------------------------------------------- #
def run_harness(bank: list, budget: int = 6, optimizer=None, weave_project: str | None = None,
                model: str | None = None, baseline: Config | None = None):
    optimizer = optimizer or RuleBasedOptimizer()
    corpus = Corpus()
    cfg = baseline or Config(topology="single_agent", retrieval_k=3, max_subquestions=3)
    if model:
        cfg = replace(cfg, model=model)

    best, best_score, best_chain = None, -1.0, -1.0
    seen = {_sig(cfg)}
    history = []

    for gen in range(budget):
        profile = run_generation(cfg, bank, corpus=corpus, weave_project=weave_project, gen=gen)
        fp = FailureProfile.from_profile(profile)
        history.append((gen, _sig(cfg), fp))
        print(f"gen {gen} [{cfg.topology} k={cfg.retrieval_k} subq={cfg.max_subquestions} "
              f"dedup={cfg.dedup_retrieval} gate={cfg.completeness_gate}]  "
              f"correct={fp.correctness:.2f} multi_recall={fp.multi_recall:.2f} "
              f"multi_chain={fp.multi_complete:.2f} redundant={fp.redundant_rate:.2f} "
              f"tokens={fp.avg_tokens:.0f}")

        # monotonic-progress guard (from gen 1 on): revert to best + halt if nothing improved.
        if gen > 0 and fp.correctness <= best_score + EPS and fp.multi_complete <= best_chain + EPS:
            print("  no progress -> revert to best config, halt")
            cfg = best
            break
        if fp.correctness > best_score:
            best, best_score = cfg, fp.correctness
        best_chain = max(best_chain, fp.multi_complete)

        proposal = optimizer.propose(fp, cfg)
        if proposal is None:
            print("  converged -> halt")
            break
        reason, nxt = proposal
        if _sig(nxt) in seen:                     # repeat-config guard: never loop forever
            print(f"  would repeat a tried config ({reason}) -> halt")
            break
        print(f"  lever: {reason}")
        seen.add(_sig(nxt))
        cfg = nxt

    print(f"\nBEST: [{best.topology} k={best.retrieval_k} subq={best.max_subquestions} "
          f"dedup={best.dedup_retrieval} gate={best.completeness_gate}]  correctness={best_score:.2f}")
    return best, history


if __name__ == "__main__":
    bank = json.load(open("qa_bank.json"))
    run_harness(bank, budget=6, weave_project=os.environ.get("WEAVE_PROJECT"))
