"""runners.py - BM25 Corpus + three team topologies as LangGraph graphs.

Each topology is built from a Config and a Corpus and exposed as a callable
`run(question) -> Rollout`. The Rollout records WHAT THE TEAM DID (not just the
answer), so the coordination scorers in scorers.py can grade inter-agent behaviour.

The config knobs are the contract (CLAUDE.md). Every topology that HAS the relevant
mechanism honours them, so the matching optimizer lever moves the metric:
  retrieval_k        -> all three (docs returned per search)
  max_subquestions   -> supervisor_workers, debate_verify (decomposition cap)
  dedup_retrieval    -> supervisor_workers, debate_verify (drop already-seen docs;
                        repeats in retrieved_docs are what the redundancy scorer reads)
  completeness_gate  -> supervisor_workers, debate_verify (audit retrieved evidence for
                        referenced-but-unfetched dependency IDs and go get them -> this
                        is what raises gold-doc recall on the dropped-hop questions)

Topology escalation (single -> supervisor -> debate) is the last-resort lever.
single_agent is the deliberate floor: one retrieve + one answer, no gate, so it
drops the low-overlap second hop on multi-hop questions.

Config and Rollout live here (the leaf module) so optimizer.py can import them
without a circular dependency.

Smoke test (needs OPENROUTER_API_KEY in ./env):  .venv/bin/python runners.py
"""
from __future__ import annotations

import json
import operator
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi

# --- optional Weave tracing: no-op decorator when weave is absent/uninitialised ---
try:
    import weave
    op = weave.op
except Exception:  # pragma: no cover
    def op(fn=None, **_):
        return fn if fn else (lambda f: f)

DOCS_DIR = os.environ.get("DOCS_DIR", "documents")
DEFAULT_MODEL = os.environ.get("RUNNER_MODEL", "openai/gpt-4o-mini")
GATE_TERM_CAP = 5  # max dependency IDs the completeness gate will chase per call


# ===================== LLM (OpenRouter) =====================
def _load_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and os.path.exists("env"):
        kv = dict(re.findall(r"^([A-Z_]+)=(.*)$", open("env").read(), re.M))
        k = kv.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        raise RuntimeError("OPENROUTER_API_KEY not set and not found in ./env")
    return k


_client = None


def _client_singleton():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_load_key())
    return _client


@op
def llm(model: str, system: str, user: str, temperature: float = 0.0,
        max_tokens: int = 700) -> tuple[str, dict]:
    """One chat completion. Returns (text, usage). Deterministic by default; retries thrice."""
    last = None
    for _ in range(3):
        try:
            r = _client_singleton().chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return (r.choices[0].message.content or "").strip(), _usage(r.usage)
        except Exception as e:
            last = e
    return f"ERROR: {last}", _usage(None)


def _usage(u) -> dict:
    g = lambda k: int(getattr(u, k, 0) or 0) if u else 0
    return {"prompt_tokens": g("prompt_tokens"),
            "completion_tokens": g("completion_tokens"),
            "total_tokens": g("total_tokens")}


def _llm_call(role: str, usage: dict) -> dict:
    return {"type": "llm", "role": role, **usage}


# ===================== Corpus (BM25) =====================
class Corpus:
    """BM25 index over documents/*.md. Searches return filenames (the unit gold_docs
    is labelled in, so the scorers line up directly)."""

    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs = {f: open(os.path.join(docs_dir, f)).read()
                     for f in sorted(os.listdir(docs_dir)) if f.endswith(".md")}
        self.names = list(self.docs)
        self._tokens = [self._tok(self.docs[n]) for n in self.names]
        self.bm25 = BM25Okapi(self._tokens)

    @staticmethod
    def _tok(s: str) -> list[str]:
        return re.findall(r"[a-z0-9\-]+", s.lower())

    def search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.names[i] for i in order[:max(0, k)]]

    def get(self, name: str) -> str:
        return self.docs[name]

    def context(self, docs: list[str]) -> str:
        """Concatenate doc texts in order (repeats included, so skipping dedup
        genuinely costs more tokens)."""
        return "\n\n".join(f"=== {d} ===\n{self.get(d)}" for d in docs)


# ===================== Contracts =====================
@dataclass
class Config:
    topology: str = "single_agent"          # single_agent | supervisor_workers | debate_verify
    retrieval_k: int = 3
    max_subquestions: int = 3
    dedup_retrieval: bool = False
    completeness_gate: bool = False
    max_verify_rounds: int = 2               # inner loop guard for debate_verify
    model: str = DEFAULT_MODEL


@dataclass
class Rollout:
    question: str
    answer: str
    retrieved_docs: list[str]                # ordered, repeats iff dedup off (redundancy signal)
    verifier_verdict: Optional[str]          # accept | revise | None (no verifier)
    agent_steps: list[dict]                  # structured per-agent actions
    tool_calls: list[dict]                   # retrieve + llm calls (with token usage)
    sub_questions: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ===================== Shared graph state =====================
class RunState(TypedDict):
    question: str
    sub_questions: list
    context_docs: Annotated[list, operator.add]
    tool_calls: Annotated[list, operator.add]
    steps: Annotated[list, operator.add]
    answer: str
    cand_a: str
    cand_b: str
    verifier_verdict: Optional[str]
    rounds: int
    missing: str


def _initial(question: str) -> RunState:
    return {"question": question, "sub_questions": [], "context_docs": [],
            "tool_calls": [], "steps": [], "answer": "", "cand_a": "", "cand_b": "",
            "verifier_verdict": None, "rounds": 0, "missing": ""}


# ===================== Prompts =====================
ANSWER_SYS = (
    "You are an SRE assistant doing root-cause analysis on the Meridian platform. "
    "Answer the question using ONLY the provided documents. Name the specific service, "
    "datastore, and third-party IDs and the numeric limits that matter. A good root-cause "
    "answer identifies the underlying resource or configuration limit, not just the symptom. "
    "Be concise (2-4 sentences). If the documents are insufficient, say exactly what is missing."
)
DECOMPOSE_SYS = (
    "You plan retrieval for a question about the Meridian platform. Break it into at most {n} "
    "focused sub-questions, each aimed at a DIFFERENT document or hop: trace one dependency edge "
    "per sub-question, and include a sub-question that looks up the deep config (connection pool, "
    "timeout, rate limit) of any datastore or third-party on the path. If the question is "
    "single-hop, return exactly one. Respond with ONLY a JSON array of strings."
)
GATE_SYS = (
    "You audit retrieval completeness for a Meridian root-cause question. Service docs reference "
    "datastores and third parties BY ID, with the deep config living in a separate catalog. "
    "Given the documents retrieved so far, list any datastore IDs (e.g. DS-GEO-3), third-party "
    "provider names, or downstream service names that are referenced AS DEPENDENCIES but whose own "
    "dedicated document is NOT present in what was retrieved - these are dropped hops. Respond with "
    "ONLY a JSON array of short search terms to look them up; return [] if retrieval is complete."
)
VERIFY_SYS = (
    "You are a verifier comparing two candidate answers against the retrieved evidence for a "
    "Meridian root-cause question. A correct answer names the underlying resource/config limit "
    "(not just the symptom) and is grounded in the documents. Respond with ONLY JSON: "
    '{"verdict": "accept" | "revise", "answer": "<the best final answer>"}. '
    'Use "revise" only when neither candidate is adequately supported and more retrieval is needed.'
)


def _parse_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(0)) if str(x).strip()]
        except Exception:
            pass
    return []


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def _retrieve(corpus: Corpus, query: str, k: int, seen: set, dedup: bool) -> tuple[list, dict]:
    """BM25 search; when dedup, drop docs already in `seen` before adding to context."""
    hits = corpus.search(query, k)
    added = [d for d in hits if d not in seen] if dedup else list(hits)
    call = {"type": "retrieve", "query": query, "k": k, "returned": hits, "added": added}
    return added, call


# ===================== Reusable nodes =====================
def _decompose_node(config: Config, corpus: Corpus):
    def decompose(state: RunState):
        text, usage = llm(config.model, DECOMPOSE_SYS.format(n=config.max_subquestions), state["question"])
        subs = (_parse_list(text) or [state["question"]])[:config.max_subquestions]
        return {"sub_questions": subs, "tool_calls": [_llm_call("supervisor", usage)],
                "steps": [{"agent": "supervisor", "action": "decompose", "sub_questions": subs}]}
    return decompose


def _workers_node(config: Config, corpus: Corpus):
    def workers(state: RunState):
        seen, added_all, calls, steps = set(state["context_docs"]), [], [], []
        for sq in state["sub_questions"]:
            added, call = _retrieve(corpus, sq, config.retrieval_k, seen, config.dedup_retrieval)
            seen.update(call["returned"])
            added_all += added
            calls.append(call)
            steps.append({"agent": "worker", "sub_question": sq, "docs": added})
        return {"context_docs": added_all, "tool_calls": calls, "steps": steps}
    return workers


def _gate_node(config: Config, corpus: Corpus):
    """Completeness gate: find dependency IDs referenced in retrieved evidence whose own
    doc is missing, then fetch them. The mechanism that recovers the dropped hop."""
    def gate(state: RunState):
        if not config.completeness_gate:
            return {}
        present = state["context_docs"]
        text, usage = llm(config.model, GATE_SYS,
                          f"Retrieved files: {sorted(set(present))}\n\n{corpus.context(present)}")
        terms = _parse_list(text)[:GATE_TERM_CAP]
        seen, added_all, calls, steps = set(present), [], [_llm_call("gate", usage)], \
            [{"agent": "gate", "action": "audit", "missing_terms": terms}]
        for t in terms:
            added, call = _retrieve(corpus, t, config.retrieval_k, seen, config.dedup_retrieval)
            seen.update(call["returned"])
            added_all += added
            calls.append(call)
            steps.append({"agent": "gate", "action": "resolve", "term": t, "docs": added})
        return {"context_docs": added_all, "tool_calls": calls, "steps": steps}
    return gate


def _answer_node(config: Config, corpus: Corpus, role: str = "solver"):
    def answer(state: RunState):
        ctx = corpus.context(state["context_docs"])
        text, usage = llm(config.model, ANSWER_SYS, f"{ctx}\n\nQUESTION: {state['question']}")
        return {"answer": text, "tool_calls": [_llm_call(role, usage)],
                "steps": [{"agent": role, "action": "answer"}]}
    return answer


# ===================== Topology 1: single agent =====================
def single_agent_runner(config: Config, corpus: Corpus) -> Callable[[str], Rollout]:
    def retrieve(state: RunState):
        added, call = _retrieve(corpus, state["question"], config.retrieval_k, set(), config.dedup_retrieval)
        return {"context_docs": added, "tool_calls": [call],
                "steps": [{"agent": "solo", "action": "retrieve", "docs": added}]}

    g = StateGraph(RunState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", _answer_node(config, corpus, "solo"))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return _wrap(g.compile(), config)


# ===================== Topology 2: supervisor / workers =====================
def supervisor_workers_runner(config: Config, corpus: Corpus) -> Callable[[str], Rollout]:
    def aggregate(state: RunState):
        ctx = corpus.context(state["context_docs"])
        user = (f"{ctx}\n\nQUESTION: {state['question']}\n\n"
                f"Sub-questions explored: {state['sub_questions']}")
        text, usage = llm(config.model, ANSWER_SYS, user)
        return {"answer": text, "tool_calls": [_llm_call("supervisor", usage)],
                "steps": [{"agent": "supervisor", "action": "aggregate"}]}

    g = StateGraph(RunState)
    g.add_node("decompose", _decompose_node(config, corpus))
    g.add_node("workers", _workers_node(config, corpus))
    g.add_node("gate", _gate_node(config, corpus))
    g.add_node("aggregate", aggregate)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "workers")
    g.add_edge("workers", "gate")
    g.add_edge("gate", "aggregate")
    g.add_edge("aggregate", END)
    return _wrap(g.compile(), config)


# ===================== Topology 3: debate / verify =====================
def debate_verify_runner(config: Config, corpus: Corpus) -> Callable[[str], Rollout]:
    def solve_a(state: RunState):
        ctx = corpus.context(state["context_docs"])
        text, usage = llm(config.model, ANSWER_SYS + " You are solver A.", f"{ctx}\n\nQUESTION: {state['question']}")
        return {"cand_a": text, "tool_calls": [_llm_call("solverA", usage)],
                "steps": [{"agent": "solverA", "action": "propose"}]}

    def solve_b(state: RunState):
        ctx = corpus.context(state["context_docs"])
        text, usage = llm(config.model, ANSWER_SYS + " You are solver B; argue for the most specific root cause.",
                          f"{ctx}\n\nQUESTION: {state['question']}")
        return {"cand_b": text, "tool_calls": [_llm_call("solverB", usage)],
                "steps": [{"agent": "solverB", "action": "propose"}]}

    def verify(state: RunState):
        ctx = corpus.context(state["context_docs"])
        user = (f"QUESTION: {state['question']}\n\nCANDIDATE A: {state['cand_a']}\n\n"
                f"CANDIDATE B: {state['cand_b']}\n\nEVIDENCE:\n{ctx}")
        text, usage = llm(config.model, VERIFY_SYS, user)
        v = _parse_json(text)
        verdict = "revise" if str(v.get("verdict", "accept")).lower().startswith("rev") else "accept"
        answer = str(v.get("answer", "")) or state["cand_a"] or state["cand_b"]
        return {"verifier_verdict": verdict, "answer": answer, "rounds": state["rounds"] + 1,
                "tool_calls": [_llm_call("verifier", usage)],
                "steps": [{"agent": "verifier", "action": "verify", "verdict": verdict}]}

    def route(state: RunState):
        # bounded revise loop: only useful when the gate can fetch more on the next pass
        if (state.get("verifier_verdict") == "revise"
                and config.completeness_gate and state["rounds"] < config.max_verify_rounds):
            return "gate"
        return END

    g = StateGraph(RunState)
    g.add_node("decompose", _decompose_node(config, corpus))
    g.add_node("workers", _workers_node(config, corpus))
    g.add_node("gate", _gate_node(config, corpus))
    g.add_node("solveA", solve_a)
    g.add_node("solveB", solve_b)
    g.add_node("verify", verify)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "workers")
    g.add_edge("workers", "gate")
    g.add_edge("gate", "solveA")
    g.add_edge("solveA", "solveB")
    g.add_edge("solveB", "verify")
    g.add_conditional_edges("verify", route, {"gate": "gate", END: END})
    return _wrap(g.compile(), config)


# ===================== Wrapper =====================
def _wrap(app, config: Config) -> Callable[[str], Rollout]:
    @op
    def run(question: str) -> Rollout:
        final = app.invoke(_initial(question), {"recursion_limit": 50})
        return Rollout(
            question=question,
            answer=final.get("answer", ""),
            retrieved_docs=final.get("context_docs", []),
            verifier_verdict=final.get("verifier_verdict"),
            agent_steps=final.get("steps", []),
            tool_calls=final.get("tool_calls", []),
            sub_questions=final.get("sub_questions", []),
            config=asdict(config),
        )
    return run


RUNNERS = {
    "single_agent": single_agent_runner,
    "supervisor_workers": supervisor_workers_runner,
    "debate_verify": debate_verify_runner,
}


# ===================== Smoke test =====================
if __name__ == "__main__":
    corpus = Corpus(DOCS_DIR)
    row = next(r for r in json.load(open("qa_bank.json")) if r["id"] == "M01")
    gold = set(row["gold_docs"])
    print(f"Q ({row['id']}): {row['question']}\nGOLD ({len(gold)}): {sorted(gold)}\n")

    configs = {
        "single_agent": Config("single_agent", retrieval_k=3),
        "supervisor_workers": Config("supervisor_workers", retrieval_k=3, max_subquestions=4,
                                     dedup_retrieval=True, completeness_gate=True),
        "debate_verify": Config("debate_verify", retrieval_k=3, max_subquestions=4,
                                dedup_retrieval=True, completeness_gate=True),
    }
    for name, cfg in configs.items():
        r = RUNNERS[name](cfg, corpus)(row["question"])
        got = r.retrieved_docs
        recall = len(set(got) & gold) / len(gold)
        n_llm = sum(c["type"] == "llm" for c in r.tool_calls)
        n_ret = sum(c["type"] == "retrieve" for c in r.tool_calls)
        toks = sum(c.get("total_tokens", 0) for c in r.tool_calls)
        redundant = len(got) - len(set(got))
        print(f"--- {name} ---")
        print(f"  recall {recall:.0%} ({len(set(got)&gold)}/{len(gold)})  llm={n_llm} retrieve={n_ret} "
              f"tokens={toks} redundant_docs={redundant} verdict={r.verifier_verdict}")
        print(f"  retrieved: {got}")
        print(f"  answer: {r.answer[:200]}\n")
