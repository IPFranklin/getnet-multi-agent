"""Offline evaluation harness for the multi-agent system.

Runs a labelled dataset through the system and computes:
  * routing accuracy           — final agent == expected agent
  * tool recall                — expected tools ⊆ tools actually called
  * grounding rate             — answers that should cite sources do
  * keyword coverage           — each "any-of" keyword group appears in the answer
  * safety                     — forbidden strings (e.g. raw PAN, "Correction") never appear
  * multi-turn cases           — "turns" are sent first in the same session, then the scored message
  * latency p50 / p95, error rate
  * (optional --judge) LLM-as-judge 1-5 scores for relevance and faithfulness to sources

Usage:
  python -m evals.run_evals                          # in-process, uses .env (fake or real LLM)
  python -m evals.run_evals --url http://localhost:8000
  python -m evals.run_evals --judge --min-route-acc 0.9   # CI quality gate (non-zero exit on failure)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
JUDGE_PROMPT = """You are grading a customer-support answer for Getnet (payments company).
Question: {question}
Answer: {answer}
Sources listed to the customer (ref = the [n] used in the answer): {sources}
Earlier turns of this conversation (the assistant may rely on them):
{history}
Evidence the assistant actually received from its tools (retrieved passages, account data, API results):
{evidence}
Score 1-5 for RELEVANCE (does it answer the question?) and FAITHFULNESS (every factual claim is supported by
the evidence above; citations [n] point to the right source; 5 = fully grounded). Judge only against the
evidence shown. Reply ONLY with JSON: {{"relevance": n, "faithfulness": n, "comment": "..."}}"""


def load_dataset(path: Path, tags: set[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not tags or tags & set(r.get("tags", []))]


def _session(case: dict, run_id: str) -> str:
    return f"eval-{run_id}-{case['id']}"


async def run_case_inprocess(orchestrator, case: dict, run_id: str) -> dict:
    # Multi-turn cases: earlier turns ("turns") are sent first in the same session, then the scored message.
    history = []
    for turn in case.get("turns", []):
        prev = await orchestrator.handle(turn, case["user_id"], session_id=_session(case, run_id))
        history.append({"user": turn, "assistant": prev.get("response", "")})
    out = await orchestrator.handle(case["message"], case["user_id"], session_id=_session(case, run_id))
    return {**out, "_history": history}


async def run_case_http(client: httpx.AsyncClient, url: str, case: dict, run_id: str) -> dict:
    async def send(msg: str) -> dict:
        resp = await client.post(f"{url}/chat", json={"message": msg, "user_id": case["user_id"],
                                                      "session_id": _session(case, run_id)})
        resp.raise_for_status()
        return resp.json()

    history = []
    for turn in case.get("turns", []):
        prev = await send(turn)
        history.append({"user": turn, "assistant": prev.get("response", "")})
    return {**await send(case["message"]), "_history": history}


def score_case(case: dict, out: dict) -> dict:
    answer = out.get("response", "")
    used = {t["tool"] for t in out.get("tools_used", [])}
    groups = case.get("keywords", [])
    hits = [any(k.lower() in answer.lower() for k in g) for g in groups]
    return {
        "id": case["id"],
        "route_ok": out.get("agent") == case["expected_agent"],
        "tools_ok": set(case.get("expected_tools", [])) <= used,
        "grounding_ok": (bool(out.get("sources")) == case["expect_sources"]) if case["expect_sources"] else True,
        "keyword_coverage": (sum(hits) / len(hits)) if hits else 1.0,
        "safety_ok": not any(f in answer for f in case.get("forbidden", [])),
        "error": bool(out.get("errors")),
        "agent": out.get("agent"),
        "tools": sorted(used),
        "latency_ms": out.get("latency_ms", 0.0),
        "answer": answer,
        "sources": out.get("sources", []),
        "evidence": out.get("evidence", []),
        "history": out.get("_history", []),
    }


async def judge(model, case: dict, result: dict) -> dict:
    from langchain_core.messages import HumanMessage

    evidence = "\n\n".join(f"<{e['tool']}>\n{e['output']}" for e in result.get("evidence", [])) or "(none returned)"
    prompt = JUDGE_PROMPT.format(question=case["message"], answer=result["answer"][:3000],
                                 sources=json.dumps(result["sources"], ensure_ascii=False)[:2000],
                                 evidence=evidence[:12000],
                                 history=json.dumps(result.get("history", []), ensure_ascii=False)[:4000] or "(none)")
    msg = await model.ainvoke([HumanMessage(content=prompt)])
    text = msg.content if isinstance(msg.content, str) else "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        return {"relevance": None, "faithfulness": None, "comment": "unparseable judge output"}


def pct(xs: list[bool]) -> float:
    return round(100 * sum(xs) / len(xs), 1) if xs else 0.0


def summarise(results: list[dict]) -> dict:
    lat = sorted(r["latency_ms"] for r in results)
    summary = {
        "cases": len(results),
        "route_accuracy_pct": pct([r["route_ok"] for r in results]),
        "tool_recall_pct": pct([r["tools_ok"] for r in results]),
        "grounding_pct": pct([r["grounding_ok"] for r in results]),
        "keyword_coverage_pct": round(100 * statistics.mean(r["keyword_coverage"] for r in results), 1),
        "safety_pct": pct([r["safety_ok"] for r in results]),
        "error_rate_pct": pct([r["error"] for r in results]),
        "latency_p50_ms": lat[len(lat) // 2] if lat else 0,
        "latency_p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0,
    }
    judged = [r["judge"] for r in results if r.get("judge") and r["judge"].get("relevance")]
    if judged:
        summary["judge_relevance_avg"] = round(statistics.mean(j["relevance"] for j in judged), 2)
        summary["judge_faithfulness_avg"] = round(statistics.mean(j["faithfulness"] for j in judged), 2)
    return summary


def write_report(summary: dict, results: list[dict], provider: str) -> Path:
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"report-{stamp}.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [f"# Evaluation report — {stamp} (provider: {provider})", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "| case | agent | route | tools | grounded | keywords | safe | latency ms |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        ok = lambda b: "✅" if b else "❌"  # noqa: E731
        lines.append(f"| {r['id']} | {r['agent']} | {ok(r['route_ok'])} | {ok(r['tools_ok'])} | {ok(r['grounding_ok'])} | "
                     f"{r['keyword_coverage']:.0%} | {ok(r['safety_ok'])} | {r['latency_ms']} |")
    path = out_dir / f"report-{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset.jsonl"))
    ap.add_argument("--url", help="Evaluate a running API instead of in-process")
    ap.add_argument("--tags", help="Comma-separated tag filter")
    ap.add_argument("--judge", action="store_true", help="Add LLM-as-judge scores (needs a real LLM provider)")
    ap.add_argument("--min-route-acc", type=float, default=0.0, help="Fail (exit 1) below this routing accuracy (0-1)")
    ap.add_argument("--min-keyword", type=float, default=0.0, help="Fail below this keyword coverage (0-1)")
    args = ap.parse_args()

    cases = load_dataset(Path(args.dataset), set(args.tags.split(",")) if args.tags else None)
    from app.config import get_settings

    settings = get_settings()
    results = []
    t0 = time.perf_counter()
    run_id = datetime.now().strftime("%H%M%S")  # fresh sessions on every run (memory is per session)
    if args.url:
        async with httpx.AsyncClient(timeout=120) as client:
            outs = [await run_case_http(client, args.url, c, run_id) for c in cases]
    else:
        from app.container import build_container
        from app.graph import Orchestrator
        from app.observability import configure_logging

        configure_logging("WARNING")
        settings.expose_trace = True  # the judge needs the tool evidence
        orchestrator = Orchestrator(build_container(settings))
        outs = [await run_case_inprocess(orchestrator, c, run_id) for c in cases]

    judge_model = None
    if args.judge and settings.llm_provider != "fake":
        from app.llm import build_chat_model

        judge_model = build_chat_model(settings, "agent")
    for case, out in zip(cases, outs, strict=True):
        r = score_case(case, out)
        if judge_model and r["agent"] in {"knowledge", "support"}:
            r["judge"] = await judge(judge_model, case, r)
        results.append(r)

    summary = summarise(results)
    summary["wall_time_s"] = round(time.perf_counter() - t0, 1)
    report = write_report(summary, results, settings.llm_provider)
    print(json.dumps(summary, indent=2))
    print(f"report: {report}")
    for r in results:
        if not (r["route_ok"] and r["tools_ok"] and r["grounding_ok"] and r["safety_ok"]):
            print(f"  FAIL {r['id']}: agent={r['agent']} tools={r['tools']}")
    failed = summary["route_accuracy_pct"] < args.min_route_acc * 100 or summary["keyword_coverage_pct"] < args.min_keyword * 100
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
