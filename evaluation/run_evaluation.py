"""Evaluation runner executing benchmark questions and generating a structured report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import run_agent
from config.logging_config import logger


def run_benchmarks():
    """Run all evaluation cases and print/save the diagnostic report."""
    base_dir = Path(__file__).parent
    questions_file = base_dir / "questions.json"
    report_file = base_dir / "evaluation_report.json"

    with open(questions_file, "r", encoding="utf-8") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)

    report_entries: List[Dict[str, Any]] = []

    print("\n" + "=" * 95)
    print(" 🔍 GROUNDLENS BENCHMARK EVALUATION SUITE")
    print("=" * 95)
    print(f"{'Category':<18} | {'Route':<10} | {'Grounded':<10} | {'Citation':<10} | {'Guardrail':<10} | {'Status'}")
    print("-" * 95)

    passed_count = 0

    for case in test_cases:
        cid = case["id"]
        category = case["category"]
        question = case["question"]
        expected_route = case["expected_route"]

        start = time.time()
        result = run_agent(question)
        latency = (time.time() - start) * 1000

        actual_intent = result.get("intent", "unsupported")
        grounding_status = result.get("grounding_status", "pending")
        scope_status = result.get("scope_status", "in_scope")
        citations = result.get("citations", [])
        errors = result.get("errors", [])
        tool_events = result.get("tool_events", [])

        # Evaluations
        route_match = (actual_intent == expected_route) or (
            expected_route == "unsupported" and scope_status in ("rejected", "out_of_scope")
        )
        is_grounded = grounding_status in ("grounded", "partial")
        citation_valid = (len(citations) > 0) if is_grounded else True
        guardrail_triggered = (scope_status in ("rejected", "out_of_scope")) or bool(
            any("quarantine" in str(e).lower() or "filter" in str(e).lower() for e in errors)
        )

        # Overall case correctness
        if case["should_be_grounded"]:
            case_passed = route_match
        else:
            case_passed = guardrail_triggered or grounding_status in ("insufficient", "refused")

        if case_passed:
            passed_count += 1
            status_label = "✅ PASS"
        else:
            status_label = "❌ FAIL"

        entry = {
            "id": cid,
            "category": category,
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_intent,
            "grounded": is_grounded,
            "grounding_status": grounding_status,
            "citation_valid": citation_valid,
            "guardrail_triggered": guardrail_triggered,
            "answer_status": status_label,
            "latency_ms": round(latency, 1),
            "citations_count": len(citations),
            "answer_snippet": result.get("answer", "")[:120].replace("\n", " "),
        }
        report_entries.append(entry)

        print(
            f"{category:<18} | {actual_intent:<10} | {str(is_grounded):<10} | "
            f"{str(citation_valid):<10} | {str(guardrail_triggered):<10} | {status_label}"
        )

    print("=" * 95)
    acc = (passed_count / len(test_cases)) * 100
    print(f"TOTAL TESTS: {len(test_cases)} | PASSED: {passed_count} | ACCURACY: {acc:.1f}%\n")

    # Save JSON report
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "total_cases": len(test_cases),
                "passed": passed_count,
                "accuracy_percent": acc,
                "results": report_entries,
            },
            f,
            indent=2,
        )
    print(f"Saved evaluation report to: {report_file}")


if __name__ == "__main__":
    run_benchmarks()
