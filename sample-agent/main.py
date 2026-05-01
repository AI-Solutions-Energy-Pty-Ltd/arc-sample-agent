"""
OGChallenge sample agent — harness entry point.

Usage:
    # Run all tasks in a full session
    python main.py

    # Run a single task by spec_id (development mode)
    python main.py --spec notification_raise

Environment variables (see .env.example):
    OGC_BASE_URL   — OGChallenge server URL (default: https://ai-agents-challenge.digital)
    OGC_API_KEY    — API key for the platform
    OPENAI_API_KEY  — OpenAI API key
    MODEL_ID        — model to use (default: gpt-4.1-2025-04-14)
"""
from __future__ import annotations

import argparse
import os
import textwrap

from dotenv import load_dotenv

load_dotenv()

from ogchallenge_client import CoreClient

from agent import run_agent

MODEL_ID = os.getenv("MODEL_ID", "gpt-4.1-2025-04-14")

CLI_RED = "\x1b[31m"
CLI_GREEN = "\x1b[32m"
CLI_BLUE = "\x1b[34m"
CLI_CLR = "\x1b[0m"


def make_client() -> CoreClient:
    base_url = os.getenv("OGC_BASE_URL", "https://ai-agents-challenge.digital")
    api_key = os.getenv("OGC_API_KEY", "")
    auth_token = os.getenv("OGC_AUTH_TOKEN", "")
    return CoreClient(
        base_url=base_url,
        api_key=api_key or None,
        auth_token=auth_token or None,
    )


def run_session(api: CoreClient, workspace: str) -> None:
    """Start a full session and run all tasks."""
    print(f"Starting session (benchmark=maintenance-ops, workspace={workspace!r})...")
    session = api.start_session(
        benchmark="maintenance-ops",
        workspace=workspace,
        name=f"sample-agent ({MODEL_ID})",
        architecture="OpenAI structured-output agent",
    )
    print(f"Session ID: {session.session_id}  tasks: {session.task_count}\n")

    status = api.session_status(session.session_id)
    scores = []

    for task_info in status.tasks:
        print("=" * 60)
        api.start_task(task_info)

        try:
            run_agent(api, task_info, model=MODEL_ID)
        except Exception as exc:
            print(f"  {CLI_RED}ERROR: {exc}{CLI_CLR}")

        result = api.complete_task(task_info)
        if result.eval:
            score = result.eval.score
            scores.append((task_info.spec_id, score))
            style = CLI_GREEN if score >= 0.8 else CLI_RED
            explain = textwrap.indent(result.eval.logs, "    ")
            print(f"\n  {style}SCORE: {score:.2f}{CLI_CLR}\n{explain}")

    print("\n" + "=" * 60)
    submitted = api.submit_session(session.session_id)
    print(f"Session submitted — status: {submitted.status}  score: {submitted.score:.2f}")

    if scores:
        print()
        for spec_id, score in scores:
            style = CLI_GREEN if score >= 0.8 else CLI_RED
            print(f"  {spec_id}: {style}{score:.2f}{CLI_CLR}")
        total = sum(s for _, s in scores) / len(scores) * 100
        print(f"\n  FINAL: {total:.1f}%")


def run_single_task(api: CoreClient, spec_id: str) -> None:
    """Start a standalone task by spec_id (for development/testing)."""
    print(f"Starting standalone task: spec={spec_id!r}\n")
    task_info = api.start_new_task(benchmark="maintenance-ops", spec_id=spec_id)

    try:
        run_agent(api, task_info, model=MODEL_ID)
    except Exception as exc:
        print(f"{CLI_RED}ERROR: {exc}{CLI_CLR}")

    result = api.complete_task(task_info)
    if result.eval:
        style = CLI_GREEN if result.eval.score >= 0.8 else CLI_RED
        explain = textwrap.indent(result.eval.logs, "    ")
        print(f"\n{style}SCORE: {result.eval.score:.2f}{CLI_CLR}\n{explain}")
    else:
        print(f"\nStatus: {result.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OGChallenge sample agent")
    parser.add_argument("--spec", help="Run a single task by spec_id (skips session)")
    parser.add_argument("--workspace", default="dev", help="Session workspace tag (default: dev)")
    args = parser.parse_args()

    api = make_client()

    if args.spec:
        run_single_task(api, args.spec)
    else:
        run_session(api, workspace=args.workspace)


if __name__ == "__main__":
    main()
