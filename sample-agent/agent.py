"""
Maintenance-ops sample agent — structured-output loop.

Uses OpenAI structured output (response_format) with a Pydantic discriminated
union of all available API requests.  The MaintenanceClient.dispatch() method
handles routing — no manual tool dispatcher needed.

Swap OpenAI for any provider that supports structured output / tool_use.
"""
from __future__ import annotations

import time
from typing import Annotated, List, Union

from annotated_types import MaxLen, MinLen
from openai import OpenAI
from pydantic import BaseModel, Field

from ogchallenge_client import CoreClient, MaintenanceClient, TaskInfo, ApiException
from ogchallenge_client.dtos import (
    # Identity
    Req_WhoAmI,
    # Equipment
    Req_EquipmentList, Req_GetEquipment, Req_UpdateEquipment, Req_EquipmentSearch,
    # Employees
    Req_EmployeeList, Req_GetEmployee, Req_UpdateEmployee, Req_EmployeeSearch,
    # Materials
    Req_MaterialList, Req_MaterialGet, Req_MaterialSearch, Req_MaterialReorder,
    # Notifications
    Req_NotifCreate, Req_NotifGet, Req_NotifSearch, Req_NotifUpdate,
    # Work Orders
    Req_WOList, Req_WOSearch, Req_WOCreate, Req_WOGet, Req_WOUpdate,
    # Operations
    Req_OperationAdd, Req_OperationUpdate, Req_OperationList,
    # Wiki
    Req_WikiList, Req_WikiLoad, Req_WikiUpdate,
    # Respond (final answer)
    Req_Respond,
)


# ── ANSI colours ─────────────────────────────────────────────────────────────

CLI_GREEN = "\x1b[32m"
CLI_RED = "\x1b[31m"
CLI_CYAN = "\x1b[36m"
CLI_YELLOW = "\x1b[33m"
CLI_BLUE = "\x1b[34m"
CLI_CLR = "\x1b[0m"


# ── Structured-output model ─────────────────────────────────────────────────
#
# The LLM returns a NextStep on every iteration.  The `function` field is a
# discriminated union over all Req_* DTOs — the `type` literal on each model
# acts as the discriminator key.

# All available API actions as a discriminated union.
Action = Union[
    Req_WhoAmI,
    # Equipment
    Req_EquipmentList, Req_GetEquipment, Req_UpdateEquipment, Req_EquipmentSearch,
    # Employees
    Req_EmployeeList, Req_GetEmployee, Req_UpdateEmployee, Req_EmployeeSearch,
    # Materials
    Req_MaterialList, Req_MaterialGet, Req_MaterialSearch, Req_MaterialReorder,
    # Notifications
    Req_NotifCreate, Req_NotifGet, Req_NotifSearch, Req_NotifUpdate,
    # Work Orders
    Req_WOList, Req_WOSearch, Req_WOCreate, Req_WOGet, Req_WOUpdate,
    # Operations
    Req_OperationAdd, Req_OperationUpdate, Req_OperationList,
    # Wiki
    Req_WikiList, Req_WikiLoad, Req_WikiUpdate,
    # Final answer
    Req_Respond,
]


class NextStep(BaseModel):
    """Structured output returned by the LLM on each reasoning step."""

    current_state: str = Field(
        ..., description="Brief summary of what you know so far",
    )
    plan: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(
        ..., description="Remaining steps to complete the task (most important first)",
    )
    task_completed: bool = Field(
        False, description="Set to true only when calling respond",
    )
    function: Action = Field(
        ...,
        discriminator="type",
        description="The next API call to execute",
    )


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a maintenance operations agent on NOVA-7, a gas production platform.
You interact with the platform's maintenance management system through API calls.

Your workflow:
1. Start with who_am_i to learn your role and today's date.
2. Read relevant wiki documents to understand policies and SOPs before acting.
3. Investigate the situation using search/get/list endpoints.
4. Take action if your role permits it — or refuse if policy forbids it.
5. Call respond with a clear summary, the correct outcome code, and entity links.

Outcome codes:
- ok_answer              — task completed, clear answer given
- ok_not_found           — requested information doesn't exist
- denied_security        — your role or policy doesn't permit the action
- none_clarification_needed — task is ambiguous, need more info
- none_unsupported       — can't do this with available tools
- error_internal         — unexpected error

Always check your authority in raci.md before performing write actions.
Always consult RAM.md and incidents.md before assigning risk assessments.
Include links to entities you referenced or acted on in your respond call.
"""


# ── Agent loop ───────────────────────────────────────────────────────────────

MAX_STEPS = 30


def run_agent(
    api: CoreClient,
    task: TaskInfo,
    *,
    model: str = "gpt-4.1-2025-04-14",
) -> None:
    """Run the agent for a single task."""

    client = OpenAI()
    maint = api.get_maintenance_client(task)

    print(f"\n{CLI_CYAN}Task {task.num}: {task.spec_id}{CLI_CLR}")
    print(f"  {task.task_text}\n")

    # ── Bootstrap: auto-run essential queries before the LLM starts ──────
    bootstrap_log = _bootstrap(maint)

    # ── Build initial message log ────────────────────────────────────────
    log: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Feed bootstrap results as context
    for label, text in bootstrap_log:
        print(f"  {CLI_GREEN}AUTO {label}{CLI_CLR}: {text[:120]}")
        log.append({"role": "user", "content": f"[{label}]\n{text}"})

    # Task instruction
    log.append({"role": "user", "content": task.task_text})

    # ── Main loop ────────────────────────────────────────────────────────
    for i in range(MAX_STEPS):
        step_id = f"step_{i + 1}"
        print(f"  Step {i + 1}... ", end="", flush=True)

        t0 = time.time()
        resp = client.beta.chat.completions.parse(
            model=model,
            response_format=NextStep,
            messages=log,
            max_completion_tokens=4096,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        step = resp.choices[0].message.parsed
        if step is None:
            print(f"{CLI_RED}LLM returned unparseable response{CLI_CLR}")
            break

        fn = step.function
        fn_type = fn.type  # type: ignore[union-attr]
        print(f"{CLI_CYAN}{fn_type}{CLI_CLR} — {step.plan[0]}  ({elapsed_ms}ms)")

        # Log LLM call
        try:
            api.log_llm(
                task_id=task.task_id,
                completion=step.plan[0],
                model=model,
                duration_sec=(time.time() - t0),
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
                completion_tokens=resp.usage.completion_tokens if resp.usage else None,
            )
        except Exception:
            pass

        # Append assistant message as a tool call (for OpenAI message format)
        log.append({
            "role": "assistant",
            "content": step.plan[0],
            "tool_calls": [{
                "type": "function",
                "id": step_id,
                "function": {
                    "name": type(fn).__name__,
                    "arguments": fn.model_dump_json(exclude_none=True),
                },
            }],
        })

        # ── Dispatch ─────────────────────────────────────────────────────
        try:
            result = maint.dispatch(fn)
            result_text = result.model_dump_json(exclude_none=True)
            print(f"    {CLI_GREEN}→{CLI_CLR} {result_text[:200]}")
        except ApiException as exc:
            result_text = f'{{"error": "{exc.api_error.error}", "code": "{exc.api_error.code}"}}'
            print(f"    {CLI_RED}ERR: {exc.api_error.error}{CLI_CLR}")
        except Exception as exc:
            result_text = f'{{"error": "{exc}"}}'
            print(f"    {CLI_RED}ERR: {exc}{CLI_CLR}")

        log.append({"role": "tool", "content": result_text, "tool_call_id": step_id})

        # ── Check if agent is done ───────────────────────────────────────
        if isinstance(fn, Req_Respond):
            print(f"\n  {CLI_GREEN}Agent responded: {fn.outcome}{CLI_CLR}")
            print(f"  {CLI_BLUE}{fn.message}{CLI_CLR}")
            if fn.links:
                for link in fn.links:
                    print(f"    link: {link.type} → {link.id}")
            break
    else:
        print(f"\n  {CLI_YELLOW}Reached max steps ({MAX_STEPS}) without responding.{CLI_CLR}")


# ── Bootstrap helpers ────────────────────────────────────────────────────────

def _bootstrap(maint: MaintenanceClient) -> list[tuple[str, str]]:
    """Run essential queries before the LLM loop starts.

    Returns a list of (label, text) pairs to inject into the conversation.
    Modify this to change what context the agent sees before reasoning.
    """
    results = []

    # 1. Identity — who am I, what role, what date
    try:
        whoami = maint.who_am_i()
        results.append(("whoami", whoami.model_dump_json()))
    except Exception as exc:
        results.append(("whoami", f"error: {exc}"))

    # 2. Available wiki documents
    try:
        wiki = maint.wiki_list()
        results.append(("wiki_list", ", ".join(wiki.paths)))
    except Exception as exc:
        results.append(("wiki_list", f"error: {exc}"))

    return results
