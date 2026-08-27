"""
RecoverIQ Phase 4 -- Gemini tool-calling agent.

Public API:
    run_agent_decision(payment_id, api_key=None) -> dict
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

import google.generativeai as genai  # noqa: E402
from google.api_core.exceptions import ResourceExhausted  # noqa: E402
from google.generativeai.types import FunctionDeclaration, Tool  # noqa: E402

try:
    from prompts import SYSTEM_PROMPT  # noqa: E402
    from tools import TOOL_FUNCTIONS, VALID_INTERVENTION_LIST  # noqa: E402
except ImportError:
    from agent.prompts import SYSTEM_PROMPT  # noqa: E402
    from agent.tools import TOOL_FUNCTIONS, VALID_INTERVENTION_LIST  # noqa: E402

MAX_TOOL_ITERATIONS = 8
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
VALID_DECISIONS = set(VALID_INTERVENTION_LIST)

PAYMENT_ID_PARAMETERS = {
    "type": "object",
    "properties": {
        "payment_id": {
            "type": "string",
            "description": "Failed payment id, e.g. PAY-0000001",
        }
    },
    "required": ["payment_id"],
}

INTERVENTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "payment_id": {
            "type": "string",
            "description": "Failed payment id, e.g. PAY-0000001",
        },
        "intervention": {
            "type": "string",
            "enum": list(VALID_INTERVENTION_LIST),
            "description": "Recovery intervention to score",
        },
    },
    "required": ["payment_id", "intervention"],
}

GEMINI_FUNCTION_DECLARATIONS = [
    FunctionDeclaration(
        name="get_payment",
        description="Load the failed payment event by payment_id.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
    FunctionDeclaration(
        name="get_customer_history",
        description="Load this customer's other failed-payment events by customer_id.",
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer id from get_payment, e.g. CUST-00001",
                }
            },
            "required": ["customer_id"],
        },
    ),
    FunctionDeclaration(
        name="get_failure_details",
        description="Load failure_reason, method, amount, and related failure fields.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
    FunctionDeclaration(
        name="calculate_recovery_probability",
        description=(
            "Return P(recovery) for one intervention via the Phase 3 EV calculator. "
            "Do not guess this number yourself."
        ),
        parameters=INTERVENTION_PARAMETERS,
    ),
    FunctionDeclaration(
        name="calculate_expected_value_tool",
        description=(
            "Return expected_value, probability_used, and cost for one intervention. "
            "Call this for each candidate before choosing."
        ),
        parameters=INTERVENTION_PARAMETERS,
    ),
    FunctionDeclaration(
        name="retry_payment",
        description="Would retry the charge. Do NOT call — this agent only decides.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
    FunctionDeclaration(
        name="generate_payment_link",
        description="Would send a new payment link. Do NOT call — this agent only decides.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
    FunctionDeclaration(
        name="send_notification",
        description="Would notify the customer. Do NOT call — this agent only decides.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
    FunctionDeclaration(
        name="escalate_to_human",
        description="Would escalate to a human. Do NOT call — this agent only decides.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
    FunctionDeclaration(
        name="stop_recovery",
        description="Would stop recovery. Do NOT call — this agent only decides.",
        parameters=PAYMENT_ID_PARAMETERS,
    ),
]

GEMINI_TOOLS = [Tool(function_declarations=GEMINI_FUNCTION_DECLARATIONS)]

_TOOL_CONFIG_NONE = {"function_calling_config": {"mode": "NONE"}}

_JSON_RETRY_PROMPT = (
    "Your previous reply was not valid JSON. Output ONLY this JSON object, "
    "nothing else (no markdown, no commentary): "
    '{"payment_id": "...", "decision": "retry|payment_link|notification|escalate|stop", '
    '"delay_hours": <number or null>, "reason": "...", "expected_value": <number>}'
)


def _dispatch_tool(name: str, tool_input: dict) -> dict:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**tool_input)
    except TypeError as exc:
        return {"error": f"Bad tool arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 — surface tool failures to the model
        return {"error": str(exc)}


def _proto_to_python(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {k: _proto_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_proto_to_python(v) for v in value]
    if hasattr(value, "items"):
        try:
            return {k: _proto_to_python(v) for k, v in value.items()}
        except (TypeError, ValueError):
            pass
    return value


def _function_calls(response) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return calls
    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if not fc or not getattr(fc, "name", None):
            continue
        raw_args = dict(fc.args) if fc.args else {}
        calls.append((fc.name, {k: _proto_to_python(v) for k, v in raw_args.items()}))
    return calls


def _extract_text(response) -> str:
    try:
        text = response.text
        if text:
            return text.strip()
    except (ValueError, AttributeError):
        pass
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    parts = []
    for part in getattr(candidates[0].content, "parts", None) or []:
        text = getattr(part, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _function_response_parts(calls: list[tuple[str, dict]]) -> list:
    parts = []
    for name, args in calls:
        result = _dispatch_tool(name, args)
        parts.append(
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=name,
                    response={"result": json.dumps(result)},
                )
            )
        )
    return parts


def _parse_decision_json(text: str, payment_id: str) -> dict:
    if not text:
        raise ValueError("Model returned no text to parse as a decision")

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object in model output: {text[:500]}")
        cleaned = cleaned[start : end + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Decision JSON must be an object")

    decision = data.get("decision")
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid decision {decision!r}; expected one of {sorted(VALID_DECISIONS)}")

    delay = data.get("delay_hours", None)
    if delay is not None:
        delay = float(delay)

    expected_value = data.get("expected_value")
    if expected_value is None:
        raise ValueError("Decision JSON missing expected_value")

    return {
        "payment_id": str(data.get("payment_id") or payment_id),
        "decision": decision,
        "delay_hours": delay,
        "reason": str(data.get("reason", "")),
        "expected_value": float(expected_value),
    }


def _chat_send(chat, parts, **kwargs):
    """Send a chat turn; retry once after the free-tier 429 retry_delay."""
    try:
        return chat.send_message(parts, **kwargs)
    except ResourceExhausted as exc:
        delay = 45.0
        retry_delay = getattr(exc, "retry_delay", None)
        seconds = getattr(retry_delay, "seconds", None) if retry_delay is not None else None
        if seconds:
            delay = float(seconds) + 2.0
        time.sleep(delay)
        return chat.send_message(parts, **kwargs)


def run_agent_decision(payment_id: str, api_key: str | None = None) -> dict:
    """
    Run the tool-calling loop for one failed payment and return the decision dict.

    If api_key is None, reads GEMINI_API_KEY from the environment.
    """
    key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env or pass api_key=...")

    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        system_instruction=SYSTEM_PROMPT,
        tools=GEMINI_TOOLS,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
        ),
    )
    chat = model.start_chat()

    user_prompt = (
        f"Decide the best recovery action for payment_id={payment_id}. "
        "Follow the gather-first protocol, score candidates with "
        "calculate_expected_value_tool, then output only the final JSON."
    )

    pending_parts: list | str = user_prompt
    response = None
    for iteration in range(MAX_TOOL_ITERATIONS):
        send_kwargs = {}
        if iteration == MAX_TOOL_ITERATIONS - 1:
            send_kwargs["tool_config"] = _TOOL_CONFIG_NONE

        response = _chat_send(chat, pending_parts, **send_kwargs)
        calls = _function_calls(response)
        if not calls:
            break
        pending_parts = _function_response_parts(calls)

    if response is None:
        raise RuntimeError("Agent loop produced no model response")

    text = _extract_text(response)
    try:
        return _parse_decision_json(text, payment_id)
    except (ValueError, json.JSONDecodeError):
        retry = _chat_send(chat, _JSON_RETRY_PROMPT, tool_config=_TOOL_CONFIG_NONE)
        return _parse_decision_json(_extract_text(retry), payment_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RecoverIQ agent on one payment")
    parser.add_argument("payment_id")
    args = parser.parse_args()
    decision = run_agent_decision(args.payment_id)
    print(json.dumps(decision, indent=2))
