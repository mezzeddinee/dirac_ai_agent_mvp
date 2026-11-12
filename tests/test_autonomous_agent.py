from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from mcp import Client

from agent import run_cycle, validate_final_decision
from mcp_server import mcp


def tool_call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
    )


def tool_response(*calls: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(output=list(calls), output_text="")


def final_response(action: str, ticket: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        output=[SimpleNamespace(type="message")],
        output_text=json.dumps({"action": action, "ticket": ticket}),
    )


class ScriptedResponses:
    def __init__(self, responses: list[SimpleNamespace]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected model request")
        return self.responses.pop(0)


def fake_openai(*responses: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(responses=ScriptedResponses(list(responses)))


class FinalDecisionValidationTests(unittest.TestCase):
    def test_ticket_requires_nonempty_string(self) -> None:
        with self.assertRaises(ValueError):
            validate_final_decision('{"action":"ticket","ticket":null}')

    def test_no_action_requires_null_ticket(self) -> None:
        with self.assertRaises(ValueError):
            validate_final_decision(
                '{"action":"no_action","ticket":"unexpected"}'
            )

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_final_decision(
                '{"action":"no_action","ticket":null,"reason":"healthy"}'
            )


class AutonomousCycleTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, model):
        async with Client(mcp) as client:
            return await run_cycle(
                openai_client=model,
                mcp_client=client,
                model="test-model",
            )

    async def test_evidence_tool_then_ticket(self) -> None:
        ticket = (
            "Subject: SiteDirector service requires investigation\n\n"
            "The service-status evidence reports a confirmed failure."
        )
        model = fake_openai(
            tool_response(
                tool_call(
                    "get_service_status",
                    {"service": "WorkloadManagement/SiteDirector"},
                    "status-1",
                )
            ),
            final_response("ticket", ticket),
        )

        decision = await self._run(model)

        self.assertEqual(decision, {"action": "ticket", "ticket": ticket})
        self.assertEqual(len(model.responses.requests), 2)
        self.assertEqual(model.responses.requests[0]["tool_choice"], "auto")
        self.assertFalse(model.responses.requests[0]["parallel_tool_calls"])
        self.assertEqual(
            model.responses.requests[0]["text"]["format"]["type"],
            "json_schema",
        )

    async def test_direct_no_action_after_automatic_log_fetch(self) -> None:
        model = fake_openai(final_response("no_action", None))

        decision = await self._run(model)

        self.assertEqual(decision, {"action": "no_action", "ticket": None})
        initial_message = json.loads(
            model.responses.requests[0]["input"][0]["content"]
        )
        self.assertEqual(initial_message["event"], "hourly_site_director_check")
        self.assertGreater(
            initial_message["site_director_logs"]["result"]["returned_lines"],
            0,
        )

    async def test_multiple_tool_calls_become_no_action(self) -> None:
        model = fake_openai(
            tool_response(
                tool_call(
                    "get_service_status",
                    {"service": "WorkloadManagement/SiteDirector"},
                    "status-1",
                ),
                tool_call(
                    "get_service_status",
                    {"service": "WorkloadManagement/JobMonitoring"},
                    "status-2",
                ),
            )
        )

        decision = await self._run(model)

        self.assertEqual(decision, {"action": "no_action", "ticket": None})

    async def test_plain_text_becomes_no_action(self) -> None:
        model = fake_openai(
            SimpleNamespace(
                output=[SimpleNamespace(type="message")],
                output_text="Everything looks healthy.",
            )
        )

        decision = await self._run(model)

        self.assertEqual(decision, {"action": "no_action", "ticket": None})

    async def test_evidence_loop_continues_until_final_decision(self) -> None:
        model = fake_openai(
            *(
                tool_response(
                    tool_call(
                        "get_logs",
                        {
                            "source": "job_monitoring",
                            "lines": 20,
                            "contains": "ERROR",
                        },
                        f"logs-{index}",
                    )
                )
                for index in range(1, 7)
            ),
            final_response("no_action", None),
        )

        decision = await self._run(model)

        self.assertEqual(decision, {"action": "no_action", "ticket": None})
        self.assertEqual(len(model.responses.requests), 7)
        self.assertTrue(
            all(
                request["tool_choice"] == "auto"
                for request in model.responses.requests
            )
        )

    async def test_unavailable_tool_becomes_no_action(self) -> None:
        model = fake_openai(
            tool_response(tool_call("restart_service", {}, "unknown-1"))
        )

        decision = await self._run(model)

        self.assertEqual(decision, {"action": "no_action", "ticket": None})


if __name__ == "__main__":
    unittest.main()
