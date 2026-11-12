from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from agent import discover_openai_tools, validate_tool_arguments
from mcp_server import mcp, read_recent_logs, read_service_status


class ToolImplementationTests(unittest.TestCase):
    def test_get_logs_returns_latest_lines(self) -> None:
        result = read_recent_logs("site_director", lines=2, contains="")
        self.assertEqual(result["returned_lines"], 2)
        self.assertIn("pilots submitted", result["entries"][0])
        self.assertIn("Cycle was successful", result["entries"][1])

    def test_get_logs_filters_case_insensitively(self) -> None:
        result = read_recent_logs(
            "site_director", lines=10, contains="SUBMITTED"
        )
        self.assertEqual(result["returned_lines"], 2)

    def test_get_logs_rejects_unlisted_source(self) -> None:
        with self.assertRaises(ValueError):
            read_recent_logs("../../etc/passwd", lines=10, contains="")

    def test_get_logs_enforces_line_limit(self) -> None:
        with self.assertRaises(ValueError):
            read_recent_logs("site_director", lines=201, contains="")

    def test_get_logs_redacts_common_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "sitedirector.log").write_text(
                "Authorization: Bearer secret-value\napi_key=do-not-return\n",
                encoding="utf-8",
            )
            (data_dir / "jobmonitoring.log").write_text("ok\n", encoding="utf-8")
            (data_dir / "service_status.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"DIRAC_MCP_DATA_DIR": str(data_dir)}):
                result = read_recent_logs(
                    "site_director", lines=10, contains=""
                )

        joined = "\n".join(result["entries"])
        self.assertNotIn("secret-value", joined)
        self.assertNotIn("do-not-return", joined)
        self.assertEqual(joined.count("<redacted>"), 2)

    def test_service_status_is_allowlisted(self) -> None:
        result = read_service_status("WorkloadManagement/JobMonitoring")
        self.assertEqual(result["status"], "AVAILABLE")
        with self.assertRaises(ValueError):
            read_service_status("Configuration/Server")


class MCPProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_exposes_exactly_two_read_only_tools(self) -> None:
        async with Client(mcp) as client:
            listed = await client.list_tools()
            self.assertEqual(
                {tool.name for tool in listed.tools},
                {"get_logs", "get_service_status"},
            )
            result = await client.call_tool(
                "get_logs",
                {"source": "site_director", "lines": 1, "contains": ""},
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["returned_lines"], 1)

    async def test_mcp_schemas_become_strict_openai_tools(self) -> None:
        async with Client(mcp) as client:
            tools = await discover_openai_tools(client)

        self.assertEqual(
            [tool["name"] for tool in tools],
            ["get_logs", "get_service_status"],
        )
        self.assertTrue(all(tool["strict"] for tool in tools))
        self.assertTrue(
            all(not tool["parameters"]["additionalProperties"] for tool in tools)
        )
        self.assertEqual(
            set(tools[0]["parameters"]["properties"]),
            {"source", "lines", "contains"},
        )

        valid = {"source": "site_director", "lines": 20, "contains": "ERROR"}
        self.assertEqual(validate_tool_arguments("get_logs", valid), valid)
        with self.assertRaises(ValueError):
            validate_tool_arguments("get_logs", {**valid, "path": "/etc/passwd"})


if __name__ == "__main__":
    unittest.main()
