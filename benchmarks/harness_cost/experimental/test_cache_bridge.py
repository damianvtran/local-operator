"""Check namespace mutation after the REAL Codex bridge + LiteLLM mappings.

Why: salting before search would contaminate BM25; salt only the final Messages
wire body after Codex/tool-search and Responses->Messages translation. This
$0 integration test imports the installed LiteLLM 1.106.6 transformation code
and the lane's actual bridge functions; no stubs replace those mappings.
It proves request mapping/search-result expansion and response-item restoration,
not a fresh live Codex process or stochastic model tool choice. Existing paid
bridge runs and the public echo controls are separate evidence.

Run with the existing LiteLLM environment's Python (no dependency install):
  <litellm-python> -m unittest -v test_cache_bridge
No servers, keychain calls, model requests or private fixture content.
"""

import copy
import importlib
import json
import unittest
from types import SimpleNamespace
from typing import Protocol, runtime_checkable
from unittest.mock import patch

from cache_namespace import instrument, namespace_tag
from codex_toolsearch_bridge import WireObject, _convert_item, rewrite_request
from jsonschema import ValidationError, validate


@runtime_checkable
class ResponsesMapping(Protocol):
    def transform_responses_api_tools_to_chat_completion_tools(
        self, tools: list[WireObject]
    ) -> tuple[list[WireObject], object]: ...


@runtime_checkable
class AnthropicMapping(Protocol):
    def _map_tool_helper(self, tool: WireObject) -> tuple[WireObject, object]: ...


def load_mappings() -> tuple[ResponsesMapping, AnthropicMapping]:
    # LiteLLM is an optional benchmark plugin, not a core/CI dependency. Validate
    # its two used interfaces instead of making the whole integration untyped.
    responses = importlib.import_module(
        "litellm.responses.litellm_completion_transformation.transformation"
    ).LiteLLMCompletionResponsesConfig
    anthropic = importlib.import_module(
        "litellm.llms.anthropic.chat.transformation"
    ).AnthropicConfig()
    if (
        not isinstance(responses, ResponsesMapping)
        or not callable(responses.transform_responses_api_tools_to_chat_completion_tools)
        or not isinstance(anthropic, AnthropicMapping)
        or not callable(anthropic._map_tool_helper)
    ):
        raise TypeError("Installed LiteLLM does not expose the benchmark mapping interfaces")
    return responses, anthropic


class PluginBoundaryTests(unittest.TestCase):
    def test_validated_interfaces_use_installed_objects(self):
        class Responses:
            @staticmethod
            def transform_responses_api_tools_to_chat_completion_tools(tools):
                return tools, None

        class Anthropic:
            def _map_tool_helper(self, tool):
                return tool, None

        with patch.object(
            importlib,
            "import_module",
            side_effect=[
                SimpleNamespace(LiteLLMCompletionResponsesConfig=Responses),
                SimpleNamespace(AnthropicConfig=Anthropic),
            ],
        ):
            responses, anthropic = load_mappings()
        self.assertIs(responses, Responses)
        tool = {"name": "echo"}
        tools, _ = responses.transform_responses_api_tools_to_chat_completion_tools([tool])
        mapped, _ = anthropic._map_tool_helper(tools[0])
        self.assertIs(mapped, tool)

    def test_missing_or_noncallable_interfaces_fail_closed(self):
        for invalid in (
            object(),
            SimpleNamespace(
                transform_responses_api_tools_to_chat_completion_tools=42,
                _map_tool_helper=42,
            ),
        ):
            with (
                self.subTest(plugin=invalid),
                patch.object(
                    importlib,
                    "import_module",
                    side_effect=[
                        SimpleNamespace(LiteLLMCompletionResponsesConfig=invalid),
                        SimpleNamespace(AnthropicConfig=lambda: invalid),
                    ],
                ),
                self.assertRaises(TypeError),
            ):
                load_mappings()

    def test_missing_plugin_is_not_silent_coverage(self):
        with (
            patch.object(importlib, "import_module", side_effect=ModuleNotFoundError("litellm")),
            self.assertRaises(ModuleNotFoundError),
        ):
            load_mappings()


class BridgeNamespaceTests(unittest.TestCase):
    def test_roundtrip_after_final_mapping(self):
        Responses, anthropic = load_mappings()
        spec = {
            "type": "namespace",
            "name": "public_echo",
            "description": "Public echo tools.",
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "description": "Return public text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ],
        }
        search = {
            "type": "tool_search",
            "execution": "client",
            "description": "Search original deferred metadata.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        original = {"tools": [search], "input": []}
        saved = copy.deepcopy(original)
        mapped, _ = rewrite_request(copy.deepcopy(original))
        chat_tools, _ = Responses.transform_responses_api_tools_to_chat_completion_tools(
            mapped["tools"]
        )
        final_tools = [anthropic._map_tool_helper(t)[0] for t in chat_tools]
        body = {
            "tools": final_tools,
            "system": [{"type": "text", "text": "Public rules."}],
            "messages": [],
        }
        namespaced, _ = instrument(body, "cachedev:bridge:trial-A")
        self.assertEqual(namespaced["tools"][0]["name"], "tool_search")
        validate({"query": "echo"}, namespaced["tools"][0]["input_schema"])
        call = _convert_item(
            {
                "type": "function_call",
                "id": "call-item",
                "call_id": "call1",
                "name": "tool_search",
                "arguments": '{"query":"echo"}',
            }
        )
        self.assertEqual(call["type"], "tool_search_call")
        self.assertEqual(call["arguments"], {"query": "echo"})
        original["input"] = [
            call,
            {
                "type": "tool_search_output",
                "execution": "client",
                "status": "completed",
                "call_id": "call1",
                "tools": [spec],
            },
        ]
        mapped2, stats = rewrite_request(copy.deepcopy(original))
        self.assertEqual(stats["loaded_specs"], 1)
        chat2, _ = Responses.transform_responses_api_tools_to_chat_completion_tools(
            mapped2["tools"]
        )
        final2 = [anthropic._map_tool_helper(t)[0] for t in chat2]
        expanded, _ = instrument({**body, "tools": final2}, "cachedev:bridge:trial-A")
        self.assertEqual(expanded["tools"][0], namespaced["tools"][0])
        self.assertEqual(expanded["tools"][1]["name"], "public_echo__echo")
        self.assertEqual(expanded["tools"][1]["description"], final2[1]["description"])
        self.assertNotIn(namespace_tag("cachedev:bridge:trial-A"), json.dumps(mapped2))
        self.assertEqual(saved["tools"], original["tools"])  # original search metadata unchanged
        schema = expanded["tools"][1]["input_schema"]
        validate({"text": "hello"}, schema)
        with self.assertRaises(ValidationError):
            validate({"text": 123}, schema)
        with self.assertRaises(ValidationError):
            validate({}, schema)
        self.assertEqual(mapped2["input"][0]["type"], "function_call")
        self.assertEqual(mapped2["input"][1]["type"], "function_call_output")


if __name__ == "__main__":
    unittest.main()
