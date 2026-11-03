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
import json
import unittest
from cache_namespace import instrument, namespace_tag
from codex_toolsearch_bridge import rewrite_request, _convert_item
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig as Responses
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from jsonschema import validate, ValidationError


class BridgeNamespaceTests(unittest.TestCase):
    def test_roundtrip_after_final_mapping(self):
        spec = {"type": "namespace", "name": "public_echo", "description": "Public echo tools.",
                "tools": [{"type": "function", "name": "echo", "description": "Return public text.",
                           "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
        search = {"type": "tool_search", "execution": "client", "description": "Search original deferred metadata.",
                  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
        original = {"tools": [search], "input": []}
        saved = copy.deepcopy(original)
        mapped, _ = rewrite_request(copy.deepcopy(original))
        chat_tools, _ = Responses.transform_responses_api_tools_to_chat_completion_tools(mapped["tools"])
        final_tools = [AnthropicConfig()._map_tool_helper(t)[0] for t in chat_tools]
        body = {"tools": final_tools, "system": [{"type": "text", "text": "Public rules."}], "messages": []}
        namespaced, _ = instrument(body, "cachedev:bridge:trial-A")
        self.assertEqual(namespaced["tools"][0]["name"], "tool_search")
        validate({"query": "echo"}, namespaced["tools"][0]["input_schema"])
        call = _convert_item({"type": "function_call", "id": "call-item", "call_id": "call1", "name": "tool_search", "arguments": '{"query":"echo"}'})
        self.assertEqual(call["type"], "tool_search_call")
        self.assertEqual(call["arguments"], {"query": "echo"})
        original["input"] = [call, {"type": "tool_search_output", "execution": "client", "status": "completed", "call_id": "call1", "tools": [spec]}]
        mapped2, stats = rewrite_request(copy.deepcopy(original))
        self.assertEqual(stats["loaded_specs"], 1)
        chat2, _ = Responses.transform_responses_api_tools_to_chat_completion_tools(mapped2["tools"])
        final2 = [AnthropicConfig()._map_tool_helper(t)[0] for t in chat2]
        expanded, _ = instrument({**body, "tools": final2}, "cachedev:bridge:trial-A")
        self.assertEqual(expanded["tools"][0], namespaced["tools"][0])
        self.assertEqual(expanded["tools"][1]["name"], "public_echo__echo")
        self.assertEqual(expanded["tools"][1]["description"], final2[1]["description"])
        self.assertNotIn(namespace_tag("cachedev:bridge:trial-A"), json.dumps(mapped2))
        self.assertEqual(saved["tools"], original["tools"])  # original search metadata unchanged
        schema = expanded["tools"][1]["input_schema"]
        validate({"text": "hello"}, schema)
        with self.assertRaises(ValidationError): validate({"text": 123}, schema)
        with self.assertRaises(ValidationError): validate({}, schema)
        self.assertEqual(mapped2["input"][0]["type"], "function_call")
        self.assertEqual(mapped2["input"][1]["type"], "function_call_output")


if __name__ == "__main__": unittest.main()
