"""Offline invariants for the proposed final-wire namespace policy.

No provider calls, filesystem fixture content or credentials. This checks the
mutation whitelist, exact tool/schema/marker/TTL preservation and refusal paths;
it does not prove anything about proprietary serialization or cache hits.
"""
import copy
import unittest
from cache_namespace import instrument, namespace_tag


class NamespaceTests(unittest.TestCase):
    def body(self):
        return {"model": "test", "tools": [
            {"name": "echo", "description": "Return the supplied text.",
             "input_schema": {"type": "object", "properties": {"cache_control": {"type": "string"}}},
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"name": "later", "description": "Second.", "defer_loading": True}],
            "system": [{"type": "text", "text": "Rules", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "opaque", "signature": "signature"}]}]}

    def test_only_two_text_fields_change(self):
        b = self.body(); original = copy.deepcopy(b)
        out, meta = instrument(b, "schedule:trial:1")
        self.assertEqual(b, original)
        tag = namespace_tag("schedule:trial:1")
        self.assertEqual(out["tools"][0]["description"], tag + b["tools"][0]["description"])
        self.assertEqual(out["system"][0]["text"], tag + b["system"][0]["text"])
        out["tools"][0]["description"] = b["tools"][0]["description"]
        out["system"][0]["text"] = b["system"][0]["text"]
        self.assertEqual(out, original)
        self.assertEqual(meta["locations"], ["/tools/0/description", "/system/0/text"])

    def test_scope_stability(self):
        self.assertEqual(namespace_tag("A"), namespace_tag("A"))
        self.assertNotEqual(namespace_tag("A"), namespace_tag("B"))
        self.assertEqual(len(namespace_tag("A")), len(namespace_tag("B")))
        base = self.body()
        a, _ = instrument(base, "A")
        base["messages"] += [{"role": "user", "content": "a summary"}]
        b, _ = instrument(base, "A")
        self.assertEqual(a["tools"], b["tools"])
        self.assertEqual(a["system"], b["system"])

    def test_changed_first_tool_gets_same_single_tag(self):
        b = self.body(); b["tools"] = [{"name": "new", "description": "New."}] + b["tools"]
        out, _ = instrument(b, "A")
        self.assertTrue(out["tools"][0]["description"].startswith(namespace_tag("A")))
        self.assertEqual(out["tools"][1:], b["tools"][1:])

    def test_absent_system_and_no_tools(self):
        out, _ = instrument({"messages": []}, "A")
        self.assertEqual(out["system"][0]["text"], namespace_tag("A"))
        out, _ = instrument({"system": "original"}, "A")
        self.assertEqual(out["system"], namespace_tag("A") + "original")

    def test_refusals(self):
        for patch in ({"description": ""}, {"defer_loading": True}, {"type": "tool_search_tool_regex_20251119"}):
            b = self.body(); b["tools"][0].update(patch)
            with self.assertRaises(ValueError): instrument(b, "A")
        b = self.body(); b["system"][0]["type"] = "image"
        with self.assertRaises(ValueError): instrument(b, "A")
        out, _ = instrument(self.body(), "A")
        with self.assertRaises(ValueError): instrument(out, "A")
        with self.assertRaises(ValueError): namespace_tag("")


if __name__ == "__main__":
    unittest.main()
