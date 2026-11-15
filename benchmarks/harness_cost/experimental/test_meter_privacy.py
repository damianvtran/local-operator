"""Real in-process ASGI -> prepared upstream regression; never use external HTTP.

Reuse the established meter/ledger fixtures rather than constructing a different
proxy or testing only its provider-object assignment. Provider authority remains
benchmark-owned while explicit supported caller privacy constraints must survive.
"""
import json
import unittest

import test_adapters as fixtures

meter = fixtures.meter


class ProviderPrivacyTests(unittest.IsolatedAsyncioTestCase):
    # Share only fixture helpers, not AdapterTests' unrelated test matrix.
    asyncSetUp = fixtures.AdapterTests.asyncSetUp
    asyncTearDown = fixtures.AdapterTests.asyncTearDown
    serve = fixtures.AdapterTests.serve
    post = fixtures.AdapterTests.post

    async def test_preserves_explicit_strict_privacy_on_prepared_dispatch(self):
        response = await self.post(provider={"zdr": True, "data_collection": "deny"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.calls), 1)
        provider = json.loads(self.calls[0].content)["provider"]
        self.assertIs(provider.get("zdr"), True)
        self.assertEqual(provider.get("data_collection"), "deny")

    async def test_false_allow_are_preserved_without_loosening_authority(self):
        response = await self.post(provider={
            "zdr": False, "data_collection": "allow", "order": ["unapproved"],
            "allow_fallbacks": True, "max_price": {"prompt": 999},
        })
        self.assertEqual(response.status_code, 200)
        provider = json.loads(self.calls[0].content)["provider"]
        self.assertIs(provider.get("zdr"), False)
        self.assertEqual(provider.get("data_collection"), "allow")
        self.assertEqual(provider["order"], ["anthropic"])
        self.assertIs(provider["allow_fallbacks"], False)
        self.assertEqual(provider["max_price"], {"prompt": 1.0, "completion": 1.0})

    async def test_no_policy_baseline_retains_exact_provider_object(self):
        self.assertEqual((await self.post()).status_code, 200)
        self.assertEqual(json.loads(self.calls[0].content)["provider"], {
            "order": ["anthropic"], "allow_fallbacks": False,
            "max_price": {"prompt": 1.0, "completion": 1.0},
        })

    async def test_each_supported_privacy_field_and_combination_survives(self):
        policies = [{"zdr": True}, {"zdr": False},
                    {"data_collection": "deny"}, {"data_collection": "allow"},
                    {"zdr": True, "data_collection": "allow"},
                    {"zdr": False, "data_collection": "deny"}]
        for index, policy in enumerate(policies):
            with self.subTest(policy=policy):
                response = await self.post("policy-" + str(index), provider=policy)
                self.assertEqual(response.status_code, 200)
                prepared = json.loads(self.calls[-1].content)["provider"]
                for name in ("zdr", "data_collection"):
                    if name in policy:
                        self.assertEqual(prepared[name], policy[name])
                        self.assertIs(type(prepared[name]), type(policy[name]))
                    else:
                        self.assertNotIn(name, prepared)
                self.assertEqual(prepared["order"], ["anthropic"])
                self.assertIs(prepared["allow_fallbacks"], False)

    async def test_invalid_or_unknown_privacy_rejected_before_dispatch(self):
        invalid = [None, [], "deny", True,
                   {"zdr": 0}, {"zdr": 1}, {"zdr": "true"}, {"zdr": None},
                   {"zdr": []}, {"data_collection": True}, {"data_collection": None},
                   {"data_collection": "DENY"}, {"data_collection": "unknown"},
                   {"data_collection": {}}, {"unknown_privacy_flag": True},
                   {"zdr": True, "data_collection": "deny", "only": ["other"]}]
        for index, provider in enumerate(invalid):
            with self.subTest(index=index):
                response = await self.post("invalid-" + str(index), provider=provider)
                self.assertEqual(response.status_code, 402)
                self.assertEqual(self.calls, [])

    async def test_privacy_does_not_bypass_existing_request_or_tool_guards(self):
        invalid = [
            {"service_tier": "priority"}, {"max_tokens": 128001},
            {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]},
            {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
            {"unexpected_server_extension": True},
        ]
        for index, updates in enumerate(invalid):
            with self.subTest(index=index):
                response = await self.post("guard-" + str(index),
                    provider={"zdr": True, "data_collection": "deny"}, **updates)
                self.assertEqual(response.status_code, 402)
                self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
