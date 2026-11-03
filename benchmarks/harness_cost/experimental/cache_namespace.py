"""Instrument final Anthropic Messages bodies for trial-scoped prompt caches.

Why: tools precede system/messages in the cache prefix. A system-only nonce
cannot isolate native tool breakpoints. Apply AFTER translators/bridges, before
metering, once tool array order is final. Caller supplies the schedule-derived
trial scope, stable across main, summary and side calls; never derive it from
request contents. No tokens, credentials or private content are needed.

Policy cache-namespace-decimal-v2 (proposed, NOT authorized for headline collection):
prepend one fixed-format administrative tag to tools[0].description, if tools
exist, and to system[0].text. Preserve EVERY original tool, schema, marker,
TTL, message and thinking block. No sorting or per-tool salt. Reject missing
first descriptions, non-text first system blocks, or pre-instrumented input.
Absent system gets an administrative-only text block: this additive intervention
needs explicit qualification. No tools is supported. Server-native first tool
variants need separate on-route validation; reject first defer_loading=True
because deferred descriptions may not enter the initial model prefix.

Native cache_control breakpoints are preserved, not normalized: every native
breakpoint in tools/system/messages is after a tagged location. We do not
recursively delete keys from schemas (a cache_control property can be user
schema data). See CACHE-QUALIFICATION.md for limits, measured overhead and TTL
fallback. Serialization hashes refer to JSON, NOT proprietary model token order.
"""
from __future__ import annotations
import copy
import hashlib
import json

POLICY = "cache-namespace-decimal-v2"


def digest(value):
    """Hash canonical JSON without publishing its potentially private content."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def prefix_fingerprints(body: dict) -> list[dict]:
    """Hash conceptual cache prefixes at actual marker locations.

    Byte lengths are canonical JSON bytes, NOT token or proprietary serialized
    positions. Actual token-length evidence is the provider's read/write usage.
    Include model/tool_choice/thinking to avoid claiming lineage across
    incompatible settings. No prompt contents are returned.
    """
    settings = {k: body[k] for k in ("model", "tool_choice", "thinking", "output_config") if k in body}
    tools = body.get("tools") or []
    system = body.get("system")
    systems = system if isinstance(system, list) else ([{"type": "text", "text": system}] if system else [])
    result = []
    def add(path, marker, prefix):
        wire = json.dumps(prefix, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        result.append({"path": path, "cache_control": marker, "conceptual_prefix_sha256": hashlib.sha256(wire).hexdigest(),
                       "canonical_json_prefix_bytes": len(wire)})
    for i, t in enumerate(tools):
        if "cache_control" in t:
            add(f"/tools/{i}", t["cache_control"], {**settings, "tools": tools[:i+1]})
    for i, s in enumerate(systems):
        if "cache_control" in s:
            add(f"/system/{i}", s["cache_control"], {**settings, "tools": tools, "system": systems[:i+1]})
    previous = []
    for i, m in enumerate(body.get("messages") or []):
        content = m.get("content")
        if isinstance(content, list):
            for j, block in enumerate(content):
                if "cache_control" in block:
                    add(f"/messages/{i}/content/{j}", block["cache_control"],
                        {**settings, "tools": tools, "system": systems, "messages": previous + [{**m, "content": content[:j+1]}]})
        previous.append(m)
    if "cache_control" in body:
        add("/cache_control", body["cache_control"], {**settings, "tools": tools, "system": systems, "messages": previous})
    return result


def namespace_tag(scope: str) -> str:
    """32 separated decimal digits (~106-bit scope); schedule, not task text.

    Initial equal-length hex tags failed the preregistered token tolerance on
    the actual route (up to 12 tokens/request variation). Digits separated by
    spaces prevent variable-length hex-word token merges. Still qualify every
    frozen schedule's tag set: this is a measured format, not a tokenizer proof.
    """
    if not isinstance(scope, str) or not scope:
        raise ValueError("nonempty schedule scope required")
    value = int(hashlib.sha256(scope.encode()).hexdigest()[:32], 16) % (10 ** 32)
    code = " ".join(f"{value:032d}")
    return f"[Administrative cache namespace: {code}. Ignore for task behavior.]\n"


def instrument(body: dict, scope: str) -> tuple[dict, dict]:
    """Deep-copy and transform final Messages input, or fail before dispatch."""
    if not isinstance(body, dict):
        raise ValueError("Messages body must be an object")
    b = copy.deepcopy(body)
    tag = namespace_tag(scope)
    locations = []
    tools = b.get("tools") or []
    if not isinstance(tools, list):
        raise ValueError("final Messages tools must be an array")
    if tools:
        t = tools[0]
        if not isinstance(t, dict) or t.get("type") not in (None, "custom"):
            raise ValueError("unqualified server-native first tool; use TTL isolation")
        if not isinstance(t.get("description"), str) or not t["description"]:
            raise ValueError("first serialized tool has no description; use TTL isolation")
        if t.get("defer_loading"):
            raise ValueError("first tool is deferred; serialized-prefix qualification required")
        if t["description"].startswith("[Administrative cache namespace:"):
            raise ValueError("already instrumented")
        t["description"] = tag + t["description"]
        locations.append("/tools/0/description")
    system = b.get("system")
    if isinstance(system, str):
        if system.startswith("[Administrative cache namespace:"):
            raise ValueError("already instrumented")
        b["system"] = tag + system
        locations.append("/system")
    elif system is None or system == []:
        b["system"] = [{"type": "text", "text": tag}]
        locations.append("/system/0/text (added block)")
    elif isinstance(system, list) and isinstance(system[0], dict) and system[0].get("type") == "text" and isinstance(system[0].get("text"), str):
        if system[0]["text"].startswith("[Administrative cache namespace:"):
            raise ValueError("already instrumented")
        system[0]["text"] = tag + system[0]["text"]
        locations.append("/system/0/text")
    else:
        raise ValueError("first system block is not text; use TTL isolation")
    return b, {"policy": POLICY, "scope_sha256": hashlib.sha256(scope.encode()).hexdigest(),
               "tag_chars": len(tag), "locations": locations, "before_sha256": digest(body),
               "after_sha256": digest(b), "tools_sha256": digest(b.get("tools")),
               "system_sha256": digest(b.get("system")), "prefix_fingerprints": prefix_fingerprints(b)}
