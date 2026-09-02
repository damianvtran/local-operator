"""A2 evidence: drive the REAL picker catalogue against a stale-capture document.

Plants a capture-1 models.dev projection (built from the real cached document,
so the row set is the operator's actual one) into an ISOLATED HOME, then drives
``ProviderController.live_catalogue`` — the same call the picker's row builder
makes — and reports the price the picker would render for the 8 rows the round-5
review named, open by open.

Not a unit test: the point is that the repair happens on the real surface, with
the real documents and the real network, rather than through a stub.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

WATCHED = [
    ("openai", "gpt-5.3-codex-spark"),
    ("openai-device", "gpt-5.3-codex-spark"),
    ("xai", "grok-4.20-0309-reasoning"),
    ("xai", "grok-4.20-0309-non-reasoning"),
    ("xai", "grok-4.20-multi-agent-0309"),
    ("xai-oauth", "grok-4.20-0309-reasoning"),
    ("xai-oauth", "grok-4.20-0309-non-reasoning"),
    ("xai-oauth", "grok-4.20-multi-agent-0309"),
]


def main() -> int:
    real_cache = Path.home() / ".local-operator" / "cache"
    home = Path(tempfile.mkdtemp(prefix="lop-a2-home-"))
    cache = home / ".local-operator" / "cache"
    cache.mkdir(parents=True)
    for doc in real_cache.glob("*.listing.json"):
        stored = json.loads(doc.read_text())
        # Re-stamped fresh so no PROVIDER listing is refetched during the probe:
        # the only document under test is models.dev, and a provider listing
        # past its hard TTL would put unrelated network calls (and unrelated
        # repairs) inside the window the watched rows are measured in.
        stored["fetched_at"] = time.time()
        (cache / doc.name).write_text(json.dumps(stored))
    os.environ["HOME"] = str(home)

    from local_operator.model import prices
    from local_operator.model.catalogue import _revalidation_threads
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    path = cache / f"{prices.PRICE_CATALOGUE_KEY}.json"

    def capture() -> object:
        if not path.exists():
            return "ABSENT"
        return json.loads(path.read_text())["payload"].get("capture")

    def plant_capture(version: int, age_s: float = 0.0) -> None:
        stored = json.loads(path.read_text()) if path.exists() else None
        assert stored is not None, "need a real document to downgrade"
        stored["payload"]["capture"] = version
        stored["fetched_at"] = time.time() - age_s
        path.write_text(json.dumps(stored))

    # Built the way cli.py builds the TUI's controller, against the isolated
    # HOME. The watched rows come from the xai and openai listings, which are
    # only READ for a provider that has a credential — so the store is seeded
    # with placeholders. They are never spent: every listing document was
    # re-stamped fresh above, so discovery serves from disk and issues no
    # request with them.
    store = AuthStore()
    store.upsert_credential("xai", {"key": "probe-not-a-real-key", "source": "login"})
    # ``gpt-5.3-codex-spark`` is only in the ACCOUNT-SCOPED ChatGPT catalogue,
    # whose document name hashes the account id, so an OAuth row is seeded and
    # the real document re-keyed onto the probe's synthetic account. Content is
    # untouched: only the scope key moves, so the rows under test are the real
    # ones. Without this the openai/openai-device pair never appears at all.
    account_id = "probe-account"
    store.upsert_credential(
        "openai",
        {"access": "probe-not-a-real-token", "refresh": "probe", "account_id": account_id},
    )
    scope = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:12]
    scoped = cache / f"openai.oauth.{scope}.listing.json"
    for name in (
        "openai.oauth.87fdefa5f580.listing.json",
        "openai.codex.87fdefa5f580.listing.json",
    ):
        source = cache / name
        if source.exists() and not scoped.exists():
            shutil.copy2(source, scoped)
    controller = ProviderController(store, None)

    def open_picker(label: str) -> dict[tuple[str, str], str]:
        from local_operator.tui.widgets.model_picker import format_price_pair

        before = capture()
        entries, _statuses = asyncio.run(controller.live_catalogue())
        for thread in _revalidation_threads():
            thread.join(timeout=30.0)
        rendered = {
            (e.provider, e.model_id): format_price_pair(e.input_price, e.output_price)
            for e in entries
        }
        watched = {k: rendered.get(k, "<missing row>") for k in WATCHED}
        priced = sum(1 for v in watched.values() if v not in ("", "<missing row>"))
        print(f"{label}: capture {before} -> {capture()}  watched priced {priced}/8")
        for k, v in watched.items():
            print(f"    {k[0]}/{k[1]:<34} {v!r}")
        return watched

    print("=== baseline: the document as it really is ===")
    open_picker("run 0")

    # The repair is bounded by REVALIDATE_BACKOFF_S, which is the point of the
    # loop test — but it also means a probe that just used its one attempt on
    # the baseline would measure the backoff rather than the repair. Cleared
    # here so the planted case starts from the same state a fresh session does.
    from local_operator.model import catalogue as _catalogue

    with _catalogue._revalidate_lock:
        _catalogue._revalidating.clear()
        _catalogue._last_attempt.clear()
        _catalogue._threads.clear()

    print("\n=== A2: plant capture 1, aged 30 days, and open the picker ===")
    plant_capture(prices.PRICE_CATALOGUE_CAPTURE - 1, age_s=30 * 24 * 3600)
    print(f"planted capture: {capture()}")
    first = open_picker("open 1")
    second = open_picker("open 2")

    repaired = all(v not in ("", "<missing row>") for v in second.values())
    degraded = all(v == "" for v in first.values())
    print(f"\nopen 1 all blank (degraded, no synchronous fetch): {degraded}")
    print(f"open 2 all priced (repaired within one open):        {repaired}")

    print("\n=== the openrouter/radient 3->4 path still self-repairs ===")
    for provider in ("openrouter", "radient"):
        doc = cache / f"{provider}.listing.json"
        if not doc.exists():
            print(f"  {provider}: no cached document on this machine, skipped")
            continue
        stored = json.loads(doc.read_text())
        stored["payload"]["capture"] = 3
        stored["fetched_at"] = time.time() - 30 * 24 * 3600
        doc.write_text(json.dumps(stored))
        asyncio.run(controller.live_catalogue())
        for thread in _revalidation_threads():
            thread.join(timeout=30.0)
        after = json.loads(doc.read_text())["payload"].get("capture")
        print(f"  {provider}: planted capture 3 -> {after} after one picker open")

    shutil.rmtree(home, ignore_errors=True)
    return 0 if (repaired and degraded) else 1


if __name__ == "__main__":
    sys.exit(main())
