"""Ratchet: an envelope whose text is persisted or model-facing names its locale.

``Msg.envelope()`` / ``messages.envelope()`` render ``text`` in the process's
ACTIVE locale when no ``locale=`` is passed (so the desktop UI paints the
selected language for server-originated refusals). The flip is only safe
because every site whose text is persisted or read by a model passes
``locale="en"`` explicitly. This file keeps that true as the code grows:

1. An ``.envelope(...)`` call without a ``locale=`` keyword is allowed ONLY
   under ``HUMAN_FACING_PATHS`` (the desktop/HTTP plane and the terminal
   printers). It is an allowlist, not a list of packages to police, so a NEW
   package or module that writes an envelope without saying which locale it
   means fails by default instead of passing silently. ``MODEL_FACING_PACKAGES``
   (transcript notices, prompts, tool results) must never intersect it.
2. Elsewhere, a ``.envelope(...)["text"]`` extraction without ``locale=`` is
   allowed only at the sites in ``UNPINNED_TEXT_SITES`` (human-facing CLI /
   route text that SHOULD follow the active locale). The table only shrinks:
   a new site fails, and so does a stale entry, so deleting a site forces the
   table to follow. Adding to it is a reviewed decision that the text is
   human-facing, never a way to silence the test.

The scan is syntactic (a ``.envelope`` attribute call), the same trade as the
other source-scan pins in this suite: it will not see a body kept in a variable
and subscripted later, which is why rule 1 exists for the packages where that
matters.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "local_operator"

#: Where text is written to a transcript, a prompt or a tool result. Kept as the
#: explicit negative control: no ``HUMAN_FACING_PATHS`` entry may sit inside one.
MODEL_FACING_PACKAGES = ("session", "harness", "tools", "model", "mcp", "compaction")

#: The only places an UNPINNED ``.envelope()`` may appear (rule 1). Each is a
#: consumer whose reader is a person looking at the active language, or not an
#: i18n envelope at all. Adding an entry is a reviewed decision that the text is
#: human-facing, never a way to silence the test.
HUMAN_FACING_PATHS = (
    # HTTP plane: ``detail.text`` is what the desktop UI paints.
    "local_operator/server/",
    # Terminal printers (human CLI output).
    "local_operator/cli.py",
    "local_operator/vault_agent/cli.py",
    "local_operator/hub_org/cli.py",
    # Desktop route error carrying a catalogued sentence.
    "local_operator/notifications/custom_sounds.py",
    # LinkError: ``str(error)`` paints in the active locale for the desktop/CLI/TUI;
    # the model-facing and persisted consumers re-render pinned to ``en`` from the
    # carried Msg (``hub_sync.identity.refusal_text_en``).
    "local_operator/hub_sync/skill_adoption_service.py",
    "local_operator/hub_sync/skill_storage.py",
    # Not an i18n envelope: the signed refresh-grant wire form's own ``envelope()``.
    "local_operator/operator/verify.py",
)

#: Human-facing ``.envelope()["text"]`` sites that follow the active locale.
#: The count is per file; see the module docstring for how it may change.
UNPINNED_TEXT_SITES: dict[str, int] = {
    # Terminal skill-adoption / org-hub printers (human CLI output).
    "local_operator/cli.py": 7,
    "local_operator/vault_agent/cli.py": 1,
    "local_operator/hub_org/cli.py": 1,
    # LinkError messages: the desktop 409/422 detail and the CLI/TUI refusal
    # line paint the localized ``str(error)``. The model-read org_pull / skill
    # tool results and the persisted hub-state ``last_error`` call
    # ``hub_sync.identity.refusal_text_en`` and so render English.
    "local_operator/hub_sync/skill_adoption_service.py": 1,
    "local_operator/hub_sync/skill_storage.py": 2,
    # Route bodies the desktop paints.
    "local_operator/server/routes/desktop_tunnel_setup.py": 2,
    "local_operator/server/routes/desktop_sessions.py": 2,
}


def _unpinned_envelope_calls(path: Path) -> list[tuple[int, bool]]:
    """``(line, is_text_subscript)`` for each ``.envelope(...)`` lacking ``locale=``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "envelope"
        ):
            continue
        if any(keyword.arg == "locale" for keyword in node.keywords):
            continue
        parent = parents.get(node)
        is_text = (
            isinstance(parent, ast.Subscript)
            and isinstance(parent.slice, ast.Constant)
            and parent.slice.value == "text"
        )
        found.append((node.lineno, is_text))
    return found


def test_the_scan_is_not_vacuous() -> None:
    assert (PACKAGE / "session" / "session.py").is_file()
    assert sum(len(_unpinned_envelope_calls(p)) for p in PACKAGE.rglob("*.py")) > 100


def test_unpinned_envelopes_only_under_human_facing_paths() -> None:
    # A renamed package/file must not turn the allowlist into a vacuous pass.
    assert all((PACKAGE / package).is_dir() for package in MODEL_FACING_PACKAGES)
    assert all((REPO / path).exists() for path in HUMAN_FACING_PATHS)
    # Negative control: the allowlist never covers a transcript/prompt/tool package.
    assert not [
        path
        for path in HUMAN_FACING_PATHS
        for package in MODEL_FACING_PACKAGES
        if path.startswith(f"local_operator/{package}/")
    ]
    offenders = [
        f"{path.relative_to(REPO)}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if not str(path.relative_to(REPO)).startswith(HUMAN_FACING_PATHS)
        for line, _is_text in _unpinned_envelope_calls(path)
    ]
    assert not offenders, (
        "an envelope outside HUMAN_FACING_PATHS renders in the ACTIVE locale; pass "
        'locale="en" if its text is persisted or read by a model (see '
        "i18n.messages.envelope), else add the path to HUMAN_FACING_PATHS: " + ", ".join(offenders)
    )


def test_unpinned_text_extractions_only_at_reviewed_human_facing_sites() -> None:
    actual: Counter[str] = Counter()
    for path in sorted(PACKAGE.rglob("*.py")):
        for _line, is_text in _unpinned_envelope_calls(path):
            if is_text:
                actual[str(path.relative_to(REPO))] += 1
    grown = {f: n for f, n in actual.items() if n > UNPINNED_TEXT_SITES.get(f, 0)}
    stale = {f: n for f, n in UNPINNED_TEXT_SITES.items() if actual.get(f, 0) < n}
    assert not grown, (
        'new unpinned .envelope()["text"] — pass locale="en" if the text is persisted '
        f"or model-facing; else justify it in UNPINNED_TEXT_SITES: {grown}"
    )
    assert not stale, f"UNPINNED_TEXT_SITES is stale (shrink it to match): {stale}"


def test_the_scanner_flags_a_planted_unpinned_text_site(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(
        "def f(m):\n"
        "    a = m.envelope()['text']\n"
        "    b = m.envelope(locale='en')['text']\n"
        "    c = m.envelope(code='x')\n",
        encoding="utf-8",
    )
    assert sorted(_unpinned_envelope_calls(planted)) == [(2, True), (4, False)]
