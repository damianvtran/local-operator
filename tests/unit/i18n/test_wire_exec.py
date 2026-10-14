"""The E2b slice: `wire.exec` on the catalogue.

What this file pins, and why each pin exists:

1. **The token boundary, both sides.** RFC §1.2.2: the receipt skeleton
   (`Background job `, the job id, `: `, the status token) and the
   `exec failed: ` prefix are MACHINE vocabulary — parsers match them — so
   they are composed in code and must NOT appear inside any catalogue value
   (a translation could otherwise move them). This asserts the absence in the
   shipped `en` catalogue AND the presence in the composing sources: a
   refactor that moves either token into the catalogue fails here, and so
   does one that deletes the token from the code path.
2. **The parsers still match what the code composes.** The receipt line is
   rebuilt exactly the way `_receipt_line`-side code composes it (anchor +
   catalogue suffix, failure tail included) and both consumers' regexes are
   run against it: `session/bulk_resume._JOB_LINE_RE` and the sessions tool's
   twin. This is the pin that makes the boundary real rather than nominal.
3. **The dynamic-payload refusals are pragma'd, not converted.** The three
   `exec failed: {exc}`-style sites carry the in-tree pragma with its reason;
   converting one would put an OS exception into the catalogue.
4. **Renders resolve.** A smoke render of every key through `tr("wire.exec")`
   equals the catalogue value byte-for-byte under the default locale.
"""

from __future__ import annotations

from pathlib import Path

import local_operator
from local_operator.i18n import catalogues
from local_operator.i18n.keys import wire_exec as w
from local_operator.i18n.text import tr

_t = tr("wire.exec")
_ROOT = Path(local_operator.__file__).resolve().parent.parent

SOURCES = (
    "local_operator/exec_mode.py",
    "local_operator/exec_session.py",
    "local_operator/exec_startup.py",
    "local_operator/cli.py",
)


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def test_machine_tokens_are_code_side_and_catalogue_free() -> None:
    values = catalogues.load_catalogue("en", "wire.exec").values()
    assert all("Background job" not in value for value in values)
    assert all("exec failed: " not in value for value in values)
    for rel in SOURCES:
        source = _read(rel)
        if rel.endswith("exec_mode.py"):
            assert 'f"Background job {job_id}: {status}"' in source
        # The anchor rides every module that prints the refusal line.
        assert "exec failed: " in source


def test_the_receipt_regexes_match_what_the_code_composes() -> None:
    from local_operator.session.bulk_resume import _JOB_LINE_RE

    running = "Background job ab12: running" + _t(w.receipt_suffix())
    match = _JOB_LINE_RE.match(running)
    assert match is not None
    assert match.group(1) == "ab12"
    assert match.group(2) == "running"

    failed = "Background job ab12: failed" + _t(
        w.receipt_failure_suffix(failure="worker died: boom")
    )
    match = _JOB_LINE_RE.match(failed)
    assert match is not None
    assert match.group(1) == "ab12"
    assert match.group(2) == "failed"
    # Group 3 is the REMAINDER the child reader surfaces — the failure tail
    # must survive in it.
    assert "worker died: boom" in match.group(3)


def test_the_sessions_tool_regex_agrees() -> None:
    from local_operator.tools.builtin import _SESSIONS_JOB_LINE_RE

    line = "Background job ab12: finished" + _t(w.receipt_suffix())
    match = _SESSIONS_JOB_LINE_RE.search(line)
    assert match is not None
    assert match.group(1) == "ab12"


def test_dynamic_payload_refusals_carry_the_pragma() -> None:
    mode = _read("local_operator/exec_mode.py")
    assert mode.count("# i18n: ignore stable prefix + dynamic payload") == 3
    assert "# i18n: ignore stable prefix + dynamic payload" in _read(
        "local_operator/exec_session.py"
    )


def test_every_key_renders_under_the_default_locale() -> None:
    messages = {
        "approval_advisory": w.approval_advisory(),
        "clear_goal_needs_start": w.clear_goal_needs_start(),
        "error_line": w.error_line(error="boom"),
        "error_preflight": w.error_preflight(error="boom"),
        "exec_epilog": w.exec_epilog(),
        "exec_help": w.exec_help(),
        "goal_only_sets_objective": w.goal_only_sets_objective(),
        "log_pointer": w.log_pointer(log_path="/tmp/x.log"),
        "no_exec_job": w.no_exec_job(job_id="'j1'", jobs_path="/tmp/jobs.jsonl"),
        "no_prompt": w.no_prompt(),
        "receipt_failure_suffix": w.receipt_failure_suffix(failure="boom"),
        "receipt_suffix": w.receipt_suffix(),
        "session_pointer": w.session_pointer(session_id="ab12"),
        "status_pointer": w.status_pointer(job_id="ab12"),
        "tools_unreachable": w.tools_unreachable(names="a, b", count="3"),
        "turn_did_not_complete": w.turn_did_not_complete(),
        "waiting_on_you": w.waiting_on_you(),
    }
    sources = catalogues.load_catalogue("en", "wire.exec")
    for name, message in messages.items():
        rendered = _t(message)
        assert rendered != message.code, name
        source = sources[message.code]
        if "{" not in source:
            # No parameters: the render is the source byte-for-byte.
            assert rendered == source, name
        else:
            # Parameterised: substitution happened, and no placeholder survived.
            assert "{" not in rendered and "}" not in rendered, name
