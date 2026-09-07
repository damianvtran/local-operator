"""What must never reach a GitHub issue. The highest-value file in this package.

``/info``'s export is written to be pasted somewhere public, so every assertion
here is about something that would be a real disclosure rather than a cosmetic
defect. Two of them are structural rather than textual — invariant #13 checks
that ``control_key`` is not a FIELD, so the guarantee survives a later renderer
change that a string search over today's output would not catch.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from local_operator.info.model import (
    AgentsInfo,
    EnvInfo,
    InfoSnapshot,
    InstallInfo,
    ProcessInfo,
    SessionLine,
    SessionsInfo,
    SubagentLine,
)
from local_operator.info.render import build_export, relativise_home

#: A control key of the real shape: ``SessionRecord.control_key`` is 64 hex
#: characters and IS the control socket's whole authorization story.
CONTROL_KEY = "9f" * 32
#: ``BridgeState.session_key`` carries ``Field(min_length=32)``.
SESSION_KEY = "b3" * 24
#: A credential VALUE. Its NAME is expected in the output; this is not.
CREDENTIAL_VALUE = "sk-live-ThisIsATotallyRealLookingSecretValue"


def _snapshot() -> InfoSnapshot:
    """A snapshot seeded with every secret the collectors could plausibly meet."""
    home = str(Path.home())
    return InfoSnapshot(
        install=InstallInfo(
            version="0.51.6",
            kind="uv-tool",
            prefix=f"{home}/.local/share/uv/tools/local-operator",
            executable=f"{home}/.local/share/uv/tools/local-operator/bin/python",
            source_ref="4311eb653aa9f00",
            is_git_snapshot=True,
            python_version="3.12.13",
            platform="macOS-26.6.2-arm64",
            machine="arm64",
        ),
        process=ProcessInfo(
            pid=4243,
            session_id="a3f9c21b7e40",
            conversation_name="Investigate request latency",
            cwd=f"{home}/clients/acme-merger-diligence",
            model_label="anthropic/claude-opus-5",
            config_dir=f"{home}/.local-operator",
            cache_dir=f"{home}/.local-operator/cache",
            agent_home=f"{home}/local-operator-home",
            log_dir=f"{home}/Library/Logs/local-operator",
            control_port=51234,
            protocol=5,
            uptime_s=840.0,
            kind="tui",
        ),
        sessions=SessionsInfo(
            lines=(
                SessionLine(
                    pid=4243,
                    kind="tui",
                    state="live",
                    session_id="a3f9c21b7e40",
                    conversation_name="Investigate request latency",
                    model_label="anthropic/claude-opus-5",
                    cwd=f"{home}/clients/acme-merger-diligence",
                    uptime_s=840.0,
                    rss_bytes=190_000_000,
                    is_self=True,
                ),
            ),
            total=1,
            live=1,
        ),
        agents=AgentsInfo(
            profiles=20,
            teams=3,
            tree=(SubagentLine(job_id="j1", label="reviewer", status="running"),),
            running=1,
        ),
        env=EnvInfo(
            credential_keys=("RADIENT_API_KEY", "GOOGLE_ACCESS_TOKEN"),
            theme="dusk",
            term="xterm-256color",
            browser_backend="extension",
            browser_name="Chrome",
        ),
        captured_at=1_788_602_400.0,
    )


def test_the_control_key_cannot_appear_because_the_field_does_not_exist() -> None:
    """Invariant #13, and the reason it is STRUCTURAL rather than a text search.

    A renderer that started dumping the whole session line would leak the key if
    the field merely went unrendered. There is no attribute to reach, so the
    guarantee holds through a change nobody remembers to re-audit.
    """
    assert "control_key" not in SessionLine.__dataclass_fields__
    assert "control_key" not in ProcessInfo.__dataclass_fields__
    # And the safe half IS carried: a port number is not a credential, and it is
    # the useful half for debugging an attach.
    assert "control_port" in ProcessInfo.__dataclass_fields__


def test_no_dataclass_in_the_package_carries_a_secret_shaped_field() -> None:
    """A blanket sweep, so a NEW field named like a secret fails here first."""
    forbidden = ("control_key", "session_key", "token", "secret", "password", "api_key")
    for cls in (
        InstallInfo,
        ProcessInfo,
        SessionLine,
        SessionsInfo,
        SubagentLine,
        AgentsInfo,
        EnvInfo,
        InfoSnapshot,
    ):
        for name in cls.__dataclass_fields__:
            assert not any(bad in name.lower() for bad in forbidden), f"{cls.__name__}.{name}"


def test_a_seeded_control_key_never_survives_the_export() -> None:
    """Even a 16-character substring, which is enough to be worth grepping for."""
    text = build_export(_snapshot())
    assert CONTROL_KEY not in text
    assert CONTROL_KEY[:16] not in text


def test_a_seeded_bridge_session_key_never_survives_the_export() -> None:
    text = build_export(_snapshot())
    assert SESSION_KEY not in text
    assert SESSION_KEY[:16] not in text


def test_credential_names_are_exported_and_values_are_not() -> None:
    """The name answers the diagnostic question; the value answers nothing."""
    text = build_export(_snapshot())
    assert "RADIENT_API_KEY" in text
    assert "GOOGLE_ACCESS_TOKEN" in text
    assert CREDENTIAL_VALUE not in text
    # No prefix, no length, no hash: a "first four characters" habit is exactly
    # how key prefixes end up in issues.
    assert CREDENTIAL_VALUE[:4] not in text


def test_no_absolute_home_prefix_survives_the_export() -> None:
    """``~/`` does; ``/Users/<name>/`` and ``/home/<name>/`` do not.

    A macOS username is low-sensitivity on its own, but the export is the
    artifact that leaves the machine and a ``cwd`` like
    ``clients/acme-merger-diligence`` beneath it is real information about
    someone else.
    """
    text = build_export(_snapshot())
    assert str(Path.home()) not in text
    assert not re.search(r"/Users/[^/\s]+/", text)
    assert not re.search(r"/home/[^/\s]+/", text)
    assert "~/" in text
    # The diagnostic part BELOW the home directory is deliberately kept: it is
    # what tells a maintainer which tree the reporter was in.
    assert "~/.local-operator" in text
    assert "clients/acme-merger-diligence" in text


def test_relativise_home_keeps_every_segment_below_home() -> None:
    home = str(Path.home())
    assert relativise_home(home) == "~"
    assert relativise_home(f"{home}/a/b/c") == os.path.join("~", "a", "b", "c")
    # A path that is not under home is untouched: /usr/local is not identifying.
    assert relativise_home("/usr/local/bin/lop") == "/usr/local/bin/lop"
    assert relativise_home("") == ""


def test_export_of_an_empty_snapshot_renders_and_leaks_nothing() -> None:
    """The all-defaults case still has to produce a paste-able document."""
    text = build_export(InfoSnapshot())
    assert "local-operator unknown" in text
    assert "## Install" in text and "## Environment" in text
    assert str(Path.home()) not in text


def test_export_states_the_cross_session_boundary() -> None:
    """A tree in the output must not read as a fleet-wide view."""
    text = build_export(_snapshot())
    assert "this session only" in text
    assert "other sessions report busy/pending and memory only" in text


def test_export_never_claims_up_to_date_from_an_unknown_latest() -> None:
    text = build_export(_snapshot())
    assert "unknown (never checked)" in text
    assert "up to date" not in text
