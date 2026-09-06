"""Presentation adapters over the one slash registry, never a command registry.

A native action is a request to present a destination, not an assertion that a
window closed, a clipboard changed or a destructive control was confirmed.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from local_operator.slash_commands import SLASH_COMMANDS
from local_operator.tui.autocomplete import SlashCommand

# These are execution handlers, not a copy of the catalogue. A new registry row
# must choose its desktop destination before it can be offered by this host.
OWNER_COMMANDS = frozenset(
    {
        "rename",
        "model",
        "effort",
        "fast",
        "context",
        "goal",
        "compact",
        "approvals",
        "team",
        "agent",
        "loop",
    }
)


def command_catalogue() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "aliases": list(spec.aliases),
            "arguments": spec.arguments.value,
            "echo": spec.echo,
            "consumes_prompt": spec.consumes_prompt,
            "destination": spec.desktop_destination,
            "execution": "owner" if spec.name in OWNER_COMMANDS else "native",
        }
        for spec in SLASH_COMMANDS
        if spec.desktop_destination
    ]


def native_action(spec: SlashCommand, session_id: str, args: str) -> dict[str, Any]:
    """Return a concrete, non-executing UI instruction with actionable form fields."""
    destination = spec.desktop_destination
    fields: list[dict[str, Any]] = []
    data: dict[str, Any] = {}
    endpoint = f"/v1/desktop/sessions/{session_id}"
    if spec.name in {"rename", "goal", "team", "agent", "model", "effort", "approvals", "loop"}:
        fields.append(
            {"name": "args", "kind": "text", "value": args, "required": spec.name != "goal"}
        )
        data["submit"] = {"method": "POST", "path": endpoint + "/commands", "command": spec.name}
        data["entities"] = endpoint + "/command-entities?command=" + spec.name
        if spec.name in {"model", "effort", "approvals"}:
            data["scope"] = "session"
            data["default_settings"] = "/v1/settings"
    if spec.name == "fast":
        fields.append({"name": "args", "kind": "choice", "choices": ["on", "off"], "value": args})
        data.update(
            premium_pricing=True,
            submit={"method": "POST", "path": endpoint + "/commands", "command": "fast"},
        )
    elif spec.name == "credential":
        fields.extend(
            [
                {"name": "key", "kind": "text", "required": True},
                {"name": "value", "kind": "secret", "required": True},
            ]
        )
        data.update(
            submit={"method": "POST", "path": endpoint + "/credentials"},
            actions=["list", "store", "forget"],
        )
    elif spec.name == "fork":
        fields.extend(
            [
                {"name": "message", "kind": "text", "value": args},
                {"name": "boundary", "kind": "choice", "choices": ["next_safe"]},
            ]
        )
        data["submit"] = {"method": "POST", "path": endpoint + "/fork"}
    elif spec.name == "btw":
        fields.append({"name": "text", "kind": "text", "value": args, "required": True})
        data.update(
            submit={"method": "POST", "path": endpoint + "/asides"},
            adopt_path=endpoint + "/asides/{aside_id}/adopt",
            off_record=True,
        )
    elif spec.name == "stop":
        fields.extend(
            [
                {"name": "targets", "kind": "sessions", "value": [session_id]},
                {"name": "confirmed", "kind": "boolean", "required": True},
            ]
        )
        data.update(
            source="/v1/desktop/sessions", submit={"method": "POST", "path": "/v1/desktop/stop"}
        )
    elif spec.name in {"resume", "new", "reload"}:
        if spec.name == "new":
            fields.append({"name": "cwd", "kind": "text", "required": True})
            data["submit"] = {"method": "POST", "path": "/v1/desktop/sessions"}
        data.update(
            source="/v1/desktop/sessions",
            selected=session_id if spec.name == "reload" else args,
            preserve_running=True,
        )
    elif spec.name == "clear":
        data.update(view_only=True, history_untouched=True)
    elif spec.name == "copy":
        data.update(
            source=endpoint + "/history", choices=["message", "code", "quote"], clipboard="native"
        )
    elif spec.name == "exit":
        data.update(unsaved_guard=True, detach_only=True)
    elif spec.name == "help":
        data["source"] = "/v1/desktop/commands"
    elif spec.name in {"provider", "accounts", "login", "logout"}:
        data.update(
            source="/v1/auth/providers",
            accounts="/v1/auth/status",
            login="/v1/auth/login",
            logout="/v1/auth/providers/{provider_id}/credentials",
            selection=args,
            confirmation=spec.name == "logout",
        )
    elif spec.name in {"settings", "search"}:
        data.update(source="/v1/settings", filter="web-search" if spec.name == "search" else args)
    elif spec.name == "failovers":
        data["source"] = endpoint + "/failovers"
    elif spec.name in {"usage", "skills", "analytics"}:
        data.update(
            source="/v1/desktop/"
            + spec.name
            + ("?session_id=" + session_id if spec.name == "skills" else ""),
            selection=args,
        )
    elif spec.name == "mcp":
        data.update(
            source=endpoint + "/mcp",
            selection=args,
            submit={"method": "POST", "path": endpoint + "/mcp"},
            downstream_authorization="unknown",
        )
    elif spec.name == "theme":
        data.update(scope="desktop", selection=args, palette_source="desktop")
    elif spec.name == "update":
        data.update(capabilities="/v1/capabilities", preserve_session=session_id)
    if spec.name == "team" and (args == "chart" or args.startswith("chart ")):
        name = args.partition(" ")[2]
        fields = [{"name": "name", "kind": "text", "value": name, "required": True}]
        data = {
            "mode": "chart",
            "source": endpoint
            + "/command-entities?command=team"
            + ("&name=" + quote(name, safe="") if name else ""),
        }
    elif spec.name == "approvals" and (args == "default" or args.startswith("default ")):
        fields = [
            {
                "name": "value",
                "kind": "choice",
                "choices": ["ask", "auto"],
                "value": args.partition(" ")[2],
            }
        ]
        data = {
            "scope": "default",
            "source": "/v1/settings",
            "submit": {"method": "PATCH", "path": "/v1/settings/tool_approval_mode"},
            "current_session_unchanged": True,
        }
    return {
        "kind": "native_action",
        "destination": destination,
        "session_id": session_id,
        "args": args,
        "fields": fields,
        "data": data,
    }
