"""Pre-execution guardrails for tool calls (see ``check.py`` for the entry point).

Added after the 2026-11-04 incident in which an unattended, auto-approved
session's shell call removed the whole state root through an empty command
substitution. Approval tiers are moot under auto approval, so this package is a
SECOND pre-execution checkpoint that understands what a call would destroy.
"""

from local_operator.guardrails.check import GuardOutcome, check_tool_call

__all__ = ["GuardOutcome", "check_tool_call"]
