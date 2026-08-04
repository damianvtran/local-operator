"""Agent tools for local-operator.

Two generations live in this package:

- **New harness tools** (``builtin.py``, ``registry.py``) — ``AgentTool``
  implementations with pydantic JSON Schemas, executed via native provider
  tool calling. Entry point: ``local_operator.tools.registry.create_tools``.
- **Legacy tools** (``general.py``, ``google.py``, ``screen_recorder.py``) —
  the old executed-Python tool registry used by ``operator.py``/``executor.py``.
  Untouched and unregistered by the new harness; do not import them from new
  code.
"""
