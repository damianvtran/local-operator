"""The claude.ai artifact data store: the shared database behind a published artifact.

A claude.ai artifact page may declare a small document store (collections of
JSON documents with per-document versions) that every viewer reads and writes.
Claude Code exposes it as its ``ArtifactData`` tool; this package is the same
capability for Local Operator, kept separate from :mod:`local_operator.artifacts`
(which is the GENERATION job interface for images/video and unrelated).

- :mod:`local_operator.artifact_data.client` — slug decoding, path validation,
  the ``me`` alias, the wire call and its error mapping.
- :mod:`local_operator.tools.artifact_data_tool` — the model-facing tool.

Stdlib-only at this level so the tool builder's gate stays cheap.
"""
