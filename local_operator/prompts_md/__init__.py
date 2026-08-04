"""Markdown prompt templates for the new harness.

Templates are plain ``.md`` files loaded via ``importlib.resources`` and
rendered by :func:`local_operator.prompts_api.render_template` (a tiny
``{{var}}`` / ``{{#if}}`` / ``{{#each}}`` engine). Keeping prompt text out of
Python source is deliberate: the old ``prompts.py`` grew to 176 KB of string
constants where every prompt edit required a package release and nothing was
diffable or hot-reloadable.

Files:

- ``system.md`` — the system persona and standing instructions (stable block).
- ``compaction_summary.md`` — the structured compaction summary prompt used by
  the compaction engine.
"""
