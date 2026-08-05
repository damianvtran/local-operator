"""Compaction engine: threshold triggers, cut-point selection, summarization
prompts, and cache-aware pruning.

Implements the context-full compaction strategy — see
``docs/REWRITE.md`` § C and ``docs/recon/ScoutSkillsCompact.md`` § 5-6 for
the contract. The stable consumer surface is :mod:`local_operator.compaction.api`;
the session imports it lazily inside functions so a missing compaction
package degrades to no-compaction rather than a crash.
"""

from .api import (
    CompactionResult,
    CompactionSettings,
    build_compaction_prompt,
    find_cut_point,
    prune_tool_outputs,
    summarize_messages,
)

__all__ = [
    "CompactionResult",
    "CompactionSettings",
    "build_compaction_prompt",
    "find_cut_point",
    "prune_tool_outputs",
    "summarize_messages",
]
