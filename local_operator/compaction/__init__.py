"""Compaction engine: threshold triggers, cut-point selection, summarization
prompts, and cache-aware pruning.

Implements the context-full compaction strategy — see
``docs/REWRITE.md`` § C and ``docs/recon/ScoutSkillsCompact.md`` § 5-6 for
the contract. The stable consumer surface is :mod:`local_operator.compaction.api`;
the session imports it lazily inside functions so a missing compaction
package degrades to no-compaction rather than a crash.

:mod:`local_operator.compaction.pass_` composes the engine into one
host-independent pass (prune → trigger → cut → summarize → rebuild) for
hosts that cannot use the session's pass — the evaluation runner today.
"""

from .api import (
    CompactionResult,
    CompactionSettings,
    build_compaction_prompt,
    find_cut_point,
    prune_tool_outputs,
    summarize_messages,
)
from .pass_ import CompactionPassResult, run_compaction_pass
from .pruning import STALE_FRAME_NOTICE, prune_stale_frames

__all__ = [
    "CompactionPassResult",
    "CompactionResult",
    "CompactionSettings",
    "STALE_FRAME_NOTICE",
    "build_compaction_prompt",
    "find_cut_point",
    "prune_stale_frames",
    "prune_tool_outputs",
    "run_compaction_pass",
    "summarize_messages",
]
