"""Deliberately failing probe for the merge gate.

Throwaway branch only: the merge gate must refuse to merge this PR while the
full suite is red, and must name ``Merge gate (full suite)`` as the blocking
check. Never merged.
"""


def test_the_merge_gate_refuses_a_red_full_suite() -> None:
    assert False, "probe: intentional failure for the merge-gate refusal test"
