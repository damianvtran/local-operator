"""``McpStartupOutcome``'s network-failure field and its grouping property.

The field exists so the front end can group on a FACT rather than parse prose.
The property is the question the toast's footer asks: "were ALL of these the
network?" — and getting it wrong in either direction is user-visible. False
negatives leave nine bare server names where one cause explains them; false
positives would label a config break as the user's connectivity.
"""

from __future__ import annotations

from local_operator.session.mcp_status import McpStartupOutcome


def test_the_field_defaults_to_empty_so_every_existing_construction_is_unchanged() -> None:
    outcome = McpStartupOutcome(configured=("github",), connected=("github",))
    assert outcome.network_failures == frozenset()
    assert outcome.all_failures_are_network is False


def test_every_failure_being_the_network_is_true() -> None:
    outcome = McpStartupOutcome(
        configured=("a", "b"),
        failures={"a": "network: cannot reach a.example", "b": "network: cannot resolve b.example"},
        network_failures=frozenset({"a", "b"}),
    )
    assert outcome.all_failures_are_network is True


def test_one_non_network_failure_makes_the_group_mixed() -> None:
    """One config-local failure in the group is enough.

    The footer's grouped form claims a SHARED cause; a single server that failed
    for its own reason (``command not found: gh``) would make that claim false,
    so the plain list — which names each server — is the honest rendering.
    """
    outcome = McpStartupOutcome(
        configured=("a", "b", "gh"),
        failures={
            "a": "network: cannot reach a.example",
            "b": "network: cannot resolve b.example",
            "gh": "command not found: gh",
        },
        network_failures=frozenset({"a", "b"}),
    )
    assert outcome.all_failures_are_network is False


def test_the_discovery_key_is_never_a_network_failure() -> None:
    """'discovery' means the CONFIG layer failed — no host explains it.

    It is recorded under a synthetic key rather than a server name, so a manager
    that reported it as connectivity would send the user to diagnose a link that
    is fine.
    """
    outcome = McpStartupOutcome(
        configured=(),
        failures={"discovery": "network: cannot reach mcp.example.com"},
        # The MESSAGE text is deliberately a network line: the property has to
        # decide from the recorded FACT, never by parsing the copy, and the
        # discovery key is not in the network set because no host explains a
        # config failure.
        network_failures=frozenset(),
    )
    assert outcome.all_failures_are_network is False


def test_an_empty_failure_map_is_never_labelled() -> None:
    outcome = McpStartupOutcome(configured=("a",), connected=("a",))
    assert outcome.failed is False
    assert outcome.all_failures_are_network is False


def test_the_network_set_is_a_subset_of_the_reported_failures() -> None:
    """The invariant the property leans on, asserted where it is declared."""
    outcome = McpStartupOutcome(
        configured=("a",),
        failures={"a": "network: cannot reach a.example"},
        network_failures=frozenset({"a"}),
    )
    assert outcome.network_failures <= frozenset(outcome.failures)
