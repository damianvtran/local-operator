"""§7's block wording is written TWICE; this file pins the two copies together.

The wiring renders the block itself (``session_factory._classification_block``)
rather than calling the package's ``recommend.render_block``, and the reason is
sound: dropping what the prompt already contains needs to know what the embedder
selected and what the MCP catalogue already advertises, and only the wiring knows
either. The import-graph reason is documented there too — the turn path must not
import the optional package.

What was missing is anything that notices when the two copies drift. Each copy
was pinned by its own literal in its own suite, which is exactly the arrangement
under which two spellings of "ignore the rest" can both stay green. These
assertions compare the two renderers' OUTPUT for the same input, so an edit to
either wording fails here — in the package's suite, where the drift is cheapest
to fix — instead of silently changing what the model reads.
"""

from __future__ import annotations

from local_operator import session_factory
from local_operator.classification.recommend import Recommendation, render_block
from tests.unit.classification.support import candidate


def hooks() -> session_factory._KnowledgeHooks:
    """The wiring's own hooks object, at its dataclass defaults.

    Defaults are the point: ``classification_max_recommendations`` is the cap the
    block is cut to, and using the real default means the comparison exercises the
    wiring's cap rather than a number this test invented.
    """
    return session_factory._KnowledgeHooks()


def test_the_wirings_block_agrees_with_the_packages_renderer_line_for_line() -> None:
    resources = (
        candidate("minerva-deploy", resource_url="skill://minerva-deploy"),
        candidate("tunnel", kind="guide", resource_url="guide://tunnel"),
        candidate("hubspot", kind="mcp", resource_url="mcp://hubspot"),
    )
    assert session_factory._classification_block(
        hooks(), Recommendation(resources=resources), picked=(), catalogue=""
    ) == render_block(resources)


def test_both_renderers_agree_that_an_empty_recommendation_says_nothing() -> None:
    """``""`` is the contract ("inject nothing"), and both halves must mean it."""
    wiring = session_factory._classification_block(
        hooks(), Recommendation(resources=()), picked=(), catalogue=""
    )
    assert wiring == render_block(()) == ""


def test_a_single_resource_renders_identically_on_both_sides() -> None:
    one = (candidate("minerva-deploy", resource_url="skill://minerva-deploy"),)
    assert session_factory._classification_block(
        hooks(), Recommendation(resources=one), picked=(), catalogue=""
    ) == render_block(one)


def test_the_three_block_literals_are_the_same_strings_in_both_modules() -> None:
    """The same comparison at the LITERAL level, so a failure names the line.

    The whole-block comparisons above are the real guard; this one exists so that
    when a wording edit lands on one side only, the failing assertion says which
    of the three lines changed rather than printing two five-line blocks.
    """
    rendered = render_block([candidate("x", resource_url="skill://x")]).splitlines()
    assert rendered[0] == session_factory._RECOMMENDATION_BLOCK_OPEN
    assert rendered[1] == session_factory._RECOMMENDATION_BLOCK_PREAMBLE
    assert rendered[-1] == session_factory._RECOMMENDATION_BLOCK_CLOSE
