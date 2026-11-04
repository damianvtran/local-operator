"""D7: meaningful counts use muted ink, not low-contrast decorative dim."""

from local_operator.tui import notifications_render as render


def test_count_uses_muted_not_decorative_dim_ink() -> None:
    text = render.section_header("Updates", 3, 68)
    offset = text.plain.index("3")
    styles = [span.style for span in text.spans if span.start <= offset < span.end]
    assert render.ink("muted") in styles
    assert render.ink("dim") not in styles
