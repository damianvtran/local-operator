"""Keep the model's download recipe safe for one-time provider operations."""

from pathlib import Path


def test_download_guidance_arms_before_final_confirmation() -> None:
    guide = (
        Path(__file__).resolve().parents[2] / "local_operator/guides/browser/GUIDE.md"
    ).read_text()
    section = guide.split("### Uploading local files to a page", 1)[0]
    # A late capture cannot undo the provider-side effect of the first click.
    assert "download selector=<final confirm button>" in section
    assert "before clicking" in section
    assert "future download event" in section
    assert "already dispatched or cancelled" in section
    assert "does not prove" in section
    assert "provider-side effect" in section
    assert "first, then `download`" not in section
