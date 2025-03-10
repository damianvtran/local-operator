"""
Tests for the static file hosting endpoints of the FastAPI server.

This module contains tests for the static file hosting endpoints to ensure
they correctly serve image files and handle error cases appropriately.
"""

import pytest


@pytest.mark.asyncio
async def test_get_image_file_not_found(test_app_client):
    """Test the get_image endpoint with a non-existent file path."""
    response = await test_app_client.get("/v1/static/images?path=/nonexistent/path/image.jpg")
    assert response.status_code == 404
    data = response.json()
    assert "File not found" in data["detail"]


@pytest.mark.asyncio
async def test_get_image_not_a_file(test_app_client, tmp_path):
    """Test the get_image endpoint with a directory path."""
    # Create a temporary directory
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    response = await test_app_client.get(f"/v1/static/images?path={dir_path}")
    assert response.status_code == 400
    data = response.json()
    assert "Not a file" in data["detail"]


@pytest.mark.asyncio
async def test_get_image_not_an_image(test_app_client, tmp_path):
    """Test the get_image endpoint with a non-image file."""
    # Create a temporary text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("This is a test file")

    response = await test_app_client.get(f"/v1/static/images?path={text_file}")
    assert response.status_code == 400
    data = response.json()
    assert "not an allowed image type" in data["detail"]


@pytest.mark.asyncio
async def test_get_image_success(test_app_client, tmp_path):
    """Test the get_image endpoint with a valid image file."""
    # Create a temporary image file (just a small PNG)
    image_file = tmp_path / "test.png"

    # Create a minimal valid PNG file
    # PNG header (8 bytes) + IHDR chunk (25 bytes) + IEND chunk (12 bytes)
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        # IHDR chunk
        b"\x00\x00\x00\x0dIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND chunk
    )

    with open(image_file, "wb") as f:
        f.write(png_data)

    response = await test_app_client.get(f"/v1/static/images?path={image_file}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == png_data
