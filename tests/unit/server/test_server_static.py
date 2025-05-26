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


@pytest.mark.asyncio
async def test_get_html_file_not_found(test_app_client):
    """Test the get_html endpoint with a non-existent file path."""
    response = await test_app_client.get("/v1/static/html?path=/nonexistent/path/index.html")
    assert response.status_code == 404
    data = response.json()
    assert "File not found" in data["detail"]


@pytest.mark.asyncio
async def test_get_html_not_a_file(test_app_client, tmp_path):
    """Test the get_html endpoint with a directory path."""
    # Create a temporary directory
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    response = await test_app_client.get(f"/v1/static/html?path={dir_path}")
    assert response.status_code == 400
    data = response.json()
    assert "Not a file" in data["detail"]


@pytest.mark.asyncio
async def test_get_html_not_an_html_file(test_app_client, tmp_path):
    """Test the get_html endpoint with a non-HTML file."""
    # Create a temporary text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("This is a test file")

    response = await test_app_client.get(f"/v1/static/html?path={text_file}")
    assert response.status_code == 400
    data = response.json()
    assert "not an allowed HTML type" in data["detail"]


@pytest.mark.asyncio
async def test_get_html_success(test_app_client, tmp_path):
    """Test the get_html endpoint with a valid HTML file."""
    # Create a temporary HTML file
    html_file = tmp_path / "test.html"
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a test HTML page.</p>
</body>
</html>"""

    html_file.write_text(html_content)

    response = await test_app_client.get(f"/v1/static/html?path={html_file}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.text == html_content


@pytest.mark.asyncio
async def test_get_html_xhtml_success(test_app_client, tmp_path):
    """Test the get_html endpoint with a valid XHTML file."""
    # Create a temporary XHTML file
    xhtml_file = tmp_path / "test.xhtml"
    xhtml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>Test XHTML Page</title>
</head>
<body>
    <h1>Hello XHTML World</h1>
    <p>This is a test XHTML page.</p>
</body>
</html>"""

    xhtml_file.write_text(xhtml_content)

    response = await test_app_client.get(f"/v1/static/html?path={xhtml_file}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xhtml+xml"
    assert response.text == xhtml_content
