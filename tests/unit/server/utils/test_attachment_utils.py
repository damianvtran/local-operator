import base64
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

# Module to be tested
from local_operator.server.utils import attachment_utils

# Minimal valid file content for various types
MINIMAL_PNG_CONTENT = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"  # noqa: E501
MINIMAL_JPEG_CONTENT = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x11\x11\x18!\x1e\x18\x1a\x1d(%\x18\x1c#\x1c\x1c $.' \"#\x1e\x1f\x1f+*9*+;\x00\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9"  # noqa: E501
MINIMAL_GIF_CONTENT = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"  # noqa: E501
MINIMAL_PDF_CONTENT = b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000058 00000 n\n0000000111 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF"  # noqa: E501
GENERIC_BINARY_CONTENT = b"\x01\x02\x03\x04\x05"


def to_data_url(mime_type: str, content: bytes) -> str:
    """Converts binary content and mime type to a base64 data URL."""
    encoded_content = base64.b64encode(content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_content}"


@pytest.fixture
def temp_uploads_dir(tmp_path: Path):
    """Fixture to create a temporary uploads directory and mock UPLOADS_DIR."""
    # Create a unique subdirectory within tmp_path for uploads
    # to simulate the structure and avoid conflicts if other tests use tmp_path root.
    mock_uploads_path = tmp_path / "test_uploads"
    mock_uploads_path.mkdir(parents=True, exist_ok=True)

    with patch.object(attachment_utils, "UPLOADS_DIR", mock_uploads_path):
        # Ensure the mocked directory is used by _ensure_uploads_dir_exists if it's called again
        # or if not, ensure it exists for the test.
        attachment_utils._ensure_uploads_dir_exists()
        yield mock_uploads_path
    # Cleanup is handled by tmp_path fixture automatically


class TestParseBase64DataUrl:
    def test_parse_valid_data_url(self):
        mime = "image/png"
        data_url = to_data_url(mime, MINIMAL_PNG_CONTENT)
        parsed_mime, parsed_data = attachment_utils.parse_base64_data_url(data_url)
        assert parsed_mime == mime
        assert parsed_data == MINIMAL_PNG_CONTENT

    def test_parse_invalid_format_no_base64_tag(self):
        with pytest.raises(HTTPException) as exc_info:
            attachment_utils.parse_base64_data_url("data:image/png,somedata")
        assert exc_info.value.status_code == 400
        assert "Invalid data URL format" in exc_info.value.detail

    def test_parse_invalid_format_no_data_prefix(self):
        with pytest.raises(HTTPException) as exc_info:
            attachment_utils.parse_base64_data_url("image/png;base64,somedata")
        assert exc_info.value.status_code == 400
        assert "Invalid data URL format" in exc_info.value.detail

    def test_parse_invalid_base64_data(self):
        data_url = "data:text/plain;base64,this is not valid base64!!!"
        with pytest.raises(HTTPException) as exc_info:
            attachment_utils.parse_base64_data_url(data_url)
        assert exc_info.value.status_code == 400
        assert "Invalid base64 data" in exc_info.value.detail

    def test_parse_base64_with_padding(self):
        # Example from RFC 4648: "Man" -> "TWFu"
        # "Ma" -> "TWE="
        # "M" -> "TQ=="
        content = b"Ma"
        expected_encoded = base64.b64encode(content).decode("utf-8")  # Should be "TWE="

        # Simulate data URL that might be missing padding (though b64encode adds it)
        # For this test, let's use a known string that requires padding
        # "Ma" base64 is "TWE=". If we provide "TWE", it should still decode.
        data_url_missing_padding = "data:text/plain;base64,TWE"  # Missing '='

        # parse_base64_data_url should handle this by adding padding
        mime, data = attachment_utils.parse_base64_data_url(data_url_missing_padding)
        assert mime == "text/plain"
        assert data == content

        data_url_correct_padding = f"data:text/plain;base64,{expected_encoded}"
        mime_correct, data_correct = attachment_utils.parse_base64_data_url(
            data_url_correct_padding
        )
        assert mime_correct == "text/plain"
        assert data_correct == content


class TestSaveBase64Attachment:
    @pytest.mark.parametrize(
        "mime_type, content, expected_extension_part",
        [
            ("image/png", MINIMAL_PNG_CONTENT, ".png"),
            ("image/jpeg", MINIMAL_JPEG_CONTENT, ".jpg"),
            ("image/gif", MINIMAL_GIF_CONTENT, ".gif"),
            ("application/pdf", MINIMAL_PDF_CONTENT, ".pdf"),
            ("application/octet-stream", GENERIC_BINARY_CONTENT, ".bin"),  # Unknown, should default
            ("text/plain", b"hello world", ".txt"),
        ],
    )
    def test_save_various_file_types(
        self, temp_uploads_dir: Path, mime_type: str, content: bytes, expected_extension_part: str
    ):
        data_url = to_data_url(mime_type, content)

        # Mock uuid.uuid4 to return a fixed UUID for predictable filenames
        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        with patch("uuid.uuid4", return_value=fixed_uuid):
            file_uri = attachment_utils.save_base64_attachment(data_url)

        assert file_uri.startswith("file://")
        file_path_str = file_uri[len("file://") :]
        file_path = Path(file_path_str)

        assert file_path.name == f"{fixed_uuid}{expected_extension_part}"
        assert file_path.parent == temp_uploads_dir
        assert file_path.exists()

        with open(file_path, "rb") as f:
            saved_content = f.read()
        assert saved_content == content

    def test_save_unknown_mime_type_defaults_to_bin(self, temp_uploads_dir: Path):
        data_url = to_data_url("application/x-custom-unknown", GENERIC_BINARY_CONTENT)
        fixed_uuid = uuid.UUID("abcdef98-1234-5678-1234-567812345678")
        with patch("uuid.uuid4", return_value=fixed_uuid):
            file_uri = attachment_utils.save_base64_attachment(data_url)

        file_path = Path(file_uri[len("file://") :])
        assert file_path.name == f"{fixed_uuid}.bin"
        assert file_path.exists()
        with open(file_path, "rb") as f:
            assert f.read() == GENERIC_BINARY_CONTENT

    def test_save_fails_if_cannot_write_file(self, temp_uploads_dir: Path):
        data_url = to_data_url("text/plain", b"test")
        # Make the directory non-writable (not straightforward to do
        # robustly cross-platform for a dir)
        # Instead, mock open to raise IOError
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(HTTPException) as exc_info:
                attachment_utils.save_base64_attachment(data_url)
        assert exc_info.value.status_code == 500
        assert "Failed to save attachment file" in exc_info.value.detail


@pytest.mark.asyncio
class TestProcessAttachments:
    async def test_process_attachments_mixed_urls(self, temp_uploads_dir: Path):
        png_data_url = to_data_url("image/png", MINIMAL_PNG_CONTENT)
        http_url = "http://example.com/image.jpg"

        fixed_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
        with patch("uuid.uuid4", return_value=fixed_uuid):
            processed_urls = await attachment_utils.process_attachments([png_data_url, http_url])

        assert len(processed_urls) == 2

        # Check the processed data URL
        expected_file_path = temp_uploads_dir / f"{fixed_uuid}.png"
        assert processed_urls[0] == expected_file_path.as_uri()
        assert expected_file_path.exists()
        with open(expected_file_path, "rb") as f:
            assert f.read() == MINIMAL_PNG_CONTENT

        # Check the regular URL
        assert processed_urls[1] == http_url

    async def test_process_attachments_only_data_urls(self, temp_uploads_dir: Path):
        png_data_url = to_data_url("image/png", MINIMAL_PNG_CONTENT)
        pdf_data_url = to_data_url("application/pdf", MINIMAL_PDF_CONTENT)

        uuids = [uuid.uuid4(), uuid.uuid4()]
        with patch("uuid.uuid4", side_effect=uuids):
            processed_urls = await attachment_utils.process_attachments(
                [png_data_url, pdf_data_url]
            )

        assert len(processed_urls) == 2

        expected_png_path = temp_uploads_dir / f"{uuids[0]}.png"
        assert processed_urls[0] == expected_png_path.as_uri()
        assert expected_png_path.exists()

        expected_pdf_path = temp_uploads_dir / f"{uuids[1]}.pdf"
        assert processed_urls[1] == expected_pdf_path.as_uri()
        assert expected_pdf_path.exists()

    async def test_process_attachments_only_regular_urls(self, temp_uploads_dir: Path):
        urls = ["http://example.com/file1", "https://othersite.org/file2.zip"]
        processed_urls = await attachment_utils.process_attachments(urls)
        assert processed_urls == urls  # Should remain unchanged
        # Ensure no files were created in uploads_dir
        assert not any(temp_uploads_dir.iterdir())

    async def test_process_attachments_empty_and_none(self):
        assert await attachment_utils.process_attachments([]) == []
        assert await attachment_utils.process_attachments(None) == []

    async def test_process_attachments_propagates_exception(self, temp_uploads_dir: Path):
        invalid_data_url = "data:text/plain;base64,not valid base64"
        with pytest.raises(HTTPException) as exc_info:
            await attachment_utils.process_attachments([invalid_data_url])
        assert exc_info.value.status_code == 400  # Propagated from parse_base64_data_url
        assert "Invalid base64 data" in exc_info.value.detail

    async def test_process_attachments_file_uri_passthrough(self, temp_uploads_dir: Path):
        file_uri = "file:///some/local/file.txt"
        processed_urls = await attachment_utils.process_attachments([file_uri])
        assert processed_urls == [file_uri]
        assert not any(temp_uploads_dir.iterdir())
