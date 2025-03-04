import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from local_operator.server import openapi


@pytest.fixture
def mock_app():
    app = MagicMock(spec=FastAPI)
    app.title = "Test API"
    app.version = "1.0.0"
    app.description = "Test API Description"
    app.routes = []
    app.openapi_tags = []
    app.servers = None
    return app


def test_generate_openapi_schema(mock_app):
    with patch(
        "local_operator.server.openapi.get_openapi", return_value={"test": "schema"}
    ) as mock_get_openapi:
        schema = openapi.generate_openapi_schema(mock_app)

        mock_get_openapi.assert_called_once_with(
            title=mock_app.title,
            version=mock_app.version,
            description=mock_app.description,
            routes=mock_app.routes,
            tags=mock_app.openapi_tags,
            servers=None,
        )
        assert schema == {"test": "schema"}


def test_save_openapi_schema_with_string_path(mock_app):
    schema = {"test": "schema"}

    with patch("local_operator.server.openapi.generate_openapi_schema", return_value=schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "openapi.json")

            openapi.save_openapi_schema(mock_app, output_path)

            assert os.path.exists(output_path)
            with open(output_path, "r") as f:
                saved_schema = json.load(f)
                assert saved_schema == schema


def test_save_openapi_schema_with_path_object(mock_app):
    schema = {"test": "schema"}

    with patch("local_operator.server.openapi.generate_openapi_schema", return_value=schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "openapi.json"

            openapi.save_openapi_schema(mock_app, output_path)

            assert output_path.exists()
            with open(output_path, "r") as f:
                saved_schema = json.load(f)
                assert saved_schema == schema


def test_save_openapi_schema_pretty_vs_compact(mock_app):
    schema = {"test": "schema", "nested": {"value": 123}}

    with patch("local_operator.server.openapi.generate_openapi_schema", return_value=schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            pretty_path = Path(temp_dir) / "pretty.json"
            compact_path = Path(temp_dir) / "compact.json"

            openapi.save_openapi_schema(mock_app, pretty_path, pretty=True)
            openapi.save_openapi_schema(mock_app, compact_path, pretty=False)

            with open(pretty_path, "r") as f:
                pretty_content = f.read()
            with open(compact_path, "r") as f:
                compact_content = f.read()

            assert len(pretty_content) > len(compact_content)
            assert json.loads(pretty_content) == json.loads(compact_content) == schema


def test_save_openapi_schema_creates_parent_dirs(mock_app):
    schema = {"test": "schema"}

    with patch("local_operator.server.openapi.generate_openapi_schema", return_value=schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "dirs"
            output_path = nested_dir / "openapi.json"

            openapi.save_openapi_schema(mock_app, output_path)

            assert output_path.exists()
            assert nested_dir.exists()


def test_save_openapi_schema_io_error(mock_app):
    with patch("local_operator.server.openapi.generate_openapi_schema", return_value={}):
        with patch("builtins.open", side_effect=IOError("Test IO Error")):
            with pytest.raises(IOError, match="Test IO Error"):
                openapi.save_openapi_schema(mock_app, "test.json")


def test_get_openapi_schema_path_default():
    path = openapi.get_openapi_schema_path()
    expected_path = Path.cwd() / "docs" / "openapi.json"
    assert path == expected_path


def test_get_openapi_schema_path_custom():
    custom_dir = Path("/custom/path")
    path = openapi.get_openapi_schema_path(custom_dir)
    expected_path = custom_dir / "openapi.json"
    assert path == expected_path
