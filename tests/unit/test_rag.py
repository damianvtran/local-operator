import textwrap
from pathlib import Path

import pytest

from local_operator.rag import EmbeddingManager


# Fixture to provide a temporary directory for the EmbeddingManager
@pytest.fixture
def temp_dir(tmp_path):
    return Path(tmp_path)


# Fixture to create an instance of EmbeddingManager using the temporary directory
@pytest.fixture
def embedding_manager(temp_dir):
    return EmbeddingManager(file_path=temp_dir)


# Helper function to normalize code blocks.
def normalize_code_block(code: str) -> str:
    return textwrap.dedent(code).strip()


def test_add_insight(embedding_manager):
    # Test adding an insight using the real transformer
    test_insight = "This is a test insight"
    embedding_manager.add_insight(test_insight)
    assert embedding_manager.index.ntotal == 1  # One vector should be in the index
    assert embedding_manager.metadata == [test_insight]


def test_query_insight(embedding_manager):
    # Add an insight first
    test_insight = "This is a query test insight"
    embedding_manager.add_insight(test_insight)

    # Query using the same text - should get a very close match
    results = embedding_manager.query_insight(test_insight, k=3)
    assert isinstance(results, list)
    assert len(results) >= 1
    # The first result should match our added insight with a very small distance
    first_result = results[0]
    assert first_result.insight == test_insight
    assert first_result.distance < 0.01  # Very close semantic match


@pytest.mark.parametrize(
    "query, haystack, needle",
    [
        (
            "Tell me about the best matching insight",
            ["First insight", "Best matching insight", "Another insight"],
            "Best matching insight",
        ),
        (
            "What's the fastest way to travel between cities?",
            [
                "Dogs are loyal and friendly pets",
                "High-speed rail connects major urban centers",
                "Photosynthesis converts sunlight into energy",
            ],
            "High-speed rail connects major urban centers",
        ),
    ],
)
def test_best_insight_among_multiple(embedding_manager, query, haystack, needle):
    # Add all insights from the haystack to the manager
    for ins in haystack:
        embedding_manager.add_insight(ins)

    # Query using the query text - should get closest semantic match to needle
    results = embedding_manager.query_insight(query, k=3)

    # Verify that the top result matches the expected insight
    assert results, "No results returned"
    best_result = results[0]
    assert best_result.insight == needle, f"Expected '{needle}', but got '{best_result.insight}'"


def test_add_large_text_basic(embedding_manager):
    text = """
    This is a large text document. It contains multiple sentences.
    Here is another sentence. And one more for good measure.
    """
    embedding_manager.add_large_text(text, chunk_size=100, overlap=20)

    # Query should return relevant chunks
    results = embedding_manager.query_insight("multiple sentences")
    assert len(results) == 2
    assert "multiple sentences" in results[0].insight
    assert "another sentence" in results[1].insight


def test_add_large_text_with_code_blocks(embedding_manager):
    text = """
    Here is some text about Python programming.
    ```python
    def hello():
        print("Hello world")
    ```
    Here is more text after the code block.
    ```python
    x = 1 + 1
    print(x)
    ```
    Final text section.
    """

    embedding_manager.add_large_text(text, chunk_size=200, overlap=50)

    # Verify text before code block was added as a chunk
    results = embedding_manager.query_insight("Python programming")
    assert len(results) > 0
    assert "Python programming" in results[0].insight

    # Verify first code block was preserved intact
    results = embedding_manager.query_insight("hello function")
    assert len(results) > 0
    assert '```python\n    def hello():\n        print("Hello world")\n```' in results[0].insight

    # Verify second code block was preserved intact
    results = embedding_manager.query_insight("How to add 1 and 1")
    assert len(results) > 0
    assert "```python\n    x = 1 + 1\n    print(x)\n```" in results[0].insight

    # Verify text after code blocks was added
    results = embedding_manager.query_insight("Final text")
    assert len(results) > 0
    assert "Final text section" in results[0].insight


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (0, 10),  # Invalid chunk size
        (100, -1),  # Invalid overlap
        (100, 200),  # Overlap larger than chunk size
    ],
)
def test_add_large_text_invalid_params(embedding_manager, chunk_size, overlap):
    text = "Some sample text for testing."
    with pytest.raises(ValueError):
        embedding_manager.add_large_text(text, chunk_size=chunk_size, overlap=overlap)


def test_add_large_text_empty(embedding_manager):
    with pytest.raises(ValueError):
        embedding_manager.add_large_text("")

    with pytest.raises(ValueError):
        embedding_manager.add_large_text(None)
