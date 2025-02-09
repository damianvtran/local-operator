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
    return EmbeddingManager(file_path=temp_dir, model_name="all-MiniLM-L6-v2")


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


@pytest.mark.parametrize(
    "query, insights, max_distance, expected_count",
    [
        (
            "Tell me about quantum physics",
            [
                "Baking cookies requires flour, sugar and butter",
                "Soccer is played with 11 players per team",
                "The Eiffel Tower is in Paris, France",
            ],
            1.5,
            0,
        ),
        (
            "How do computers work?",
            [
                "CPUs process instructions using logic gates",
                "Memory stores data temporarily",
                "Giraffes have long necks",
                "Lions hunt in packs",
            ],
            1.5,
            2,
        ),
        (
            "artificial intelligence",
            ["Machine learning models process data", "Neural networks are inspired by brains"],
            0.1,
            0,
        ),
        (
            "programming languages syntax",
            [
                "Python uses indentation for blocks",
                "Java requires semicolons",
                "Natural languages evolve over time",
                "Birds migrate south for winter",
            ],
            1.5,
            2,
        ),
    ],
)
def test_max_distance_filtering(embedding_manager, query, insights, max_distance, expected_count):
    # Add all test insights
    for insight in insights:
        embedding_manager.add_insight(insight)

    # Query with specified max_distance
    results = embedding_manager.query_insight(query, max_distance=max_distance)

    # Verify we get expected number of results
    assert (
        len(results) == expected_count
    ), f"Expected {expected_count} results but got {len(results)}"

    # Verify all returned results are within max_distance
    for result in results:
        assert result.distance <= max_distance


def test_add_large_text_basic(embedding_manager):
    text = (
        "This is a large text document. It contains multiple sentences. "
        "Here is another sentence. And one more for good measure."
    )
    embedding_manager.add_large_text(text, chunk_size=100, overlap=20)

    # Query the embedding manager for chunks containing "multiple sentences"
    results = embedding_manager.query_insight("multiple sentences", max_distance=1.0)
    assert len(results) == 1, f"Expected exactly 1 result, but found {len(results)}."
    insight_chunk = results[0].insight
    assert (
        "multiple sentences" in insight_chunk
    ), "Chunk does not contain expected text 'multiple sentences'."
    assert (
        "another sentence" in insight_chunk
    ), "Chunk does not contain expected text 'another sentence'."


def test_add_large_text_code_block_chunking(embedding_manager):
    """
    Test that multiple code blocks in a large text input are preserved intact when chunked.
    """
    text = (
        "Here is a description of the functionality. "
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
        "This should determine how code blocks are handled. "
        "```javascript\n"
        "function subtract(a, b) {\n"
        "    return a - b;\n"
        "}\n"
        "```\n"
        "End of code blocks."
    )
    # Use a small chunk size to force splitting and to check that code blocks stay intact.
    embedding_manager.add_large_text(text, chunk_size=80, overlap=20)

    # Find a chunk that contains the Python code block marker.
    python_chunk = next(
        (chunk for chunk in embedding_manager.metadata if "```python" in chunk), None
    )
    assert python_chunk is not None, "Python code block was not preserved in any chunk"
    # Verify that the Python code block content remains intact.
    assert (
        "def add(a, b):" in python_chunk
    ), "Python code block content missing the function definition"
    assert "return a + b" in python_chunk, "Python code block content missing the return statement"

    # Find a chunk that contains the JavaScript code block marker.
    js_chunk = next(
        (chunk for chunk in embedding_manager.metadata if "```javascript" in chunk), None
    )
    assert js_chunk is not None, "JavaScript code block was not preserved in any chunk"
    # Verify that the JavaScript code block content remains intact.
    assert (
        "function subtract(a, b)" in js_chunk
    ), "JavaScript code block missing the function definition"
    assert "return a - b" in js_chunk, "JavaScript code block missing the return statement"


def test_add_large_text_chunking(embedding_manager):
    # Create long text blocks without newlines to force mid-sentence chunking
    text = (
        "Section 1: This section talks about machine learning and artificial intelligence. "
        "Deep learning models have revolutionized many fields including computer vision and NLP. "
        "Neural networks can learn complex patterns from large amounts of data. "
        "Section 2: Here we discuss database systems and data storage. "
        "Relational databases use SQL for querying structured data. "
        "NoSQL databases provide more flexibility for unstructured data storage. "
        "Section 3: Software engineering best practices are important. "
        "Code reviews help maintain quality and share knowledge. "
        "Automated testing ensures reliability of software systems. "
        "Section 4: Cloud computing has transformed infrastructure. "
        "Services like AWS and Azure provide scalable solutions. "
        "Containerization with Docker simplifies deployment."
    )

    # Use smaller chunk size to avoid splitting important phrases
    embedding_manager.add_large_text(text, chunk_size=100, overlap=50)

    assert len(embedding_manager.metadata) >= 2

    # Test that we can find content from different sections
    ml_results = embedding_manager.query_insight("machine learning neural networks", k=5)
    assert len(ml_results) >= 1
    assert any("machine learning" in r.insight.lower() for r in ml_results)
    assert any("neural networks" in r.insight.lower() for r in ml_results)

    db_results = embedding_manager.query_insight("database SQL NoSQL", k=5)
    assert len(db_results) >= 1
    assert any("sql" in r.insight.lower() for r in db_results)
    assert any("nosql" in r.insight.lower() for r in db_results)

    se_results = embedding_manager.query_insight(
        "software engineering testing", k=5, max_distance=5
    )
    assert len(se_results) >= 1
    assert any("engineering" in r.insight.lower() for r in se_results)
    assert any("testing" in r.insight.lower() for r in se_results)

    cloud_results = embedding_manager.query_insight("cloud computing AWS", k=5, max_distance=5)
    assert len(cloud_results) >= 1
    assert any("cloud computing" in r.insight.lower() for r in cloud_results)
    assert any("aws" in r.insight.lower() for r in cloud_results)

    # Verify that overlapping chunks allow finding content that spans chunk boundaries
    spanning_results = embedding_manager.query_insight("databases provide flexibility", k=5)
    assert len(spanning_results) >= 1
    assert any(
        "databases" in r.insight.lower() and "flexibility" in r.insight.lower()
        for r in spanning_results
    )


def test_add_large_text_lorem_ipsum(embedding_manager):
    # Test with a long Lorem Ipsum text to verify chunking behavior
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
        "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
        "eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt "
        "in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis "
        "unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, "
        "totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto "
        "beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit "
        "aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione "
        "voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor "
        "sit amet, consectetur, adipisci velit."
    )

    # Use small chunk size to test chunking
    embedding_manager.add_large_text(text, chunk_size=100, overlap=50)

    assert len(embedding_manager.metadata) >= 5  # Should create multiple chunks

    # Test finding content from different parts of the text
    start_results = embedding_manager.query_insight("Lorem ipsum dolor sit amet", k=3)
    assert len(start_results) >= 1
    assert any("lorem ipsum" in r.insight.lower() for r in start_results)

    middle_results = embedding_manager.query_insight("voluptatem accusantium doloremque", k=3)
    assert len(middle_results) >= 1
    assert any("voluptatem" in r.insight.lower() for r in middle_results)

    # Test finding content that might span chunk boundaries
    spanning_results = embedding_manager.query_insight("dolor sit amet consectetur", k=3)
    assert len(spanning_results) >= 1
    assert any(
        "dolor" in r.insight.lower() and "consectetur" in r.insight.lower()
        for r in spanning_results
    )


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
