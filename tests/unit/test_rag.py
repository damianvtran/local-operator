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
    text = """
    This is a large text document. It contains multiple sentences.
    Here is another sentence. And one more for good measure.
    """
    embedding_manager.add_large_text(text, chunk_size=100, overlap=20)

    # Query should return relevant chunks
    results = embedding_manager.query_insight("multiple sentences")
    assert len(results) == 1
    assert "multiple sentences" in results[0].insight
    assert "another sentence" in results[0].insight


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
