import pytest

from local_operator import rag


# Fixture to provide a temporary directory for the EmbeddingManager
@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


# Fixture to create an instance of EmbeddingManager using the temporary directory
@pytest.fixture
def embedding_manager(temp_dir):
    return rag.EmbeddingManager(file_path=temp_dir)


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


def test_rag_tool_fetch(embedding_manager):
    # Add some test insights
    test_insights = [
        "Python is a programming language",
        "Machine learning uses data to make predictions",
        "Natural language processing analyzes text",
    ]
    for insight in test_insights:
        embedding_manager.add_insight(insight)

    # Test querying with a semantically similar query
    query = "Tell me about programming with Python"
    results = rag.rag_tool_fetch(embedding_manager, query, k=2)

    assert len(results) == 2
    assert results[0].insight == test_insights[0]  # Should match Python programming insight
    assert results[0].distance < results[1].distance  # First result should be closest match
