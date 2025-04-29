from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model
from local_operator.operator import (
    Operator,
    OperatorType,
    process_classification_response,
)
from local_operator.prompts import RequestType
from local_operator.tools import ToolRegistry
from local_operator.types import (
    ActionType,
    RelativeEffortLevel,
    RequestClassification,
    ResponseJsonSchema,
)


@pytest.fixture
def mock_model_config():
    model_configuration = MagicMock()
    model_configuration.instance = AsyncMock()
    model_configuration.instance.ainvoke = AsyncMock()
    model_configuration.instance.astream = AsyncMock()
    yield model_configuration


@pytest.fixture
def env_config():
    return {
        "RADIENT_API_KEY": "test_key",
    }


@pytest.fixture
def executor(mock_model_config):
    executor = LocalCodeExecutor(model_configuration=mock_model_config)
    executor.agent_state.conversation = []
    executor.tool_registry = ToolRegistry()
    yield executor


@pytest.fixture
def cli_operator(mock_model_config, executor, env_config):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    agent_registry = MagicMock()
    agent_registry.list_agents = MagicMock(return_value=[])

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=mock_model_config,
        config_manager=config_manager,
        type=OperatorType.CLI,
        agent_registry=agent_registry,
        current_agent=None,
        env_config=env_config,
    )

    operator._get_input_with_history = MagicMock(return_value="noop")

    yield operator


def test_cli_operator_init(mock_model_config, executor, env_config):
    credential_manager = MagicMock()
    credential_manager.get_credential = MagicMock(return_value="test_key")

    config_manager = MagicMock()
    config_manager.get_config_value = MagicMock(return_value="test_value")

    agent_registry = MagicMock()
    agent_registry.list_agents = MagicMock(return_value=[])

    operator = Operator(
        executor=executor,
        credential_manager=credential_manager,
        model_configuration=mock_model_config,
        config_manager=config_manager,
        type=OperatorType.CLI,
        agent_registry=agent_registry,
        current_agent=None,
        env_config=env_config,
    )

    assert operator.model_configuration == mock_model_config
    assert operator.credential_manager == credential_manager
    assert operator.executor is not None


@pytest.mark.asyncio
async def test_cli_operator_chat(cli_operator, mock_model_config):
    mock_response = ResponseJsonSchema(
        response="I'm done",
        code="",
        content="",
        file_path="",
        mentioned_files=[],
        replacements=[],
        action=ActionType.DONE,
        learnings="",
    )

    # Patch all required methods at once to reduce nesting
    mock_classify_request = patch.object(
        cli_operator,
        "classify_request",
        return_value=RequestClassification(
            type=RequestType.CONVERSATION,
            planning_required=True,
            relative_effort=RelativeEffortLevel.MEDIUM,
        ),
    ).start()
    mock_generate_plan = patch.object(
        cli_operator, "generate_plan", return_value=MagicMock()
    ).start()
    mock_interpret_action_response = patch.object(
        cli_operator, "interpret_action_response", return_value=mock_response
    ).start()
    patch.object(cli_operator, "_agent_should_exit", return_value=True).start()
    patch("builtins.input", return_value="exit").start()

    # Set up the mock response
    async def mock_astream(*args, **kwargs):
        message = BaseMessage(content=mock_response.model_dump_json(), type="assistant")
        yield message

    mock_model_config.instance.astream = mock_astream

    # Execute the test
    await cli_operator.chat()

    # Assertions
    assert mock_classify_request.call_count == 1
    assert mock_generate_plan.call_count == 1
    assert mock_interpret_action_response.call_count == 1
    assert "I'm done" in cli_operator.executor.agent_state.conversation[-1].content

    # Clean up patches
    patch.stopall()


def test_agent_is_done(cli_operator):
    test_cases = [
        {"name": "None response", "response": None, "expected": False},
        {
            "name": "DONE action",
            "response": ResponseJsonSchema(
                response="",
                code="",
                content="",
                file_path="",
                mentioned_files=[],
                replacements=[],
                action=ActionType.DONE,
                learnings="",
            ),
            "expected": True,
        },
        {
            "name": "Other action",
            "response": ResponseJsonSchema(
                response="",
                code="",
                content="",
                file_path="",
                mentioned_files=[],
                replacements=[],
                action=ActionType.CODE,
                learnings="",
            ),
            "expected": False,
        },
    ]

    for test_case in test_cases:
        cli_operator._agent_should_exit = MagicMock(return_value=False)
        assert (
            cli_operator._agent_is_done(test_case["response"]) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"

        # Test with agent_should_exit returning True
        if test_case["response"] is not None:
            cli_operator._agent_should_exit = MagicMock(return_value=True)
            assert (
                cli_operator._agent_is_done(test_case["response"]) is True
            ), f"Failed test case: {test_case['name']} with agent_should_exit=True"


def test_agent_requires_user_input(cli_operator):
    test_cases = [
        {"name": "None response", "response": None, "expected": False},
        {
            "name": "ASK action",
            "response": ResponseJsonSchema(
                response="",
                code="",
                content="",
                file_path="",
                mentioned_files=[],
                replacements=[],
                action=ActionType.ASK,
                learnings="",
            ),
            "expected": True,
        },
        {
            "name": "Other action",
            "response": ResponseJsonSchema(
                response="",
                code="",
                content="",
                file_path="",
                mentioned_files=[],
                replacements=[],
                action=ActionType.DONE,
                learnings="",
            ),
            "expected": False,
        },
    ]

    for test_case in test_cases:
        assert (
            cli_operator._agent_requires_user_input(test_case["response"]) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"


def test_agent_should_exit(cli_operator):
    test_cases = [
        {"name": "None response", "response": None, "expected": False},
        {
            "name": "BYE action",
            "response": ResponseJsonSchema(
                response="",
                code="",
                content="",
                file_path="",
                mentioned_files=[],
                replacements=[],
                action=ActionType.BYE,
                learnings="",
            ),
            "expected": True,
        },
        {
            "name": "Other action",
            "response": ResponseJsonSchema(
                response="",
                code="",
                content="",
                file_path="",
                mentioned_files=[],
                replacements=[],
                action=ActionType.CODE,
                learnings="",
            ),
            "expected": False,
        },
    ]

    for test_case in test_cases:
        assert (
            cli_operator._agent_should_exit(test_case["response"]) == test_case["expected"]
        ), f"Failed test case: {test_case['name']}"


@pytest.mark.asyncio
async def test_operator_print_hello_world(cli_operator):
    """Test that operator correctly handles 'print hello world' command and output
    using ChatMock."""
    # Configure mock model
    mock_model_config = configure_model("test", "", MagicMock())

    mock_executor = LocalCodeExecutor(mock_model_config)
    mock_executor.tool_registry = ToolRegistry()
    cli_operator.executor = mock_executor
    cli_operator.model_configuration = mock_model_config

    # Execute command and get response
    _, final_response = await cli_operator.handle_user_input("print hello world")

    # Verify conversation history was updated
    assert len(cli_operator.executor.agent_state.conversation) > 0
    last_conversation_message = cli_operator.executor.agent_state.conversation[-1]

    assert last_conversation_message.content == final_response
    assert final_response == "I have printed 'Hello World' to the console."


@pytest.mark.parametrize(
    "response_content,expected",
    [
        pytest.param(
            "<type>software_development</type><planning_required>true</planning_required>"
            "<relative_effort>medium</relative_effort><subject_change>false</subject_change>",
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=True,
                relative_effort=RelativeEffortLevel.MEDIUM,
                subject_change=False,
            ),
            id="xml_tag_format",
        ),
        pytest.param(
            "<type>software_development</type>",
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=False,
                relative_effort=RelativeEffortLevel.LOW,
                subject_change=False,
            ),
            id="default_values_for_missing_tags",
        ),
        pytest.param(
            "Here is some text<type>software_development</type><planning_required>true"
            "</planning_required><relative_effort>medium</relative_effort>"
            "<subject_change>false</subject_change>  Here is some more text",
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=True,
                relative_effort=RelativeEffortLevel.MEDIUM,
                subject_change=False,
            ),
            id="text_with_embedded_tags",
        ),
        pytest.param(
            """
            Here is some text
            <type>software_development</type>
            <planning_required>true</planning_required>
            <relative_effort>medium</relative_effort>
            <subject_change>false</subject_change>
            Here is some more text
            """,
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=True,
                relative_effort=RelativeEffortLevel.MEDIUM,
                subject_change=False,
            ),
            id="multiline_text_with_tags",
        ),
        pytest.param(
            """
            Here is some text
            <type>
            software_development
            </type>
            <planning_required>
            true
            </planning_required>
            <relative_effort>
            medium
            </relative_effort>
            <subject_change>
            false
            </subject_change>
            Here is some more text
            """,
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=True,
                relative_effort=RelativeEffortLevel.MEDIUM,
                subject_change=False,
            ),
            id="multiline_tags",
        ),
        pytest.param(
            "<type>conversation</type><planning_required>false</planning_required>"
            "<relative_effort>low</relative_effort><subject_change>true</subject_change>",
            RequestClassification(
                type=RequestType.CONVERSATION,
                planning_required=False,
                relative_effort=RelativeEffortLevel.LOW,
                subject_change=True,
            ),
            id="conversation_type",
        ),
        pytest.param(
            "<type>CONVERSATION</type><planning_required>false</planning_required>"
            "<relative_effort>low</relative_effort><subject_change>true</subject_change>",
            RequestClassification(
                type=RequestType.CONVERSATION,
                planning_required=False,
                relative_effort=RelativeEffortLevel.LOW,
                subject_change=True,
            ),
            id="uppercase_type_value",
        ),
        pytest.param(
            "<type>software_development</type><planning_required>True</planning_required>"
            "<relative_effort>medium</relative_effort><subject_change>False</subject_change>",
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=True,
                relative_effort=RelativeEffortLevel.MEDIUM,
                subject_change=False,
            ),
            id="mixed_case_boolean_values",
        ),
        pytest.param(
            "I think this is a <type>software_development</type> request with "
            "<planning_required>true</planning_required> "
            "planning and <relative_effort>high</relative_effort> effort. "
            "<subject_change>false</subject_change>",
            RequestClassification(
                type=RequestType.SOFTWARE_DEVELOPMENT,
                planning_required=True,
                relative_effort=RelativeEffortLevel.HIGH,
                subject_change=False,
            ),
            id="partial_xml_tags_with_text",
        ),
    ],
)
def test_process_classification_response(response_content, expected):
    """Test that process_classification_response correctly parses different formats."""
    result = process_classification_response(response_content)

    assert result.type == expected.type
    assert result.planning_required == expected.planning_required
    assert result.relative_effort == expected.relative_effort
    assert result.subject_change == expected.subject_change


@pytest.mark.parametrize(
    "invalid_content,expect_throw",
    [
        pytest.param("Here is a response with no tags", False, id="no_tags"),
        pytest.param(
            "Here is a response with incomplete xml tags <type>software_development",
            False,
            id="incomplete_tags",
        ),
        pytest.param(
            "Here is a response missing the type field <planning_required>true</planning_required>",
            True,
            id="missing_type_field",
        ),
    ],
)
def test_process_classification_response_invalid(invalid_content, expect_throw):
    """Test that process_classification_response raises ValidationError for invalid inputs."""
    if expect_throw:
        with pytest.raises(ValidationError):
            process_classification_response(invalid_content)
    else:
        classification = process_classification_response(invalid_content)

        assert classification.type == RequestType.CONTINUE
        assert classification.planning_required is False
        assert classification.relative_effort == RelativeEffortLevel.LOW
        assert classification.subject_change is False
