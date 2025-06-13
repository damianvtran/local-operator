import re
from typing import Tuple

from langchain_core.messages import BaseMessage

from local_operator.agents import AgentData
from local_operator.executor import LocalCodeExecutor
from local_operator.types import ConversationRecord, ConversationRole

GENDER_CLASSIFICATION_PROMPT = """
You are tasked with classifying the gender of an AI agent based on its name and description. This classification will be used to select an appropriate voice for text-to-speech generation.

Instructions:
1. Analyze the agent's name and description carefully
2. Determine if the agent is intended to be perceived as male or female
3. Consider cultural naming conventions, pronouns used in the description, and any explicit gender indicators
4. If the gender is ambiguous or unclear, default to "male"
5. You must respond with exactly one of two values: "male" or "female"
6. Format your response using the exact XML schema shown below

Required Response Format:
<gender>male</gender>
OR
<gender>female</gender>

Agent Information:
<agent_name>{agent_name}</agent_name>
<agent_description>{agent_description}</agent_description>

Respond now with the gender classification in the required XML format:
"""  # noqa: E501


def parse_gender_from_xml(xml_string: str) -> str:
    """
    Parses the gender from an XML string.

    Args:
        xml_string: The XML string containing the gender.

    Returns:
        The gender string ("male" or "female").
    """
    # Use regex to find <gender>...</gender> tags
    match = re.search(r"<gender>(.*?)</gender>", xml_string, re.DOTALL)
    if match:
        gender = match.group(1).strip().lower()
        if gender in ["male", "female"]:
            return gender
    return "male"  # Default gender


async def determine_voice_and_instructions(
    agent: AgentData, executor: LocalCodeExecutor
) -> Tuple[str, str]:
    """
    Determines the voice and instructions for speech generation based on the agent's gender.

    Args:
        agent: The agent data.
        executor: The LocalCodeExecutor instance.

    Returns:
        A tuple containing the voice, instructions, and input text.
    """
    prompt = GENDER_CLASSIFICATION_PROMPT.format(
        agent_name=agent.name, agent_description=agent.description
    )
    messages = [ConversationRecord(role=ConversationRole.USER, content=prompt)]
    response: BaseMessage = await executor.invoke_model(messages)
    gender = parse_gender_from_xml(str(response.content))

    voice = "nova" if gender == "female" else "ash"
    instructions = f"You are {agent.name}, {agent.description if agent.description else 'a helpful assistant'}.  Speak aloud and pay attention to potentially multilingual inputs and make sure to use native accents for all different parts of the text, especially those that are not english.  Strive for a casual and native-sounding conversational tone.  Don't over-enunciate, consider word combinations that should have silent and natural transitions, like \"raha hoon\" -> \"rahoon\" or \"je m'appelle\" -> \"jm'appelle\"."  # noqa: E501

    return voice, instructions
