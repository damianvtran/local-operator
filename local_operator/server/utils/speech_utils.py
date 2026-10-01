"""Voice selection helpers for the agent speech endpoint.

The gender classifier is a single non-agentic completion, so it runs through
``ServerExecutor.invoke_model`` (provider wire clients) rather than an agent
session — no tools, no transcript, no loop.

Two properties this module owns, both findings from the speak-aloud work:

* The classifier's answer depends only on the agent's ``(name, description)``,
  so it is cached for the process lifetime. The press path used to run a fresh
  completion on EVERY press, paying latency and model cost to re-answer the
  same question.
* The selected voice travels as an ALIAS (``"female"``/``"male"``), not as a
  provider voice name: the hub resolves the alias to a voice id, so a voice
  retirement is one change on the hub rather than an app release (the OpenAI
  voice names this module used to pick, ``nova``/``ash``, pin this app to a
  provider's voice roster). The persona ``instructions`` string the OpenAI
  path used to build is gone — ElevenLabs has no equivalent field — a small
  delivery-quality loss accepted in exchange for native multilingual
  pronunciation.
"""

import hashlib
import re
from collections import OrderedDict

from local_operator.agents import AgentData
from local_operator.server.utils.operator import ServerExecutor
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


#: Upper bound on cached classifications. One entry per (name, description)
#: pair ever spoken for; 256 is far more agents than a fleet asks about, and
#: the eviction below keeps a long-lived daemon from pinning every agent it
#: has ever seen.
_GENDER_CACHE_MAX = 256

#: The classification cache itself, insertion-ordered for LRU eviction: a
#: hit moves its key to the most-recent end, and the bound below drops the
#: front, which is the least recently USED entry. Process-wide on purpose:
#: the daemon is one process serving one user's agents, and the finding this
#: cache answers is per-press model calls.
_GENDER_CACHE: "OrderedDict[str, str]" = OrderedDict()


def _gender_cache_key(name: str, description: str) -> str:
    """The cache key for one agent's classification: sha256 over (name, description).

    Hashed rather than spelled because the description can be long and the
    cache must not pin every description it has seen; the NUL separator keeps
    ("ab", "c") from colliding with ("a", "bc").
    """
    return hashlib.sha256(f"{name}\x00{description}".encode("utf-8")).hexdigest()


async def determine_voice(agent: AgentData, executor: ServerExecutor) -> str:
    """Return the voice alias ("female"/"male") this agent's speech should use.

    A cache miss runs the one-shot classifier; a hit answers without a model
    call. A concurrent double-miss may duplicate one call — the answers agree,
    and serialising every press behind a lock costs more than it saves.

    Args:
        agent: The agent data.
        executor: The ServerExecutor providing the one-shot completion.

    Returns:
        The voice alias: exactly the classifier's answer, resolved to a
        provider voice id by the hub.
    """
    key = _gender_cache_key(agent.name, agent.description or "")
    cached = _GENDER_CACHE.get(key)
    if cached is not None:
        _GENDER_CACHE.move_to_end(key)
        return cached

    prompt = GENDER_CLASSIFICATION_PROMPT.format(
        agent_name=agent.name, agent_description=agent.description
    )
    messages = [ConversationRecord(role=ConversationRole.USER, content=prompt)]
    response = await executor.invoke_model(messages)
    gender = parse_gender_from_xml(str(response.content))

    _GENDER_CACHE[key] = gender
    if len(_GENDER_CACHE) > _GENDER_CACHE_MAX:
        _GENDER_CACHE.popitem(last=False)
    return gender
