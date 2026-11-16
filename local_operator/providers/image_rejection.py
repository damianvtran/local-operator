"""Survive a provider refusing ONE image: replace it with a note and retry once.

WHY THIS EXISTS. An image block a provider refuses lives in the conversation
history, so the refusal is not one failed request: it is the end of the turn
that discovered it (``session._degrade_if_image_rejected`` documents why that
handler cannot retry from inside the loop's end event), and then a sticky
whole-session strip of EVERY image. Observed 2026-11-11 on direct DeepSeek: a
benchmark episode read a damaged PNG with the ``read`` tool, the next request
earned ``invalid request (HTTP 400): .messages[139].image[0]: You have
uploaded an unsupported image``, and the episode ended at step 11 with
``provider-error``. Any TUI, exec or mobile turn on the same route ends the
same way.

The creation-time bound and the render-time repair in
:mod:`local_operator.imaging` now normalize the cause that was measured (a
structurally damaged PNG). This module is the net for every cause they cannot
predict: a provider that names the refused image gets that ONE image replaced
with a short text note and the request is sent once more, inside the client,
before anything has streamed — so the turn continues and no other image is
lost.

HOW THE IMAGE IS IDENTIFIED. Providers name the refused block by a JSON path
into the request body they were sent: DeepSeek says ``.messages[139].image[0]``
(the 0th image part of wire message 139, system messages included), Anthropic
and OpenAI say ``messages.3.content.1...``. The path is walked over the body
the client ACTUALLY sent (captured by the client, never rebuilt, so the
indices cannot drift), the image's base64 payload is read at that node, and the
request's :class:`ImageContent` carrying the same payload is the one replaced.
When no path resolves, a damaged PNG in the request is the culprit if there is
one, and a request carrying exactly one image is unambiguous. Otherwise the
error is re-raised unchanged and the session's existing degrade handles it.

STABLE IN HISTORY. The refused payload's digest is remembered per provider for
the life of the process, and every later request through any client has the
same block replaced with the same constant note before it is sent. So the
prompt-cache prefix changes once, at the refused image, and later requests
never fail on it again. The note is constant (no request id, no provider
wording) precisely so that it renders byte-identically every time. The
transcript is never touched: a provider that accepts the image still gets it
after a model switch, and ``/export`` keeps the original.

KNOWN LIMIT: the memory is per process. A resumed session re-sends the image
once, earns the refusal once, and is repaired by the same retry — one wasted
request per resume, never a failed turn.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from local_operator.harness.types import ChatRequest, ImageContent, Message, TextContent
from local_operator.providers.failover import (
    ProviderError,
    is_image_count_rejection,
    is_image_modality_rejection,
    is_image_rejection,
)

logger = logging.getLogger(__name__)

#: What the model sees where the refused image was. Model-facing, so English by
#: the i18n contract (AGENTS.md: model-facing prompts stay English), and
#: CONSTANT so every later request renders the same bytes at that position.
REJECTED_IMAGE_PLACEHOLDER = (
    "[Image not sent: the model provider rejected this image as unsupported or "
    "invalid, so it was replaced with this note. Its contents are not visible; "
    "re-capture or re-encode the image if it is still needed.]"
)

#: Bound on remembered refusals. A refusal is rare (one per damaged file a
#: session reads), so this never evicts in practice; it exists only so a
#: pathological process cannot grow the map without limit. Entries are a
#: provider id and a 64-character digest.
_REJECTED_MAX_ENTRIES = 4096
_REJECTED: OrderedDict[tuple[str, str], None] = OrderedDict()

#: One path segment: a key (``messages``, ``content``, ``image``) or an index.
_PATH_TOKEN = re.compile(r"[A-Za-z_]+|\d+")
#: The start of a body path in provider prose: ``.messages[139]``,
#: ``messages.3`` — the path runs until whitespace or the ``:`` before the prose.
_PATH_START = re.compile(r"\.?messages(?:\[\d+\]|\.\d+)[^\s:]*")


def _digest(data: str) -> str:
    return hashlib.sha256(data.encode("ascii", "replace")).hexdigest()


def remember_rejected_image(provider: str, data: str) -> None:
    """Record that ``provider`` refused the image whose base64 is ``data``."""
    key = (provider, _digest(data))
    _REJECTED[key] = None
    _REJECTED.move_to_end(key)
    while len(_REJECTED) > _REJECTED_MAX_ENTRIES:
        _REJECTED.popitem(last=False)


def forget_rejected_images() -> None:
    """Clear the memory (tests; a long-lived process has no reason to)."""
    _REJECTED.clear()


def _replace(request: ChatRequest, doomed: Callable[[ImageContent], bool]) -> ChatRequest | None:
    """``request`` with every image ``doomed`` accepts swapped for the note."""
    changed = False
    messages: list[Message] = []
    for message in request.messages:
        if not any(isinstance(b, ImageContent) and doomed(b) for b in message.content):
            messages.append(message)
            continue
        content = [
            (
                TextContent(text=REJECTED_IMAGE_PLACEHOLDER)
                if isinstance(block, ImageContent) and doomed(block)
                else block
            )
            for block in message.content
        ]
        messages.append(message.model_copy(update={"content": content}))
        changed = True
    return request.model_copy(update={"messages": messages}) if changed else None


def with_rejected_images_replaced(request: ChatRequest) -> ChatRequest:
    """Apply every remembered refusal for this provider to ``request``.

    Free when nothing was ever refused (the common case): no digest is
    computed unless this provider has at least one remembered refusal.
    """
    provider = request.model.provider
    if not any(key[0] == provider for key in _REJECTED):
        return request
    return _replace(request, lambda block: (provider, _digest(block.data)) in _REJECTED) or request


def _image_parts(node: Mapping[str, Any]) -> list[Any]:
    content = node.get("content")
    if not isinstance(content, list):
        return []
    return [
        part for part in content if isinstance(part, Mapping) and "image" in str(part.get("type"))
    ]


def _payload_of(node: Any) -> str | None:
    """The base64 payload of the single image at ``node``, if there is one."""
    if isinstance(node, str):
        if node.startswith("data:") and "," in node:
            return node.split(",", 1)[1]
        return None
    if isinstance(node, Mapping):
        # Anthropic ``source.data`` is the bare base64, not a data URL.
        if isinstance(node.get("data"), str) and node.get("type") == "base64":
            return node["data"]
        # OpenAI chat ``image_url.url``, Responses ``image_url`` (a string),
        # Anthropic ``source``.
        for key in ("image_url", "url", "source"):
            if key in node:
                found = _payload_of(node[key])
                if found is not None:
                    return found
        parts = _image_parts(node)
        if len(parts) == 1:
            return _payload_of(parts[0])
    return None


def _walk(body: Mapping[str, Any], path: str) -> str | None:
    """Follow a provider's body path to the refused image's base64 payload."""
    tokens = _PATH_TOKEN.findall(path)
    node: Any = body
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isdigit():
            if not isinstance(node, list) or int(token) >= len(node):
                # The provider's index does not exist in what was sent: the
                # path is not about this body, so name nothing rather than
                # guess from wherever the walk stopped.
                return None
            node = node[int(token)]
        elif isinstance(node, Mapping) and token in node:
            node = node[token]
        elif (
            isinstance(node, Mapping)
            and token == "image"
            and index + 1 < len(tokens)
            and tokens[index + 1].isdigit()
        ):
            # DeepSeek's ``.image[K]``: the K-th image PART of this message,
            # not a key — its parts are ``{"type": "image_url", ...}``.
            parts = _image_parts(node)
            position = int(tokens[index + 1])
            if position >= len(parts):
                return None
            node = parts[position]
            index += 1
        else:
            # ``image``/``source``/``base64`` past the part itself: the part
            # reached so far is the refused one.
            break
        index += 1
    return _payload_of(node)


def _named_payload(error: ProviderError, body: Mapping[str, Any] | None) -> str | None:
    if not body:
        return None
    for match in _PATH_START.finditer(error.message):
        payload = _walk(body, match.group(0).lstrip("."))
        if payload is not None:
            return payload
    return None


def _structurally_damaged(block: ImageContent) -> bool:
    import base64

    from local_operator.media import png_structure_fault

    try:
        raw = base64.b64decode(block.data, validate=True)
    except ValueError:
        return True
    return raw.startswith(b"\x89PNG") and png_structure_fault(raw, verify_crc=True) is not None


def replace_rejected_image(
    request: ChatRequest, error: BaseException, sent_body: Mapping[str, Any] | None
) -> ChatRequest | None:
    """The retry request for a per-image refusal, or ``None`` to re-raise.

    Only BLOCK refusals qualify. A count refusal ("Too much media") and a
    modality refusal ("this model takes no images") are properties of the
    provider, not of one block; replacing a single image cannot cure them and
    the session already has scoped handlers for both.
    """
    if not isinstance(error, ProviderError) or not is_image_rejection(error):
        return None
    if is_image_count_rejection(error) or is_image_modality_rejection(error):
        return None
    provider = request.model.provider
    images: Sequence[ImageContent] = [
        block for m in request.messages for block in m.content if isinstance(block, ImageContent)
    ]
    if not images:
        return None
    named = _named_payload(error, sent_body)
    if named is not None and any(block.data == named for block in images):
        doomed = {named}
        how = "named by the provider"
    else:
        doomed = {block.data for block in images if _structurally_damaged(block)}
        how = "structurally damaged"
        if not doomed and len(images) == 1:
            doomed = {images[0].data}
            how = "the only image in the request"
    if not doomed:
        return None
    for data in doomed:
        remember_rejected_image(provider, data)
    logger.warning(
        "provider %s rejected %d image(s) (%s); replaced with a note and retrying once",
        provider,
        len(doomed),
        how,
    )
    return _replace(request, lambda block: block.data in doomed)


async def stream_with_rejected_image_retry(
    request: ChatRequest,
    attempt: Callable[[ChatRequest, dict[str, Any]], AsyncIterator[Any]],
) -> AsyncIterator[Any]:
    """Run ``attempt``; on a per-image refusal before any event, retry ONCE.

    ``attempt(request, sent)`` is the client's single-shot stream; it stores
    the body it posts in ``sent["body"]`` so the provider's path can be walked
    over exactly what was sent. A refusal after an event has been yielded is
    never retried: the caller has already acted on output and a resend would
    duplicate it (in practice a 400 arrives on the status line, before any).
    Exactly one retry, because a second refusal names a different cause that
    this cannot see, and the session's degrade is the right owner of it.
    """
    request = with_rejected_images_replaced(request)
    # A cached ordinary-session refusal must not silently erase evidence in an
    # opt-in required-media request. Validate after that projection as well.
    request.validate_required_images()
    sent: dict[str, Any] = {}
    yielded = False
    try:
        async for event in attempt(request, sent):
            yielded = True
            yield event
        return
    except ProviderError as error:
        if yielded or request.required_image_count:
            # Required-media verification cannot accept an answer to a retry
            # that removed its evidence. Keep the original provider failure and
            # do not poison the ordinary-session rejection cache for this run.
            raise
        retry = replace_rejected_image(request, error, sent.get("body"))
        if retry is None:
            raise
    async for event in attempt(retry, {}):
        yield event
