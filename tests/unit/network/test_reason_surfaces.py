"""The guard on the SURFACES, not on the vocabulary: a reason may not reach a person raw.

WHY THIS FILE EXISTS, AND WHY IT IS NOT ANOTHER VOCABULARY TEST. Four consecutive
review rounds (9, 10, 11) and two QA rounds each found ONE MORE human surface still
printing a peer's raw ``reason``: `/network peers`, then `lop sessions --peer`
stderr, then `lop network doctor`, then the agent tool's own ``peers`` digest — and
Step 1 of round 11 turned up a fifth and sixth (the membership table's marker and the
sentence `lop network show` prints). Every fix was correct and the next surface was
always one nobody had looked at yet. The vocabulary guards that existed are the reason
each fix was correct; they are not the reason the next leak was found. What was
missing is a guard over the RENDERERS.

WHAT IT ASSERTS. Every place in the mesh's own modules where a ``reason``/``detail``
value is turned into TEXT — an f-string, a ``%``/``+`` concatenation, a ``str()``
coercion, a ``.format()``/``.join()`` — must either hand that value to one of the
shared glosses (``resume.peer_reason_words``, ``resume.doctor_detail_words``,
``resume.table_reason_words``) or be DECLARED below with the reason it is not a human
surface. A new un-glossed renderer fails this test the moment it is written; a
declaration that no longer describes a real read fails too (the comparison is a
multiset, both directions), so the table cannot rot into a list of dead entries.

WHAT IT CANNOT DO, STATED PLAINLY. It reads the source; it does not run the surfaces.
"Reaches a person" is not decidable statically, so the proxy is "turns the value into
text", and the boundary is drawn by the declarations below rather than by the
analysis — which is exactly why each declaration says WHY. A renderer that builds its
text somewhere else and passes the raw value along (a dataclass field, a payload key)
is invisible here by construction; those are covered by the behavioural tests in
``test_slash_network.py``/``test_tool.py`` and by the QA rounds that drive the real
binary. And the module set is derived from the source (every module that names
``local_operator.network``) rather than listed by hand, so a new mesh surface joins the
guard by importing the mesh.

WHY THE FIELD SET IS THREE NAMES. ``reason`` and ``detail`` are the two keys the mesh
writes; ``unreachable_reason`` is the attribute the sidebar's row carries. A read of an
EXCEPTION's ``.reason`` (``json.JSONDecodeError``) is not a mesh field and is excluded
by name.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import local_operator

#: The fields the mesh writes a machine token into, on their way to a surface.
_REASON_FIELDS = frozenset({"reason", "detail", "unreachable_reason"})

#: The one spellings of "say it in words" this project has. A read that is an argument
#: of one of these is glossed; every other read of the fields above must be declared.
_GLOSSES = frozenset({"peer_reason_words", "doctor_detail_words", "table_reason_words"})

#: ``(path, qualified function, field) -> (how many raw reads, why that is not a
#: person's problem)``. The count is part of the key on purpose: a second raw read
#: inside an already-declared function is a new site and must be looked at.
_DECLARED_RAW_READS: dict[tuple[str, str, str], tuple[int, str]] = {
    # ``lop prune``'s sweep vocabulary: these ARE the sentences that line prints
    # (``superseded, unreferenced``), authored in ``update.py`` for a reader.
    ("local_operator/cli.py", "_cleanup_row", "reason"): (1, "the prune vocabulary's own prose"),
    # ``lop wake``'s booking reason — likewise prose, and not a mesh value at all.
    ("local_operator/cli.py", "wake_command", "reason"): (2, "the wake booking's own prose"),
    # Assigned to the payload's ``code`` key, not to a line: the human half of that
    # refusal is ``message``.
    ("local_operator/network/cli.py", "_cmd_service", "reason"): (1, "a payload key, not a line"),
    # The relay's OWN sentence for a session op (``runtime joining``), printed verbatim
    # on purpose: the wire facts (``outcome``/``engaged``/``admitted``) are the booleans
    # beside it, and UX round 3 removed the second copy of those from these lines.
    ("local_operator/network/cli.py", "_cmd_sessions", "detail"): (
        4,
        "the relay's own sentence — the 4th read is the model receipt's detail, which "
        "says why a requested model was not applied",
    ),
    # The push's per-peer report: ``reason`` here is the CONFLICT's own sentence (the
    # peer composed it), printed verbatim so a person learns why a row was refused.
    # The refusal/conflict SENTENCES this module composes around the peer's own
    # reports (``_describe_rows`` builds ``kind 'name' (reason)`` for a person), and
    # ``_apply_agent``/``_apply_team`` read the recorded row's origin to name it.
    ("local_operator/network/definitions.py", "_describe_rows", "reason"): (
        1,
        "the peer's own reason word, rendered for a person",
    ),
    ("local_operator/network/cli.py", "_cmd_definitions", "reason"): (
        1,
        "the peer's own conflict sentence, printed verbatim",
    ),
    # The mobility slice's wire transport reads the ACK'S ``detail`` key - the
    # envelope field itself, never rendered to anyone: what a surface renders is the
    # ``message`` INSIDE it, and the codes beside it are what a front end branches on.
    (
        "local_operator/network/mobility.py",
        "LinkTransport.ask",
        "detail",
    ): (1, "the ack's own payload key, not a rendered reason"),
    # The refusing device's code and sentence, handed to ``refusal_from_pairing``, whose
    # code-to-sentence map is what a person reads (Q-R3-3).
    ("local_operator/network/cli.py", "_join_one", "detail"): (1, "input to the sentence map"),
    ("local_operator/network/cli.py", "_join_one", "reason"): (1, "input to the sentence map"),
    # The credential listing's `skipped` rows (review round 5, NIT 2). THIS VOCABULARY
    # IS THIS DEVICE'S OWN, which is why the token is shown rather than glossed or
    # mapped to prose: `pull_placement` BUILDS the list in-process (`busy`,
    # `malformed_document` — two literals in `credentials/client.py`), so no peer's
    # prose can ever reach it, unlike the mesh reasons the glosses exist for. The
    # remedy in that sentence is authored here, and the same token rides the `--json`
    # payload for a script.
    ("local_operator/network/cli.py", "_cmd_credentials", "reason"): (
        1,
        "the credential client's own two-token vocabulary, not a peer's prose",
    ),
    # Carried into the catalog row (a ``str()`` coercion, not a render). The row is
    # painted by ``network_cli._peer_line`` and the TUI panel, both through the gloss.
    ("local_operator/network/projection.py", "RelayPeerCatalog.peers", "reason"): (
        1,
        "carried into the row; its renderers are glossed",
    ),
    # The peer's own sentence for an engage op, surfaced as a ProjectionRefusal message.
    ("local_operator/network/projection.py", "RemoteOwner.engage", "detail"): (
        1,
        "the peer's own refusal sentence",
    ),
    # Frame fields copied into audit records and ``set_trust(reason=…)``: the audit
    # trail is the machine register (`lop network log` prints the record verbatim on
    # purpose, and its ``remedy`` field is the human one).
    ("local_operator/network/relay.py", "RelayServer._ctl_pair_confirm", "reason"): (
        1,
        "an audit-record field",
    ),
    ("local_operator/network/relay.py", "RelayServer._ctl_panic", "reason"): (
        1,
        "an audit-record field",
    ),
    ("local_operator/network/relay.py", "RelayServer._ctl_trust", "reason"): (
        1,
        "an audit-record field",
    ),
    ("local_operator/network/relay.py", "RelayServer._op_panic", "reason"): (
        1,
        "an audit-record field",
    ),
    # THIS device's OWN runtime's sentence, when a create's model choice was not taken
    # (``_set_model_on``), recorded as the ``detail`` of the local audit event
    # ``session.create.warm_failed``. It is not a peer's reason token — the mesh's
    # glosses have nothing to say about it — and it goes into an audit record rather
    # than onto a screen, so a fixed sentence here would lose the runtime's own words
    # from the one line an operator reads.
    ("local_operator/network/relay.py", "RelayServer._warm_after_create", "detail"): (
        1,
        "this device's own runtime sentence, recorded in the local audit log",
    ),
    # THE PEER'S OWN SENTENCES ABOUT THE CREATE IT JUST DID, carried as the reply's
    # declared ``detail`` and ``model.detail`` fields (review round 1, MAJOR 2 — the two
    # reads this declaration counts are the two fields that used to be DROPPED at this
    # boundary). Prose rather than a token from the mesh's glossed vocabulary: only the
    # hosting device knows why its runtime has not joined or why its model choice was
    # not taken. The fields BESIDE them are what a renderer branches on
    # (``warming``/``admitted``/``model.applied``), never these sentences, which is the
    # same rule the relay's own audit record follows.
    ("local_operator/server/routes/desktop_sessions.py", "create_session", "detail"): (
        2,
        "the peer's own sentences, carried as declared reply fields",
    ),
    # Carried into ``SessionRow.unreachable_reason`` / ``UnansweredPeer.reason``. Both
    # readers are glossed or silent by design: the sidebar tooltip goes through
    # ``peer_reason_words``, and the silent-peer HEADING deliberately paints no reason
    # (``SessionSidebar._silent_peer_tiers`` says why).
    ("local_operator/session/peer_rows.py", "_read", "reason"): (
        2,
        "carried; both readers gloss or hide it",
    ),
    # The credential/attention/tunnel/loop vocabularies. Adjacent surfaces, none of them
    # the mesh's: these reasons are authored for their own line and are not read here.
    ("local_operator/tui/app.py", "OperatorApp._credential_remote_flow", "reason"): (
        2,
        "not a mesh reason",
    ),
    ("local_operator/tui/app.py", "OperatorApp._poll_completion_attention", "reason"): (
        1,
        "not a mesh reason",
    ),
    ("local_operator/tui/app.py", "OperatorApp._poll_tunnel_park", "reason"): (
        1,
        "not a mesh reason",
    ),
    ("local_operator/tui/app.py", "OperatorApp._store_inline_credentials", "reason"): (
        1,
        "not a mesh reason",
    ),
    ("local_operator/tui/app.py", "_loop_status_line", "reason"): (1, "not a mesh reason"),
}

_ROOT = Path(local_operator.__file__).parent

#: A module is on the mesh's surface path when its source NAMES the mesh package. That
#: is an import edge for most of them and a string reference for the rest, and it is
#: derived rather than listed so a new surface joins by reaching for the mesh.
_MESH_REFERENCE = "local_operator.network"


def _field_read(node: ast.AST) -> str | None:
    """The field name if this node reads one of :data:`_REASON_FIELDS`, else ``None``."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "get" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value in _REASON_FIELDS:
                return str(first.value)
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and sl.value in _REASON_FIELDS:
            return str(sl.value)
    if isinstance(node, ast.Attribute) and node.attr in ("reason", "unreachable_reason"):
        return node.attr
    return None


def _text_ids(tree: ast.AST) -> set[int]:
    """Every node that sits inside an expression producing TEXT.

    THE PROXY FOR "A RENDERER". A ``reason`` read that is assigned to a field or a
    payload key is not here; one interpolated into an f-string, concatenated, coerced
    with ``str()`` or merged with ``.join()`` is. That is the shape all six leaks took,
    and it is the shape a new one would take.
    """
    inside: set[int] = set()
    for node in ast.walk(tree):
        text = (
            isinstance(node, ast.JoinedStr)
            or (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)))
            or (
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Attribute) and node.func.attr in ("format", "join"))
                    or (isinstance(node.func, ast.Name) and node.func.id in ("str", "join"))
                )
            )
        )
        if text:
            for sub in ast.walk(node):
                inside.add(id(sub))
    return inside


def _gloss_ids(tree: ast.AST) -> set[int]:
    """Every node passed to a gloss call, at any depth (``f"({gloss(x)})"`` counts)."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if name in _GLOSSES:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    for sub in ast.walk(arg):
                        out.add(id(sub))
    return out


def _enclosing(tree: ast.AST) -> dict[int, str]:
    """``id(node) -> qualified name of the outermost function containing it``."""
    out: dict[int, str] = {}
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            for sub in ast.walk(node):
                out.setdefault(id(sub), ".".join(stack))
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

    Visitor().visit(tree)
    return out


def _exception_names(tree: ast.AST) -> set[str]:
    return {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and node.name
    }


def _raw_reads() -> tuple[Counter[tuple[str, str, str]], int, int]:
    """``(undeclared-or-declared raw reads, mesh modules seen, reads glossed)``."""
    found: Counter[tuple[str, str, str]] = Counter()
    glossed = 0
    modules = 0
    for path in sorted(_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if _MESH_REFERENCE not in source:
            continue
        modules += 1
        tree = ast.parse(source)
        text = _text_ids(tree)
        gloss = _gloss_ids(tree)
        enclosing = _enclosing(tree)
        exceptions = _exception_names(tree)
        relative = str(path.relative_to(_ROOT.parent))
        for node in ast.walk(tree):
            field = _field_read(node)
            if not field or id(node) not in text:
                continue
            if id(node) in gloss:
                glossed += 1
                continue
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in exceptions
            ):
                # ``json.JSONDecodeError.reason`` — an exception attribute, not a
                # reason field this project writes.
                continue
            found[(relative, enclosing.get(id(node), "<module>"), field)] += 1
    return found, modules, glossed


def test_every_reason_read_on_a_mesh_surface_is_glossed_or_declared() -> None:
    """The guard round 11 asked for: enumerate the RENDERERS, not the vocabulary.

    Fails on a new un-glossed read (a new leak in the making) and on a declaration
    that no longer matches a real site (a table that has rotted into fiction). The
    count in each declaration is compared too, so a second raw read inside an
    already-declared function is a change and not a free pass.
    """
    found, _modules, _glossed = _raw_reads()
    declared = Counter({key: count for key, (count, _why) in _DECLARED_RAW_READS.items()})
    assert found == declared, {
        "new_or_moved_raw_reads": sorted((found - declared).items()),
        "declarations_with_no_read": sorted((declared - found).items()),
        "counts_changed": sorted(
            (key, declared[key], found[key])
            for key in set(declared) & set(found)
            if declared[key] != found[key]
        ),
    }
    for key, (_count, why) in _DECLARED_RAW_READS.items():
        assert why.strip(), key


def test_the_scan_reaches_the_tree_it_claims_to() -> None:
    """A fresh-eyes guard on the guard: a scan that found nothing would agree with a
    clean tree and with a broken walker alike."""
    found, modules, glossed = _raw_reads()
    # The mesh's own surfaces, all of which must be in scope.
    assert modules >= 20, modules
    # The glossed reads are the ones the four fixes put there (plus their siblings);
    # if the walker stops finding them it has stopped finding anything.
    assert glossed >= 5, glossed
    assert len(found) >= 15, sorted(found)
    # And the surfaces each fix belongs to are represented, so a module silently
    # dropping out of the scan cannot pass.
    paths = {key[0] for key in found}
    assert "local_operator/network/cli.py" in paths
    assert "local_operator/network/relay.py" in paths
    assert "local_operator/session/peer_rows.py" in paths
