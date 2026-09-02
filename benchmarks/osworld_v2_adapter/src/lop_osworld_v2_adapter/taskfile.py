"""Static parsing of OSWorld 2.0 task classes into a TaskDescriptor.

OSWorld V2 tasks are Python modules, not JSON: each defines a ``BaseTask``
subclass whose *class attributes* carry the task fields (``id``,
``instruction``, ``config``, ``evaluator``, ``related_apps``, ``proxy``,
``image``, ``instance_type``, ``volume_size``, ``platform``,
``user_simulator``). ``task_base.py:12-29`` declares them as plain class
attributes, so they are statically resolvable in the overwhelming majority of
cases.

There are two tiers, and the tier boundary is the security decision:

- **Tier 1 (static, this module):** parse the source with ``ast`` and read
  literal assignments. No task code executes. Every decision made *before* any
  resource exists — requirements, provisioning, the cleanup plan — is made
  from Tier 1 alone. ``inspect_requirements`` and ``prepare`` must never
  import a task module.
- **Tier 2 (import, deferred to ``reset_start``/``score``):** the live module
  object, needed only when OSWorld's own machinery requires the class
  (``setup()``, a custom ``evaluate()`` override). By then the cleanup plan is
  persisted, so executing task code cannot leak an unnamed resource.

The descriptor carries ``source_sha256`` over the exact bytes parsed. That is
what ``EpisodeSpec.task_digest`` binds to: an episode's evidence must pin the
exact task bytes that were run, not a filename that could be repointed.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

# Fields read from class-level assignments on the task class. ``config`` and
# ``evaluator`` are the nested structures every OSWorld task carries; the rest
# are V2 additions that a task may omit (defaults applied at resolve time).
_STATIC_FIELDS = (
    "id",
    "instruction",
    "config",
    "evaluator",
    "related_apps",
    "proxy",
    "image",
    "instance_type",
    "volume_size",
    "platform",
    "user_simulator",
    "disable_vnc",
    "disable_recording",
    "intermediate_eval_safe",
)


class TaskParseError(ValueError):
    """A task module could not be statically resolved to a descriptor.

    Raised rather than falling back to import: failing preflight is cheap,
    executing unknown code to decide whether to spend money is not. A task
    whose fields are not statically resolvable is reported as a requirement
    the adapter cannot satisfy, never silently executed.
    """


@dataclass(frozen=True)
class TaskDescriptor:
    """The statically-derived facts about one OSWorld V2 task.

    ``config`` and ``evaluator`` are kept as their raw literal values (list /
    dict) because their shape is OSWorld's own and the adapter never
    interprets beyond the specific keys it derives requirements from.
    """

    task_id: str
    instruction: str
    config: tuple[Any, ...] = ()
    evaluator: Any = None
    related_apps: tuple[str, ...] = ()
    proxy: bool = False
    image: str | None = None
    instance_type: str | None = None
    volume_size: int | None = None
    platform: str | None = None
    user_simulator: Any = None
    disable_vnc: bool = False
    disable_recording: bool = False
    intermediate_eval_safe: bool = False
    # Whether the task class defines its own ``evaluate(self, env)``. This is
    # how EVERY task in the pinned V2 corpus scores (108 of 108 override it;
    # none declares an ``evaluator`` dict), and ``DesktopEnv.evaluate`` calls
    # the override in preference to the dict (desktop_env.py:584-587). A
    # parser that only recorded the dict would judge the whole corpus as
    # unscorable and refuse every paid episode at ``score``.
    evaluate_override: bool = False
    source_sha256: str = ""
    # The raw module source, kept so requirement derivation can detect
    # controller references (gitlab/website imports) without re-reading the
    # file. Task code never executes to produce this; it is the parsed bytes.
    source_text: str = ""
    # The workspace-relative path to the module, used by Tier 2 to locate it
    # for import. Kept relative so the descriptor is content-bound and
    # location-relative at once.
    module_name: str = ""

    def has_evaluator(self) -> bool:
        """Whether OSWorld's own ``evaluate()`` has anything to run.

        A task with neither an ``evaluator`` dict nor a custom override is one
        OSWorld would score 0.0 via ``logger.error`` — the exact
        score-deflation error ``scoring.score_to_artifact`` exists to reject.
        """

        return bool(self.evaluator) or self.evaluate_override

    def is_infeasible(self) -> bool:
        """Whether this task grades a correct refusal rather than a completion.

        OSWorld marks these with ``evaluator.func == "infeasible"``: the task
        is impossible, and ``evaluate()`` awards 1.0 only when
        ``action_history[-1]`` is ``FAIL`` (desktop_env.py). Our runner returns
        on a ``finish`` batch WITHOUT calling ``execute``, so the adapter never
        sees the terminal action and cannot put ``FAIL`` into that history. An
        agent that correctly declared the task infeasible would therefore score
        0 — and synthesising the ``FAIL`` ourselves would report a claim the
        agent never made, which is score fraud rather than a workaround.

        So these tasks are REFUSED at ``reset_start``, before anything is
        allocated. Detection is on the evaluator's own field, so it follows the
        task file rather than a maintained list that would drift from it.
        """

        evaluator = self.evaluator
        if isinstance(evaluator, dict) and evaluator.get("func") == "infeasible":
            return True
        # A multi-phase or conjunctive evaluator can nest the marker; the
        # repr scan is deliberately broad because a MISSED infeasible task
        # scores an honest refusal as a failure, while a false positive only
        # refuses a task we would rather not run yet.
        return "'infeasible'" in repr(evaluator)


def _literal(node: ast.AST) -> Any:
    """Resolve a class-level assignment value to a plain Python literal.

    Only ``ast.literal_eval``-compatible nodes are accepted: constants, lists,
    dicts, tuples, booleans, None. Anything else — a call, a name, an f-string
    — means the field is not statically resolvable, which is a ``TaskParseError``
    rather than a reason to execute the module.
    """

    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError) as error:
        raise TaskParseError(
            f"task field is not statically resolvable: {ast.dump(node)[:200]}"
        ) from error


def load_static(source: bytes, *, module_name: str) -> TaskDescriptor:
    """Parse one task module's source bytes into a TaskDescriptor.

    Raises ``TaskParseError`` if no task class is found or a required field is
    not a literal. ``source_sha256`` is always over the raw bytes given, so a
    descriptor is content-bound regardless of where the bytes came from.
    """

    try:
        tree = ast.parse(source.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise TaskParseError(f"task module does not parse: {error}") from error

    # Find the class whose body assigns the task fields. We do not require a
    # specific class name because V2 names classes Task001, TaskChrome01, etc.;
    # we require only that exactly one class assigns ``instruction`` and ``id``.
    values: dict[str, Any] = {}
    found = False
    evaluate_override = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        body_assigns = [
            stmt for stmt in node.body if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        ]
        names = {
            target.id
            for stmt in body_assigns
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        if "instruction" not in names or "id" not in names:
            continue
        found = True
        for stmt in body_assigns:
            target = stmt.targets[0]
            assert isinstance(target, ast.Name)
            if target.id in _STATIC_FIELDS:
                values[target.id] = _literal(stmt.value)
        # Detected on the SAME class that carries the task fields, by AST,
        # so a helper module defining an unrelated ``evaluate`` cannot claim
        # the task is scorable. Statically observable and side-effect free.
        evaluate_override = any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "evaluate"
            for stmt in node.body
        )
        break
    if not found:
        raise TaskParseError("no task class with 'id' and 'instruction' assignments found")

    source_sha256 = hashlib.sha256(source).hexdigest()
    if "id" not in values or "instruction" not in values:
        raise TaskParseError("task class assigns neither 'id' nor 'instruction'")

    config = values.get("config", [])
    if not isinstance(config, (list, tuple)):
        raise TaskParseError("task 'config' is not a list")
    related_apps = values.get("related_apps", [])
    if not isinstance(related_apps, (list, tuple)):
        related_apps = []

    return TaskDescriptor(
        task_id=str(values["id"]),
        instruction=str(values["instruction"]),
        config=tuple(config),
        evaluator=values.get("evaluator"),
        related_apps=tuple(str(item) for item in related_apps),
        proxy=bool(values.get("proxy", False)),
        image=values.get("image"),
        instance_type=values.get("instance_type"),
        volume_size=values.get("volume_size"),
        platform=values.get("platform"),
        user_simulator=values.get("user_simulator"),
        disable_vnc=bool(values.get("disable_vnc", False)),
        disable_recording=bool(values.get("disable_recording", False)),
        intermediate_eval_safe=bool(values.get("intermediate_eval_safe", False)),
        evaluate_override=evaluate_override,
        source_sha256=source_sha256,
        source_text=source.decode("utf-8", errors="replace"),
        module_name=module_name,
    )


def load_imported(descriptor: TaskDescriptor) -> Any:
    """Tier 2: import the live task class. Only called from reset_start/score.

    This is the boundary where task code first executes in the worker. It must
    only ever be reached after ``prepare`` has persisted the cleanup plan,
    because an import that allocates a resource the descriptor does not name
    is the leak the two-stage persistence exists to prevent. The import is
    done by file location, never through the workspace's sys.path, so a task
    module cannot shadow an installed package.
    """

    import importlib.util

    if not descriptor.module_name:
        raise TaskParseError("descriptor carries no module_name for import")
    path = descriptor.module_name
    spec = importlib.util.spec_from_file_location(f"osworld_task_{descriptor.task_id}", path)
    if spec is None or spec.loader is None:
        raise TaskParseError(f"cannot build an import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
