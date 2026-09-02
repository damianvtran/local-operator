"""Fixture OSWorld V2 task modules, written to disk per test.

These mirror the real V2 ``BaseTask`` subclass shape: plain class-attribute
assignments over ``id``/``instruction``/``config``/``evaluator`` plus the V2
provisioning fields. They exist as STRINGS (not committed .py files) so the
tests that parse them exercise the real ``ast`` path against bytes the test
controls, and so a fixture can be adversarial (a non-literal field) without
needing a syntactically-valid-but-broken committed module.
"""

from __future__ import annotations

# A minimal, dependency-free task: the common case.
PLAIN = """
class Task001:
    id = "task_plain"
    instruction = "Open the text editor and write hello."
    config = [{"type": "launch", "app": "gedit"}]
    evaluator = {"func": "check_include", "result": {"type": "vm_file", "path": "/tmp/x"}}
    related_apps = ["gedit"]
"""

# proxy:true, which must pull in the proxy secret and endpoint.
PROXY = """
class Task002:
    id = "task_proxy"
    instruction = "Browse to a geo-restricted page."
    config = [{"type": "launch", "app": "chrome"}]
    evaluator = {"func": "check_url"}
    proxy = True
"""

# References the gitlab controller, which raises at import without its env var.
GITLAB = """
from desktop_env.controllers import gitlab

class Task003:
    id = "task_gitlab"
    instruction = "Merge the MR."
    config = [{"type": "launch", "app": "chrome"}]
    evaluator = {"func": "check_gitlab_state"}
"""

# References the website controller / WEBSITE_HOST_SUFFIX.
WEBSITE = """
import desktop_env.controllers.website as website

class Task004:
    id = "task_website"
    instruction = "Buy the item on the mocked shop."
    config = [{"type": "launch", "app": "chrome"}]
    evaluator = {"func": "check_cart"}
"""

# Imports the LLM judge client directly (the shape of the real task_008).
JUDGED = """
from desktop_env.evaluators.model_client import generate_text

class Task008:
    id = "task_judged"
    instruction = "Write a summary the judge will grade."
    config = [{"type": "launch", "app": "gedit"}]
    evaluator = {"func": "compare_text_with_llm"}
"""

# Reaches the judge through an llm_metrics metric rather than the client.
JUDGED_VIA_METRICS = """
from desktop_env.evaluators.metrics import llm_metrics

class Task009:
    id = "task_judged_metrics"
    instruction = "Edit the image the judge will compare."
    config = [{"type": "launch", "app": "gimp"}]
    evaluator = {"func": "compare_images_with_llm"}
"""

# A googledrive config entry needs Google account credentials.
GOOGLEDRIVE = """
class Task005:
    id = "task_gdrive"
    instruction = "Open the shared sheet."
    config = [{"type": "googledrive", "file_id": "abc123"}]
    evaluator = {"func": "check_cell"}
"""

# An LLM-backed user simulator needs an API key.
LLM_SIMULATOR = """
class Task006:
    id = "task_llmsim"
    instruction = "Ask the user for their name, then type it."
    config = [{"type": "launch", "app": "gedit"}]
    evaluator = {"func": "check_include"}
    user_simulator = {"type": "llm", "provider": "openai", "model": "gpt-4o"}
"""

# A scripted (non-LLM) user simulator: needs NO API key.
SCRIPTED_SIMULATOR = """
class Task007:
    id = "task_scriptedsim"
    instruction = "Confirm the dialog."
    config = [{"type": "launch", "app": "zenity"}]
    evaluator = {"func": "check_dialog"}
    user_simulator = {"type": "scripted", "responses": ["yes"]}
"""

# A relative-time evaluator needs the host to pin the episode clock.
CLOCK = """
class Task008:
    id = "task_clock"
    instruction = "Create an event tomorrow."
    config = [{"type": "launch", "app": "calendar"}]
    evaluator = {"func": "check_event", "rule_relativeTime": {"days": 1}}
"""

# Custom provisioning fields override the defaults.
CUSTOM_INSTANCE = """
class Task009:
    id = "task_custom"
    instruction = "Compile the kernel."
    config = [{"type": "execute", "command": "make"}]
    evaluator = {"func": "check_build"}
    image = "ami-0123456789abcdef0"
    instance_type = "t3.2xlarge"
    volume_size = 100
"""

# A task with NO evaluator: score() must raise, never return 0.0.
NO_EVALUATOR = """
class Task010:
    id = "task_noeval"
    instruction = "Just look around."
    config = [{"type": "launch", "app": "files"}]
"""

# The shape of EVERY task in the pinned V2 corpus: no ``evaluator`` dict, an
# ``evaluate(self, env)`` override on the task class instead.
EVALUATE_OVERRIDE = """
from desktop_env.task_base import BaseTask

class Task011(BaseTask):
    id = "task_override"
    instruction = "Create the report."
    config = [{"type": "launch", "app": "gedit"}]

    def evaluate(self, env):
        return 0.5
"""

# An ``evaluate`` defined OUTSIDE the task class must not count.
EVALUATE_ELSEWHERE = """
def evaluate(env):
    return 1.0

class Task012:
    id = "task_helper_only"
    instruction = "Nothing scores this."
    config = []
"""

# An infeasible-style task (V2 shape). Excluded from scoring support.
INFEASIBLE = """
class Task011:
    id = "task_infeasible"
    instruction = "Do something impossible."
    config = [{"type": "launch", "app": "files"}]
    evaluator = {"func": "infeasible"}
"""

# A field that is NOT statically resolvable: must raise TaskParseError, never
# execute the module to find out.
NON_LITERAL = """
import os

class Task012:
    id = "task_nonliteral"
    instruction = "Computed instruction."
    config = [{"type": "launch", "app": "files"}]
    evaluator = {"func": os.environ["HOME"]}
"""
