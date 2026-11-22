"""Compare actual prior/current paint functions, including styled fragment geometry."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

from rich.cells import cell_len
from rich.style import Style

import local_operator
from local_operator.agent_profiles import AgentFact
from local_operator.i18n import text
from local_operator.i18n.messages import RenderedMessage
from local_operator.tui.widgets import settings_view as view

BASE = "e12cb57872faf05f74fcef66b926d286983ce32f"
PATH = "local_operator/tui/widgets/settings_view.py"
old_source = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"], text=True)
old_node = next(
    n
    for n in ast.parse(old_source).body
    if isinstance(n, ast.FunctionDef) and n.name == "_paint_facts_lines"
)
old_node.name = "_before_paint_facts_lines"
namespace = view.__dict__
exec(
    compile(ast.Module(body=[old_node], type_ignores=[]), "before-settings-view.py", "exec"),
    namespace,
)
before = namespace[old_node.name]
after = view._paint_facts_lines
muted, faint = Style(color="white"), Style(color="grey50")
original = view.agent_fact_record
rows = []


def serialized(lines):
    return [
        [
            [(fragment, str(style), cell_len(fragment)) for fragment, style in line],
            cell_len("".join(fragment for fragment, _ in line)),
        ]
        for line in lines
    ]


for locale in ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"):
    text.set_active_locale(locale)
    assert text.active_locale() == locale
    for expansion in ("actual", "long-kind", "long-all"):

        def record(fact):
            result = original(fact)
            if expansion == "long-all" or (expansion == "long-kind" and fact.code == "kind"):
                return RenderedMessage("⟦" + result.text * 4 + "⟧", result.source_locale)
            return result

        view.agent_fact_record = record
        for kind in ("role", "specialist"):
            k = AgentFact("kind", kind)
            p = AgentFact("source", "packaged")
            m = AgentFact("action_class", "proactive")
            s = AgentFact("scope", "session")
            opaque = AgentFact("model", "proactive · " + original(s).text)
            effort = AgentFact("effort", "custom-tier")
            cases = (
                (k, p, m, s, opaque, effort),
                (opaque, effort, k, p, m, s),
                (k, s, opaque, m, s, m),
                (m, s, m, s, k),
                (),
                (opaque,),
                "role · proactive · this conversation",
            )
            for case_index, facts in enumerate(cases):
                for room in (0, 1, 5, 15, 26, 27, 28, 40, 80, 120, 500):
                    a = serialized(before(facts, room, muted=muted, faint=faint))
                    b = serialized(after(facts, room, muted=muted, faint=faint))
                    assert a == b, (locale, expansion, kind, case_index, room, a, b)
                    rows.append([locale, expansion, kind, case_index, room, a])
view.agent_fact_record = original
encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
output = {
    "base": BASE,
    "source": local_operator.__file__,
    "cases": len(rows),
    "locales": 8,
    "expansions": ["actual", "long-kind", "long-all"],
    "equal": True,
    "painted_fragments_styles_geometry_sha256": hashlib.sha256(encoded).hexdigest(),
    "before_function_ast_sha256": hashlib.sha256(ast.dump(old_node).encode()).hexdigest(),
    "current_file_sha256": hashlib.sha256(Path(PATH).read_bytes()).hexdigest(),
    "limits": "Projection proof only; no new native/font/shaping or global-English certification.",
}
print(json.dumps(output, indent=2))
