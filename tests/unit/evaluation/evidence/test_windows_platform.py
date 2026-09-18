"""The evidence store on a platform with no ``fcntl`` (audit D5 / B19).

The defect was not that the store lacked a platform guard — it has a complete
one, ``EvidenceWriter._supported``, whose message already names the reason. The
defect was that ``import fcntl`` sat at module scope, so nothing on Windows ever
reached the guard: the import of this module failed first, and because
``evaluation.runner.episode`` imports it at module scope, the whole evaluation
runner was unimportable there.

So the assertions below are about ORDER: the module must import, and the
refusal a caller then meets must be the guard's own sentence. A blocked module
rather than a deleted attribute, because the import machinery is what has to
fail — a ``fcntl = None`` stand-in would not reproduce the
``ModuleNotFoundError`` the audit recorded.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

_BLOCK_FCNTL = """
import sys

class _NoFcntl:
    def find_spec(self, name, path=None, target=None):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'")
        return None

sys.meta_path.insert(0, _NoFcntl())
sys.modules.pop("fcntl", None)
"""


def _run(tmp_path: Path, statement: str) -> subprocess.CompletedProcess[str]:
    """Run ``statement`` in a fresh interpreter where ``fcntl`` cannot be imported."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(tmp_path / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        PYTHONPATH=str(REPO_ROOT),
    )
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_FCNTL + statement],
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )


def test_the_evidence_store_imports_where_fcntl_does_not_exist(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "from local_operator.evaluation.evidence.store import EvidenceWriter\n"
        "print('imported', EvidenceWriter.__name__)",
    )
    assert result.returncode == 0, result.stderr
    assert "imported EvidenceWriter" in result.stdout


def test_the_refusal_is_the_guards_own_sentence_not_an_import_error(tmp_path: Path) -> None:
    """The public entry point, reached on such a platform, refuses BY NAME.

    ``_supported`` is called before any syscall, so the caller gets the
    documented ``EvidenceUnsupported`` rather than a traceback from inside the
    locking helper — which is what makes the dead-code guard live again.
    """
    bundle = tmp_path / "bundle"
    result = _run(
        tmp_path,
        "from pathlib import Path\n"
        "from tests.unit.evaluation.evidence.test_models import manifest\n"
        "from tests.unit.evaluation.evidence.test_store import redactions\n"
        "from local_operator.evaluation.evidence.store import EvidenceUnsupported, EvidenceWriter\n"
        "try:\n"
        f"    EvidenceWriter.create(Path({str(bundle)!r}), manifest(), redactions('secret'))\n"
        "except EvidenceUnsupported as exc:\n"
        "    print('REFUSED:', exc)\n"
        "else:\n"
        "    print('NOT REFUSED')\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("REFUSED:"), result.stdout + result.stderr
    assert "POSIX flock and directory descriptors" in result.stdout
