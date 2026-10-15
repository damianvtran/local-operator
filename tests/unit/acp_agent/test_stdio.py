"""``claim_stdio`` (D4): the protocol pipes become private before anything runs.

Exercised in a SUBPROCESS because the claim rewires this process's fd 0 and 1,
which pytest itself is using. The probe proves the three properties the design
rests on, each by observation rather than by reading the code:

* a child process started after the claim reads EOF on stdin — so a ``bash``
  tool's ``cat`` cannot swallow JSON-RPC frames meant for the agent;
* a stray ``print`` after the claim lands on stderr, never on the protocol
  stream;
* the private descriptors are not inheritable, and writing to the private
  writer reaches the ORIGINAL stdout, which is where the client reads.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_PROBE = r"""
import json, os, subprocess, sys
from local_operator.acp_agent.stdio import claim_stdio

reader, writer = claim_stdio()
child = subprocess.run(
    [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()), file=sys.stderr)"],
    capture_output=True, text=True, timeout=60,
)
print("STRAY-PRINT")
sys.stdout.flush()
report = {
    "child_stdin": child.stderr.strip(),
    "reader_inheritable": os.get_inheritable(reader),
    "writer_inheritable": os.get_inheritable(writer),
    "frame_from_parent": os.read(reader, 64).decode(),
}
os.write(writer, (json.dumps(report) + "\n").encode())
"""


def test_claim_makes_the_protocol_pipes_private() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        input="FRAME-ON-STDIN",
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    # The child saw /dev/null, not the parent's protocol stdin.
    assert report["child_stdin"] == "''"
    # The frame the client wrote is still there for the agent to read.
    assert report["frame_from_parent"] == "FRAME-ON-STDIN"
    assert report["reader_inheritable"] is False
    assert report["writer_inheritable"] is False
    # A stray print after the claim reached stderr, and stdout carries only the
    # private writer's frame.
    assert "STRAY-PRINT" in proc.stderr
    assert "STRAY-PRINT" not in proc.stdout
    assert proc.stdout.strip().splitlines() == [proc.stdout.strip()]
