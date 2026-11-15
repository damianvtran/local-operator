"""The pairing SAS is computed on two ends that share no code: this computer
(``pair_op.sas_code``) and the phone (the PWA's WebCrypto HKDF, later the native
apps). A drift in either one shows up only as "the codes never match" on a real
device, so both sides pin the SAME vector file. This test is the Python half; the
PWA's ``src/lib/pair-sas.test.ts`` is the other. Regenerating the file is a
deliberate protocol change (the label ``lop-pair-sas-v1`` would change with it).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from local_operator.operator import pair_op

VECTORS = (
    Path(__file__).resolve().parents[3]
    / "local_operator/mobile/web/src/fixtures/pair-sas-vectors.json"
)


def _raw(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def test_the_shared_vectors_pin_this_derivation() -> None:
    doc = json.loads(VECTORS.read_text())
    assert doc["label"] == pair_op.SAS_LABEL.decode()
    assert len(doc["vectors"]) >= 3
    for vector in doc["vectors"]:
        np_ = _raw(vector["np"])
        assert pair_op.sas_commit(np_) == vector["commit"]
        assert (
            pair_op.sas_code(
                np=np_,
                nd=_raw(vector["nd"]),
                device_spki=_raw(vector["device_spki"]),
                operator_spki=_raw(vector["operator_spki"]),
                approval_id=vector["approval_id"],
            )
            == vector["sas"]
        )
