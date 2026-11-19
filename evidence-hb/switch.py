import sys; from pathlib import Path
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
root=Path(sys.argv[1]); assert "/scratchpad/hb/iso/.local-operator" in str(root)
o=OwnerSnapshot(credential_id=7, account_id="acc_2", tenant_id="3f2a1111-2222-4333-8444-555566667777", endpoint_id="ep_2", canonical_base_url=sys.argv[2], enrollment_generation=1)
Store(root).commit_enrollment(o, OwnerReceipt("bridge_consent_v1","ref-2"), cos_session_id="abcdef012345"); print("enrolled acc_2/ep_2")
