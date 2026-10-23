"""Exact Python3.12 native proof prerequisite; not a substitute native run."""
import hashlib
import json
from pathlib import Path
import sys

VERDICT='GENERIC_NATIVE_PRODUCER_QUALIFIED_NOT_S2_OR_PRODUCT_EXONERATION'


def permits_s2(proof, rc, python, executable_sha256, sources):
    if type(rc) is not int or rc!=0 or type(proof) is not dict:return False
    if type(python) is not list or len(python)!=3 or any(type(x) is not int for x in python) or python[:2]!=[3,12]:return False
    if proof.get('qualified') is not True or proof.get('full_proof_qualified') is not True:return False
    if proof.get('verdict')!=VERDICT or proof.get('proof_stage')!='QUALIFIED' or proof.get('root_deletion')!='REMOVED':return False
    life=proof.get('lifecycle')
    if type(life) is not dict or set(life)!={'rc','reaped','owned_settled','errors','synthetic_root_removed'}:return False
    if type(life['rc']) is not int or life['rc']!=0 or life['errors']!=[]:return False
    if any(life[k] is not True for k in ('reaped','owned_settled','synthetic_root_removed')):return False
    ident=proof.get('identity')
    if type(ident) is not dict or set(ident)!={'python','executable_sha256','pid','ppid','pgid','sources'}:return False
    if ident['python']!=python or any(type(x) is not int for x in ident['python']):return False
    if ident['executable_sha256']!=executable_sha256 or ident['sources']!=sources:return False
    if any(type(ident[k]) is not int or ident[k]<=0 for k in ('pid','ppid','pgid')):return False
    return True


def main():
    allowed=False
    try:
        if sys.version_info[:2]!=(3,12):raise ValueError('version')
        source=Path(sys.argv[1])
        if source.is_symlink() or not source.is_file() or source.stat().st_size>262144:raise ValueError('proof')
        proof=json.loads(source.read_bytes())
        root=Path(__file__).resolve().parents[3]
        manifest=json.loads((root/'scripts/diagnostics/native_worker_6911/manifest.json').read_bytes())
        sources={p:manifest['sources'][p] for p in manifest['loaded_sources']}
        allowed=permits_s2(proof,int(sys.argv[3]),list(sys.version_info[:3]),hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),sources)
    except Exception:pass
    Path(sys.argv[2]).write_text(json.dumps(dict(native_gate='QUALIFIED' if allowed else 'UNKNOWN_OR_REJECTED',s2='AUTHORIZED_ONE_ORIGINAL_CALL' if allowed else 'UNRUN_NO_DISPATCH'))+'\n')
    if not allowed:raise SystemExit(1)


if __name__=='__main__':main()
