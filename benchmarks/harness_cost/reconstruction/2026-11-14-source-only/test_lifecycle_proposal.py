"""Full lifecycle is exercised through stubs, never Desktop/Docker/containers."""
import hashlib
import unittest
from unittest import mock

try:
    import lifecycle_proposal as target
except ImportError:
    target = None


class StubOps:
    def __init__(self, memory=4_000_000_000, helper=True, foreign=False, missing=False, digest=False):
        self.memory=memory; self.helper=helper; self.foreign=foreign; self.missing=missing; self.digest=digest
        self.events=[]; self.running=False; self.ticks=0
    def preflight(self): self.events.append('fresh-preflight')
    def existing_desktop(self): return self.foreign
    def start_owned(self): self.running=True; self.events.append('start-owned'); return {'pid':123,'identity':'same-start/exe'}
    def info(self): self.events.append('info'); return {'MemTotal':self.memory,'NCPU':2}
    def vm_helper(self, owner):
        if not self.helper: raise RuntimeError('unidentified VM helper')
        self.events.append('attributed-rss'); return {'pid':456,'rss_bytes':200_000_000,'measurement':'RSS-not-footprint'}
    def check_disk(self): self.events.append('disk')
    def cached_carrier(self): return None if self.missing else {'Architecture':'amd64','Os':'linux','RepoDigests':['python@sha256:' + ('b' if self.digest else 'a')*64]}
    def pull(self): self.events.append('PULL'); self.missing=False
    def wheel_hash(self): return 'c'*64
    def begin_resolver(self): self.events.append('resolver')
    def resolver_done(self): self.ticks+=1; return self.ticks>1
    def sleep(self, seconds): pass
    def now(self): return float(self.ticks)*10
    def stop_owned_resolver(self): self.events.append('stop-resolver')
    def quit_owned(self, owner): self.events.append('quit-owned'); self.running=False
    def record(self, value): self.receipt=value


def admission(pull=False, kind='full-scope'):
    return {'go':True,'scope':'resolver-only','full_scope_review':{'kind':kind,'reviewer':'independent','allow_digest_pull':pull},'authorize_digest_pull':pull}


class LifecycleTests(unittest.TestCase):
    def run_case(self, ops, **kwargs):
        self.assertIsNotNone(target, 'new lifecycle contract absent')
        return target.run_lifecycle(ops, admission(**kwargs), {'base_image':'python@sha256:'+'a'*64,'uv_wheel':{'sha256':'c'*64}}, require_runtime_scope=False)
    def test_overcap_aborts_and_quits_only_owned_without_container(self):
        ops=StubOps(memory=4_300_000_001)
        with self.assertRaises(RuntimeError): self.run_case(ops)
        self.assertIn('quit-owned',ops.events);self.assertNotIn('resolver',ops.events)
    def test_unidentified_vm_fails_closed(self):
        ops=StubOps(helper=False)
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertIn('quit-owned',ops.events);self.assertNotIn('resolver',ops.events)
    def test_existing_desktop_is_neither_started_nor_quit(self):
        ops=StubOps(foreign=True)
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertNotIn('start-owned',ops.events);self.assertNotIn('quit-owned',ops.events)
    def test_default_no_pull_and_missing_artifact_aborts(self):
        ops=StubOps(missing=True)
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertNotIn('PULL',ops.events);self.assertNotIn('resolver',ops.events)
    def test_optional_digest_pull_requires_full_scope_signoff(self):
        ops=StubOps(missing=True);self.run_case(ops,pull=True)
        self.assertEqual(ops.events.count('PULL'),1)
        ops=StubOps(missing=True)
        with self.assertRaises(RuntimeError):self.run_case(ops,pull=True,kind='delta-only')
        self.assertNotIn('PULL',ops.events)
    def test_digest_and_wheel_hash_mismatch_never_installs(self):
        for ops in (StubOps(digest=True),StubOps()):
            if not ops.digest:ops.wheel_hash=lambda:'wrong'
            with self.assertRaises(RuntimeError):self.run_case(ops)
            self.assertNotIn('resolver',ops.events);self.assertNotIn('install',ops.events)
    def test_rss_overcap_and_termination_cleanup(self):
        ops=StubOps();ops.vm_helper=lambda owner:{'pid':456,'rss_bytes':4_300_000_001}
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertIn('quit-owned',ops.events)
        ops=StubOps();ops.sleep=lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):self.run_case(ops)
        self.assertIn('stop-resolver',ops.events);self.assertIn('quit-owned',ops.events)
    def test_cpu_overcap_missing_wheel_and_work_deadline_abort(self):
        ops=StubOps();ops.info=lambda:{'MemTotal':4_000_000_000,'NCPU':3}
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertNotIn('resolver',ops.events)
        ops=StubOps();ops.wheel_hash=lambda:(_ for _ in ()).throw(FileNotFoundError('missing staged wheel'))
        with self.assertRaises(FileNotFoundError):self.run_case(ops)
        self.assertNotIn('resolver',ops.events);self.assertIn('quit-owned',ops.events)
        ops=StubOps();ops.now=lambda:float(ops.ticks)*900
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertIn('stop-resolver',ops.events);self.assertIn('quit-owned',ops.events)

    def test_vm_attribution_excludes_unrelated_and_rejects_ambiguity(self):
        self.assertIsNotNone(target)
        owner={'pid':123,'ppid':1,'start':'owner','exe':'/Applications/Docker.app/Contents/MacOS/Docker','rss_kib':1}
        old={'pid':9,'ppid':1,'start':'old','exe':'/System/com.apple.Virtualization.VirtualMachine','rss_kib':999}
        helper={'pid':456,'ppid':1,'start':'new','exe':'/System/com.apple.Virtualization.VirtualMachine','rss_kib':100}
        chosen=target.choose_helper([old],[owner,old,helper],owner,{456})
        self.assertEqual(chosen['pid'],456);self.assertEqual(chosen['rss_bytes'],100*1024)
        self.assertIn('not-footprint',chosen['measurement'])
        with self.assertRaises(RuntimeError):target.choose_helper([old],[owner,old,helper],owner,set())
        other=dict(helper,pid=457)
        with self.assertRaises(RuntimeError):target.choose_helper([old],[owner,old,helper,other],owner,{456,457})

    def test_cleanup_ownership_failure_is_not_reported_success(self):
        ops=StubOps();ops.quit_owned=lambda owner:(_ for _ in ()).throw(RuntimeError('owner identity changed'))
        with self.assertRaises(RuntimeError):self.run_case(ops)
        self.assertEqual(ops.receipt['status'],'cleanup-unverified')
        self.assertFalse(ops.receipt['runtime_qualified'])

    def test_concrete_quit_refuses_changed_owner_before_any_signal(self):
        self.assertIsNotNone(target)
        ops=object.__new__(target.NativeOps)
        owner={'pid':123,'ppid':1,'start':'owned','exe':'/Applications/Docker.app/Contents/MacOS/Docker','rss_kib':1}
        ops.owner=owner;ops.before=[];ops.monitor_enabled=True;ops.app_root='/Applications/Docker.app/'
        ops.snapshot=lambda:[dict(owner,start='different-start')]
        ops.proc=mock.Mock();ops.proc.poll.return_value=None
        with mock.patch.object(target.os,'kill') as kill:
            with self.assertRaises(RuntimeError):ops.quit_owned(owner)
            kill.assert_not_called()

    def test_termination_during_resolver_start_still_requests_owned_cleanup(self):
        ops=StubOps();ops.begin_resolver=lambda:(_ for _ in ()).throw(KeyboardInterrupt('pending SIGTERM at launch'))
        with self.assertRaises(KeyboardInterrupt):self.run_case(ops)
        self.assertIn('stop-resolver',ops.events);self.assertIn('quit-owned',ops.events)

    def test_no_full_scope_review_has_no_effects(self):
        self.assertIsNotNone(target)
        ops=StubOps()
        with self.assertRaises(RuntimeError):target.run_lifecycle(ops,{'go':False},{},require_runtime_scope=False)
        self.assertEqual(ops.events,[])


if __name__=='__main__':unittest.main()
