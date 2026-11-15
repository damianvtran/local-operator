"""R15: actual command waits on real monotonic time; never runs codesign/app."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import unittest
from unittest import mock
import lifecycle_proposal as target
import test_causal_startup as prior


class IdentityProofTests(unittest.TestCase):
    def adapter(self,folder):
        ops,d,patch=prior.CausalStartupTests().adapter(folder)
        with patch:ops.start_owned()
        ops.owned_processes.clear();ops.owner=None;ops.verified_process_paths.clear()
        ops.now=time.monotonic;ops.deadline=ops.now()+840;ops.startup_deadline=ops.now()+120
        ops.command=types.MethodType(target.NativeOps.command,ops)
        ops.verify_owned_path=types.MethodType(target.NativeOps.verify_owned_path,ops)
        # The signature subprocess is an inert transport; path checks stay
        # deterministic, not dependent on a user's Docker installation.
        ops.snapshot=lambda:copy.deepcopy(d.rows)
        return ops,d

    def transport(self,ops,code=0,duration=0.65):
        processes=[]
        class Process:
            pid=990002
            def __init__(self):self.start=time.monotonic();self.returncode=None;processes.append(self)
            def poll(self):return self.returncode
            def communicate(self,timeout=None):
                if self.returncode is not None:return '',''
                remaining=duration-(time.monotonic()-self.start)
                if remaining>0:
                    delay=min(remaining,timeout);time.sleep(delay)
                    if delay<remaining:raise subprocess.TimeoutExpired('INERT codesign',timeout)
                self.returncode=code;return '','inert signature refusal' if code else ''
        def popen(args,**kwargs):
            if args[0]!='/usr/bin/codesign':raise AssertionError('unexpected process dispatch')
            return Process()
        return mock.patch.object(target.subprocess,'Popen',side_effect=popen),processes

    def test_slow_signature_proof_checks_resources_without_recursive_observation(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f);checks=[]
            ops.sample_swap=lambda phase:checks.append((time.monotonic(),phase,ops.owner))
            original=ops.observe_owned;active=0
            def observe():
                nonlocal active
                if active:raise AssertionError('recursive startup ownership observation')
                active+=1
                try:return original()
                finally:active-=1
            ops.observe_owned=observe
            transport,processes=self.transport(ops);begin=time.monotonic()
            with transport,mock.patch.object(Path,'is_file',return_value=True),mock.patch.object(Path,'resolve',lambda p,**kwargs:p):
                ops.startup_guard()
            elapsed=time.monotonic()-begin
            self.assertGreaterEqual(elapsed,0.60);self.assertEqual(len(processes),1)
            self.assertTrue(any(at-begin>=0.45 and owner is None for at,phase,owner in checks),'resource guard must run during the >0.5s unproven interval')
            self.assertEqual(ops.owner['pid'],2);self.assertIn(prior.EXE,ops.verified_process_paths)

    def test_slow_proof_outside_startup_guard_never_recurses(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f);transport,processes=self.transport(ops);begin=time.monotonic()
            with transport,mock.patch.object(Path,'is_file',return_value=True),mock.patch.object(Path,'resolve',lambda p,**kwargs:p):
                ops.observe_owned()
            self.assertGreaterEqual(time.monotonic()-begin,0.60);self.assertEqual(len(processes),1)
            self.assertEqual(ops.owner['pid'],2);self.assertEqual(ops.ownership_scan_depth,0);self.assertEqual(ops.identity_proof_depth,0)

    def test_signature_failure_deadline_and_resource_error_retain_ticket_not_false_proof(self):
        for kind in ('signature','deadline','swap','slot','rss'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,d=self.adapter(f);transport,processes=self.transport(ops,code=1 if kind=='signature' else 0)
                if kind=='deadline':ops.startup_deadline=time.monotonic()+0.55
                if kind=='swap':ops.sample_swap=mock.Mock(side_effect=RuntimeError('inert swap abort during proof'))
                if kind=='slot':ops.a['slot_end_utc']='2000-01-01T00:00:00+00:00'
                if kind=='rss':d.rows.append(dict(d.rows[0],pid=3,ppid=2,exe='/Applications/Docker.app/Contents/MacOS/com.docker.virtualization',rss_kib=target.CAP//1024+1))
                def reap(pid,signum):
                    self.assertEqual(pid,990002)
                    for p in processes:p.returncode=-9
                with transport,mock.patch.object(target.os,'killpg',side_effect=reap),mock.patch.object(Path,'is_file',return_value=True),mock.patch.object(Path,'resolve',lambda p,**kwargs:p),self.assertRaises(RuntimeError):
                    ops.observe_owned()
                self.assertIsNotNone(ops.broker.ticket);self.assertIsNone(ops.owner)
                self.assertNotIn(prior.EXE,ops.verified_process_paths)
                self.assertEqual(ops.identity_proof_depth,0);self.assertEqual(ops.ownership_scan_depth,0);self.assertFalse(ops.resource_guard_running)

    def test_blocked_topology_admission_stops_before_any_native_command(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=prior.CausalStartupTests().adapter(f);ops.a['_admission_path']='inert-admission'
            with mock.patch.object(target,'validated_registered_root'),mock.patch.object(target,'validated_admission_path'),self.assertRaisesRegex(RuntimeError,'topology remains unproven'):
                ops.preflight()
            ops.command.assert_not_called();self.assertIsNone(ops.broker)

    def test_native2_topology_remains_unproven_not_adopted_from_timing(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            driver=prior.Driver();driver.ticket['pid']=75147
            def pump(seconds):
                driver.time+=seconds
                if driver.fired:return
                driver.fired=True
                root={'pid':75147,'ppid':75032,'pgid':75032,'start':'Sun Nov 15 03:48:02 2026','exe':prior.EXE,'rss_kib':206640}
                parent=dict(root,pid=75032,ppid=1);sibling=dict(root,pid=75149,ppid=75032)
                vm=dict(root,pid=75501,ppid=75147,start='Sun Nov 15 03:48:04 2026',exe='/Applications/Docker.app/Contents/MacOS/com.docker.virtualization')
                driver.rows=[parent,root,sibling,vm];driver.callback(driver.ticket,None)
            driver.pump=pump
            driver.matches=lambda ticket:ticket==driver.ticket and any(p['pid']==75147 for p in driver.rows)
            ops,d,patch=prior.CausalStartupTests().adapter(f,driver)
            with patch,self.assertRaisesRegex(RuntimeError,'foreign/unobserved'):ops.start_owned()
            self.assertIn(75147,ops.owned_processes);self.assertIn(75501,ops.owned_processes)
            self.assertNotIn(75032,ops.owned_processes);self.assertNotIn(75149,ops.owned_processes)
            self.assertFalse(d.terminated)


if __name__=='__main__':unittest.main()
