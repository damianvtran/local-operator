"""R12/R13 actual adapter regressions; inert OS/daemon transports only."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import lifecycle_proposal as target
import test_causal_startup as prior


class CleanupFenceTests(unittest.TestCase):
    def adapter(self,folder):
        ops,d,patch=prior.CausalStartupTests().adapter(folder)
        with patch:ops.start_owned()
        ops.begin_cleanup()
        return ops,d

    def test_available_overcap_engine_must_probe_foreign_containers(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f)
            ops.command.side_effect=[(0,json.dumps({'MemTotal':8_200_000_000,'NCPU':14}),''),(0,'foreign-container','')]
            with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertFalse(d.terminated)
            self.assertEqual([c.args[0][1] for c in ops.command.call_args_list],['info','ps'])

    def test_root_journal_io_cannot_erase_in_memory_cleanup_owner(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=prior.CausalStartupTests().adapter(f);original=ops.startup_event
            def event(row):
                if row['event']=='causal-root-observed':raise OSError('inert journal I/O failure')
                original(row)
            ops.startup_event=event
            with patch,self.assertRaises(OSError):ops.start_owned()
            self.assertIsNotNone(ops.owner)
            self.assertTrue(ops.owned_processes)
            ops.begin_cleanup();ops.command.return_value=(1,'','Cannot connect to the Docker daemon at unix:///var/run/docker.sock.')
            ops.cleanup_pending_launch();ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated)


    def test_engine_state_matrix_never_treats_unknown_as_unavailable(self):
        ready=(0,json.dumps({'MemTotal':8_200_000_000,'NCPU':14}),'')
        unavailable=(1,'','Cannot connect to the Docker daemon at unix:///var/run/docker.sock.')
        cases=[('available-clean',[ready,(0,'','')],True),('foreign',[ready,(0,'foreign','')],False),
               ('failed-ps',[ready,(1,'','permission denied')],False),('unknown-info',[(1,'','permission denied')],False),
               ('malformed-info',[(0,'{}','')],False),('never-ready',[unavailable],True),
               ('availability-lost',[unavailable],False),('unknown-history',[unavailable],False)]
        for kind,responses,allowed in cases:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,d=self.adapter(f);ops.command.side_effect=responses
                if kind=='availability-lost':ops.daemon_observed_available=True
                if kind=='unknown-history':ops.engine_history_unknown=True
                if allowed:ops.quit_owned(ops.owner)
                else:
                    with self.assertRaises(Exception):ops.quit_owned(ops.owner)
                self.assertEqual(d.terminated,allowed)

    def test_actual_lifecycle_overcap_auto_restart_blocks_termination(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=prior.CausalStartupTests().adapter(f);ops.preflight=mock.Mock();ops.record=lambda value:setattr(ops,'receipt_fixture',value)
            ready=(0,json.dumps({'MemTotal':8_200_000_000,'NCPU':14}),'')
            ops.command.side_effect=[ready,ready,(0,'foreign-autorestarted-container','')]
            admission={'go':True,'scope':'resolver-only','full_scope_review':{'kind':'full-scope','reviewer':'independent-fixture'}}
            with patch,self.assertRaises(RuntimeError):target.run_lifecycle(ops,admission,{})
            self.assertFalse(d.terminated);self.assertEqual(ops.receipt_fixture['status'],'cleanup-unverified')
            self.assertEqual([c.args[0][1] for c in ops.command.call_args_list],['info','info','ps'])

    def test_journal_failures_at_callback_root_and_edge_still_retire_authority(self):
        for stage in ('causal-launch-ticket','causal-root-observed','observed-owned-edge'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,d,patch=prior.CausalStartupTests().adapter(f);original=ops.startup_event
                def event(row):
                    if row['event']==stage:raise OSError('inert journal failure: '+stage)
                    original(row)
                ops.startup_event=event
                if stage=='observed-owned-edge':
                    with patch:ops.start_owned()
                    d.rows.append({'pid':3,'ppid':2,'pgid':2,'start':'proven-child','exe':prior.EXE,'rss_kib':1})
                    with self.assertRaises(OSError):ops.observe_owned()
                else:
                    with patch,self.assertRaises((OSError,RuntimeError)):ops.start_owned()
                ops.begin_cleanup();ops.cleanup_pending_launch()
                self.assertTrue(ops.has_cleanup_authority());self.assertIsNotNone(ops.owner)
                ops.quit_owned(ops.owner);self.assertTrue(d.terminated)

    def test_retained_map_dispatches_even_if_owner_attribute_is_lost(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=prior.CausalStartupTests().adapter(f);ops.preflight=mock.Mock();ops.record=lambda value:setattr(ops,'receipt_fixture',value)
            original=ops.start_owned
            def abort_after_proof():original();ops.owner=None;raise OSError('inert late ownership journal failure')
            ops.start_owned=abort_after_proof
            admission={'go':True,'scope':'resolver-only','full_scope_review':{'kind':'full-scope','reviewer':'independent-fixture'}}
            with patch,self.assertRaises(OSError):target.run_lifecycle(ops,admission,{})
            self.assertTrue(d.terminated);self.assertTrue(ops.has_cleanup_authority())
            self.assertNotEqual(ops.receipt_fixture['status'],'resolved-not-installed-not-qualified')

    def test_startup_owner_file_failure_does_not_skip_cleanup(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=prior.CausalStartupTests().adapter(f);original=Path.open
            def opened(path,*args,**kwargs):
                if path.name=='startup-owner.json':raise OSError('inert owner receipt failure')
                return original(path,*args,**kwargs)
            with patch,mock.patch.object(Path,'open',opened),self.assertRaises(OSError):ops.start_owned()
            self.assertTrue(ops.has_cleanup_authority());ops.begin_cleanup();ops.cleanup_pending_launch();ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated)

    def test_refused_terminate_and_broken_journal_still_try_exact_pid_recovery(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f)
            d.terminate=lambda ticket:False
            def broken(row):raise OSError('inert cleanup journal unavailable')
            ops.startup_event=broken;ops.broker.emit=broken
            def signal_owned(pid,signum):
                self.assertEqual(pid,2);self.assertEqual(signum,target.signal.SIGTERM);d.rows=[]
            with mock.patch.object(target.os,'kill',side_effect=signal_owned) as signaled,self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            signaled.assert_called_once();self.assertEqual(d.rows,[]);self.assertTrue(ops.ownership_journal_errors)

    def test_native_final_receipt_retains_journal_error_and_cannot_be_success(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            root=Path(f);ops=target.NativeOps({'full_scope_review':{'source_commit':'fixture'}},{})
            ops.root=root;ops.directory=root;ops.raw=root/'Docker.raw';ops.raw.write_bytes(b'');ops.baseline=0
            ops.command=mock.Mock(return_value=(0,'used = 0.00M',''))
            ops.ownership_journal_errors=[{'event':'fixture','error':'retained I/O failure'}]
            with mock.patch.object(target.os,'statvfs',return_value=mock.Mock(f_bavail=40*1024**3,f_frsize=1)),self.assertRaises(RuntimeError):
                ops.record({'status':'resolved-not-installed-not-qualified','runtime_qualified':False})
            receipt=json.loads((root/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'cleanup-unverified');self.assertEqual(receipt['ownership_journal_errors'],ops.ownership_journal_errors)

    def test_terminate_journal_failure_reports_unverified_after_retirement(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f);original=ops.startup_event
            def event(row):
                if row['event']=='causal-terminate-request':raise OSError('inert terminate journal failure')
                original(row)
            ops.startup_event=event;ops.broker.emit=event
            with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated);self.assertTrue(ops.ownership_journal_errors)


if __name__=='__main__':unittest.main()
