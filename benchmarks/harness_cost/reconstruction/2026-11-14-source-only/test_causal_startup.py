"""R11 actual adapter + fake causal OS driver; NO app/service/Docker launch."""
import copy
import datetime
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import lifecycle_proposal as target
from launch_services_broker import CausalLaunch

ROOT='/Applications/Docker.app';EXE=ROOT+'/Contents/MacOS/com.docker.backend'


class Driver:
    def __init__(self,delay=0.5,error=None):
        self.time=0.;self.delay=delay;self.error=error;self.callback=None;self.fired=False;self.rows=[];self.terminated=False
        self.ticket={'pid':2,'bundle':ROOT,'executable':EXE,'identifier':'com.docker.docker','launch_date':12345.,'new_instance_requested':True}
    def begin(self,bundle,callback):self.callback=callback
    def pump(self,seconds):
        self.time+=seconds
        if not self.fired and self.time>=self.delay:
            self.fired=True
            if not self.error:self.rows=[{'pid':2,'ppid':1,'pgid':2,'start':'ticket-root','exe':EXE,'rss_kib':100}]
            self.callback(None if self.error else self.ticket,self.error)
    def matches(self,ticket):return ticket==self.ticket and any(p['pid']==2 for p in self.rows)
    def terminate(self,ticket):
        if ticket!=self.ticket:raise RuntimeError('foreign ticket')
        self.terminated=True;self.rows=[];return True


class CausalStartupTests(unittest.TestCase):
    def adapter(self,folder,driver=None):
        d=driver or Driver();ops=target.NativeOps({'attempt_id':'linux-amd64-02','slot_end_utc':(datetime.datetime.now(datetime.UTC)+datetime.timedelta(hours=1)).isoformat()},{})
        ops.directory=Path(folder);ops.root=Path(folder);ops.deadline=840;ops.now=lambda:d.time
        ops.desktop_layout={'bundles':[{'entrypoint':'com.docker.backend'}]}
        ops.snapshot=lambda:copy.deepcopy(d.rows);ops.verify_owned_path=mock.Mock();ops.sample_swap=mock.Mock();ops.check_disk=mock.Mock()
        ops.command=mock.Mock();ops.d=['NO-DOCKER']
        def factory(bundle,exe,before,emit):return CausalLaunch(bundle,exe,before,emit,driver=d)
        return ops,d,mock.patch.object(target,'CausalLaunch',side_effect=factory)
    def test_causal_ticket_root_and_observed_reparented_descendant_only(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f)
            with patch:owner=ops.start_owned()
            d.rows.append({'pid':3,'ppid':2,'pgid':3,'start':'observed-child','exe':EXE,'rss_kib':100})
            ops.observe_owned();d.rows=[dict(d.rows[1],ppid=1)]
            rows,live=ops.observe_owned();self.assertEqual(ops.effective_owner(live)['pid'],3)
            self.assertEqual(ops.ownership_edges[0]['parent']['pid'],2)
            self.assertEqual(json.loads((Path(f)/'startup-owner.json').read_text())['pid'],2)
    def test_singleton_time_window_without_observed_edge_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f)
            with patch:ops.start_owned()
            d.rows=[{'pid':3,'ppid':1,'pgid':3,'start':'new-but-unproven','exe':EXE,'rss_kib':100}]
            with self.assertRaisesRegex(RuntimeError,'foreign/unobserved'):ops.observe_owned()
    def test_foreign_race_and_unknown_pid_preserved(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f)
            with patch:ops.start_owned()
            d.rows.append({'pid':99,'ppid':1,'pgid':99,'start':'foreign','exe':EXE,'rss_kib':100})
            with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertFalse(d.terminated)
    def test_old_pid_or_wrong_identity_ticket_cannot_be_adopted(self):
        for field,value in [('pid',0),('identifier','foreign.app'),('executable','/untrusted')]:
            d=Driver();d.ticket[field]=value;events=[];b=CausalLaunch(ROOT,EXE,set(),events.append,driver=d);b.begin();b.pump(1)
            self.assertIsNone(b.ticket);self.assertIsNotNone(b.error)
            with self.assertRaises(RuntimeError):b.terminate()
        d=Driver();b=CausalLaunch(ROOT,EXE,{2},lambda row:None,driver=d);b.begin();b.pump(1);self.assertIsNone(b.ticket)
    def test_slow_daemon_info_is_first_api_and_guarded(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f)
            with patch:ops.start_owned()
            error='Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?'
            ops.command.side_effect=[(1,'',error),(1,'',error),(0,json.dumps({'MemTotal':4_000_000_000,'NCPU':2}),'')]
            info=ops.info();self.assertEqual(info['NCPU'],2)
            self.assertTrue(all(c.args[0][1:2]==['info'] for c in ops.command.call_args_list))
            self.assertGreaterEqual(ops.sample_swap.call_count,6);self.assertFalse(ops.startup_active)
    def test_refused_info_and_overcap_are_not_retried(self):
        for response in [(1,'','permission denied'),(0,json.dumps({'MemTotal':8_200_000_000,'NCPU':14}),'')]:
            with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,d,patch=self.adapter(f)
                with patch:ops.start_owned()
                ops.command.return_value=response
                with self.assertRaises(RuntimeError):ops.info()
                self.assertEqual(ops.command.call_count,1);self.assertFalse((Path(f)/'readiness.json').exists())
    def test_callback_timeout_and_late_capture_are_not_skipped(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f,Driver(delay=140))
            with patch,self.assertRaisesRegex(RuntimeError,'deadline'):ops.start_owned()
            self.assertLessEqual(d.time,120)
            ops.begin_cleanup();ops.cleanup_pending_launch();ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated);self.assertTrue(ops.broker.ticket)
    def test_never_returned_callback_is_unresolved_not_guessed(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f,Driver(delay=999))
            with patch,self.assertRaises(RuntimeError):ops.start_owned()
            ops.begin_cleanup()
            with self.assertRaisesRegex(RuntimeError,'UNRESOLVED'):ops.cleanup_pending_launch()
            self.assertFalse(d.terminated);self.assertLessEqual(d.time,180)
    def test_launch_error_stops_before_any_docker_api(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f,Driver(error='launch denied'))
            with patch,self.assertRaises(RuntimeError):ops.start_owned()
            ops.command.assert_not_called()
    def test_daemon_absent_cleanup_needs_no_docker_ps(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f)
            with patch:ops.start_owned()
            ops.begin_cleanup();ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated);ops.command.assert_not_called()
    def test_journal_failure_keeps_authority_for_cleanup(self):
        d=Driver()
        def emit(row):
            if row['event']=='causal-launch-ticket':raise OSError('fixture journal failure')
        b=CausalLaunch(ROOT,EXE,set(),emit,driver=d);b.begin();b.pump(1)
        self.assertTrue(b.ticket);self.assertTrue(b.error);b.terminate();self.assertTrue(d.terminated)
    def test_readiness_120s_resource_slot_and_work_bounds_stop_before_more_apis(self):
        for failure in ('timeout','swap','slot','work','validity'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,d,patch=self.adapter(f)
                with patch:ops.start_owned()
                ops.command.return_value=(1,'','Cannot connect to the Docker daemon at unix:///var/run/docker.sock.')
                if failure=='swap':ops.sample_swap.side_effect=RuntimeError('swap fixture breach')
                if failure=='slot':ops.a['slot_end_utc']=(datetime.datetime.now(datetime.UTC)+datetime.timedelta(seconds=30)).isoformat()
                if failure=='work':ops.deadline=d.time
                if failure=='validity':ops.a['valid_until_utc']=(datetime.datetime.now(datetime.UTC)-datetime.timedelta(seconds=1)).isoformat()
                with self.assertRaises(RuntimeError):ops.info()
                self.assertLessEqual(d.time,120)
                if failure!='timeout':ops.command.assert_not_called()
    def test_old_pid_ticket_and_duplicate_begin_cannot_reuse_authority(self):
        d=Driver();events=[];b=CausalLaunch(ROOT,EXE,{2},events.append,driver=d)
        b.begin();b.pump(1);self.assertIsNone(b.ticket);self.assertTrue(b.error)
        with self.assertRaises(RuntimeError):b.begin()
    def test_attempt_namespace_is_explicit_and_original_is_never_selected(self):
        code=(Path(target.__file__).parent/'RESOLVER-COMMAND.md').read_text()
        self.assertIn("A = ROOT / 'resolver' / admission['attempt_id']",code)
        self.assertNotIn("A = ROOT / 'resolver/linux-amd64-01'",code)
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f);original=Path(f)/'resolver/linux-amd64-01';original.mkdir(parents=True);(original/'keep').write_text('immutable original')
            ops.recover_owned_container()
            self.assertEqual((original/'keep').read_text(),'immutable original')
    def test_fresh_admission_paths_reject_legacy_and_escaping_aliases(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            root=Path(f).resolve();(root/'admissions').mkdir();p=root/'admissions/linux-amd64-02.json';p.write_text('{}')
            self.assertEqual(target.validated_admission_path(str(p),root,'linux-amd64-02'),p)
            for invalid in (root/'desk-admission.json',root/'admissions/../admissions/linux-amd64-02.json'):
                with self.assertRaises(RuntimeError):target.validated_admission_path(str(invalid),root,'linux-amd64-02')
    def test_signal_style_unwind_retains_and_captures_pending_authority(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d,patch=self.adapter(f,Driver(delay=2))
            original=d.pump
            def interrupted(seconds):d.pump=original;raise KeyboardInterrupt('SIGTERM fixture')
            d.pump=interrupted
            with patch,self.assertRaises(KeyboardInterrupt):ops.start_owned()
            ops.begin_cleanup();ops.cleanup_pending_launch();ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated)


if __name__=='__main__':unittest.main()
