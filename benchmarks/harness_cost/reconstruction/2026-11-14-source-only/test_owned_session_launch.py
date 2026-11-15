"""Prospective direct outer backend authority; ALL process/app transports inert."""
import copy
import ctypes
import datetime
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest import mock
import lifecycle_proposal as target
from owned_session_launch import OwnedSessionLaunch,KIND,backend_environment
from darwin_process_identity import BSDInfo,DarwinIdentity

ROOT='/Applications/Docker.app';EXE=ROOT+'/Contents/MacOS/com.docker.backend'
UNAVAILABLE=(1,'','Cannot connect to the Docker daemon at unix:///var/run/docker.sock.')
READY=(0,json.dumps({'MemTotal':4_000_000_000,'NCPU':2}),'')


class Board:
    def __init__(self,root_exit=False):
        self.rows=[];self.spawned=[];self.signals=[];self.waited=False;self.root_exit=root_exit
        board=self
        class Creator:
            pid=75032;returncode=None
            def poll(self):raise AssertionError('creator MUST remain unreaped; poll is forbidden')
            def wait(self,timeout):
                if any(r['sid']==self.pid and not r['state'].startswith('Z') for r in board.rows):raise AssertionError('reap before live members retired')
                self.returncode=0;board.waited=True;board.rows=[r for r in board.rows if r['pid']!=self.pid];return 0
        self.creator=Creator()
    def row(self,pid,ppid,exe=EXE,sid=75032,pgid=75032,state='live'):
        return {'pid':pid,'ppid':ppid,'pgid':pgid,'sid':sid,'state':state,'uid':os.getuid(),'exe':exe,'start':'1794732482.'+str(pid).zfill(6),'rss_kib':100}
    def spawn(self,args,**kwargs):
        self.spawned.append((args,kwargs))
        assert args==[EXE] and kwargs['start_new_session'] is True
        assert {signal.SIGINT,signal.SIGTERM,signal.SIGALRM}<=signal.pthread_sigmask(signal.SIG_BLOCK,[])
        self.rows=[self.row(75032,1,state='Z' if self.root_exit else 'live'),self.row(75147,1 if self.root_exit else 75032),
                   self.row(75149,75032),self.row(75446,75149,ROOT+'/Contents/MacOS/com.docker.build'),
                   self.row(75457,75149,ROOT+'/Contents/MacOS/Docker Desktop.app/Contents/MacOS/Docker Desktop'),
                   self.row(75501,75147,ROOT+'/Contents/MacOS/com.docker.virtualization')]
        return self.creator
    def kill(self,pid,signum):
        if signum==0:
            if any(r['pid']==pid for r in self.rows):return
            raise ProcessLookupError
        assert signum==signal.SIGTERM
        row=next(r for r in self.rows if r['pid']==pid)
        assert row['sid']==75032 or row['ppid']==75032
        self.signals.append(pid)
        if pid==75032:row['state']='Z'
        else:self.rows=[r for r in self.rows if r['pid']!=pid]


class SessionTests(unittest.TestCase):
    def adapter(self,folder,board=None):
        board=board or Board();ops=target.NativeOps({'attempt_id':'linux-amd64-03','slot_end_utc':(datetime.datetime.now(datetime.UTC)+datetime.timedelta(hours=1)).isoformat()},
              {'launch_authority':KIND,'r15_topology_admission':{'ready':True}})
        ops.root=Path(folder);ops.directory=Path(folder);ops.deadline=time.monotonic()+840
        ops.desktop_layout={'bundles':[{'entrypoint':'com.docker.backend'}]}
        ops.snapshot=lambda:copy.deepcopy(board.rows)
        ops.verify_owned_path=mock.Mock();ops.sample_swap=mock.Mock();ops.check_disk=mock.Mock()
        ops.command=mock.Mock(return_value=UNAVAILABLE);ops.d=['NO-DOCKER']
        def factory(bundle,exe,before,emit,log_path):
            authority=OwnedSessionLaunch(bundle,exe,before,emit,spawn=board.spawn,log_path=log_path)
            self.addCleanup(lambda:authority.log.close() if authority.log and not authority.log.closed else None)
            return authority
        return ops,board,mock.patch.object(target,'OwnedSessionLaunch',side_effect=factory)
    def retire(self,ops,b):
        ops.begin_cleanup()
        with mock.patch.object(target.os,'kill',side_effect=b.kill):ops.quit_owned(ops.owner)

    def test_exact_watchdog_backend_ui_build_topology_is_owned_by_created_session(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch:ops.start_owned()
            self.assertEqual(set(ops.owned_processes),{75032,75147,75149,75446,75457,75501})
            self.assertEqual(ops.broker.ticket['sid'],75032);self.assertEqual(ops.broker.ticket['initial_pgid'],75032)
            self.assertEqual(ops.broker.ticket['birth'],b.rows[0]['start']);self.assertNotIn('new_instance_requested',ops.broker.ticket)
            self.assertEqual(b.spawned[0][1]['env']['HOME'],str(Path.home()))
            self.assertNotIn('docker-home',b.spawned[0][1]['env']['HOME'])
            self.retire(ops,b);self.assertTrue(b.waited);self.assertTrue(ops.broker.released)

    def test_creator_exit_reparent_and_child_process_group_do_not_reap_authority(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f,Board(root_exit=True))
            with patch:ops.start_owned()
            b.rows[1]['pgid']=75147
            _,live=ops.observe_owned();self.assertIn(75147,{p['pid'] for p in live});self.assertFalse(b.waited)
            self.assertTrue(ops.broker.matches(ops.broker.ticket))
            self.retire(ops,b);self.assertTrue(b.waited)

    def test_same_signed_path_foreign_session_and_unobserved_escape_are_not_adopted(self):
        for kind in ('foreign','escape','reuse'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,b,patch=self.adapter(f)
                with patch:ops.start_owned()
                if kind=='foreign':b.rows.append(b.row(99001,1,sid=99001,pgid=99001))
                if kind=='escape':b.rows[1].update(sid=75147,pgid=75147,ppid=1)
                if kind=='reuse':b.rows[1]['start']='1794732482.999999'
                with self.assertRaises(RuntimeError):ops.observe_owned()
                self.assertFalse(b.signals);self.assertFalse(b.waited)

    def test_observed_parent_edge_can_cover_a_session_escape_not_a_foreign_group(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch:ops.start_owned()
            b.rows.append(b.row(76000,75032,sid=76000,pgid=76000))
            ops.observe_owned();self.assertEqual(ops.owned_processes[76000]['authority'],'observed-parent-edge')
            self.retire(ops,b);self.assertIn(76000,b.signals)

    def test_published_token_survives_journal_or_signal_style_failure(self):
        for failure in (OSError('inert journal failure'),KeyboardInterrupt('inert delivered signal')):
            with self.subTest(kind=type(failure).__name__),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,b,patch=self.adapter(f);original=ops.startup_event;mask=signal.pthread_sigmask(signal.SIG_BLOCK,[])
                def event(row):
                    if row['event']=='owned-session-created':
                        self.assertTrue(ops.broker.ticket['creator_unreaped']);self.assertIs(ops.broker.creator,b.creator)
                        raise failure
                    original(row)
                ops.startup_event=event
                with patch,self.assertRaises(type(failure)):ops.start_owned()
                self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK,[]),mask)
                self.assertTrue(ops.has_cleanup_authority());self.assertIsNotNone(ops.broker.ticket)
                ops.begin_cleanup();ops.cleanup_pending_launch()
                with mock.patch.object(target.os,'kill',side_effect=b.kill):ops.quit_owned(ops.owner)
                self.assertTrue(b.waited)

    def test_signature_failure_blocks_work_but_not_private_session_retirement(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f);ops.verify_owned_path.side_effect=RuntimeError('inert bad signature')
            with patch,self.assertRaises(RuntimeError):ops.start_owned()
            self.assertTrue(ops.has_cleanup_authority());self.assertTrue(ops.identity_proof_errors)
            ops.begin_cleanup();ops.cleanup_pending_launch()
            with mock.patch.object(target.os,'kill',side_effect=b.kill):ops.quit_owned(ops.owner)
            self.assertTrue(b.waited);self.assertTrue(ops.identity_proof_errors)

    def test_r12_available_foreign_engine_and_r14_info_journal_history_still_block_signals(self):
        for kind in ('foreign','history'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,b,patch=self.adapter(f)
                with patch:ops.start_owned()
                if kind=='foreign':ops.command.side_effect=[READY,(0,'foreign-container','')]
                else:
                    original=ops.startup_event
                    def event(row):
                        if row['event']=='first-api-readiness-info':raise OSError('inert info journal failure')
                        original(row)
                    ops.startup_event=event;replies=iter([READY,UNAVAILABLE])
                    def command(args,**kwargs):
                        if args[0]=='/usr/sbin/lsof':return (0,'n'+str(ops.raw),'')
                        return next(replies)
                    ops.command.side_effect=command
                    with self.assertRaises(OSError):ops.info()
                    self.assertTrue(ops.daemon_observed_available)
                ops.begin_cleanup()
                with mock.patch.object(target.os,'kill',side_effect=b.kill),self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
                self.assertFalse(b.signals);self.assertFalse(b.waited)

    def test_unowned_external_vm_holder_is_preserved_and_shutdown_unverified(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch:ops.start_owned()
            b.rows.append(b.row(79000,1,'/System/Library/com.apple.Virtualization.VirtualMachine',sid=444,pgid=444))
            ops.command.side_effect=lambda args,**kwargs:(0,'n'+str(ops.raw),'') if args[0]=='/usr/sbin/lsof' else UNAVAILABLE
            with self.assertRaisesRegex(RuntimeError,'shutdown association'):self.retire(ops,b)
            self.assertTrue(b.waited);self.assertNotIn(79000,b.signals)
            self.assertTrue(any(p['pid']==79000 for p in b.rows))

    def test_creator_reaped_or_birth_reused_invalidates_all_session_authority(self):
        for kind in ('reaped','birth'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,b,patch=self.adapter(f)
                with patch:ops.start_owned()
                if kind=='reaped':b.creator.returncode=0
                else:b.rows[0]['start']='replacement-birth'
                with self.assertRaises(RuntimeError):ops.observe_owned()
                self.assertFalse(b.signals)

    def test_missing_qualification_helper_and_cleanup_deadline_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch:ops.start_owned()
            ops.command.return_value=(1,'','unavailable fixture raw association')
            with self.assertRaises(RuntimeError):ops.vm_helper(ops.owner)
            ops.begin_cleanup();ops.cleanup_until=ops.now()-1
            with mock.patch.object(target.os,'kill',side_effect=b.kill),self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertFalse(b.signals)

    def test_backend_environment_filters_agent_secrets_but_preserves_settings_home(self):
        with mock.patch.dict(os.environ,{'HARNESS_TEST_API_KEY':'synthetic-marker','CMUX_SOCKET':'synthetic-marker','DOCKER_CONFIG':'/original/settings'}):
            env=backend_environment()
            self.assertNotIn('HARNESS_TEST_API_KEY',env);self.assertNotIn('CMUX_SOCKET',env)
            self.assertEqual(env['HOME'],str(Path.home()));self.assertEqual(env['DOCKER_CONFIG'],'/original/settings')

    def test_signal_at_mask_restore_sees_published_token_before_any_journal(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f);real_mask=signal.pthread_sigmask
            def restore(how,mask):
                previous=real_mask(how,mask)
                if how==signal.SIG_SETMASK and ops.broker and ops.broker.ticket is not None:
                    raise KeyboardInterrupt('inert delivery at mask restore')
                return previous
            with patch,mock.patch('owned_session_launch.signal.pthread_sigmask',side_effect=restore),self.assertRaises(KeyboardInterrupt):ops.start_owned()
            self.assertTrue(ops.has_cleanup_authority());self.assertEqual(ops.broker.ticket['sid'],75032)
            self.retire(ops,b);self.assertTrue(b.waited)

    def test_spawn_exec_failure_has_no_false_authority_or_mask_leak(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f);mask=signal.pthread_sigmask(signal.SIG_BLOCK,[])
            with patch,mock.patch.object(b,'spawn',side_effect=OSError('inert exec failure')),self.assertRaises(OSError):ops.start_owned()
            self.assertFalse(ops.has_cleanup_authority());self.assertIsNone(ops.broker.creator)
            self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK,[]),mask);self.assertFalse(b.signals)

    def test_actual_session_lifecycle_overcap_stops_before_pull_and_reaps_creator(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f);ops.preflight=mock.Mock();ops.record=mock.Mock()
            replies=iter([(0,json.dumps({'MemTotal':8_200_000_000,'NCPU':14}),''),(0,json.dumps({'MemTotal':8_200_000_000,'NCPU':14}),''),(0,'','')])
            def command(args,**kwargs):
                if args[0]=='/usr/sbin/lsof':return 0,'n'+str(ops.raw),''
                self.assertNotIn('pull',args);self.assertNotIn('run',args)
                return next(replies)
            ops.command.side_effect=command
            admission={'go':True,'scope':'resolver-only','full_scope_review':{'kind':'full-scope','reviewer':'independent-fixture'}}
            with patch,mock.patch.object(target.os,'kill',side_effect=b.kill),self.assertRaises(RuntimeError):target.run_lifecycle(ops,admission,{})
            self.assertTrue(b.waited);self.assertIsNone(ops.resolver);self.assertFalse(ops.pull_attempted)
            self.assertNotEqual(ops.record.call_args.args[0]['status'],'resolved-not-installed-not-qualified')

    def test_actual_snapshot_closure_includes_kernel_sid_and_microsecond_birth(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch:ops.start_owned()
            rows={p['pid']:p for p in b.rows}
            text='\n'.join(str(p['pid'])+' '+str(p['ppid'])+' '+str(p['pgid'])+' 100 S Sun Nov 15 03:48:02 2026 '+p['exe'] for p in b.rows)
            ops.command.return_value=(0,text,'');ops.command.reset_mock()
            ops.kernel_identity=mock.Mock();ops.kernel_identity.record.side_effect=lambda pid,**kwargs:copy.deepcopy(rows[pid])
            with mock.patch.object(target.os,'getsid',side_effect=lambda pid:rows[pid]['sid']):actual=target.NativeOps.snapshot(ops)
            self.assertEqual({p['pid']:p['start'] for p in actual},{pid:p['start'] for pid,p in rows.items()})
            self.assertTrue(all(p['sid']==75032 for p in actual));ops.command.assert_called_once()

    def test_transient_apple_helper_is_qualified_only_inside_our_created_session(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            b=Board();original=b.spawn
            def spawn(*args,**kwargs):
                creator=original(*args,**kwargs);b.rows.append(b.row(76000,75032,'/usr/bin/sw_vers'));return creator
            b.spawn=spawn;ops,b,patch=self.adapter(f,b)
            ops.command.side_effect=lambda args,**kwargs:(0,'','') if args[0]=='/usr/bin/codesign' else UNAVAILABLE
            with patch,mock.patch.object(Path,'is_file',return_value=True),mock.patch.object(Path,'resolve',lambda p,**kwargs:p):ops.start_owned()
            self.assertIn(76000,ops.owned_processes)
            self.assertTrue(any(call.args[0][-2:]==['=anchor apple','/usr/bin/sw_vers'] for call in ops.command.call_args_list))
            b.rows.append(b.row(76001,1,'/usr/bin/sw_vers',sid=444,pgid=444));ops.observe_owned()
            self.assertNotIn(76001,ops.owned_processes)
            b.rows=[p for p in b.rows if p['pid']!=76001];self.retire(ops,b);self.assertTrue(b.waited)

    def test_ui_acknowledgement_is_required_before_any_native_effect(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f);ops.a['_admission_path']='inert-admission'
            with mock.patch.object(target,'validated_registered_root'),mock.patch.object(target,'validated_admission_path'),self.assertRaisesRegex(RuntimeError,'UI effects'):
                ops.preflight()
            ops.command.assert_not_called();self.assertFalse(b.spawned)

    def test_sigchld_auto_reap_is_refused_before_creating_any_session(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch,mock.patch('owned_session_launch.signal.getsignal',return_value=signal.SIG_IGN),self.assertRaises(RuntimeError):ops.start_owned()
            self.assertFalse(b.spawned);self.assertFalse(ops.has_cleanup_authority())

    def test_threaded_controller_is_refused_before_process_creation(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,b,patch=self.adapter(f)
            with patch,mock.patch('owned_session_launch.threading.active_count',return_value=2),self.assertRaises(RuntimeError):ops.start_owned()
            self.assertFalse(b.spawned);self.assertFalse(ops.has_cleanup_authority())

    def test_creator_live_to_zombie_keeps_high_resolution_reserved_identity(self):
        identity=object.__new__(DarwinIdentity)
        live=BSDInfo();live.pid=75032;live.ppid=1;live.pgid=75032;live.start_sec=123;live.start_usec=45;live.status=2
        dead=BSDInfo();ctypes.memmove(ctypes.byref(dead),ctypes.byref(live),ctypes.sizeof(live));dead.status=5
        identity.bsd=mock.Mock(side_effect=[live,dead]);identity.path=mock.Mock(return_value=None)
        with mock.patch('darwin_process_identity.os.getsid',side_effect=ProcessLookupError):row=identity.record(75032,creator_sid=75032,creator_executable=EXE)
        self.assertEqual(row['sid'],75032);self.assertEqual(row['start'],'123.000045');self.assertEqual(row['state'],'Z');self.assertEqual(row['exe'],EXE)

    def test_kernel_record_layout_and_pid_birth_race_are_explicit(self):
        self.assertEqual(ctypes.sizeof(BSDInfo),136);self.assertEqual(BSDInfo.start_sec.offset,120)
        identity=object.__new__(DarwinIdentity)
        a=BSDInfo();a.pid=10;a.ppid=1;a.pgid=10;a.start_sec=123;a.start_usec=1
        b=BSDInfo();ctypes.memmove(ctypes.byref(b),ctypes.byref(a),ctypes.sizeof(a));b.start_usec=2
        identity.bsd=mock.Mock(side_effect=[a,b]);identity.path=mock.Mock(return_value=EXE)
        with mock.patch('darwin_process_identity.os.getsid',return_value=10),self.assertRaises(RuntimeError):identity.record(10)


if __name__=='__main__':unittest.main()
