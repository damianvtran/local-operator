"""R14: real NativeOps.info AND command dispatch; only Popen/OS are inert."""
import json
import os
import subprocess
import tempfile
import types
import unittest
from unittest import mock
import lifecycle_proposal as target
import test_causal_startup as prior

UNAVAILABLE=(1,'','Cannot connect to the Docker daemon at unix:///var/run/docker.sock.')
AVAILABLE=(0,json.dumps({'MemTotal':4_000_000_000,'NCPU':2}),'')


class EngineHistoryTests(unittest.TestCase):
    def adapter(self,folder):
        ops,d,patch=prior.CausalStartupTests().adapter(folder)
        with patch:ops.start_owned()
        ops.command=types.MethodType(target.NativeOps.command,ops)
        return ops,d

    def transport(self,ops,d,replies):
        class Process:
            pid=990001
            def __init__(self,reply):self.reply=reply;self.returncode=None
            def poll(self):return self.returncode
            def communicate(self,timeout=None):
                if isinstance(self.reply,BaseException):
                    error=self.reply;self.returncode=0
                    if isinstance(error,subprocess.TimeoutExpired):d.time+=1000
                    raise error
                self.returncode,out,err=self.reply;return out,err
        responses=iter(replies)
        def popen(*args,**kwargs):
            self.assertTrue(ops.engine_query_inflight,'must be conservative BEFORE actual command dispatch')
            return Process(next(responses))
        return mock.patch.object(target.subprocess,'Popen',side_effect=popen)

    def test_success_then_journal_failure_retains_available_before_disconnect(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f);original=ops.startup_event
            def event(row):
                if row['event']=='first-api-readiness-info':raise OSError('inert readiness journal failure')
                original(row)
            ops.startup_event=event
            with self.transport(ops,d,[AVAILABLE,UNAVAILABLE]):
                with self.assertRaises(OSError):ops.info()
                self.assertTrue(ops.daemon_observed_available)
                ops.begin_cleanup()
                with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertFalse(d.terminated)

    def test_dispatch_deadline_failure_is_unknown_before_disconnect(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f)
            with self.transport(ops,d,[subprocess.TimeoutExpired('INERT',1),UNAVAILABLE]):
                with self.assertRaises(RuntimeError):ops.info()
                self.assertTrue(ops.engine_history_unknown)
                ops.begin_cleanup()
                with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertFalse(d.terminated)

    def test_signal_and_malformed_response_history_cannot_be_erased(self):
        for reply in (KeyboardInterrupt('inert signal'),(0,'not-json',''),(0,'{}',''),(0,'[]',''),(1,'','permission denied')):
            with self.subTest(reply=type(reply).__name__),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                ops,d=self.adapter(f)
                with self.transport(ops,d,[reply,UNAVAILABLE,UNAVAILABLE]):
                    with self.assertRaises(BaseException):ops.info()
                    self.assertTrue(ops.engine_history_unknown)
                    ops.begin_cleanup()
                    for _ in range(2):
                        with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
                        self.assertTrue(ops.engine_history_unknown)
                self.assertFalse(d.terminated)

    def test_cleanup_query_exception_also_latches_unknown_history(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f);ops.begin_cleanup()
            with self.transport(ops,d,[OSError('inert dispatch failure'),UNAVAILABLE]):
                with self.assertRaises(OSError):ops.quit_owned(ops.owner)
                self.assertTrue(ops.engine_history_unknown)
                with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertFalse(d.terminated)

    def test_actual_successful_info_then_disconnect_is_not_never_ready(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f)
            with self.transport(ops,d,[AVAILABLE,UNAVAILABLE]):
                self.assertEqual(ops.info()['NCPU'],2);ops.begin_cleanup()
                with self.assertRaises(RuntimeError):ops.quit_owned(ops.owner)
            self.assertTrue(ops.daemon_observed_available);self.assertFalse(d.terminated)

    def test_recognized_never_ready_reply_keeps_owned_cleanup_exception(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
            ops,d=self.adapter(f);original=ops.startup_event
            def event(row):
                if row['event']=='first-api-readiness-info':raise OSError('inert journal failure AFTER classified unavailable')
                original(row)
            ops.startup_event=event
            with self.transport(ops,d,[UNAVAILABLE,UNAVAILABLE]):
                with self.assertRaises(OSError):ops.info()
                self.assertFalse(ops.daemon_observed_available);self.assertFalse(ops.engine_history_unknown)
                ops.begin_cleanup();ops.quit_owned(ops.owner)
            self.assertTrue(d.terminated)


if __name__=='__main__':unittest.main()
