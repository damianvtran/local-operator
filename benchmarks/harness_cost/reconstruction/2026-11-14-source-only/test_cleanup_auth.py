"""R6/R7: actual adapter methods with mocks/sentinel CLI, never Docker/keychain."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

import lifecycle_proposal as target


class CleanupTests(unittest.TestCase):
    def adapter(self, root):
        ops=object.__new__(target.NativeOps)
        ops.root=root;ops.resolver=mock.Mock();ops.d=['stub-container-cli'];ops.directory=root
        ops.now=lambda:1000.0;ops.deadline=0;ops.cleanup_until=1060
        ops.monitor_enabled=True;ops.monitoring=False;ops.next_rss=0;ops.owner={'pid':123}
        ops.samples=[];ops.raw=root/'Docker.raw';ops.raw.write_bytes(b'');ops.baseline=0;ops.raw_identity=None
        ops.sleep=lambda seconds:None
        ops.check_cleanup_resources=lambda:(_ for _ in ()).throw(RuntimeError('cleanup resource monitor failed'))
        ops.check_disk=lambda:(_ for _ in ()).throw(RuntimeError('expired work deadline'))
        ops.vm_helper=lambda owner:(_ for _ in ()).throw(RuntimeError('monitor unavailable'))
        return ops

    def test_monitor_failure_still_reaches_known_cid_removal_and_verification(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);cid=root/'resolver/linux-amd64-01/container.cid';cid.parent.mkdir(parents=True);cid.write_text('a'*64)
            ops=self.adapter(root);ops.resolver.poll.side_effect=[None,None,None,None,0];ops.resolver.pid=424242
            calls=[]
            def command(args,**kwargs):
                calls.append(args)
                return (1,'','No such object') if 'inspect' in args else (0,'','')
            ops.command=command
            with mock.patch.object(target.os,'killpg'):
                with self.assertRaisesRegex(RuntimeError,'cleanup resource monitor failed'):
                    ops.stop_owned_resolver()
            self.assertTrue(any(args[-3:]==['rm','--force','a'*64] for args in calls),calls)
            self.assertTrue(any(args[-3:]==['container','inspect','a'*64] for args in calls),calls)

    def test_expired_work_healthy_cleanup_resources_succeeds(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);cid=root/'resolver/linux-amd64-01/container.cid';cid.parent.mkdir(parents=True);cid.write_text('a'*64)
            ops=self.adapter(root);ops.next_rss=float('inf');ops.resolver.pid=424242
            ops.resolver.poll.side_effect=[None,None,0,0]
            ops.check_cleanup_resources=lambda:target.NativeOps.check_cleanup_resources(ops)
            ops.command=lambda args,**kw:(1,'','No such object') if 'inspect' in args else (0,'','')
            fs=mock.Mock(f_bavail=40*1024**3,f_frsize=1)
            with mock.patch.object(target.os,'killpg'),mock.patch.object(target.os,'statvfs',return_value=fs):
                ops.stop_owned_resolver()
            self.assertTrue(any(x.get('absence_verified') for x in ops.samples))

    def test_remove_error_still_attempts_authoritative_absence_check(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);cid=root/'resolver/linux-amd64-01/container.cid';cid.parent.mkdir(parents=True);cid.write_text('a'*64)
            ops=self.adapter(root);calls=[]
            def command(args,**kw):
                calls.append(args)
                if 'rm' in args:raise RuntimeError('remove transport failed')
                return 1,'','No such object'
            ops.command=command;ops.recover_owned_container()
            self.assertEqual(len(calls),2);self.assertTrue(ops.samples[-1]['absence_verified'])
            self.assertIn('transport failed',ops.samples[-1]['remove_error'])

    def test_invalid_cid_and_exhausted_cleanup_budget_never_spawn(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);cid=root/'resolver/linux-amd64-01/container.cid';cid.parent.mkdir(parents=True);cid.write_text('not-an-owned-cid')
            ops=self.adapter(root);ops.clients=[]
            with mock.patch.object(target.subprocess,'Popen') as launch:
                with self.assertRaises(RuntimeError):ops.recover_owned_container()
                cid.write_text('a'*64);ops.cleanup_until=999
                with self.assertRaises(RuntimeError):ops.recover_owned_container()
                launch.assert_not_called()

    def test_absent_cid_never_broad_cleans_and_daemon_error_is_unverified(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);ops=self.adapter(root);ops.command=mock.Mock()
            ops.recover_owned_container();ops.command.assert_not_called()
            cid=root/'resolver/linux-amd64-01/container.cid';cid.parent.mkdir(parents=True);cid.write_text('a'*64)
            ops.command.side_effect=[(0,'',''),(1,'','Cannot connect to daemon')]
            with self.assertRaisesRegex(RuntimeError,'absence not proven'):ops.recover_owned_container()
            self.assertFalse(ops.samples[-1]['absence_verified'])

    def test_actual_command_does_not_reenter_failed_work_monitor_in_cleanup(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            ops=self.adapter(Path(folder));ops.clients=[]
            process=mock.Mock();process.communicate.side_effect=[target.subprocess.TimeoutExpired('stub',0.5),('ok','')];process.poll.return_value=0;process.returncode=0
            with mock.patch.object(target.subprocess,'Popen',return_value=process),mock.patch.object(target.signal,'pthread_sigmask',return_value=set()):
                self.assertEqual(ops.command(['stub-command'])[1],'ok')
            self.assertEqual(process.communicate.call_count,2)


class AuthTests(unittest.TestCase):
    def test_actual_anonymous_config_suppresses_sentinel_autodetection(self):
        self.assertTrue(hasattr(target,'write_anonymous_cli_config'),'explicit anonymous config helper missing')
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);config=root/'docker-config';config.mkdir();home=root/'docker-home';home.mkdir()
            sentinel=root/'sentinel';driver=root/'driver.py';bin_dir=root/'bin';bin_dir.mkdir()
            helper=bin_dir/'docker-credential-osxkeychain'
            helper.write_text('#!'+sys.executable+'\nimport pathlib\npathlib.Path('+repr(str(sentinel))+').write_text("TEST-ONLY-HELPER")\n')
            helper.chmod(0o700)
            # This is a SOURCE-derived test driver, NOT Docker. Its helper is a
            # test-only marker callback, never the real macOS credential helper.
            driver.write_text('import json,os,pathlib,shutil,subprocess,sys\nc=json.loads(pathlib.Path(sys.argv[1]).read_text()); contains=bool(c.get("credsStore") or c.get("credHelpers") or c.get("auths"))\nif not contains:\n h=shutil.which("docker-credential-osxkeychain"); assert h and pathlib.Path(h).parent==pathlib.Path(sys.argv[2]),"refuse any real helper"\n subprocess.run([h,"get"],check=True)\nprint("auth-env-present" if "DOCKER_AUTH_CONFIG" in os.environ else "anonymous-env")\n')
            blank=config/'blank.json';blank.write_text('{}')
            import subprocess
            subprocess.run([sys.executable,str(driver),str(blank),str(bin_dir)],env={'PATH':str(bin_dir),'HOME':str(home)},capture_output=True,check=True)
            self.assertTrue(sentinel.exists(),'negative control did not detect old empty config')
            sentinel.unlink()  # own test-only marker
            target.write_anonymous_cli_config(config)
            ops=object.__new__(target.NativeOps);ops.directory=root;ops.cleanup_until=None;ops.owner=None;ops.clients=[];ops.monitor_enabled=False;ops.monitoring=False
            ops.now=target.time.monotonic
            real_environment=target.anonymous_cli_environment
            def environment(directory):
                value=real_environment(directory);value['PATH']=str(bin_dir)+os.pathsep+value['PATH'];return value
            with mock.patch.dict(os.environ,{'DOCKER_AUTH_CONFIG':'SYNTHETIC-ENV-SENTINEL'}),mock.patch.object(target,'anonymous_cli_environment',side_effect=environment):
                result=ops.command([sys.executable,str(driver),str(config/'config.json'),str(bin_dir)])
            self.assertFalse(sentinel.exists(),'actual adapter config allowed helper autodetection')
            self.assertIn('anonymous-env',result[1])
            self.assertEqual(json.loads((config/'config.json').read_text())['auths'],{'https://index.docker.io/v1/':{}})


if __name__=='__main__':unittest.main()
