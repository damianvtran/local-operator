"""R8: actual adapter launch environment -> emitted supervisor read fragment.

No Docker/manager/helper executes. Fixtures are text, not a real uv artifact.
The account-home lookup alone is mocked, independently of isolated CLI HOME.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import lifecycle_proposal as target


class RegisteredRootTests(unittest.TestCase):
    def fixture(self, home):
        root=home/'benchmark-runs/reconstruction-20261114-source-v1'
        directory=root/'lifecycle/linux-amd64-01';directory.mkdir(parents=True)
        (directory/'docker-home').mkdir()
        (directory/'startup-owner.json').write_text('{"kind":"source-only-fixture"}')
        (root/'desk-admission.json').write_text('{"attempt_id":"linux-amd64-01","kind":"source-only-fixture"}')
        (root/'artifacts').mkdir();(root/'artifacts/source-fixture.txt').write_bytes(b'not a wheel; source fixture only')
        pins={'uv_wheel':{'filename':'source-fixture.txt','sha256':hashlib.sha256(b'not a wheel; source fixture only').hexdigest()}}
        ops=object.__new__(target.NativeOps);ops.root=root;ops.directory=directory;ops.d=['NO-DOCKER']
        ops.command=mock.Mock(return_value=(0,'',''))
        with mock.patch.object(target.subprocess,'Popen') as launch:
            ops.begin_resolver()
        return root,pins,launch.call_args.kwargs['env'],(directory/'resolver-supervisor.py').read_text()

    def fragment(self, code):
        selected=[]
        for node in ast.parse(code).body:
            names={t.id for t in node.targets if isinstance(t,ast.Name)} if isinstance(node,ast.Assign) else set()
            if names & {'ROOT','admission','lifecycle','W'}:selected.append(node)
            elif isinstance(node,ast.Assert) and any(isinstance(x,ast.Name) and x.id=='W' for x in ast.walk(node)):selected.append(node)
            elif isinstance(node,ast.If) and isinstance(node.test,ast.BoolOp) and any(isinstance(x,ast.Name) and x.id=='lifecycle' for x in ast.walk(node.test)):selected.append(node)
        self.assertEqual(sum(isinstance(n,ast.Assign) for n in selected),4)
        return compile(ast.Module(body=selected,type_ignores=[]),'<actual-supervisor-root-admission-artifact-fragment>','exec')

    def run_fragment(self, fragment, env, pins):
        scope={'pathlib':__import__('pathlib'),'os':os,'json':json,'hashlib':hashlib,'pins':pins}
        if hasattr(target,'validated_registered_root'):scope['validated_registered_root']=target.validated_registered_root
        with mock.patch.dict(os.environ,env,clear=True):exec(fragment,scope)
        return scope

    def test_actual_adapter_environment_reads_registered_admission_and_artifact(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            home=Path(folder).resolve()
            # Path.home remains environment-based; only the independent account
            # metadata provider uses this source-only fixture home.
            with mock.patch('pwd.getpwuid',return_value=SimpleNamespace(pw_dir=str(home))):
                root,pins,env,code=self.fixture(home)
                self.assertNotEqual(env['HOME'],str(home))
                self.assertNotIn('DOCKER_AUTH_CONFIG',env)
                result=self.run_fragment(self.fragment(code),env,pins)
                self.assertEqual(result['ROOT'],root)
                self.assertEqual(result['admission']['kind'],'source-only-fixture')
                self.assertEqual(result['W'].read_bytes(),b'not a wheel; source fixture only')

    def test_missing_wrong_relative_and_parent_escape_roots_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            home=Path(folder).resolve()
            with mock.patch('pwd.getpwuid',return_value=SimpleNamespace(pw_dir=str(home))):
                root,pins,env,code=self.fixture(home);fragment=self.fragment(code)
                for raw in (None,'relative/root',str(home/'other'),str(root/'..'/root.name)):
                    with self.subTest(raw=raw):
                        value=dict(env)
                        if raw is None:value.pop('BENCH_REGISTERED_ROOT',None)
                        else:value['BENCH_REGISTERED_ROOT']=raw
                        with self.assertRaises((RuntimeError,KeyError,FileNotFoundError)):self.run_fragment(fragment,value,pins)

    def test_symlink_escape_rejected_before_adapter_launch(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            home=Path(folder);real=home/'real';real.mkdir();alias=home/'alias';alias.symlink_to(real,target_is_directory=True)
            root=alias/'benchmark-runs/reconstruction-20261114-source-v1';root.mkdir(parents=True)
            ops=object.__new__(target.NativeOps);ops.root=root
            with mock.patch('pwd.getpwuid',return_value=SimpleNamespace(pw_dir=str(alias))),mock.patch.object(target.subprocess,'Popen') as launch:
                with self.assertRaises(RuntimeError):ops.preflight()
                with self.assertRaises(RuntimeError):ops.begin_resolver()
                launch.assert_not_called()


if __name__=='__main__':unittest.main()
