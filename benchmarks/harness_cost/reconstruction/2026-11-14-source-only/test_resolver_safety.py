"""Execute only the proposed supervisor tail against owned stub commands.

No Docker/VM/resolver/network work is dispatched. Leaked stub processes from a
RED reproduction are reaped by their exact recorded PID in the test's finally.
"""
import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


def proposed_code():
    return (HERE / 'RESOLVER-COMMAND.md').read_text().split('```python\n', 1)[1].split('\n```', 1)[0]


def functions(code):
    tree = ast.parse(code)
    return '\n\n'.join(ast.get_source_segment(code, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))


class ResolverSafetyTests(unittest.TestCase):
    def test_optimization_guard_precedes_effects(self):
        code = proposed_code()
        prefix = code.split('R = pathlib.Path.cwd().resolve()', 1)[0]
        for flags, optimized in (([], False), (['-O'], True), (['-OO'], True)):
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, *flags, '-c', prefix + '\nprint("EFFECT")'], capture_output=True, text=True, timeout=5)
                self.assertEqual('EFFECT' in result.stdout, not optimized)
        env = dict(os.environ, PYTHONOPTIMIZE='1')
        result = subprocess.run([sys.executable, '-c', prefix + '\nprint("EFFECT")'], env=env, capture_output=True, text=True, timeout=5)
        self.assertNotIn('EFFECT', result.stdout)
        self.assertNotEqual(result.returncode, 0)

    def exercise_supervisor(self, terminate):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            a = Path(folder)
            client = a / 'client.py'
            client.write_text('import os,pathlib,sys,time\np=pathlib.Path(sys.argv[1]);(p/"client.pid").write_text(str(os.getpid()));(p/"container.cid").write_text("owned-stub");time.sleep(30)\n')
            docker = a / 'stub_container.py'
            docker.write_text('import pathlib,sys\np=pathlib.Path(sys.argv[1]);args=sys.argv[2:]\nif args[0]=="rm": (p/"removed").write_text(args[-1]);sys.exit(0)\nif (p/"removed").exists(): print("No such object",file=sys.stderr);sys.exit(1)\nsys.exit(0)\n')
            code = proposed_code()
            tail = code[code.index("(A / 'command.json').write_text"):]
            prefix = 'import hashlib,json,os,pathlib,resource,signal,subprocess,sys,time\n' + functions(code)
            prefix += '\nA=pathlib.Path(' + repr(str(a)) + ');source_head="0"*40\n'
            prefix += 'docker_raw=A/"stub.raw";raw_baseline=0\ndef disk_sample(*args): return {"docker_raw_allocated_bytes":0,"free_bytes":40*1024**3}\n'
            prefix += 'cmd=[sys.executable,' + repr(str(client)) + ',str(A)];D=[sys.executable,' + repr(str(docker)) + ',str(A)]\n'
            if not terminate:
                prefix += '_clock=time.monotonic;time.monotonic=lambda:_clock()*10000\n'
            proc = subprocess.Popen([sys.executable, '-c', prefix + tail], stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not (a / 'client.pid').exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue((a / 'client.pid').exists())
                child_pid = int((a / 'client.pid').read_text())
                if terminate:
                    proc.send_signal(signal.SIGTERM)
                proc.communicate(timeout=10)
                gone = False
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    gone = True
                self.assertTrue(gone, 'owned stub client leaked')
                self.assertEqual((a / 'removed').read_text(), 'owned-stub')
                self.assertTrue((a / 'receipt.json').exists())
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate(timeout=5)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_outer_sigterm_reaps_owned_client_and_container(self):
        self.exercise_supervisor(True)

    def test_outer_timeout_reaps_owned_client_and_container(self):
        self.exercise_supervisor(False)

    def test_disk_fences_use_allocated_growth_and_free_space(self):
        namespace = {}
        exec('import os,time\n' + functions(proposed_code()), namespace)
        check = namespace['check_disk']
        check({'docker_raw_allocated_bytes': 5*1024**3, 'free_bytes': 25*1024**3}, 0)
        for sample in ({'docker_raw_allocated_bytes': 5*1024**3+1, 'free_bytes': 30*1024**3},
                       {'docker_raw_allocated_bytes': 0, 'free_bytes': 25*1024**3-1}):
            with self.assertRaises(RuntimeError):
                check(sample, 0)

    def test_inner_optimization_guard_precedes_effects(self):
        tree = ast.parse(proposed_code())
        inner = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INNER' for t in n.targets))
        prefix = inner.split('child = None', 1)[0]
        for flags, environment in (([], {}), (['-O'], {}), ([], {'PYTHONOPTIMIZE':'1'})):
            result = subprocess.run([sys.executable, *flags, '-c', prefix+'\nprint("EFFECT")'], env=dict(os.environ, **environment), capture_output=True, text=True, timeout=5)
            self.assertEqual('EFFECT' in result.stdout, not flags and not environment)

    def test_independent_container_deadline_exists(self):
        tree = ast.parse(proposed_code())
        inner = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'INNER' for t in n.targets))
        self.assertIn('signal.alarm(840)', inner)
        self.assertIn('os.killpg', inner)
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            path = Path(folder) / 'child.pid'
            handler = next(n for n in ast.parse(inner).body if isinstance(n, ast.FunctionDef) and n.name == 'container_deadline')
            script = 'import os,signal,subprocess,sys,time,pathlib\n' + ast.get_source_segment(inner, handler)
            payload = 'import os,time,pathlib;pathlib.Path('+repr(str(path))+').write_text(str(os.getpid()));time.sleep(30)'
            script += '\nchild=subprocess.Popen([sys.executable,"-c",'+repr(payload)+'],start_new_session=True)\nsignal.signal(signal.SIGALRM,container_deadline)\nsignal.alarm(1)\nchild.wait()\n'
            proc = subprocess.Popen([sys.executable, '-c', script], start_new_session=True)
            child_pid = None
            try:
                self.assertEqual(proc.wait(timeout=5), 124)
                child_pid = int(path.read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if child_pid is None and path.exists():
                    child_pid = int(path.read_text())
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
