"""Exact source-gate and inert entrypoint regressions; no runtime dispatch.

Tamper filesystem reads, never the shared reviewer-frozen source bytes. The REAL
validator consumes each altered view. CLI tests run actual --verify-source paths
without Git, accounts, builds, services or NativeOps construction.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
R = KIT/'reconstruction/2026-11-14-source-only'
spec = importlib.util.spec_from_file_location('source_gate_under_test', KIT/'source_generation.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.overlay = KIT/gate.OVERLAY
        cls.trusted = os.environ['BENCH_TEST_GENERATION_SHA256']
        cls.raw = cls.overlay.read_bytes()
        cls.data = json.loads(cls.raw)

    def verify(self, trusted=None):
        return gate.validate_selection(self.overlay, self.trusted if trusted is None else trusted)

    @contextlib.contextmanager
    def tamper(self, changes):
        original_bytes = Path.read_bytes
        original_text = Path.read_text
        def altered_bytes(path):
            return changes.get(path, original_bytes(path))
        def altered_text(path, *args, **kwargs):
            if path in changes:
                return changes[path].decode('utf-8')
            return original_text(path, *args, **kwargs)
        with patch.object(Path,'read_bytes',altered_bytes), patch.object(Path,'read_text',altered_text):
            yield

    def test_exact_standalone_source_without_git(self):
        self.assertFalse((KIT/'.git').exists())
        with patch.object(subprocess,'run',side_effect=AssertionError('no Git/process allowed')), patch.object(subprocess,'Popen',side_effect=AssertionError('no process allowed')):
            result=self.verify()
        self.assertTrue(result['source_verified'])
        self.assertFalse(result['runtime_qualified'])
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['git_required'])

    def test_new_meter_new_tests_runtime_and_historical_tamper_reject(self):
        paths=[gate.METER,gate.METER_TEST,gate.HISTORICAL,gate.BASELINE,
               gate.R+'owned_session_launch.py','experimental/lop_edge.py','source_generation.py',
               'experimental/test_source_generation.py']
        for name in paths:
            with self.subTest(path=name):
                target=KIT/name
                with self.tamper({target:target.read_bytes()+b'\n# altered'}):
                    with self.assertRaises(ValueError):self.verify()

    def test_resealed_unrelated_runtime_and_restricted_source_still_reject(self):
        # Even a recomputed overlay digest cannot waive immutable baseline files.
        for name in (gate.R+'owned_session_launch.py','experimental/lop_edge.py'):
            with self.subTest(path=name):
                target=KIT/name;changed=target.read_bytes()+b'\n# altered'
                overlay=json.loads(self.raw);overlay['files'][name]=gate.record(changed)
                raw=json.dumps(overlay,sort_keys=True).encode()
                with self.tamper({target:changed,self.overlay:raw}):
                    with self.assertRaisesRegex(ValueError,'restricted unchanged baseline'):
                        self.verify(gate.digest(raw))

    def test_resealed_wiring_change_is_not_a_new_admission(self):
        name=gate.R+'lifecycle_proposal.py';target=KIT/name
        changed=target.read_bytes()+b'\n# undeclared runtime change'
        overlay=json.loads(self.raw);overlay['files'][name]=gate.record(changed)
        raw=json.dumps(overlay).encode()
        with self.tamper({target:changed,self.overlay:raw}):
            with self.assertRaisesRegex(ValueError,'reviewed wiring identity changed'):
                self.verify(gate.digest(raw))

    def test_overlay_record_digest_unknown_version_generation_and_wrong_base_reject(self):
        changes=[('schema_version',999),('schema_version',2.0),('base_commit','0'*40),
                 ('generation','old-default'),('baseline_sha256','0'*64),
                 ('runtime_qualified',True),('execution_authorized',True),
                 ('unknown_record',True)]
        for field,value in changes:
            with self.subTest(field=field,value=value):
                overlay=json.loads(self.raw);overlay[field]=value;raw=json.dumps(overlay).encode()
                with self.tamper({self.overlay:raw}):
                    with self.assertRaises(ValueError):self.verify()  # trusted digest mismatch
                    with self.assertRaises(ValueError):self.verify(gate.digest(raw))  # schema/base anchor

    def test_missing_unknown_selection_and_receipt_boolean_cannot_admit(self):
        for manifest,sha in ((None,None),(self.overlay,None),(R/'missing.json',self.trusted),(self.overlay,'0'*64)):
            with self.subTest(manifest=manifest):
                with self.assertRaises((ValueError,OSError)):
                    gate.validate_selection(manifest,sha)
        # An otherwise plausible receipt cannot turn an absent selection into proof.
        receipt={'source_verified':True,'runtime_qualified':True}
        self.assertTrue(receipt['source_verified'])
        with self.assertRaises(ValueError):gate.validate_selection(receipt.get('source_generation'),receipt.get('source_generation_sha256'))

    def test_missing_file_unknown_inventory_or_derived_manifest_reject(self):
        original=gate.public_files
        current=original(KIT)
        for actual in (current-{gate.METER},current|{'unlisted_runtime.py'}):
            with patch.object(gate,'public_files',return_value=actual):
                with self.assertRaises(ValueError):self.verify()
        target=KIT/gate.DERIVED
        with self.tamper({target:target.read_bytes().replace(b'\n',b'\r\n')}):
            with self.assertRaisesRegex(ValueError,'derived source manifest'):self.verify()
        with self.tamper({target:target.read_bytes()+b'\n'}):
            with self.assertRaisesRegex(ValueError,'derived source manifest'):self.verify()

    def test_actual_crlf_file_normalizes_in_old_read_text_but_new_gate_rejects(self):
        import tempfile
        target=KIT/gate.DERIVED;original=Path.read_bytes
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as directory:
            fixture=Path(directory)/'manifest.crlf';fixture.write_bytes(target.read_bytes().replace(b'\n',b'\r\n'))
            self.assertEqual(fixture.read_text(),target.read_text())  # actual old normalization repro
            self.assertNotEqual(fixture.read_bytes(),target.read_bytes())
            def actual_file_view(path):return original(fixture if path==target else path)
            with patch.object(Path,'read_bytes',actual_file_view):
                with self.assertRaisesRegex(ValueError,'derived source manifest'):
                    self.verify()

    def test_duplicate_json_and_unlisted_overlay_delta_reject(self):
        raw=self.raw.rstrip()[:-1]+b',"schema_version":2}'
        with self.tamper({self.overlay:raw}):
            with self.assertRaisesRegex(ValueError,'duplicate JSON'):self.verify(gate.digest(raw))
        overlay=json.loads(self.raw);overlay['changes'].append('experimental/lop_edge.py');raw=json.dumps(overlay).encode()
        with self.tamper({self.overlay:raw}):
            with self.assertRaisesRegex(ValueError,'unknown source delta'):self.verify(gate.digest(raw))

    def test_real_verify_only_clis_and_default_missing_selection(self):
        paths=[KIT/'source_generation.py',KIT/'experimental/coding/driver.py',
               KIT/'experimental/runner_v2.py',R/'lifecycle_proposal.py']
        for path in paths:
            with self.subTest(entrypoint=path.name):
                flags=[] if path.name=='source_generation.py' else ['--verify-source']
                command=[sys.executable,'-B',str(path),*flags]
                missing=subprocess.run(command,capture_output=True,text=True,timeout=15)
                self.assertNotEqual(missing.returncode,0)
                result=subprocess.run(command+['--source-generation',str(self.overlay),'--source-generation-sha256',self.trusted],capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                data=json.loads(result.stdout)
                self.assertTrue(data['source_verified'])
                self.assertFalse(data['runtime_qualified'])

    def test_native_execute_missing_protected_final_go_has_no_admission(self):
        result=subprocess.run([sys.executable,'-B',str(R/'lifecycle_proposal.py'),'--execute',
            '--source-generation',str(self.overlay),'--source-generation-sha256',self.trusted],
            capture_output=True,text=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('explicit desk admission path required',result.stderr)

    def test_actual_driver_preflight_rejects_receipt_before_legacy_dispatch(self):
        path=KIT/'experimental/coding/driver.py'
        spec=importlib.util.spec_from_file_location('actual_driver_gate_test',path)
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        with patch.object(driver,'block_paid_coding',side_effect=AssertionError('gate was bypassed')):
            with self.assertRaisesRegex(ValueError,'explicit NEW generation'):
                driver.preflight(types.SimpleNamespace(source_verified=True))

    def test_actual_native_preflight_ignores_forged_receipt_and_requires_live_permit(self):
        import runpy
        with patch.object(sys,'path',[str(R),*sys.path]):
            ns=runpy.run_path(str(R/'lifecycle_proposal.py'))
        ops=object.__new__(ns['NativeOps'])
        ops.a={'source_verified':True}
        with self.assertRaisesRegex(ValueError,'explicit NEW generation'):
            ops.preflight()
        ops.a={'_source_generation':str(self.overlay),'_source_generation_sha256':self.trusted}
        with self.assertRaisesRegex(ValueError,'active issuer permit required'):
            ops.preflight()

    def test_actual_extracted_supervisor_valid_source_still_stops_before_runtime(self):
        code=(R/'RESOLVER-COMMAND.md').read_text().split('```python\n',1)[1].split('\n```',1)[0]
        env={'BENCH_SOURCE_GENERATION':str(self.overlay),'BENCH_SOURCE_GENERATION_SHA256':self.trusted,'BENCH_FINAL_PERMIT_FD':''}
        # Actual producer supplies PYTHONPATH=R to the extracted child.
        with patch.dict(os.environ,env), patch.object(sys,'path',[str(R),*sys.path]), patch.object(Path,'cwd',return_value=R), patch.object(subprocess,'check_output',side_effect=AssertionError('command before hold')), patch.object(subprocess,'Popen',side_effect=AssertionError('process before hold')):
            with self.assertRaisesRegex(ValueError,'inherited one-use pipe missing'):
                exec(compile(code,'actual-resolver-supervisor','exec'),{'__name__':'source_only_probe'})

    def test_actual_extracted_supervisor_rejects_missing_gate_before_socket_or_commands(self):
        code=(R/'RESOLVER-COMMAND.md').read_text().split('```python\n',1)[1].split('\n```',1)[0]
        env={k:v for k,v in os.environ.items() if k not in ('BENCH_SOURCE_GENERATION','BENCH_SOURCE_GENERATION_SHA256')}
        with patch.dict(os.environ,env,clear=True), patch.object(Path,'cwd',return_value=R), patch.object(subprocess,'check_output',side_effect=AssertionError('command before gate')), patch.object(subprocess,'Popen',side_effect=AssertionError('process before gate')):
            with self.assertRaisesRegex(ValueError,'explicit NEW generation'):
                exec(compile(code,'actual-resolver-supervisor','exec'),{'__name__':'source_only_probe'})


if __name__=='__main__':
    unittest.main()
