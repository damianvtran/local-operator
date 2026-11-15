"""Final03 authority with REAL private temp files, actual source gate, inert effects.

Only pwd/clock identity and effect factories are mocked. No real registered HOLD
claim, Docker/native helper, network, account, settings or model is touched.
"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import lifecycle_proposal as life
import final_attempt_admission as auth
import test_lifecycle_proposal as fixtures

HERE=Path(__file__).resolve().parent
KIT=HERE.parents[1]


class FinalAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
        self.home=Path(self.temp.name);self.root=self.home/'benchmark-runs/reconstruction-20261114-source-v1'
        self.root.mkdir(parents=True,mode=0o700);(self.root/'admissions').mkdir(mode=0o700)
        self.path=self.root/'admissions/linux-amd64-03.json'
        self.manifest=HERE/'generation-overlay-v4.json';self.source_sha=os.environ['BENCH_TEST_GENERATION_SHA256'];self.head='a'*40
        self.pwd=patch.object(auth.pwd,'getpwuid',return_value=types.SimpleNamespace(pw_dir=str(self.home)));self.pwd.start()
        now=auth.utc();end=now+datetime.timedelta(minutes=60)
        self.base={
            'attempt_id':auth.ATTEMPT,'scope':'resolver-only','go':True,
            'operator_decisions':copy.deepcopy(auth.DECISIONS),'limits':copy.deepcopy(auth.LIMITS),
            'carrier':auth.CARRIER,'platform':'linux/amd64','runtime_qualified':False,
            'operator_ack_id':life.ACK,'operator_ack_quote':life.ACK_QUOTE,'prestart_cap_validation':'skipped-with-operator-ack',
            'exclusive_desktop_lease':True,'direct_backend_ui_effects':'acknowledged-unqualified-default-backend',
            'docker_raw_path':str(self.home/'Docker.raw'),
            'slot_start_utc':(now-datetime.timedelta(seconds=1)).isoformat(),'slot_end_utc':end.isoformat(),
            'valid_until_utc':end.isoformat(),'fleet_mb':20000,'fresh_memory_go':True,'authorize_digest_pull':False,
            'desk_go':{'receipt_id':'inert-fixture-go-0001','source_commit':self.head,
                'source_generation_sha256':self.source_sha,'issued_utc':(now-datetime.timedelta(seconds=1)).isoformat(),
                'start_by_utc':(now+datetime.timedelta(minutes=5)).isoformat(),'expires_utc':end.isoformat()},
            'full_scope_review':{'kind':'full-scope','clean':True,'reviewer':'independent-inert-fixture',
                'source_commit':self.head,'source_generation_sha256':self.source_sha,
                'file_sha256':{n:auth.digest((HERE/n).read_bytes()) for n in ('lifecycle_proposal.py','RESOLVER-COMMAND.md','resolver-pins.json','launch_services_broker.py','owned_session_launch.py','darwin_process_identity.py')}}}
        self.effects=[];self.save(self.base)

    def tearDown(self):
        for permit in list(auth._ACTIVE.values()):auth.retire_permit(permit)
        auth._ACTIVE.clear();self.pwd.stop();self.temp.cleanup()

    def save(self,data):
        raw=(json.dumps(data,sort_keys=True)+'\n').encode()
        self.path.write_bytes(raw);self.path.chmod(0o600);self.go_sha=auth.digest(raw)

    def claim(self):
        return auth.claim_final_attempt(self.path,self.go_sha,self.manifest,self.source_sha,self.head)

    def drive(self,fail=False):
        data,permit=self.claim();auth.begin_controller(permit);auth.begin_native_start(permit)
        self.effects.append('INERT-effect')
        if fail:raise KeyboardInterrupt('inert canceled final attempt')
        return permit

    def test_one_correct_claim_once_across_restart_and_exception(self):
        permit=self.drive();self.assertEqual(self.effects,['INERT-effect'])
        with self.assertRaises(ValueError):auth.begin_controller(permit)
        with self.assertRaises(ValueError):auth.begin_native_start(permit)
        with self.assertRaises(FileExistsError):self.drive()
        auth._ACTIVE.clear()  # simulate fresh interpreter registry, not release
        with self.assertRaises(FileExistsError):self.drive()
        self.assertEqual(self.effects,['INERT-effect'])
        claim=self.root/'claims/linux-amd64-03.claim.json'
        self.assertEqual(claim.stat().st_mode&0o777,0o600)

    def test_concurrent_claim_has_exactly_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        def compete():
            try:self.claim();return 'claimed'
            except FileExistsError:return 'rejected'
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:compete(),range(2)))
        self.assertEqual(sorted(results),['claimed','rejected']);self.assertEqual(self.effects,[])

    def test_cancellation_burns_claim_without_retry(self):
        with self.assertRaises(KeyboardInterrupt):self.drive(fail=True)
        self.assertTrue((self.root/'claims/linux-amd64-03.claim.json').exists())
        with self.assertRaises(FileExistsError):self.drive()
        self.assertEqual(len(self.effects),1)

    def test_invalid_go_decisions_namespace_scope_digest_head_expiry_no_effect(self):
        variants=[]
        for key,value in [('attempt_id','linux-amd64-02'),('scope','paid-campaign'),('go',False),('operator_decisions',{'verified':True}),('platform','linux/arm64'),('carrier','python:latest'),('runtime_qualified',True),('fresh_memory_go',False),('fleet_mb',26001),('exclusive_desktop_lease',False),('operator_ack_quote','verified'),('vm_caps_verified_before_start',True)]:
            item=copy.deepcopy(self.base);item[key]=value;variants.append(item)
        item=copy.deepcopy(self.base);item['operator_decisions']['a-9509']['answer']='verified';variants.append(item)
        item=copy.deepcopy(self.base);item['limits']['vm_rss_max_bytes']=8600000000;variants.append(item)
        item=copy.deepcopy(self.base);item['limits']['container_cpus']=True;variants.append(item)
        for field,value in [('source_commit','b'*40),('source_generation_sha256','0'*64),('start_by_utc',(auth.utc()-datetime.timedelta(seconds=5)).isoformat()),('issued_utc',(auth.utc()-datetime.timedelta(days=1)).isoformat()),('expires_utc',(auth.utc()-datetime.timedelta(seconds=1)).isoformat())]:
            item=copy.deepcopy(self.base);item['desk_go'][field]=value;variants.append(item)
        item=copy.deepcopy(self.base);item['full_scope_review']['kind']='delta-only';variants.append(item)
        item=copy.deepcopy(self.base);item['full_scope_review']['file_sha256']={};variants.append(item)
        for index,item in enumerate(variants):
            with self.subTest(index=index):
                self.save(item)
                with self.assertRaises((ValueError,FileNotFoundError)):self.drive()
                self.assertEqual(self.effects,[])
                self.assertFalse((self.root/'claims/linux-amd64-03.claim.json').exists())
        self.save(self.base)
        with self.assertRaises(ValueError):auth.claim_final_attempt(self.path,'0'*64,self.manifest,self.source_sha,self.head)
        with self.assertRaises(ValueError):auth.claim_final_attempt(self.path,self.go_sha,self.manifest,self.source_sha,'b'*40)
        self.assertEqual(self.effects,[])

    def test_go_mutation_private_modes_and_forged_permit_reject(self):
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):self.claim()
        self.path.chmod(0o600);data,permit=self.claim()
        with self.assertRaises(ValueError):auth.validate_live_permit({'verified':True})
        changed=copy.deepcopy(self.base);changed['desk_go']['receipt_id']='changed-receipt';self.save(changed)
        with self.assertRaises(ValueError):auth.validate_live_permit(permit)
        self.assertEqual(self.effects,[])

    def test_existing_partial_claim_and_symlink_are_terminal(self):
        directory=self.root/'claims';directory.mkdir(mode=0o700)
        target=directory/'linux-amd64-03.claim.json';target.write_bytes(b'partial');target.chmod(0o600)
        with self.assertRaises(FileExistsError):self.claim()
        self.assertEqual(target.read_bytes(),b'partial')
        # No deletion/replacement here: the incomplete claim itself is terminal.
        self.assertEqual(self.effects,[])

    def test_symlinked_admission_cannot_mint_claim(self):
        target=self.root/'fixture-go.json';self.path.rename(target);self.path.symlink_to(target)
        with self.assertRaises(ValueError):self.claim()
        self.assertEqual(self.effects,[])
        self.assertFalse((self.root/'claims/linux-amd64-03.claim.json').exists())

    def test_live_handoff_issue_is_monotone_and_claim_has_no_bearer(self):
        data,permit=self.claim();auth.begin_controller(permit);auth.begin_native_start(permit)
        fd=auth.issue_resolver_handoff(permit)
        try:
            with self.assertRaises(ValueError):auth.issue_resolver_handoff(permit)
            self.assertNotIn(permit.secret.hex(),permit.path.read_text())
            self.assertNotIn('secret',permit.path.read_text())
        finally:os.close(fd)
        auth.retire_permit(permit)
        self.assertTrue(auth.marker_path(permit,'retired').exists())
        self.assertEqual(self.effects,[])

    def test_unfilled_or_regular_nonlive_descriptors_cannot_authorize(self):
        readfd,writefd=os.pipe()
        try:
            with self.assertRaises(ValueError):auth.consume_resolver_handoff(readfd,self.manifest,self.source_sha,self.path)
        finally:os.close(writefd)
        fd=os.open(self.path,os.O_RDONLY)
        with self.assertRaises(ValueError):auth.consume_resolver_handoff(fd,self.manifest,self.source_sha,self.path)
        self.assertEqual(self.effects,[])

    def test_actual_cli_with_inert_driver_claims_once_and_missing_go_never_constructs_native(self):
        argv=['lifecycle_proposal.py','--execute','--admission',str(self.path),
              '--admission-sha256',self.go_sha,'--reviewed-source-head',self.head,
              '--source-generation',str(self.manifest),'--source-generation-sha256',self.source_sha]
        def factory(admission,pins,permit):
            auth.begin_native_start(permit)
            return types.SimpleNamespace(final_permit=permit)
        def inert(ops,admission,pins):auth.validate_live_permit(ops.final_permit,starting=True);self.effects.append('INERT-cli')
        with patch.object(Path,'home',return_value=self.home),patch.object(sys,'argv',argv),patch.object(life,'NativeOps',side_effect=factory),patch.object(life,'run_lifecycle',side_effect=inert),patch.object(life.signal,'alarm'),patch.object(life.signal,'signal'):
            life.main()
            with self.assertRaises(FileExistsError):life.main()
        self.assertEqual(self.effects,['INERT-cli'])

    def test_actual_cli_missing_and_forged_go_have_zero_native_construction(self):
        for argv in (['lifecycle','--execute','--source-generation',str(self.manifest),'--source-generation-sha256',self.source_sha],
                     ['lifecycle','--execute','--admission',str(self.path),'--admission-sha256','0'*64,'--reviewed-source-head',self.head,'--source-generation',str(self.manifest),'--source-generation-sha256',self.source_sha]):
            with patch.object(Path,'home',return_value=self.home),patch.object(sys,'argv',argv),patch.object(life,'NativeOps',side_effect=AssertionError('native effect forbidden')):
                with self.assertRaises((ValueError,RuntimeError)):life.main()
        self.assertEqual(self.effects,[])


class CapacityTests(unittest.TestCase):
    def test_configured_int_bounds_do_not_change_rss_cap(self):
        self.assertEqual(life.CAP,4300000000)
        for memory,cpu in ((8200000000,14),(8600000000,14),(4000000000,2)):
            self.assertEqual(life.verify_caps({'MemTotal':memory,'NCPU':cpu})['MemTotal'],memory)
        for info in (None,[],True,'invalid'):
            with self.assertRaises(RuntimeError):life.verify_caps(info)
        for memory,cpu in ((8600000001,14),(8200000000,15),(True,14),(8200000000,True),(8200000000.0,14),(8200000000,'14'),(0,14)):
            with self.subTest(memory=memory,cpu=cpu):
                with self.assertRaises(RuntimeError):life.verify_caps({'MemTotal':memory,'NCPU':cpu})

    def test_actual_lifecycle_configured_overcap_and_rss_overcap_before_pull_or_container(self):
        for memory,cpu,rss in ((8600000001,14,1),(8200000000,15,1),(8200000000,14,4300000001)):
            ops=fixtures.StubOps(memory=memory);ops.info=lambda:{'MemTotal':memory,'NCPU':cpu};ops.vm_helper=lambda owner:{'rss_bytes':rss}
            with self.assertRaises(RuntimeError):life.run_lifecycle(ops,fixtures.admission(),{'base_image':'python@sha256:'+'a'*64,'uv_wheel':{'sha256':'c'*64}},require_runtime_scope=False)
            self.assertNotIn('PULL',ops.events);self.assertNotIn('resolver',ops.events);self.assertIn('quit-owned',ops.events)

    def test_exact_rss_boundary_allowed_with_larger_configured_capacity(self):
        ops=fixtures.StubOps(memory=8200000000);ops.info=lambda:{'MemTotal':8200000000,'NCPU':14};ops.vm_helper=lambda owner:{'rss_bytes':4300000000}
        result=life.run_lifecycle(ops,fixtures.admission(),{'base_image':'python@sha256:'+'a'*64,'uv_wheel':{'sha256':'c'*64}},require_runtime_scope=False)
        self.assertEqual(result,'resolved-not-installed-not-qualified');self.assertIn('resolver',ops.events)


if __name__=='__main__':unittest.main()
