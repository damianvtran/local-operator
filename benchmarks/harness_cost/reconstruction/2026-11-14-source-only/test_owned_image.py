"""R9 actual adapter cleanup with inert transports; never Docker or keychain."""
import unittest
import copy
import json
import signal
import os
import tempfile
from pathlib import Path
from unittest import mock
import lifecycle_proposal as target


class OwnedImageTests(unittest.TestCase):
    def test_image_cleanup_hook_is_before_quit_in_actual_lifecycle(self):
        import test_lifecycle_proposal as prior
        ops=prior.StubOps();ops.snapshot_assets=lambda:ops.events.append('snapshot-assets')
        ops.cleanup_owned_image=lambda:ops.events.append('cleanup-image')
        prior.LifecycleTests().run_case(ops)
        self.assertIn('cleanup-image',ops.events)
        self.assertLess(ops.events.index('snapshot-assets'),ops.events.index('resolver'))
        self.assertLess(ops.events.index('stop-resolver'),ops.events.index('cleanup-image'))
        self.assertLess(ops.events.index('cleanup-image'),ops.events.index('quit-owned'))

    def adapter(self,preexisting=False):
        old='sha256:'+'1'*64;new='sha256:'+'2'*64;shared='sha256:'+'3'*64
        ref='docker.io/library/python@sha256:'+'a'*64
        image={'Id':new,'Architecture':'amd64','Os':'linux','RootFS':{'Layers':[shared,'sha256:'+'4'*64]},'RepoTags':[],'RepoDigests':[ref]}
        baseline={'Id':old,'RootFS':{'Layers':[shared]},'RepoTags':['baseline:keep'],'RepoDigests':[]}
        state={'images':{old:baseline},'cache':['baseline-cache'],'containers':{}}
        if preexisting:state['images'][new]=copy.deepcopy(image)
        ops=target.NativeOps({}, {'base_image':ref});ops.d=['NO-DOCKER'];ops.cleanup_until=target.time.monotonic()+60;ops.next_rss=float('inf')
        calls=[];options={'partial':False,'rm_failure':False}
        def command(args,**kwargs):
            calls.append(args);a=args[1:]
            if a[:2]==['image','ls']:return 0,'\n'.join(state['images']),''
            if a[:2]==['system','df']:return 0,'\n'.join(json.dumps({'ID':x,'InUse':'true' if x.endswith('*') else 'false'}) for x in state['cache']),''
            if a[:2]==['ps','--all']:return 0,'\n'.join(state['containers']),''
            if a[:2]==['container','inspect']:return 0,json.dumps(state['containers'][a[-1]]),''
            if a[:2]==['image','inspect']:
                if '--format' in a:
                    ids=a[a.index('--format')+2:]
                    rows=[dict(Id=i,Layers=state['images'][i]['RootFS']['Layers'],RepoTags=state['images'][i]['RepoTags'],RepoDigests=state['images'][i]['RepoDigests']) for i in ids]
                    return 0,'\n'.join(json.dumps(x) for x in rows),''
                if new not in state['images']:return 1,'','No such image'
                return 0,json.dumps([state['images'][new]]),''
            if a and a[0]=='pull':
                if options['partial']:raise RuntimeError('inert partial pull transport failure')
                state['images'][new]=copy.deepcopy(image);return 0,'',''
            if a[:2]==['image','rm']:
                self.assertEqual(a,['image','rm',new])
                if options.get('foreign_during_rm'):
                    state['containers']['7'*64]={'Image':new,'Running':False}
                    raise RuntimeError('inert concurrent foreign use')
                if options['rm_failure']:raise RuntimeError('inert removal failure')
                state['images'].pop(new);return 0,'',''
            if a[:2]==['ps','-q']:return 0,'',''
            raise AssertionError('Unexpected inert CLI invocation: '+repr(a))
        ops.command=mock.Mock(side_effect=command);ops.check_disk=mock.Mock()
        ops.snapshot_assets()
        return ops,state,calls,options,old,new,image

    def test_actual_adapter_lifecycle_orders_recovery_image_retirement_then_quit(self):
        ops,state,calls,options,old,new,image=self.adapter();ops.assets_before=None
        ops.p['uv_wheel']={'sha256':'c'*64};ops.preflight=mock.Mock();ops.info=lambda:{'MemTotal':4_000_000_000,'NCPU':2}
        owner={'pid':123,'ppid':1,'start':'source-fixture','exe':'/Applications/Docker.app/Contents/MacOS/Docker','rss_kib':1}
        ops.owner=owner;ops.start_owned=lambda:owner;ops.snapshot=lambda:[]
        ops.vm_helper=lambda owner:{'rss_bytes':200_000_000};ops.wheel_hash=lambda:'c'*64
        ops.begin_resolver=mock.Mock();ops.resolver_done=lambda:True
        original_stop=ops.stop_owned_resolver;original_quit=ops.quit_owned
        def stop():calls.append(['event','recover-containers']);original_stop()
        def quit(owner):calls.append(['event','quit']);original_quit(owner)
        ops.stop_owned_resolver=stop;ops.quit_owned=quit
        ops.record=lambda value:setattr(ops,'unit_receipt',value)
        admission={'go':True,'scope':'resolver-only','authorize_digest_pull':True,'full_scope_review':{'kind':'full-scope','reviewer':'independent-source-fixture','allow_digest_pull':True}}
        with mock.patch.object(target.os,'kill') as kill:
            target.run_lifecycle(ops,admission,ops.p)
            kill.assert_not_called()
        self.assertLess(calls.index(['event','recover-containers']),calls.index(['NO-DOCKER','image','rm',new]))
        self.assertLess(calls.index(['NO-DOCKER','image','rm',new]),calls.index(['event','quit']))
        self.assertEqual(ops.unit_receipt['status'],'resolved-not-installed-not-qualified')

    def test_actual_new_image_removed_exactly_and_shared_layers_preserved(self):
        ops,state,calls,options,old,new,image=self.adapter();ops.pull();ops.cleanup_owned_image()
        self.assertEqual(set(state['images']),{old});self.assertEqual(state['cache'],['baseline-cache'])
        self.assertEqual(ops.owned_image_id,new)
        self.assertTrue(ops.asset_receipt['shared_layers_preserved'])
        removals=[a for a in calls if 'rm' in a];self.assertEqual(removals,[['NO-DOCKER','image','rm',new]])
        self.assertFalse(any('prune' in a or '--force' in a for a in calls))

    def test_preexisting_carrier_preserved_even_after_exact_pull(self):
        ops,state,calls,options,old,new,image=self.adapter(preexisting=True);ops.pull();ops.cleanup_owned_image()
        self.assertIn(new,state['images']);self.assertIsNone(ops.owned_image_id)
        self.assertFalse(any('rm' in a for a in calls));self.assertIn(new,ops.asset_receipt['after']['images'])

    def test_partial_pull_is_unverified_not_pruned(self):
        ops,state,calls,options,old,new,image=self.adapter();options['partial']=True
        with self.assertRaises(RuntimeError):ops.pull()
        with self.assertRaisesRegex(RuntimeError,'partial pull'):ops.cleanup_owned_image()
        self.assertFalse(any('rm' in a or 'prune' in a for a in calls));self.assertFalse(ops.foreign_asset_activity)

    def test_failed_pull_materialized_exact_new_id_can_be_retired(self):
        ops,state,calls,options,old,new,image=self.adapter();options['partial']=True
        with self.assertRaises(RuntimeError):ops.pull()
        state['images'][new]=copy.deepcopy(image)  # Engine completed despite lost client.
        ops.cleanup_owned_image()
        self.assertNotIn(new,state['images']);self.assertFalse(ops.pull_complete)
        self.assertEqual(ops.owned_image_id,new)

    def test_foreign_users_cache_and_image_aliases_are_preserved(self):
        for kind in ('container','cache','tag','other-image'):
            with self.subTest(kind=kind):
                ops,state,calls,options,old,new,image=self.adapter();ops.pull()
                if kind=='container':state['containers']['5'*64]={'Image':new,'Running':False}
                elif kind=='cache':state['cache'].append('foreign-build-cache')
                elif kind=='tag':state['images'][new]['RepoTags']=['foreign:keep']
                else:state['images']['sha256:'+'6'*64]=dict(image,Id='sha256:'+'6'*64)
                with self.assertRaises(RuntimeError):ops.cleanup_owned_image()
                self.assertTrue(ops.foreign_asset_activity);self.assertFalse(any('rm' in a for a in calls))
                with self.assertRaisesRegex(RuntimeError,'foreign asset activity'):ops.quit_owned({'fixture':True})

    def test_cleanup_failure_not_success_and_absence_still_checked(self):
        ops,state,calls,options,old,new,image=self.adapter();ops.pull();options['rm_failure']=True
        with self.assertRaises(RuntimeError):ops.cleanup_owned_image()
        self.assertIn(new,ops.asset_receipt['after']['images']);self.assertIsNotNone(ops.asset_receipt['removal_error'])

    def test_foreign_use_after_failed_remove_blocks_desktop_quit(self):
        ops,state,calls,options,old,new,image=self.adapter();ops.pull();options['foreign_during_rm']=True
        with self.assertRaisesRegex(RuntimeError,'asset namespace changed'):ops.cleanup_owned_image()
        self.assertTrue(ops.foreign_asset_activity);self.assertIn(new,state['images'])
        with self.assertRaisesRegex(RuntimeError,'foreign asset activity'):ops.quit_owned({'fixture':True})

    def test_expired_budget_cannot_start_asset_cleanup_commands(self):
        ops,state,calls,options,old,new,image=self.adapter();ops.pull();count=len(calls)
        ops.cleanup_until=ops.now()-1
        with self.assertRaises(RuntimeError):ops.cleanup_owned_image()
        self.assertEqual(len(calls),count)

    def test_signal_during_pull_still_runs_asset_cleanup_before_quit(self):
        import test_lifecycle_proposal as prior
        ops=prior.StubOps(missing=True)
        ops.pull=lambda:(_ for _ in ()).throw(KeyboardInterrupt('source-only SIGTERM fixture'))
        with self.assertRaises(KeyboardInterrupt):prior.LifecycleTests().run_case(ops,pull=True)
        self.assertLess(ops.events.index('cleanup-image'),ops.events.index('quit-owned'))

    def test_swap_gate_actual_adapter_stops_before_start_and_keeps_sample(self):
        ops=target.NativeOps({},{});ops.command=mock.Mock(return_value=(0,'vm.swapusage: used = 11738.12M',''))
        with self.assertRaisesRegex(RuntimeError,'swap exceeds'):ops.sample_swap('PRE-startup')
        self.assertGreater(ops.samples[-1]['swap']['conservative_used_bytes_upper'],12_000_000_000)
        self.assertEqual(ops.samples[-1]['swap']['phase'],'PRE-startup')

    def test_active_baseline_cache_blocks_pull_without_deletion(self):
        ops,state,calls,options,old,new,image=self.adapter();ops.assets_before=None;state['cache']=['existing-cache*']
        with self.assertRaisesRegex(RuntimeError,'foreign running/build'):ops.snapshot_assets()
        self.assertFalse(any('pull' in a or 'rm' in a for a in calls));self.assertTrue(ops.foreign_asset_activity)

    def test_failed_snapshot_cannot_authorize_pull_or_desktop_quit(self):
        ops=target.NativeOps({},{});ops.d=['NO-DOCKER'];ops.command=mock.Mock(side_effect=RuntimeError('inert unsupported inventory'))
        with self.assertRaises(RuntimeError):ops.snapshot_assets()
        self.assertTrue(ops.foreign_asset_activity);self.assertIsNone(ops.assets_before)
        with self.assertRaises(RuntimeError):ops.pull()
        with self.assertRaisesRegex(RuntimeError,'foreign asset activity'):ops.quit_owned({'fixture':True})

    def test_container_recovery_failure_still_attempts_image_cleanup_then_quit(self):
        import test_lifecycle_proposal as prior
        ops=prior.StubOps();ops.stop_owned_resolver=lambda:(_ for _ in ()).throw(RuntimeError('inert CID recovery failure'))
        with self.assertRaises(RuntimeError):prior.LifecycleTests().run_case(ops)
        self.assertLess(ops.events.index('cleanup-image'),ops.events.index('quit-owned'))
        self.assertEqual(ops.receipt['status'],'cleanup-unverified')

    def test_actual_record_keeps_final_swap_failure_and_asset_receipt(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
            root=Path(folder);ops=target.NativeOps({'full_scope_review':{'source_commit':'source-fixture'}},{})
            ops.root=root;ops.directory=root;ops.raw=root/'Docker.raw';ops.raw.write_bytes(b'');ops.baseline=0
            ops.command=mock.Mock(return_value=(0,'vm.swapusage: used = 11738.12M',''))
            ops.asset_receipt={'kind':'inert ownership receipt fixture'}
            with mock.patch.object(target.os,'statvfs',return_value=mock.Mock(f_bavail=40*1024**3,f_frsize=1)),self.assertRaisesRegex(RuntimeError,'final swap gate'):
                ops.record({'status':'resolved-not-installed-not-qualified','runtime_qualified':False})
            receipt=json.loads((root/'receipt.json').read_text())
            self.assertEqual(receipt['status'],'swap-final-unverified-or-over-limit')
            self.assertEqual(receipt['asset_ownership'],ops.asset_receipt)
            self.assertGreater(receipt['samples'][-1]['swap']['conservative_used_bytes_upper'],12_000_000_000)

    def test_image_failure_still_attempts_quit_but_never_returns_success(self):
        import test_lifecycle_proposal as prior
        ops=prior.StubOps();ops.cleanup_owned_image=lambda:(_ for _ in ()).throw(RuntimeError('source-only image retirement failed'))
        with self.assertRaises(RuntimeError):prior.LifecycleTests().run_case(ops)
        self.assertIn('quit-owned',ops.events)
        self.assertEqual(ops.receipt['status'],'cleanup-unverified')

    def test_swap_threshold_uses_binary_sysctl_units_without_rounding_down(self):
        self.assertTrue(hasattr(target,'parse_swap_used_bytes'),'missing explicit swap parser')
        self.assertGreater(target.parse_swap_used_bytes('vm.swapusage: total = 13312.00M used = 11738.12M free = 1573.88M'),12_000_000_000)
        self.assertEqual(target.parse_swap_used_bytes('used = 0.000001M'),2)
        self.assertGreater(target.swap_measurement('used = 11444.09M')['conservative_used_bytes_upper'],12_000_000_000)
        with self.assertRaises(RuntimeError):target.parse_swap_used_bytes('unavailable')


if __name__=='__main__':unittest.main()
