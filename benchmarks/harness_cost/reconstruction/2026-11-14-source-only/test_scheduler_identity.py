"""R16 real observe -> snapshot -> record path; no process/app launch."""
import copy
import ctypes
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import lifecycle_proposal as target
from darwin_process_identity import DarwinIdentity,BSDInfo
import test_owned_session_launch as fixture


def bsd(pid,status):
    item=BSDInfo();item.pid=pid;item.ppid=1;item.pgid=pid;item.uid=os.getuid();item.start_sec=1794732482;item.start_usec=pid;item.status=status
    return item


class SchedulerIdentityTests(unittest.TestCase):
    def test_scheduler_changes_populate_actual_adapter_identity_not_replacement(self):
        for statuses in ((2,3),(3,2),(2,4),(4,3),(3,3)):
            with self.subTest(statuses=statuses),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                helper=fixture.SessionTests();self.addCleanup(helper.doCleanups)
                ops,b,patch=helper.adapter(f)
                with patch:ops.start_owned()
                rows={p['pid']:p for p in b.rows};identity=object.__new__(DarwinIdentity)
                records=[]
                for pid,row in rows.items():
                    a=bsd(pid,statuses[0]);a.ppid=row['ppid'];a.pgid=row['pgid']
                    z=bsd(pid,statuses[1]);z.ppid=row['ppid'];z.pgid=row['pgid'];z.flags=0x8000;z.nice=12;z.nfiles=9
                    records.extend((a,z))
                identity.bsd=mock.Mock(side_effect=records);identity.path=mock.Mock(side_effect=lambda pid:rows[pid]['exe'])
                ops.kernel_identity=identity
                text='\n'.join(str(p['pid'])+' '+str(p['ppid'])+' '+str(p['pgid'])+' 100 S Sun Nov 15 03:48:02 2026 '+p['exe'] for p in rows.values())
                ops.command.return_value=(0,text,'');ops.snapshot=types.MethodType(target.NativeOps.snapshot,ops)
                with mock.patch.object(target.os,'getsid',return_value=75032):actual,live=ops.observe_owned()
                self.assertEqual(len(live),6);self.assertEqual(set(ops.owned_processes),set(rows))
                self.assertTrue(all(p['sid']==75032 and p['state']=='live' for p in actual))
                self.assertTrue(all(p['start'].endswith(str(p['pid']).zfill(6)) for p in actual))
                self.assertFalse(b.signals);self.assertFalse(b.waited)

    def test_actual_adapter_rejects_changed_kernel_identity_during_capture(self):
        for kind in ('pid','birth','path','sid','exit'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as f:
                helper=fixture.SessionTests();self.addCleanup(helper.doCleanups)
                ops,b,patch=helper.adapter(f)
                with patch:ops.start_owned()
                rows={p['pid']:p for p in b.rows};first=bsd(75032,2);second=bsd(75032,3)
                if kind=='pid':second.pid+=1
                if kind=='birth':second.start_usec+=1
                # The held creator's legitimate live->zombie transition remains
                # supported; use an invalid resurrection in its negative case.
                if kind=='exit':first.status=5;second.status=3
                identity=object.__new__(DarwinIdentity);identity.bsd=mock.Mock(side_effect=[first,second])
                identity.path=mock.Mock(side_effect=[rows[75032]['exe'],'/different/path'] if kind=='path' else [rows[75032]['exe']]*2)
                ops.kernel_identity=identity;ops.snapshot=types.MethodType(target.NativeOps.snapshot,ops)
                text='75032 1 75032 100 S Sun Nov 15 03:48:02 2026 '+rows[75032]['exe']
                ops.command.return_value=(0,text,'')
                with mock.patch.object(target.os,'getsid',side_effect=[75032,75032,444] if kind=='sid' else [75032]*3),self.assertRaises(RuntimeError):ops.observe_owned()
                self.assertFalse(b.signals);self.assertFalse(b.waited)

    def test_birth_pid_path_and_sid_changes_still_fail_closed(self):
        for kind in ('pid','birth','path','sid','exit'):
            with self.subTest(kind=kind):
                a=bsd(10,2);z=bsd(10,3)
                if kind=='pid':z.pid=11
                if kind=='birth':z.start_usec+=1
                if kind=='exit':z.status=5
                identity=object.__new__(DarwinIdentity);identity.bsd=mock.Mock(side_effect=[a,z])
                identity.path=mock.Mock(side_effect=['/Applications/Docker.app/backend','/different/path'] if kind=='path' else ['/Applications/Docker.app/backend']*2)
                with mock.patch('darwin_process_identity.os.getsid',side_effect=[10,11] if kind=='sid' else [10,10]),self.assertRaises(RuntimeError):identity.record(10)

    def test_creator_zombie_classification_is_not_ordinary_scheduler_noise(self):
        a=bsd(10,5);z=bsd(10,3);identity=object.__new__(DarwinIdentity);identity.bsd=mock.Mock(side_effect=[a,z])
        with self.assertRaises(RuntimeError):identity.record(10,creator_sid=10,creator_executable='/Applications/Docker.app/backend')


if __name__=='__main__':unittest.main()
