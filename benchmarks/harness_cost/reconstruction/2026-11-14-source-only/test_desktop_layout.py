"""R10 static bundle validation only; fixture binaries are never executed."""
import os
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest import mock
import lifecycle_proposal as target


class LayoutTests(unittest.TestCase):
    def fixture(self,root,relative):
        nested='Docker Desktop.app/' in relative
        def bundle(path,identifier,exe):
            (path/'Contents/MacOS').mkdir(parents=True,exist_ok=True)
            (path/'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':identifier,'CFBundleExecutable':exe}))
            binary=path/'Contents/MacOS'/exe;binary.write_text('INERT FIXTURE');binary.chmod(0o700)
        bundle(root,'com.docker.docker','com.docker.backend' if nested else Path(relative).name)
        if nested:bundle(root/'Contents/MacOS/Docker Desktop.app','com.electron.dockerdesktop','Docker Desktop')
        exe=root/relative;exe.write_text('INERT FIXTURE; DO NOT EXECUTE');exe.chmod(0o700)
        def signed(args,**kw):
            if '--verify' in args:
                self.assertIn('-R',args);self.assertIn('anchor apple generic',args[args.index('-R')+1]);return 0,'',''
            identifier='com.electron.dockerdesktop' if args[-1].endswith('Docker Desktop.app') else 'com.docker.docker'
            return 0,'','Identifier='+identifier+'\nTeamIdentifier=9BNSXJN65R\n'
        return exe,signed

    def test_explicit_nested_and_both_flat_layouts(self):
        self.assertTrue(hasattr(target,'validate_desktop_layout'),'missing static layout gate')
        for relative in ('Contents/MacOS/Docker','Contents/MacOS/Docker Desktop','Contents/MacOS/Docker Desktop.app/Contents/MacOS/Docker Desktop'):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
                root=Path(folder).resolve()/'Docker.app';exe,runner=self.fixture(root,relative)
                with mock.patch.object(target,'DOCKER_BUNDLE',root):
                    value=target.validate_desktop_layout(exe,runner)
                self.assertEqual(value['executable'],str(exe))

    def test_escaping_plist_executables_and_symlinked_outer_inner_fail(self):
        self.assertTrue(hasattr(target,'validate_desktop_layout'),'missing static layout gate')
        for kind in ('outer-plist','inner-plist','outer-symlink','inner-symlink','executable-symlink','untrusted-signature','unlisted-path'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as folder:
                home=Path(folder).resolve();root=home/'Docker.app';rel='Contents/MacOS/Docker Desktop.app/Contents/MacOS/Docker Desktop'
                exe,runner=self.fixture(root,rel);inner=root/'Contents/MacOS/Docker Desktop.app'
                if kind.endswith('plist'):
                    p=(root if kind=='outer-plist' else inner)/'Contents/Info.plist';d=plistlib.loads(p.read_bytes());d['CFBundleExecutable']='../../escape';p.write_bytes(plistlib.dumps(d))
                elif kind=='outer-symlink':
                    alias=home/'alias.app';alias.symlink_to(root,target_is_directory=True);root=alias;exe=root/rel
                elif kind=='inner-symlink':
                    moved=home/'inner.app';inner.rename(moved);inner.symlink_to(moved,target_is_directory=True)
                elif kind=='executable-symlink':
                    moved=home/'inert';exe.rename(moved);exe.symlink_to(moved)
                elif kind=='untrusted-signature':runner=lambda args,**kw:(0,'','Identifier=com.docker.docker\nTeamIdentifier=UNTRUSTED\n')
                else:exe=root/'Contents/MacOS/arbitrary'
                with mock.patch.object(target,'DOCKER_BUNDLE',root),self.assertRaises((RuntimeError,FileNotFoundError)):
                    target.validate_desktop_layout(exe,runner)

    def test_nested_prefix_and_existing_helper_attribution_are_not_broadened(self):
        nested='/Applications/Docker.app/Contents/MacOS/Docker Desktop.app/Contents/MacOS/Docker Desktop'
        ops=target.NativeOps({},{});row={'pid':101,'ppid':1,'start':'fixture','exe':nested,'rss_kib':1}
        self.assertEqual(ops.desktop_records([row]),[row])
        helper=dict(pid=102,ppid=101,start='new',exe='/Applications/Docker.app/Contents/MacOS/com.docker.virtualization',rss_kib=1)
        self.assertEqual(target.choose_helper([], [row,helper],row,set())['pid'],102)
        with self.assertRaises(RuntimeError):target.choose_helper([helper],[row,helper],row,set())


if __name__=='__main__':unittest.main()
