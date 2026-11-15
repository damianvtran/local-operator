"""Direct signed outer-backend launch; authority is a session WE create.

Never discover/adopt a LaunchServices ancestor/group. Keep the session creator
UNREAPED until all owned live members are gone: its reserved PID prevents numeric
SID reuse, including a fast fork followed by creator exit. No poll()/wait() before
release. The controller must retain this object; SIGCHLD auto-reaping is refused.
No activation/headless/prompt flags are claimed for direct backend execution.
"""
import os
from pathlib import Path
import signal
import subprocess
import time
import threading

KIND='owned-posix-session-v1'


def backend_environment():
    # Preserve operator settings/HOME, not Docker CLI's isolated anonymous HOME.
    # Do not propagate agent API keys, harness routing or guessed launch flags.
    keys=('HOME','USER','LOGNAME','SHELL','PATH','TMPDIR','LANG','LC_ALL','LC_CTYPE','XDG_CONFIG_HOME','DOCKER_CONFIG')
    env={k:os.environ[k] for k in keys if k in os.environ}
    env['HOME']=str(Path.home())
    return env


class OwnedSessionLaunch:
    kind=KIND
    def __init__(self,bundle,executable,before_pids,emit,spawn=None,log_path=None):
        self.bundle=str(bundle);self.executable=str(executable);self.before=set(before_pids);self.emit=emit
        self.spawn=spawn or subprocess.Popen;self.log_path=log_path;self.log=None
        self.creator=None;self.ticket=None;self.dispatched=False;self.done=False;self.error=None
        self.released=False;self.driver=self
    def begin(self):
        if self.dispatched or self.creator is not None:raise RuntimeError('direct launch is one-shot')
        if threading.active_count()!=1:raise RuntimeError('signal-mask preexec restoration requires the single-threaded native controller')
        if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:raise RuntimeError('creator PID reservation requires default SIGCHLD disposition')
        self.emit({'event':'launch-intent','authority':KIND,'executable':self.executable,'new_session_requested':True,'ui_behavior':'UNQUALIFIED; no activation/headless/prompt flags'})
        if self.log_path is not None:
            fd=os.open(self.log_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);self.log=os.fdopen(fd,'wb',buffering=0)
        blocked={signal.SIGTERM,signal.SIGINT,signal.SIGALRM};mask=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
        try:
            self.creator=self.spawn([self.executable],stdin=subprocess.DEVNULL,stdout=self.log or subprocess.DEVNULL,stderr=subprocess.STDOUT,
                                    env=backend_environment(),cwd='/',start_new_session=True,
                                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,mask))
            # Popen returns only after setsid + successful exec. Publish authority
            # BEFORE restoring signals or performing any fallible journal I/O.
            self.dispatched=True
            self.ticket={'authority':KIND,'pid':self.creator.pid,'sid':self.creator.pid,'initial_pgid':self.creator.pid,
                         'bundle':self.bundle,'executable':self.executable,'birth':None,'birth_pending':True,'creator_unreaped':True,
                         'home_policy':'original-operator-settings','ui_behavior':'UNQUALIFIED'}
            self.done=True
            # Stale numeric pre-start PIDs are not identities: fork/setsid
            # allocated THIS child/session. Birth + unreaped reservation bind it.
        except BaseException as error:
            self.done=True;self.error=type(error).__name__+': '+str(error)
            if self.creator is None and self.log is not None:self.log.close()
            raise
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        self.emit({'event':'owned-session-created','ticket':dict(self.ticket)})
    def matches(self,ticket):
        # Merely inspect returncode; poll()/wait() would destroy PID reservation.
        return bool(ticket is self.ticket and self.creator is not None and self.creator.returncode is None and not self.released)
    def bind_birth(self,rows):
        if not self.matches(self.ticket):raise RuntimeError('creator reaped/released; no session authority')
        root=next((p for p in rows if p['pid']==self.creator.pid),None)
        if root is None:raise RuntimeError('unreaped creator absent from coherent snapshot')
        if root.get('sid')!=self.ticket['sid']:raise RuntimeError('creator session identity changed')
        if self.ticket['birth'] is None:self.ticket['birth']=root['start'];self.ticket['birth_pending']=False
        if root['start']!=self.ticket['birth']:raise RuntimeError('creator PID/birth reused; refuse adoption')
        return root
    def pump(self,seconds):
        if seconds>0:time.sleep(seconds)  # No app discovery, callback or polling.
    def terminate(self):
        if not self.matches(self.ticket):raise RuntimeError('lost session creation authority')
        self.emit({'event':'owned-session-exact-pid-retirement','authority':KIND})
        # NativeOps validates a coherent membership set and signals exact PIDs.
        # This method neither guesses a process group nor invokes global quit.
    def release_creator(self,timeout):
        if self.released:return
        if self.creator is None:return
        self.creator.wait(timeout=max(0.001,timeout));self.released=True
        self.ticket['creator_unreaped']=False
        if self.log is not None:self.log.close()
    def public_state(self):
        return {'authority':KIND,'dispatched':self.dispatched,'done':self.done,'ticket':self.ticket,'error':self.error,'creator_released':self.released}
