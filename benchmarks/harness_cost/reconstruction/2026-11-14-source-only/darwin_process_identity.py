"""Read-only Darwin process identity, no app/helper launch or native compilation.

ABI: Apple's xnu bsd/sys/proc_info.h proc_bsdinfo / PROC_PIDTBSDINFO=3.
Use microsecond kernel birth, path and SID together; coarse ps lstart alone is
not enough to reject a rapid PID reuse. Return no authority on unstable reads.
"""
import ctypes as C
import errno
import os


class BSDInfo(C.Structure):
    _fields_=[(n,C.c_uint32) for n in ('flags','status','xstatus','pid','ppid','uid','gid','ruid','rgid','svuid','svgid','reserved')]+[
        ('comm',C.c_char*16),('name',C.c_char*32)]+[(n,C.c_uint32) for n in ('nfiles','pgid','jobc','tdev','tpgid')]+[
        ('nice',C.c_int32),('start_sec',C.c_uint64),('start_usec',C.c_uint64)]


class DarwinIdentity:
    def __init__(self):
        if C.sizeof(BSDInfo)!=136:raise RuntimeError('unsupported proc_bsdinfo ABI')
        self.lib=C.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        self.lib.proc_pidinfo.argtypes=[C.c_int,C.c_int,C.c_uint64,C.c_void_p,C.c_int];self.lib.proc_pidinfo.restype=C.c_int
        self.lib.proc_pidpath.argtypes=[C.c_int,C.c_void_p,C.c_uint32];self.lib.proc_pidpath.restype=C.c_int
    def bsd(self,pid):
        value=BSDInfo();size=self.lib.proc_pidinfo(pid,3,0,C.byref(value),C.sizeof(value))
        if size==0 and C.get_errno() in (errno.ESRCH,errno.ENOENT):return None
        if size!=C.sizeof(value) or value.pid!=pid:raise RuntimeError('process identity unavailable/unsupported ABI')
        return value
    def path(self,pid):
        value=C.create_string_buffer(4096)
        if self.lib.proc_pidpath(pid,value,len(value))<=0:return None
        return value.value.decode('utf-8','strict')
    def record(self,pid,creator_sid=None,creator_executable=None):
        first=self.bsd(pid)
        if first is None:return None
        def held_zombie(value):
            return {'pid':pid,'ppid':int(value.ppid),'pgid':int(value.pgid),'sid':creator_sid,'uid':int(value.uid),
                    'start':str(value.start_sec)+'.'+str(value.start_usec).zfill(6),'exe':creator_executable,'state':'Z'}
        if first.status==5 and creator_sid is not None:
            second=self.bsd(pid)
            if second is None or (first.start_sec,first.start_usec)!=(second.start_sec,second.start_usec):raise RuntimeError('held creator birth unavailable')
            return held_zombie(second)
        path=self.path(pid)
        try:sid=os.getsid(pid)
        except ProcessLookupError:
            second=self.bsd(pid)
            if creator_sid is not None and second is not None and second.status==5 and (first.start_sec,first.start_usec)==(second.start_sec,second.start_usec):return held_zombie(second)
            raise
        second=self.bsd(pid);path2=self.path(pid)
        if creator_sid is not None and second is not None and second.status==5 and (first.start_sec,first.start_usec)==(second.start_sec,second.start_usec):return held_zombie(second)
        fields=('pid','ppid','pgid','uid','start_sec','start_usec','status')
        if second is None or any(getattr(first,k)!=getattr(second,k) for k in fields) or path!=path2:
            raise RuntimeError('process identity changed during snapshot closure')
        if not path:raise RuntimeError('live process executable identity unavailable')
        return {'pid':pid,'ppid':int(first.ppid),'pgid':int(first.pgid),'sid':sid,'uid':int(first.uid),
                'start':str(first.start_sec)+'.'+str(first.start_usec).zfill(6),'exe':path,'state':'Z' if first.status==5 else 'live'}
