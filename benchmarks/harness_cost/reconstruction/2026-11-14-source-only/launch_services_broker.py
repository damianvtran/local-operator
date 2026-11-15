"""Causal macOS launch binding. Default/no-launch probe cannot launch any app.

Only system frameworks and ctypes; no compiler, dependencies or AppleEvents.
Objective-C message signatures are local CFUNCTYPE wrappers (never mutate a
shared objc_msgSend signature while a callback may run on another thread).
"""
import ctypes as C
import json
import os
import sys
import time


class Bridge:
    def __init__(self):
        self.objc=C.CDLL('/usr/lib/libobjc.A.dylib')
        self.foundation=C.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        self.appkit=C.CDLL('/System/Library/Frameworks/AppKit.framework/AppKit')
        self.objc.objc_getClass.argtypes=[C.c_char_p];self.objc.objc_getClass.restype=C.c_void_p
        self.objc.sel_registerName.argtypes=[C.c_char_p];self.objc.sel_registerName.restype=C.c_void_p
    def cls(self,name):
        result=self.objc.objc_getClass(name.encode())
        if not result:raise RuntimeError('unavailable Objective-C class: '+name)
        return result
    def sel(self,name):return self.objc.sel_registerName(name.encode())
    def send(self,obj,name,restype=C.c_void_p,argtypes=(),args=()):
        fn=C.CFUNCTYPE(restype,C.c_void_p,C.c_void_p,*argtypes)(('objc_msgSend',self.objc))
        return fn(obj,self.sel(name),*args)
    def string(self,value):return self.send(self.cls('NSString'),'stringWithUTF8String:',argtypes=(C.c_char_p,),args=(value.encode(),))
    def text(self,obj):
        if not obj:return None
        value=self.send(obj,'UTF8String',C.c_char_p)
        return value.decode() if value else None


class BlockDescriptor(C.Structure):
    _fields_=[('reserved',C.c_ulong),('size',C.c_ulong),('signature',C.c_char_p)]


class BlockLiteral(C.Structure):
    _fields_=[('isa',C.c_void_p),('flags',C.c_int),('reserved',C.c_int),('invoke',C.c_void_p),('descriptor',C.POINTER(BlockDescriptor))]


class NativeBlock:
    """Keep descriptor/callback alive until release; runtime owns an explicit copy."""
    def __init__(self,callback,signature=b'v@?@@'):
        self.system=C.CDLL('/usr/lib/libSystem.B.dylib')
        self.callback=C.CFUNCTYPE(None,C.c_void_p,C.c_void_p,C.c_void_p)(callback)
        self.descriptor=BlockDescriptor(0,C.sizeof(BlockLiteral),signature)
        isa=C.addressof(C.c_void_p.in_dll(self.system,'_NSConcreteStackBlock'))
        self.literal=BlockLiteral(isa,1<<30,0,C.cast(self.callback,C.c_void_p),C.pointer(self.descriptor))
        self.system._Block_copy.argtypes=[C.c_void_p];self.system._Block_copy.restype=C.c_void_p
        self.system._Block_release.argtypes=[C.c_void_p]
        self.pointer=self.system._Block_copy(C.byref(self.literal))
        if not self.pointer:raise RuntimeError('block copy failed')
    def release(self):
        if self.pointer:self.system._Block_release(self.pointer);self.pointer=None


def no_launch_probe():
    """No workspace instance/open/activation/quit call; current-process metadata only."""
    b=Bridge();workspace=b.cls('NSWorkspace');configuration=b.cls('NSWorkspaceOpenConfiguration')
    selector=b.sel('openApplicationAtURL:configuration:completionHandler:')
    available=bool(b.send(workspace,'instancesRespondToSelector:',C.c_bool,(C.c_void_p,),(selector,)))
    config=b.send(configuration,'configuration')
    new_selector=b.sel('setCreatesNewApplicationInstance:')
    new_available=all(bool(b.send(config,'respondsToSelector:',C.c_bool,(C.c_void_p,),(b.sel(s),))) for s in ('setCreatesNewApplicationInstance:','setActivates:','setPromptsUserIfNeeded:','setAllowsRunningApplicationSubstitution:'))
    current=b.send(b.cls('NSRunningApplication'),'currentApplication')
    pid=b.send(current,'processIdentifier',C.c_int) if current else None
    # Apple documents -1 for an application with no PID. Unbundled Python is
    # not registered as a GUI app; do not turn this into app registration/UI.
    # A process-local NSObject fixture tests this exact selector's pid_t ABI.
    b.objc.objc_allocateClassPair.argtypes=[C.c_void_p,C.c_char_p,C.c_size_t];b.objc.objc_allocateClassPair.restype=C.c_void_p
    b.objc.class_addMethod.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_char_p];b.objc.class_addMethod.restype=C.c_bool
    b.objc.objc_registerClassPair.argtypes=[C.c_void_p]
    fixture_cls=b.objc.objc_allocateClassPair(b.cls('NSObject'),('BrokerPIDFixture'+str(os.getpid())).encode(),0)
    getter=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)(lambda obj,sel:os.getpid())
    if not b.objc.class_addMethod(fixture_cls,b.sel('processIdentifier'),C.cast(getter,C.c_void_p),b'i@:'):raise RuntimeError('fixture method binding failed')
    b.objc.objc_registerClassPair(fixture_cls)
    fixture=b.send(b.send(fixture_cls,'alloc'),'init');fixture_pid=b.send(fixture,'processIdentifier',C.c_int)
    observed=[]
    block=NativeBlock(lambda block,obj,stop:observed.append(b.send(obj,'processIdentifier',C.c_int)),b'v@?@^B')
    try:
        collection=b.send(b.cls('NSSet'),'setWithObject:',argtypes=(C.c_void_p,),args=(fixture,))
        b.send(collection,'enumerateObjectsUsingBlock:',None,(C.c_void_p,),(block.pointer,))
    finally:block.release()
    result={'kind':'NO-LAUNCH-BINDING-PROBE','workspace_open_selector_available':available,
            'new_instance_selector_available':new_available,'current_application_pid':pid,
            'own_os_pid':os.getpid(),'current_application_has_registered_pid':pid==os.getpid(),
            'fixture_pid':fixture_pid,'fixture_pid_matches_own_process':fixture_pid==os.getpid(),
            'native_block_fixture_callback_pids':observed,'block_pointer_abi_smoke':observed==[os.getpid()],
            'launch_calls':0,'activation_calls':0,'quit_calls':0,'native_attempt_consumed':False}
    if not available or not new_available or fixture_pid!=os.getpid() or observed!=[os.getpid()]:raise RuntimeError('unsupported causal launch binding: '+json.dumps(result))
    return result


class WorkspaceDriver:
    """Retains the exact returned NSRunningApplication, never re-finds a singleton."""
    def __init__(self):
        self.b=Bridge();self.app=None;self.block=None;self.ticket=None
    def begin(self,bundle,callback):
        b=self.b;config=b.send(b.cls('NSWorkspaceOpenConfiguration'),'configuration')
        for selector,value in [('setCreatesNewApplicationInstance:',True),('setActivates:',False),('setPromptsUserIfNeeded:',False),('setAllowsRunningApplicationSubstitution:',False)]:
            if not b.send(config,'respondsToSelector:',C.c_bool,(C.c_void_p,),(b.sel(selector),)):raise RuntimeError('unsupported launch configuration: '+selector)
            b.send(config,selector,None,(C.c_bool,),(value,))
        if not b.send(config,'createsNewApplicationInstance',C.c_bool):raise RuntimeError('new-instance flag not accepted')
        url=b.send(b.cls('NSURL'),'fileURLWithPath:',argtypes=(C.c_void_p,),args=(b.string(str(bundle)),))
        def completed(block,app,error):
            if self.app is not None:
                callback(None,'duplicate OS launch completion; original authority retained');return
            # Retain before metadata extraction. An exception must not lose the
            # authoritative app object or pretend that launch never happened.
            if app:self.app=b.send(app,'retain')
            try:
                if error or not app:callback(None,b.text(b.send(error,'localizedDescription')) if error else 'nil launched app');return
                exe=b.send(app,'executableURL');date=b.send(app,'launchDate')
                ticket={'pid':b.send(app,'processIdentifier',C.c_int),'bundle':b.text(b.send(b.send(app,'bundleURL'),'path')),
                        'executable':b.text(b.send(exe,'path')),'identifier':b.text(b.send(app,'bundleIdentifier')),
                        'launch_date':b.send(date,'timeIntervalSince1970',C.c_double) if date else None,'new_instance_requested':True}
                self.ticket=ticket;callback(ticket,None)
            except BaseException as exc:callback(None,'callback metadata error: '+type(exc).__name__)
        self.block=NativeBlock(completed)
        b.send(b.send(b.cls('NSWorkspace'),'sharedWorkspace'),'openApplicationAtURL:configuration:completionHandler:',None,
               (C.c_void_p,C.c_void_p,C.c_void_p),(url,config,self.block.pointer))
    def pump(self,seconds):
        started=time.monotonic()
        b=self.b;date=b.send(b.cls('NSDate'),'dateWithTimeIntervalSinceNow:',argtypes=(C.c_double,),args=(seconds,))
        b.send(b.send(b.cls('NSRunLoop'),'currentRunLoop'),'runUntilDate:',None,(C.c_void_p,),(date,))
        remaining=seconds-(time.monotonic()-started)
        if remaining>0:time.sleep(remaining)  # No empty-runloop busy spin.
    def matches(self,ticket):
        if self.app is None or ticket!=self.ticket:return False
        b=self.b
        if b.send(self.app,'isTerminated',C.c_bool):return False
        current=b.send(b.cls('NSRunningApplication'),'runningApplicationWithProcessIdentifier:',argtypes=(C.c_int,),args=(ticket['pid'],))
        return bool(current and b.send(self.app,'isEqual:',C.c_bool,(C.c_void_p,),(current,)))
    def terminate(self,ticket):
        if self.app is None or ticket!=self.ticket:raise RuntimeError('no exact retained application authority')
        b=self.b
        if b.send(self.app,'isTerminated',C.c_bool):return True
        current=b.send(b.cls('NSRunningApplication'),'runningApplicationWithProcessIdentifier:',argtypes=(C.c_int,),args=(ticket['pid'],))
        if not current or not b.send(self.app,'isEqual:',C.c_bool,(C.c_void_p,),(current,)):raise RuntimeError('application identity changed; no PID adoption')
        return bool(b.send(self.app,'terminate',C.c_bool))


class CausalLaunch:
    """Journal precedes notification; callback loss is unresolved, never success.

    A late callback remains captured during bounded cleanup. If the OS never
    returns authority, cleanup reports UNRESOLVED and cannot guess a process.
    This explicit failure branch must block any subsequent native attempt.
    """
    def __init__(self,bundle,expected_executable,before_pids,emit,driver=None):
        self.bundle=str(bundle);self.expected=str(expected_executable);self.before=set(before_pids);self.emit=emit
        self.driver=driver if driver is not None else WorkspaceDriver()
        self.ticket=None;self.done=False;self.error=None;self.dispatched=False
    def begin(self):
        if self.dispatched:raise RuntimeError('launch request cannot be retried')
        self.emit({'event':'launch-intent','bundle':self.bundle,'new_instance_requested':True})
        self.dispatched=True
        self.driver.begin(self.bundle,self._completed)
    def _completed(self,ticket,error):
        if self.done:
            self.error='duplicate completion; original launch authority retained'
            try:self.emit({'event':'duplicate-launch-completion-rejected'})
            except BaseException as exc:self.error+='; duplicate-event journal failure: '+type(exc).__name__
            return
        try:
            if error:raise RuntimeError(error)
            if (not ticket or ticket.get('pid',0)<=0 or ticket['pid'] in self.before or ticket.get('bundle')!=self.bundle or
                ticket.get('executable')!=self.expected or ticket.get('identifier')!='com.docker.docker' or ticket.get('new_instance_requested') is not True or not ticket.get('launch_date')):
                raise RuntimeError('invalid/old/foreign causal launch ticket')
            # Persist FIRST, including when the parent is unwinding a signal.
            self.ticket=ticket  # Retain authority even if the durable journal fails.
            self.emit({'event':'causal-launch-ticket','ticket':ticket})
        except BaseException as exc:
            self.error=type(exc).__name__+': '+str(exc)
            try:self.emit({'event':'launch-authority-unresolved','error':self.error})
            except BaseException:self.error+='; journal failure (authority retained if already validated)'
        finally:self.done=True
    def pump(self,seconds):self.driver.pump(seconds)
    def terminate(self):
        if self.ticket is None:raise RuntimeError('launch authority UNRESOLVED; do not guess/skip cleanup')
        result=self.driver.terminate(self.ticket)
        self.emit({'event':'causal-terminate-request','ticket':self.ticket,'accepted':result})
        if not result:raise RuntimeError('causal application termination refused')


if __name__=='__main__':
    if sys.argv[1:]!=['--probe']:raise SystemExit('STOP: only explicit NO-LAUNCH probe implemented/allowed')
    print(json.dumps(no_launch_probe(),sort_keys=True))
