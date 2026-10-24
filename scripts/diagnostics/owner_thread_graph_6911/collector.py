"""Diagnostic-only instrumented owner-thread CPU, not cProfile/getstats.

Edges represent immediate synchronous resumption-stack nesting, never await or
creation causality. Inclusive values overlap. C execution stays in Python self
CPU; C events are not nodes. No calibrated bias is subtracted. A future driver
must install ``engine.callback`` DIRECTLY, immediately call its reviewed root,
and unconditionally call cleanup. Installing inside a returning helper would
introduce an unsupported startup return: this collector deliberately rejects it.
"""
from types import CodeType, FrameType
import re

SCHEMA='owner_thread_python_resumption_graph_v1'
CAPS=dict(codes=128,edges=256,stack=64,events=20000)
HEADER={'python','executable_sha256','producer_sha256','pid','ppid','pgid','owner_thread','static_ids'}
REASONS={'HOOK_CONFLICT','FOREIGN_THREAD','EVENT','FRAME','ROOT','STACK','CAP','CLOCK_TYPE','CLOCK_NEGATIVE','HOOK','UNFINISHED','SERIALIZATION','QUALITY'}


def header_valid(value,expected):
    if type(value) is not dict or set(value)!=HEADER or value!=expected:return False
    if type(value['python']) is not list or len(value['python'])!=3 or any(type(v) is not int or v<0 for v in value['python']) or value['python'][:2]!=[3,12]:return False
    for key in ('pid','ppid','pgid','owner_thread'):
        if type(value[key]) is not int or value[key]<=0:return False
    ids=value['static_ids']
    if type(ids) is not list or len(ids)>CAPS['codes'] or any(type(v) is not str or not re.fullmatch('PRIMITIVE:[A-Z_]{1,40}',v) for v in ids) or len(set(ids))!=len(ids):return False
    return all(type(value[k]) is str and re.fullmatch('[0-9a-f]{64}',value[k]) for k in ('executable_sha256','producer_sha256'))


class Collector:
    def __init__(self,root,catalog,identity,expected,clock,thread_id,get_hook,set_hook):
        # The outer reviewed root must RETURN, not yield/suspend: a generator
        # boundary cannot distinguish trailing completion from its first yield.
        if type(root) is not CodeType or root.co_flags & (0x20|0x80|0x200) or type(catalog) is not tuple or not header_valid(identity,expected):raise ValueError('identity STOP')
        if any(type(code) is not CodeType or type(label) is not str or not re.fullmatch('PRIMITIVE:[A-Z_]{1,40}',label) for code,label in catalog):raise ValueError('catalog STOP')
        if len(catalog)>CAPS['codes'] or len({id(code) for code,_ in catalog})!=len(catalog) or len({label for _,label in catalog})!=len(catalog) or not any(code is root for code,_ in catalog):raise ValueError('catalog STOP')
        if identity['static_ids']!=[label for _,label in catalog]:raise ValueError('catalog binding STOP')
        self.root=root;self.catalog=catalog;self.identity=dict(identity);self.clock=clock;self.thread_id=thread_id;self.get_hook=get_hook;self.set_hook=set_hook
        self.callback=self._event;self.codes=[];self.nodes=[];self.edges=[];self.stack=[];self.last=None;self.events=0;self.c_events=0;self.ready=False;self.trailing=False;self.failure=None;self.signed_value=None;self.argument_used=None;self.hook_seen_at_export=None;self.hook_after_cleanup=None
        # Freeze the entry observation; cleanup must never rewrite it.
        self.hook_entry_none=get_hook() is None
        if not self.hook_entry_none:self.fail('HOOK_CONFLICT')

    def fail(self,reason,value=None):
        if self.failure is None:
            self.failure=reason;self.signed_value=value if type(value) is int else None

    def node(self,code):
        for index,known in enumerate(self.codes):
            if known is code:return index
        if len(self.codes)>=CAPS['codes']:self.fail('CAP');return None
        self.codes.append(code);label=next((label for known,label in self.catalog if known is code),None)
        self.nodes.append(dict(static_id=label,segments=0,own_ns=0,inclusive_ns=0));return len(self.codes)-1

    def _event(self,frame,event,arg):
        # Reject foreign delivery before consulting the clock, stack or frame.
        if self.thread_id()!=self.identity['owner_thread']:self.fail('FOREIGN_THREAD');return
        if self.failure is not None:return
        try:
            if type(event) is not str or event not in ('call','return','c_call','c_return','c_exception'):self.fail('EVENT');return
            if self.events>=CAPS['events']:self.fail('CAP');return
            self.events+=1
            if event.startswith('c_'):self.c_events+=1;return
            if type(frame) is not FrameType or type(frame.f_code) is not CodeType:self.fail('FRAME');return
            code=frame.f_code;token=id(frame)
            if not self.ready:
                if event!='call' or code is not self.root:self.fail('ROOT');return
                self.argument_used=self.get_hook() is self.callback
                if self.argument_used is not True:self.fail('HOOK');return
                self.ready=True
            if self.trailing:self.fail('ROOT');return
            now=self.clock()
            if type(now) is not int or now<0:self.fail('CLOCK_TYPE');return
            if self.last is not None:
                delta=now-self.last
                if delta<0:self.fail('CLOCK_NEGATIVE',delta);return
                for _,index,edge in self.stack:
                    self.nodes[index]['inclusive_ns']+=delta
                    if edge is not None:self.edges[edge]['inclusive_ns']+=delta
                if self.stack:
                    _,index,edge=self.stack[-1];self.nodes[index]['own_ns']+=delta
                    if edge is not None:self.edges[edge]['own_ns']+=delta
            self.last=now
            if event=='call':
                if len(self.stack)>=CAPS['stack'] or any(old==token for old,_,_ in self.stack):self.fail('STACK');return
                index=self.node(code)
                if index is None:return
                edge=None
                if self.stack:
                    parent=self.stack[-1][1];edge=next((i for i,row in enumerate(self.edges) if row['parent']==parent and row['child']==index),None)
                    if edge is None:
                        if len(self.edges)>=CAPS['edges']:self.fail('CAP');return
                        edge=len(self.edges);self.edges.append(dict(parent=parent,child=index,segments=0,own_ns=0,inclusive_ns=0))
                    self.edges[edge]['segments']+=1
                self.nodes[index]['segments']+=1;self.stack.append((token,index,edge))
            else:
                if not self.stack or self.stack[-1][0]!=token or self.codes[self.stack[-1][1]] is not code:self.fail('STACK');return
                self.stack.pop()
                if not self.stack:
                    if code is not self.root:self.fail('ROOT');return
                    self.trailing=True;self.cleanup()
        except Exception:self.fail('EVENT')

    def cleanup(self):
        # Never remove an observer installed by someone else, even after failure.
        try:
            hook=self.get_hook()
            if hook is self.callback:self.set_hook(None)
            elif hook is not None:self.fail('HOOK')
            self.hook_after_cleanup=self.get_hook() is None
            if not self.hook_after_cleanup:self.fail('HOOK')
        except Exception:self.fail('HOOK');self.hook_after_cleanup=False

    def export(self):
        # The exact root-return callback releases its hook while callback
        # recursion is suppressed. Export observes NONE, not historical ownership.
        self.hook_seen_at_export=self.get_hook() is None
        if not self.ready or not self.trailing or self.stack:self.fail('UNFINISHED')
        record=dict(schema=SCHEMA,identity=dict(self.identity),unit='instrumented_owner_thread_cpu_ns',caps=dict(CAPS),ready=self.ready,trailing=self.trailing,balanced=not self.stack,hook_entry_none=self.hook_entry_none,hook_argument_used=self.argument_used,hook_seen_at_export=self.hook_seen_at_export,hook_after_cleanup=self.hook_after_cleanup,events=self.events,c_events=self.c_events,failure=self.failure,signed_value=self.signed_value,nodes=[dict(row) for row in self.nodes],edges=[dict(row) for row in self.edges],completed=False)
        try:
            validate(record,self.identity,require_complete=False)
            # Serialization success is observed, not assumed by a default flag.
            import json
            json.dumps(record,allow_nan=False)
            record['completed']=True
        except (ValueError,TypeError,OverflowError):record['failure']=record['failure'] or 'SERIALIZATION'
        return record


KEYS={'schema','identity','unit','caps','ready','trailing','balanced','hook_entry_none','hook_argument_used','hook_seen_at_export','hook_after_cleanup','events','c_events','failure','signed_value','nodes','edges','completed'}


def validate(record,expected,require_complete=True):
    if type(record) is not dict or set(record)!=KEYS or record['schema']!=SCHEMA or record['unit']!='instrumented_owner_thread_cpu_ns' or record['caps']!=CAPS or not header_valid(record['identity'],expected):raise ValueError('schema STOP')
    for key in ('ready','trailing','balanced','hook_entry_none','hook_seen_at_export','completed'):
        if type(record[key]) is not bool:raise ValueError('bool STOP')
    for key in ('hook_argument_used','hook_after_cleanup'):
        if record[key] is not None and type(record[key]) is not bool:raise ValueError('hook STOP')
    if record['failure'] is not None and record['failure'] not in REASONS:raise ValueError('reason STOP')
    if record['signed_value'] is not None and type(record['signed_value']) is not int:raise ValueError('signed STOP')
    for key in ('events','c_events'):
        if type(record[key]) is not int or not 0<=record[key]<=CAPS['events']:raise ValueError('count STOP')
    if record['c_events']>record['events']:raise ValueError('count STOP')
    if type(record['nodes']) is not list or len(record['nodes'])>CAPS['codes'] or type(record['edges']) is not list or len(record['edges'])>CAPS['edges']:raise ValueError('cap STOP')
    # Every UNKNOWN node/edge is validated before any trusted projection.
    for row in record['nodes']:
        if type(row) is not dict or set(row)!={'static_id','segments','own_ns','inclusive_ns'}:raise ValueError('node STOP')
        if row['static_id'] is not None and (type(row['static_id']) is not str or row['static_id'] not in expected['static_ids']):raise ValueError('static STOP')
    for row in record['edges']:
        if type(row) is not dict or set(row)!={'parent','child','segments','own_ns','inclusive_ns'}:raise ValueError('edge STOP')
        if any(type(row[k]) is not int or not 0<=row[k]<len(record['nodes']) for k in ('parent','child')):raise ValueError('endpoint STOP')
    for row in record['nodes']+record['edges']:
        for key in ('segments','own_ns','inclusive_ns'):
            if type(row[key]) is not int or row[key]<0:raise ValueError('quality STOP')
    if require_complete and (record['failure'] is not None or not record['nodes'] or any(record[k] is not True for k in ('ready','trailing','balanced','hook_entry_none','hook_argument_used','hook_seen_at_export','hook_after_cleanup','completed'))):raise ValueError('incomplete STOP')
    return True


def finalize(record,expected):
    """No arbitrary-record echo. Signed diagnostics are not measurement rows."""
    safe=dict(schema=SCHEMA,qualified=False,failure=None,signed_value=None,completed=None)
    if type(record) is dict:
        if type(record.get('failure')) is str and record['failure'] in REASONS:safe['failure']=record['failure']
        if type(record.get('signed_value')) is int:safe['signed_value']=record['signed_value']
        if type(record.get('completed')) is bool:safe['completed']=record['completed']
    try:validate(record,expected);safe['qualified']=True
    except (ValueError,TypeError,KeyError,IndexError):pass
    return safe
