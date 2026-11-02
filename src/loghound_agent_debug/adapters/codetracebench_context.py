"""Label-free CodeTraceBench action records with explicit native-ID mapping.

Step conventions checked against NJU-LINK/CodeTracer commit
2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34 (MIT), skills/seed/*/parser.py.
This adapter is a format preflight, not a root-cause benchmark evaluator.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..context_map import FIELDS


def _pack(task: str, records: list[dict]) -> dict:
    if not task.strip() or not records:
        raise ValueError('Missing task or action records')
    steps, native, provenance = [], [], []
    for index, record in enumerate(records):
        step = {'id':str(index), **{f:'' for f in FIELDS}}
        for field in ('thought','response','action','observation'):
            step[field] = str(record.get(field) or '')
        step['state'] = 'native_step_id='+str(record['native_id'])
        steps.append(step)
        native.append(str(record['native_id']))
        provenance.append(record['provenance'])
    if len(native)!=len(set(native)):
        raise ValueError('Duplicate native step identifiers')
    return {'case':{'task':task,'test_code':'','error':'','steps':steps},
            'native_step_ids':native,'provenance':provenance,
            'id_convention':'Internal zero-based ordinal; native_step_ids maps predictions back to published action IDs.'}


def miniswe_context(raw: dict, task: str | None = None) -> dict:
    messages=raw['messages']
    if task is None:
        task=next((m['content'] for m in messages if m.get('role')=='user' and isinstance(m.get('content'),str)),'')
    actions=[i for i,m in enumerate(messages) if m.get('role')=='assistant' and
             isinstance(m.get('content'),str) and '```bash' in m['content']]
    records=[]
    for native_id,index in enumerate(actions,1):
        content=messages[index]['content'];match=re.search(r'```bash\s*\n(.*?)\n?```',content,re.S)
        action=match[1].strip() if match else content.strip()
        observation=next((j for j in range(index+1,len(messages)) if messages[j].get('role')=='user' and
                         isinstance(messages[j].get('content'),str) and '<returncode>' in messages[j]['content']),None)
        records.append({'native_id':native_id,'thought':content[:match.start()] if match else '',
            'response':content[match.end():] if match else '', 'action':action,
            'observation':messages[observation]['content'] if observation is not None else '',
            'provenance':{'action_message_index':index,'observation_message_index':observation,
                          'command_character_range':[match.start(1),match.end(1)] if match else [0,len(content)]}})
    return _pack(task,records)


def _openhands_command(event: dict) -> str | None:
    if event.get('action')=='run_ipython':
        return str((event.get('args') or {}).get('code') or '').strip() or None
    metadata=event.get('tool_call_metadata') or {}
    args=metadata.get('args') or {}
    if isinstance(args,dict) and args.get('command'):
        return str(args['command']).strip() or None
    response=metadata.get('model_response') or {}
    for choice in response.get('choices') or []:
        for call in (choice.get('message') or {}).get('tool_calls') or []:
            encoded=(call.get('function') or {}).get('arguments')
            if not isinstance(encoded,str):continue
            try:decoded=json.loads(encoded)
            except ValueError:continue
            if isinstance(decoded,dict) and decoded.get('command'):
                return str(decoded['command']).strip() or None
    message=event.get('message') or ''
    if isinstance(message,str) and message.startswith('Running command:'):
        return message[len('Running command:'):].strip() or None
    return None


def openhands_context(events: list[dict], task: str) -> dict:
    # The public seed parser covers shell/IPython runs only. Released trajectories
    # can also number file reads/edits; dropping these would shift every later ID.
    unsupported = sorted({str(e['action']) for e in events if e.get('action') not in
                          (None, 'system', 'message', 'run', 'run_ipython', 'recall', 'think', 'finish')})
    if unsupported:
        raise ValueError('Unverified OpenHands native action mapping: '+', '.join(unsupported))
    actions,observations={},{}
    for event in sorted(events,key=lambda e:e.get('id') if isinstance(e.get('id'),int) else 10**9):
        eid=event.get('id')
        if not isinstance(eid,int) or eid==0:continue
        if event.get('action') in ('run','run_ipython') and _openhands_command(event) is not None:
            actions[eid]=event
        cause=event.get('cause')
        if isinstance(cause,int) and 'observation' in event:
            observations.setdefault(cause,event)
    records=[]
    for native_id,eid in enumerate(sorted(actions),1):
        action=actions[eid];observation=observations.get(eid,{})
        records.append({'native_id':native_id,'action':_openhands_command(action),
            'thought':(action.get('args') or {}).get('thought',''),
            'observation':observation.get('content',''),
            'provenance':{'action_event_id':eid,'observation_event_id':observation.get('id')}})
    return _pack(task,records)


def terminus_context(episodes: dict[int, dict[str,str]], task: str) -> dict:
    if not episodes:raise ValueError('Empty episode list')
    records=[]
    for episode_id in range(max(episodes)+1):
        response=episodes.get(episode_id,{}).get('response')
        observation=episodes.get(episode_id+1,{}).get('prompt')
        if response is None and observation is None:continue
        thought='';action=''
        if response is not None:
            try:obj=json.loads(response)
            except ValueError:obj=None
            if isinstance(obj,dict):
                thought='\n'.join(str(v) for v in (obj.get('analysis') or obj.get('state_analysis'),
                                  obj.get('plan') or obj.get('explanation')) if v)
                commands=obj.get('commands') or []
                action='\n'.join(str(c.get('keystrokes','') if isinstance(c,dict) else c)
                                  for c in commands if isinstance(c,(str,dict))) if isinstance(commands,list) and commands else thought
            else:action=response.strip()
        marker='New Terminal Output:'
        if observation is not None:
            observation=observation.split(marker,1)[-1].strip()
        records.append({'native_id':episode_id+1,'thought':thought,'action':action,'observation':observation,
            'provenance':{'response_episode':episode_id,'observation_prompt_episode':episode_id+1 if observation is not None else None}})
    return _pack(task,records)


def native_predictions(internal_ids: list[str], packet: dict) -> list[str]:
    native=packet['native_step_ids']
    if len(set(internal_ids))!=len(internal_ids) or any(not i.isdigit() or not 0<=int(i)<len(native) for i in internal_ids):
        raise ValueError('Invalid internal prediction identifiers')
    return [native[int(i)] for i in internal_ids]
