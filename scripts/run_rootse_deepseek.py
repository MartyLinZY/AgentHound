#!/usr/bin/env python3
"""Frozen six-method RootSE suite, subprocess isolation and per-call replay."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from deepseek_suite_transport import save,now,digest
from run_rootse_deepseek_worker import METHODS
from run_rootse_full_baseline import ensure_source

ROOT=Path(__file__).resolve().parents[1]


def prepare(source:Path,out:Path,config:dict,config_path:Path)->dict:
    expected=json.loads((ROOT/'experiments/rootse_full/dataset_manifest.json').read_text())
    actual={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'RootSE').rglob('*.json')}
    if actual!=expected or len(actual)!=102:raise ValueError('RootSE differs from fixed release')
    source_paths=[source/'utils.py',source/'TrajAudit/template.yaml']+[source/(m.replace('.','/')+'.py') for m,_ in METHODS.values()]
    source_paths+=sorted((source/'baseline/FAMAS/reference_trajs').rglob('*.json'))
    source_paths+=sorted((source/'baseline/FAMAS/selected_refs').rglob('*.json'))
    source_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    code_paths=[ROOT/'scripts'/name for name in ['run_rootse_deepseek.py','run_rootse_deepseek_worker.py','deepseek_suite_transport.py']]
    code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    protocol={'config':config,'dataset_hash':digest(expected),'upstream_hashes':source_hashes,'code_hashes':code_hashes}
    fingerprint=digest(protocol)
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['fingerprint']!=fingerprint:raise ValueError('Frozen protocol changed; use a new output directory')
        return manifest
    snapshot=out/'snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for p in code_paths:shutil.copy2(p,snapshot/p.name)
    for p in source_paths:
        if p.suffix!='.json':
            target=snapshot/'upstream'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    save(snapshot/'config.json',config);save(snapshot/'dataset_manifest.json',expected)
    shutil.copy2(ROOT/'experiments/rootse_full/paper_reference.json',snapshot/'paper_reference.json')
    files=sorted(expected,key=lambda rel:(-(source/rel).stat().st_size,rel))
    if config.get('sample_limit'):
        files=files[:config['sample_limit']]
    cases=[]
    for rel in files:
        case_id=hashlib.sha256(rel.encode()).hexdigest()[:16]
        for method in config['methods']:
            cases.append({'method':method,'case_id':case_id,'source_file':rel,'directory':f'{method}/{case_id}'})
    manifest={'protocol':protocol,'fingerprint':fingerprint,'cases':cases,'n':len(cases),'started_at':now(),'status':'prepared'}
    save(out/'manifest.json',manifest)
    return manifest


def run_job(case:dict,out:Path,source:Path,key_path:Path)->dict:
    directory=out/case['directory'];directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'state.json'
    if state_path.exists():
        old=json.loads(state_path.read_text())
        if old['status'] not in {'prepared','running','worker_error'}:return old
    cmd=[sys.executable,str(ROOT/'scripts/run_rootse_deepseek_worker.py'),'--source',str(source),
         '--out',str(directory),'--config',str(out/'snapshot/config.json'),'--method',case['method'],
         '--file',case['source_file'],'--api-key-file',str(key_path)]
    with (directory/'execution.log').open('a') as log:
        result=subprocess.run(cmd,cwd=source,env={**os.environ,'PYTHONHASHSEED':'0'},stdout=log,stderr=subprocess.STDOUT)
    if not state_path.exists():
        state={'method':case['method'],'source_file':case['source_file'],'status':'worker_error','exit_code':result.returncode}
        save(state_path,state)
        return state
    state=json.loads(state_path.read_text())
    if state['status']=='running':
        state.update(status='worker_error',exit_code=result.returncode);save(state_path,state)
    return state


def write_summary(out: Path, manifest: dict) -> dict:
    """Audit actual completed cases and tokens without a fixed 102-case denominator."""
    from collections import Counter
    statuses = Counter()
    calls = []
    for case in manifest['cases']:
        directory = out / case['directory']
        state_path = directory / 'state.json'
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        statuses[state['status']] += 1
        for call_path in sorted(directory.glob('calls/*/state.json')):
            call = json.loads(call_path.read_text())
            data = json.loads((call_path.parent / call['response_file']).read_text()) if call.get('response_file') else {}
            choice = next(iter(data.get('choices', [])), {})
            message = choice.get('message', {})
            request = json.loads((call_path.parent / 'request.json').read_text())
            calls.append({'method': case['method'], 'source_file': case['source_file'],
                          'call_index': call['call_index'], 'status': call['status'],
                          'case_status': state['status'], 'finish_reason': choice.get('finish_reason'),
                          'response_model': data.get('model'), 'usage': data.get('usage'),
                          'input_characters': sum(len(m.get('content') or '') for m in request['messages']),
                          'content_characters': len(message.get('content') or ''),
                          'reasoning_characters': len(message.get('reasoning_content') or ''),
                          'attempts': call['attempts']})
    summary = {'model': manifest['protocol']['config']['model'],
               'endpoint': manifest['protocol']['config']['endpoint'],
               'workers': manifest['protocol']['config']['workers'],
               'selected_trajectories': len({c['source_file'] for c in manifest['cases']}),
               'method_cases': manifest['n'], 'statuses': dict(statuses),
               'incomplete_cases': manifest['n'] - sum(statuses.values()) + statuses.get('running', 0),
               'output_truncated_calls': sum(c['finish_reason'] == 'length' for c in calls),
               'context_overflow_cases': statuses.get('context_overflow', 0),
               'calls': calls, 'evaluated_at': now()}
    save(out / 'deepseek_summary.json', summary)
    return summary


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT.parent/'TrajAudit')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1')
    p.add_argument('--config',type=Path,default=ROOT/'experiments/rootse_suite/config_deepseek_official_flash.json')
    p.add_argument('--api-key-file',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--limit',type=int,help='Select N largest trajectories, for a separate smoke run')
    p.add_argument('--methods',nargs='+',choices=METHODS)
    p.add_argument('--workers',type=int,help='Concurrent isolated method/case workers')
    a=p.parse_args();source=a.source.resolve();out=a.out.resolve();config=json.loads(a.config.read_text())
    if a.limit is not None:
        if not 1 <= a.limit <= 102:p.error('--limit must be between 1 and 102')
        if out == (ROOT/'outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1').resolve():
            out=out.with_name(out.name+f'-smoke-{a.limit}')
        config['sample_limit']=a.limit
    if a.methods:config['methods']=list(dict.fromkeys(a.methods))
    if a.workers is not None:config['workers']=a.workers
    if config['workers'] < 1:p.error('--workers must be positive')
    approved = {('deepseek-flash', 'https://api.deepseek.com/chat/completions')}
    if (config['model'], config['endpoint']) not in approved:
        raise ValueError('Unapproved deployment')
    ensure_source(source,config);out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=prepare(source,out,config,a.config)
        if a.prepare_only:print(f"Prepared {manifest['n']} method/cases, no API calls");return 0
        if not a.report_only:
            if not a.api_key_file or not a.api_key_file.read_text().strip():p.error('--api-key-file is required')
            manifest['status']='running';save(out/'manifest.json',manifest)
            with ThreadPoolExecutor(max_workers=config['workers']) as pool:
                futures=[pool.submit(run_job,c,out,source,a.api_key_file.resolve()) for c in manifest['cases']]
                for n,future in enumerate(as_completed(futures),1):
                    state=future.result();print(f"[{n}/{manifest['n']}] {state['method']} {state['source_file']}: {state['status']}",flush=True)
            manifest.update(status='complete',finished_at=now());save(out/'manifest.json',manifest)
        summary=write_summary(out,manifest)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        if summary['incomplete_cases'] or any(status != 'valid' for status in summary['statuses']):return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
