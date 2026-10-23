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
from suite_transport import save,now,digest
from run_rootse_suite_worker import METHODS
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
    code_paths=[ROOT/'scripts'/name for name in ['run_rootse_suite.py','run_rootse_suite_worker.py','suite_transport.py']]
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
    cmd=[sys.executable,str(ROOT/'scripts/run_rootse_suite_worker.py'),'--source',str(source),
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


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT.parent/'TrajAudit')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/rootse-suite-deepseek-v41-flash-32k-v1')
    p.add_argument('--config',type=Path,default=ROOT/'experiments/rootse_suite/config.json')
    p.add_argument('--api-key-file',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();source=a.source.resolve();out=a.out.resolve();config=json.loads(a.config.read_text())
    approved = {
        ('deepseek-v4.1-flash', 'https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions'),
        ('deepseek-v4-flash-0731', 'https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions'),
        ('deepseek-v4-flash', 'https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions'),
        ('qwen3.8-flash', 'https://llm-qyefq7nc90xr3msm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions'),
    }
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
        subprocess.run([sys.executable,str(ROOT/'scripts/report_rootse_suite.py'),'--out',str(out),'--source',str(source)],check=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
