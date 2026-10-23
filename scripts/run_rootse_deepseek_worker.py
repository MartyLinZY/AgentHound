#!/usr/bin/env python3
"""Run one upstream method/case with no gold fields exposed to the algorithm."""
from __future__ import annotations
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import ModuleType, SimpleNamespace
from deepseek_suite_transport import RecordedClient, ExperimentAbort, save, now, digest

METHODS = {
 'all_at_once': ('baseline.who_and_when.all_at_once','analyze_all_at_once'),
 'step_by_step': ('baseline.who_and_when.step_by_step','analyze_step_by_step'),
 'binary_search': ('baseline.who_and_when.binary_search','analyze_binary_search'),
 'TRAIL': ('baseline.trail.trail_adapted','analyze_trail'),
 'FAMAS': ('baseline.FAMAS.famas','analyze_famas'),
 'agent_locate': ('TrajAudit.fault_localization_generic','analyze_agent_locate_generic'),
}


def setup_source(source: Path, config: dict):
    # Inject a credential-free config before importing reviewed upstream modules.
    cfg = ModuleType('config')
    for name,value in {'MODEL':config['model'],'API_KEY':'RUNTIME_TRANSPORT_ONLY',
                       'BASE_URL':config['endpoint'].removesuffix('/chat/completions'),
                       'TEMPERATURE':config['temperature'],'TIMEOUT':config['timeout_seconds'],
                       'WITH_GROUND_TRUTH':False,'WITH_CONTEXT_AWARE_TRUNCATION':True,'WITH_DIAGNOSIS':True}.items():
        setattr(cfg,name,value)
    sys.modules['config'] = cfg
    sys.path.insert(0,str(source))
    return importlib.import_module('utils')


def safe_failure(raw: dict, utils):
    test = next(iter(raw['failed_tests'].values()))
    fields = {'index','thought','response','action','observation','execution_time','state','extra_info'}
    steps = [json.dumps({k:v for k,v in step.items() if k in fields},indent=2) for step in raw['original_traj']]
    return utils.Failure(instance_id=raw['instance_id'],agent=raw['agent'],model=raw['model'],
                        original_mission_prompt=raw['original_mission_prompt'],original_traj=steps,
                        failed_test_code=test['code'],failed_test_error_description=test['error_description'],
                        failed_test_expectation='',gold_patch='',failure_id='WITHHELD')


def execute_case(source: Path, out: Path, config: dict, method: str, rel: str, key: str) -> dict:
    state_path = out/'state.json'
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old['status'] not in {'prepared','running','worker_error'}:
            return old
    utils = setup_source(source,config)
    raw = json.loads((source/rel).read_text())
    failure = safe_failure(raw,utils)
    ids=[str(s['index']) for s in raw['original_traj']]
    if ids != [str(i) for i in range(len(ids))]:
        raise ValueError('upstream algorithms require zero-based contiguous indices')
    state={'method':method,'source_file':rel,'trajectory_id':'rootse/'+rel.removeprefix('RootSE/').removesuffix('.json'),
           'agent':failure.agent,'trajectory_model':failure.model,'instance_id':failure.instance_id,
           'status':'running','started_at':now(),'step_ids':ids}
    save(state_path,state)
    client=RecordedClient(out,config,key)
    module=importlib.import_module(METHODS[method][0])
    def factory(**kwargs):
        if kwargs.get('http_client') is not None:
            kwargs['http_client'].close()
        return client
    module.OpenAI=factory
    if method=='agent_locate':
        # Upstream constructs an unused insecure httpx client; all actual traffic
        # goes through RecordedClient with normal TLS verification.
        module.httpx=SimpleNamespace(Client=lambda **kwargs: SimpleNamespace(close=lambda:None))
    upstream_dir=out/'upstream'
    if upstream_dir.exists():
        shutil.rmtree(upstream_dir)
    if method=='FAMAS':
        analyzer=module.FAMASAnalyzer.__new__(module.FAMASAnalyzer)
        analyzer.failure=failure
        refs,outcomes=analyzer._load_reference_trajs()
        state['reference_trajectories']={'n':len(refs),'passing':sum(outcomes),'failing':len(refs)-sum(outcomes),
                                         'trajectory_hashes':[digest(r) for r in refs], 'steps':[len(r) for r in refs]}
        save(state_path,state)
    try:
        getattr(module,METHODS[method][1])(failure,str(upstream_dir),1)
        results=list(upstream_dir.rglob('result.csv'))
        rows=[]
        for p in results:
            with p.open(newline='') as f: rows.extend(csv.DictReader(f))
        if len(rows)==1:
            row=rows[0]
            state.update(failure_id=str(row['failure_id']),failure_reason=row.get('failure_reason',''),upstream_result=row)
            state['status']='valid' if state['failure_id'] in ids else 'no_prediction'
        elif not rows:
            state.update(status='no_prediction',failure_reason='Upstream returned without submitting a result')
        else:
            state.update(status='algorithm_error',failure_reason='Multiple upstream predictions; refusing to select one')
    except ExperimentAbort as exc:
        state.update(status=exc.status,failure_reason=str(exc))
    except Exception as exc:
        state.update(status='algorithm_error',failure_reason=type(exc).__name__+': '+str(exc).replace(key,'[REDACTED]'))
        print(traceback.format_exc().replace(key,'[REDACTED]'))
    state.update(finished_at=now(),logical_calls=client.index)
    save(state_path,state)
    return state


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--file',required=True);p.add_argument('--api-key-file',type=Path,required=True)
    a=p.parse_args();os.chdir(a.source)
    result=execute_case(a.source,a.out,json.loads(a.config.read_text()),a.method,a.file,a.api_key_file.read_text().strip())
    print(json.dumps({'method':a.method,'source_file':a.file,'status':result['status']},ensure_ascii=False))
    return 2 if result['status']=='worker_error' else 0

if __name__=='__main__':raise SystemExit(main())
