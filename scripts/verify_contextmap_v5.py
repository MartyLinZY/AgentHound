#!/usr/bin/env python3
"""Exact reconstruction and no-network replay for v5."""
from __future__ import annotations

import argparse
from pathlib import Path

import run_contextmap as transport
from run_contextmap_v5 import OUTPUTS, PARENTS, METHODS
from run_contextmap import read, sha
from run_contextmap_balanced import check_manifest
from suite_transport import save, digest, now
from loghound_agent_debug.context_map import tokens
from loghound_agent_debug.context_units import structural_units
from loghound_agent_debug.context_evidence import build_evidence_view, review_messages
from loghound_agent_debug.context_coverage import build_coverage_view


def inputs(dataset: str) -> dict:
    out,parent=OUTPUTS[dataset],PARENTS[dataset]
    manifest=check_manifest(out)
    count=spans=0
    for c in manifest['cases']:
        cid=c['case_id'];case=read(out/'cases'/f'{cid}.json');mapping=read(out/'maps'/f'{cid}.json')
        original=read(parent/'prepared/balanced'/f'{cid}.json')
        candidates=None
        if dataset=='rootse':
            candidates=sorted(read(parent/'runs/balanced'/cid/'state.json')['decision']['ranked_step_ids'],key=int)
            expected=build_evidence_view(case,mapping,original['view'],candidates)
        else:
            expected=build_coverage_view(case,mapping,original['view'])
        for method,view in zip(METHODS[dataset],(original['view'],expected)):
            p=read(out/'prepared'/method/f'{cid}.json')
            assert p['view']==view
            if dataset=='rootse':
                messages=review_messages(original['messages'],original['view'],view,candidates)
                assert p['initial_candidate_ids']==candidates
            else:
                from copy import deepcopy
                messages=deepcopy(original['messages'])
                messages[1]['content']=messages[1]['content'].replace(original['view']['content'],view['content'],1)
            assert p['messages']==messages and digest(messages)==p['messages_sha256']
            assert tokens(view['content'])==view['context_tokens']<=16384
            for f in view['fragments']:
                source=case['steps'][int(f['step_id'])][f['field']]
                assert f['text']==source[f['start']:f['end']]
                assert (f['start'],f['end']) in {(a,b) for a,b,_ in structural_units(source)}
                spans+=1
            count+=1
    result={'prepared_requests':count,'exact_spans':spans,'fingerprint':manifest['fingerprint'],
            'network_calls':0,'time':now()}
    save(out/'verification/input_reconstruction.json',result)
    print(dataset,result,flush=True)
    return result


def replay(dataset: str) -> dict:
    out=OUTPUTS[dataset];manifest=check_manifest(out)
    assert read(out/'run_status.json')['completed']==manifest['n']*2
    before={str(p):sha(p) for p in (out/'runs').rglob('*') if p.is_file()}
    protected={**read(out/'historical_hashes.json'),**read(out/'parent_run_hashes.json')}
    def forbidden(*args,**kwargs):
        raise RuntimeError('Replay attempted network')
    saved=transport.RecordedClient;transport.RecordedClient=forbidden
    count=0
    try:
        for c in manifest['cases']:
            for m in METHODS[dataset]:
                assert transport.run_one(out,manifest,c['case_id'],m,'NOT_A_CREDENTIAL')['status']!='running'
                count+=1
    finally:
        transport.RecordedClient=saved
    for path,h in {**before,**protected}.items():
        assert sha(Path(path))==h
    result={'cached_tasks':count,'network_calls':0,'unchanged_run_files':len(before),
            'unchanged_historical_files':len(protected),'time':now()}
    save(out/'verification/replay.json',result)
    print(dataset,result,flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('rootse','whowhen','both'),default='both')
    parser.add_argument('--stage',choices=('inputs','replay'),required=True)
    args=parser.parse_args()
    for d in OUTPUTS:
        if args.dataset in (d,'both'):
            (inputs if args.stage=='inputs' else replay)(d)


if __name__=='__main__':
    main()
