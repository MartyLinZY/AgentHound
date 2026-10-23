#!/usr/bin/env python3
"""Apply one frozen attention checkpoint to label-free features and step IDs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loghound_agent_debug.rankers.attention_map import AttentionMapRanker, VERSION


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.npz').exists():p.error('Output already exists')
    torch.set_num_threads(1)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=AttentionMapRanker(checkpoint['input_dim'],hidden=checkpoint['hidden'],variant=checkpoint['variant'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    source=json.loads(a.input.read_text());source=source.get('item',source)
    ids=[str(s['source_step_id']) for s in source['steps']]
    with np.load(a.features,allow_pickle=False) as z:
        values=[torch.from_numpy(z[k]) for k in ('x','failure','prior')]
    if len(ids)!=len(values[0]) or len(ids)!=len(set(ids)):raise ValueError('Step ID mismatch')
    with torch.no_grad():result=model(*values)
    order=sorted(range(len(ids)),key=lambda i:(-float(result['scores'][i]),i))
    ranking=[{'rank':rank,'source_step_id':ids[i],'score':float(result['scores'][i]),
              'local':float(result['local'][i]),'relation':float(result['relation'][i])} for rank,i in enumerate(order,1)]
    response={'trajectory_id':source['trajectory_id'],'version':VERSION,'variant':checkpoint['variant'],
              'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'features_sha256':hashlib.sha256(a.features.read_bytes()).hexdigest(),'ranking':ranking}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if 'attention' in result:
        np.savez_compressed(a.out.with_suffix('.npz'),attention=result['attention'].numpy(),
                            seeds=result['seeds'].numpy(),mass=result['mass'].numpy(),step_ids=np.array(ids))
    a.out.write_text(json.dumps(response,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({'trajectory_id':source['trajectory_id'],'top1':ranking[0]['source_step_id'],'out':str(a.out)}))


if __name__=='__main__':main()
