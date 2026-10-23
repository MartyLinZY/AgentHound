#!/usr/bin/env python3
"""Read-only, offline preflight of existing dependency extraction requests."""
import json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.rootse_dependencies import select_candidates,_build_request

items,_=load_rootse(ROOT.parent/'TrajAudit/RootSE')
rows=[];totals=Counter()
for item in items:
    row=Counter(steps=len(item.steps));sizes=[]
    for target in range(len(item.steps)):
        candidates=select_candidates(item.steps,target)
        if not candidates:row['no_candidates']+=1;continue
        request,meta=_build_request(item.steps,target,candidates)
        if request is None:row[meta['status']]+=1;continue
        row['planned_logical_calls']+=1
        row['selected_candidates']+=len(candidates)
        row['shown_candidates']+=len(meta['candidate_ordinals'])
        fields=[v for fields in meta['truncation'].values() for v in fields.values()]
        row['field_occurrences']+=len(fields)
        row['truncated_field_occurrences']+=sum(v['truncated'] for v in fields)
        row['calls_with_truncated_field']+=int(any(v['truncated'] for v in fields))
        sizes.append(meta['messages_chars'])
    totals.update(row)
    rows.append({'trajectory_id':item.trajectory_id,**dict(row),'max_input_chars':max(sizes,default=0)})
result={'scope':'offline request construction only; no model or credential used','trajectories':len(items),'totals':dict(totals),
        'max_logical_calls_per_trajectory':max(r.get('planned_logical_calls',0) for r in rows),
        'max_input_chars':max(r['max_input_chars'] for r in rows),'per_trajectory':rows}
out=ROOT/'experiments/rootse_next';out.mkdir(exist_ok=True)
(out/'readiness_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='per_trajectory'},ensure_ascii=False,indent=2))
