#!/usr/bin/env python3
"""Inspect downloaded CodeTraceBench archives using a strict raw-evidence file allowlist."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

from run_contextmap import ROOT, read, sha
from suite_transport import save, now
from loghound_agent_debug.adapters.codetracebench_context import miniswe_context,openhands_context,terminus_context

DEST=ROOT.parents[2]/'output/attentionmap_v5_evidence_2026-10-30/codetracebench-format'


def raw_evidence(archive: Path) -> dict[str,bytes]:
    selected={}
    process=subprocess.Popen(['/opt/homebrew/bin/zstd','-dc',str(archive)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
        for member in tar:
            name=PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:continue
            allowed=(name.name.endswith('.traj.json') or name.name=='results.json' or
                (name.suffix=='.json' and name.parent.name=='sessions') or
                bool(re.search(r'/sessions/sessions/[^/]+/events/\d+\.json$',member.name)) or
                bool(re.search(r'/agent-logs/episode-\d+/(prompt|response)\.txt$',member.name)))
            if allowed:
                selected[member.name]=tar.extractfile(member).read()
    process.stdout.close()
    if process.wait()!=0:raise ValueError('Archive decompression failed')
    return selected


def normalize(meta: dict, files: dict[str,bytes]) -> dict:
    result_files=[b for name,b in files.items() if name.endswith('/results.json')]
    task=str(json.loads(result_files[0]).get('instruction') or '') if result_files else ''
    if meta['agent']=='mini-SWE-agent':
        raw=[json.loads(b) for name,b in files.items() if name.endswith('.traj.json')]
        if len(raw)!=1:raise ValueError('Ambiguous mini-SWE trajectory')
        packet=miniswe_context(raw[0],task or None)
    elif meta['agent']=='OpenHands':
        names=sorted(name for name in files if re.search(r'/sessions/sessions/[^/]+/events/\d+\.json$',name))
        if names:
            session=names[0].rsplit('/events/',1)[0]
            events=[json.loads(files[name]) for name in names if name.startswith(session+'/events/')]
        else:
            flat=[(name,b) for name,b in files.items() if PurePosixPath(name).parent.name=='sessions' and name.endswith('.json')]
            events=json.loads(max(flat,key=lambda x:len(x[1]))[1])
        packet=openhands_context(events,task)
    elif meta['agent']=='Terminus2':
        episodes={}
        for name,b in files.items():
            match=re.search(r'/episode-(\d+)/(prompt|response)\.txt$',name)
            if match:episodes.setdefault(int(match[1]),{})[match[2]]=b.decode('utf-8')
        packet=terminus_context(episodes,task)
    else:raise ValueError('Unsupported agent')
    return packet


def main() -> None:
    selection=read(DEST/'selected_metadata.json')
    audit=[]
    for meta in selection['cases']:
        cid=meta['traj_id'];archive=DEST/(cid+'.tar.zst');files=raw_evidence(archive)
        row={'traj_id':cid,'agent':meta['agent'],'expected_steps':meta['step_count'],
             'archive_sha256':sha(archive),'raw_members_read':list(files),
             'annotation_fields_used':0,'model_calls':0}
        try:
            packet=normalize(meta,files)
            row.update(parsed_steps=len(packet['case']['steps']),native_step_ids=packet['native_step_ids'])
            if row['expected_steps']!=row['parsed_steps']:
                raise ValueError('Native action count mismatch')
        except ValueError as exc:
            row.update(status='quarantined',error=str(exc))
        else:
            save(DEST/'normalized'/(cid+'.json'),packet)
            row['status']='count_and_format_matched'
        audit.append(row)
        print({k:row.get(k) for k in ('agent','expected_steps','parsed_steps','status','error')},flush=True)
    save(DEST/'normalization_audit.json',{'time':now(),'dataset_revision':selection['revision'],
         'parser_reference_commit':read(DEST/'official_code_tree.json')['sha'],'cases':audit,
         'adapter_sha256':sha(ROOT/'src/loghound_agent_debug/adapters/codetracebench_context.py'),
         'scope':'four-source format inspection; no model evaluation and no attribution labels read'})


if __name__=='__main__':
    main()
