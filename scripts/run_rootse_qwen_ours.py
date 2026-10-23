#!/usr/bin/env python3
"""Rerun our graph and earliest-error judge with Qwen thinking disabled."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

import run_rootse_migration as graph_runner
from migration_core import ROOT
from suite_transport import RecordedClient, ExperimentAbort, save, digest, now
from loghound_agent_debug.adapters.rootse import load_rootse
from loghound_agent_debug.schema import TrajectoryInput
from loghound_agent_debug.evidence import load_clue_rules
from loghound_agent_debug.rankers.loghound_static import load_config
from loghound_agent_debug.rankers.anchor_path import rank_trajectory
from loghound_agent_debug.rankers.causal_judge import build_messages, parse_and_validate, promote_decision, VERSION

DEFAULT = ROOT/'outputs/rootse-ours-qwen38-nothinking-causal-v1'
CONFIG = ROOT/'experiments/rootse_migration/config_qwen38_nothinking_causal.json'


def require_nothinking(config: dict) -> None:
    if config.get('enable_thinking') is not False or config.get('stream') is not True:
        raise ValueError('Qwen must explicitly disable thinking with audited streaming')
    if config.get('model') != 'qwen3.8-flash' or config.get('workers', 0) < 1:
        raise ValueError('Unexpected model or invalid worker count')
    if 'thinking' in config or 'reasoning_effort' in config:
        raise ValueError('Conflicting thinking parameters')


def judge_case(item: TrajectoryInput, graph: dict, retrievals: dict, out: Path,
               config: dict, key: str, client_factory=RecordedClient) -> dict:
    require_nothinking(config)
    messages = build_messages(item, graph['dependencies'], retrievals)
    signature = digest({'messages': messages, 'config': config, 'version': VERSION})
    path = out/'state.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['request_sha256'] != signature:
            raise ValueError('Judge protocol changed; cannot mix cached results')
        if previous['status'] != 'running':
            return previous
    state = {'trajectory_id': item.trajectory_id, 'status': 'running',
             'request_sha256': signature, 'started_at': now(), 'decision': None,
             'messages_chars': sum(len(m['content']) for m in messages)}
    save(path, state)
    try:
        answer = client_factory(out, config, key).create(model=config['model'], messages=messages)
        state['decision'] = parse_and_validate(answer.choices[0].message.content, item)
        state['status'] = 'valid'
    except ExperimentAbort as exc:
        if exc.status in {'model_mismatch', 'replay_mismatch', 'call_limit'}:
            raise RuntimeError(str(exc)) from exc
        state.update(status=exc.status, failure_reason=str(exc))
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        state.update(status='format_error', failure_reason=f'{type(exc).__name__}: {str(exc)[:300]}')
    state['finished_at'] = now()
    save(path, state)
    return state


def judge_all(source: Path, out: Path, config: dict, key_path: Path, pilot: bool = False) -> dict:
    items, _ = load_rootse(source/'RootSE', protocol='postmortem')
    cfg = load_config(out/'snapshot/configs/default.yaml')
    rules = load_clue_rules(out/'snapshot/configs/clue_rules.yaml')
    key = key_path.read_text().strip()
    if not key:
        raise ValueError('Empty credential')
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Full graph extraction must finish before adjudication')
    prepared = []
    for item in items:
        cid = hashlib.sha256(item.trajectory_id.encode()).hexdigest()[:16]
        graph = json.loads((out/'graphs'/f'{cid}.json').read_text())
        retrievals = {v: rank_trajectory(item, cfg, rules, graph['dependencies'], v)
                      for v in ['baseline', 'anchors-global', 'anchors-backtrace']}
        prepared.append((item, cid, graph, retrievals))
    if pilot:
        prepared = [max(prepared, key=lambda p: sum(len(m['content']) for m in build_messages(p[0], p[2]['dependencies'], p[3])))]
    base = out/'judge_pilot' if pilot else out/'judge'
    counts = Counter()
    def job(data: tuple) -> dict:
        item, cid, graph, retrievals = data
        state = judge_case(item, graph, retrievals, base/cid, config, key)
        if not pilot:
            for variant, ranking in retrievals.items():
                save(out/'retrievals'/variant/f'{cid}.json', ranking)
            save(out/'rankings'/f'{cid}.json', promote_decision(retrievals['anchors-backtrace'], state['decision']))
        return state
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures = [pool.submit(job, p) for p in prepared]
        for n, future in enumerate(as_completed(futures), 1):
            state = future.result(); counts[state['status']] += 1
            progress = {'stage': 'judge_pilot' if pilot else 'judge', 'seen': n,
                        'planned': len(prepared), 'status_counts': dict(counts), 'updated_at': now()}
            save(base/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    result = {'status': 'complete', 'n': len(prepared), 'status_counts': dict(counts), 'finished_at': now()}
    if pilot:
        result['technical_gate_passed'] = counts['valid'] == len(prepared)
    save(base/'run_status.json', result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT.parent/'TrajAudit')
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--api-key-file', type=Path)
    p.add_argument('--stage', choices=['prepare','pilot','graph','judge','all'], default='all')
    a = p.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text()); require_nothinking(config)
    # The shared graph runner already freezes all src files and its own runner/core.
    own_paths = [Path(__file__).resolve()]
    config['driver_hashes'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in own_paths}
    with (out/'.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = graph_runner.prepare(a.source.resolve(), out, config)
        for f in own_paths:
            dest = out/'snapshot'/f.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dest)
        if a.stage == 'prepare':
            print(json.dumps({k:m[k] for k in ['fingerprint','n_cases','n_targets','max_input_chars']},indent=2)); return 0
        if a.api_key_file is None:
            p.error('--api-key-file required for execution')
        if a.stage in {'pilot','all'}:
            pilot = graph_runner.run(m, out, a.api_key_file, True)
            if not pilot['technical_gate_passed']: return 2
        if a.stage in {'graph','all'}:
            pilot_path = out/'pilot/run_status.json'
            if not pilot_path.exists() or not json.loads(pilot_path.read_text()).get('technical_gate_passed'):
                raise ValueError('Technical graph pilot must pass before full extraction')
            result = graph_runner.run(m, out, a.api_key_file, False)
            if result['status'] != 'complete': return 2
        if a.stage in {'judge','all'}:
            probe = judge_all(a.source.resolve(), out, config, a.api_key_file, pilot=True)
            if not probe['technical_gate_passed']: return 2
            judge_all(a.source.resolve(), out, config, a.api_key_file)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
