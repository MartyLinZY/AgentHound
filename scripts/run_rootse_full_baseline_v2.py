#!/usr/bin/env python3
"""Frozen, resumable RootSE All-at-Once experiment; evaluation is a separate step."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Callable

import requests
from run_qwen_baseline import endpoint_for, messages_for, save

def validate_model(model: str) -> None:
    # Explicit user-selected MoMA million-context deployment; never Claude.
    if model != 'deepseek-v4.1-flash':
        raise ValueError('v2 is frozen to the selected DeepSeek deployment')


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'valid', 'context_overflow', 'format_error', 'invalid_step', 'output_truncated',
            'transport_error', 'service_error', 'model_mismatch'}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def upstream_extractor(source: Path) -> Callable[[str], str]:
    """Compile only the reviewed pure extract_json function, not upstream config."""
    tree = ast.parse((source / 'utils.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_json')
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {'re': re}
    exec(compile(module, str(source / 'utils.py'), 'exec'), namespace)
    return namespace['extract_json']


def classify_response(data: dict, step_ids: list[str], extract: Callable[[str], str]) -> dict:
    reported = str(data.get('model', ''))
    if 'claude' in reported.lower():
        return {'status': 'model_mismatch'}
    try:
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop':
            return {'status': 'output_truncated'}
        prediction = json.loads(extract(choice['message']['content']))
        step = str(prediction['failure_id'])
        reason = prediction['failure_reason']
        if not isinstance(reason, str):
            return {'status': 'format_error'}
        if step not in step_ids:
            return {'status': 'invalid_step', 'returned_step_id': step}
        return {'status': 'valid', 'failure_id': step, 'failure_reason': reason}
    except (ValueError, TypeError, KeyError, IndexError):
        return {'status': 'format_error'}


def run_case(state_path: Path, out: Path, config: dict, key: str,
             extract: Callable[[str], str], post: Callable = requests.post,
             sleep: Callable = time.sleep) -> dict:
    state = json.loads(state_path.read_text())
    if state['status'] in TERMINAL:
        return state
    payload_path = out / state['request_file']
    payload = json.loads(payload_path.read_text())
    if sha(canonical(payload)) != state['request_sha256']:
        raise ValueError('saved request changed; refusing resume')
    # An interrupted attempt with a saved response can be parsed without another
    # billable request. An attempt interrupted before response receipt is unknown.
    for attempt in state['attempts']:
        if attempt['status'] == 'started':
            response_path = out / attempt['response_file']
            if response_path.exists():
                saved = json.loads(response_path.read_text())
                attempt.update(saved['metadata'], status='received')
                state.update(classify_response(saved['data'], state['step_ids'], extract))
                save(state_path, state)
                return state
            attempt.update(status='interrupted_unknown', usage_unknown=True)
    while len(state['attempts']) < config['transport_attempts']:
        number = len(state['attempts']) + 1
        response_file = f"responses/{state['case_id']}-{number}.json"
        attempt = {'number': number, 'status': 'started', 'started_at': now(),
                   'response_file': response_file}
        state['attempts'].append(attempt)
        state['status'] = 'running'
        save(state_path, state)
        started = time.monotonic()
        retry = False
        try:
            response = post(config['endpoint'], headers={'Authorization': 'Bearer ' + key}, json=payload,
                            timeout=(20, config['timeout_seconds']), allow_redirects=False)
            attempt.update(http_status=response.status_code, elapsed_seconds=time.monotonic() - started)
            if response.status_code != 200:
                text = response.text.replace(key, '[REDACTED]')
                attempt.update(status='http_error', error=text[:3000], usage_unknown=True)
                if response.status_code == 400 and ('context' in text.lower() or 'input_tokens' in text.lower()):
                    state['status'] = 'context_overflow'
                else:
                    state['status'] = 'service_error'
                    retry = response.status_code == 429 or response.status_code >= 500
            else:
                data = response.json()
                safe = json.loads(json.dumps(data).replace(key, '[REDACTED]'))
                attempt.update(status='received', usage=data.get('usage') or {},
                               response_model=data.get('model'), finished_at=now())
                save(out / response_file, {'metadata': attempt.copy(), 'data': safe})
                state.update(classify_response(safe, state['step_ids'], extract))
        except requests.RequestException as exc:
            attempt.update(status='transport_error', error_type=type(exc).__name__,
                           elapsed_seconds=time.monotonic() - started, usage_unknown=True)
            state['status'] = 'transport_error'
            retry = True
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            attempt.update(status='response_error', error_type=type(exc).__name__)
            state['status'] = 'format_error'
        if retry and number < config['transport_attempts']:
            state['status'] = 'retry_pending'
        state['updated_at'] = now()
        save(state_path, state)
        if not retry or number == config['transport_attempts']:
            break
        sleep(min(5 * number, 15))
    if state['status'] not in TERMINAL:
        state['status'] = 'transport_error'
        save(state_path, state)
    return state


def ensure_source(source: Path, config: dict) -> None:
    if not source.exists():
        subprocess.run(['git', 'clone', config['source_repository'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', config['source_commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != config['source_commit']:
        raise ValueError('upstream commit differs from frozen configuration')


def prepare(source: Path, out: Path, config: dict, config_path: Path) -> dict:
    expected = json.loads((config_path.parent / config['dataset_manifest']).read_text())
    actual_files = {str(p.relative_to(source)) for p in (source / 'RootSE').rglob('*.json')}
    if actual_files != set(expected) or len(expected) != 102:
        raise ValueError('the experiment requires the exact frozen 102-case RootSE release')
    for rel, digest in expected.items():
        if sha((source / rel).read_bytes()) != digest:
            raise ValueError(f'dataset hash mismatch: {rel}')
    code_paths = [Path(__file__), Path(__file__).with_name('run_qwen_baseline.py')]
    prompt_paths = [source / 'utils.py', source / 'baseline/who_and_when/all_at_once.py']
    protocol = {'config': config, 'dataset_hash': sha(canonical(expected)),
                'code_hashes': {p.name: sha(p.read_bytes()) for p in code_paths},
                'prompt_hashes': {str(p.relative_to(source)): sha(p.read_bytes()) for p in prompt_paths}}
    fingerprint = sha(canonical(protocol))
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('configuration/code/data changed; use a new output directory')
        return manifest
    for name in ('states', 'requests', 'responses', 'snapshot'):
        (out / name).mkdir(parents=True, exist_ok=True)
    for p in code_paths:
        shutil.copyfile(p, out / 'snapshot' / p.name)
    for p in prompt_paths:
        shutil.copyfile(p, out / 'snapshot' / p.name)
    save(out / 'snapshot/config.json', config)
    save(out / 'snapshot/dataset_manifest.json', expected)
    shutil.copyfile(config_path.parent / config['paper_reference'], out / 'snapshot/paper_reference.json')
    cases = []
    for rel in sorted(expected):
        raw = json.loads((source / rel).read_text())
        case_id = sha(rel.encode())[:16]
        step_ids = [str(s['index']) for s in raw['original_traj']]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError('duplicate step IDs')
        payload = {'model': config['model'], 'messages': messages_for(raw, source),
                   'temperature': config['temperature'], 'max_tokens': config['max_tokens']}
        request_file = f'requests/{case_id}.json'
        save(out / request_file, payload)
        state = {'case_id': case_id, 'trajectory_id': 'rootse/' + rel.removeprefix('RootSE/').removesuffix('.json'),
                 'source_file': rel, 'agent': raw['agent'], 'trajectory_model': raw['model'],
                 'instance_id': raw['instance_id'], 'step_ids': step_ids,
                 'request_file': request_file, 'request_sha256': sha(canonical(payload)),
                 'input_chars': len(json.dumps(payload['messages'])), 'status': 'prepared', 'attempts': []}
        save(out / f'states/{case_id}.json', state)
        cases.append({'case_id': case_id, 'input_chars': state['input_chars']})
    # Longest cases first: expose deployment limits early, without label access.
    cases.sort(key=lambda x: (-x['input_chars'], x['case_id']))
    manifest = {'fingerprint': fingerprint, 'protocol': protocol, 'started_at': now(),
                'status': 'prepared', 'cases': cases, 'n': len(cases), 'model': config['model']}
    save(manifest_path, manifest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/rootse_full/config.json')
    p.add_argument('--source', type=Path, default=ROOT.parent / 'TrajAudit')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/rootse-full-qwen25-14b-1m-v1')
    p.add_argument('--api-key-file', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    validate_model(config['model'])
    config['endpoint'] = endpoint_for(config['endpoint'])
    ensure_source(args.source, config)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.source, args.out, config, args.config)
        if args.prepare_only:
            print(f"Prepared {manifest['n']} full untruncated inputs; no requests sent.")
            return 0
        if not args.report_only:
            if args.api_key_file is None:
                p.error('--api-key-file is required for inference')
            key = args.api_key_file.read_text().strip()
            if not key:
                p.error('empty API key file')
            extract = upstream_extractor(args.source)
            manifest['status'] = 'running'
            save(args.out / 'manifest.json', manifest)
            with ThreadPoolExecutor(max_workers=config['workers']) as pool:
                futures = [pool.submit(run_case, args.out / f"states/{c['case_id']}.json",
                                       args.out, config, key, extract) for c in manifest['cases']]
                for count, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    print(f"[{count}/{manifest['n']}] {result['trajectory_id']}: {result['status']}", flush=True)
            manifest.update(status='complete', finished_at=now())
            save(args.out / 'manifest.json', manifest)
        subprocess.run([sys.executable, str(ROOT / 'scripts/report_rootse_baseline.py'),
                        '--source', str(args.source), '--out', str(args.out)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
