#!/usr/bin/env python3
"""Probe a named Qwen deployment on the longest RootSE prompt, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests
from run_qwen_baseline import endpoint_for, messages_for, save, select_files, validate_model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('../TrajAudit'))
    p.add_argument('--api-key-file', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--endpoint', default='https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=300)
    args = p.parse_args()
    validate_model(args.model)
    if args.out.exists():
        p.error('refusing to overwrite a previous probe')
    ranked = [(len(json.dumps(messages_for(json.loads(f.read_text()), args.source))), f)
              for f in select_files(args.source / 'RootSE', 0)]
    chars, path = max(ranked)
    payload = {'model': args.model, 'messages': messages_for(json.loads(path.read_text()), args.source),
               'temperature': 0, 'max_tokens': 1}
    key = args.api_key_file.read_text().strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'started', 'model': args.model, 'endpoint': endpoint_for(args.endpoint),
              'file': str(path.relative_to(args.source)), 'input_chars': chars, 'max_tokens': 1,
              'input_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
              'api_requests': 1, 'purpose': 'context_acceptance_only_not_diagnosis'}
    save(args.out, result)
    print(f'Probing {args.model}: longest input {chars} characters; max_tokens=1', flush=True)
    started = time.monotonic()
    try:
        response = requests.post(result['endpoint'], headers={'Authorization': 'Bearer ' + key},
                                 json=payload, timeout=(20, args.timeout), allow_redirects=False)
        result.update(http_status=response.status_code, status='returned',
                      response=response.text.replace(key, '[REDACTED]'))
    except requests.RequestException as exc:
        result.update(status='transport_error', error_type=type(exc).__name__)
    result['elapsed_seconds'] = time.monotonic() - started
    save(args.out, result)
    print(json.dumps(result, ensure_ascii=False)[:3500])
    return 0 if result.get('http_status') == 200 else 2


if __name__ == '__main__':
    raise SystemExit(main())
