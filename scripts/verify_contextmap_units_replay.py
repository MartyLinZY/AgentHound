#!/usr/bin/env python3
"""Replay all terminal states with network construction forbidden; verify immutability."""
import run_contextmap as base
from run_contextmap_units import OUT, METHODS, prepare
from suite_transport import save


def main() -> None:
    manifest = prepare(OUT)
    before = {str(p.relative_to(OUT)): base.sha(p) for p in (OUT / 'runs').rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Replay unexpectedly attempted an API client')
    base.RecordedClient = forbidden
    states = []
    for case in manifest['cases']:
        for method in METHODS:
            state = base.run_one(OUT, manifest, case['case_id'], method, 'not-a-credential')
            if state['status'] == 'running':
                raise ValueError('Incomplete run')
            states.append(state)
    after = {str(p.relative_to(OUT)): base.sha(p) for p in (OUT / 'runs').rglob('*') if p.is_file()}
    if before != after:
        raise ValueError('Replay modified raw results')
    historical = base.read(OUT / 'historical_hashes.json')
    from pathlib import Path
    for path, expected in historical.items():
        if base.sha(Path(path)) != expected:
            raise ValueError('Historical artifact changed')
    result = {'terminal_tasks_replayed': len(states), 'new_api_calls': 0,
              'new_run_files_unchanged': len(before), 'historical_files_unchanged': len(historical)}
    save(OUT / 'analysis/replay_verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
