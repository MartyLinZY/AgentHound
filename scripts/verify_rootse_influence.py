#!/usr/bin/env python3
"""Verify complete cached replay and protected artifacts with network creation forbidden."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import json
from run_rootse_influence import DEFAULT,prepare,run,rank_all,sha
from suite_transport import RecordedClient,save


def main():
    out=DEFAULT;m=prepare(out)
    if m['status']!='complete': raise ValueError('Cannot verify incomplete run')
    paths=[p for p in out.rglob('*') if p.is_file() and
           (p.name in {'state.json','request.json'} or p.name.startswith('response-') or 'rankings' in p.relative_to(out).parts)]
    before={str(p):sha(p) for p in paths}
    def forbidden(*args,**kwargs): raise RuntimeError('Unexpected uncached request during offline verification')
    RecordedClient.create=forbidden
    with tempfile.NamedTemporaryFile(mode='w') as key:
        key.write('offline-cache-test');key.flush()
        with contextlib.redirect_stdout(io.StringIO()):
            result=run(m,out,Path(key.name));rank_all(m,out)
    changed=[p for p,h in before.items() if sha(Path(p))!=h]
    if changed: raise ValueError('Replay changed saved results: '+str(changed[:5]))
    protected=json.loads((out/'preservation_before.json').read_text())
    protected_changed=[p for p,h in protected.items() if not Path(p).exists() or sha(Path(p))!=h]
    if protected_changed: raise ValueError('Protected files changed')
    record={'terminal_jobs_replayed':result['n_jobs'],'new_api_requests':0,'unchanged_raw_and_ranking_files':len(paths),
            'changed_files':changed,'protected_files':len(protected),'protected_changes':protected_changed}
    save(out/'analysis/cache_verification.json',record);print(json.dumps(record))


if __name__=='__main__':main()
