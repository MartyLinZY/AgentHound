#!/usr/bin/env python3
"""After completion, rerun the one-command entry and verify zero new requests."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from migration_core import ROOT
from run_rootse_migration import DEFAULT
from suite_transport import save, now
from report_rootse_migration import accounting


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def hashes(out: Path) -> dict:
    paths = list(out.glob("targets/*/*/state.json")) + list(
        out.glob("targets/*/*/calls/*/state.json")
    )
    paths += list(out.glob("pilot/targets/*/*/state.json")) + list(
        out.glob("pilot/targets/*/*/calls/*/state.json")
    )
    paths += list(out.glob("ablations/*/*/rankings.jsonl"))
    return {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--api-key-file", type=Path, required=True)
    a = p.parse_args(argv)
    out = a.out.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    require(
        manifest["status"] == "complete",
        "Integrity check failed: manifest['status']=='complete'",
    )
    before = hashes(out)
    counts = [
        accounting(out)["api_requests"],
        accounting(out / "pilot")["api_requests"],
    ]
    save(out / "verification_before_hashes.json", before)
    with (out / "verification_reproduction.log").open("w") as log:
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/reproduce_rootse_migration.sh"),
                str(a.api_key_file.resolve()),
                str(out),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    after = hashes(out)
    counts_after = [
        accounting(out)["api_requests"],
        accounting(out / "pilot")["api_requests"],
    ]
    if not (result.returncode == 0 and before == after and counts == counts_after):
        raise ValueError(
            "Integrity check failed: result.returncode==0 and before==after and counts==counts_after"
        )
    secret = a.api_key_file.read_bytes().strip()
    require(len(secret) > 10, "Integrity check failed: len(secret)>10")
    files = [p for p in out.rglob("*") if p.is_file()]
    leaks = sum(secret in p.read_bytes() for p in files)
    require(leaks == 0, "Integrity check failed: leaks==0")
    save(
        out / "verification.json",
        {
            "verified_at": now(),
            "entry_exit_code": result.returncode,
            "all_target_call_and_ranking_hashes_unchanged": True,
            "hashed_files": len(before),
            "requests_before_formal_pilot": counts,
            "requests_after_formal_pilot": counts_after,
            "new_requests": 0,
            "secret_scan_files": len(files),
            "secret_matches": leaks,
            "tests": "passed within one-command entry; see verification_reproduction.log",
        },
    )
    print(
        "Verified unchanged states/rankings, zero new requests and no credential matches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
