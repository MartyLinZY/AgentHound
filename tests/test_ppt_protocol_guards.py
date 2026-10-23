"""The public RootSE path must not silently use generic same-agent edges."""

from pathlib import Path
import subprocess
import sys

import pytest

from loghound_agent_debug.rankers.loghound_static import rank_trajectory
from loghound_agent_debug.schema import Step, TrajectoryInput

ROOT = Path(__file__).resolve().parents[1]


def test_rootse_requires_explicit_dependency_graph():
    item = TrajectoryInput(
        "0.1.0",
        "rootse",
        "t",
        "task",
        "coding",
        "failure",
        [],
        [
            Step(0, "0", ["0"], "agent", "assistant", ""),
            Step(1, "1", ["1"], "agent", "assistant", ""),
        ],
    )
    with pytest.raises(ValueError, match="explicit"):
        rank_trajectory(item, {}, {"rules": []})
    result = rank_trajectory(item, {}, {"rules": []}, [])
    assert result["graph_stats"]["edges"] == 0


@pytest.mark.parametrize(
    "module",
    [
        "verify_rootse_migration",
        "audit_rootse_migration_results",
        "plot_rootse_migration",
    ],
)
def test_report_tools_do_not_run_on_import(module):
    # Plotting uses its optional analysis environment; test import guards without
    # requiring matplotlib in the inference environment.
    if module == "plot_rootse_migration":
        source = (ROOT / "scripts" / f"{module}.py").read_text()
        assert 'if __name__ == "__main__":' in source
        return
    code = f"import sys;sys.path.insert(0,{str(ROOT / 'scripts')!r});import {module}"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_optimized_python_still_rejects_incomplete_verification(tmp_path):
    (tmp_path / "manifest.json").write_text('{"status":"running"}')
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(ROOT / "scripts/verify_rootse_migration.py"),
            "--out",
            str(tmp_path),
            "--api-key-file",
            str(tmp_path / "not-read.key"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "complete" in result.stderr
    assert not (tmp_path / "verification_before_hashes.json").exists()
