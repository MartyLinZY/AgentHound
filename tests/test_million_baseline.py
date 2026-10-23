"""The new deployment must not change the frozen inference protocol."""
import ast
from pathlib import Path
import pytest
from run_rootse_full_baseline_v2 import validate_model


def test_only_model_guard_changed():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    normalized = []
    for name in ['run_rootse_full_baseline.py', 'run_rootse_full_baseline_v2.py']:
        tree = ast.parse((scripts / name).read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'validate_model')]
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'run_qwen_baseline':
                node.names = [n for n in node.names if n.name != 'validate_model']
        normalized.append(ast.dump(tree, include_attributes=False))
    assert normalized[0] == normalized[1]


def test_selected_deployment_only():
    validate_model('deepseek-v4.1-flash')
    for model in ['claude-sonnet-4.5', 'YOUR_MODEL', 'DeepSeek-V3']:
        with pytest.raises(ValueError):
            validate_model(model)
