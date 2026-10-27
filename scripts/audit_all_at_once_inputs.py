#!/usr/bin/env python3
"""Audit saved All-at-Once requests, without invoking an inference service."""
from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/attentionmap-aa-input-audit-20261027"
SOURCE = ROOT / "outputs/attentionmap-semantic-v2/semantic"
SUITES = {
    "qwen38": ROOT / "outputs/rootse-suite-aliyun-qwen38-flash-thinking-32k-v1",
    "deepseek_official": ROOT / "outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1",
    "moma_deepseek41": ROOT.parent / "AgentHound/outputs/rootse-suite-deepseek-v41-flash-32k-v1",
}
FIELDS = {"index", "thought", "response", "action", "observation", "execution_time", "state", "extra_info"}
TOOL_KEYS = {"tools", "tool_choice", "function_call", "functions", "tool_calls", "enable_search", "search_options",
             "search_info", "web_search_call", "web_extractor_call", "plugins", "citations", "server_tool_use"}
GOLD_KEYS = {"failure_id", "failure_reason", "gold_patch", "root_cause_steps", "root_cause", "ground_truth"}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assigned_literal(path, name):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)


def keys_in(value, selected, prefix=""):
    results = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k in selected:
                results.append({"path": prefix+k, "nonempty": bool(v)})
            results.extend(keys_in(v, selected, prefix+k+"."))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            results.extend(keys_in(v, selected, prefix+str(i)+"."))
    return results


def main():
    OUT.mkdir(exist_ok=True)
    manifest = read(SOURCE / "manifest.json")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    labels = read(SOURCE / "labels.json")
    baseline = read(ROOT / "outputs/attentionmap-top1-diagnosis-20261027/baseline_cases.json")
    hashes = {}

    def record(p):
        hashes[str(p)] = sha(p)

    for path in (SOURCE / "manifest.json", SOURCE / "labels.json",
                 ROOT / "outputs/attentionmap-top1-diagnosis-20261027/baseline_cases.json"):
        record(path)
    rows = []
    for suite, directory in SUITES.items():
        code = directory / "snapshot/upstream/baseline/who_and_when/all_at_once.py"
        utils = directory / "snapshot/upstream/utils.py"
        worker = "run_rootse_deepseek_worker.py" if suite == "deepseek_official" else "run_rootse_suite_worker.py"
        transport = "deepseek_suite_transport.py" if suite == "deepseek_official" else "suite_transport.py"
        for path in (code, utils, directory / "snapshot" / worker, directory / "snapshot" / transport):
            record(path)
        system = assigned_literal(code, "system_prompt")
        template = assigned_literal(code, "user_prompt")
        guidelines = assigned_literal(utils, "annotation_guidelines")
        # Execute only the reviewed pure JSON extraction function, not module imports.
        parser_node = next(n for n in ast.parse(utils.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "extract_json")
        parser_env = {"re": re}
        exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(utils), "exec"), parser_env)
        parse_answer = parser_env["extract_json"]

        def render(raw):
            steps = [json.dumps({k:v for k,v in step.items() if k in FIELDS}, indent=2) for step in raw["original_traj"]]
            test = next(iter(raw["failed_tests"].values()))
            user = template.format(problem_statement=raw["original_mission_prompt"],
                trajectories="<trajectory>\n"+"\n\n".join(steps)+"\n</trajectory>",
                test_code=test["code"], error_message=test["error_description"],
                annotation_guidelines=guidelines, ground_truth="\n")
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]

        for cid, c in cases.items():
            rawpath = Path(c["sources"]["raw"]["path"])
            record(rawpath)
            if hashes[str(rawpath)] != c["sources"]["raw"]["sha256"]:
                raise ValueError("Raw data drift")
            raw = read(rawpath)
            expected = render(raw)
            poisoned = copy.deepcopy(raw)
            for name in GOLD_KEYS:
                poisoned[name] = "AUDIT_FORBIDDEN_ANSWER_SENTINEL_"+name
            if render(poisoned) != expected:
                raise ValueError("Gold fields influence rendered input")
            baseline_suite = {"qwen38": "qwen38_thinking", "deepseek_official": "deepseek_official_thinking",
                              "moma_deepseek41": "moma_deepseek41"}[suite]
            case_dir = Path(baseline[baseline_suite+"/all_at_once"][cid]["state_path"]).parent
            if not case_dir.is_relative_to(directory):
                raise ValueError("Unexpected baseline directory")
            requests = sorted(case_dir.glob("calls/*/request.json"))
            if not requests:
                raise ValueError("Missing saved request")
            statepath = case_dir / "state.json";record(statepath);state = read(statepath)
            if state["trajectory_id"] != c["trajectory_id"]:
                raise ValueError("Baseline trajectory identity mismatch")
            record_data = {"suite": suite, "case_id": cid, "trajectory_id": c["trajectory_id"],
                "calls": len(requests), "status": state["status"], "prediction": state.get("failure_id"),
                "gold": labels[cid], "exact_safe_prompt_matches": 0, "request_protocol_tool_fields": [],
                "response_protocol_tool_fields": [], "sse_protocol_tool_fields": [], "sse_parse_errors": 0,
                "gold_reason_full_match": False, "gold_patch_full_match": False,
                "nested_gold_field_names_in_allowed_input": [], "reasoning_characters": 0,
                "final_characters": 0, "attempts": 0, "request_keys": [], "guidelines_characters": len(guidelines),
                "raw_response_matches": 0, "valid_prediction_verified_from_response": False, "correct": False}
            permitted = {"task": raw["original_mission_prompt"], "steps": [{k:v for k,v in s.items() if k in FIELDS} for s in raw["original_traj"]],
                         "test": {k:v for k,v in next(iter(raw["failed_tests"].values())).items() if k in {"code", "error_description"}}}
            record_data["nested_gold_field_names_in_allowed_input"] = keys_in(permitted, GOLD_KEYS)
            prompt = "\n".join(m["content"] for m in expected)
            for name, key in (("failure_reason", "gold_reason_full_match"), ("gold_patch", "gold_patch_full_match")):
                value = raw.get(name)
                record_data[key] = isinstance(value, str) and len(value.strip()) >= 40 and value.strip() in prompt
            for reqpath in requests:
                record(reqpath);request = read(reqpath)
                callstatepath = reqpath.parent / "state.json";record(callstatepath);callstate = read(callstatepath)
                actual_sha = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if actual_sha != callstate["request_sha256"]:
                    raise ValueError("Request recorded digest mismatch")
                if request["messages"] != expected:
                    raise ValueError(f"Saved prompt differs from allowed-input rendering: {suite}/{cid}")
                record_data["exact_safe_prompt_matches"] += 1
                record_data["request_keys"] = sorted(request)
                record_data["request_protocol_tool_fields"] += keys_in({k:v for k,v in request.items() if k != "messages"}, TOOL_KEYS)
                record_data["attempts"] += len(callstate["attempts"])
                if state["status"] == "valid" and reqpath == requests[-1]:
                    answer = read(reqpath.parent / callstate["response_file"])
                    parsed = json.loads(parse_answer(answer["choices"][0]["message"]["content"]))
                    if str(parsed["failure_id"]) != str(state["failure_id"]):
                        raise ValueError("Case prediction differs from actual model response")
                    record_data["valid_prediction_verified_from_response"] = True
                    record_data["correct"] = int(parsed["failure_id"]) in labels[cid]
                for responsepath in reqpath.parent.glob("response-*.json"):
                    record(responsepath);response = read(responsepath)
                    record_data["response_protocol_tool_fields"] += keys_in(response, TOOL_KEYS)
                    for choice in response.get("choices", []):
                        message = choice.get("message", {})
                        record_data["reasoning_characters"] += len(message.get("reasoning_content") or "")
                        record_data["final_characters"] += len(message.get("content") or "")
                    textpath = responsepath.with_suffix(".txt")
                    if textpath.exists():
                        record(textpath)
                        if read(textpath) != response:
                            raise ValueError("Raw HTTP response differs from normalized JSON")
                        record_data["raw_response_matches"] += 1
                for rawresponse in reqpath.parent.glob("response-*.sse"):
                    record(rawresponse)
                    content_fragments = []
                    with rawresponse.open() as stream:
                        for line in stream:
                            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                record_data["sse_parse_errors"] += 1
                                continue
                            record_data["sse_protocol_tool_fields"] += keys_in(chunk, TOOL_KEYS)
                            for choice in chunk.get("choices", []):
                                if choice.get("index", 0) == 0 and choice.get("delta", {}).get("content"):
                                    content_fragments.append(choice["delta"]["content"])
                    normalized = rawresponse.with_suffix(".json")
                    if normalized.exists():
                        message = read(normalized)["choices"][0]["message"]
                        if "".join(content_fragments) != (message.get("content") or ""):
                            raise ValueError("Raw SSE answer differs from normalized response")
                        record_data["raw_response_matches"] += 1
            rows.append(record_data)
        print(json.dumps({"completed_suite": suite}), flush=True)

    summaries = {}
    for suite in SUITES:
        subset = [r for r in rows if r["suite"] == suite]
        summaries[suite] = {"n": len(subset), "calls": sum(r["calls"] for r in subset),
            "attempts": sum(r["attempts"] for r in subset),
            "exact_safe_prompt_matches": sum(r["exact_safe_prompt_matches"] for r in subset),
            "raw_response_matches": sum(r["raw_response_matches"] for r in subset),
            "valid_predictions_verified_from_response": sum(r["valid_prediction_verified_from_response"] for r in subset),
            "exact_hits_from_response": sum(r["correct"] for r in subset),
            "request_protocol_tool_fields": sum(len(r["request_protocol_tool_fields"]) for r in subset),
            "response_protocol_tool_fields": sum(len(r["response_protocol_tool_fields"]) for r in subset),
            "sse_protocol_tool_fields": sum(len(r["sse_protocol_tool_fields"]) for r in subset),
            "nonempty_request_tool_fields": sum(v["nonempty"] for r in subset for v in r["request_protocol_tool_fields"]),
            "nonempty_response_tool_fields": sum(v["nonempty"] for r in subset for v in r["response_protocol_tool_fields"]),
            "nonempty_sse_tool_fields": sum(v["nonempty"] for r in subset for v in r["sse_protocol_tool_fields"]),
            "sse_parse_errors": sum(r["sse_parse_errors"] for r in subset),
            "gold_reason_full_matches": sum(r["gold_reason_full_match"] for r in subset),
            "gold_patch_full_matches": sum(r["gold_patch_full_match"] for r in subset),
            "nested_gold_field_names": sum(len(r["nested_gold_field_names_in_allowed_input"]) for r in subset),
            "statuses": dict(Counter(r["status"] for r in subset)),
            "reasoning_characters": sum(r["reasoning_characters"] for r in subset),
            "cases_with_reasoning": sum(r["reasoning_characters"] > 0 for r in subset)}
    qrows = []
    for cid, c in cases.items():
        nodepath = Path(c["sources"]["nodes"]["path"]);record(nodepath)
        featurepath = Path(c["sources"]["features"]["path"]);record(featurepath)
        nodes = read(nodepath);features = read(featurepath)
        root = str(labels[cid][0]);node = next((r for r in (nodes.get("rows") or []) if str(r["step_id"]) == root), None)
        b = baseline["qwen38_thinking/all_at_once"][cid]
        qrows.append({"case_id": cid, "qwen_aa_correct": b["correct"], "root": root,
            "q": features["q"][int(root)], "q_missing": node is None, "node": node,
            "q_status": nodes["status"]})
    qsummary = {}
    for group, subset in (("all", qrows), ("qwen_aa_correct", [r for r in qrows if r["qwen_aa_correct"]])):
        qsummary[group] = {"n": len(subset), "root_q_missing": sum(r["q_missing"] for r in subset),
            "root_valid_q_positive": sum(not r["q_missing"] and r["q"] > 0 for r in subset),
            "root_valid_q_zero": sum(not r["q_missing"] and r["q"] == 0 for r in subset),
            "valid_root_kinds": dict(Counter(r["node"]["kind"] for r in subset if r["node"])),
            "valid_root_recovered": sum(r["node"]["recovered"] for r in subset if r["node"])}
    for path, digest in hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError("Source modified")
    save(OUT / "request-audit.json", rows)
    save(OUT / "summary.json", summaries)
    save(OUT / "q-root-audit.json", qrows)
    save(OUT / "q-summary.json", qsummary)
    save(OUT / "source-hashes.json", hashes)
    save(OUT / "verification.json", {"files_unchanged": len(hashes), "new_api_calls": 0,
        "new_training_fits": 0, "script_sha256": sha(Path(__file__))})
    print(json.dumps({"request_audit": summaries, "q_audit": qsummary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
