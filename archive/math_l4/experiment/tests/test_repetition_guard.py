"""Exercise token-based loop detection and guarded score contracts without a model."""
import ast
import copy
import json
from pathlib import Path
import random
import runpy

root = Path(__file__).resolve().parents[1]
policy = json.loads((root / "configs/math_l4.json").read_text())["evaluation"]["generation_guard"]
path = root / "scripts/repetition_guard.py"
code = compile(path.read_text(), str(path), "exec")
assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(ast.parse(path.read_text())))
checks = 1
for period in (1, 2, 7, 32, 128, 255, 256):
    span = period * max(4, (1024 + period - 1) // period)
    count = ((span + 31) // 32) * 32
    pattern = list(range(100, 100+period))
    ids = (pattern * ((count + period-1)//period))[:count]
    state = {"TOKEN_IDS": ids, "GUARD_POLICY": policy}
    exec(code, state)
    result = state["GUARD_RESULT"]
    assert result["triggered"], (period, count, result)
    e = result["evidence"]
    assert e["period_tokens"] == period and e["span_tokens"] >= 1024 and e["repeat_count"] >= 4
    assert e["detected_at_tokens"] == count
    checks += 1
rng = random.Random(42)
negative_inputs = [[], [8]*1023, list(range(20000)), [rng.randrange(200000) for _ in range(8192)],
    [token for index in range(2000) for token in (7, 8, index+1000, 9)],
    [7,8,9,10]*200, [7]*1025]
for ids in negative_inputs:
    state = {"TOKEN_IDS": ids, "GUARD_POLICY": policy}
    exec(code, state)
    assert not state["GUARD_RESULT"]["triggered"], len(ids)
    checks += 1
# Prefix checking must ignore later data, including a later correction or EOS.
state = {"TOKEN_IDS": [9]*1024 + list(range(5000)), "CHECK_AT_TOKENS": 1024, "GUARD_POLICY": policy}
exec(code, state)
assert state["GUARD_RESULT"]["evidence"]["detected_at_tokens"] == 1024
checks += 1
for key, value in (("version", "other"), ("check_every_tokens", True), ("min_repeats", 2), ("max_period_tokens", 100000), ("min_repeat_span_tokens", 100000)):
    mutated = dict(policy); mutated[key] = value
    try:
        exec(code, {"TOKEN_IDS": [7]*1024, "GUARD_POLICY": mutated})
    except ValueError:
        checks += 1
    else:
        raise AssertionError(key)

settings = {"generation_guard": policy, "eos_token_id": 248044, "max_prompt_length": 2048, "context_limit": 262144}
base = {"answer": "17", "text": "#### 17", "prompt_tokens": 5, "generation_limit": 262139,
    "output_token_ids": [11,12,248044], "output_tokens": 3, "seconds": 1.0,
    "termination_reason": "eos", "scoring_evaluable": True, "guard_evidence": None, "timeout_evidence": None}
score = runpy.run_path(str(root / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [base]})["SCORE_OUTPUT"][0]
base.update(score)
validator = root / "scripts/validate_guarded_sample.py"
assert runpy.run_path(str(validator), init_globals={"SAMPLE": base, "SETTINGS": settings})["VALIDATED"]
checks += 1
repeat = copy.deepcopy(base)
repeat.update(output_token_ids=[7]*1024, output_tokens=1024, seconds=30, termination_reason="repetition_guard",
    scoring_evaluable=False, correct=None, reward=None, extracted=None, format_compliant=None, status="unscored")
s = {"TOKEN_IDS": repeat["output_token_ids"], "GUARD_POLICY": policy};exec(code,s)
repeat["guard_evidence"] = s["GUARD_RESULT"]["evidence"]
timeout = copy.deepcopy(base)
timeout.update(output_token_ids=[11,12], output_tokens=2, seconds=301, termination_reason="answer_timeout", scoring_evaluable=False,
    correct=False, reward=0.0, extracted=None, format_compliant=None, status="timeout",
    timeout_evidence={"timeout_seconds":300, "elapsed_seconds":300, "last_token_age_seconds":1, "observed_tokens_at_timeout":2})
stall = copy.deepcopy(timeout)
stall.update(output_token_ids=[], output_tokens=0, seconds=121, termination_reason="no_token_timeout", correct=None, reward=None, status="unscored",
    timeout_evidence={"timeout_seconds":120, "elapsed_seconds":120, "last_token_age_seconds":120, "observed_tokens_at_timeout":0})
for sample in (repeat, timeout, stall):
    assert runpy.run_path(str(validator), init_globals={"SAMPLE": sample, "SETTINGS": settings})["VALIDATED"]
    checks += 1
for name in ("guard_false_score", "timeout_unknown", "timeout_wrong_reason", "timeout_too_soon", "fake_eos", "fake_guard", "late_guard", "token_count", "bad_evidence", "missing_guard", "fake_stall", "boolean_token"):
    sample = copy.deepcopy(repeat if name.startswith("guard") or name in ("fake_guard", "late_guard", "bad_evidence", "missing_guard") else stall if name=="fake_stall" else timeout if name.startswith("timeout") else base)
    if name == "guard_false_score": sample["correct"] = False
    elif name == "timeout_unknown": sample["correct"] = None
    elif name == "timeout_wrong_reason": sample["status"] = "incorrect"
    elif name == "timeout_too_soon": sample["timeout_evidence"]["elapsed_seconds"] = 299
    elif name == "fake_eos": sample["output_token_ids"][-1] = 13
    elif name == "fake_guard": sample["output_token_ids"][-1] = 8
    elif name == "late_guard": sample["guard_evidence"]["detected_at_tokens"] = 1056
    elif name == "token_count": sample["output_tokens"] += 1
    elif name == "bad_evidence": sample["guard_evidence"]["pattern_sha256"] = "0"*64
    elif name == "missing_guard": sample["termination_reason"] = "eos"; sample["scoring_evaluable"] = True; sample["guard_evidence"] = None; sample["output_token_ids"].append(248044); sample["output_tokens"] += 1; sample.update(score)
    elif name == "fake_stall": sample["timeout_evidence"]["last_token_age_seconds"] = 119
    elif name == "boolean_token": sample["output_token_ids"][0] = True
    try:
        runpy.run_path(str(validator), init_globals={"SAMPLE": sample, "SETTINGS": settings})
    except ValueError:
        checks += 1
    else:
        raise AssertionError(name)
print(f"✅ [REPETITION GUARD] {checks} token-period, negative-control, deadline-zero-score and unknown-stop contract checks; no model/GPU/network")
