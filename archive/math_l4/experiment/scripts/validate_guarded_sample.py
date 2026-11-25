"""Replay one guarded generation record and its scoring/termination contract."""
import math
from pathlib import Path

if "SAMPLE" not in globals() or "SETTINGS" not in globals():
    raise SystemExit("Internal script: supply SAMPLE and SETTINGS.")
_row, _settings = SAMPLE, SETTINGS
_script_root = Path(__file__).resolve().parent
if not isinstance(_row, dict) or not isinstance(_settings, dict):
    raise ValueError("Invalid guarded sample/settings")
_policy = _settings.get("generation_guard")
if not isinstance(_policy, dict):
    raise ValueError("Missing declared generation guard")
for _field in ("heartbeat_seconds", "no_token_timeout_seconds", "answer_timeout_seconds", "stop_grace_seconds"):
    if type(_policy.get(_field)) not in (int, float) or not math.isfinite(_policy[_field]) or _policy[_field] <= 0:
        raise ValueError(f"Invalid guard timer: {_field}")
if type(_settings.get("eos_token_id")) is not int or _settings["eos_token_id"] < 0:
    raise ValueError("Missing fixed EOS token ID")
if (type(_row.get("prompt_tokens")) is not int or not 1 <= _row["prompt_tokens"] <= _settings["max_prompt_length"]
        or type(_row.get("generation_limit")) is not int
        or _row["generation_limit"] != _settings["context_limit"] - _row["prompt_tokens"]):
    raise ValueError("Invalid per-answer native context budget")
_ids = _row.get("output_token_ids")
if (not isinstance(_ids, list) or any(type(_token) is not int or _token < 0 for _token in _ids)
        or type(_row.get("output_tokens")) is not int or _row["output_tokens"] != len(_ids)
        or not 0 <= len(_ids) <= _row["generation_limit"]):
    raise ValueError("Invalid generated token trace or count")
if (not isinstance(_row.get("text"), str) or type(_row.get("seconds")) not in (int, float)
        or not math.isfinite(_row["seconds"]) or _row["seconds"] < 0
        or type(_row.get("truncated")) is not bool):
    raise ValueError("Invalid output text, duration or truncation flag")
_reason = _row.get("termination_reason")
if _reason not in {"eos", "repetition_guard", "context_limit", "no_token_timeout", "answer_timeout"}:
    raise ValueError("Unknown termination reason")
if type(_row.get("scoring_evaluable")) is not bool or _row["scoring_evaluable"] != (_reason == "eos"):
    raise ValueError("Only natural-EOS answers are mathematically scored")
if _settings["eos_token_id"] in _ids[:-1]:
    raise ValueError("Token trace continued after EOS")
if not _ids and _reason not in {"no_token_timeout", "answer_timeout"}:
    raise ValueError("Empty generation cannot have this termination reason")
if _row["truncated"] != (_reason == "context_limit"):
    raise ValueError("Truncation must identify native context exhaustion only")
if _reason == "context_limit" and (len(_ids) != _row["generation_limit"] or _ids[-1] == _settings["eos_token_id"]):
    raise ValueError("Context exhaustion contradicts the token trace")
if _reason == "eos" and (not _ids or _ids[-1] != _settings["eos_token_id"]):
    raise ValueError("Natural completion must end with the fixed EOS token")
_timeout = _row.get("timeout_evidence")
if _reason in {"no_token_timeout", "answer_timeout"}:
    _timer = "no_token_timeout_seconds" if _reason == "no_token_timeout" else "answer_timeout_seconds"
    if (not isinstance(_timeout, dict) or set(_timeout) != {"timeout_seconds", "elapsed_seconds", "last_token_age_seconds", "observed_tokens_at_timeout"}
            or any(type(_timeout.get(_field)) not in (int, float) or not math.isfinite(_timeout[_field]) or _timeout[_field] < 0 for _field in ("timeout_seconds", "elapsed_seconds", "last_token_age_seconds"))
            or type(_timeout.get("observed_tokens_at_timeout")) is not int
            or not 0 <= _timeout["observed_tokens_at_timeout"] <= len(_ids)
            or _timeout["last_token_age_seconds"] > _timeout["elapsed_seconds"]
            or _timeout["timeout_seconds"] != _policy[_timer]
            or _timeout["elapsed_seconds"] < _policy[_timer]
            or _row["seconds"] < _timeout["elapsed_seconds"]
            or (_reason == "no_token_timeout" and _timeout["last_token_age_seconds"] < _policy[_timer])):
        raise ValueError("Timeout evidence contradicts the declared deadline")
elif _timeout is not None:
    raise ValueError("Unexpected timeout evidence")

_guard_scan_tokens = _timeout["observed_tokens_at_timeout"] if _reason in {"no_token_timeout", "answer_timeout"} else len(_ids)
_guard_code = compile((_script_root / "repetition_guard.py").read_text(), str(_script_root / "repetition_guard.py"), "exec")
_probe = {"TOKEN_IDS": [], "GUARD_POLICY": _policy}
exec(_guard_code, _probe)
_evidence = _row.get("guard_evidence")
_first_guard = None
for _boundary in range(_policy["check_every_tokens"], _guard_scan_tokens+1, _policy["check_every_tokens"]):
    if _boundary < _policy["min_generated_tokens"]:
        continue
    if _ids[_boundary-1] == _settings["eos_token_id"]:
        break
    _probe = {"TOKEN_IDS": _ids, "GUARD_POLICY": _policy, "CHECK_AT_TOKENS": _boundary}
    exec(_guard_code, _probe)
    if _probe["GUARD_RESULT"]["triggered"]:
        _first_guard = _probe["GUARD_RESULT"]["evidence"]
        break
if _reason == "repetition_guard":
    if not isinstance(_evidence, dict) or _evidence != _first_guard:
        raise ValueError("Repetition stop evidence differs from the earliest fixed detector trigger")
elif _evidence is not None:
    raise ValueError("Unexpected repetition evidence")
elif _first_guard is not None:
    raise ValueError("Token trace crossed a repetition trigger without recording it")
if _reason == "eos":
    if any(type(_row.get(_field)) is not bool for _field in ("correct", "format_compliant")) or type(_row.get("reward")) not in (int, float):
        raise ValueError("Invalid natural-completion scoring types")
    _score_namespace = {"SCORE_INPUT": [_row]}
    exec(compile((_script_root / "verify_answers.py").read_text(), str(_script_root / "verify_answers.py"), "exec"), _score_namespace)
    _score = _score_namespace["SCORE_OUTPUT"][0]
    if _score["status"] not in {"correct", "incorrect", "extraction_failed", "ambiguous"} or any(_row.get(_field) != _score[_field] for _field in ("correct", "reward", "extracted", "status", "format_compliant", "truncated")):
        raise ValueError("Completed answer score differs from fixed-verifier replay")
elif _reason == "answer_timeout":
    if (_row.get("correct") is not False or type(_row.get("reward")) not in (int, float) or _row["reward"] != 0
            or _row.get("extracted") is not None or _row.get("format_compliant") is not None or _row.get("status") != "timeout"):
        raise ValueError("The user-defined answer timeout must be recorded as a zero-score failure")
else:
    if any(_row.get(_field) is not None for _field in ("correct", "reward", "extracted", "format_compliant")) or _row.get("status") != "unscored":
        raise ValueError("Other interrupted answers must remain unscored")
VALIDATED = True
