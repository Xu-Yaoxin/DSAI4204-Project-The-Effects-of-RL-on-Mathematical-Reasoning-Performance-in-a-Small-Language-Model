"""Detect a long exact periodic suffix in generated token IDs, without using gold."""
import hashlib
import json
import math

if "TOKEN_IDS" not in globals() or "GUARD_POLICY" not in globals():
    raise SystemExit("Internal script: supply TOKEN_IDS and GUARD_POLICY.")
_policy = GUARD_POLICY
if not isinstance(_policy, dict) or _policy.get("version") != "exact-periodic-suffix-v1":
    raise ValueError("Unsupported repetition guard policy")
for _key in ("check_every_tokens", "min_generated_tokens", "min_repeat_span_tokens", "max_period_tokens", "min_repeats"):
    if type(_policy.get(_key)) is not int or _policy[_key] < 1:
        raise ValueError(f"Invalid guard policy: {_key}")
if (_policy["min_repeats"] < 4 or _policy["max_period_tokens"] > 1024
        or _policy["min_repeat_span_tokens"] > 65536
        or _policy["min_generated_tokens"] < _policy["min_repeat_span_tokens"]):
    raise ValueError("Invalid or unbounded repetition guard policy")
if not isinstance(TOKEN_IDS, list):
    raise ValueError("Guard requires generated token IDs as a list")
_count = globals().get("CHECK_AT_TOKENS", len(TOKEN_IDS))
if type(_count) is not int or not 0 <= _count <= len(TOKEN_IDS):
    raise ValueError("Invalid repetition check boundary")
GUARD_RESULT = {"triggered": False, "reason": None, "evidence": None}
if _count >= _policy["min_generated_tokens"] and _count % _policy["check_every_tokens"] == 0:
    _window = max(_policy["min_repeat_span_tokens"] + _policy["max_period_tokens"],
                  _policy["min_repeats"] * _policy["max_period_tokens"])
    if any(type(_token) is not int or _token < 0 for _token in TOKEN_IDS[max(0, _count-_window):_count]):
        raise ValueError("Invalid generated token ID")
    for _period in range(1, min(_policy["max_period_tokens"], _count // _policy["min_repeats"]) + 1):
        _repeats = max(_policy["min_repeats"], math.ceil(_policy["min_repeat_span_tokens"] / _period))
        _span = _period * _repeats
        if _span > _count:
            continue
        _pattern = TOKEN_IDS[_count-_period:_count]
        if TOKEN_IDS[_count-_span:_count] == _pattern * _repeats:
            GUARD_RESULT = {"triggered": True, "reason": "repetition_guard", "evidence": {
                "detector": _policy["version"], "detected_at_tokens": _count,
                "period_tokens": _period, "repeat_count": _repeats, "span_tokens": _span,
                "pattern_token_ids": _pattern,
                "pattern_sha256": hashlib.sha256(json.dumps(_pattern, separators=(",", ":")).encode()).hexdigest()}}
            break
