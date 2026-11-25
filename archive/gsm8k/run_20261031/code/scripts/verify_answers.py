"""Conservative exact numerical scoring; CLI or runpy with SCORE_INPUT."""

import argparse
import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path


_score_cli = "SCORE_INPUT" not in globals()
if _score_cli:
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--input", required=True, help="JSONL with text and answer fields")
    _parser.add_argument("--output", help="Output JSONL; stdout when omitted")
    _args = _parser.parse_args()
    SCORE_INPUT = [json.loads(_line) for _line in Path(_args.input).read_text(encoding="utf-8").split("\n") if _line.strip()]

_scalar_pattern = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
SCORE_OUTPUT = []

for _record in SCORE_INPUT:
    _truncated = bool(_record.get("truncated", False))
    _result = {"correct": False, "reward": 0.0, "extracted": None,
               "status": "extraction_failed", "format_compliant": False,
               "truncated": _truncated}
    if _record.get("infrastructure_error"):
        _result["status"] = "infrastructure_error"
        SCORE_OUTPUT.append(_result)
        continue

    _text = str(_record.get("text", ""))
    _candidates = []
    _marked = False
    _malformed_box = False
    _format_candidates = re.findall(r"^[ \t]*####[ \t]+([^\n]+)$", _text, flags=re.MULTILINE)

    for _match in re.finditer(r"(?<!#)####(?!#)[ \t]*([^\n]*?)(?=(?<!#)####(?!#)|\n|\Z)", _text):
        _marked = True
        _candidates.append(_match.group(1).strip())

    for _match in re.finditer(r"\\boxed\s*\{", _text):
        _marked = True
        _depth = 1
        _position = _match.end()
        _start = _position
        while _position < len(_text) and _depth:
            if _text[_position] == "{":
                _depth += 1
            elif _text[_position] == "}":
                _depth -= 1
            _position += 1
        if _depth:
            _malformed_box = True
        else:
            _candidates.append(_text[_start:_position - 1].strip())

    _explicit_pattern = r"(?:^|(?<=[.!?])\s+)[ \t]*(?:(?:therefore|thus|hence|so)[,:]?[ \t]+)?(?:the[ \t]+)?(?:final[ \t]+answer|answer)[ \t]*(?::|=|\bis\b)[ \t]*([^\n]+)"
    for _match in re.finditer(_explicit_pattern, _text, flags=re.IGNORECASE | re.MULTILINE):
        _marked = True
        _candidates.append(_match.group(1).strip())

    _lines = [_line.strip() for _line in _text.splitlines() if _line.strip()]
    if _lines and not _marked:
        _candidates.append(_lines[-1])
    elif _lines and re.fullmatch(_scalar_pattern + r"(?:\s*/\s*" + _scalar_pattern + r")?\.?", _lines[-1]):
        _candidates.append(_lines[-1])

    _values = []
    _invalid_candidate = _malformed_box
    _gold_value = None
    _format_values = []
    for _candidate_kind, _candidate in [("gold", str(_record.get("answer", "")))] + [("prediction", _candidate) for _candidate in _candidates] + [("format", _candidate) for _candidate in _format_candidates]:
        _candidate = unicodedata.normalize("NFKC", _candidate).replace("−", "-").strip()
        _value = None
        if len(_candidate) <= 512:
            if _candidate.endswith("。"):
                _candidate = _candidate[:-1].rstrip()
            if _candidate.endswith("."):
                _candidate = _candidate[:-1].rstrip()
            _previous = None
            while _candidate != _previous:
                _previous = _candidate
                for _left, _right in [("**", "**"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
                    if _candidate.startswith(_left) and _candidate.endswith(_right) and len(_candidate) > len(_left) + len(_right):
                        _candidate = _candidate[len(_left):-len(_right)].strip()
                _box = re.fullmatch(r"\\boxed\s*\{(.*)\}", _candidate, flags=re.DOTALL)
                if _box:
                    _candidate = _box.group(1).strip()
            _candidate = _candidate.replace(r"\left", "").replace(r"\right", "")
            for _spacing in [r"\,", r"\!", r"\ "]:
                _candidate = _candidate.replace(_spacing, "")
            _currency = re.fullmatch(r"([+-]?)\s*[$£€]\s*(.+)", _candidate)
            if _currency:
                _candidate = _currency.group(1) + _currency.group(2)
            _parts = []
            _outer_sign = 1
            _latex_fraction = re.fullmatch(r"([+-]?)\s*\\(?:frac|dfrac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", _candidate)
            if _latex_fraction:
                _outer_sign = -1 if _latex_fraction.group(1) == "-" else 1
                _parts = [_latex_fraction.group(2).strip(), _latex_fraction.group(3).strip()]
            elif _candidate.count("/") == 1:
                _parts = [_part.strip() for _part in _candidate.split("/")]
            else:
                _parts = [_candidate]
            _parsed_parts = []
            for _part in _parts:
                _exponent = re.search(r"[eE]([+-]?[0-9]+)$", _part)
                if re.fullmatch(_scalar_pattern, _part) and (not _exponent or abs(int(_exponent.group(1))) <= 1000):
                    try:
                        _parsed_parts.append(Fraction(_part.replace(",", "")))
                    except (ValueError, ZeroDivisionError, OverflowError):
                        break
                else:
                    break
            if len(_parsed_parts) == len(_parts):
                if len(_parsed_parts) == 1:
                    _value = _parsed_parts[0]
                elif _parsed_parts[1] != 0:
                    _value = _outer_sign * _parsed_parts[0] / _parsed_parts[1]
        if _candidate_kind == "gold":
            _gold_value = _value
        elif _candidate_kind == "format":
            if _value is not None:
                _format_values.append(_value)
        elif _value is None:
            _invalid_candidate = True
        else:
            _values.append(_value)

    _unique_values = set(_values)
    if _gold_value is None:
        _result["status"] = "invalid_gold"
    elif len(_unique_values) > 1:
        _result["status"] = "ambiguous"
    elif _invalid_candidate or not _unique_values:
        _result["status"] = "extraction_failed"
    else:
        _answer_value = _values[0]
        _result["extracted"] = str(_answer_value.numerator) if _answer_value.denominator == 1 else f"{_answer_value.numerator}/{_answer_value.denominator}"
        _result["correct"] = _answer_value == _gold_value
        _result["reward"] = float(_result["correct"])
        _result["status"] = "correct" if _result["correct"] else "incorrect"
        _result["format_compliant"] = bool(_format_values) and all(_value == _answer_value for _value in _format_values)
    SCORE_OUTPUT.append(_result)

if _score_cli:
    _output_text = "".join(json.dumps({**_record, **_score}, ensure_ascii=False, sort_keys=True) + "\n" for _record, _score in zip(SCORE_INPUT, SCORE_OUTPUT))
    if _args.output:
        Path(_args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(_args.output).write_text(_output_text, encoding="utf-8")
    else:
        print(_output_text, end="")
