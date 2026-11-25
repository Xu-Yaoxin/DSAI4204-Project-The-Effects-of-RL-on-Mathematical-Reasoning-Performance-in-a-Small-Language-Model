"""Offline exact-answer regressions; no models, packages, GPU, or network needed."""

import copy
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


code_root = Path(__file__).resolve().parents[1]
verifier_path = code_root / "scripts/verify_answers.py"

# Verbatim archived completion: run_20261031/test_base, id gsm8k-test-001010,
# sample_index 0. The fixture is embedded so this suite never needs the archive.
archived_text = (
    "1. Bethany can run 10 laps.\n"
    "2. Trey can run 10 + 4 = 14 laps.\n"
    "3. Shaelyn can run 14 / 2 = 7 laps.\n"
    "4. Quinn can run 7 - 2 = 5 laps.\n"
    "5. Bethany can run 10 - 5 = 5 more laps than Quinn.\n\n"
    "Answer: #### 5"
)

# Expected values are hand-written, including contrasting wrong answers in
# the same notation. A successful extraction need not be a correct answer.
cases = [
    ("archived_nested_marker", archived_text, "5", "correct", "5", False),
    ("archived_numeric_contrast", archived_text.replace("Answer: #### 5", "Answer: #### 6"), "5", "incorrect", "6", False),
    ("nested_final_marker", "Final answer: #### 17", "17", "correct", "17", False),
    ("nested_final_wrong", "Final answer: #### 18", "17", "incorrect", "18", False),
    ("is_colon", "The final answer is: 17", "17", "correct", "17", False),
    ("is_colon_wrong", "The final answer is: 18", "17", "incorrect", "18", False),
    ("is_equals", "The answer is = 17", "17", "correct", "17", False),
    ("bold_label_colon_inside", "**Answer:** 17", "17", "correct", "17", False),
    ("bold_label_colon_outside", "**Final answer**:17", "17", "correct", "17", False),
    ("bold_label_wrong_inside", "**Answer:** 18", "17", "incorrect", "18", False),
    ("bold_label_wrong_outside", "**Final answer**:18", "17", "incorrect", "18", False),
    ("bold_label_conflict_inside", "#### 17\n**Answer:** 18", "17", "ambiguous", None, False),
    ("bold_label_conflict_outside", "#### 17\n**Final answer**:18", "17", "ambiguous", None, False),
    ("bold_label_unclosed", "**Answer: 17", "17", "extraction_failed", None, False),
    ("bold_label_unopened", "Answer**: 17", "17", "extraction_failed", None, False),
    ("bold_label_sentence_start", "The calculation is complete. **Answer:** 17", "17", "correct", "17", False),
    ("bold_label_with_valid_hash", "**Answer:** 17\n#### 17", "17", "correct", "17", True),
    ("is_colon_currency", "Thus, the answer is: $1,250.00$.", "1250", "correct", "1250", False),
    ("is_colon_fraction", r"Answer is: \frac{3}{4}", ".75", "correct", "3/4", False),
    ("nested_box_marker", r"Answer: #### \boxed{17}", "17", "correct", "17", False),
    ("nested_box_marker_wrong", r"Answer: #### \boxed{18}", "17", "incorrect", "18", False),
    ("nested_fraction", r"Final answer: #### \boxed{\frac{3}{4}}", ".75", "correct", "3/4", False),
    ("nested_negative_fraction", r"Answer: #### \boxed{-\frac{3}{4}}", "-.75", "correct", "-3/4", False),
    ("marker_box", r"#### \boxed{17}", "17", "correct", "17", True),
    ("explicit_box", r"The answer is: \boxed{17}.", "17", "correct", "17", False),
    ("same_line_repeated_markers", "Answer: #### 17 #### 17", "17", "correct", "17", False),
    ("negative_zero", "Answer: #### -0.000", "0", "correct", "0", False),
    ("exact_large_integer", "Answer: #### 9007199254740993", "9007199254740993", "correct", "9007199254740993", False),
    ("no_float_rounding", "Answer: #### 9007199254740992", "9007199254740993", "incorrect", "9007199254740992", False),
    ("scientific_fraction", "Answer: #### 1.5e2/2.5e1", "6", "correct", "6", False),
    ("decimal_exactness", "Answer: #### 0.30000000000000004", ".3", "incorrect", "7500000000000001/25000000000000000", False),
    ("fullwidth_number", "Answer: #### １７．００", "17", "correct", "17", False),
    ("unicode_minus", "Answer: #### −１７", "-17", "correct", "-17", False),
    ("crlf", "We subtract 3 from 20.\r\nAnswer is: 17\r\n#### 17\r\n", "17", "correct", "17", True),
    ("tabs", "\tAnswer:\t####\t17\t", "17", "correct", "17", False),
    ("formatted_wrong", "The final answer is: 18.\n#### 18", "17", "incorrect", "18", True),
    ("xml_scalar", "<answer>17</answer>", "17", "correct", "17", False),
    ("xml_wrong", "<answer>18</answer>", "17", "incorrect", "18", False),
    ("xml_case_and_space", "<AnSwEr> 3/4 </aNsWeR>", ".75", "correct", "3/4", False),
    ("xml_empty", "<answer></answer>", "17", "extraction_failed", None, False),
    ("xml_unclosed", "<answer>17\n#### 17", "17", "extraction_failed", None, False),
    ("xml_mismatched", "<answer>17</result>", "17", "extraction_failed", None, False),
    ("xml_attributes", '<answer confidence="1">17</answer>', "17", "extraction_failed", None, False),
    ("xml_contradiction", "<answer>17</answer>\n#### 18", "17", "ambiguous", None, False),
    ("xml_units", "<answer>17 apples</answer>", "17", "extraction_failed", None, False),
    ("xml_nested_hash", "#### <answer>17</answer>", "17", "correct", "17", True),
    ("xml_agreement", "<answer>17</answer>\n<answer>17.00</answer>", "17", "correct", "17", False),
    ("xml_disagreement", "<answer>17</answer>\n<answer>18</answer>", "17", "ambiguous", None, False),
    ("xml_empty_after_valid", "<answer>17</answer>\n<answer></answer>", "17", "extraction_failed", None, False),
    ("xml_nested_tags", "<answer><answer>17</answer></answer>", "17", "extraction_failed", None, False),
    ("xml_extra_closing_tag", "<answer>17</answer>\n</answer>", "17", "extraction_failed", None, False),
    ("heading_numeric", "Final Answer:\n17", "17", "correct", "17", False),
    ("heading_marker", "Final Answer:\n\n#### 17", "17", "correct", "17", True),
    ("heading_box", "Answer:\n" + r"\boxed{17}", "17", "correct", "17", False),
    ("heading_trailing", "#### 17\nAnswer:", "17", "extraction_failed", None, False),
    ("heading_not_a_scalar", "Answer:\nI obtained 17 apples.\n#### 17", "17", "extraction_failed", None, False),
    ("heading_explicit", "Answer:\nFinal answer: 17", "17", "correct", "17", False),
    ("heading_explicit_wrong", "Answer:\nFinal answer: 18", "17", "incorrect", "18", False),
    ("heading_chain", "Answer:\nFinal answer:\n#### 17", "17", "correct", "17", True),
    ("heading_chain_contradiction", "Answer:\nFinal answer: 18\n#### 17", "17", "ambiguous", None, False),
    ("heading_chain_empty", "#### 17\nAnswer:\nFinal answer:", "17", "extraction_failed", None, False),
    ("units_nested", "Answer: #### 17 apples", "17", "extraction_failed", None, False),
    ("units_is_colon", "The answer is: 17 dollars.", "17", "extraction_failed", None, False),
    ("expression", "Answer: #### 20 - 3", "17", "extraction_failed", None, False),
    ("equation", "Answer is: 20 - 3 = 17", "17", "extraction_failed", None, False),
    ("percentage", "Answer: #### 17%", "17", "extraction_failed", None, False),
    ("nan", "Answer: #### NaN", "0", "extraction_failed", None, False),
    ("infinity", "Answer: #### Infinity", "0", "extraction_failed", None, False),
    ("unicode_infinity", "Answer: #### ∞", "0", "extraction_failed", None, False),
    ("zero_denominator", r"Answer: #### \boxed{\frac{17}{0}}", "17", "extraction_failed", None, False),
    ("malformed_grouping", "Answer: #### 1,70", "170", "extraction_failed", None, False),
    ("empty_hash_marker", "Answer: ####", "17", "extraction_failed", None, False),
    ("empty_box", r"Answer: #### \boxed{}", "17", "extraction_failed", None, False),
    ("unclosed_box", r"Answer: #### \boxed{17", "17", "extraction_failed", None, False),
    ("empty_marker_after_valid", "Answer: #### 17\n####", "17", "extraction_failed", None, False),
    ("malformed_marker_after_valid", "Answer: #### 17\n#### unknown", "17", "extraction_failed", None, False),
    ("valid_does_not_rescue_units", "Answer: #### 17 apples\n#### 17", "17", "extraction_failed", None, False),
    ("unsupported_set", "Answer: #### {17, 18}", "17", "extraction_failed", None, False),
    ("unsupported_range", "Answer: #### 17–18", "17", "extraction_failed", None, False),
    ("no_arbitrary_last_number", "I considered 18 but computed 17 apples.", "17", "extraction_failed", None, False),
    ("conditional_is_not_final", "If the answer is: 17, then we are done.", "17", "extraction_failed", None, False),
    ("code_not_executed", "Answer: #### __import__('os').system('false')", "0", "extraction_failed", None, False),
    ("bounded_exponent", "Answer: #### 1e999999", "1", "extraction_failed", None, False),
    ("contradictory_hash", "Answer: #### 17\n#### 18", "17", "ambiguous", None, False),
    ("contradictory_explicit", "Answer: #### 17\nThe answer is: 18", "17", "ambiguous", None, False),
    ("contradictory_box", "Answer: #### 17\n" + r"\boxed{18}", "17", "ambiguous", None, False),
    ("contradictory_bare", "Answer: #### 17\n18", "17", "ambiguous", None, False),
    ("contradictory_same_line", "Answer: #### 17 #### 18", "17", "ambiguous", None, False),
    ("contradictory_fraction", "Answer: #### .75\nFinal answer: 3/5", ".75", "ambiguous", None, False),
    ("same_number_multiple_formats", "Answer: #### .75\nThe answer is: 3/4\n" + r"\boxed{\frac{6}{8}}", ".75", "correct", "3/4", False),
    ("invalid_gold_expression", "Answer: #### 17", "20 - 3", "invalid_gold", None, False),
    ("invalid_gold_nonfinite", "Answer: #### 17", "NaN", "invalid_gold", None, False),
]

records = [{"text": text, "answer": gold} for _, text, gold, _, _, _ in cases]
records_before = copy.deepcopy(records)
scores = runpy.run_path(str(verifier_path), init_globals={"SCORE_INPUT": records})["SCORE_OUTPUT"]
assert records == records_before, "Scoring must not mutate source records"
assert len(scores) == len(cases)
score_fields = {"correct", "reward", "extracted", "status", "format_compliant", "truncated"}
for (label, text, gold, status, extracted, compliant), score in zip(cases, scores):
    assert set(score) == score_fields, (label, score)
    assert score["status"] == status, (label, text, gold, score, status)
    assert score["extracted"] == extracted, (label, score, extracted)
    assert score["format_compliant"] is compliant, (label, score, compliant)
    assert score["correct"] is (status == "correct"), (label, score)
    assert type(score["reward"]) is float and score["reward"] == float(status == "correct"), (label, score)
    assert score["truncated"] is False, (label, score)

# Gold must only determine correctness, never which predicted number is chosen.
gold_independent_texts = [
    ("Answer: #### 17", "17"),
    ("The answer is: 17", "17"),
    ("Answer: #### 17\n#### 18", None),
    ("Answer: #### 17 apples\n#### 17", None),
]
gold_cases = [{"text": text, "answer": gold} for text, _ in gold_independent_texts for gold in ["17", "18", "999"]]
gold_scores = runpy.run_path(str(verifier_path), init_globals={"SCORE_INPUT": gold_cases})["SCORE_OUTPUT"]
for offset, (text, expected_extracted) in enumerate(gold_independent_texts):
    group = gold_scores[offset * 3:offset * 3 + 3]
    assert all(score["extracted"] == expected_extracted for score in group), (text, group)
    assert len({score["format_compliant"] for score in group}) == 1, (text, group)
    if expected_extracted is not None:
        assert [score["status"] for score in group] == ["correct", "incorrect", "incorrect"], (text, group)
    else:
        assert len({score["status"] for score in group}) == 1, (text, group)
        assert all(score["reward"] == 0.0 for score in group), (text, group)

# Repetition, equivalent representations and whitespace must not change value.
equivalent_texts = [
    "Answer: #### .75",
    "Answer: #### .75\nAnswer: #### 3/4",
    "Answer: #### .75\nFinal answer is: 6/8\n" + r"\boxed{0.750}",
    "\r\n\tAnswer:\t####\t.750\t\r\n",
    "Answer: #### .75\n#### 3/4\n#### 6/8\n#### .7500",
]
equivalent_scores = runpy.run_path(str(verifier_path), init_globals={"SCORE_INPUT": [{"text": text, "answer": ".75"} for text in equivalent_texts]})["SCORE_OUTPUT"]
assert all(score["status"] == "correct" and score["extracted"] == "3/4" for score in equivalent_scores), equivalent_scores

flag_records = [
    {"text": "Answer: #### 17", "answer": "17", "truncated": True},
    {"text": "Answer: ####", "answer": "17", "truncated": True},
    {"text": "Answer: #### 17", "answer": "17", "infrastructure_error": "generation failed", "truncated": True},
]
flag_scores = runpy.run_path(str(verifier_path), init_globals={"SCORE_INPUT": flag_records})["SCORE_OUTPUT"]
assert [score["status"] for score in flag_scores] == ["correct", "extraction_failed", "infrastructure_error"], flag_scores
assert all(score["truncated"] is True for score in flag_scores), flag_scores
assert flag_scores[2]["extracted"] is None and flag_scores[2]["reward"] == 0.0, flag_scores

# -S excludes site-packages. Literal Unicode line separators inside JSON strings
# must not split JSONL records, and source payloads must survive rescoring intact.
with tempfile.TemporaryDirectory(prefix="dsai4207-verifier-regressions-") as temporary_directory:
    temporary_root = Path(temporary_directory)
    input_path = temporary_root / "input.jsonl"
    cli_records = [
        {"id": "unicode-line", "text": "Calculation\u2028continues.\nAnswer: #### 17", "answer": "17", "question": "First\u0085second", "sample_index": 0},
        {"id": "crlf", "text": "A calculation.\r\nThe answer is: 18\r\n", "answer": "17", "sample_index": 1},
        {"id": "conflict", "text": "Answer: #### 17\nFinal answer is: 18", "answer": "18", "sample_index": 2},
    ]
    input_bytes = ("\n" + "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in cli_records)).encode("utf-8")
    input_path.write_bytes(input_bytes)
    completed = subprocess.run([sys.executable, "-S", str(verifier_path), "--input", "input.jsonl", "--output", "nested/scores.jsonl"], cwd=temporary_root, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    cli_scores = [json.loads(line) for line in (temporary_root / "nested/scores.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
    assert [score["status"] for score in cli_scores] == ["correct", "incorrect", "ambiguous"], cli_scores
    assert input_path.read_bytes() == input_bytes, "CLI must not overwrite its input"
    for record, score in zip(cli_records, cli_scores):
        assert all(score[key] == value for key, value in record.items()), (record, score)
    stdout_completed = subprocess.run([sys.executable, "-S", str(verifier_path), "--input", "input.jsonl"], cwd=temporary_root, capture_output=True, text=True, timeout=15)
    assert stdout_completed.returncode == 0, stdout_completed.stderr
    stdout_scores = [json.loads(line) for line in stdout_completed.stdout.split("\n") if line.strip()]
    assert stdout_scores == cli_scores, (stdout_scores, cli_scores)

scenario_count = len(cases) + len(gold_cases) + len(equivalent_texts) + len(flag_records) + 2 * len(cli_records)
print(f"[PASS] {scenario_count} verifier regression scenarios | archived fixture, exact arithmetic, gold independence, malformed answers and stdlib-only CLI")
