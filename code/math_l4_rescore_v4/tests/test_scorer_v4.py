"""Offline v4 regressions with literal expectations and frozen v3 coverage."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scorer", type=Path, default=root / "scorer.py")
parser.add_argument("--skip-legacy", action="store_true")
parser.add_argument("--receipt", type=Path)
args = parser.parse_args()
verifier = args.scorer.resolve()

# Literal expected scalars are independent of production extraction logic.
cases = [
    ("term_heading", "#### Term 1: $2+3$\nWorking.\n#### 17", "17", "correct", "17", True),
    ("term_wrong_final", "#### Term 1: $2+3$\n#### 18", "17", "incorrect", "18", True),
    ("part_heading", "#### Part A: $x+1$\n#### 17", "17", "correct", "17", True),
    ("equation_heading", "#### Equation (2): $x+1=4$\n#### 17", "17", "correct", "17", True),
    ("ordinal_heading", "#### First equation: $x+1=4$\n#### 17", "17", "correct", "17", True),
    ("bold_heading", "#### **Third Term: $2+3$**\n#### 17", "17", "correct", "17", True),
    ("step_heading", "#### Step 2: Simplify the expression\n#### 17", "17", "correct", "17", True),
    ("case_heading", "#### Case 1: All sides equal\n#### 17", "17", "correct", "17", True),
    ("heading_without_final", "#### Term 17: $17$\nWorking is incomplete.", "17", "extraction_failed", None, False),
    ("heading_only", "#### Term 17: $17$", "17", "extraction_failed", None, False),
    ("unknown_hash", "#### unknown\n#### 17", "17", "extraction_failed", None, False),
    ("arbitrary_heading", "#### Maybe seventeen\n#### 17", "17", "extraction_failed", None, False),
    ("heading_unknown_claim", "#### Step 1: unknown\n#### 17", "17", "extraction_failed", None, False),
    ("heading_na_claim", "#### Step 1: N/A\n#### 17", "17", "extraction_failed", None, False),
    ("heading_uncertain_claim", "#### Term 1: undetermined\n#### 17", "17", "extraction_failed", None, False),
    ("heading_final_number_claim", "#### Step 1: Final 18\n#### 17", "17", "extraction_failed", None, False),
    ("heading_question_claim", "#### Step 1: ?\n#### 17", "17", "extraction_failed", None, False),
    ("heading_final_claim", "#### Step 1: Answer is 18\n#### 17", "17", "extraction_failed", None, False),
    ("heading_negation", "#### Step 1: The answer is not 17\n#### 17", "17", "extraction_failed", None, False),
    ("inline_heading_not_markdown", "Text #### Step 1: simplify\n#### 17", "17", "extraction_failed", None, False),
    ("hash_conflict_preserved", "#### Term 1: $2+3$\n#### 18\n#### 17", "17", "ambiguous", None, False),
    ("hash_answer_heading", "#### Final Answer:\n17", "17", "correct", "17", False),
    ("hash_answer_inline", "#### Final Answer: 17", "17", "correct", "17", False),
    ("hash_answer_wrong", "#### Final Answer: 18", "17", "incorrect", "18", False),
    ("hash_answer_conflict", "#### Final Answer: 18\n#### 17", "17", "ambiguous", None, False),
    ("hash_answer_unknown", "#### Final Answer: unknown\n#### 17", "17", "extraction_failed", None, False),
    ("hash_answer_empty", "#### 17\n#### Final Answer:", "17", "extraction_failed", None, False),
    ("hash_answer_display", "#### Final Answer:\n$$\n#### 17\n$$", "17", "correct", "17", True),
    ("hash_answer_display_wrong", "#### Final Answer:\n$$\n#### 18\n$$", "17", "incorrect", "18", True),
    ("hash_answer_display_unclosed", "#### Final Answer:\n$$\n#### 17", "17", "extraction_failed", None, False),
    ("hash_answer_display_prose", "#### Final Answer:\n$$\nI computed #### 17\n$$", "17", "extraction_failed", None, False),
    ("prose_area", "Final Answer: The maximum area of the triangle is 2500 square units.", "2500", "correct", "2500", False),
    ("prose_next_line", "**Final Answer:**\nThe maximum area of the triangle is 2500 square units.\n#### 2500", "2500", "correct", "2500", True),
    ("prose_minimum", "Final Answer: The minimum length is 17 meters.", "17", "correct", "17", False),
    ("prose_fraction", r"Final Answer: The probability is $\frac{3}{4}$.", ".75", "correct", "3/4", False),
    ("prose_decimal", "Answer: The total cost is 17.25 dollars.", "69/4", "correct", "69/4", False),
    ("prose_wrong", "Final Answer: The maximum area of the triangle is 2501 square units.", "2500", "incorrect", "2501", False),
    ("prose_wrong_with_marker", "Final Answer: The maximum area of the triangle is 2501 square units.\n#### 2500", "2500", "ambiguous", None, False),
    ("prose_negation", "Final Answer: The maximum area of the triangle is not 2500 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_multiple_or", "Final Answer: The maximum area is 2500 or 2501 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_multiple_and", "Final Answer: The area is 2500 and the length is 50.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_numeric_subject", "Final Answer: The area of the 3 triangles is 2500 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_uncertain", "Final Answer: The area is approximately 2500 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_qualified_subject", "Final Answer: The possible area is 2500 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_incorrect_subject", "Final Answer: The value of the incorrect answer is 17.\n#### 17", "17", "extraction_failed", None, False),
    ("prose_wrong_subject", "Final Answer: The value of the wrong answer is 17.\n#### 17", "17", "extraction_failed", None, False),
    ("prose_false_subject", "Final Answer: The value of the false result is 17.\n#### 17", "17", "extraction_failed", None, False),
    ("prose_unmarked", "The maximum area of the triangle is 2500 square units.", "2500", "extraction_failed", None, False),
    ("prose_extra_sentence", "Final Answer: The area is 2500 square units. It may be wrong.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_percent", "Final Answer: The probability is 75%.\n#### 0.75", ".75", "extraction_failed", None, False),
    ("prose_unknown_units", "Final Answer: The area is 2500 magic units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_malformed_math", "Final Answer: The area is $2500 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_malformed_xml", "Final Answer: The area is <answer>2500 square units.\n#### 2500", "2500", "extraction_failed", None, False),
    ("prose_malformed_box", r"Final Answer: The area is \boxed{2500 square units."+"\n#### 2500", "2500", "extraction_failed", None, False),
    ("square_units_scalar", "Final Answer: 2500 square units", "2500", "correct", "2500", False),
    ("square_units_no_conversion", "Final Answer: 1 square meter", "10000", "incorrect", "1", False),
    ("large_exact_prose", "Final Answer: The value is 9007199254740993.", "9007199254740993", "correct", "9007199254740993", False),
    ("large_inexact_prose", "Final Answer: The value is 9007199254740992.", "9007199254740993", "incorrect", "9007199254740992", False),
]

records = [{"text": text, "answer": gold} for _, text, gold, _, _, _ in cases]
before = copy.deepcopy(records)
state = runpy.run_path(str(verifier), init_globals={"SCORE_INPUT": records})
scores = state["SCORE_OUTPUT"]
assert records == before
assert len(scores) == len(cases)
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for node in ast.walk(ast.parse(verifier.read_text())))
failures = []
checks = 0
for (label, _, _, status, value, compliant), score in zip(cases, scores):
    expected = (status, value, compliant)
    actual = (score["status"], score["extracted"], score["format_compliant"])
    if actual != expected:
        failures.append({"label": label, "expected": expected, "actual": actual})
    assert set(score) == {"correct", "reward", "extracted", "status", "format_compliant", "truncated"}
    assert score["correct"] == (score["status"] == "correct")
    assert score["reward"] == float(score["correct"])
    checks += 1

# Every literal case also runs with a different gold and an added contradiction.
variants = []
for _, text, gold, _, _, _ in cases:
    variants.extend([{"text": text, "answer": "999999999999999999999"},
                     {"text": text + "\n#### 999999999999999999999", "answer": gold}])
variant_scores = runpy.run_path(str(verifier), init_globals={"SCORE_INPUT": variants})["SCORE_OUTPUT"]
for index, case in enumerate(cases):
    score, counterfactual, contradiction = scores[index], variant_scores[2 * index], variant_scores[2 * index + 1]
    assert counterfactual["extracted"] == score["extracted"] and not counterfactual["correct"], case[0]
    assert counterfactual["format_compliant"] == score["format_compliant"], case[0]
    assert not contradiction["correct"], case[0]
    checks += 2

fixtures = json.loads((root / "tests/fixtures/real_scorer_regressions.json").read_text())
assert len(fixtures) == 6
for fixture in fixtures:
    assert hashlib.sha256(fixture["text"].encode()).hexdigest() == fixture["text_sha256"]
    assert hashlib.sha256(fixture["raw_text"].encode()).hexdigest() == fixture["raw_text_sha256"]
    assert fixture["source_status"] == "extraction_failed" and fixture["termination_reason"] == "eos"
    fixture_scores = runpy.run_path(str(verifier), init_globals={"SCORE_INPUT": [
        {"text": fixture["text"], "answer": fixture["answer"]},
        {"text": fixture["text"], "answer": "999999999999999999999"},
        {"text": fixture["text"] + "\n#### 999999999999999999999", "answer": fixture["answer"]},
    ]})["SCORE_OUTPUT"]
    if (fixture_scores[0]["status"], fixture_scores[0]["extracted"]) != (fixture["expected_status"], fixture["expected_extracted"]):
        failures.append({"label": fixture["id"] + ":" + str(fixture["sample_index"]),
                         "expected": [fixture["expected_status"], fixture["expected_extracted"]], "actual": fixture_scores[0]})
    assert fixture_scores[1]["extracted"] == fixture_scores[0]["extracted"] and not fixture_scores[1]["correct"]
    assert not fixture_scores[2]["correct"]
    checks += 3

# Only the verifier path and version assertion change in the v3 test source.
# All 294 old expectations, CLI checks and long-input cases remain unchanged.
legacy_checks = 0
if not args.skip_legacy:
    legacy_path = root.parent / "math_l4_v1/tests/test_scorer.py"
    legacy_source = legacy_path.read_text()
    assert legacy_source.count('verifier = root / "scripts/verify_answers.py"') == 1
    assert legacy_source.count('state["SCORER_VERSION"] == "3.0.0"') == 1
    legacy_source = legacy_source.replace('verifier = root / "scripts/verify_answers.py"', "verifier = Path(" + repr(str(verifier)) + ")")
    legacy_source = legacy_source.replace('state["SCORER_VERSION"] == "3.0.0"', 'state["SCORER_VERSION"] == ' + repr(state["SCORER_VERSION"]))
    with tempfile.TemporaryDirectory(prefix="scorer_v4_legacy_") as directory:
        adapted_path = Path(directory) / "legacy_test.py"
        adapted_path.write_text(legacy_source)
        legacy_state = runpy.run_path(str(adapted_path))
    legacy_checks = legacy_state["checks"]
    assert legacy_checks == 294

receipt = {"scorer_version": state["SCORER_VERSION"], "scorer_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
           "literal_cases": len(cases), "real_fixtures": len(fixtures), "new_checks": checks,
           "legacy_checks": legacy_checks, "total_checks": checks + legacy_checks,
           "failed_expectations": failures, "passed": not failures}
if args.receipt:
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
assert not failures, f"{len(failures)} regression expectations failed"
if verifier == (root / "scorer.py").resolve():
    assert state["SCORER_VERSION"] == "4.0.0"
