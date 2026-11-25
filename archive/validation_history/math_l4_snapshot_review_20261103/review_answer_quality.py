"""Read-only source-gold screening and sampled answer-quality diagnostics."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import runpy

import pyarrow.parquet as pq

snapshot = Path('/Users/xu./Downloads/math_l4_snapshot_20261103')
workspace = Path('/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207')
output = Path(__file__).resolve().parent
trusted = workspace / 'project/code/math_l4_v1'
samples_path = snapshot / 'outputs/math_l4_v1_fix1/dev/checkpoint-128/samples.jsonl'
rows = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
scorer_path = trusted / 'scripts/verify_answers.py'
assert hashlib.sha256(scorer_path.read_bytes()).digest() == hashlib.sha256((snapshot / 'scripts/verify_answers.py').read_bytes()).digest()
source_cache = {}
multibox = []
all_value_queries = []
screen_counts = {}
for split in ('train', 'dev', 'test'):
    records = [json.loads(line) for line in (snapshot / f'data/processed/{split}.jsonl').read_text().splitlines()]
    screen_counts[split] = len(records)
    for row in records:
        source_file = row['source']['source_file']
        source_path = snapshot / 'data/raw/math_full' / source_file
        if source_file not in source_cache:
            source_cache[source_file] = pq.read_table(source_path).to_pylist()
        original = source_cache[source_file][row['source']['row_index']]
        assert original['problem'] == row['question']
        boxes = []
        for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', original['solution']):
            start = position = match.end()
            depth = 1
            while position < len(original['solution']) and depth:
                if original['solution'][position] == '{':
                    depth += 1
                elif original['solution'][position] == '}':
                    depth -= 1
                position += 1
            assert depth == 0
            boxes.append(original['solution'][start:position-1])
        if len(boxes) > 1:
            multibox.append({'split': split, 'id': row['id'], 'stored_answer': row['answer'],
                'boxed_values': boxes, 'distinct_literal_boxes': len(set(boxes)),
                'question': row['question'], 'source': row['source'],
                'source_file_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'source_solution': original['solution']})
        if re.search(r'(?:find|list|enter) all (?:real numbers|positive real numbers|values|possible values|solutions)', row['question'], re.I):
            all_value_queries.append({'split': split, 'id': row['id'], 'question': row['question'],
                'stored_answer': row['answer'], 'source_solution': original['solution']})

issues = [row for row in multibox if row['distinct_literal_boxes'] > 1]
assert {row['id'] for row in issues} == {'math-l4-train-precalculus-0660', 'math-l4-train-precalculus-0471'}
for row in issues:
    row['classification'] = 'confirmed_multi_value_answer_admitted_as_scalar'
    row['root_cause'] = 'prepare_data.py lines 112-127 retains only the last boxed expression'
    row['observed_responses'] = [{k: x[k] for k in ('sample_index', 'correct', 'status', 'termination_reason', 'extracted', 'output_tokens')}
        for x in rows if x['id'] == row['id']]
    if row['id'].endswith('0660'):
        row['correct_answer_set'] = ['-3', '3/2']
        row['independent_witnesses'] = []
        for a, b, c in ((1, 1, 1), (1, 1, -2)):
            determinant = 3*a*b*c - a**3-b**3-c**3
            value = Fraction(a, b+c)+Fraction(b, a+c)+Fraction(c, a+b)
            assert determinant == 0
            row['independent_witnesses'].append({'a': a, 'b': b, 'c': c, 'determinant': determinant, 'expression': str(value)})
        row['diagnosis'] = 'Sample 3 gives the complete correct answer set but the scalar scorer rejects it; samples 0 and 1 are incomplete or wrong, sample 2 is unscored repetition. No sample in this snapshot is credited for answering only -3.'
    else:
        row['correct_answer_set'] = ['-4', '4']
        row['independent_witnesses'] = []
        for cosine, sine in ((Fraction(1), Fraction(0)), (Fraction(-15, 17), Fraction(-8, 17))):
            assert cosine*cosine+sine*sine == 1 and cosine-4*sine == 1
            row['independent_witnesses'].append({'cosine': str(cosine), 'sine': str(sine), 'expression': str(sine+4*cosine)})
        row['diagnosis'] = 'No response for this question has been downloaded yet. Its source-only defect is independent of observed outcomes.'

false_negative = next(row for row in rows if row['id'] == 'math-l4-train-number_theory-0194' and row['sample_index'] == 1)
assert false_negative['text'].count('<think>') == 0 and false_negative['text'].count('</think>') == 1
edited_text = false_negative['text'].replace('</think>', '')
replay = runpy.run_path(str(scorer_path), init_globals={'SCORE_INPUT': [
    {'text': false_negative['text'], 'answer': '378'},
    {'text': edited_text, 'answer': '378'},
    {'text': edited_text, 'answer': '379'}]})['SCORE_OUTPUT']
assert [x['correct'] for x in replay] == [False, True, False]
assert Fraction(3780*18, 180) == 378

ambiguous = [row for row in rows if row['status'] == 'ambiguous']
ambiguous_review = []
for row in ambiguous:
    ambiguous_review.append({'id': row['id'], 'sample_index': row['sample_index'], 'answer': row['answer'],
        'markers': re.findall(r'^.*(?:####|boxed|answer>).*$', row['text'], flags=re.M),
        'review': 'Terminal revised answer is correct on manual inspection, but an earlier visible final-answer marker gives a different value. The frozen conservative contradiction policy intentionally rejects the whole response; do not silently convert it to correct.'})

positive_sample = random.Random(20261103).sample([row for row in rows if row['correct'] is True], 12)
positive_reviews = []
for row in positive_sample:
    note = 'Final numerical answer checked against the question; no answer-level false positive found in this selected sample.'
    if row['id'] == 'math-l4-train-algebra-0245' and row['sample_index'] == 2:
        note += ' Reasoning contains arithmetic mistakes (4*4.9*28 written as548.48 instead of548.8), despite landing on the correct final answer2. Exact-answer correctness is not a proof of sound reasoning.'
    if row['id'] == 'math-l4-train-number_theory-0806' and row['sample_index'] == 2:
        note += ' Reasoning includes65 as a divisor and confused unordered-pair wording, while the final count2 is correct.'
    positive_reviews.append({'id': row['id'], 'sample_index': row['sample_index'], 'answer': row['answer'], 'review': note})

error_examples = []
for question_id, sample_index, diagnosis in [
    ('math-l4-train-counting_and_probability-0098', 3, 'Misreads B and C as six green balls rather than four; predicts3/5 instead of7/15.'),
    ('math-l4-train-algebra-0771', 0, 'Confuses ceiling interval(14,15] with[15,16); predicts31 instead of29.'),
    ('math-l4-train-algebra-1122', 2, 'Completes the square correctly but reads center x-coordinate as-9 instead of9; predicts0 instead of18.'),
    ('math-l4-train-geometry-0779', 2, 'Rejects the equality boundary2+9=11 as if11>11; predicts7 instead of6.'),
    ('math-l4-train-prealgebra-0762', 2, 'Counts the two middle palindrome digits independently; predicts900 instead of90.'),
    ('math-l4-train-prealgebra-0725', 1, 'Reaches the correct area400 in the partial text but keeps reconsidering and reaches the300-second deadline. It remains incorrect under the requested termination policy.'),
    ('math-l4-train-intermediate_algebra-0936', 3, 'Repeats the same polynomial-elimination sentence; repetition guard terminates generation. No completed final answer is inferred.')]:
    row = next(row for row in rows if row['id'] == question_id and row['sample_index'] == sample_index)
    error_examples.append({'id': question_id, 'sample_index': sample_index, 'gold': row['answer'],
        'status': row['status'], 'termination_reason': row['termination_reason'], 'correct': row['correct'],
        'diagnosis': diagnosis, 'tail_excerpt': row['text'][-1000:]})

result = {
    'scope': 'Independent diagnostic review; no source/downloaded outputs were modified and no model, GPU, pickle, or network was used.',
    'samples_path': str(samples_path), 'samples_sha256': hashlib.sha256(samples_path.read_bytes()).hexdigest(),
    'rows': len(rows), 'statuses': dict(Counter(row['status'] for row in rows)),
    'termination_counts': dict(Counter(row['termination_reason'] for row in rows)),
    'source_screen': {'counts': screen_counts, 'multi_box_record_count': len(multibox),
        'distinct_box_defects': issues, 'all_value_query_count': len(all_value_queries),
        'all_value_queries': all_value_queries,
        'limits': 'All840 prepared records were scanned for multiple boxed source expressions; explicit find/list/enter-all-value questions were reviewed against their source solutions. This is not a full mathematical proof of every label.'},
    'confirmed_scalar_math_false_negative': {'id': false_negative['id'], 'sample_index': 1,
        'gold': '378', 'final_line': false_negative['text'].splitlines()[-1],
        'root_cause': 'One unmatched closing think tag makes scorer lines48-60 flag malformed_think and clear the complete text.',
        'original_replay': replay[0], 'diagnostic_remove_only_unmatched_closing_tag': replay[1],
        'counterfactual_gold379': replay[2], 'production_or_saved_scores_changed': False},
    'ambiguous_correct_terminal_answers': ambiguous_review,
    'positive_sample_seed': 20261103, 'positive_answer_reviews': positive_reviews,
    'error_examples': error_examples,
    'recommendation': 'Preserve the completed training and immutable raw outputs. The dataset demonstrably challenges this model, but current dev labels include two multi-value questions incorrectly admitted as scalars. Resolve these source-defined eligibility defects through a documented separately versioned selection/reanalysis rule before trusting the checkpoint choice. Audit malformed-think extraction separately without changing the running protocol. Do not restart training on this evidence, and do not claim RL improvement without matched Base/RL evaluation.',
    'limits': ['Partial dev is a sequential prefix, not a random sample of the128 dev questions.',
        'Six terminally correct revisions remain ambiguous under the frozen public-answer conflict rule; they are not included in official correct counts.',
        'No answer-level false positive was found among12 deterministic sampled positives; this does not validate all positives or all reasoning.',
        'Train and test had no distinct-multiple-box defects in this screen; other semantic errors remain possible.']
}
(output / 'quality_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'rows': len(rows), 'source_counts': screen_counts, 'distinct_box_defects': [row['id'] for row in issues],
    'confirmed_scalar_math_false_negative': false_negative['id'], 'ambiguous_rows': len(ambiguous),
    'sampled_positive_checks': len(positive_reviews), 'saved': str(output / 'quality_review.json')}, indent=2))
