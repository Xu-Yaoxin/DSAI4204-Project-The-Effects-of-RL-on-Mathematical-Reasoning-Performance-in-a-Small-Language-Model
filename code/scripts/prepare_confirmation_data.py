"""Freeze 200 unseen scalar calibration questions; source data only, no model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import runpy
import ssl
import urllib.request

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('data/boundary_confirm_v2'))
parser.add_argument('--math-cache', type=Path, help='Optional local copy of the pinned full-MATH repository files')
parser.add_argument('--offline', action='store_true')
parser.add_argument('--verify', action='store_true')
args = parser.parse_args()
output = (root / args.output).absolute()
if any(p.is_symlink() for p in (output, *output.parents)):
    raise SystemExit('Output must not contain symlinks')
output = output.resolve()
if output == root or output in root.parents:
    raise SystemExit('Use a dedicated dataset directory')
seed = 20261101
apple_revision = '93b5b3758d9d9841ffe81d6cd2ae2b030685b078'
math_revision = '21a5633873b6a120296cce3e2df9d5550074f4a3'
math_hashes = {
    'algebra': '0e04a850dac9cfc758eac0c80afd2d0cc65e29ea72f916c8ac1045b12af937f5',
    'counting_and_probability': 'c324818949c75a6227c0328a8499d58023cbbf6757cfc1736f5ab24cc8d87966',
    'geometry': '98720f3a8134ad9784fd8951195245264495df03ffc5c9485a9a54ff5ae2431a',
    'intermediate_algebra': 'e859f72486c951c2017da880541ca729940eaa986b0364eb23e238983df9bfeb',
    'number_theory': 'ba156c9c9fa2040836ba12135fd964e6e67d6bccd8c469ec4a3ad941f8f6fbe9',
    'prealgebra': '301e5d8ade5c7b5038032532e498bdc9dca799e7f1061a63f2e7d980803240cc',
    'precalculus': 'da37474ba3761ae5727af3844a7ba963a735bd6a14c65e6137593669a4fe3b9c',
}
source_specs = [
    {'dataset': 'apple/GSM-Symbolic', 'revision': apple_revision, 'remote': 'main/test.jsonl', 'local': 'gsm_symbolic/main.jsonl', 'git_blob_sha1': '8644707d8edb3179cc53876357b4d0f847f5335f', 'license': 'CC-BY-NC-ND-4.0'},
    {'dataset': 'apple/GSM-Symbolic', 'revision': apple_revision, 'remote': 'README.md', 'local': 'gsm_symbolic/README.md', 'git_blob_sha1': 'f5a7fa6b1fe8c4f4da817c60edbc5e9364e8e628', 'license': 'CC-BY-NC-ND-4.0'},
    {'dataset': 'apple/GSM-Symbolic', 'revision': apple_revision, 'remote': 'LICENSE-CC-BY-NC-ND-4.0.md', 'local': 'gsm_symbolic/LICENSE.md', 'git_blob_sha1': 'eece3f3d351c27555403d89a3700b68c56bfffd6', 'license': 'CC-BY-NC-ND-4.0'},
]
for subject, digest in math_hashes.items():
    source_specs.append({'dataset': 'EleutherAI/hendrycks_math', 'revision': math_revision, 'remote': f'{subject}/test-00000-of-00001.parquet', 'local': f'math_full/{subject}/test-00000-of-00001.parquet', 'sha256': digest, 'license': 'Original MATH repository: MIT'})
source_specs += [
    {'dataset': 'EleutherAI/hendrycks_math', 'revision': math_revision, 'remote': 'README.md', 'local': 'math_full/README.md', 'git_blob_sha1': '9ab6422553ffa7c65f5c11386d1599fc77149101', 'license': 'Original MATH repository: MIT'},
    {'dataset': 'hendrycks/math', 'revision': '985bdc1696e88e8643f081a0ff4719da39f2ae2a', 'remote': 'LICENSE', 'local': 'math_full/ORIGINAL_MATH_LICENSE', 'license': 'MIT', 'url': 'https://raw.githubusercontent.com/hendrycks/math/985bdc1696e88e8643f081a0ff4719da39f2ae2a/LICENSE'},
]
old_root = root / 'data/boundary_v1'
previous = json.loads((output / 'manifest.json').read_text()) if (output / 'manifest.json').is_file() else None
if args.verify and previous is None:
    raise SystemExit('No completed confirmation manifest to verify')
print('[CONFIRM DATA 1/4] 0% | PINNED SOURCES | models=0', flush=True)
sources = []
for index, spec in enumerate(source_specs, 1):
    target = output / 'raw' / spec['local']
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise SystemExit(f'Unsafe source symlink: {target}')
    url = spec.get('url', f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}/{spec['remote']}")
    cache = None
    if spec['dataset'] == 'apple/GSM-Symbolic':
        cache = old_root / 'raw' / spec['local']
    elif spec['dataset'] == 'hendrycks/math':
        cache = old_root / 'raw/math500/ORIGINAL_MATH_LICENSE'
    elif args.math_cache:
        cache = args.math_cache / spec['remote']
    if target.is_file():
        content = target.read_bytes()
    elif args.verify or previous:
        raise SystemExit(f'Missing frozen source: {target}')
    elif cache is not None and cache.is_file() and not cache.is_symlink():
        content = cache.read_bytes()
    elif args.offline:
        raise SystemExit(f'Missing pinned source: {target}; provide --math-cache or run online')
    else:
        context = ssl.create_default_context()
        try:
            import certifi
        except ImportError:
            pass
        else:
            context = ssl.create_default_context(cafile=certifi.where())
        print(f'[DOWNLOAD {index}/{len(source_specs)}] {url}', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'DSAI4207-confirmation-data/2'})
        with urllib.request.urlopen(request, context=context, timeout=60) as response:
            content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    if (spec.get('sha256') and spec['sha256'] != digest) or (spec.get('git_blob_sha1') and spec['git_blob_sha1'] != blob):
        raise SystemExit(f'Pinned source hash mismatch: {target}')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_suffix(target.suffix + '.part').write_bytes(content)
        target.with_suffix(target.suffix + '.part').replace(target)
    sources.append({**spec, 'url': url, 'path': str(target.relative_to(output)), 'bytes': len(content), 'sha256': digest})
    print(f'[SOURCE {index}/{len(source_specs)}] verified {len(content):,} bytes', flush=True)

history_hashes = {}
old_questions = set()
old_gsm_ids = set()
for relative in ['data/processed/' + name + '.jsonl' for name in ('train', 'pilot', 'engineering', 'dev', 'test')] + ['data/boundary_v1/questions.jsonl']:
    path = root / relative
    content = path.read_bytes()
    history_hashes[relative] = hashlib.sha256(content).hexdigest()
    for row in [json.loads(line) for line in content.decode().split('\n') if line.strip()]:
        old_questions.add(' '.join(row['question'].casefold().split()))
        if row.get('source', {}).get('official_split') == 'test':
            old_gsm_ids.add(row['source']['row_index'])
        elif str(row.get('id', '')).startswith('gsm8k-test-'):
            old_gsm_ids.add(int(row['id'].rsplit('-', 1)[-1]))
        elif row.get('source', {}).get('dataset') == 'apple/GSM-Symbolic':
            old_gsm_ids.add(row['source']['original_id'])
old_manifest = json.loads((old_root / 'manifest.json').read_text())
old_digest = hashlib.sha256((old_root / 'questions.jsonl').read_bytes()).hexdigest()
if old_manifest['files']['questions.jsonl']['sha256'] != old_digest:
    raise SystemExit('Previous boundary dataset differs from its frozen manifest')
history_hashes['data/boundary_v1/manifest.json'] = hashlib.sha256((old_root / 'manifest.json').read_bytes()).hexdigest()
verifier = root / 'scripts/verify_answers.py'
verifier_hash = hashlib.sha256(verifier.read_bytes()).hexdigest()
print('[CONFIRM DATA 2/4] 25% | HISTORY EXCLUSION AND EXACT SCALAR GOLD', flush=True)
gsm_rows = [json.loads(line) for line in (output / 'raw/gsm_symbolic/main.jsonl').read_text().split('\n') if line.strip()]
records = []
gsm_exclusions = Counter()
seen = set()
for index, row in enumerate(gsm_rows):
    normalized = ' '.join(row['question'].casefold().split())
    if row['original_id'] in old_gsm_ids:
        gsm_exclusions['previous_gsm_family'] += 1
    elif normalized in old_questions:
        gsm_exclusions['previous_normalized_question'] += 1
    elif normalized in seen:
        gsm_exclusions['duplicate_normalized_question'] += 1
    else:
        seen.add(normalized)
        records.append({'row_index': index, 'raw': row, 'gold': row['answer'].rsplit('####', 1)[-1].strip()})
scores = runpy.run_path(str(verifier), init_globals={'SCORE_INPUT': [{'text': '#### ' + item['gold'], 'answer': item['gold']} for item in records]})['SCORE_OUTPUT']
gsm_pool = []
for item, score in zip(records, scores):
    if score['correct']:
        item['canonical_answer'] = score['extracted']
        gsm_pool.append(item)
    else:
        gsm_exclusions['unsupported_non_scalar_gold'] += 1

import pyarrow.parquet as pq
math_records = []
math_exclusions = Counter()
math_source_counts = {}
seen = set()
for subject in math_hashes:
    source_file = f'{subject}/test-00000-of-00001.parquet'
    rows = pq.read_table(output / 'raw/math_full' / source_file).to_pylist()
    math_source_counts[subject] = len(rows)
    for index, row in enumerate(rows):
        normalized = ' '.join(row['problem'].casefold().split())
        if row.get('level') not in (4, 'Level 4', '4'):
            math_exclusions['not_level4'] += 1
            continue
        if normalized in old_questions:
            math_exclusions['previous_normalized_question'] += 1
            continue
        if normalized in seen:
            math_exclusions['duplicate_normalized_question'] += 1
            continue
        if any(marker in row['problem'].lower() for marker in ('[asy]', '\\includegraphics', '<img')):
            math_exclusions['requires_diagram'] += 1
            continue
        seen.add(normalized)
        matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', row['solution']))
        if not matches:
            math_exclusions['missing_boxed_source_gold'] += 1
            continue
        match = matches[-1]
        start = position = match.end()
        depth = 1
        while position < len(row['solution']) and depth:
            if row['solution'][position] == '{':
                depth += 1
            elif row['solution'][position] == '}':
                depth -= 1
            position += 1
        if depth:
            math_exclusions['unclosed_boxed_source_gold'] += 1
            continue
        gold = row['solution'][start:position - 1].strip()
        math_records.append({'row_index': index, 'raw': row, 'gold': gold, 'source_file': source_file, 'subject': subject, 'question_hash': hashlib.sha256(normalized.encode()).hexdigest()})
scores = runpy.run_path(str(verifier), init_globals={'SCORE_INPUT': [{'text': '#### ' + item['gold'], 'answer': item['gold']} for item in math_records]})['SCORE_OUTPUT']
math_pool = []
for item, score in zip(math_records, scores):
    if score['correct']:
        item['canonical_answer'] = score['extracted']
        math_pool.append(item)
    else:
        math_exclusions['unsupported_non_scalar_gold'] += 1

families = sorted({item['raw']['original_id'] for item in gsm_pool})
if len(families) < 50 or len(math_pool) < 100:
    raise SystemExit(f'Insufficient unseen pool: GSM families={len(families)}, MATH Level4 scalar={len(math_pool)}')
rng = random.Random(seed)
chosen_families = sorted(rng.sample(families, min(100, len(families))))
second_families = set(rng.sample(chosen_families, 100 - len(chosen_families)))
groups = [{'id': 'gsm_main_confirm', 'label': 'GSM-Symbolic main confirmation', 'count': 100}, {'id': 'math_l4_confirm', 'label': 'MATH Level4 scalar confirmation', 'count': 100}]
selected = []
for family in chosen_families:
    candidates = [item for item in gsm_pool if item['raw']['original_id'] == family]
    quota = 2 if family in second_families else 1
    if len(candidates) < quota:
        raise SystemExit(f'GSM family {family} has insufficient distinct eligible instances')
    for item in sorted(rng.sample(candidates, quota), key=lambda item: item['row_index']):
        row = item['raw']
        selected.append({'id': f"gsm-main-confirm-{family:04d}-{row['instance']:02d}", 'question': row['question'], 'answer': item['canonical_answer'], 'source_answer': row['answer'], 'group_id': groups[0]['id'], 'group_label': groups[0]['label'], 'family_id': f'gsm8k-test-{family:06d}', 'difficulty_label': 'main', 'source': {'dataset': 'apple/GSM-Symbolic', 'revision': apple_revision, 'split': 'test', 'subset': 'main', 'row_index': item['row_index'], 'original_id': family, 'instance': row['instance'], 'canary': row.get('canary')}})
for item in sorted(rng.sample(math_pool, 100), key=lambda item: (item['subject'], item['row_index'])):
    source_id = f"{item['source_file']}#row={item['row_index']}"
    selected.append({'id': f"math-l4-confirm-{item['subject']}-{item['row_index']:04d}", 'question': item['raw']['problem'], 'answer': item['canonical_answer'], 'source_answer': item['gold'], 'group_id': groups[1]['id'], 'group_label': groups[1]['label'], 'family_id': 'math-problem-' + item['question_hash'], 'difficulty_label': 'Level 4', 'source': {'dataset': 'EleutherAI/hendrycks_math', 'revision': math_revision, 'split': 'test', 'source_file': item['source_file'], 'row_index': item['row_index'], 'original_id': source_id, 'original_id_kind': 'pinned_mirror_file_and_row_not_original_math_filename', 'level': 4, 'subject': item['subject']}})
normalized = [' '.join(row['question'].casefold().split()) for row in selected]
if len(selected) != 200 or len({row['id'] for row in selected}) != 200 or len(set(normalized)) != 200 or set(normalized) & old_questions:
    raise SystemExit('Selected questions must be unique and disjoint from all known previous question texts')
content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in selected).encode()
gsm_counts = Counter(row['family_id'] for row in selected if row['group_id'] == groups[0]['id'])
manifest = {'schema_version': 1, 'purpose': 'exploratory_baseline_calibration', 'stage': 'fresh_source_group_calibration_confirmation', 'seed': seed,
    'model_scores_used_for_selection': False, 'source_group_choice_informed_by_prior_screen': True,
    'files': {'questions.jsonl': {'sha256': hashlib.sha256(content).hexdigest(), 'rows': 200}}, 'groups': groups, 'sources': sources,
    'verifier_sha256': verifier_hash, 'preparation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'history_files_sha256': history_hashes,
    'source_rows': {'gsm_main': len(gsm_rows), 'math_full_test': sum(math_source_counts.values()), 'math_full_test_by_subject': math_source_counts},
    'eligible_unseen_rows': {'gsm_main': len(gsm_pool), 'math_level4_scalar': len(math_pool)}, 'eligible_unseen_gsm_families': len(families),
    'exclusions': {'gsm_main': dict(gsm_exclusions), 'math_full_test': dict(math_exclusions)}, 'excluded_gsm_original_ids': sorted(old_gsm_ids),
    'selected_gsm_original_ids': chosen_families, 'gsm_family_counts': dict(sorted(gsm_counts.items())), 'gsm_family_count_histogram': dict(Counter(gsm_counts.values())),
    'math_selected_subject_counts': dict(Counter(row['source']['subject'] for row in selected if row['group_id'] == groups[1]['id'])),
    'semantic_gold_review': 'not_performed; pinned source gold passed exact scalar parsing only',
    'limitations': ['Source groups were chosen after the previous screen; individual new questions were sampled without model scores.', 'GSM100 rows are clustered within64 original families in this build; use family-cluster uncertainty rather than100-independent-family claims.', 'MATH uses fresh original-test Level4 scalar text-only questions across seven subjects, not the full benchmark score.', 'Missing final boxed source gold, non-rational expressions and diagram questions were excluded; this selects a subset.', 'Exact normalized text and known GSM-family exclusion do not prove absence of semantic overlap or pretraining contamination.', 'Gold, source solutions and canary metadata are provenance only; generation prompts must contain question text alone.', 'Confirmation refers to source-group difficulty calibration, not a formal Base/RL hypothesis test.']}
if previous:
    if previous != json.loads(json.dumps(manifest)) or (output / 'questions.jsonl').read_bytes() != content:
        raise SystemExit('Existing confirmation content or provenance changed; preserve it and use a new output')
else:
    (output / 'questions.jsonl').write_bytes(content)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Fresh baseline calibration: 200 questions', '', '100 GSM-Symbolic main questions and100 original MATH Level4 scalar questions. Verbatim problem text; answers are stored separately in questions.jsonl and must not enter prompts.', '']
    for group in groups:
        lines += [f"## {group['label']}", '']
        for row in selected:
            if row['group_id'] == group['id']:
                lines += [f"### {row['id']}", '', row['question'], '']
    (output / 'questions.md').write_text('\n'.join(lines))
    (output / 'README.md').write_text('# Fresh source-group calibration v2\n\n200 verbatim, previously unused question texts:100 GSM-Symbolic main and100 original MATH test Level4 scalar text-only questions. Seed20261101. The two source groups were chosen after the prior screen; individual questions were not selected using model scores. This confirms source-group difficulty, not a formal Base/RL hypothesis test.\n\nAll previous boundary questions and complete GSM source families are excluded, as are known previously evaluated GSM8K test families and exact normalized train/dev/test questions. The GSM100 rows cover64 families:36 families contribute2 rows,28 contribute1. Treat shared-family responses as clustered. MATH uses all seven original-test subjects before fixed random selection; scalar and diagram filtering changes benchmark coverage.\n\nSources: https://huggingface.co/datasets/apple/GSM-Symbolic (CC-BY-NC-ND-4.0); https://huggingface.co/datasets/EleutherAI/hendrycks_math (full MATH mirror at pinned revision); https://github.com/hendrycks/math (original MIT license included separately). Source bytes, revisions, hashes, filters and history hashes are in manifest.json. Raw pools and licenses are in raw/. The full-MATH mirror does not retain original JSON filenames; source.original_id is explicitly a pinned mirror file/row locator, and family_id hashes normalized problem text.\n\nGold is extracted from the final balanced boxed/fbox source-solution answer and validated as an exact scalar by the frozen project scorer. No model is called. This is parsing validation, not a semantic review of200 source answers. No target accuracy is guaranteed before inference. Source gold, solutions and canaries never belong in generation prompts.\n\nFrom the main code directory: python3 scripts/prepare_confirmation_data.py --verify. Preparation uses pyarrow already present in the GPU requirements; the uploaded inference package only needs questions.jsonl, manifest.json, questions.md and licenses, not raw parquet pools.\n')
print(f'[CONFIRM DATA 3/4] 90% | GSM rows=100 families={len(gsm_counts)} | MATH Level4=100 | eligible_math={len(math_pool)}', flush=True)
print(f"✅ [CONFIRM DATA 4/4] 100% | {'VERIFIED' if previous else 'SAVED'} | {output / 'questions.jsonl'}", flush=True)
