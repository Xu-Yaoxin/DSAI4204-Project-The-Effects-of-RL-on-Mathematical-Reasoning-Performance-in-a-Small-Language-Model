"""Rescore frozen Base/RL answers locally without generating or selecting anything."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import shutil
import tempfile
from types import SimpleNamespace

package_root = Path(__file__).resolve().parent
code_root = package_root.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root', default=str(code_root / 'outputs/validation/full_math_l4_review_20261106/experiment'))
parser.add_argument('--output', default=str(code_root / 'outputs/rescore_20261106'))
args = parser.parse_args()
root = Path(args.experiment_root).expanduser().absolute()
output = Path(args.output).expanduser().absolute()
for path in (root, output):
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('Source and output paths must not contain symlinks.')
root = root.resolve()
output = output.resolve()
if not root.is_dir():
    raise ValueError('The frozen experiment root does not exist.')
for protected in (root, package_root, code_root / 'math_l4_v1', code_root / 'math_l4_rl_split_v1'):
    if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
        raise ValueError('Output must be separate from source and scorer trees.')

schema = 'math-l4-posthoc-rescore-v1'
scorer_version = '4.0.0'
score_fields = ('correct', 'reward', 'extracted', 'status', 'format_compliant', 'truncated')
frozen_code = {
    'analyze.py': '3cfdf3dd6a650a72fb68ad1a5f3411869b8a8be2a265105a04fdb38eae08a028',
    'repetition_guard.py': '56597e251b32476ce497511903ed806513b1768f790a591d89eaccb426e8f33c',
    'validate_evaluation.py': '7b33b9962b57cc1a2d83a8508a8332ce459f12d017110e6950de483085757c8c',
    'validate_guarded_sample.py': '3eaf004fe6cbd531da1464858313782fe5967a4ccff1e72f308e4cae35681ff6',
    'validate_selection.py': 'b8c8fc384c315d34d1d64032f7bc3fa6b6e8774eee372079d8775a172c1a3d97',
    'verify_answers.py': 'f133f1b99257d0c4677b11c44145577a87edefcc55a3d2cb42bcebbe9aa7944f',
}
for name, expected in frozen_code.items():
    path = root / 'scripts' / name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Frozen validation/scoring/statistics source changed: {name}')
statistics_source = (root / 'scripts/analyze.py').read_text()
statistics_block = statistics_source[statistics_source.index('records = {label:'):statistics_source.index('limitations = [')]
statistics_sha256 = hashlib.sha256(statistics_block.encode()).hexdigest()
if statistics_sha256 != 'c08fa7f5655a7a6dc8572aed74017b721b216b88137734152c3d69f3735378ea':
    raise ValueError('Frozen statistics block differs from the approved exact extraction.')
statistics_code = compile(statistics_block, str(root / 'scripts/analyze.py') + ':statistics', 'exec')

print('[RESCORE 1/5] 0% | checking immutable sources and frozen validation code', flush=True)
source_files = {}
for path in sorted(root.rglob('*')):
    if path.is_symlink():
        raise ValueError(f'Frozen source contains a symlink: {path}')
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        source_files[str(path.relative_to(root))] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
audit_paths = [root.parent / 'source_inventory.json', root.parent / 'integrity_audit.json']
audit_bindings = {}
if any(path.exists() for path in audit_paths):
    if not all(path.is_file() and not path.is_symlink() for path in audit_paths):
        raise ValueError('The snapshot source inventory and integrity audit must both be present and safe.')
    inventory = json.loads(audit_paths[0].read_bytes())
    audit = json.loads(audit_paths[1].read_bytes())
    if inventory.get('source_modified') is not False or audit.get('complete') is not True:
        raise ValueError('Snapshot inventory/audit does not describe an immutable complete snapshot.')
    if audit.get('source_inventory_sha256') != hashlib.sha256(audit_paths[0].read_bytes()).hexdigest():
        raise ValueError('Snapshot integrity audit does not bind the source inventory.')
    if not isinstance(inventory.get('files'), dict) or not inventory['files']:
        raise ValueError('Snapshot source inventory is empty or malformed.')
    for relative, expected in inventory['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or source_files.get(relative) != expected:
            raise ValueError(f'Snapshot source inventory mismatch: {relative}')
    for field, relative in (
            ('analysis_ready_sha256', 'outputs/math_l4_v1_fix1/main_analysis/ready.json'),
            ('merge_provenance_sha256', 'outputs/math_l4_v1_fix1/test_rl_split_merged/merge_provenance.json')):
        if field in audit and source_files.get(relative, {}).get('sha256') != audit[field]:
            raise ValueError(f'Snapshot audit artifact changed: {relative}')
    audit_bindings = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in audit_paths}

cfg = json.loads((root / 'configs/math_l4.json').read_bytes())
validated = {}
source_lines = {}
for label, relative in (('base', 'outputs/math_l4_v1_fix1/test_base'), ('rl', 'outputs/math_l4_v1_fix1/test_rl_split_merged')):
    path = root / relative
    validated[label] = runpy.run_path(str(root / 'scripts/validate_evaluation.py'), init_globals={
        'EVALUATION_PATH': path, 'VALIDATE_CONFIG': cfg})['EVALUATION_RESULT']
    manifest = validated[label]['manifest']
    if (manifest['model_label'] != label or manifest['split'] != 'test'
            or manifest['n_samples'] != cfg['evaluation']['n_samples']
            or manifest['question_count'] != cfg['data']['test_size']):
        raise ValueError('Rescoring requires the complete configured Base/RL test grids.')
    if any('old_score' in row or 'rescore' in row for row in validated[label]['rows']):
        raise ValueError('Inputs must be original generation records, not previous rescore outputs.')
    source_lines[label] = {(row['id'], row['sample_index']): line_number
        for line_number, line in enumerate((path / 'samples.jsonl').read_text().splitlines(), 1)
        if line.strip() for row in [json.loads(line)]}
    validated[label]['source_path'] = str(path)
    print(f'[RESCORE 1/5] {10 if label == "base" else 20}% | {label}: {len(validated[label]["rows"])} frozen scores and stop traces validated', flush=True)
base = validated['base']['manifest']
rl = validated['rl']['manifest']
for field in ('protocol_id', 'settings', 'dataset_sha256', 'dataset_manifest_sha256', 'n_samples', 'question_count', 'model_identity', 'model_files', 'code_hashes'):
    if base[field] != rl[field]:
        raise ValueError(f'Unmatched Base/RL generation provenance: {field}')
if validated['base']['dataset'] != validated['rl']['dataset']:
    raise ValueError('Base/RL question and gold metadata differ.')
paired_rows = {label: {(row['id'], row['sample_index']): row for row in validated[label]['rows']} for label in ('base', 'rl')}
if set(paired_rows['base']) != set(paired_rows['rl']):
    raise ValueError('Base/RL question/sample grids are not paired.')
if any(paired_rows['base'][key]['sample_seed'] != paired_rows['rl'][key]['sample_seed'] for key in paired_rows['base']):
    raise ValueError('Base/RL paired sample seeds differ.')
selection = runpy.run_path(str(root / 'scripts/validate_selection.py'), init_globals={
    'SELECTION_PATH': root / rl['selection_path'], 'SELECTION_CONFIG': cfg})['SELECTION_RESULT']
if (selection['selection_id'] != rl['selection_identity'] or selection['adapter_identity'] != rl['adapter_identity']
        or selection['adapter_path'] != rl['adapter_path'] or selection['adapter_files'] != rl['adapter_files']):
    raise ValueError('RL test adapter differs from the frozen development selection.')
if audit_bindings and any(audit.get(field) != expected for field, expected in (
        ('question_count', base['question_count']), ('base_answers', len(validated['base']['rows'])),
        ('rl_answers', len(validated['rl']['rows'])))):
    raise ValueError('Snapshot audit counts differ from the validated paired grids.')

scorer_path = package_root / 'scorer.py'
if scorer_path.is_symlink() or not scorer_path.is_file():
    raise ValueError('The local v4 scorer is missing or unsafe.')
scorer_bytes = scorer_path.read_bytes()
scorer_sha256 = hashlib.sha256(scorer_bytes).hexdigest()
scorer_code = compile(scorer_bytes, str(scorer_path), 'exec')
scorer_probe = {'SCORE_INPUT': [], '__file__': str(scorer_path), '__name__': 'posthoc_scorer'}
exec(scorer_code, scorer_probe)
if scorer_probe.get('SCORER_VERSION') != scorer_version or scorer_probe.get('SCORE_OUTPUT') != []:
    raise ValueError('The rescore runner requires the v4.0.0 six-field scorer interface.')
identity = {'schema_version': schema, 'experiment_root': str(root), 'source_files': source_files,
    'source_audit_sha256': audit_bindings,
    'evaluations': {label: {key: validated[label][key] for key in ('manifest_sha256', 'samples_sha256')} for label in ('base', 'rl')},
    'selection_identity': selection['selection_id'], 'scorer_version': scorer_version, 'scorer_sha256': scorer_sha256,
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'statistics_source_sha256': frozen_code['analyze.py'], 'statistics_block_sha256': statistics_sha256,
    'bootstrap': 2000, 'bootstrap_seed': 42, 'stopped_answer_policy': 'preserve all original non-EOS score fields'}
identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if output.exists():
    ready_path = output / 'ready.json'
    if not ready_path.is_file() or ready_path.is_symlink():
        raise ValueError('Existing output is partial; preserve it and choose a new directory.')
    ready = json.loads(ready_path.read_bytes())
    if ready.get('complete') is not True or ready.get('identity') != identity or ready.get('identity_sha256') != identity_sha256:
        raise ValueError('Existing output belongs to a different source/scorer/runner comparison.')
    inventory = {}
    for path in output.rglob('*'):
        if path.is_symlink():
            raise ValueError('Completed rescore output contains a symlink.')
        if path.is_file() and path != ready_path:
            inventory[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if inventory != ready.get('artifacts_sha256'):
        raise ValueError('Completed rescore artifacts changed.')
    print(f'[RESCORE CACHED] {output}', flush=True)
    raise SystemExit(0)
print('[RESCORE 2/5] 40% | complete paired grids, original scores and frozen development selection verified', flush=True)

rescored = {}
changes = []
change_counts = {}
generation_hashes = {}
for label in ('base', 'rl'):
    source_rows = validated[label]['rows']
    eos_rows = [row for row in source_rows if row['termination_reason'] == 'eos']
    score_namespace = {'SCORE_INPUT': [{key: row[key] for key in ('text', 'answer', 'truncated')} for row in eos_rows],
        '__file__': str(scorer_path), '__name__': 'posthoc_scorer'}
    exec(scorer_code, score_namespace)
    scores = score_namespace.get('SCORE_OUTPUT')
    if score_namespace.get('SCORER_VERSION') != scorer_version or not isinstance(scores, list) or len(scores) != len(eos_rows):
        raise ValueError('The v4 scorer returned an invalid result grid.')
    scored_by_key = {(row['id'], row['sample_index']): score for row, score in zip(eos_rows, scores)}
    rescored[label] = {**validated[label], 'rows': []}
    original_payload_hash = hashlib.sha256()
    rescored_payload_hash = hashlib.sha256()
    counts = Counter()
    for row in source_rows:
        key = (row['id'], row['sample_index'])
        old_score = {field: row[field] for field in score_fields}
        new_score = scored_by_key[key] if row['termination_reason'] == 'eos' else old_score.copy()
        if not isinstance(new_score, dict) or set(new_score) != set(score_fields):
            raise ValueError('The v4 scorer changed the exact six-field scoring contract.')
        if row['termination_reason'] == 'eos' and (
                type(new_score['correct']) is not bool or type(new_score['format_compliant']) is not bool
                or type(new_score['reward']) not in (int, float) or new_score['reward'] != float(new_score['correct'])
                or new_score['truncated'] is not False
                or new_score['status'] not in ('correct', 'incorrect', 'extraction_failed', 'ambiguous')
                or (new_score['status'] == 'correct') != new_score['correct']):
            raise ValueError('The v4 scorer returned invalid EOS scoring values.')
        source_key = {'model_label': label, 'id': row['id'], 'sample_index': row['sample_index'],
            'sample_seed': row['sample_seed'], 'line_number': source_lines[label][key]}
        new_row = {**row, **new_score, 'old_score': old_score,
            'rescore': {'schema_version': schema, 'scorer_version': scorer_version, 'posthoc': True,
                'method': 'posthoc',
                'scorer_sha256': scorer_sha256, 'source_samples_sha256': validated[label]['samples_sha256'],
                'source_key': source_key, 'scoring_applied': row['termination_reason'] == 'eos'}}
        original_payload = {field: value for field, value in row.items() if field not in score_fields}
        new_payload = {field: value for field, value in new_row.items() if field not in (*score_fields, 'old_score', 'rescore')}
        if original_payload != new_payload:
            raise ValueError('Rescoring changed immutable generation/question metadata.')
        for payload, digest in ((original_payload, original_payload_hash), (new_payload, rescored_payload_hash)):
            digest.update((json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
        changed_fields = [field for field in score_fields if old_score[field] != new_score[field]]
        counts['eos_rescored' if row['termination_reason'] == 'eos' else 'stopped_preserved'] += 1
        if changed_fields:
            changes.append({**source_key, 'source_key': source_key, 'source_samples_sha256': validated[label]['samples_sha256'],
                'termination_reason': row['termination_reason'], 'old_score': old_score, 'new_score': new_score,
                'changed_fields': changed_fields})
            counts['changed_samples'] += 1
            counts['correctness_changes'] += old_score['correct'] != new_score['correct']
            counts['false_to_true'] += old_score['correct'] is False and new_score['correct'] is True
            counts['true_to_false'] += old_score['correct'] is True and new_score['correct'] is False
            for field in changed_fields:
                counts['field_' + field] += 1
        rescored[label]['rows'].append(new_row)
    if original_payload_hash.hexdigest() != rescored_payload_hash.hexdigest():
        raise ValueError('Generation payload digest changed.')
    generation_hashes[label] = original_payload_hash.hexdigest()
    change_counts[label] = dict(counts)
print('[RESCORE 3/5] 60% | EOS answers rescored uniformly; stopped answers and generation payloads preserved', flush=True)

statistics_runs = {}
for name, values in (('original', validated), ('rescored', rescored)):
    namespace = {'validated': values, 'base': base, 'args': SimpleNamespace(seed=42), 'repeats': 2000,
        'Counter': Counter, 'defaultdict': defaultdict, 'math': math, 'random': random}
    exec(statistics_code, namespace)
    statistics_runs[name] = {key: namespace[key] for key in (
        'per_question', 'metrics', 'coverage', 'diagnostics', 'count', 'families')}
current = statistics_runs['rescored']
original = statistics_runs['original']
original_questions = {question['id']: question for question in original['per_question']}
for question in current['per_question']:
    question['original_scores'] = {label: original_questions[question['id']][label] for label in ('base', 'rl')}
generation_provenance = {'model_identity': base['model_identity'], 'adapter_identity': rl['adapter_identity'],
    'selection_identity': selection['selection_id'], 'selected_step': selection['selected_step'],
    'source_protocol_id': base['protocol_id'], 'settings': base['settings'],
    'checkpoint_selection_unchanged': True, 'new_generation_performed': False}
summary = {'schema_version': schema, 'posthoc': True, 'scorer_version': scorer_version,
    'scorer_sha256': scorer_sha256, 'identity_sha256': identity_sha256,
    'generation_provenance': generation_provenance, 'question_count': current['count'],
    'family_count': len(current['families']), 'n_samples': base['n_samples'], 'bootstrap': 2000, 'bootstrap_seed': 42,
    'interval_method': 'paired family-cluster percentile bootstrap,95%,linear quantiles',
    'statistics_provenance': {'method': 'exact checked extraction from frozen analyze.py',
        'source_sha256': frozen_code['analyze.py'], 'block_sha256': statistics_sha256,
        'start_marker': 'records = {label:', 'end_marker_exclusive': 'limitations = ['},
    'metrics': current['metrics'], 'observed_coverage_counts': dict(current['coverage']), 'diagnostics': current['diagnostics'],
    'original_statistics': {'metrics': original['metrics'], 'observed_coverage_counts': dict(original['coverage']),
        'diagnostics': original['diagnostics']}, 'changes': change_counts, 'changed_samples': len(changes),
    'limitations': [
        'These are post-hoc v4 scores of fixed original generations; the original training rewards and development-selected checkpoint remain unchanged.',
        'Only natural-EOS answers are rescored. Answer timeouts remain false/zero; repetition, context and no-token stops retain unknown correctness.',
        'All requested attempts remain in the denominator. Unknown-outcome bounds are distinct from confidence intervals.',
        'Intervals use paired whole-family bootstrap with 2000 replicates and seed 42; repeated answers are not independent questions.',
        'This comparison uses one training seed and one filtered MATH Level 4 subset. Final-answer equivalence does not validate every reasoning step.'
    ]}
print('[RESCORE 4/5] 80% | original and rescored paired family statistics computed with the frozen calculation', flush=True)

output.parent.mkdir(parents=True, exist_ok=True)
stage = Path(tempfile.mkdtemp(prefix='.rescore-stage-', dir=output.parent))
try:
    for label in ('base', 'rl'):
        destination = stage / label
        destination.mkdir()
        samples = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in rescored[label]['rows']).encode()
        (destination / 'samples.jsonl').write_bytes(samples)
        manifest = {'schema_version': schema, 'complete': True, 'model_label': label, 'posthoc': True,
            'scorer_version': scorer_version, 'scorer_sha256': scorer_sha256, 'identity_sha256': identity_sha256,
            'question_count': base['question_count'], 'n_samples': base['n_samples'], 'total_samples': len(rescored[label]['rows']),
            'samples_sha256': hashlib.sha256(samples).hexdigest(), 'score_fields': list(score_fields),
            'generation_provenance': {'source_path': validated[label]['source_path'],
                'source_manifest_sha256': validated[label]['manifest_sha256'], 'source_samples_sha256': validated[label]['samples_sha256'],
                'source_manifest': validated[label]['manifest'], 'payload_sha256': generation_hashes[label],
                'payload_hash_policy': 'ordered JSONL of all original fields except the six score fields, sorted keys, UTF-8, compact separators',
                'generation_payload_preserved': True}, 'changes': change_counts[label]}
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (stage / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (stage / 'per_question.jsonl').write_text(''.join(json.dumps(question, sort_keys=True, ensure_ascii=False) + '\n' for question in current['per_question']))
    (stage / 'changes.jsonl').write_text(''.join(json.dumps(change, sort_keys=True, ensure_ascii=False) + '\n' for change in changes))
    final_source_files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Frozen source acquired a symlink during rescoring.')
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            final_source_files[str(path.relative_to(root))] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    if final_source_files != source_files:
        raise ValueError('Frozen source changed during rescoring; no completed output will be published.')
    if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != scorer_sha256 or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != identity['runner_sha256']:
        raise ValueError('Scorer or runner changed during rescoring.')
    if audit_bindings != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in audit_paths if path.exists()}:
        raise ValueError('Snapshot audit envelope changed during rescoring.')
    artifacts = {str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest() for path in stage.rglob('*') if path.is_file()}
    ready = {'schema_version': schema, 'complete': True, 'posthoc': True, 'scorer_version': scorer_version,
        'identity': identity, 'identity_sha256': identity_sha256, 'artifacts_sha256': artifacts,
        'source_validation': {'frozen_evaluation_replay': True, 'frozen_selection_replay': True,
            'paired_full_grids': True, 'generation_payloads_preserved': True, 'sources_unchanged_after_scoring': True},
        'question_count': base['question_count'], 'answers_per_model': len(rescored['base']['rows'])}
    (stage / 'ready.json').write_text(json.dumps(ready, indent=2, ensure_ascii=False) + '\n')
    if output.exists() or output.is_symlink():
        raise ValueError('Output appeared during rescoring; refusing to replace it.')
    stage.rename(output)
finally:
    if stage.exists():
        shutil.rmtree(stage)
print(f'✅ [RESCORE 5/5] 100% | {len(changes)} changed answer scores | [SAVED] {output}', flush=True)
