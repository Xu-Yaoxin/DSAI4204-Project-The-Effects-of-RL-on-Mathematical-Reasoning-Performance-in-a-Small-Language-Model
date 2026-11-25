"""Validate and merge both RL shards into one complete, provenance-bound test grid."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile

bundle_root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root', default=str(bundle_root.parent / 'math_l4_v1'))
parser.add_argument('--run-dir', default='outputs/math_l4_v1_fix1')
parser.add_argument('--output', default='outputs/math_l4_v1_fix1/test_rl_split_merged')
args = parser.parse_args()
experiment = Path(args.experiment_root).expanduser().absolute()
run = (experiment / args.run_dir).absolute()
output = (experiment / args.output).absolute()
if any(p.is_symlink() for path in (experiment, run, output) for p in (path, *path.parents)):
    raise ValueError('Experiment, run and output paths must not contain symlinks.')
experiment, run, output = experiment.resolve(), run.resolve(), output.resolve()
if (not run.is_relative_to(experiment / 'outputs') or run == experiment / 'outputs'
        or not output.is_relative_to(experiment / 'outputs') or output == experiment / 'outputs'
        or output == run or output in run.parents or output.is_relative_to(run / 'rl_shards')):
    raise ValueError('Use a separate merged output under experiment outputs/.')
run_relative = run.relative_to(experiment).as_posix()
config = json.loads((experiment / 'configs/math_l4.json').read_text())
descriptions = []
for index in (0, 1):
    described = runpy.run_path(str(bundle_root / 'scripts/evaluate_shard.py'), init_globals={
        'DESCRIBE_ONLY': True, 'SHARD_ARGUMENTS': ['--experiment-root', str(experiment),
            '--run-dir', run_relative, '--shard-index', str(index)]})
    descriptions.append(described['SHARD_EXTENSION'])
selection = runpy.run_path(str(experiment / 'scripts/validate_selection.py'), init_globals={
    'SELECTION_PATH': run / 'selection.json', 'SELECTION_CONFIG': config})['SELECTION_RESULT']
source_validator = experiment / 'scripts/validate_evaluation.py'
original_validator = source_validator.read_text()
if hashlib.sha256(source_validator.read_bytes()).hexdigest() != '7b33b9962b57cc1a2d83a8508a8332ce459f12d017110e6950de483085757c8c':
    raise ValueError('Unrecognized original evaluation validator.')
validator_replacements = [
    ('if manifest.get("sharding") is not None:\n    raise ValueError("This experiment requires unsharded serial evaluations.")',
     'if manifest.get("sharding") != EXPECTED_SHARDING:\n    raise ValueError("Unexpected declared shard identity.")'),
    ('if len(rows) != len(dataset) * n or manifest.get("total_samples") != len(rows):',
     'if len(rows) != len(ASSIGNED_IDS) * n or manifest.get("total_samples") != len(rows):'),
    ('if row.get("id") not in by_id or type(row.get("sample_index")) is not int or not 0 <= row["sample_index"] < n:',
     'if row.get("id") not in ASSIGNED_IDS or type(row.get("sample_index")) is not int or not 0 <= row["sample_index"] < n:'),
]
shard_validator = original_validator
for original, replacement in validator_replacements:
    if shard_validator.count(original) != 1:
        raise ValueError('The declared shard validation extension no longer matches exactly once.')
    shard_validator = shard_validator.replace(original, replacement, 1)
test_rows = [json.loads(line) for line in (experiment / 'data/processed/test.jsonl').read_text().splitlines() if line.strip()]
by_id = {row['id']: row for row in test_rows}
all_rows = {}
validated = []
sources = []
for index in (0, 1):
    source = run / 'rl_shards' / f'shard_{index}'
    assigned_ids = {row['id'] for position, row in enumerate(test_rows) if position % 2 == index}
    namespace = {'__name__': '__main__', '__file__': str(source_validator),
        'EVALUATION_PATH': source, 'VALIDATE_CONFIG': config, 'ASSIGNED_IDS': assigned_ids,
        'EXPECTED_SHARDING': {'count': 2, 'index': index, 'assignment': 'question_index_modulo'}}
    exec(compile(shard_validator, str(source_validator), 'exec'), namespace)
    checked = namespace['EVALUATION_RESULT']
    manifest = checked['manifest']
    if (manifest.get('execution_extension') != descriptions[index] or manifest['model_label'] != 'rl'
            or manifest['split'] != 'test' or manifest['n_samples'] != 16
            or manifest['selection_identity'] != selection['selection_id']
            or manifest['selection_path'] != run_relative + '/selection.json'
            or manifest['adapter_path'] != selection['adapter_path']
            or manifest['adapter_identity'] != selection['adapter_identity']
            or manifest['adapter_files'] != selection['adapter_files']
            or manifest.get('gpu') != 'NVIDIA GeForce RTX 4090'
            or manifest.get('execution_cpu_threads') != {'intraop': 4, 'interop': 1}):
        raise ValueError(f'Shard {index} execution extension, adapter, GPU or thread contract differs.')
    expected_keys = {(question_id, sample) for question_id in assigned_ids for sample in range(16)}
    keys = {(row['id'], row['sample_index']) for row in checked['rows']}
    if keys != expected_keys or len(keys) != len(checked['rows']):
        raise ValueError('Incomplete or duplicate shard sample grid.')
    for row in checked['rows']:
        key = (row['id'], row['sample_index'])
        if key in all_rows:
            raise ValueError('Overlapping shards cannot be merged.')
        all_rows[key] = row
    validated.append(checked)
    sources.append({'shard_index': index, 'path': str(source.relative_to(experiment)),
        'manifest_sha256': checked['manifest_sha256'], 'samples_sha256': checked['samples_sha256'],
        'execution_extension': manifest['execution_extension']})
    print(f'✅ [MERGE {index+1}/3] Verified shard={index} | questions={len(assigned_ids)} | answers={len(keys)}', flush=True)
left, right = [item['manifest'] for item in validated]
for field in ('schema_version', 'experiment_type', 'synthetic', 'model_label', 'split', 'adapter_path',
        'adapter_identity', 'adapter_files', 'selection_identity', 'selection_path', 'dataset_path',
        'dataset_sha256', 'dataset_manifest_path', 'dataset_manifest_sha256', 'n_samples', 'question_count',
        'config', 'settings', 'code_hashes', 'protocol_id', 'path_base', 'model_identity', 'model_files',
        'gpu', 'groups', 'execution_cpu_threads'):
    if left.get(field) != right.get(field):
        raise ValueError('Unmatched shard protocol: ' + field)
reference_path = bundle_root / 'reference/base_manifest.json'
reference = json.loads(reference_path.read_text())
if reference.get('model_label') != 'base' or reference.get('split') != 'test':
    raise ValueError('The original Base test reference manifest is missing or invalid.')
for field in ('protocol_id', 'settings', 'dataset_sha256', 'dataset_manifest_sha256', 'n_samples',
        'question_count', 'model_identity', 'model_files', 'code_hashes', 'gpu', 'execution_cpu_threads'):
    if reference.get(field) != left.get(field):
        raise ValueError('RL shards do not match the original Base reference: ' + field)
expected_grid = {(row['id'], sample) for row in test_rows for sample in range(16)}
if set(all_rows) != expected_grid or len(all_rows) != 3200:
    raise ValueError('Merged RL test must cover all 200 questions exactly 16 times.')
merge_provenance = {'schema_version': 1, 'kind': 'verified-question-shard-merge-v1',
    'sources': sources, 'selection_identity': selection['selection_id'],
    'merge_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'original_validator_sha256': hashlib.sha256(source_validator.read_bytes()).hexdigest(),
    'extended_validator_sha256': hashlib.sha256(shard_validator.encode()).hexdigest(),
    'base_reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    'question_count': 200, 'samples_per_question': 16, 'total_samples': 3200,
    'sample_content_modified': False, 'original_scientific_protocol_preserved': True}
if output.exists():
    existing = runpy.run_path(str(source_validator), init_globals={
        'EVALUATION_PATH': output, 'VALIDATE_CONFIG': config})['EVALUATION_RESULT']
    if (existing['manifest'].get('execution_extension') != merge_provenance
            or not (output / 'merge_provenance.json').is_file()
            or json.loads((output / 'merge_provenance.json').read_text()) != merge_provenance
            or {(row['id'], row['sample_index']): row for row in existing['rows']} != all_rows):
        raise ValueError('Existing merge differs from these complete source shards; never overwrite.')
    print(f'✅ [MERGE CACHED] 3200 verified answers | {output}', flush=True)
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.rl-shard-merge-', dir=output.parent))
    try:
        sample_bytes = ''.join(json.dumps(all_rows[(row['id'], sample)], ensure_ascii=False) + '\n'
            for row in test_rows for sample in range(16)).encode()
        merged = copy.deepcopy(left)
        for name in ('sharding', 'invocation_seconds', 'invocation_output_tokens', 'invocation_tokens_per_second', 'peak_allocated_gib'):
            merged.pop(name, None)
        merged.update(complete=True, total_samples=3200, samples_sha256=hashlib.sha256(sample_bytes).hexdigest(),
            execution_extension=merge_provenance)
        for name, content in (('samples.jsonl', sample_bytes),
                ('manifest.json', (json.dumps(merged, ensure_ascii=False, indent=2) + '\n').encode()),
                ('merge_provenance.json', (json.dumps(merge_provenance, indent=2) + '\n').encode())):
            with (stage / name).open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        final_check = runpy.run_path(str(source_validator), init_globals={
            'EVALUATION_PATH': stage, 'VALIDATE_CONFIG': config})['EVALUATION_RESULT']
        if len(final_check['rows']) != 3200:
            raise ValueError('Final original-validator check did not return the complete grid.')
        if output.exists():
            raise FileExistsError('Merged output appeared during validation; preserve it.')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'✅ [MERGE 3/3] COMPLETE | original validator passed | 200 questions | 3200 answers | [SAVED] {output}', flush=True)
