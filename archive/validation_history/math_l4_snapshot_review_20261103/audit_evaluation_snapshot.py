"""Read-only replay and descriptive analysis of an incomplete evaluation snapshot."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import statistics
import time

snapshot = Path('/Users/xu./Downloads/math_l4_snapshot_20261103')
trusted = Path(__file__).resolve().parents[3] / 'math_l4_v1'
output = Path(__file__).resolve().parent
run = snapshot / 'outputs/math_l4_v1_fix1'
cfg = json.loads((snapshot / 'configs/math_l4.json').read_text())
assert (snapshot / 'configs/math_l4.json').read_bytes() == (trusted / 'configs/math_l4.json').read_bytes()
validator = trusted / 'scripts/validate_guarded_sample.py'
validator_code = compile(validator.read_text(), str(validator), 'exec')
started = time.perf_counter()
result = {'snapshot': str(snapshot), 'evaluations': {}, 'limitations': [
    'This is a running-job download, not an atomic server filesystem snapshot.',
    'Partial dev rows are an ordered prefix; descriptive scores do not establish full-dev or held-out-test accuracy.',
    'There is no matched Base evaluation on these dev questions in this snapshot.',
    'Original Base model content is not included; its recorded content inventory identity is checked, not its physical weight bytes.',
    'No model inference, GPU, network, pickle or PT deserialization is used.',
]}

for relative in ['engineering_base', 'dev/checkpoint-128']:
    folder = run / relative
    manifest_bytes = (folder / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    settings = manifest['settings']
    assert manifest['config'] == cfg
    assert manifest['path_base'] == 'code_root'
    assert manifest['synthetic'] is False
    assert manifest['experiment_type'] == 'math_l4_base_rl'
    assert manifest.get('sharding') is None
    required_code = {'evaluate.py', 'verify_answers.py', 'load_model.py', 'fingerprint_model.py',
                     'repetition_guard.py', 'validate_guarded_sample.py', 'doctor.py',
                     'validate_evaluation.py', 'validate_selection.py'}
    assert set(manifest['code_hashes']) == required_code
    for name, expected in manifest['code_hashes'].items():
        assert hashlib.sha256((snapshot / 'scripts' / name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((trusted / 'scripts' / name).read_bytes()).hexdigest() == expected
    protocol = {key: manifest[key] for key in ('dataset_sha256', 'dataset_manifest_sha256', 'n_samples', 'settings', 'code_hashes', 'experiment_type')}
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == manifest['protocol_id']
    assert hashlib.sha256(json.dumps(manifest['model_files'], sort_keys=True).encode()).hexdigest() == manifest['model_identity'] == cfg['model']['expected_identity']
    expected_settings = {key: cfg['evaluation'][key] for key in ('temperature', 'top_p', 'top_k', 'max_new_tokens', 'max_prompt_length', 'seed', 'context_limit', 'length_policy', 'generation_guard')}
    expected_settings.update(prompt_template=cfg['prompt']['template'], dtype='bfloat16', backend='transformers-sdpa',
        batch_size=1, world_size=1, add_special_tokens=True, stopping_rule='tokenizer-eos-or-native-context-or-guard',
        repetition_penalty=1.0, eos_token_id=cfg['model']['expected_eos_token_id'])
    assert all(settings.get(key) == value for key, value in expected_settings.items())
    assert isinstance(settings.get('package_versions'), dict)
    assert manifest['dataset_path'] == str(Path(cfg['data']['output_dir']) / (manifest['split'] + '.jsonl'))
    assert manifest['dataset_manifest_path'] == str(Path(cfg['data']['output_dir']) / 'manifest.json')
    for path_key, digest_key in [('dataset_path', 'dataset_sha256'), ('dataset_manifest_path', 'dataset_manifest_sha256')]:
        assert hashlib.sha256((snapshot / manifest[path_key]).read_bytes()).hexdigest() == manifest[digest_key]
    dataset = [json.loads(line) for line in (snapshot / manifest['dataset_path']).read_text().splitlines() if line.strip()]
    data_manifest = json.loads((snapshot / manifest['dataset_manifest_path']).read_text())
    prepared = data_manifest['files'][Path(manifest['dataset_path']).name]
    assert prepared['sha256'] == manifest['dataset_sha256'] and prepared['rows'] == len(dataset)
    by_id = {row['id']: row for row in dataset}
    assert len(by_id) == len(dataset) == manifest['question_count']
    n_samples = manifest['n_samples']
    expected_n = cfg['evaluation']['engineering_samples'] if manifest['split'] == 'engineering' else cfg['evaluation']['dev_samples']
    assert n_samples == expected_n
    if manifest['model_label'] == 'base':
        assert all(manifest.get(key) is None for key in ('adapter_path', 'adapter_identity', 'adapter_files', 'selection_identity', 'selection_path'))
    else:
        assert manifest['model_label'] == 'rl'
        assert set(manifest['adapter_files']) == {'adapter_config.json', 'adapter_model.safetensors'}
        adapter = snapshot / manifest['adapter_path']
        assert adapter.resolve().is_relative_to(run.resolve())
        digest = hashlib.sha256()
        for filename in ('adapter_config.json', 'adapter_model.safetensors'):
            content = (adapter / filename).read_bytes()
            assert hashlib.sha256(content).hexdigest() == manifest['adapter_files'][filename]
            digest.update(filename.encode())
            digest.update(content)
        assert digest.hexdigest() == manifest['adapter_identity']
        assert manifest.get('selection_identity') is None and manifest.get('selection_path') is None
        receipt = json.loads((adapter / 'resume_state.json').read_text())
        for filename, digest_value in manifest['adapter_files'].items():
            assert receipt['files'][filename]['sha256'] == digest_value
    sample_bytes = (folder / 'samples.jsonl').read_bytes()
    lines = sample_bytes.splitlines(keepends=True)
    trailing_bytes = b''
    if lines and not lines[-1].endswith(b'\n'):
        trailing_bytes = lines.pop()
    rows = [json.loads(line) for line in lines if line.strip()]
    expected_order = [(row['id'], index) for row in dataset for index in range(n_samples)]
    seen = set()
    for row_index, row in enumerate(rows):
        assert row['id'] in by_id and type(row['sample_index']) is int and 0 <= row['sample_index'] < n_samples
        key = (row['id'], row['sample_index'])
        assert key not in seen
        assert all(row.get(field) == value for field, value in by_id[row['id']].items())
        expected_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{row['sample_index']}".encode()).digest()[:4], 'big')
        assert row['sample_seed'] == expected_seed
        namespace = {'SAMPLE': row, 'SETTINGS': settings, '__file__': str(validator)}
        exec(validator_code, namespace)
        assert namespace['VALIDATED'] is True
        seen.add(key)
    ordered_prefix = [(row['id'], row['sample_index']) for row in rows] == expected_order[:len(rows)]
    if manifest['complete']:
        assert not trailing_bytes
        assert hashlib.sha256(sample_bytes).hexdigest() == manifest['samples_sha256']
        assert set(expected_order) == seen and manifest['total_samples'] == len(rows)
    correct = sum(row['correct'] is True for row in rows)
    wrong = sum(row['correct'] is False for row in rows)
    unknown = sum(row['correct'] is None for row in rows)
    grouped = defaultdict(list)
    by_reason = defaultdict(list)
    by_subject = defaultdict(list)
    for row in rows:
        grouped[row['id']].append(row)
        by_reason[row['termination_reason']].append(row)
        by_subject[row['source']['subject']].append(row)
    full = [group for group in grouped.values() if len(group) == n_samples]
    full_rows = [row for group in full for row in group]
    full_correct = sum(row['correct'] is True for row in full_rows)
    full_unknown = sum(row['correct'] is None for row in full_rows)
    covered = sum(any(row['correct'] is True for row in group) for group in full)
    possibly_covered = sum(any(row['correct'] is True or row['correct'] is None for row in group) for group in full)
    duration = sum(row['seconds'] for row in rows)
    result['evaluations'][relative] = summary = {
        'manifest_complete': manifest['complete'], 'model_label': manifest['model_label'], 'split': manifest['split'],
        'requested_questions': len(dataset), 'samples_per_question': n_samples,
        'requested_answers': len(expected_order), 'downloaded_complete_answers': len(rows),
        'complete_questions': len(full), 'partial_questions': len(grouped) - len(full), 'unstarted_questions': len(dataset) - len(grouped),
        'trailing_incomplete_bytes_excluded': len(trailing_bytes), 'ordered_prefix': ordered_prefix,
        'replayed_answer_count': len(rows), 'replay_failures': 0,
        'correct': correct, 'wrong': wrong, 'unknown': unknown,
        'observed_answer_accuracy_lower': correct / len(rows), 'observed_answer_accuracy_upper': (correct + unknown) / len(rows),
        'complete_question_pass1_lower': full_correct / len(full_rows),
        'complete_question_pass1_upper': (full_correct + full_unknown) / len(full_rows),
        'complete_question_pass_at_n_lower': covered / len(full),
        'complete_question_pass_at_n_upper': possibly_covered / len(full),
        'complete_questions_with_confirmed_success': covered,
        'complete_questions_with_possible_success': possibly_covered,
        'scoring_status': dict(Counter(row['status'] for row in rows)),
        'termination': {}, 'subjects': {},
        'answer_seconds_sum': duration, 'mean_answer_seconds': duration / len(rows),
        'median_answer_seconds': statistics.median(row['seconds'] for row in rows),
        'output_tokens_sum': sum(row['output_tokens'] for row in rows),
        'mean_output_tokens': statistics.mean(row['output_tokens'] for row in rows),
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'samples_sha256_snapshot': hashlib.sha256(sample_bytes).hexdigest(),
        'first_question_id': rows[0]['id'], 'last_question_id': rows[-1]['id'],
        'last_sample_index': rows[-1]['sample_index'],
        'adapter_bytes_verified': manifest['model_label'] == 'rl',
    }
    for reason, group in by_reason.items():
        seconds = sum(row['seconds'] for row in group)
        summary['termination'][reason] = {'count': len(group), 'answer_fraction': len(group) / len(rows),
            'seconds_sum': seconds, 'time_fraction': seconds / duration,
            'seconds_mean': seconds / len(group), 'seconds_median': statistics.median(row['seconds'] for row in group),
            'tokens_mean': statistics.mean(row['output_tokens'] for row in group)}
    for subject, group in by_subject.items():
        count_correct = sum(row['correct'] is True for row in group)
        count_unknown = sum(row['correct'] is None for row in group)
        summary['subjects'][subject] = {'answers': len(group), 'questions_started': len({row['id'] for row in group}),
            'questions_in_frozen_split': sum(row['source']['subject'] == subject for row in dataset),
            'correct': count_correct, 'unknown': count_unknown,
            'accuracy_lower': count_correct / len(group), 'accuracy_upper': (count_correct + count_unknown) / len(group),
            'termination': dict(Counter(row['termination_reason'] for row in group))}
    print(f'[REPLAY] {relative}: {len(rows)} verified records, correct={correct}, wrong={wrong}, unknown={unknown}', flush=True)

state = json.loads((run / 'run_state.json').read_text())
dev = result['evaluations']['dev/checkpoint-128']
dev_total = cfg['data']['dev_size'] * cfg['evaluation']['dev_samples'] * cfg['training']['epochs']
test_total = cfg['data']['test_size'] * cfg['evaluation']['n_samples'] * 2
remaining = dev_total + test_total - dev['downloaded_complete_answers']
result['runtime_projection'] = {
    'remaining_answers_at_samples_snapshot': remaining,
    'projected_hours_at_observed_dev_answer_mean': remaining * dev['mean_answer_seconds'] / 3600,
    'observed_dev_answer_seconds_sum': dev['answer_seconds_sum'],
    'state_phase': state['phase'], 'state_phase_elapsed_seconds': state['phase_elapsed_seconds'],
    'state_total_elapsed_seconds': state['elapsed_seconds'],
    'scope': 'Conditional projection using completed checkpoint-128 dev answers only; later subjects, adapters and Base/RL lengths may differ.'
}
if isinstance(state.get('answer'), dict):
    result['runtime_projection']['state_termination_counts'] = state['answer'].get('termination_counts')
    result['runtime_projection']['state_current_question'] = state['answer'].get('id')
    result['runtime_projection']['state_current_sample'] = state['answer'].get('sample_index')
result['audit_elapsed_seconds'] = time.perf_counter() - started
result['all_available_evaluation_records_replayed'] = True
(output / 'evaluation_diagnostic.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
