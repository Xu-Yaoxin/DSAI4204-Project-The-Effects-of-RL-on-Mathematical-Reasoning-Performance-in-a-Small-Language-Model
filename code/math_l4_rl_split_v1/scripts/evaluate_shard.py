"""Run a declared two-way scheduling extension over the unchanged frozen evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

bundle_root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root', default=str(bundle_root.parent / 'math_l4_v1'))
parser.add_argument('--run-dir', default='outputs/math_l4_v1_fix1')
parser.add_argument('--shard-index', type=int, choices=(0, 1), required=True)
parser.add_argument('--plan', action='store_true')
parser.add_argument('--cache-only', action='store_true')
args = parser.parse_args(globals().get('SHARD_ARGUMENTS'))
if args.plan and args.cache_only:
    parser.error('--plan and --cache-only are mutually exclusive.')
experiment = Path(args.experiment_root).expanduser().absolute()
if any(path.is_symlink() for path in (experiment, *experiment.parents)):
    raise ValueError('Experiment path must not contain symlinks.')
experiment = experiment.resolve()
run = (experiment / args.run_dir).absolute()
if any(path.is_symlink() for path in (run, *run.parents)):
    raise ValueError('Run path must not contain symlinks.')
run = run.resolve()
if not run.is_relative_to(experiment / 'outputs') or run == experiment / 'outputs':
    raise ValueError('Run must be a dedicated directory under experiment outputs/.')
run_relative = run.relative_to(experiment).as_posix()
source_manifest_path = experiment / 'bundle_manifest.json'
source_manifest_bytes = source_manifest_path.read_bytes()
if hashlib.sha256(source_manifest_bytes).hexdigest() != 'deebc68649df15c1280e36dad8a5d98f3a900bf292f0cde60976e067ca84ba32':
    raise ValueError('Expected the original math-l4-1.0.1 frozen bundle.')
source_manifest = json.loads(source_manifest_bytes)
for relative, expected in source_manifest['files_sha256'].items():
    path = experiment / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Unsafe frozen source path.')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen source changed or is missing: ' + relative)
config = json.loads((experiment / 'configs/math_l4.json').read_text())
selection_path = run / 'selection.json'
if selection_path.is_symlink() or not selection_path.is_file():
    raise ValueError('Download the frozen selection.json before planning or running.')
selection = json.loads(selection_path.read_text())
reference_selection = json.loads((bundle_root / 'reference/selection.json').read_text())
if selection != reference_selection:
    raise ValueError('Selection differs from the frozen migration reference; do not reselect on test outcomes.')
selection_payload = {key: value for key, value in selection.items() if key != 'selection_id'}
if (selection.get('complete') is not True or selection.get('selected_step') != 384
        or selection.get('test_outcomes_used') is not False or selection.get('selection_split') != 'dev'
        or selection.get('model_identity') != config['model']['expected_identity']
        or selection.get('config_identity') != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        or selection.get('prepared_manifest_sha256') != source_manifest['data_manifest_sha256']
        or selection.get('adapter_path') != run_relative + '/formal/checkpoint-384'
        or selection.get('selection_id') != hashlib.sha256(json.dumps(selection_payload, sort_keys=True).encode()).hexdigest()):
    raise ValueError('Frozen selection identity/configuration differs from the selected checkpoint-384 run.')
test_bytes = (experiment / 'data/processed/test.jsonl').read_bytes()
test_rows = [json.loads(line) for line in test_bytes.splitlines() if line.strip()]
if len(test_rows) != 200 or config['evaluation']['n_samples'] != 16:
    raise ValueError('The declared comparison requires exactly 200 test questions and 16 samples each.')
assigned_ids = [row['id'] for index, row in enumerate(test_rows) if index % 2 == args.shard_index]
source_path = experiment / 'scripts/evaluate.py'
original_source = source_path.read_text()
original_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
if original_sha256 != '21321104395abad6bac1e37ab2ee5dc608de4928e6f43575c8c3550532bac244':
    raise ValueError('Unrecognized original evaluator; do not adapt a changed implementation silently.')
replacements = [
    ('type=int, choices=(1,), default=1, help="Serial only; one complete split per invocation"',
     'type=int, choices=(1, 2), default=1, help="Declared two-way question scheduling extension"'),
    ('work_rows = rows\n', 'work_rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]\n'),
    ('existing = {}\nprevious_manifest = None\n',
     'expected_metadata["execution_extension"] = EXECUTION_EXTENSION\nexisting = {}\nprevious_manifest = None\n'),
]
transformed_source = original_source
for original, replacement in replacements:
    if transformed_source.count(original) != 1:
        raise ValueError('The declared scheduling patch no longer matches exactly once.')
    transformed_source = transformed_source.replace(original, replacement, 1)
SHARD_EXTENSION = {
    'schema_version': 1, 'kind': 'math-l4-two-machine-question-shards-v1',
    'original_evaluator_sha256': original_sha256,
    'transformed_evaluator_sha256': hashlib.sha256(transformed_source.encode()).hexdigest(),
    'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'patches': ['allow_two_shards', 'question_index_modulo', 'bind_execution_extension'],
    'assignment': 'question_index_modulo', 'shard_count': 2, 'shard_index': args.shard_index,
    'assigned_question_ids_sha256': hashlib.sha256(json.dumps(assigned_ids, separators=(',', ':')).encode()).hexdigest(),
    'assigned_questions': len(assigned_ids), 'samples_per_question': 16,
    'dataset_sha256': hashlib.sha256(test_bytes).hexdigest(),
    'selection_identity': selection['selection_id'],
    'original_bundle_sha256': hashlib.sha256(source_manifest_bytes).hexdigest(),
    'generation_scoring_and_seed_code_unchanged': True,
}
if not globals().get('DESCRIBE_ONLY', False):
    print('[EXECUTION EXTENSION] ' + json.dumps(SHARD_EXTENSION, sort_keys=True), flush=True)
    print(f'[SHARD PLAN] index={args.shard_index}/2 | questions={len(assigned_ids)}/200 | responses=1600 | selected=checkpoint-384', flush=True)
    sys.argv = [str(source_path), '--config', 'configs/math_l4.json', '--model', 'rl', '--split', 'test',
        '--samples', '16', '--num-shards', '2', '--shard-index', str(args.shard_index),
        '--selection', run_relative + '/selection.json',
        '--output', run_relative + f'/rl_shards/shard_{args.shard_index}', '--resume']
    if args.plan:
        sys.argv.append('--plan')
    if args.cache_only:
        sys.argv.append('--cache-only')
    exec(compile(transformed_source, str(source_path), 'exec'), {
        '__name__': '__main__', '__file__': str(source_path), 'EXECUTION_EXTENSION': SHARD_EXTENSION})
