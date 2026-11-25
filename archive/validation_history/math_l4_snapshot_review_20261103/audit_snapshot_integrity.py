"""Read-only verification of a downloaded, still-running experiment snapshot."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

source = Path('/Users/xu./Downloads/math_l4_snapshot_20261103')
trusted = Path(__file__).resolve().parents[3] / 'math_l4_v1'
output = Path(__file__).resolve().parent
run = source / 'outputs/math_l4_v1_fix1'
inventory = {}
failures = []
for path in sorted(source.rglob('*')):
    if path.is_symlink():
        failures.append({'kind': 'symlink', 'path': str(path.relative_to(source))})
        continue
    if not path.is_file():
        continue
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    relative = path.relative_to(source).as_posix()
    inventory[relative] = {'bytes': after.st_size, 'sha256': digest.hexdigest(), 'mtime_ns': after.st_mtime_ns}
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        failures.append({'kind': 'changed_during_read', 'path': relative})

bundle = json.loads((source / 'bundle_manifest.json').read_text())
local_bundle = json.loads((trusted / 'bundle_manifest.json').read_text())
if bundle != local_bundle:
    failures.append({'kind': 'bundle_differs_from_trusted_local'})
for relative, expected in bundle['files_sha256'].items():
    if inventory.get(relative, {}).get('sha256') != expected:
        failures.append({'kind': 'bundle_hash', 'path': relative})
    local = trusted / relative
    if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != expected:
        failures.append({'kind': 'trusted_source_hash', 'path': relative})

cfg = json.loads((source / 'configs/math_l4.json').read_text())
state = json.loads((run / 'run_state.json').read_text())
if state['identity'] != {
    'config_sha256': inventory['configs/math_l4.json']['sha256'],
    'bundle_sha256': inventory['bundle_manifest.json']['sha256'],
}:
    failures.append({'kind': 'pipeline_identity'})

splits = {}
for split in ('train', 'dev', 'test', 'engineering', 'pilot'):
    rows = [json.loads(line) for line in (source / f'data/processed/{split}.jsonl').read_text().splitlines()]
    ids = {row['id'] for row in rows}
    hashes = {row['normalized_question_sha256'] for row in rows}
    if len(ids) != len(rows) or len(hashes) != len(rows):
        failures.append({'kind': 'duplicate_data', 'split': split})
    required_source = 'test' if split == 'test' else 'train'
    if any(row['source']['split'] != required_source or row['split'] != split for row in rows):
        failures.append({'kind': 'split_metadata', 'split': split})
    splits[split] = {'count': len(rows), 'ids': ids, 'hashes': hashes}
for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
    if splits[left]['hashes'] & splits[right]['hashes']:
        failures.append({'kind': 'split_leakage', 'splits': [left, right]})
if not splits['engineering']['ids'] <= splits['pilot']['ids'] <= splits['train']['ids']:
    failures.append({'kind': 'nested_training_subsets'})

checkpoint_results = []
completion_count = 0
for mode, expected_steps in [('engineering', 2), ('pilot', 96), ('formal', 384)]:
    training = run / mode
    report = json.loads((training / 'report.json').read_text())
    manifest = json.loads((training / 'manifest.json').read_text())
    if report['status'] != 'completed' or report['optimizer_steps'] != expected_steps or not all(report['gates'].values()):
        failures.append({'kind': 'training_status', 'mode': mode})
    protocol = manifest['training_protocol']
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if protocol_hash != report['config_training_hash'] or manifest['config'] != cfg:
        failures.append({'kind': 'training_protocol', 'mode': mode})
    for relative, expected in report['completion_files'].items():
        key = (training / relative).relative_to(source).as_posix()
        if inventory.get(key, {}).get('sha256') != expected:
            failures.append({'kind': 'completion_hash', 'mode': mode, 'path': key})
        completion_count += 1
    for split, expected in report['split_sha256'].items():
        if inventory[f'data/processed/{split}.jsonl']['sha256'] != expected:
            failures.append({'kind': 'training_split_hash', 'mode': mode, 'split': split})
    for checkpoint in sorted(training.glob('checkpoint-*')):
        receipt = json.loads((checkpoint / 'resume_state.json').read_text())
        step = int(checkpoint.name.removeprefix('checkpoint-'))
        trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
        if receipt['step'] != step or receipt['binding'] != manifest['resume_binding'] or trainer['global_step'] != step or trainer['max_steps'] != expected_steps:
            failures.append({'kind': 'checkpoint_binding', 'path': checkpoint.name, 'mode': mode})
        observed = {p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob('*') if p.is_file() and p.name != 'resume_state.json'}
        if observed != set(receipt['files']):
            failures.append({'kind': 'checkpoint_inventory', 'path': str(checkpoint)})
        for relative, record in receipt['files'].items():
            entry = inventory.get((checkpoint / relative).relative_to(source).as_posix(), {})
            if entry.get('bytes') != record['size'] or entry.get('sha256') != record['sha256']:
                failures.append({'kind': 'checkpoint_file', 'path': str(checkpoint / relative)})
        for rank, audit in receipt['audits'].items():
            snapshot = checkpoint / audit['snapshot']
            raw = snapshot.read_bytes()
            if len(raw) != audit['bytes'] or hashlib.sha256(raw).hexdigest() != audit['sha256'] or len(raw.splitlines()) != audit['groups']:
                failures.append({'kind': 'checkpoint_audit', 'path': str(snapshot)})
            with (training / audit['target']).open('rb') as stream:
                prefix = stream.read(audit['bytes'])
            if prefix != raw:
                failures.append({'kind': 'checkpoint_audit_prefix', 'path': str(snapshot)})
        checkpoint_results.append({'mode': mode, 'step': step, 'sealed_files': len(receipt['files'])})

for relative, entry in inventory.items():
    stat = (source / relative).stat()
    if (stat.st_size, stat.st_mtime_ns) != (entry['bytes'], entry['mtime_ns']):
        failures.append({'kind': 'source_changed', 'path': relative})

result = {
    'source': str(source), 'audited_utc': datetime.now(timezone.utc).isoformat(),
    'passed': not failures, 'failures': failures, 'downloaded_files': len(inventory),
    'downloaded_bytes': sum(row['bytes'] for row in inventory.values()),
    'frozen_bundle_files_verified': len(bundle['files_sha256']),
    'training_completion_artifacts_verified': completion_count,
    'sealed_checkpoints': checkpoint_results,
    'split_counts': {name: row['count'] for name, row in splits.items()},
    'snapshot_phase': state['phase'], 'snapshot_phase_seconds': state['phase_elapsed_seconds'],
    'snapshot_pipeline_seconds': state['elapsed_seconds'],
    'source_unchanged_during_audit': not any(row['kind'] in ['source_changed', 'changed_during_read'] for row in failures),
    'limits': ['Hashes validate saved artifacts, not semantic correctness of gold labels.',
               'Base weights were not downloaded or rehashed. No GPU execution or optimizer deserialization.',
               'Live output files were copied at different times and do not form an atomic remote snapshot.'],
}
(output / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
(output / 'integrity_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'sealed_checkpoints'}, indent=2))
print('Sealed checkpoints verified:', len(checkpoint_results))
if failures:
    raise SystemExit(1)
