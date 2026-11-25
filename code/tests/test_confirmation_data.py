"""Check real frozen source selection and reject tampering using isolated copies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
data = root / 'data/boundary_confirm_v2'
manifest = json.loads((data / 'manifest.json').read_text())
question_bytes = (data / 'questions.jsonl').read_bytes()
rows = [json.loads(line) for line in question_bytes.decode().split('\n') if line.strip()]
checks = 0
assert manifest['files']['questions.jsonl'] == {'sha256': hashlib.sha256(question_bytes).hexdigest(), 'rows': 200}
assert Counter(row['group_id'] for row in rows) == {'gsm_main_confirm': 100, 'math_l4_confirm': 100}
assert len({row['id'] for row in rows}) == 200 and len({' '.join(row['question'].casefold().split()) for row in rows}) == 200
checks += 3
old_rows = []
for relative in manifest['history_files_sha256']:
    path = root / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['history_files_sha256'][relative]
    if path.suffix == '.jsonl':
        old_rows += [json.loads(line) for line in path.read_text().split('\n') if line.strip()]
old_text = {' '.join(row['question'].casefold().split()) for row in old_rows}
assert not old_text.intersection(' '.join(row['question'].casefold().split()) for row in rows)
assert not set(manifest['excluded_gsm_original_ids']).intersection(manifest['selected_gsm_original_ids'])
checks += 2
family_counts = Counter(row['family_id'] for row in rows if row['group_id'] == 'gsm_main_confirm')
assert len(family_counts) == 64 and Counter(family_counts.values()) == {1: 28, 2: 36}
assert all(row['source']['split'] == 'test' and row['source']['level'] == 4 for row in rows if row['group_id'] == 'math_l4_confirm')
assert all(row['source']['revision'] == '21a5633873b6a120296cce3e2df9d5550074f4a3' for row in rows if row['group_id'] == 'math_l4_confirm')
assert manifest['model_scores_used_for_selection'] is False and manifest['source_group_choice_informed_by_prior_screen'] is True
checks += 4
scores = runpy.run_path(str(root / 'scripts/verify_answers.py'), init_globals={'SCORE_INPUT': [{'text': '#### ' + row['answer'], 'answer': row['answer']} for row in rows]})['SCORE_OUTPUT']
assert len(scores) == 200 and all(row['correct'] for row in scores)
checks += 1
for source in manifest['sources']:
    assert hashlib.sha256((data / source['path']).read_bytes()).hexdigest() == source['sha256']
checks += 1
with tempfile.TemporaryDirectory(prefix='dsai-confirm-data-') as temporary:
    fixture = Path(temporary) / 'code'
    (fixture / 'scripts').mkdir(parents=True)
    for filename in ('prepare_confirmation_data.py', 'verify_answers.py'):
        shutil.copyfile(root / 'scripts' / filename, fixture / 'scripts' / filename)
    for relative in manifest['history_files_sha256']:
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, path)
    target = fixture / 'data/boundary_confirm_v2'
    shutil.copytree(data, target)
    command = [sys.executable, str(fixture / 'scripts/prepare_confirmation_data.py'), '--verify']
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0 and 'VERIFIED' in result.stdout, result.stdout + result.stderr
    assert (target / 'questions.jsonl').read_bytes() == question_bytes
    checks += 1
    original = (target / 'questions.jsonl').read_bytes()
    (target / 'questions.jsonl').write_bytes(original.replace(b'question', b'questioN', 1))
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and 'Existing confirmation content or provenance changed' in result.stderr
    (target / 'questions.jsonl').write_bytes(original)
    checks += 1
    raw = target / 'raw/math_full/algebra/test-00000-of-00001.parquet'
    original_raw = raw.read_bytes()
    raw.write_bytes(original_raw + b'changed')
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and 'Pinned source hash mismatch' in result.stderr
    raw.write_bytes(original_raw)
    checks += 1
    history = fixture / 'data/processed/train.jsonl'
    original_history = history.read_bytes()
    history.write_bytes(original_history + b'\n')
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and 'Existing confirmation content or provenance changed' in result.stderr
    history.write_bytes(original_history)
    checks += 1
    previous = fixture / 'data/boundary_v1/questions.jsonl'
    old_content = previous.read_bytes()
    previous.write_bytes(old_content + b'\n')
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and 'Previous boundary dataset differs' in result.stderr
    previous.write_bytes(old_content)
    checks += 1
    raw.unlink()
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and 'Missing frozen source' in result.stderr
    raw.write_bytes(original_raw)
    checks += 1
    frozen_manifest = (target / 'manifest.json').read_bytes()
    (target / 'manifest.json').unlink()
    (target / 'questions.jsonl').unlink()
    result = subprocess.run([sys.executable, str(fixture / 'scripts/prepare_confirmation_data.py'), '--offline'], text=True, capture_output=True)
    assert result.returncode == 0 and (target / 'questions.jsonl').read_bytes() == question_bytes, result.stderr
    assert (target / 'manifest.json').read_bytes() == frozen_manifest
    checks += 1
    actual = subprocess.run([sys.executable, '-c', "import json; from pathlib import Path; import pyarrow.parquet as pq; p=Path('data/boundary_confirm_v2'); a=[json.loads(x) for x in (p/'questions.jsonl').read_text().splitlines()]; g=[json.loads(x) for x in (p/'raw/gsm_symbolic/main.jsonl').read_text().splitlines()]; m={};\nfor r in a:\n s=r['source']\n if r['group_id']=='gsm_main_confirm':\n  assert r['question']==g[s['row_index']]['question'] and r['source_answer']==g[s['row_index']]['answer']\n else:\n  if s['source_file'] not in m:m[s['source_file']]=pq.read_table(p/'raw/math_full'/s['source_file']).to_pylist()\n  assert r['question']==m[s['source_file']][s['row_index']]['problem']\nprint('VERBATIM200')"], cwd=fixture, text=True, capture_output=True)
    assert actual.returncode == 0 and 'VERBATIM200' in actual.stdout, actual.stderr
    checks += 1
print(f'✅ [CONFIRMATION DATA TEST 100%] {checks} source, history, family, scalar, determinism and tamper checks; no model/GPU/network')
