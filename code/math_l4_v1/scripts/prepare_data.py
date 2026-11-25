"""Freeze independent official-split MATH Level 4 rational, text-only datasets."""
import argparse
from collections import Counter, defaultdict
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
parser.add_argument('--output', default='data/processed')
parser.add_argument('--verify', action='store_true', help='Verify existing data using only the standard library')
parser.add_argument('--offline', action='store_true', help='Refuse missing pinned source downloads')
args = parser.parse_args()
output = (root / args.output).absolute()
if any(p.is_symlink() for p in (output, *output.parents)):
    raise SystemExit('Dataset output must not contain symlinks')
output = output.resolve()
if not output.is_relative_to(root / 'data') or output == root / 'data':
    raise SystemExit('Use a dedicated directory within data/')
sizes = {'engineering': 32, 'pilot': 128, 'train': 512, 'dev': 128, 'test': 200}
seed = 20261102
cfg = {'data': {'output_dir': str(output), 'seed': seed, **{name+'_size': size for name,size in sizes.items()}}}
if args.verify:
    report = runpy.run_path(str(root / 'scripts/doctor.py'), init_globals={'DOCTOR_CONFIG': cfg})['report']
    if not report['data_ready']:
        raise SystemExit(json.dumps(report, indent=2))
    print('✅ [DATA VERIFIED] 100% | train=512 dev=128 test=200 | independent official splits | no model/GPU', flush=True)
    raise SystemExit(0)
revision = '21a5633873b6a120296cce3e2df9d5550074f4a3'
subjects = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
fixed = {'provenance/math_source_tree.json': '2214dac8812eb1702f205b91272e91359e98a3f7224ea577514d6e06745bfa8f',
    'provenance/gold_parser.py': '0e0121cb6994f1a5206e4d1d85fa560dbf350e4fbb9b5904f218ef2d13475ae1',
    'data/raw/math_full/README.md': '2372e22a154ca3ed5865378e270b32d8ed5b615c4592fad42630a731a950bcc0',
    'data/raw/math_full/ORIGINAL_MATH_LICENSE': '8f6401fb0b1fbfcac17aac8d61e4e1b0d2de9aa190d8aa655fedc5fd783b2ace'}
for relative, expected in fixed.items():
    path = root / relative
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Missing or changed pinned provenance: {relative}')
history_path = root / 'provenance/history_exclusions.json'
history = json.loads(history_path.read_text())
excluded = set(history['excluded_normalized_question_sha256'])
if history.get('selection_uses_observed_scores') is not False or len(excluded) != history['known_prior_question_count']:
    raise SystemExit('Invalid history exclusion ledger')
tree = json.loads((root / 'provenance/math_source_tree.json').read_text())
specs = [s for s in tree if s['path'].endswith('.parquet')]
if len(specs) != 14:
    raise SystemExit('Expected seven subjects with two official split files each')
source_files = []
print('[DATA 1/4] 0% | verify pinned public MATH source files', flush=True)
for number, spec in enumerate(specs, 1):
    relative = spec['path']
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise SystemExit('Unsafe pinned source path')
    path = root / 'data/raw/math_full' / relative
    url = f'https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/{revision}/{relative}'
    if path.is_symlink():
        raise SystemExit('Pinned source cannot be a symlink')
    if not path.exists():
        if args.offline or (output / 'manifest.json').exists():
            raise SystemExit(f'Missing pinned source: {path}')
        context = ssl.create_default_context()
        try:
            import certifi
        except ImportError:
            pass
        else:
            context = ssl.create_default_context(cafile=certifi.where())
        print(f'[DOWNLOAD {number}/14] {relative} | bytes={spec["size"]}', flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'DSAI4207-MATH-L4/1'}), context=context, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != spec['lfs']['oid'] or len(content) != spec['size']:
            raise SystemExit('Downloaded source differs from pinned SHA-256 and size')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix('.part').write_bytes(content)
        path.with_suffix('.part').replace(path)
    content = path.read_bytes()
    if len(content) != spec['size'] or hashlib.sha256(content).hexdigest() != spec['lfs']['oid']:
        raise SystemExit(f'Pinned source changed: {path}')
    source_files.append({'path': str(path.relative_to(root)), 'url': url, 'sha256': spec['lfs']['oid'], 'bytes': len(content)})
    print(f'[SOURCE {number}/14] verified {relative}', flush=True)
import pyarrow.parquet as pq
candidates = []
exclusions = Counter()
source_counts = {}
diagram_markers = ('[asy]', '\\includegraphics', '<img', 'diagram', 'figure shown', 'graph shown', 'shown below', 'pictured', 'illustrated')
for spec in source_files:
    path = root / spec['path']
    subject = path.parent.name
    official_split = path.name.split('-', 1)[0]
    if subject not in subjects or official_split not in ('train', 'test'):
        raise SystemExit('Unexpected official subject or split')
    raw_rows = pq.read_table(path).to_pylist()
    source_counts[f'{subject}/{official_split}'] = len(raw_rows)
    for index, row in enumerate(raw_rows):
        if row.get('level') not in ('Level 4', '4', 4):
            exclusions['not_level4'] += 1
            continue
        question = row['problem']
        normalized = ' '.join(question.casefold().split())
        question_hash = hashlib.sha256(normalized.encode()).hexdigest()
        if question_hash in excluded:
            exclusions['prior_question_text'] += 1
            continue
        if any(marker in normalized for marker in diagram_markers):
            exclusions['diagram_or_visual_reference'] += 1
            continue
        matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', row['solution']))
        if not matches:
            exclusions['missing_boxed_source_gold'] += 1
            continue
        start = position = matches[-1].end()
        depth = 1
        while position < len(row['solution']) and depth:
            if row['solution'][position] == '{':
                depth += 1
            elif row['solution'][position] == '}':
                depth -= 1
            position += 1
        if depth:
            exclusions['unclosed_boxed_source_gold'] += 1
            continue
        source_answer = row['solution'][start:position-1].strip()
        source_file = f'{subject}/{path.name}'
        candidates.append({'id': f'math-l4-{official_split}-{subject}-{index:04d}', 'question': question,
            'source_answer': source_answer, 'normalized_question_sha256': question_hash,
            'family_id': 'math-problem-'+question_hash, 'group_id': 'math_l4',
            'group_label': 'MATH Level 4 scalar text-only', 'difficulty_label': 'Level 4',
            'source': {'dataset': 'EleutherAI/hendrycks_math', 'revision': revision, 'split': official_split,
                'source_file': source_file, 'row_index': index, 'original_id': source_file+f'#row={index}',
                'original_id_kind': 'pinned_mirror_file_and_row_not_original_math_filename', 'level': 4, 'subject': subject}})
scores = runpy.run_path(str(root / 'provenance/gold_parser.py'), init_globals={'SCORE_INPUT': [
    {'text': '#### '+r['source_answer'], 'answer': r['source_answer']} for r in candidates]})['SCORE_OUTPUT']
by_question = defaultdict(list)
for record, score in zip(candidates, scores):
    if not score['correct']:
        exclusions['unsupported_non_rational_gold'] += 1
        continue
    record['answer'] = score['extracted']
    by_question[record['normalized_question_sha256']].append(record)
pool = []
for question_hash, items in sorted(by_question.items()):
    if len({r['source']['split'] for r in items}) > 1:
        exclusions['cross_official_split_duplicate'] += len(items)
        continue
    if len({r['answer'] for r in items}) > 1:
        exclusions['conflicting_duplicate_gold'] += len(items)
        continue
    items.sort(key=lambda r: r['id'])
    pool.append(items[0])
    exclusions['within_official_split_duplicate'] += len(items)-1
pool.sort(key=lambda r: r['id'])
if sum(source_counts[k] for k in source_counts if k.endswith('/train')) != 7500 or sum(source_counts[k] for k in source_counts if k.endswith('/test')) != 5000:
    raise SystemExit('Pinned source row totals differ from official 7500/5000 split sizes')
pool_bytes = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in pool).encode()
pool_path = root / 'provenance/eligible_pool.jsonl'
if pool_path.exists() and pool_path.read_bytes() != pool_bytes:
    raise SystemExit('Eligible pool changed; use a new version instead of overwriting')
if not pool_path.exists():
    pool_path.write_bytes(pool_bytes)
print('[DATA 2/4] 40% | rational/text-only/history and duplicate filtering complete', flush=True)
selected = {}
quotas = {}
remaining = {'train': [r for r in pool if r['source']['split']=='train'], 'test': [r for r in pool if r['source']['split']=='test']}
for split in ('train', 'dev', 'test', 'pilot', 'engineering'):
    if split == 'pilot':
        available = selected['train']
    elif split == 'engineering':
        available = selected['pilot']
    else:
        available = remaining['test' if split=='test' else 'train']
    count = sizes[split]
    if len(available) < count:
        raise SystemExit(f'Insufficient independent {split} pool: {len(available)} < {count}')
    by_subject = {subject: [r for r in available if r['source']['subject']==subject] for subject in subjects}
    quotas[split] = {subject: count*len(by_subject[subject])//len(available) for subject in subjects}
    order = sorted(subjects, key=lambda subject: (-(count*len(by_subject[subject]) % len(available)), subject))
    for subject in order[:count-sum(quotas[split].values())]:
        quotas[split][subject] += 1
    chosen = []
    for subject in subjects:
        rng_seed = int.from_bytes(hashlib.sha256(f'{seed}:{split}:{subject}'.encode()).digest()[:8], 'big')
        rng = random.Random(rng_seed)
        candidates = sorted(by_subject[subject], key=lambda r: r['id'])
        chosen.extend(rng.sample(candidates, quotas[split][subject]))
    rng = random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{split}:order'.encode()).digest()[:8], 'big'))
    rng.shuffle(chosen)
    selected[split] = [{**r, 'split': split} for r in chosen]
    if split in ('train', 'dev', 'test'):
        used = {r['id'] for r in chosen}
        official = 'test' if split=='test' else 'train'
        remaining[official] = [r for r in remaining[official] if r['id'] not in used]
file_bytes = {}
files = {}
for split in sizes:
    records = selected[split]
    payload = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in records).encode()
    filename = split+'.jsonl'
    file_bytes[filename] = payload
    files[filename] = {'sha256': hashlib.sha256(payload).hexdigest(), 'rows': len(records),
        'ids': [r['id'] for r in records], 'subject_counts': quotas[split]}
manifest = {'schema_version': 1, 'kind': 'math_level4_splits', 'purpose': 'formal_base_rl_comparison',
    'seed': seed, 'sizes': sizes, 'model_scores_used_for_selection': False, 'files': files,
    'source_provenance': {'status': 'verified', 'repository': 'https://huggingface.co/datasets/EleutherAI/hendrycks_math',
        'revision': revision, 'license': 'Original MATH repository MIT',
        'license_url': 'https://raw.githubusercontent.com/hendrycks/math/985bdc1696e88e8643f081a0ff4719da39f2ae2a/LICENSE',
        'official_train_rows': 7500, 'official_test_rows': 5000, 'source_files': source_files},
    'fixed_provenance_sha256': fixed, 'history_exclusions_path': 'provenance/history_exclusions.json',
    'history_exclusions_sha256': hashlib.sha256(history_path.read_bytes()).hexdigest(),
    'eligible_pool_path': 'provenance/eligible_pool.jsonl', 'eligible_pool_sha256': hashlib.sha256(pool_bytes).hexdigest(),
    'eligible_pool_rows': len(pool), 'eligible_by_official_split': dict(Counter(r['source']['split'] for r in pool)),
    'eligible_subject_counts': {split: dict(Counter(r['source']['subject'] for r in pool if r['source']['split']==split)) for split in ('train','test')},
    'source_rows_by_subject_split': source_counts, 'exclusions': dict(sorted(exclusions.items())),
    'stratification': 'Hamilton largest remainder per subject; train then dev from remaining official train; test from official test; nested pilot and engineering',
    'preparation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'normalization': 'casefold then split/join whitespace', 'diagram_markers': list(diagram_markers),
    'semantic_gold_review': 'not_performed; exact rational parsing of pinned source final boxed answer only',
    'limitations': ['Exact normalized text groups do not establish semantic-template or pretraining independence.',
        'Calibration guided the source/level choice; individual questions were sampled without outcomes.',
        'Only rational, text-only Level4 questions are represented, not the full MATH benchmark.',
        'All prior known question texts were excluded, including the conservative whole MATH500 source pool.',
        'Source solutions and gold must never be included in generation prompts.']}
manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
output.mkdir(parents=True, exist_ok=True)
if (output / 'manifest.json').exists():
    if (output / 'manifest.json').read_bytes()!=manifest_payload or any(not (output/name).is_file() or (output/name).read_bytes()!=payload for name,payload in file_bytes.items()):
        raise SystemExit('Existing dataset or manifest differs; preserve it and prepare a new version')
else:
    if any((output/name).exists() for name in file_bytes):
        raise SystemExit('Partial dataset exists without manifest; preserve it and use a new output')
    for name,payload in file_bytes.items():
        (output/name).write_bytes(payload)
    (output / 'manifest.json').write_bytes(manifest_payload)
print(f'[DATA 3/4] 90% | eligible={manifest["eligible_by_official_split"]} | quotas={json.dumps(quotas,sort_keys=True)}', flush=True)
print(f'✅ [DATA 4/4] 100% | train=512 dev=128 test=200 | [SAVED] {output}', flush=True)
