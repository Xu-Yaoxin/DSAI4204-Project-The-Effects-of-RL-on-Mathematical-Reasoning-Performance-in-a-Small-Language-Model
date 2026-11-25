"""Freeze one adapter using the complete development grid, never test scores."""
import argparse
import json
import os
import tempfile
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='configs/math_l4.json')
parser.add_argument('--training-report', required=True)
parser.add_argument('--dev-root', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = json.loads((root / args.config).read_text())
output = (root / args.output).absolute()
if any(p.is_symlink() for p in (output, *output.parents)):
    raise ValueError('Unsafe selection output path.')
output = output.resolve()
if not output.is_relative_to(root / 'outputs'):
    raise ValueError('Selection must remain under outputs/.')
selected = runpy.run_path(str(root / 'scripts/validate_selection.py'), init_globals={
    'SELECTION_CONFIG': config, 'TRAINING_REPORT_PATH': root / args.training_report,
    'DEVELOPMENT_ROOT': root / args.dev_root})['SELECTION_RESULT']
if output.exists():
    if json.loads(output.read_text()) != selected:
        raise ValueError('Existing selection differs; never replace a frozen selection.')
    print(f'[SELECTION CACHED] step={selected["selected_step"]} | {output}', flush=True)
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.selection-', dir=output.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(json.dumps(selected, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'✅ [SELECTION FROZEN] step={selected["selected_step"]} | development only | [SAVED] {output}', flush=True)
