"""Create explicitly synthetic outputs to exercise verification, statistics and plotting."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="outputs/validation/demo")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = (root / args.output).resolve()
if output.exists():
    raise FileExistsError(f"Synthetic output already exists: {output}")
output.mkdir(parents=True)
fixture_rows = [{"id": f"fixture-{i}", "question": f"What is {i} plus 10?", "answer": str(i+10),
                 "split": "test", "family_id": f"fixture-{i}", "variant_type": "original"} for i in range(8)]
dataset_hash = hashlib.sha256(json.dumps(fixture_rows, sort_keys=True).encode()).hexdigest()
settings = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 512, "seed": 42,
            "prompt_template": "Synthetic fixture only: {question}", "dtype": "none", "backend": "synthetic-fixture"}
for label, counts in {"base": [16, 1, 0, 0, 8, 0, 2, 0], "rl": [16, 0, 1, 0, 12, 0, 8, 0]}.items():
    directory = output / label
    directory.mkdir()
    samples = []
    for question_index, row in enumerate(fixture_rows):
        for sample_index in range(16):
            value = int(row["answer"]) if sample_index < counts[question_index] else int(row["answer"]) + 1
            completion = f"#### {value}" if sample_index % 7 else str(value)
            score = runpy.run_path(str(root / "scripts/verify_answers.py"), init_globals={
                "SCORE_INPUT": [{"text": completion, "answer": row["answer"], "truncated": False}]})["SCORE_OUTPUT"][0]
            samples.append({**row, **score, "sample_index": sample_index, "text": completion, "output_tokens": 4})
    (directory / "samples.jsonl").write_text("".join(json.dumps(item) + "\n" for item in samples))
    manifest = {"schema_version": 1, "model_label": label, "n_samples": 16, "split": "test",
                "dataset_sha256": dataset_hash, "protocol_id": "synthetic-fixture-v1",
                "model_identity": "fixture-base-no-model-loaded", "adapter_identity": "fixture-rl" if label == "rl" else None,
                "synthetic": True, "complete": True, "settings": settings, "question_count": len(fixture_rows)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
subprocess.run([sys.executable, str(root / "scripts/analyze.py"), "--base", str(output / "base"),
    "--rl", str(output / "rl"), "--output", str(output / "analysis"), "--bootstrap", "2000"], check=True)
if args.plot:
    subprocess.run([sys.executable, str(root / "scripts/plot_results.py"), "--summary", str(output / "analysis/summary.json"),
                    "--output", str(output / "analysis/overview.png")], check=True)
print(f"✅ [SYNTHETIC SMOKE ONLY] {output} | no model loaded, no training performed")
