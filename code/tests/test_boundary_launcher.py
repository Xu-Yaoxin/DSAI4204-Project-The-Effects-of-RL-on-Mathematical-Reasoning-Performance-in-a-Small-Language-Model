"""Check the isolated shell launcher without GPUs, weights or network access."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="boundary-launch-") as temporary:
    parent = Path(temporary)
    package = parent / "probe with spaces"
    (package / "scripts").mkdir(parents=True)
    shutil.copy2(root / "run_boundary.sh", package / "run_boundary.sh")
    shutil.copy2(root / "run_boundary_parallel.sh", package / "run_boundary_parallel.sh")
    evaluator = package / "scripts/evaluate_boundary.py"
    evaluator.write_text('import json,sys\nassert sys.flags.no_site == 1\nprint(json.dumps(sys.argv[1:]))\n')
    environment = dict(os.environ, RLVR_PYTHON=sys.executable)
    for arguments, quick in ((["--plan"], False), (["--plan", "--quick"], True)):
        completed = subprocess.run(["bash", str(package / "run_boundary.sh"), *arguments], cwd=parent, env=environment, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
        actual = json.loads(completed.stdout)
        assert actual[actual.index("--config") + 1] == "configs/boundary_v1.json"
        assert actual[actual.index("--output") + 1] == ("outputs/boundary_v1_quick/base" if quick else "outputs/boundary_v1/base")
        assert "--plan" in actual and "--resume" not in actual
        assert ("--samples" in actual) is quick
        if quick:
            assert actual[actual.index("--samples") + 1] == "1"
        assert not (package / "outputs").exists()
    supervisor = package / "scripts/run_boundary_parallel.py"
    supervisor.write_text('import json,sys\nassert sys.flags.no_site == 1\nprint(json.dumps(sys.argv[1:]))\n')
    for extra in ([], ["--quick"], ["--sequential"]):
        completed = subprocess.run(["bash", str(package / "run_boundary_parallel.sh"), "--plan", *extra], cwd=parent, env=environment, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
        actual = json.loads(completed.stdout)
        assert actual[actual.index("--output") + 1] == ("outputs/boundary_v1_parallel_quick" if "--quick" in extra else "outputs/boundary_v1_parallel")
        assert "--plan" in actual
        assert ("--samples" in actual) is ("--quick" in extra)
        assert ("--workers" in actual) is ("--sequential" in extra)
        if "--workers" in actual:
            assert actual[actual.index("--workers") + 1] == "1"
        assert not (package / "outputs").exists()
    completed = subprocess.run(["bash", str(package / "run_boundary.sh"), "--unknown"], cwd=parent, env=environment, text=True, capture_output=True)
    assert completed.returncode == 2 and "Usage:" in completed.stderr
    (package / "bundle_manifest.json").write_text(json.dumps({"files_sha256": {"scripts/evaluate_boundary.py": hashlib.sha256(evaluator.read_bytes()).hexdigest()}}))
    completed = subprocess.run(["bash", str(package / "run_boundary.sh"), "--plan"], cwd=parent, env=environment, text=True, capture_output=True)
    assert completed.returncode == 0 and "[BUNDLE VERIFIED]" in completed.stdout, completed.stderr
    evaluator.write_text('raise SystemExit("must not execute changed code")\n')
    completed = subprocess.run(["bash", str(package / "run_boundary.sh"), "--plan"], cwd=parent, env=environment, text=True, capture_output=True)
    assert completed.returncode != 0 and "Bundle file changed" in completed.stderr
    assert "must not execute changed code" not in completed.stderr
print("[PASS] 8 boundary launcher checks | Bash array portability, relocated paths, quick isolation and upload integrity | no GPU/network")
