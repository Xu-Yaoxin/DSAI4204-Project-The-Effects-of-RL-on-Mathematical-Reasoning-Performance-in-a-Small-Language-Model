"""Offline fault-injection checks for pinned, atomic GSM8K acquisition."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import ssl
import subprocess
import sys
import tempfile
import urllib.error
from unittest.mock import DEFAULT, patch


root = Path(__file__).resolve().parents[1]
script = root / "scripts/download_data.py"
revision = "3101c7d5072418e28b9008a6636bde82a006892c"
train_bytes = "".join(json.dumps({"question": f"Training fixture {index}:\u2028what is {index} plus 1?", "answer": f"Add one.\n#### {index + 1}"}, ensure_ascii=False) + "\n" for index in range(7473)).encode("utf-8")
test_bytes = "".join(json.dumps({"question": f"Testing fixture {index}:\u0085what is {index} plus 2?", "answer": f"Add two.\n#### {index + 2}"}, ensure_ascii=False) + "\n" for index in range(1319)).encode("utf-8")
license_bytes = b"MIT License\n\nCopyright (c) 2021 OpenAI\n\nSynthetic transport-test fixture.\n"
checks = 0

with tempfile.TemporaryDirectory(prefix="dsai4207-download-test-") as directory:
    temporary_root = Path(directory)
    raw_dir = temporary_root / "raw"
    config_path = temporary_root / "config.json"
    config = {"data": {"raw_train": str(raw_dir / "train.jsonl"), "raw_test": str(raw_dir / "test.jsonl"),
                       "output_dir": str(temporary_root / "processed"), "train_size": 8, "dev_size": 4, "test_size": 4,
                       "pilot_size": 4, "engineering_size": 2, "seed": 42}}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    argv = [str(script), "--config", str(config_path)]
    with patch.object(sys, "argv", argv), patch("urllib.request.urlopen", side_effect=[io.BytesIO(train_bytes), io.BytesIO(test_bytes), io.BytesIO(license_bytes)]) as network, contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script))
    assert network.call_count == 3
    for call in network.call_args_list:
        assert f"/{revision}/" in call.args[0].full_url
        assert call.kwargs["context"].verify_mode == ssl.CERT_REQUIRED
        assert call.kwargs["context"].check_hostname is True
    snapshot = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_dir.iterdir()}
    manifest = json.loads(snapshot["source_manifest.json"][0])
    assert manifest["revision"] == revision and manifest["license"] == "MIT"
    assert manifest["files"]["train"]["rows"] == 7473 and manifest["files"]["test"]["rows"] == 1319
    for key, filename in [("train", "train.jsonl"), ("test", "test.jsonl"), ("license", "LICENSE")]:
        assert manifest["files"][key]["sha256"] == hashlib.sha256(snapshot[filename][0]).hexdigest()
    checks += 1

    with patch.object(sys, "argv", argv + ["--offline"]), patch("urllib.request.urlopen", side_effect=AssertionError("Offline rerun attempted network")) as network, contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script))
    assert network.call_count == 0
    assert snapshot == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_dir.iterdir()}
    checks += 1

    (raw_dir / "LICENSE").unlink()
    with patch.object(sys, "argv", argv), patch("urllib.request.urlopen", side_effect=[io.BytesIO(license_bytes)]) as network, contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script))
    assert network.call_count == 1 and network.call_args.args[0].full_url.endswith("/LICENSE")
    assert (raw_dir / "LICENSE").read_bytes() == license_bytes
    assert all((raw_dir / name).stat().st_mtime_ns == snapshot[name][1] for name in ["train.jsonl", "test.jsonl", "source_manifest.json"])
    checks += 1

    (raw_dir / "train.jsonl").write_bytes(train_bytes + b"\n")
    with patch.object(sys, "argv", argv), patch("urllib.request.urlopen", side_effect=AssertionError("Tamper detection attempted network")) as network:
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert "Raw-data hash mismatch" in str(error), str(error)
        else:
            raise AssertionError("Raw-data tampering was accepted")
    assert network.call_count == 0 and (raw_dir / "train.jsonl").read_bytes() == train_bytes + b"\n"
    (raw_dir / "train.jsonl").write_bytes(train_bytes)
    checks += 1

    with patch.object(sys, "argv", argv + ["--revision", "0" * 40]), patch("urllib.request.urlopen", side_effect=AssertionError("Revision conflict attempted network")):
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert "Provenance conflict" in str(error), str(error)
        else:
            raise AssertionError("Revision change overwrote existing data")
    assert json.loads((raw_dir / "source_manifest.json").read_text())["revision"] == revision
    checks += 1

    prepare_command = [sys.executable, str(root / "scripts/prepare_data.py"), "--config", str(config_path)]
    completed = subprocess.run(prepare_command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    prepared = json.loads((temporary_root / "processed/manifest.json").read_text())
    assert prepared["source_provenance"]["status"] == "verified"
    assert prepared["source_provenance"]["source_manifest_sha256"] == hashlib.sha256((raw_dir / "source_manifest.json").read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (temporary_root / "processed/train.jsonl").read_text().split("\n") if line.strip()]
    assert len(rows) == 8 and all("\u2028" in row["question"] for row in rows)
    checks += 1
    (raw_dir / "test.jsonl").write_bytes(test_bytes + b"\n")
    completed = subprocess.run(prepare_command, text=True, capture_output=True)
    assert completed.returncode != 0 and "source hash/count/URL mismatch" in completed.stderr, completed.stderr
    (raw_dir / "test.jsonl").write_bytes(test_bytes)
    checks += 1

    for scenario in ["network_failure", "install_failure", "invalid_json", "invalid_count", "unmanifested_conflict", "offline_missing"]:
        case_root = temporary_root / scenario
        case_raw = case_root / "raw"
        case_config = {"data": {**config["data"], "raw_train": str(case_raw / "train.jsonl"), "raw_test": str(case_raw / "test.jsonl"), "output_dir": str(case_root / "processed")}}
        case_root.mkdir()
        case_config_path = case_root / "config.json"
        case_config_path.write_text(json.dumps(case_config), encoding="utf-8")
        case_argv = [str(script), "--config", str(case_config_path)]
        side_effect = [io.BytesIO(train_bytes), io.BytesIO(test_bytes), io.BytesIO(license_bytes)]
        expected_type = SystemExit
        expected_text = ""
        if scenario == "network_failure":
            side_effect = [io.BytesIO(train_bytes), urllib.error.URLError("simulated interrupted response")]
            expected_text = "No dataset files were installed"
        elif scenario == "install_failure":
            expected_type = OSError
            expected_text = "simulated disk failure"
        elif scenario == "invalid_json":
            side_effect = [io.BytesIO(b"not JSON\n")]
            expected_text = "Invalid official train JSONL"
        elif scenario == "invalid_count":
            side_effect = [io.BytesIO(b'{"question":"Toy?", "answer":"#### 1"}\n')]
            expected_text = "Unexpected row count"
        elif scenario == "unmanifested_conflict":
            case_raw.mkdir()
            (case_raw / "train.jsonl").write_bytes(b"user content must survive\n")
            expected_text = "Refusing to overwrite"
        else:
            case_argv.append("--offline")
            side_effect = AssertionError("Offline missing-source check attempted network")
            expected_text = "Offline verification requires complete"
        original_link = os.link
        link_effect = [DEFAULT, OSError("simulated disk failure")] if scenario == "install_failure" else None
        with patch.object(sys, "argv", case_argv), patch("urllib.request.urlopen", side_effect=side_effect), patch("os.link", wraps=original_link, side_effect=link_effect), contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script))
            except expected_type as error:
                assert expected_text in str(error), (scenario, str(error))
            else:
                raise AssertionError(f"Failure scenario was accepted: {scenario}")
        if scenario == "unmanifested_conflict":
            assert (case_raw / "train.jsonl").read_bytes() == b"user content must survive\n"
            assert {path.name for path in case_raw.iterdir()} == {"train.jsonl"}
        else:
            assert not case_raw.exists() or not list(case_raw.iterdir()), scenario
        checks += 1

    cli_input = temporary_root / "verifier_input.jsonl"
    cli_output = temporary_root / "verifier_output.jsonl"
    cli_input.write_text(json.dumps({"text": "Reasoning\u2028has a separator.\n#### 2", "answer": "2"}, ensure_ascii=False) + "\n", encoding="utf-8")
    completed = subprocess.run([sys.executable, str(root / "scripts/verify_answers.py"), "--input", str(cli_input), "--output", str(cli_output)], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(cli_output.read_text())["correct"] is True
    checks += 1

print(f"[PASS] {checks} downloader/provenance/Unicode scenarios; all network responses mocked, no external downloads during tests")
