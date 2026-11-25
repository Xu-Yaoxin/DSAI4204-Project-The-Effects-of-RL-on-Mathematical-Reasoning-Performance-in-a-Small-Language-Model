"""Exercise model integrity and persistent HTTPS Range resume without external network or GPU."""

import ast
import contextlib
import fcntl
import hashlib
import http.client
import io
import json
from pathlib import Path
import runpy
import shutil
import ssl
import sys
import tempfile
from types import ModuleType
import urllib.error
from unittest.mock import MagicMock, patch


root = Path(__file__).resolve().parents[1]
script = root / "scripts/download_model.py"
official_manifest = json.loads((root / "models/qwen3.5-2b-base-manifest.json").read_text())
revision = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
assert official_manifest["repo_id"] == "Qwen/Qwen3.5-2B-Base" and official_manifest["revision"] == revision
assert len(official_manifest["files"]) == 12 and official_manifest["total_bytes"] == 4571206192
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for node in ast.walk(ast.parse(script.read_text())))
checks = 1
payloads = {"config.json": b'{"fixture":true}\n', "model.safetensors": b"synthetic model bytes, not a usable model"}
entries = []
for name, payload in payloads.items():
    entry = {"path": name, "size": len(payload)}
    if name.endswith(".safetensors"):
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        entry["git_blob_sha1"] = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    entries.append(entry)
fixture_manifest = {"schema_version": 1, "repo_id": "Qwen/Qwen3.5-2B-Base", "revision": revision,
                    "total_bytes": sum(map(len, payloads.values())), "files": entries}
progress_module = ModuleType("tqdm")
progress_module.tqdm = MagicMock()
progress = progress_module.tqdm.return_value.__enter__.return_value
module_patches = {"tqdm": progress_module}

with tempfile.TemporaryDirectory(prefix="dsai4207-model-download-") as directory:
    test_root = Path(directory)
    manifest_path = test_root / "manifest.json"
    manifest_path.write_text(json.dumps(fixture_manifest))
    output = test_root / "model with spaces"
    output.mkdir()
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    argv = [str(script), "--manifest", str(manifest_path), "--output-dir", str(output)]
    before = {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns) for name in payloads}
    with patch.object(sys, "argv", argv + ["--offline"]), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=AssertionError("Offline run attempted network")) as network, contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script))
    receipt = json.loads((output / "source_manifest.json").read_text())
    assert receipt["status"] == "verified" and receipt["all_files_verified"] is True
    assert receipt["downloaded_files"] == 0 and receipt["reused_files"] == 2
    assert receipt["files"]["config.json"]["sha256"] == hashlib.sha256(payloads["config.json"]).hexdigest()
    assert receipt["model_loaded"] is False and receipt["training_executed"] is False
    assert before == {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns) for name in payloads}
    assert network.call_count == 0 and progress_module.tqdm.call_args.kwargs["disable"] is False
    assert sum(call.args[0] for call in progress.update.call_args_list) == fixture_manifest["total_bytes"]
    checks += 1

    original = payloads["config.json"]
    corrupted = b"!" + original[1:]
    (output / "config.json").write_bytes(corrupted)
    (output / "model.safetensors").unlink()
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=AssertionError("Corrupt existing file triggered download")) as network, contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except ValueError as error:
            assert "Model hash mismatch" in str(error)
        else:
            raise AssertionError("Same-size Git blob corruption was accepted")
    receipt = json.loads((output / "source_manifest.json").read_text())
    assert receipt["status"] == "failed" and receipt["all_files_verified"] is False
    assert (output / "config.json").read_bytes() == corrupted and network.call_count == 0
    (output / "config.json").write_bytes(original)
    checks += 1

    with patch.object(sys, "argv", argv + ["--offline"]), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=AssertionError("Incomplete offline run attempted network")) as network, contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except FileNotFoundError as error:
            assert "Offline model verification requires all files" in str(error)
        else:
            raise AssertionError("Offline verification accepted an incomplete model")
    assert network.call_count == 0
    checks += 1

    # Bytes survive a failed invocation; the next invocation requests the exact remaining range.
    weight = payloads["model.safetensors"]
    size = len(weight)
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.headers = {"Content-Length": str(size)}
    response.geturl.return_value = "https://cdn.example/model"
    response.read.side_effect = [weight[:8], http.client.IncompleteRead(b"discarded incomplete read", size - 8)]
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=[response, urllib.error.URLError("offline"), urllib.error.URLError("offline")]) as network, patch("time.sleep"), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except RuntimeError as error:
            assert "Partial bytes are retained" in str(error)
        else:
            raise AssertionError("Interrupted transfer was hidden")
    partial = output / ".cache/rlvr-download" / revision / (hashlib.sha256(b"model.safetensors").hexdigest() + ".part")
    assert partial.read_bytes() == weight[:8] and not (output / "model.safetensors").exists()
    assert [call.args[0].get_header("Range") for call in network.call_args_list] == ["bytes=0-", "bytes=8-", "bytes=8-"]
    assert json.loads((output / "source_manifest.json").read_text())["status"] == "failed"
    checks += 1

    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 206
    response.headers = {"Content-Length": str(size - 8), "Content-Range": f"bytes 8-{size - 1}/{size}"}
    response.geturl.return_value = "https://cdn.example/model"
    response.read.side_effect = [weight[8:], b""]
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", return_value=response) as network, contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script))
    assert network.call_count == 1
    request = network.call_args.args[0]
    assert request.full_url == f"https://huggingface.co/Qwen/Qwen3.5-2B-Base/resolve/{revision}/model.safetensors?download=true"
    assert request.get_header("Range") == "bytes=8-" and request.get_header("Accept-encoding") == "identity"
    assert network.call_args.kwargs["context"].verify_mode == ssl.CERT_REQUIRED and network.call_args.kwargs["context"].check_hostname
    assert (output / "model.safetensors").read_bytes() == weight and not partial.exists()
    receipt = json.loads((output / "source_manifest.json").read_text())
    assert receipt["status"] == "verified" and receipt["downloaded_files"] == 1 and receipt["reused_files"] == 1
    assert receipt["resumed_bytes"] == 8 and receipt["transferred_bytes"] == size - 8
    assert any(call.kwargs.get("initial") == 8 and call.kwargs.get("disable") is False for call in progress_module.tqdm.call_args_list)
    checks += 1

    for extra in ([], ["--verify"]):
        with patch.object(sys, "argv", argv + extra), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=AssertionError("Complete files should be reused")) as network, contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script))
        assert network.call_count == 0
        checks += 1

    (output / "model.safetensors").write_bytes(b"!" + weight[1:])
    with patch.object(sys, "argv", argv + ["--offline"]), patch.dict(sys.modules, module_patches), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except ValueError as error:
            assert "Model hash mismatch" in str(error)
        else:
            raise AssertionError("SHA256 model corruption was accepted")
    (output / "model.safetensors").unlink()
    checks += 1

    # Reject malformed/ignored ranges before touching the existing partial file.
    partial.write_bytes(weight[:8])
    for status, headers, expected in [
        (200, {"Content-Length": str(size)}, "ignored a resume Range"),
        (206, {"Content-Range": f"bytes 0-{size - 1}/{size}"}, "Unexpected model Content-Range"),
        (206, {"Content-Range": f"bytes 8-{size - 1}/{size}", "Content-Length": str(size)}, "Unexpected model Content-Length"),
    ]:
        response = MagicMock()
        response.__enter__.return_value = response
        response.status, response.headers = status, headers
        response.geturl.return_value = "https://cdn.example/model"
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", return_value=response) as network, contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script))
            except ValueError as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError("Invalid range response accepted")
        assert network.call_count == 1 and partial.read_bytes() == weight[:8]
        assert not response.read.called and not (output / "model.safetensors").exists()
        checks += 1

    partial.unlink()
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.headers = {"Content-Length": str(size)}
    response.geturl.return_value = "https://cdn.example/model"
    bad_weight = b"!" + weight[1:]
    response.read.side_effect = [bad_weight, b""]
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", return_value=response), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except ValueError as error:
            assert "Model hash mismatch" in str(error)
        else:
            raise AssertionError("Bad downloaded bytes were installed")
    assert partial.read_bytes() == bad_weight and not (output / "model.safetensors").exists()
    assert json.loads((output / "source_manifest.json").read_text())["all_files_verified"] is False
    checks += 1

    partial.unlink()
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.headers = {"Content-Length": str(size)}
    response.geturl.return_value = "https://cdn.example/model"
    response.read.side_effect = [weight[:8], KeyboardInterrupt()]
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", return_value=response) as network, contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Keyboard interrupt was swallowed")
    assert network.call_count == 1 and partial.read_bytes() == weight[:8]
    assert not (output / "model.safetensors").exists()
    assert json.loads((output / "source_manifest.json").read_text())["status"] == "failed"
    checks += 1

    # A verified complete partial can be committed without another network request.
    partial.write_bytes(weight)
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=AssertionError("Complete partial attempted network")), contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(script))
    assert (output / "model.safetensors").read_bytes() == weight and not partial.exists()
    checks += 1

    for name in ["generation_config.json", "added_tokens.json", "special_tokens_map.json", "adapter_config.json", "adapter_model.bin"]:
        (output / name).write_text("{}")
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), patch("urllib.request.urlopen", side_effect=AssertionError("Extra settings triggered network")), contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script))
            except ValueError as error:
                assert "Unlisted model-affecting" in str(error)
            else:
                raise AssertionError(f"Extra model-affecting file accepted: {name}")
        assert (output / name).read_text() == "{}"
        (output / name).unlink()
        checks += 1

    (output / "model.safetensors").unlink()
    foreign = test_root / "foreign.bin"
    foreign.write_bytes(weight)
    (output / "model.safetensors").symlink_to(foreign)
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except ValueError as error:
            assert "Refusing symlink" in str(error)
        else:
            raise AssertionError("Symlink model destination was accepted")
    assert foreign.read_bytes() == weight
    (output / "model.safetensors").unlink()
    checks += 1

    shutil.rmtree(output / ".cache")
    foreign_cache = test_root / "foreign-cache"
    foreign_cache.mkdir()
    (output / ".cache").symlink_to(foreign_cache, target_is_directory=True)
    with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except ValueError as error:
            assert "symlink model download cache" in str(error)
        else:
            raise AssertionError("External symlink download cache was accepted")
    assert not list(foreign_cache.iterdir())
    (output / ".cache").unlink()
    checks += 1

    receipt_before = (output / "source_manifest.json").read_bytes()
    with (output / ".model-download.lock").open("a+b") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script))
            except SystemExit as error:
                assert "Another model download/verification" in str(error)
            else:
                raise AssertionError("Concurrent model writers were permitted")
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
    assert (output / "source_manifest.json").read_bytes() == receipt_before
    checks += 1

    for scenario in ("../escape", "/absolute", "config.json/../escape", "duplicate", "revision"):
        bad = json.loads(json.dumps(fixture_manifest))
        if scenario == "duplicate":
            bad["files"].append(bad["files"][0])
        elif scenario == "revision":
            bad["revision"] = "main"
        else:
            bad["files"][0]["path"] = scenario
        manifest_path.write_text(json.dumps(bad))
        with patch.object(sys, "argv", argv), patch.dict(sys.modules, module_patches), contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script))
            except SystemExit:
                pass
            else:
                raise AssertionError(f"Invalid manifest accepted: {scenario}")
        assert not (test_root / "escape").exists()
        checks += 1

print(f"✅ [MODEL DOWNLOAD] {checks} checks | pinned hashes, persistent Range resume, offline reuse, progress and path safety | no real weights downloaded")
