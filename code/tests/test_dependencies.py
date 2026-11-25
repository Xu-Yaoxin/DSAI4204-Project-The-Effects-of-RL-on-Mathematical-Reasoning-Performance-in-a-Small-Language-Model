"""Check locked target dependencies, resumable download integrity, and setup guards."""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name


root = Path(__file__).resolve().parents[1]
script = root / 'scripts/prepare_packages.py'
manifest_path = root / 'packages/linux-cp311-cu128/manifest.json'
manifest = json.loads(manifest_path.read_text())
packages = {entry['name']: entry for entry in manifest['packages']}
assert len(packages) == manifest['package_count']
assert manifest['target']['os'] == 'Linux' and manifest['target']['machine'] == 'x86_64'
assert manifest['target']['python'] == '3.11' and manifest['target']['cuda'] == '12.8'
assert manifest['marker_environment']['sys_platform'] == 'linux'
assert manifest['marker_environment']['python_version'] == '3.11'
assert packages['torch']['version'] == '2.10.0+cu128'
assert packages['accelerate']['version'] == '1.10.1'
assert packages['transformers']['version'] == '5.3.0'
assert packages['peft']['version'] == '0.18.1'
assert packages['trl']['version'] == '1.16.0'
assert {'nvidia-cublas-cu12','nvidia-cudnn-cu12','nvidia-nccl-cu12','nvidia-cuda-runtime-cu12','triton'} <= set(packages)
assert 'flash-linear-attention' not in packages and 'causal-conv1d' not in packages
assert manifest['dependency_closure']['verified'] is True
assert manifest['validation']['gpu_runtime_tested'] is False
assert manifest['total_bytes'] == sum(entry['bytes'] for entry in packages.values())
lock = manifest_path.parent / 'requirements-lock.txt'
assert hashlib.sha256(lock.read_bytes()).hexdigest() == manifest['lock_sha256']
assert hashlib.sha256((root/'requirements-gpu.txt').read_bytes()).hexdigest() == manifest['requirements_sha256']
assert not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.walk(ast.parse(script.read_text())))
extras={name:set() for name in packages}
changed=True
while changed:
    changed=False
    for name,entry in packages.items():
        assert not entry['requires_python'] or SpecifierSet(entry['requires_python']).contains('3.11.0')
        for text in entry['requires_dist']:
            dependency=Requirement(text)
            if dependency.marker and not any(dependency.marker.evaluate({**manifest['marker_environment'],'extra':extra}) for extra in extras[name]|{''}):
                continue
            required=canonicalize_name(dependency.name)
            assert required in packages,(name,text)
            assert dependency.specifier.contains(packages[required]['version'],prereleases=True),(name,text)
            new_extras=set(dependency.extras)-extras[required]
            if new_extras:
                extras[required].update(new_extras)
                changed=True
assert 'http' in extras['fsspec']
checks = 2

with tempfile.TemporaryDirectory(prefix='dsai4207-package-test-') as directory:
    bundle = Path(directory)
    wheels = bundle/'wheels'
    wheels.mkdir()
    payload = b'a small transport fixture, not an executable wheel'
    entry = {'name':'fixture','version':'1.0','filename':'fixture-1.0-py3-none-any.whl','url':'https://files.pythonhosted.org/fixture.whl',
             'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'etag':'"stable-fixture"','accept_ranges':True}
    fixture_lock = f"fixture==1.0 --hash=sha256:{entry['sha256']}\n"
    fixture_manifest = {'schema_version':1,'target':{'id':'linux-cp311-cu128'},'packages':[entry],'package_count':1,'total_bytes':len(payload),
                        'lock_sha256':hashlib.sha256(fixture_lock.encode()).hexdigest(),
                        'requirements_sha256':hashlib.sha256((root/'requirements-gpu.txt').read_bytes()).hexdigest()}
    (bundle/'requirements-lock.txt').write_text(fixture_lock)
    (bundle/'manifest.json').write_text(json.dumps(fixture_manifest))
    argv = [str(script),'--bundle',str(bundle)]
    complete_part = wheels/(entry['filename']+f".part-{0:012d}-{len(payload)-1:012d}")
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.headers = {}
    response.read.side_effect = [payload,b'']
    with patch.object(sys,'argv',argv+['--fetch-part','fixture','--range-start','0','--range-end',str(len(payload)-1)]), patch('urllib.request.urlopen',return_value=response) as network, contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert error.code == 0
    assert complete_part.read_bytes() == payload
    assert network.call_args.args[0].get_header('If-match') == '"stable-fixture"'
    checks += 1

    completed = subprocess.run([sys.executable,str(script),'--bundle',str(bundle),'--workers','2'],text=True,capture_output=True)
    assert completed.returncode == 0,(completed.stdout,completed.stderr)
    wheel = wheels/entry['filename']
    assert wheel.read_bytes() == payload
    assert not complete_part.exists() and not list(wheels.glob('*.assemble'))
    checks += 1

    with patch.object(sys,'argv',argv+['--verify']), patch('urllib.request.urlopen',side_effect=AssertionError('Offline verify attempted network')), contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert error.code == 0
    checks += 1
    wheel.write_bytes(payload+b'tampered')
    with patch.object(sys,'argv',argv+['--verify']):
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert 'Wheel hash mismatch' in str(error)
        else:
            raise AssertionError('Tampered wheel accepted')
    assert wheel.read_bytes()==payload+b'tampered'
    wheel.unlink()
    checks += 1

    start,end=7,26
    part = wheels/(entry['filename']+f'.part-{start:012d}-{end:012d}')
    part.write_bytes(payload[start:start+3])
    response = MagicMock()
    response.__enter__.return_value = response
    response.status=206
    response.headers={'Content-Range':f'bytes {start+3}-{end}/{len(payload)}'}
    response.read.side_effect=[payload[start+3:end+1],b'']
    with patch.object(sys,'argv',argv+['--fetch-part','fixture','--range-start',str(start),'--range-end',str(end)]), patch('urllib.request.urlopen',return_value=response) as network, contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert error.code==0
    assert network.call_args.args[0].get_header('Range')==f'bytes={start+3}-{end}'
    assert part.read_bytes()==payload[start:end+1]
    checks += 1

    part.write_bytes(payload[start:start+3])
    response = MagicMock()
    response.__enter__.return_value=response
    response.status=206
    response.headers={'Content-Range':f'bytes 0-{end}/{len(payload)}'}
    with patch.object(sys,'argv',argv+['--fetch-part','fixture','--range-start',str(start),'--range-end',str(end)]), patch('urllib.request.urlopen',return_value=response):
        try:
            runpy.run_path(str(script))
        except ValueError as error:
            assert 'Unexpected range' in str(error)
        else:
            raise AssertionError('Mismatched resume range accepted')
    assert part.read_bytes()==payload[start:start+3]
    checks += 1

    fixture_manifest['packages'][0]['sha256']='0'*64
    (bundle/'manifest.json').write_text(json.dumps(fixture_manifest))
    with patch.object(sys,'argv',argv+['--verify']):
        try:
            runpy.run_path(str(script))
        except SystemExit as error:
            assert 'manifest and exact hash lock disagree' in str(error)
        else:
            raise AssertionError('Manifest/lock disagreement accepted')
    checks += 1

for name in ['setup_gpu.sh','download_packages.sh']:
    completed=subprocess.run(['bash','-n',str(root/name)],text=True,capture_output=True)
    assert completed.returncode==0,completed.stderr
    completed=subprocess.run(['bash',str(root/name),'--help'],text=True,capture_output=True,cwd='/private/tmp' if platform.system()=='Darwin' else '/tmp')
    assert completed.returncode==0 and 'Usage:' in completed.stdout
    checks += 1
if platform.system()!='Linux':
    existed=(root/'.venv').exists()
    completed=subprocess.run(['bash',str(root/'setup_gpu.sh'),'--offline'],text=True,capture_output=True)
    assert completed.returncode==2 and 'requires Linux' in completed.stderr
    assert (root/'.venv').exists()==existed
    checks += 1

with tempfile.TemporaryDirectory(prefix='dsai4207-setup-order-') as directory:
    fixture_root=Path(directory)
    fake_bin=fixture_root/'bin'
    fake_bin.mkdir()
    (fake_bin/'uname').write_text('#!/bin/sh\nif [ "$1" = "-s" ]; then echo Linux; else echo x86_64; fi\n')
    (fake_bin/'nvidia-smi').write_text('#!/bin/sh\nexit 0\n')
    fake_python=fake_bin/'fixture-python'
    fake_driver=fixture_root/'fixture_driver.py'
    fake_python.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(fake_driver))+' "$@"\n')
    fake_driver.write_text('''
import os
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(os.environ['SETUP_FIXTURE_ROOT'])
mode=os.environ['SETUP_FIXTURE_MODE']
args=sys.argv[1:]
with (root/'calls.txt').open('a') as output:
    output.write(repr(args)+'\\n')
if args==['-']:
    sys.stdin.read()
    raise SystemExit(0)
if args[:1]==['-c']:
    if args[1]=='import venv, ensurepip':
        raise SystemExit(1 if mode=='missing_support' else 0)
    raise SystemExit(subprocess.run([sys.executable,*args]).returncode)
if args[:2]==['-m','venv']:
    target=root/'.venv/bin'
    target.mkdir(parents=True)
    shutil.copy2(Path(os.environ['SETUP_FIXTURE_PYTHON']),target/'python')
    raise SystemExit(0)
if args[:2]==['-m','ensurepip']:
    if mode=='pip_bootstrap_failure':
        raise SystemExit(1)
    (root/'pip-ready').write_text('ready')
    raise SystemExit(0)
if args[:3]==['-m','pip','--version']:
    raise SystemExit(0 if (root/'pip-ready').exists() else 1)
if args[:1]==['scripts/prepare_packages.py']:
    (root/'download-called').write_text('called')
    raise SystemExit(73)
raise SystemExit('Unexpected fixture invocation: '+repr(args))
''')
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)
    for mode in ['unmanaged','symlink','broken_managed','missing_support','pip_bootstrap_failure','new_ok','managed_ok','managed_missing_pip']:
        project=fixture_root/mode
        project.mkdir()
        shutil.copy2(root/'setup_gpu.sh',project/'setup_gpu.sh')
        lock_directory=project/'packages/linux-cp311-cu128'
        lock_directory.mkdir(parents=True)
        (lock_directory/'requirements-lock.txt').write_bytes(lock.read_bytes())
        if mode in ['unmanaged','broken_managed','managed_ok','managed_missing_pip']:
            (project/'.venv/bin').mkdir(parents=True)
        if mode=='symlink':
            (project/'.venv').symlink_to(project/'absent-env',target_is_directory=True)
        if mode in ['broken_managed','managed_ok','managed_missing_pip']:
            (project/'.venv/.rlvr-package-lock-sha256').write_text(manifest['lock_sha256']+'\n')
        if mode in ['managed_ok','managed_missing_pip']:
            shutil.copy2(fake_python,project/'.venv/bin/python')
        if mode=='managed_ok':
            (project/'pip-ready').write_text('ready')
        environment={**os.environ,'PATH':str(fake_bin)+os.pathsep+os.environ['PATH'],
                     'RLVR_SYSTEM_PYTHON':str(fake_python),'SETUP_FIXTURE_PYTHON':str(fake_python),
                     'SETUP_FIXTURE_ROOT':str(project),'SETUP_FIXTURE_MODE':mode}
        completed=subprocess.run(['bash',str(project/'setup_gpu.sh')],env=environment,text=True,capture_output=True,cwd=directory)
        if mode in ['new_ok','managed_ok','managed_missing_pip']:
            assert completed.returncode==73,(mode,completed.stdout,completed.stderr)
            assert (project/'download-called').exists()
            assert (project/'pip-ready').exists()
            assert (project/'.venv/.rlvr-package-lock-sha256').read_text().strip()==manifest['lock_sha256']
            calls=(project/'calls.txt').read_text().split('\n')
            assert next(i for i,line in enumerate(calls) if "'pip', '--version'" in line)<next(i for i,line in enumerate(calls) if 'scripts/prepare_packages.py' in line)
        else:
            assert completed.returncode==2,(mode,completed.stdout,completed.stderr)
            assert not (project/'download-called').exists(),mode
        if mode=='pip_bootstrap_failure':
            assert (project/'.venv/.rlvr-package-lock-sha256').read_text().strip()==manifest['lock_sha256']
        if mode=='missing_support':
            assert not (project/'.venv').exists()
        if mode=='symlink':
            assert (project/'.venv').is_symlink()
        checks+=1
print(f'[PASS] {checks} dependency lock/download/setup checks; GPU installation and training remain unrun on this host')
