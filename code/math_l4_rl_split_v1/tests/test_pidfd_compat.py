"""Exercise native and raw-syscall pidfd paths without sending any real signal."""
import ast
from contextlib import ExitStack
import ctypes
import errno
import os
from pathlib import Path
import runpy
import signal
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

helper = Path(__file__).resolve().parents[1] / 'scripts/pidfd_compat.py'
checks = 0
for native_flags in [(True, True), (False, False), (True, False), (False, True)]:
    fake_os = SimpleNamespace(getpid=MagicMock(return_value=12345), strerror=os.strerror, close=MagicMock())
    fake_signal = SimpleNamespace(NSIG=65)
    if native_flags[0]:
        fake_os.pidfd_open = MagicMock(return_value=0)
    if native_flags[1]:
        fake_signal.pidfd_send_signal = MagicMock()
    syscall = MagicMock(side_effect=[0] * 2)
    cdll = MagicMock(return_value=SimpleNamespace(syscall=syscall))
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {'os': fake_os, 'signal': fake_signal}))
        stack.enter_context(patch('platform.system', return_value='Linux'))
        stack.enter_context(patch('platform.machine', return_value='x86_64'))
        stack.enter_context(patch('ctypes.CDLL', cdll))
        result = runpy.run_path(str(helper), init_globals={'PIDFD_OPERATION': 'probe'})['PIDFD_RESULT']
    assert result['supported'] is True and result['probe_signal'] == 0 and result['probed_pid'] == 12345
    assert result['backend'] == ('python' if all(native_flags) else 'linux-syscall' if not any(native_flags) else 'mixed')
    fake_os.close.assert_called_once_with(0)
    if native_flags[0]:
        fake_os.pidfd_open.assert_called_once_with(12345, 0)
    if native_flags[1]:
        fake_signal.pidfd_send_signal.assert_called_once_with(0, 0)
    if all(native_flags):
        cdll.assert_not_called()
    else:
        cdll.assert_called_once_with(None, use_errno=True)
        assert syscall.restype is ctypes.c_long and syscall.argtypes == [ctypes.c_long]
        expected_numbers = ([434] if not native_flags[0] else []) + ([424] if not native_flags[1] else [])
        assert [call.args[0].value for call in syscall.call_args_list] == expected_numbers
        for call in syscall.call_args_list:
            if call.args[0].value == 434:
                assert [type(value) for value in call.args] == [ctypes.c_long, ctypes.c_int, ctypes.c_uint]
                assert [value.value for value in call.args] == [434, 12345, 0]
            else:
                assert [type(value) for value in call.args] == [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
                assert [value.value for value in call.args] == [424, 0, 0, None, 0]
    checks += 4

for operation in ['open', 'send']:
    fake_os = SimpleNamespace(getpid=MagicMock(return_value=12345), strerror=os.strerror, close=MagicMock())
    fake_signal = SimpleNamespace(NSIG=65)
    syscall = MagicMock(return_value=0)
    with patch.dict(sys.modules, {'os': fake_os, 'signal': fake_signal}), patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), patch('ctypes.CDLL', return_value=SimpleNamespace(syscall=syscall)):
        result = runpy.run_path(str(helper), init_globals={'PIDFD_OPERATION': operation, 'PIDFD_PID': 12345, 'PIDFD_FD': 0, 'PIDFD_SIGNAL': 0})['PIDFD_RESULT']
    assert operation != 'open' or result['fd'] == 0
    fake_os.close.assert_not_called()
    checks += 2

for use_native in [False, True]:
    for operation in ['open', 'send']:
        for number, exception_type in [(errno.ENOSYS, OSError), (errno.EPERM, PermissionError), (errno.EACCES, PermissionError), (errno.ESRCH, ProcessLookupError)]:
            fake_os = SimpleNamespace(getpid=MagicMock(return_value=12345), strerror=os.strerror, close=MagicMock())
            fake_signal = SimpleNamespace(NSIG=65)
            if use_native:
                fake_os.pidfd_open = MagicMock(side_effect=OSError(number, os.strerror(number)))
                fake_signal.pidfd_send_signal = MagicMock(side_effect=OSError(number, os.strerror(number)))
            syscall = MagicMock(return_value=-1)
            with patch.dict(sys.modules, {'os': fake_os, 'signal': fake_signal}), patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), patch('ctypes.CDLL', return_value=SimpleNamespace(syscall=syscall)), patch('ctypes.get_errno', return_value=number):
                try:
                    runpy.run_path(str(helper), init_globals={'PIDFD_OPERATION': operation, 'PIDFD_PID': 12345, 'PIDFD_FD': 9, 'PIDFD_SIGNAL': 0})
                except exception_type as error:
                    assert error.errno == number
                    if number != errno.ESRCH:
                        assert 'no PID-based signal fallback' in str(error)
                else:
                    raise AssertionError('A pidfd error was accepted.')
            fake_os.close.assert_not_called()
            checks += 1

for native_send in [False, True]:
    fake_os = SimpleNamespace(getpid=MagicMock(return_value=12345), strerror=os.strerror, close=MagicMock())
    fake_signal = SimpleNamespace(NSIG=65)
    if native_send:
        fake_os.pidfd_open = MagicMock(return_value=0)
        fake_signal.pidfd_send_signal = MagicMock(side_effect=PermissionError(errno.EPERM, 'denied'))
    syscall = MagicMock(side_effect=[0, -1])
    with patch.dict(sys.modules, {'os': fake_os, 'signal': fake_signal}), patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'), patch('ctypes.CDLL', return_value=SimpleNamespace(syscall=syscall)), patch('ctypes.get_errno', return_value=errno.EPERM):
        try:
            runpy.run_path(str(helper), init_globals={'PIDFD_OPERATION': 'probe'})
        except PermissionError:
            pass
        else:
            raise AssertionError('A denied self-probe was accepted.')
    fake_os.close.assert_called_once_with(0)
    checks += 1

for system, machine, word_size in [('Darwin', 'x86_64', sys.maxsize), ('Linux', 'aarch64', sys.maxsize), ('Linux', 'x86_64', 2 ** 31 - 1)]:
    fake_os = SimpleNamespace(getpid=MagicMock(return_value=12345), strerror=os.strerror, close=MagicMock())
    fake_signal = SimpleNamespace(NSIG=65)
    cdll = MagicMock()
    with patch.dict(sys.modules, {'os': fake_os, 'signal': fake_signal}), patch('platform.system', return_value=system), patch('platform.machine', return_value=machine), patch.object(sys, 'maxsize', word_size), patch('ctypes.CDLL', cdll):
        try:
            runpy.run_path(str(helper), init_globals={'PIDFD_OPERATION': 'probe'})
        except RuntimeError as error:
            assert 'no process was signalled' in str(error)
        else:
            raise AssertionError('An unsupported syscall ABI was accepted.')
    cdll.assert_not_called()
    fake_os.close.assert_not_called()
    checks += 1

for initial in [{'PIDFD_OPERATION': 'invalid'}, {'PIDFD_OPERATION': 'open', 'PIDFD_PID': -1}, {'PIDFD_OPERATION': 'open', 'PIDFD_PID': True}, {'PIDFD_OPERATION': 'send', 'PIDFD_FD': -1, 'PIDFD_SIGNAL': 0}, {'PIDFD_OPERATION': 'send', 'PIDFD_FD': 0, 'PIDFD_SIGNAL': -1}, {'PIDFD_OPERATION': 'send', 'PIDFD_FD': 0, 'PIDFD_SIGNAL': True}]:
    fake_os = SimpleNamespace(getpid=MagicMock(return_value=12345), strerror=os.strerror, close=MagicMock(), pidfd_open=MagicMock())
    fake_signal = SimpleNamespace(NSIG=65, pidfd_send_signal=MagicMock())
    with patch.dict(sys.modules, {'os': fake_os, 'signal': fake_signal}), patch('platform.system', return_value='Linux'):
        try:
            runpy.run_path(str(helper), init_globals=initial)
        except ValueError:
            pass
        else:
            raise AssertionError('An invalid pidfd argument was accepted.')
    fake_os.pidfd_open.assert_not_called()
    fake_signal.pidfd_send_signal.assert_not_called()
    checks += 1

tree = ast.parse(helper.read_text())
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) for node in ast.walk(tree))
assert not any(isinstance(node, ast.Attribute) and node.attr in {'kill', 'killpg'} for node in ast.walk(tree))
checks += 2
print(f'[PASS] {checks} pidfd compatibility checks | native, missing/partial wrappers, exact syscall ABI, errno, descriptor cleanup and unsupported hosts | all signals mocked')
