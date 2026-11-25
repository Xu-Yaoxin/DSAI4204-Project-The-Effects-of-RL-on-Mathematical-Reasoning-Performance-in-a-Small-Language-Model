"""Perform one race-safe Linux pidfd operation without requiring Python wrappers."""
import ctypes
import errno
import os
import platform
import signal
import sys

operation = globals().get('PIDFD_OPERATION')
if operation not in {'open', 'send', 'probe'}:
    raise ValueError('PIDFD_OPERATION must be open, send or probe.')
if platform.system() != 'Linux':
    raise RuntimeError('Safe process control requires Linux; no process was signalled.')

native_open = getattr(os, 'pidfd_open', None)
native_send = getattr(signal, 'pidfd_send_signal', None)
open_backend = 'python' if callable(native_open) else 'linux-syscall'
send_backend = 'python' if callable(native_send) else 'linux-syscall'
actions = ['open', 'send'] if operation == 'probe' else [operation]
requires_syscall = ('open' in actions and open_backend == 'linux-syscall') or ('send' in actions and send_backend == 'linux-syscall')
syscall = None
if requires_syscall:
    if platform.machine().lower() not in {'x86_64', 'amd64'} or ctypes.sizeof(ctypes.c_void_p) != 8 or sys.maxsize <= 2 ** 32:
        raise RuntimeError('The pidfd compatibility backend requires 64-bit Linux x86_64; no process was signalled.')
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long]

target_pid = os.getpid() if operation == 'probe' else globals().get('PIDFD_PID')
target_fd = globals().get('PIDFD_FD')
target_signal = 0 if operation == 'probe' else globals().get('PIDFD_SIGNAL')
probe_fd = None
PIDFD_RESULT = {'operation': operation, 'open_backend': open_backend, 'send_backend': send_backend,
                'backend': open_backend if open_backend == send_backend else 'mixed'}
try:
    for action in actions:
        if action == 'open':
            if type(target_pid) is not int or target_pid <= 0:
                raise ValueError('PIDFD_PID must be a positive process ID.')
            try:
                if callable(native_open):
                    result = native_open(target_pid, 0)
                else:
                    ctypes.set_errno(0)
                    result = syscall(ctypes.c_long(434), ctypes.c_int(target_pid), ctypes.c_uint(0))
                    if result == -1:
                        error_number = ctypes.get_errno()
                        raise OSError(error_number, os.strerror(error_number))
            except OSError as error:
                if error.errno in {errno.ENOSYS, errno.EPERM, errno.EACCES}:
                    detail = 'unavailable in this kernel or blocked by the container' if error.errno == errno.ENOSYS else 'denied by permissions or container policy'
                    raise OSError(error.errno, f'pidfd_open is {detail}; no PID-based signal fallback is used.') from error
                raise
            target_fd = result
            if operation == 'probe':
                probe_fd = result
            else:
                PIDFD_RESULT['fd'] = result
        else:
            if type(target_fd) is not int or target_fd < 0:
                raise ValueError('PIDFD_FD must be a nonnegative file descriptor.')
            if isinstance(target_signal, bool) or not isinstance(target_signal, int) or not 0 <= target_signal < signal.NSIG:
                raise ValueError('PIDFD_SIGNAL must be a valid nonnegative signal number.')
            try:
                if callable(native_send):
                    native_send(target_fd, target_signal)
                else:
                    ctypes.set_errno(0)
                    result = syscall(ctypes.c_long(424), ctypes.c_int(target_fd), ctypes.c_int(target_signal), ctypes.c_void_p(None), ctypes.c_uint(0))
                    if result == -1:
                        error_number = ctypes.get_errno()
                        raise OSError(error_number, os.strerror(error_number))
            except OSError as error:
                if error.errno in {errno.ENOSYS, errno.EPERM, errno.EACCES}:
                    detail = 'unavailable in this kernel or blocked by the container' if error.errno == errno.ENOSYS else 'denied by permissions or container policy'
                    raise OSError(error.errno, f'pidfd_send_signal is {detail}; no PID-based signal fallback is used.') from error
                raise
    if operation == 'probe':
        PIDFD_RESULT.update(supported=True, probed_pid=target_pid, probe_signal=0)
finally:
    if probe_fd is not None:
        os.close(probe_fd)
