"""Shared test utilities for the quent test suite."""
from __future__ import annotations

import asyncio
import functools
import operator
from typing import Any


# --- Sync/Async callable pairs ---

def sync_fn(x):
  return x + 1

async def async_fn(x):
  return x + 1

def sync_identity(x):
  return x

async def async_identity(x):
  return x

def raise_fn(x=None):
  raise ValueError('test error')

async def async_raise_fn(x=None):
  raise ValueError('test error')

def sync_side_effect(tracker, x):
  tracker.append(x)
  return 'side_effect_result'

async def async_side_effect(tracker, x):
  tracker.append(x)
  return 'side_effect_result'


# --- Callable type fixtures ---

class Adder:
  """Not callable -- used as a non-callable class constructor test."""
  def __init__(self, x):
    self.x = x

class CallableObj:
  def __call__(self, x):
    return x + 1

class AsyncCallableObj:
  async def __call__(self, x):
    return x + 1

class BoundMethodHolder:
  def method(self, x):
    return x + 1

partial_fn = functools.partial(operator.add, 10)


# --- Context manager fixtures ---

class SyncCM:
  def __init__(self):
    self.entered = False
    self.exited = False
  def __enter__(self):
    self.entered = True
    return 'ctx_value'
  def __exit__(self, *args):
    self.exited = True
    return False

class AsyncCM:
  def __init__(self):
    self.entered = False
    self.exited = False
  async def __aenter__(self):
    self.entered = True
    return 'ctx_value'
  async def __aexit__(self, *args):
    self.exited = True
    return False

class SyncCMSuppresses:
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, *args):
    return True

class AsyncCMSuppresses:
  async def __aenter__(self):
    return 'ctx_value'
  async def __aexit__(self, *args):
    return True

class SyncCMRaisesOnEnter:
  def __enter__(self):
    raise RuntimeError('enter error')
  def __exit__(self, *args):
    return False

class SyncCMRaisesOnExit:
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, *args):
    raise RuntimeError('exit error')

class DualCM:
  """Has both sync and async context manager protocols."""
  def __enter__(self):
    return 'sync_ctx'
  def __exit__(self, *args):
    return False
  async def __aenter__(self):
    return 'async_ctx'
  async def __aexit__(self, *args):
    return False


# --- Async iterable fixtures ---

class AsyncRange:
  def __init__(self, n):
    self.n = n
  def __aiter__(self):
    self._i = 0
    return self
  async def __anext__(self):
    if self._i >= self.n:
      raise StopAsyncIteration
    val = self._i
    self._i += 1
    return val

class AsyncEmpty:
  def __aiter__(self):
    return self
  async def __anext__(self):
    raise StopAsyncIteration


# --- Utility ---

def run_sync(coro):
  """Run a coroutine synchronously."""
  return asyncio.run(coro)


# --- Edge-case context managers & iterables ---

class SyncCMWithAwaitableExit:
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, *args):
    async def _exit():
      return False
    return _exit()

class SyncCMSuppressesAwaitable:
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, *args):
    async def _exit():
      return True
    return _exit()

class AsyncCMNoop:
  async def __aenter__(self):
    return 'ctx_value'
  async def __aexit__(self, *args):
    return False

class AsyncRangeRaises:
  def __init__(self, n, raise_at):
    self.n = n
    self.raise_at = raise_at
  def __aiter__(self):
    self._i = 0
    return self
  async def __anext__(self):
    if self._i >= self.n:
      raise StopAsyncIteration
    val = self._i
    self._i += 1
    if val == self.raise_at:
      raise RuntimeError('iteration error')
    return val


# --- Traceback edge-case fixtures ---

class ObjWithBadName:
  """__name__/__qualname__ raise RuntimeError (not AttributeError), so
  getattr(obj, '__name__', None) propagates instead of returning default.
  This triggers the ``except Exception`` path in _get_obj_name.
  Falls back to repr().
  """
  def __getattr__(self, name):
    if name in ('__name__', '__qualname__'):
      raise RuntimeError(f'no {name}')
    raise AttributeError(name)
  def __repr__(self):
    return '<ObjWithBadName>'

class ObjWithBadNameAndRepr:
  """Same as ObjWithBadName but repr() also raises, so _get_obj_name
  falls all the way back to type(obj).__name__.
  """
  def __getattr__(self, name):
    if name in ('__name__', '__qualname__'):
      raise RuntimeError(f'no {name}')
    raise AttributeError(name)
  def __repr__(self):
    raise RuntimeError('no repr')

class AsyncCMRaisesOnExit:
  async def __aenter__(self):
    return self

  async def __aexit__(self, *args):
    raise RuntimeError('async exit error')


# --- Stateful callables ---

class StatefulCallable:
  """Tracks call count and arguments for verification."""
  def __init__(self):
    self.calls = []
  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return len(self.calls)

class AsyncStatefulCallable:
  """Async version of StatefulCallable."""
  def __init__(self):
    self.calls = []
  async def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return len(self.calls)


# --- Additional context manager fixtures ---

class AsyncCMRaisesOnEnter:
  """Async CM whose __aenter__ raises."""
  async def __aenter__(self):
    raise RuntimeError('async enter error')
  async def __aexit__(self, *args):
    return False

class SyncCMRaisesOnExitFrom:
  """__exit__ raises with `from exc` chaining."""
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_val is not None:
      raise RuntimeError('exit error') from exc_val
    return False

class SyncCMExitReturnsAwaitableOnException:
  """Sync CM whose __exit__ returns an awaitable only on exception."""
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      async def _exit():
        return False
      return _exit()
    return False

class TrackingCM:
  """CM that records enter/exit calls and arguments."""
  def __init__(self):
    self.entered = False
    self.exited = False
    self.exit_args = None
    self.enter_result = 'tracked_ctx'
  def __enter__(self):
    self.entered = True
    return self.enter_result
  def __exit__(self, *args):
    self.exited = True
    self.exit_args = args
    return False

class AsyncTrackingCM:
  """Async CM that records enter/exit calls."""
  def __init__(self):
    self.entered = False
    self.exited = False
    self.exit_args = None
    self.enter_result = 'async_tracked_ctx'
  async def __aenter__(self):
    self.entered = True
    return self.enter_result
  async def __aexit__(self, *args):
    self.exited = True
    self.exit_args = args
    return False

class SyncCMEnterReturnsNone:
  """CM whose __enter__ returns None."""
  def __enter__(self):
    return None
  def __exit__(self, *args):
    return False

class SyncCMEnterReturnsSelf:
  """CM whose __enter__ returns self."""
  def __enter__(self):
    return self
  def __exit__(self, *args):
    return False

class SyncCMExitRaisesOnSuccess:
  """CM whose __exit__ raises even on success (no body exception)."""
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    raise RuntimeError('exit error on success')


# --- Additional iterable fixtures ---

class RaisingIterable:
  """Sync iterable that raises at a specific index."""
  def __init__(self, n, raise_at):
    self.n = n
    self.raise_at = raise_at
  def __iter__(self):
    for i in range(self.n):
      if i == self.raise_at:
        raise RuntimeError('iteration error')
      yield i

class InfiniteIterable:
  """Iterable that yields forever (for break testing)."""
  def __init__(self, start=0):
    self.start = start
  def __iter__(self):
    i = self.start
    while True:
      yield i
      i += 1

class AsyncInfiniteIterable:
  """Async iterable that yields forever."""
  def __init__(self, start=0):
    self.start = start
  def __aiter__(self):
    self._i = self.start
    return self
  async def __anext__(self):
    val = self._i
    self._i += 1
    return val


# --- Exception fixtures ---

class CustomException(Exception):
  """Custom exception for testing exception type filtering."""
  pass

class CustomBaseException(BaseException):
  """Custom BaseException for testing BaseException handling."""
  pass

class NestedCustomException(CustomException):
  """Subclass of CustomException for inheritance testing."""
  pass


# --- Utility functions ---

def make_tracker():
  """Return a callable that records calls and a list to inspect."""
  calls = []
  def fn(*args, **kwargs):
    calls.append((args, kwargs))
    return 'tracked'
  fn.calls = calls
  return fn

def make_async_tracker():
  """Return an async callable that records calls and a list to inspect."""
  calls = []
  async def fn(*args, **kwargs):
    calls.append((args, kwargs))
    return 'tracked'
  fn.calls = calls
  return fn
