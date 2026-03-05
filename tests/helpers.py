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
