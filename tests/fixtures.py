"""Shared test fixtures: sentinels, result capture, callable pairs, CMs, iterables, variant axes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

_UNSET = object()


# ---------------------------------------------------------------------------
# Result capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Result:
  """Captured pipeline execution result."""

  success: bool
  value: Any = None
  exc_type: type | None = None
  exc_message: str | None = None
  sub_exc_types: frozenset[type] | None = None


def _extract_sub_exc_types(exc: BaseException) -> frozenset[type] | None:
  """Extract sub-exception types from an ExceptionGroup, or None."""
  if hasattr(exc, 'exceptions') and exc.exceptions:
    return frozenset(type(e) for e in exc.exceptions)
  return None


async def capture(fn: Any) -> Result:
  """Execute fn(), await if coroutine, capture result or exception."""
  try:
    result = fn()
    if asyncio.iscoroutine(result):
      result = await result
    return Result(success=True, value=result)
  except BaseException as exc:
    return Result(
      success=False,
      exc_type=type(exc),
      exc_message=str(exc),
      sub_exc_types=_extract_sub_exc_types(exc),
    )


# ---------------------------------------------------------------------------
# Callable fixtures -- sync/async pairs
# ---------------------------------------------------------------------------


def sync_fn(x: Any) -> Any:
  return x + 1


async def async_fn(x: Any) -> Any:
  return x + 1


def sync_identity(x: Any) -> Any:
  return x


async def async_identity(x: Any) -> Any:
  return x


def sync_double(x: Any) -> Any:
  return x * 2


async def async_double(x: Any) -> Any:
  return x * 2


def sync_is_even(x: Any) -> bool:
  return x % 2 == 0


async def async_is_even(x: Any) -> bool:
  return x % 2 == 0


def sync_is_truthy(x: Any) -> bool:
  return bool(x)


async def async_is_truthy(x: Any) -> bool:
  return bool(x)


def sync_raise(x: Any) -> Any:
  raise ValueError('test error')


async def async_raise(x: Any) -> Any:
  raise ValueError('test error')


def sync_noop(x: Any) -> None:
  return None


async def async_noop(x: Any) -> None:
  return None


def sync_always_true(x: Any) -> bool:
  return True


async def async_always_true(x: Any) -> bool:
  return True


def sync_always_false(x: Any) -> bool:
  return False


async def async_always_false(x: Any) -> bool:
  return False


# Multi-arg fixtures for calling convention testing
def sync_add(a: Any, b: Any) -> Any:
  return a + b


async def async_add(a: Any, b: Any) -> Any:
  return a + b


def sync_kw(*, key: Any) -> Any:
  return key


async def async_kw(*, key: Any) -> Any:
  return key


def sync_triple(x: Any) -> Any:
  return x * 3


async def async_triple(x: Any) -> Any:
  return x * 3


def sync_gt0(x: Any) -> bool:
  return x > 0


async def async_gt0(x: Any) -> bool:
  return x > 0


# Error handler fixtures (for except_() axis testing)
def sync_handler(info: Any) -> str:
  return 'handled'


async def async_handler(info: Any) -> str:
  return 'handled'


# Cleanup handler that raises (for finally_() failure testing)
def sync_bad_cleanup(rv: Any) -> None:
  raise RuntimeError('cleanup boom')


async def async_bad_cleanup(rv: Any) -> None:
  raise RuntimeError('cleanup boom')


# ---------------------------------------------------------------------------
# Context manager fixtures
# ---------------------------------------------------------------------------


class SyncCM:
  """Sync CM returning a numeric value for pipeline compatibility."""

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  def __enter__(self) -> Any:
    return self._value

  def __exit__(self, *args: Any) -> bool:
    return False


class AsyncCM:
  """Async CM returning a numeric value for pipeline compatibility."""

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  async def __aenter__(self) -> Any:
    return self._value

  async def __aexit__(self, *args: Any) -> bool:
    return False


class SyncCMSuppresses:
  def __enter__(self) -> Any:
    return 10

  def __exit__(self, *args: Any) -> bool:
    return True


class AsyncCMSuppresses:
  async def __aenter__(self) -> Any:
    return 10

  async def __aexit__(self, *args: Any) -> bool:
    return True


class DualProtocolCM:
  """Dual-protocol CM: sync __enter__/__exit__ + async __aenter__/__aexit__.

  The bridge runner exercises both protocols naturally:
  - Pure sync permutations use __enter__/__exit__
  - Async permutations (with running event loop) prefer __aenter__/__aexit__
  Both return the same numeric value for pipeline compatibility.
  """

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  def __enter__(self) -> Any:
    return self._value

  def __exit__(self, *args: Any) -> bool:
    return False

  async def __aenter__(self) -> Any:
    return self._value

  async def __aexit__(self, *args: Any) -> bool:
    return False


class SyncCMAsyncExit:
  """Sync CM whose __exit__ returns a coroutine -- triggers async transition on exit.

  Has sync __enter__ but __exit__ returns an awaitable, exercising
  the _await_exit_success / _await_exit_suppress / _await_exit_signal
  code paths in _sync_cm.
  """

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  def __enter__(self) -> Any:
    return self._value

  def __exit__(self, *args: Any) -> Any:
    async def _exit() -> bool:
      return False

    return _exit()


# ---------------------------------------------------------------------------
# Iterable fixtures
# ---------------------------------------------------------------------------


class AsyncRange:
  """Async iterable over range(n)."""

  def __init__(self, n: int) -> None:
    self._n = n

  def __aiter__(self) -> Any:
    return self._gen()

  async def _gen(self) -> Any:
    for i in range(self._n):
      yield i


class AsyncPair:
  """Async iterable yielding [x, x+1] for pipeline-numeric compatibility."""

  def __init__(self, x: Any) -> None:
    self._x = x

  def __aiter__(self) -> Any:
    return self._gen()

  async def _gen(self) -> Any:
    yield self._x
    yield self._x + 1


# ---------------------------------------------------------------------------
# Exception fixtures
# ---------------------------------------------------------------------------


class CustomError(Exception):
  """Custom exception for type-specific testing."""


class CustomBaseError(BaseException):
  """Custom BaseException for BaseException filtering tests."""


# ---------------------------------------------------------------------------
# Variant axes -- sync/async dimensions
# ---------------------------------------------------------------------------

VariantAxis = list[tuple[str, Any]]

V_FN: VariantAxis = [('sync', sync_fn), ('async', async_fn)]
V_IDENTITY: VariantAxis = [('sync', sync_identity), ('async', async_identity)]
V_DOUBLE: VariantAxis = [('sync', sync_double), ('async', async_double)]
V_IS_EVEN: VariantAxis = [('sync', sync_is_even), ('async', async_is_even)]
V_IS_TRUTHY: VariantAxis = [('sync', sync_is_truthy), ('async', async_is_truthy)]
V_RAISE: VariantAxis = [('sync', sync_raise), ('async', async_raise)]
V_NOOP: VariantAxis = [('sync', sync_noop), ('async', async_noop)]
V_TRUE: VariantAxis = [('sync', sync_always_true), ('async', async_always_true)]
V_FALSE: VariantAxis = [('sync', sync_always_false), ('async', async_always_false)]
V_CM: VariantAxis = [('sync', SyncCM), ('async', AsyncCM)]
V_CM_SUPPRESSES: VariantAxis = [('sync', SyncCMSuppresses), ('async', AsyncCMSuppresses)]
V_ITER: VariantAxis = [('list', list(range(5))), ('async', AsyncRange(5))]
V_ADD: VariantAxis = [('sync', sync_add), ('async', async_add)]
V_KW: VariantAxis = [('sync', sync_kw), ('async', async_kw)]
V_TRIPLE: VariantAxis = [('sync', sync_triple), ('async', async_triple)]
V_GT0: VariantAxis = [('sync', sync_gt0), ('async', async_gt0)]
V_HANDLER: VariantAxis = [('sync', sync_handler), ('async', async_handler)]
V_BAD_CLEANUP: VariantAxis = [('sync', sync_bad_cleanup), ('async', async_bad_cleanup)]
