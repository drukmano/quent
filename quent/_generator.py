# SPDX-License-Identifier: MIT
"""Generator wrappers for pipeline iteration (iterate/iterate_do)."""

from __future__ import annotations

import sys
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ._buffer_ops import _async_buffer_iter, _sync_buffer_iter
from ._engine import _run_async_finally, _run_sync_finally
from ._eval import _eval_signal_value, _evaluate_value, _isawaitable, _should_use_async_protocol
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._traceback import _modify_traceback
from ._types import Null, _Break, _ControlFlowSignal, _Return, _UncopyableMixin

_T = TypeVar('_T')

if TYPE_CHECKING:
  from ._q import Q

# Sentinel returned by _resolve_signal_value when exc.value is Null (no value to yield).
_NO_VALUE = object()


def _handle_iterate_exc(
  exc: BaseException, link: Link | None, q: Q[Any] | None, ignore_result: bool, item: Any, idx: int
) -> None:
  """Attach source metadata to an iteration exception for traceback display."""
  __tracebackhide__ = True
  if link is not None:
    _set_link_temp_args(exc, link, item=item, index=idx)
    method = 'iterate_do' if ignore_result else 'iterate'
    _modify_traceback(exc, q, link, q._root_link if q else None, extra_links=[(link, method)])


def _resolve_signal_value(
  exc: _ControlFlowSignal,
  link: Link | None,
  q: Q[Any] | None,
  ignore_result: bool,
  idx: int,
) -> Any:
  """Evaluate a control flow signal's value, attaching traceback metadata on failure.

  Returns ``_NO_VALUE`` when the signal carries no value (``exc.value is Null``).
  Otherwise returns the resolved value (which may be an awaitable — the caller
  is responsible for handling that).  Raises on evaluation failure after
  attaching iteration metadata to the exception.
  """
  __tracebackhide__ = True
  if exc.value is Null:
    return _NO_VALUE
  try:
    return _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
  except BaseException as eval_exc:
    _handle_iterate_exc(eval_exc, link, q, ignore_result, exc.value, idx)
    raise eval_exc  # explicit raise for modified __traceback__ on Python <3.11


def _close_and_raise_sync(awaitable: Any, signal_name: str) -> None:
  """Close an awaitable that cannot be consumed synchronously, then raise TypeError."""
  __tracebackhide__ = True
  if hasattr(awaitable, 'close'):
    awaitable.close()
  msg = (
    f'iterate() {signal_name} value resolved to a coroutine. '
    f'Use "async for" with __aiter__ instead of "for" with __iter__.'
  )
  raise TypeError(msg) from None


def _run_deferred_finally_sync(q: Q[Any], deferred: list[Any]) -> None:
  """Run a deferred finally handler synchronously after iteration ends."""
  __tracebackhide__ = True
  root_value, root_link, exec_id = deferred[0], deferred[1], deferred[2]
  active_exc = sys.exc_info()[1]
  if isinstance(active_exc, (GeneratorExit, _ControlFlowSignal)):
    active_exc = None
  result = _run_sync_finally(q, root_value, root_link, active_exc, exec_id=exec_id)
  if result is not None:
    # Handler returned a coroutine — can't await in sync context.
    if hasattr(result, 'close'):
      result.close()
    msg = "Sync iteration pipeline's finally_() handler returned a coroutine; use 'async for' instead of 'for'."
    raise TypeError(msg)


async def _run_deferred_finally_async(q: Q[Any], deferred: list[Any]) -> None:
  """Run a deferred finally handler asynchronously after async iteration ends."""
  __tracebackhide__ = True
  root_value, root_link, exec_id = deferred[0], deferred[1], deferred[2]
  active_exc = sys.exc_info()[1]
  if isinstance(active_exc, (GeneratorExit, _ControlFlowSignal)):
    active_exc = None
  await _run_async_finally(q, root_value, root_link, active_exc, exec_id=exec_id)


def _sync_cm_exit_clean(cm: Any) -> None:
  """Exit a sync CM on the success/signal path. Warns if __exit__ returns a coroutine."""
  __tracebackhide__ = True
  exit_result = cm.__exit__(None, None, None)
  if _isawaitable(exit_result):
    if hasattr(exit_result, 'close'):
      exit_result.close()
    warnings.warn(
      'iterate(): CM __exit__ returned a coroutine during sync iteration; '
      "the __exit__ was not awaited. Use 'async for' for async CMs.",
      RuntimeWarning,
      stacklevel=2,
    )


async def _async_cm_exit_clean(cm: Any, use_async: bool) -> None:
  """Exit a CM (async or sync) on the success/signal path."""
  __tracebackhide__ = True
  if use_async:
    await cm.__aexit__(None, None, None)
  else:
    exit_result = cm.__exit__(None, None, None)
    if _isawaitable(exit_result):
      await exit_result


def _cm_exc_exit_sync(cm: Any) -> bool:
  """Call sync CM __exit__ with active exception info. Returns True if exception was suppressed."""
  __tracebackhide__ = True
  exc_info = sys.exc_info()
  if not isinstance(exc_info[1], (GeneratorExit, _ControlFlowSignal)):
    suppress = cm.__exit__(*exc_info)
    if _isawaitable(suppress):
      if hasattr(suppress, 'close'):
        suppress.close()
      return False
    return bool(suppress)
  _sync_cm_exit_clean(cm)
  return False


async def _cm_exc_exit_async(cm: Any, use_async: bool) -> bool:
  """Call CM __exit__/__aexit__ with active exception info. Returns True if exception was suppressed."""
  __tracebackhide__ = True
  exc_info = sys.exc_info()
  if not isinstance(exc_info[1], (GeneratorExit, _ControlFlowSignal)):
    if use_async:
      suppress = await cm.__aexit__(*exc_info)
    else:
      suppress = cm.__exit__(*exc_info)
      if _isawaitable(suppress):
        suppress = await suppress
    return bool(suppress)
  await _async_cm_exit_clean(cm, use_async)
  return False


# ---- Shared helpers for sync/async generator deduplication ----


def _build_run_kwargs(
  deferred: list[Any] | None,
  deferred_with: tuple[Link, bool] | None,
) -> dict[str, Any] | None:
  """Build keyword arguments for the pipeline run call.

  Returns ``None`` when no extra kwargs are needed (the common fast path),
  otherwise returns a dict with ``deferred_finally`` and/or ``deferred_with``
  entries.
  """
  if deferred is None and deferred_with is None:
    return None
  kw: dict[str, Any] = {}
  if deferred is not None:
    kw['deferred_finally'] = deferred
  if deferred_with is not None:
    kw['deferred_with'] = True
  return kw


def _run_pipeline(
  q_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  run_kw: dict[str, Any] | None,
) -> Any:
  """Execute the pipeline run callable with optional keyword arguments.

  Returns the raw result (may be awaitable -- caller handles that).
  """
  if run_kw is not None:
    return q_run(*run_args, **run_kw)
  return q_run(*run_args)


def _call_fn_sync(
  fn: Callable[[Any], Any],
  item: Any,
  link: Link | None,
  q: Q[Any] | None,
  ignore_result: bool,
  idx: int,
) -> Any:
  """Call the iteration callback synchronously, with error handling.

  Raises TypeError if the callback returns a coroutine.
  Attaches traceback metadata on non-control-flow exceptions.
  """
  __tracebackhide__ = True
  try:
    fn_result = fn(item)
    if _isawaitable(fn_result):
      if hasattr(fn_result, 'close'):
        fn_result.close()
      msg = (
        f'iterate() callback {fn!r} returned a coroutine. '
        f'Use "async for" with __aiter__ instead of "for" with __iter__.'
      )
      raise TypeError(msg)
    return fn_result
  except _ControlFlowSignal:  # Must propagate — not a regular exception.
    raise
  except BaseException as exc:
    _handle_iterate_exc(exc, link, q, ignore_result, item, idx)
    raise exc  # Use `raise exc` (not bare `raise`) so the modified __traceback__ is respected on Python <3.11.


async def _call_fn_async(
  fn: Callable[[Any], Any],
  item: Any,
  link: Link | None,
  q: Q[Any] | None,
  ignore_result: bool,
  idx: int,
) -> Any:
  """Call the iteration callback with async support and error handling.

  If the callback returns an awaitable, it is awaited.
  Attaches traceback metadata on non-control-flow exceptions.
  """
  __tracebackhide__ = True
  try:
    result = fn(item)
    if _isawaitable(result):
      result = await result
    return result
  except _ControlFlowSignal:  # Must propagate — not a regular exception.
    raise
  except BaseException as exc:
    _handle_iterate_exc(exc, link, q, ignore_result, item, idx)
    raise exc  # Use `raise exc` (not bare `raise`) so the modified __traceback__ is respected on Python <3.11.


def _flush_sync(flush: Callable[[], Any]) -> Any:
  """Call the flush callable synchronously, raising TypeError if it returns a coroutine."""
  __tracebackhide__ = True
  flush_result = flush()
  if _isawaitable(flush_result):
    if hasattr(flush_result, 'close'):
      flush_result.close()
    msg = "iterate(): flush callable returned a coroutine. Use 'async for' with __aiter__ instead."
    raise TypeError(msg)
  return flush_result


async def _flush_async(flush: Callable[[], Any]) -> Any:
  """Call the flush callable with async support."""
  __tracebackhide__ = True
  flush_result = flush()
  if _isawaitable(flush_result):
    flush_result = await flush_result
  return flush_result


# Module-level functions (not methods) to avoid binding `self` in the generator closure.


def _sync_generator(
  q_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  q: Q[Any] | None = None,
  link: Link | None = None,
  deferred_with: tuple[Link, bool] | None = None,
  flat: bool = False,
  flush: Callable[[], Any] | None = None,
  buffer_size: int | None = None,
) -> Iterator[Any]:
  # SYNC MIRROR of _async_generator — keep both in sync when modifying.
  # Intentional divergences: sync closes awaitables + raises TypeError; async awaits them.
  """Synchronous generator that yields each element of the pipeline's output."""
  __tracebackhide__ = True
  _has_deferred = q is not None and q._on_finally_link is not None
  _deferred: list[Any] | None = [Null, None, 0] if _has_deferred else None
  _with_cm: Any = None  # Tracks entered CM for cleanup
  _cm_exited = False

  try:
    # Run the pipeline
    result = _run_pipeline(q_run, run_args, _build_run_kwargs(_deferred, deferred_with))

    if _isawaitable(result):
      # Coroutines have .close(); Tasks/Futures have .cancel().
      if hasattr(result, 'close'):
        result.close()
      elif hasattr(result, 'cancel'):
        result.cancel()
      msg = "Cannot use sync iteration on an async pipeline; use 'async for' instead"
      raise TypeError(msg)

    # Enter deferred CM if active
    if deferred_with is not None:
      _dw_inner_link, _dw_ignore_result = deferred_with
      cm = result
      if not hasattr(cm, '__enter__'):
        msg = "Context manager does not support sync protocol; use 'async for' instead."
        raise TypeError(msg)
      ctx = cm.__enter__()
      _with_cm = cm
      inner_result = _evaluate_value(_dw_inner_link, ctx)
      if _isawaitable(inner_result):
        if hasattr(inner_result, 'close'):
          inner_result.close()
        msg = "iterate(): deferred with_ inner function returned a coroutine. Use 'async for' with __aiter__ instead."
        raise TypeError(msg)
      result = cm if _dw_ignore_result else inner_result

    # Wrap with buffer if requested
    if buffer_size is not None:
      result = _sync_buffer_iter(result, buffer_size)

    # Iteration loop
    idx = 0
    try:
      for item in result:
        if fn is None:
          if flat:
            for sub in item:
              yield sub
          else:
            yield item
        else:
          fn_result = _call_fn_sync(fn, item, link, q, ignore_result, idx)
          if flat:
            if ignore_result:
              for _unused in fn_result:
                pass
              yield item
            else:
              for sub in fn_result:
                yield sub
          else:
            yield item if ignore_result else fn_result
        idx += 1

      # flush after source exhaustion (flat mode only)
      if flush is not None:
        for sub in _flush_sync(flush):
          yield sub

    except (_Break, _Return) as exc:
      resolved = _resolve_signal_value(exc, link, q, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          _close_and_raise_sync(resolved, type(exc).__name__)
        yield resolved
      return

  except BaseException:
    # CM exception-path exit — handles __exit__(exc) and suppression.
    if _with_cm is not None:
      _cm_exited = True
      if _cm_exc_exit_sync(_with_cm):
        return  # CM suppressed the exception.
    raise

  finally:
    # Cleanup: CM clean-exit (non-exception paths) + deferred finally.
    # Exception-path CM exit is handled in except above; _cm_exited prevents double-exit.
    try:
      if _with_cm is not None and not _cm_exited:
        _sync_cm_exit_clean(_with_cm)
    finally:
      if _deferred is not None:
        assert q is not None  # guaranteed by _has_deferred guard
        _run_deferred_finally_sync(q, _deferred)


async def _aiter_wrap(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
  """Wrap a synchronous iterator as an async iterator."""
  __tracebackhide__ = True
  for item in sync_iter:
    yield item


async def _async_generator(
  q_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  q: Q[Any] | None = None,
  link: Link | None = None,
  deferred_with: tuple[Link, bool] | None = None,
  flat: bool = False,
  flush: Callable[[], Any] | None = None,
  buffer_size: int | None = None,
) -> AsyncIterator[Any]:
  # ASYNC MIRROR of _sync_generator — keep both in sync when modifying.
  # Intentional divergences: async awaits awaitables; sync closes them + raises TypeError.
  """Asynchronous generator that yields each element of the pipeline's output."""
  __tracebackhide__ = True
  _has_deferred = q is not None and q._on_finally_link is not None
  _deferred: list[Any] | None = [Null, None, 0] if _has_deferred else None
  _with_cm: Any = None
  _use_async_cm = False
  _cm_exited = False

  try:
    # Run the pipeline
    iterator = _run_pipeline(q_run, run_args, _build_run_kwargs(_deferred, deferred_with))
    if _isawaitable(iterator):
      iterator = await iterator

    # Enter deferred CM if active
    if deferred_with is not None:
      _dw_inner_link, _dw_ignore_result = deferred_with
      cm = iterator
      _use_async_cm_result = _should_use_async_protocol(cm, '__enter__', '__aenter__')
      if _use_async_cm_result is True:
        _use_async_cm = True
        ctx = await cm.__aenter__()
      elif _use_async_cm_result is False:
        ctx = cm.__enter__()
      else:
        msg = (
          f'{type(cm).__name__} object does not support the context manager protocol '
          f'(__enter__/__exit__ or __aenter__/__aexit__). '
          f'Ensure the pipeline value at this step is a context manager.'
        )
        raise TypeError(msg)
      _with_cm = cm
      inner_result = _evaluate_value(_dw_inner_link, ctx)
      if _isawaitable(inner_result):
        inner_result = await inner_result
      iterator = cm if _dw_ignore_result else inner_result

    # Wrap with buffer if requested
    if buffer_size is not None:
      iterator = _async_buffer_iter(iterator, buffer_size)

    if not hasattr(iterator, '__aiter__'):
      iterator = _aiter_wrap(iterator)

    # Iteration loop
    idx = 0
    try:
      async for item in iterator:
        if fn is None:
          if flat:
            if hasattr(item, '__aiter__'):
              async for sub in item:
                yield sub
            else:
              for sub in item:
                yield sub
          else:
            yield item
        else:
          result = await _call_fn_async(fn, item, link, q, ignore_result, idx)
          if flat:
            if ignore_result:
              if hasattr(result, '__aiter__'):
                async for _unused in result:
                  pass
              else:
                for _unused in result:
                  pass
              yield item
            else:
              if hasattr(result, '__aiter__'):
                async for sub in result:
                  yield sub
              else:
                for sub in result:
                  yield sub
          else:
            yield item if ignore_result else result
        idx += 1

      # flush after source exhaustion
      if flush is not None:
        flush_result = await _flush_async(flush)
        if hasattr(flush_result, '__aiter__'):
          async for sub in flush_result:
            yield sub
        else:
          for sub in flush_result:
            yield sub

    except (_Break, _Return) as exc:
      resolved = _resolve_signal_value(exc, link, q, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          try:
            resolved = await resolved
          except BaseException as await_exc:
            _handle_iterate_exc(await_exc, link, q, ignore_result, exc.value, idx)
            raise await_exc
        yield resolved
      return

  except BaseException:
    # CM exception-path exit — handles __exit__(exc) / __aexit__(exc) and suppression.
    if _with_cm is not None:
      _cm_exited = True
      if await _cm_exc_exit_async(_with_cm, _use_async_cm):
        return  # CM suppressed the exception.
    raise

  finally:
    # Cleanup: CM clean-exit (non-exception paths) + deferred finally.
    # Exception-path CM exit is handled in except above; _cm_exited prevents double-exit.
    try:
      if _with_cm is not None and not _cm_exited:
        await _async_cm_exit_clean(_with_cm, _use_async_cm)
    finally:
      if _deferred is not None:
        assert q is not None  # guaranteed by _has_deferred guard
        await _run_deferred_finally_async(q, _deferred)


class QuentIterator(_UncopyableMixin, Generic[_T]):
  """Wraps pipeline output as a dual sync/async iterable.

  Created by ``Q.iterate()``. Supports both ``__iter__`` and
  ``__aiter__``, choosing the appropriate generator at iteration time.
  Calling the instance returns a new ``QuentIterator`` with updated run args,
  making generators reusable with different inputs.

  Generic over ``_T``, the element type yielded during iteration.
  Currently ``_T`` is ``Any`` (the element type cannot be statically
  inferred from ``Q[T]``), but the parameter is in place for future
  refinement.
  """

  __slots__ = (
    '_buffer_size',
    '_deferred_with',
    '_flat',
    '_flush',
    '_fn',
    '_ignore_result',
    '_link',
    '_q',
    '_q_run',
    '_run_args',
  )

  _buffer_size: int | None
  _q: Q[Any] | None
  _q_run: Callable[..., Any]
  _deferred_with: tuple[Link, bool] | None
  _flat: bool
  _flush: Callable[[], Any] | None
  _fn: Callable[[Any], Any] | None
  _ignore_result: bool
  _link: Link | None
  _run_args: tuple[Any, tuple[Any, ...], dict[str, Any]]

  def __init__(
    self,
    q_run: Callable[..., Any],
    fn: Callable[[Any], Any] | None,
    ignore_result: bool,
    q: Q[Any] | None = None,
    link: Link | None = None,
    deferred_with: tuple[Link, bool] | None = None,
    flat: bool = False,
    flush: Callable[[], Any] | None = None,
    buffer_size: int | None = None,
  ) -> None:
    self._q_run = q_run
    self._fn = fn
    self._ignore_result = ignore_result
    self._q = q
    self._link = link
    self._deferred_with = deferred_with
    self._flat = flat
    self._flush = flush
    self._buffer_size = buffer_size
    self._run_args: tuple[Any, tuple[Any, ...], dict[str, Any]] = (Null, (), {})

  def __call__(self, v: Any = Null, *args: Any, **kwargs: Any) -> QuentIterator[_T]:
    """Return a new ``QuentIterator`` with updated ``_run_args``, enabling reuse with different inputs."""
    g: QuentIterator[_T] = QuentIterator(
      self._q_run,
      self._fn,
      self._ignore_result,
      self._q,
      self._link,
      deferred_with=self._deferred_with,
      flat=self._flat,
      flush=self._flush,
      buffer_size=self._buffer_size,
    )
    g._run_args = (v, args, kwargs)
    return g

  def __iter__(self) -> Iterator[_T]:
    """Delegate to the module-level ``_sync_generator`` function."""
    return _sync_generator(
      self._q_run,
      self._run_args,
      self._fn,
      self._ignore_result,
      self._q,
      self._link,
      deferred_with=self._deferred_with,
      flat=self._flat,
      flush=self._flush,
      buffer_size=self._buffer_size,
    )

  def __aiter__(self) -> AsyncIterator[_T]:
    """Delegate to the module-level ``_async_generator`` function."""
    return _async_generator(
      self._q_run,
      self._run_args,
      self._fn,
      self._ignore_result,
      self._q,
      self._link,
      deferred_with=self._deferred_with,
      flat=self._flat,
      flush=self._flush,
      buffer_size=self._buffer_size,
    )

  def __repr__(self) -> str:
    return '<quent.QuentIterator>'
