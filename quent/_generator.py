# SPDX-License-Identifier: MIT
"""Generator wrappers for pipeline iteration (iterate/iterate_do)."""

from __future__ import annotations

import sys
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ._buffer_ops import _async_buffer_iter, _sync_buffer_iter
from ._engine import _perf_counter_ns, _record_step, _run_async_finally, _run_sync_finally, _timing_ctx
from ._engine import _run as _engine_run
from ._eval import _bind_callback, _evaluate_value, _isawaitable, _should_use_async_protocol
from ._eval import _resolve_signal_value as _eval_resolved_signal_value
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._traceback import _modify_traceback
from ._types import Null, _Break, _ControlFlowSignal, _Exit, _Return, _UncopyableMixin

_T = TypeVar('_T')

if TYPE_CHECKING:
  from ._q import Q

# Returned by _resolve_signal_value when exc.value is Null.
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
  *,
  guard_awaitable: bool = True,
) -> Any:
  """Evaluate a signal's value; returns _NO_VALUE if Null. Caller handles awaitables."""
  __tracebackhide__ = True
  if exc.value is Null:
    return _NO_VALUE
  value = exc.value
  try:
    return _eval_resolved_signal_value(exc, guard_awaitable=guard_awaitable)
  except BaseException as eval_exc:
    _handle_iterate_exc(eval_exc, link, q, ignore_result, value, idx)
    raise eval_exc  # explicit so modified __traceback__ is respected on Python <3.11


def _close_and_raise_sync(awaitable: Any, signal_name: str) -> None:
  __tracebackhide__ = True
  if hasattr(awaitable, 'close'):
    awaitable.close()
  msg = (
    f'iterate() {signal_name} value resolved to a coroutine. '
    f'Use "async for" with __aiter__ instead of "for" with __iter__.'
  )
  raise TypeError(msg) from None


def _run_deferred_finally_sync(q: Q[Any], deferred: list[Any]) -> None:
  __tracebackhide__ = True
  root_value, root_link, exec_id = deferred[0], deferred[1], deferred[2]
  active_exc = sys.exc_info()[1]
  if isinstance(active_exc, (GeneratorExit, _ControlFlowSignal)):
    active_exc = None
  result = _run_sync_finally(q, root_value, root_link, active_exc, exec_id=exec_id, wrap_awaitable=False)
  if result is not None:
    if hasattr(result, 'close'):
      result.close()
    msg = "Sync iteration pipeline's finally_() handler returned a coroutine; use 'async for' instead of 'for'."
    raise TypeError(msg)


async def _run_deferred_finally_async(q: Q[Any], deferred: list[Any]) -> None:
  __tracebackhide__ = True
  root_value, root_link, exec_id = deferred[0], deferred[1], deferred[2]
  active_exc = sys.exc_info()[1]
  if isinstance(active_exc, (GeneratorExit, _ControlFlowSignal)):
    active_exc = None
  await _run_async_finally(q, root_value, root_link, active_exc, exec_id=exec_id)


def _sync_cm_exit_clean(cm: Any) -> None:
  """Exit sync CM cleanly; warns if __exit__ returns a coroutine."""
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
  __tracebackhide__ = True
  if use_async:
    await cm.__aexit__(None, None, None)
  else:
    exit_result = cm.__exit__(None, None, None)
    if _isawaitable(exit_result):
      await exit_result


def _cm_exc_exit_sync(cm: Any) -> bool:
  """Sync CM __exit__ with active exception info. Returns True if suppressed."""
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
  """CM __exit__/__aexit__ with active exception info. Returns True if suppressed."""
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


def _build_run_kwargs(
  deferred: list[Any] | None,
  deferred_with: tuple[Link, bool] | None,
) -> dict[str, Any] | None:
  """Returns None on fast path; otherwise dict with deferred_finally/deferred_with."""
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
  """Call iteration callback; raises TypeError if it returns a coroutine."""
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
  except _ControlFlowSignal:
    raise
  except BaseException as exc:
    _handle_iterate_exc(exc, link, q, ignore_result, item, idx)
    raise exc  # `raise exc` (not bare `raise`) so modified __traceback__ is respected on Python <3.11.


async def _call_fn_async(
  fn: Callable[[Any], Any],
  item: Any,
  link: Link | None,
  q: Q[Any] | None,
  ignore_result: bool,
  idx: int,
) -> Any:
  """Call iteration callback; awaits awaitable results."""
  __tracebackhide__ = True
  try:
    result = fn(item)
    if _isawaitable(result):
      result = await result
    return result
  except _ControlFlowSignal:
    raise
  except BaseException as exc:
    _handle_iterate_exc(exc, link, q, ignore_result, item, idx)
    raise exc


def _flush_sync(flush: Callable[[], Any]) -> Any:
  __tracebackhide__ = True
  flush_result = flush()
  if _isawaitable(flush_result):
    if hasattr(flush_result, 'close'):
      flush_result.close()
    msg = "iterate(): flush callable returned a coroutine. Use 'async for' with __aiter__ instead."
    raise TypeError(msg)
  return flush_result


async def _flush_async(flush: Callable[[], Any]) -> Any:
  __tracebackhide__ = True
  flush_result = flush()
  if _isawaitable(flush_result):
    flush_result = await flush_result
  return flush_result


def _start_deferred_with(q: Q[Any] | None) -> tuple[Any, bool, int] | None:
  if q is None:
    return None
  on_step, debug, needs_timing = _timing_ctx(q)
  return (on_step, debug, _perf_counter_ns()) if needs_timing else None


def _record_deferred_with(
  q: Q[Any] | None,
  timing: tuple[Any, bool, int] | None,
  cm: Any,
  result: Any,
  ignore_result: bool,
  deferred: list[Any] | None,
  exception: BaseException | None = None,
) -> None:
  if timing is None or q is None:
    return
  link = q._current_link or q._first_link
  if link is None:
    return
  on_step, debug, started = timing
  _record_step(
    q,
    link,
    q._root_link,
    cm,
    result,
    started,
    on_step,
    debug,
    step_name='with_do' if ignore_result else 'with_',
    exec_id=deferred[2] if deferred is not None else 0,
    exception=exception,
  )


# Module-level so we don't bind `self` in the generator closure.


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
  __tracebackhide__ = True
  _has_deferred = q is not None and q._on_finally_link is not None
  _deferred: list[Any] | None = [Null, None, 0] if _has_deferred else None
  _with_cm: Any = None
  _cm_exited = False

  try:
    run_kw = _build_run_kwargs(_deferred, deferred_with)
    if isinstance(q_run, partial) and q_run.func is _engine_run:
      # Only our engine supports raw rejection; public custom runners keep their signature.
      if run_kw is None:
        run_kw = {}
      run_kw['raw_return_awaitable'] = True
    result = _run_pipeline(q_run, run_args, run_kw)

    if _isawaitable(result):
      # Coroutines have .close(); Tasks/Futures have .cancel().
      if hasattr(result, 'close'):
        result.close()
      elif hasattr(result, 'cancel'):
        result.cancel()
      msg = "Cannot use sync iteration on an async pipeline; use 'async for' instead"
      raise TypeError(msg)

    idx = 0
    _buffer = None
    try:
      if deferred_with is not None:
        _dw_inner_link, _dw_ignore_result = deferred_with
        cm = result
        _with_timing = _start_deferred_with(q)
        try:
          if not hasattr(cm, '__enter__'):
            msg = "Context manager does not support sync protocol; use 'async for' instead."
            raise TypeError(msg)
          ctx = cm.__enter__()
          _with_cm = cm
          inner_result = _evaluate_value(_dw_inner_link, ctx)
          if _isawaitable(inner_result):
            if hasattr(inner_result, 'close'):
              inner_result.close()
            msg = (
              "iterate(): deferred with_ inner function returned a coroutine. Use 'async for' with __aiter__ instead."
            )
            raise TypeError(msg)
          result = cm if _dw_ignore_result else inner_result
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _record_deferred_with(q, _with_timing, cm, None, _dw_ignore_result, _deferred, exc)
          raise
        else:
          _record_deferred_with(q, _with_timing, cm, result, _dw_ignore_result, _deferred)

      if buffer_size is not None:
        _buffer = _sync_buffer_iter(result, buffer_size)
        result = _buffer

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

        if flush is not None:
          for sub in _flush_sync(flush):
            yield sub

      finally:
        if _buffer is not None:
          active_exc = sys.exc_info()[1]
          try:
            _buffer.close()
          except BaseException as cleanup_exc:
            if active_exc is not None and cleanup_exc is not active_exc:
              cleanup_exc.__context__ = active_exc
            raise

    except (_Break, _Return, _Exit) as exc:
      resolved = _resolve_signal_value(exc, link, q, ignore_result, idx, guard_awaitable=False)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          _close_and_raise_sync(resolved, type(exc).__name__)
        yield resolved
      return

  except BaseException:
    if _with_cm is not None:
      _cm_exited = True
      if _cm_exc_exit_sync(_with_cm):
        return  # CM suppressed.
    raise

  finally:
    # Clean-exit (non-exception paths) + deferred finally.
    # Exception-path CM exit handled in except above; _cm_exited prevents double-exit.
    try:
      if _with_cm is not None and not _cm_exited:
        _sync_cm_exit_clean(_with_cm)
    finally:
      if _deferred is not None:
        assert q is not None  # guaranteed by _has_deferred guard
        _run_deferred_finally_sync(q, _deferred)


async def _aiter_wrap(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
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
  __tracebackhide__ = True
  _has_deferred = q is not None and q._on_finally_link is not None
  _deferred: list[Any] | None = [Null, None, 0] if _has_deferred else None
  _with_cm: Any = None
  _use_async_cm = False
  _cm_exited = False

  try:
    iterator = _run_pipeline(q_run, run_args, _build_run_kwargs(_deferred, deferred_with))
    if _isawaitable(iterator):
      iterator = await iterator

    idx = 0
    _buffer = None
    try:
      if deferred_with is not None:
        _dw_inner_link, _dw_ignore_result = deferred_with
        cm = iterator
        _with_timing = _start_deferred_with(q)
        try:
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
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _record_deferred_with(q, _with_timing, cm, None, _dw_ignore_result, _deferred, exc)
          raise
        else:
          _record_deferred_with(q, _with_timing, cm, iterator, _dw_ignore_result, _deferred)

      if buffer_size is not None:
        _buffer = _async_buffer_iter(iterator, buffer_size)
        iterator = _buffer

      if not hasattr(iterator, '__aiter__'):
        iterator = _aiter_wrap(iterator)

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

        if flush is not None:
          flush_result = await _flush_async(flush)
          if hasattr(flush_result, '__aiter__'):
            async for sub in flush_result:
              yield sub
          else:
            for sub in flush_result:
              yield sub

      finally:
        if _buffer is not None:
          active_exc = sys.exc_info()[1]
          try:
            await _buffer.aclose()
          except BaseException as cleanup_exc:
            # aclose injects GeneratorExit; retain the consumer failure it replaces.
            if active_exc is not None and cleanup_exc is not active_exc:
              cleanup_exc.__context__ = active_exc
            raise

    except (_Break, _Return, _Exit) as exc:
      signal_value = exc.value
      resolved = _resolve_signal_value(exc, link, q, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          try:
            resolved = await resolved
          except BaseException as await_exc:
            _handle_iterate_exc(await_exc, link, q, ignore_result, signal_value, idx)
            raise await_exc
        yield resolved
      return

  except BaseException:
    if _with_cm is not None:
      _cm_exited = True
      if await _cm_exc_exit_async(_with_cm, _use_async_cm):
        return  # CM suppressed.
    raise

  finally:
    try:
      if _with_cm is not None and not _cm_exited:
        await _async_cm_exit_clean(_with_cm, _use_async_cm)
    finally:
      if _deferred is not None:
        assert q is not None  # guaranteed by _has_deferred guard
        await _run_deferred_finally_async(q, _deferred)


class QuentIterator(_UncopyableMixin, Generic[_T]):
  """Dual sync/async iterable over pipeline output.

  Created by Q.iterate(). Supports both __iter__ and __aiter__. Calling the
  instance returns a new QuentIterator with updated run args (reusable).
  _T is the yielded element type (currently always Any).
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
    self._fn = _bind_callback(fn) if fn is not None else None
    self._ignore_result = ignore_result
    self._q = q
    self._link = link
    self._deferred_with = deferred_with
    self._flat = flat
    self._flush = flush
    self._buffer_size = buffer_size
    self._run_args: tuple[Any, tuple[Any, ...], dict[str, Any]] = (Null, (), {})

  def __call__(self, v: Any = Null, *args: Any, **kwargs: Any) -> QuentIterator[_T]:
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
