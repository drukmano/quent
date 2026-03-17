# SPDX-License-Identifier: MIT
"""Generator wrappers for chain iteration (iterate/iterate_do)."""

from __future__ import annotations

import sys
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ._engine import _run_async_finally, _run_sync_finally
from ._eval import _eval_signal_value, _evaluate_value, _isawaitable, _should_use_async_protocol
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._traceback import _modify_traceback
from ._types import Null, _Break, _ControlFlowSignal, _Return, _UncopyableMixin

_T = TypeVar('_T')

if TYPE_CHECKING:
  from ._chain import Chain

# Sentinel returned by _resolve_signal_value when exc.value is Null (no value to yield).
_NO_VALUE = object()


def _handle_iterate_exc(
  exc: BaseException, link: Link | None, chain: Chain[Any] | None, ignore_result: bool, item: Any, idx: int
) -> None:
  """Attach source metadata to an iteration exception for traceback display."""
  __tracebackhide__ = True
  if link is not None:
    _set_link_temp_args(exc, link, item=item, index=idx)
    method = 'iterate_do' if ignore_result else 'iterate'
    _modify_traceback(exc, chain, link, chain.root_link if chain else None, extra_links=[(link, method)])


def _resolve_signal_value(
  exc: _ControlFlowSignal,
  link: Link | None,
  chain: Chain[Any] | None,
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
    _handle_iterate_exc(eval_exc, link, chain, ignore_result, exc.value, idx)
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


def _run_deferred_finally_sync(chain: Chain[Any], deferred: list[Any]) -> None:
  """Run a deferred finally handler synchronously after iteration ends."""
  __tracebackhide__ = True
  root_value, root_link, exec_id = deferred[0], deferred[1], deferred[2]
  active_exc = sys.exc_info()[1]
  if isinstance(active_exc, (GeneratorExit, _ControlFlowSignal)):
    active_exc = None
  result = _run_sync_finally(chain, root_value, root_link, active_exc, exec_id=exec_id)
  if result is not None:
    # Handler returned a coroutine — can't await in sync context.
    if hasattr(result, 'close'):
      result.close()
    warnings.warn(
      'iterate(): finally handler returned a coroutine during sync iteration; '
      "the handler was not awaited. Use 'async for' to await async finally handlers.",
      RuntimeWarning,
      stacklevel=2,
    )


async def _run_deferred_finally_async(chain: Chain[Any], deferred: list[Any]) -> None:
  """Run a deferred finally handler asynchronously after async iteration ends."""
  __tracebackhide__ = True
  root_value, root_link, exec_id = deferred[0], deferred[1], deferred[2]
  active_exc = sys.exc_info()[1]
  if isinstance(active_exc, (GeneratorExit, _ControlFlowSignal)):
    active_exc = None
  await _run_async_finally(chain, root_value, root_link, active_exc, exec_id=exec_id)


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


# Module-level functions (not methods) to avoid binding `self` in the generator closure.


def _sync_generator(
  chain_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  chain: Chain[Any] | None = None,
  link: Link | None = None,
  deferred_with: tuple[Link | None, bool] | None = None,
  flat: bool = False,
  on_exhaust: Callable[[], Any] | None = None,
) -> Iterator[Any]:
  # SYNC MIRROR of _async_generator — keep both in sync when modifying.
  # Intentional divergences: sync closes awaitables + raises TypeError; async awaits them.
  """Synchronous generator that yields each element of the chain's output."""
  __tracebackhide__ = True
  _has_deferred = chain is not None and chain.on_finally_link is not None
  _deferred: list[Any] | None = [Null, None, 0] if _has_deferred else None
  _with_cm: Any = None  # Tracks entered CM for cleanup

  try:
    # Run the chain pipeline
    _has_kw = _deferred is not None or deferred_with is not None
    if _has_kw:
      _kw: dict[str, Any] = {}
      if _deferred is not None:
        _kw['deferred_finally'] = _deferred
      if deferred_with is not None:
        _kw['deferred_with'] = True
      result = chain_run(*run_args, **_kw)
    else:
      result = chain_run(*run_args)

    if _isawaitable(result):
      # Coroutines have .close(); Tasks/Futures have .cancel().
      if hasattr(result, 'close'):
        result.close()
      elif hasattr(result, 'cancel'):
        result.cancel()
      msg = "Cannot use sync iteration on an async chain; use 'async for' instead"
      raise TypeError(msg)

    # Enter deferred CM if active
    if deferred_with is not None:
      _dw_inner_link, _dw_ignore_result = deferred_with
      cm = result
      ctx = cm.__enter__()
      _with_cm = cm
      if _dw_inner_link is not None:
        inner_result = _evaluate_value(_dw_inner_link, ctx)
        if _isawaitable(inner_result):
          if hasattr(inner_result, 'close'):
            inner_result.close()
          msg = "iterate(): deferred with_ inner function returned a coroutine. Use 'async for' with __aiter__ instead."
          raise TypeError(msg)
        result = cm if _dw_ignore_result else inner_result
      else:
        # Bare with_(): context value IS the iterable
        result = ctx

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
          except _ControlFlowSignal:  # Must propagate — not a regular exception.
            raise
          except BaseException as exc:
            _handle_iterate_exc(exc, link, chain, ignore_result, item, idx)
            raise exc  # Use `raise exc` (not bare `raise`) so the modified __traceback__ is respected on Python <3.11.
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

      # on_exhaust after source exhaustion (flat mode only)
      if on_exhaust is not None:
        exhaust_result = on_exhaust()
        if _isawaitable(exhaust_result):
          if hasattr(exhaust_result, 'close'):
            exhaust_result.close()
          msg = "iterate(): on_exhaust callable returned a coroutine. Use 'async for' with __aiter__ instead."
          raise TypeError(msg)
        for sub in exhaust_result:
          yield sub

    except _Break as exc:
      resolved = _resolve_signal_value(exc, link, chain, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          _close_and_raise_sync(resolved, '_Break')
        yield resolved
      return
    except _Return as exc:
      resolved = _resolve_signal_value(exc, link, chain, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          _close_and_raise_sync(resolved, '_Return')
        yield resolved
      return

  finally:
    # CM exit + deferred finally — single cleanup path for all exit modes.
    # Structure: try { cm.__exit__ } finally { deferred_finally }
    # ensures deferred finally runs even if __exit__ raises.
    _cm_suppress = False
    try:
      if _with_cm is not None:
        _exc_info = sys.exc_info()
        if _exc_info[1] is not None and not isinstance(_exc_info[1], (GeneratorExit, _ControlFlowSignal)):
          _cm_suppress = _with_cm.__exit__(*_exc_info)
          if _isawaitable(_cm_suppress):
            if hasattr(_cm_suppress, 'close'):
              _cm_suppress.close()
            _cm_suppress = False
          else:
            _cm_suppress = bool(_cm_suppress)
        else:
          _sync_cm_exit_clean(_with_cm)
    finally:
      if _deferred is not None:
        assert chain is not None  # guaranteed by _has_deferred guard
        _run_deferred_finally_sync(chain, _deferred)
    if _cm_suppress:
      return  # noqa: B012 — intentional: CM suppression semantics require return in finally


async def _aiter_wrap(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
  """Wrap a synchronous iterator as an async iterator."""
  __tracebackhide__ = True
  for item in sync_iter:
    yield item


async def _async_generator(
  chain_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  chain: Chain[Any] | None = None,
  link: Link | None = None,
  deferred_with: tuple[Link | None, bool] | None = None,
  flat: bool = False,
  on_exhaust: Callable[[], Any] | None = None,
) -> AsyncIterator[Any]:
  # ASYNC MIRROR of _sync_generator — keep both in sync when modifying.
  # Intentional divergences: async awaits awaitables; sync closes them + raises TypeError.
  """Asynchronous generator that yields each element of the chain's output."""
  __tracebackhide__ = True
  _has_deferred = chain is not None and chain.on_finally_link is not None
  _deferred: list[Any] | None = [Null, None, 0] if _has_deferred else None
  _with_cm: Any = None
  _use_async_cm = False

  try:
    # Run the chain pipeline
    _has_kw = _deferred is not None or deferred_with is not None
    if _has_kw:
      _kw: dict[str, Any] = {}
      if _deferred is not None:
        _kw['deferred_finally'] = _deferred
      if deferred_with is not None:
        _kw['deferred_with'] = True
      iterator = chain_run(*run_args, **_kw)
    else:
      iterator = chain_run(*run_args)
    if _isawaitable(iterator):
      iterator = await iterator

    # Enter deferred CM if active
    if deferred_with is not None:
      _dw_inner_link, _dw_ignore_result = deferred_with
      cm = iterator
      # In _async_generator we are always in an async context, so prefer
      # __aenter__ when available.  _should_use_async_protocol checks for a
      # running asyncio loop, which returns False under trio/other runtimes.
      _has_aenter = hasattr(cm, '__aenter__')
      _use_async_cm_result = True if _has_aenter else _should_use_async_protocol(cm, '__enter__', '__aenter__')
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
      if _dw_inner_link is not None:
        inner_result = _evaluate_value(_dw_inner_link, ctx)
        if _isawaitable(inner_result):
          inner_result = await inner_result
        iterator = cm if _dw_ignore_result else inner_result
      else:
        # Bare with_(): context value IS the iterable
        iterator = ctx

    if not hasattr(iterator, '__aiter__'):
      iterator = _aiter_wrap(iterator)

    # Iteration loop
    idx = 0
    try:
      async for item in iterator:
        if fn is None:
          if flat:
            for sub in item:
              yield sub
          else:
            yield item
        else:
          try:
            result = fn(item)
            if _isawaitable(result):
              result = await result
          except _ControlFlowSignal:  # Must propagate — not a regular exception.
            raise
          except BaseException as exc:
            _handle_iterate_exc(exc, link, chain, ignore_result, item, idx)
            raise exc  # Use `raise exc` (not bare `raise`) so the modified __traceback__ is respected on Python <3.11.
          if flat:
            if ignore_result:
              for _unused in result:
                pass
              yield item
            else:
              for sub in result:
                yield sub
          else:
            yield item if ignore_result else result
        idx += 1

      # on_exhaust after source exhaustion
      if on_exhaust is not None:
        exhaust_result = on_exhaust()
        if _isawaitable(exhaust_result):
          exhaust_result = await exhaust_result
        for sub in exhaust_result:
          yield sub

    except _Break as exc:
      resolved = _resolve_signal_value(exc, link, chain, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          try:
            resolved = await resolved
          except BaseException as await_exc:
            _handle_iterate_exc(await_exc, link, chain, ignore_result, exc.value, idx)
            raise await_exc  # explicit raise for modified __traceback__ on Python <3.11
        yield resolved
      return
    except _Return as exc:
      resolved = _resolve_signal_value(exc, link, chain, ignore_result, idx)
      if resolved is not _NO_VALUE:
        if _isawaitable(resolved):
          try:
            resolved = await resolved
          except BaseException as await_exc:
            _handle_iterate_exc(await_exc, link, chain, ignore_result, exc.value, idx)
            raise await_exc  # explicit raise for modified __traceback__ on Python <3.11
        yield resolved
      return

  finally:
    # CM exit + deferred finally — single cleanup path for all exit modes.
    _cm_suppress = False
    try:
      if _with_cm is not None:
        _exc_info = sys.exc_info()
        if _exc_info[1] is not None and not isinstance(_exc_info[1], (GeneratorExit, _ControlFlowSignal)):
          if _use_async_cm:
            _cm_suppress = await _with_cm.__aexit__(*_exc_info)
          else:
            _cm_suppress = _with_cm.__exit__(*_exc_info)
            if _isawaitable(_cm_suppress):
              _cm_suppress = await _cm_suppress
          _cm_suppress = bool(_cm_suppress)
        else:
          await _async_cm_exit_clean(_with_cm, _use_async_cm)
    finally:
      if _deferred is not None:
        assert chain is not None  # guaranteed by _has_deferred guard
        await _run_deferred_finally_async(chain, _deferred)
    if _cm_suppress:
      return  # noqa: B012 — intentional: CM suppression semantics require return in finally


class ChainIterator(_UncopyableMixin, Generic[_T]):
  """Wraps chain output as a dual sync/async iterable.

  Created by ``Chain.iterate()``. Supports both ``__iter__`` and
  ``__aiter__``, choosing the appropriate generator at iteration time.
  Calling the instance returns a new ``ChainIterator`` with updated run args,
  making generators reusable with different inputs.

  Generic over ``_T``, the element type yielded during iteration.
  Currently ``_T`` is ``Any`` (the element type cannot be statically
  inferred from ``Chain[T]``), but the parameter is in place for future
  refinement.
  """

  __slots__ = (
    '_chain',
    '_chain_run',
    '_deferred_with',
    '_flat',
    '_fn',
    '_ignore_result',
    '_link',
    '_on_exhaust',
    '_run_args',
  )

  _chain: Chain[Any] | None
  _chain_run: Callable[..., Any]
  _deferred_with: tuple[Link | None, bool] | None
  _flat: bool
  _fn: Callable[[Any], Any] | None
  _ignore_result: bool
  _link: Link | None
  _on_exhaust: Callable[[], Any] | None
  _run_args: tuple[Any, tuple[Any, ...], dict[str, Any]]

  def __init__(
    self,
    chain_run: Callable[..., Any],
    fn: Callable[[Any], Any] | None,
    ignore_result: bool,
    chain: Chain[Any] | None = None,
    link: Link | None = None,
    deferred_with: tuple[Link | None, bool] | None = None,
    flat: bool = False,
    on_exhaust: Callable[[], Any] | None = None,
  ) -> None:
    self._chain_run = chain_run
    self._fn = fn
    self._ignore_result = ignore_result
    self._chain = chain
    self._link = link
    self._deferred_with = deferred_with
    self._flat = flat
    self._on_exhaust = on_exhaust
    self._run_args: tuple[Any, tuple[Any, ...], dict[str, Any]] = (Null, (), {})

  def __call__(self, v: Any = Null, *args: Any, **kwargs: Any) -> ChainIterator[_T]:
    """Return a new ``ChainIterator`` with updated ``_run_args``, enabling reuse with different inputs."""
    g: ChainIterator[_T] = ChainIterator(
      self._chain_run,
      self._fn,
      self._ignore_result,
      self._chain,
      self._link,
      deferred_with=self._deferred_with,
      flat=self._flat,
      on_exhaust=self._on_exhaust,
    )
    g._run_args = (v, args, kwargs)
    return g

  def __iter__(self) -> Iterator[_T]:
    """Delegate to the module-level ``_sync_generator`` function."""
    return _sync_generator(
      self._chain_run,
      self._run_args,
      self._fn,
      self._ignore_result,
      self._chain,
      self._link,
      deferred_with=self._deferred_with,
      flat=self._flat,
      on_exhaust=self._on_exhaust,
    )

  def __aiter__(self) -> AsyncIterator[_T]:
    """Delegate to the module-level ``_async_generator`` function."""
    return _async_generator(
      self._chain_run,
      self._run_args,
      self._fn,
      self._ignore_result,
      self._chain,
      self._link,
      deferred_with=self._deferred_with,
      flat=self._flat,
      on_exhaust=self._on_exhaust,
    )

  def __repr__(self) -> str:
    return '<quent.ChainIterator>'
