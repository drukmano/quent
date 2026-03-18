# SPDX-License-Identifier: MIT
"""Generator driving operation (drive_gen)."""

from __future__ import annotations

from types import AsyncGeneratorType, GeneratorType
from typing import Any

from ._eval import _isawaitable
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import _ControlFlowSignal


class _DriveGenOp:
  """Drive a sync or async generator bidirectionally with a step function.

  Abstracts over the protocol split between sync generators
  (``next``/``.send``/``StopIteration``/``.close``) and async generators
  (``__anext__``/``.asend``/``StopAsyncIteration``/``.aclose``).

  Three-tier execution (matching quent's standard pattern)::

    Scenario                          Method pipeline
    --------------------------------  -------------------------------------------
    Sync gen + sync step_fn           __call__ -> _sync_drive (returns value)
    Sync gen + async step_fn          __call__ -> _sync_drive -> _mid_transition
                                        (returns coroutine on first awaitable)
    Async gen (any step_fn)           __call__ -> _full_async (returns coroutine)
  """

  __slots__ = ('_fn', '_link', '_link_name')

  _fn: Any
  _link: Link
  _link_name: str

  def __init__(self, link: Link) -> None:
    self._link = link
    self._fn = link.v
    self._link_name = 'drive_gen'

  def __call__(self, current_value: Any) -> Any:
    """Resolve the current value as a generator and dispatch to the appropriate tier."""
    __tracebackhide__ = True
    gen = current_value

    # If callable (but not already a generator), invoke to get the generator.
    if not isinstance(gen, (GeneratorType, AsyncGeneratorType)) and callable(gen):
      gen = gen()

    if isinstance(gen, AsyncGeneratorType):
      return self._full_async(gen)
    if isinstance(gen, GeneratorType):
      return self._sync_drive(gen)

    msg = (
      f'{type(gen).__name__} object is not a generator. '
      f'drive_gen requires a sync generator, async generator, or a callable that produces one.'
    )
    raise TypeError(msg)

  def _sync_drive(self, gen: Any) -> Any:
    """Tier 1: sync fast path -- sync generator + sync step_fn.

    Does NOT use a finally block because of the sync-to-async transition:
    when ``_mid_transition`` is returned as a coroutine, the generator must
    remain open for the continuation.  Cleanup is explicit in each exit path.
    """
    __tracebackhide__ = True
    try:
      yielded = next(gen)
    except StopIteration:
      gen.close()
      return None

    while True:
      try:
        last_result = self._fn(yielded)
      except _ControlFlowSignal:
        gen.close()
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._link, current_value=yielded)
        gen.close()
        raise

      if _isawaitable(last_result):
        # Transfer generator ownership to _mid_transition -- it handles cleanup.
        return self._mid_transition(gen, last_result)

      try:
        yielded = gen.send(last_result)
      except StopIteration:
        gen.close()
        return last_result
      except BaseException:
        gen.close()
        raise

  async def _mid_transition(self, gen: Any, first_awaitable: Any) -> Any:
    """Tier 2: sync gen + async step_fn -- await results, use sync gen.send()."""
    __tracebackhide__ = True
    try:
      last_result = await first_awaitable

      while True:
        try:
          yielded = gen.send(last_result)
        except StopIteration:
          return last_result

        try:
          last_result = self._fn(yielded)
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _set_link_temp_args(exc, self._link, current_value=yielded)
          raise

        if _isawaitable(last_result):
          last_result = await last_result
    finally:
      gen.close()

  async def _full_async(self, gen: Any) -> Any:
    """Tier 3: async generator -- all generator operations awaited."""
    __tracebackhide__ = True
    try:
      try:
        yielded = await gen.__anext__()
      except StopAsyncIteration:
        return None

      while True:
        try:
          last_result = self._fn(yielded)
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _set_link_temp_args(exc, self._link, current_value=yielded)
          raise

        if _isawaitable(last_result):
          last_result = await last_result

        try:
          yielded = await gen.asend(last_result)
        except StopAsyncIteration:
          return last_result
    finally:
      await gen.aclose()
