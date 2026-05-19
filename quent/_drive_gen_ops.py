# SPDX-License-Identifier: MIT
"""Generator driving operation (drive_gen)."""

from __future__ import annotations

from types import AsyncGeneratorType, GeneratorType
from typing import Any

from ._eval import _handle_return_exc, _isawaitable
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import _ControlFlowSignal, _Return


class _DriveGenOp:
  """Drive a sync or async generator bidirectionally with a step function.

  Abstracts over the protocol split between sync (next/send/StopIteration/close)
  and async (__anext__/asend/StopAsyncIteration/aclose) generators.

  Three tiers:
    Sync gen + sync fn   → __call__ → _sync_drive
    Sync gen + async fn  → __call__ → _sync_drive → _mid_transition (returns coro)
    Async gen (any fn)   → __call__ → _full_async (returns coro)
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
    __tracebackhide__ = True
    gen = current_value

    # Callable but not yet a generator? Invoke to obtain one.
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
    """Tier 1: sync gen + sync fn.

    No outer finally — generator must remain open across the sync→async
    transition. Cleanup is explicit on each exit path.
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
      except _Return as ret_exc:
        # Q.return_() inside fn returns from fn — value becomes pipeline CV.
        last_result = _handle_return_exc(ret_exc)
        if _isawaitable(last_result):
          # Lazy fn returned an awaitable — transition to async to await it.
          return self._mid_transition(gen, last_result)
        gen.close()
        return last_result
      except _ControlFlowSignal:
        gen.close()
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._link, current_value=yielded)
        gen.close()
        raise

      if _isawaitable(last_result):
        # Transfer ownership to _mid_transition for cleanup.
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
    """Tier 2: sync gen + async fn — await results, use sync gen.send()."""
    __tracebackhide__ = True
    try:
      # First-awaitable wait — may raise _Return inside the coroutine.
      try:
        last_result = await first_awaitable
      except _Return as ret_exc:
        last_result = _handle_return_exc(ret_exc)
        if _isawaitable(last_result):
          last_result = await last_result
        return last_result

      while True:
        try:
          yielded = gen.send(last_result)
        except StopIteration:
          return last_result

        # Call fn AND await its result inside the same try/except _Return —
        # for async fns, _Return is raised on `await`, not on call.
        try:
          last_result = self._fn(yielded)
          if _isawaitable(last_result):
            last_result = await last_result
        except _Return as ret_exc:
          last_result = _handle_return_exc(ret_exc)
          if _isawaitable(last_result):
            last_result = await last_result
          return last_result
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _set_link_temp_args(exc, self._link, current_value=yielded)
          raise
    finally:
      gen.close()

  async def _full_async(self, gen: Any) -> Any:
    """Tier 3: async generator — all generator ops awaited."""
    __tracebackhide__ = True
    try:
      try:
        yielded = await gen.__anext__()
      except StopAsyncIteration:
        return None

      while True:
        # Call fn AND await its result inside the same try/except _Return —
        # for async fns, _Return is raised on `await`, not on call.
        try:
          last_result = self._fn(yielded)
          if _isawaitable(last_result):
            last_result = await last_result
        except _Return as ret_exc:
          last_result = _handle_return_exc(ret_exc)
          if _isawaitable(last_result):
            last_result = await last_result
          return last_result
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _set_link_temp_args(exc, self._link, current_value=yielded)
          raise

        try:
          yielded = await gen.asend(last_result)
        except StopAsyncIteration:
          return last_result
    finally:
      await gen.aclose()
