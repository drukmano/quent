# SPDX-License-Identifier: MIT
"""Bounded backpressure buffer for iteration terminals.

Sync: queue.Queue + background thread. Async: asyncio.Queue + background task.
Producer blocks on full, consumer blocks on empty.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ._concurrency import _create_task_fn

# Producer-finished marker (see _types.py for the sentinel landscape).
_END = object()


class _ProducerError:
  """Wraps an exception raised by the producer."""

  __slots__ = ('exc',)

  def __init__(self, exc: BaseException) -> None:
    self.exc = exc


def _sync_buffer_iter(iterable: Any, maxsize: int) -> Iterator[Any]:
  """Yield items from *iterable* via a bounded queue.Queue.

  A daemon thread feeds the queue. On consumer early exit, _stop is set so
  the producer notices on its next put() and exits.
  """
  buf: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
  stop_event = threading.Event()

  def _producer() -> None:
    try:
      for item in iterable:
        if stop_event.is_set():
          return
        # Polling put so we can periodically check stop_event.
        while True:
          if stop_event.is_set():
            return
          try:
            buf.put(item, timeout=0.05)
            break
          except queue.Full:
            continue
      buf.put(_END)
    except BaseException as exc:
      try:
        buf.put(_ProducerError(exc), timeout=1.0)
      except queue.Full:
        # Consumer gone — re-raise system exceptions so they're not silently dropped.
        if not isinstance(exc, Exception):
          raise

  t = threading.Thread(target=_producer, daemon=True)
  t.start()
  try:
    while True:
      item = buf.get()
      if item is _END:
        return
      if isinstance(item, _ProducerError):
        raise item.exc
      yield item
  finally:
    stop_event.set()
    # Drain so producer can unblock if stuck on put().
    try:
      while not buf.empty():
        buf.get_nowait()
    except queue.Empty:
      pass
    t.join(timeout=5.0)
    if t.is_alive():
      warnings.warn(
        'quent: buffer producer thread did not terminate within 5s; it will continue as a daemon thread.',
        RuntimeWarning,
        stacklevel=2,
      )


async def _async_buffer_iter(iterable: Any, maxsize: int) -> AsyncIterator[Any]:
  """Yield items from *iterable* via a bounded asyncio.Queue.

  Background task feeds the queue; on consumer early exit, the task is cancelled.
  """
  buf: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
  stop = False

  async def _producer() -> None:
    nonlocal stop
    try:
      if hasattr(iterable, '__aiter__'):
        async for item in iterable:
          if stop:
            return
          await buf.put(item)
      else:
        for item in iterable:
          if stop:
            return
          await buf.put(item)
      await buf.put(_END)
    except asyncio.CancelledError:
      return
    except BaseException as exc:
      await buf.put(_ProducerError(exc))

  task = _create_task_fn(_producer())
  try:
    while True:
      item = await buf.get()
      if item is _END:
        return
      if isinstance(item, _ProducerError):
        raise item.exc
      yield item
  finally:
    stop = True
    if not task.done():
      task.cancel()
      try:
        await task
      except (asyncio.CancelledError, Exception):
        pass
