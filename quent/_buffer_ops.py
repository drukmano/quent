# SPDX-License-Identifier: MIT
"""Backpressure-aware buffer operation for iteration pipelines.

Interposes a bounded queue between a producer (the pipeline's iterable output)
and a consumer (the iteration loop).  When the buffer is full the producer
blocks (backpressure); when the buffer is empty the consumer blocks.

Sync path: ``queue.Queue`` + a background ``threading.Thread``.
Async path: ``asyncio.Queue`` + a background ``asyncio.Task``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ._concurrency import _create_task_fn

# Sentinel signalling that the producer has finished (normally or with error).
# See _types.py for the full sentinel landscape.
_END = object()


class _ProducerError:
  """Wrapper carrying an exception raised by the producer."""

  __slots__ = ('exc',)

  def __init__(self, exc: BaseException) -> None:
    self.exc = exc


# ---------------------------------------------------------------------------
# Sync buffered iteration
# ---------------------------------------------------------------------------


def _sync_buffer_iter(iterable: Any, maxsize: int) -> Iterator[Any]:
  """Yield items from *iterable* via a bounded ``queue.Queue``.

  A background daemon thread feeds the queue.  When the queue is full the
  producer thread blocks (backpressure).  The consumer (this generator)
  blocks on ``queue.get()`` when the buffer is empty.

  Cleanup: if the consumer exits early (``break``, ``GeneratorExit``), the
  ``_stop`` event is set so the producer thread notices on its next
  ``put()`` attempt and exits.
  """
  buf: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
  stop_event = threading.Event()

  def _producer() -> None:
    try:
      for item in iterable:
        if stop_event.is_set():
          return
        # Use a polling put so we can check the stop event periodically.
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
        # Queue is full and consumer is gone -- re-raise system exceptions
        # (KeyboardInterrupt, SystemExit) so they are never silently dropped.
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
    # Drain the queue so the producer can unblock if it is stuck on put().
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


# ---------------------------------------------------------------------------
# Async buffered iteration
# ---------------------------------------------------------------------------


async def _async_buffer_iter(iterable: Any, maxsize: int) -> AsyncIterator[Any]:
  """Yield items from *iterable* via a bounded ``asyncio.Queue``.

  A background ``asyncio.Task`` feeds the queue.  When the queue is full
  the producer task awaits (backpressure).  The consumer (this async
  generator) awaits ``queue.get()`` when the buffer is empty.

  Cleanup: if the consumer exits early, the producer task is cancelled.
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
