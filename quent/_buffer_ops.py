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
from collections.abc import AsyncGenerator, Generator
from contextvars import copy_context
from typing import Any

from ._concurrency import _create_task_fn

# Producer-finished marker (see _types.py for the sentinel landscape).
_END = object()


class _ProducerError:
  """Wraps an exception raised by the producer."""

  __slots__ = ('exc',)

  def __init__(self, exc: BaseException) -> None:
    self.exc = exc


def _sync_buffer_iter(iterable: Any, maxsize: int) -> Generator[Any, None, None]:
  """Yield items from *iterable* via a bounded queue.Queue.

  A daemon thread feeds the queue. On consumer early exit, _stop is set so
  the producer notices on its next put() and exits.
  """
  buf: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
  stop_event = threading.Event()
  cleanup_error: BaseException | None = None
  delivered_error: BaseException | None = None

  def _put(item: Any) -> bool:
    # Terminal markers obey the same backpressure and shutdown rules as data.
    while not stop_event.is_set():
      try:
        buf.put(item, timeout=0.05)
        return True
      except queue.Full:
        continue
    return False

  def _producer() -> None:
    nonlocal cleanup_error
    try:
      source = iter(iterable)
      try:
        for item in source:
          if not _put(item):
            return
      finally:
        close = getattr(source, 'close', None)
        if close is not None:
          try:
            close()
          except BaseException as exc:
            cleanup_error = exc
            raise
      _put(_END)
    except BaseException as exc:
      if not _put(_ProducerError(exc)) and exc is not cleanup_error and not isinstance(exc, Exception):
        raise

  context = copy_context()
  t = threading.Thread(target=context.run, args=(_producer,), daemon=True)
  t.start()
  try:
    while True:
      item = buf.get()
      if item is _END:
        return
      if isinstance(item, _ProducerError):
        delivered_error = item.exc
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
    elif cleanup_error is not None and cleanup_error is not delivered_error:
      raise cleanup_error


async def _async_buffer_iter(iterable: Any, maxsize: int) -> AsyncGenerator[Any, None]:
  """Yield items from *iterable* via a bounded asyncio.Queue.

  Background task feeds the queue; on consumer early exit, the task is cancelled.
  """
  buf: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
  stop = False
  cleanup_error: BaseException | None = None
  delivered_error: BaseException | None = None

  async def _producer() -> None:
    nonlocal cleanup_error
    try:
      if hasattr(iterable, '__aiter__'):
        source = aiter(iterable)
        try:
          async for item in source:
            if stop:
              return
            await buf.put(item)
        finally:
          aclose = getattr(source, 'aclose', None)
          if aclose is not None:
            try:
              await aclose()
            except BaseException as exc:
              cleanup_error = exc
              raise
      else:
        sync_source = iter(iterable)
        try:
          for item in sync_source:
            if stop:
              return
            await buf.put(item)
        finally:
          close = getattr(sync_source, 'close', None)
          if close is not None:
            try:
              close()
            except BaseException as exc:
              cleanup_error = exc
              raise
      await buf.put(_END)
    except BaseException as exc:
      if stop:
        if exc is not cleanup_error and not isinstance(exc, (Exception, asyncio.CancelledError)):
          raise
        return
      # Source cancellation is a producer failure, not a consumer shutdown.
      await buf.put(_ProducerError(exc))

  producer = _producer()
  try:
    task = _create_task_fn(producer)
  except BaseException:
    producer.close()
    raise
  try:
    while True:
      item = await buf.get()
      if item is _END:
        return
      if isinstance(item, _ProducerError):
        delivered_error = item.exc
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
    if cleanup_error is not None and cleanup_error is not delivered_error:
      raise cleanup_error
