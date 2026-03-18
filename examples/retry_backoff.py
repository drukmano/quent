"""
Retry with exponential backoff and jitter -- quent recipe.

Demonstrates how to build a reusable retry-with-backoff wrapper and
compose it into quent pipelines via .then(), .do(), and .except_().

Patterns shown:
  - retry(fn) -- wraps a sync callable with exponential-backoff retry logic
  - async_retry(fn) -- async variant using asyncio.sleep
  - Inline use as a .then() step
  - Final safety net via .except_()
  - Both sync and async demonstrations
  - Practical: simulated flaky network call
  - Nested pipeline as a reusable retry building block

Run with:
    python examples/retry_backoff.py
"""

from __future__ import annotations

import asyncio
import random
import time

from quent import Q

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def retry(
  fn,
  max_attempts: int = 3,
  base_delay: float = 0.1,
  max_delay: float = 10.0,
  jitter: bool = True,
  exceptions: tuple = (Exception,),
):
  """Wrap *fn* with synchronous exponential-backoff retry logic.

  The returned callable accepts the same arguments as *fn*. On failure it
  sleeps and retries up to *max_attempts* times before re-raising.
  """

  def _wrapped(*args, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
      try:
        return fn(*args, **kwargs)
      except exceptions as exc:
        last_exc = exc
        if attempt < max_attempts - 1:
          delay = min(base_delay * (2 ** attempt), max_delay)
          if jitter:
            delay *= 0.5 + random.random()
          print(f'  [retry] attempt {attempt + 1}/{max_attempts} failed: {exc!r} -- sleeping {delay:.3f}s')
          time.sleep(delay)
        else:
          print(f'  [retry] attempt {attempt + 1}/{max_attempts} failed: {exc!r} -- giving up')
    raise last_exc  # type: ignore[misc]

  return _wrapped


def async_retry(
  fn,
  max_attempts: int = 3,
  base_delay: float = 0.1,
  max_delay: float = 10.0,
  jitter: bool = True,
  exceptions: tuple = (Exception,),
):
  """Wrap *fn* with asynchronous exponential-backoff retry logic.

  The returned coroutine function accepts the same arguments as *fn*.
  On failure it awaits asyncio.sleep and retries up to *max_attempts* times.
  """

  async def _wrapped(*args, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
      try:
        result = fn(*args, **kwargs)
        if asyncio.isfuture(result) or asyncio.iscoroutine(result):
          return await result
        return result
      except exceptions as exc:
        last_exc = exc
        if attempt < max_attempts - 1:
          delay = min(base_delay * (2 ** attempt), max_delay)
          if jitter:
            delay *= 0.5 + random.random()
          print(f'  [async_retry] attempt {attempt + 1}/{max_attempts} failed: {exc!r} -- sleeping {delay:.3f}s')
          await asyncio.sleep(delay)
        else:
          print(f'  [async_retry] attempt {attempt + 1}/{max_attempts} failed: {exc!r} -- giving up')
    raise last_exc  # type: ignore[misc]

  return _wrapped


# ---------------------------------------------------------------------------
# Simulated flaky services
# ---------------------------------------------------------------------------

class FlakyService:
  """Fails the first *fail_count* calls with ConnectionError, then succeeds."""

  def __init__(self, name: str, fail_count: int = 2) -> None:
    self.name = name
    self.fail_count = fail_count
    self._calls = 0

  def reset(self) -> None:
    self._calls = 0

  def fetch(self, url: str) -> dict:
    self._calls += 1
    if self._calls <= self.fail_count:
      raise ConnectionError(f'{self.name}: transient failure on call {self._calls}')
    return {'url': url, 'status': 200, 'body': f'response from {self.name}'}

  async def async_fetch(self, url: str) -> dict:
    self._calls += 1
    if self._calls <= self.fail_count:
      raise ConnectionError(f'{self.name}: async transient failure on call {self._calls}')
    await asyncio.sleep(0)  # yield to event loop
    return {'url': url, 'status': 200, 'body': f'async response from {self.name}'}


# ---------------------------------------------------------------------------
# Nested pipeline for retry with per-attempt error handling
# ---------------------------------------------------------------------------

def make_retry_pipeline(service: FlakyService, max_attempts: int = 3) -> Q:
  """Build a pipeline that retries a service call with backoff.

  This demonstrates using a nested pipeline as a reusable retry building block.
  The nested pipeline has its own except_() for per-attempt logging; the outer
  pipeline can add its own except_() for final failure handling.
  """
  wrapped = retry(service.fetch, max_attempts=max_attempts, base_delay=0.05)

  return (
    Q()
    .then(wrapped)
    .do(lambda r: print(f'  [nested] success: {r["body"]}'))
  )


# ---------------------------------------------------------------------------
# Main demos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  random.seed(42)  # reproducible jitter for demo output

  # -- Sync demo ----------------------------------------------------------
  print('=' * 60)
  print('SYNC DEMO: retry() as a .then() step')
  print('=' * 60)

  svc = FlakyService('sync-svc', fail_count=2)
  result = (
    Q('https://api.example.com/data')
    .then(retry(svc.fetch, max_attempts=4, base_delay=0.05))
    .do(lambda r: print(f'  [do] status={r["status"]}'))
    # except_ with Ellipsis: handler receives only the exception.
    .except_(lambda ei: print(f'  [except] gave up: {ei.exc}') or {})
    .run()
  )
  print(f'  Result: {result}\n')

  # -- Async demo ----------------------------------------------------------
  print('=' * 60)
  print('ASYNC DEMO: async_retry() with asyncio.run()')
  print('=' * 60)

  async_svc = FlakyService('async-svc', fail_count=2)
  result = asyncio.run(
    Q('https://api.example.com/async-data')
    .then(async_retry(async_svc.async_fetch, max_attempts=4, base_delay=0.05))
    .do(lambda r: print(f'  [do] body={r["body"]!r}'))
    .except_(lambda ei: print(f'  [except] gave up: {ei.exc}') or {})
    .run()
  )
  print(f'  Result: {result}\n')

  # -- Nested pipeline demo ---------------------------------------------------
  print('=' * 60)
  print('NESTED PIPELINE DEMO: reusable retry pipeline as a .then() step')
  print('=' * 60)

  nested_svc = FlakyService('nested-svc', fail_count=1)
  retry_q = make_retry_pipeline(nested_svc, max_attempts=3)

  result = (
    Q('https://api.example.com/nested')
    # The nested retry pipeline runs with the current value (the URL) as input.
    .then(retry_q)
    .then(lambda r: f'Final answer: {r["body"]}')
    .run()
  )
  print(f'  Result: {result}\n')

  # -- All retries exhausted demo ------------------------------------------
  print('=' * 60)
  print('FAILURE DEMO: all retries exhausted, except_ catches final error')
  print('=' * 60)

  hopeless_svc = FlakyService('hopeless-svc', fail_count=100)  # always fails
  result = (
    Q('https://api.example.com/hopeless')
    .then(retry(hopeless_svc.fetch, max_attempts=3, base_delay=0.02))
    # except_ default calling convention: handler(exc, root_value).
    # root_value is the URL string passed to Q().
    .except_(lambda ei: {'error': str(ei.exc), 'url': ei.root_value})
    .run()
  )
  print(f'  Result: {result}')

  print()
  print('All demos complete.')
