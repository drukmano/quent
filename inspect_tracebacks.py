"""Visual inspection of quent exception traceback formatting.

Run with:  python3 inspect_tracebacks.py
"""
from __future__ import annotations

import asyncio
import traceback as tb_mod
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from quent import Chain, QuentException


# ---------------------------------------------------------------------------
# Case counters
# ---------------------------------------------------------------------------
_total_cases = 0
_traceback_cases = 0
_caught_by_except_cases = 0


def _header(label: str) -> None:
  width = max(60, len(label) + 6)
  bar = '═' * width
  print(f'\n{bar}')
  print(f'  {label}')
  print(f'{bar}')


def run_case(label: str, fn) -> None:
  """Run a sync case, printing header + traceback (or result) + separator."""
  global _total_cases, _traceback_cases
  _total_cases += 1
  _header(label)
  try:
    result = fn()
    print(f'[no exception]  result = {result!r}')
  except Exception:
    _traceback_cases += 1
    tb_mod.print_exc()
  print()


def run_case_caught(label: str, fn) -> None:
  """Run a sync case expected to be handled by except_, printing the result."""
  global _total_cases, _caught_by_except_cases
  _total_cases += 1
  _caught_by_except_cases += 1
  _header(label)
  try:
    result = fn()
    print(f'[caught by except_]  result = {result!r}')
  except Exception:
    tb_mod.print_exc()
  print()


def run_async_case(label: str, coro_fn) -> None:
  """Run an async case, printing header + traceback (or result) + separator."""
  global _total_cases, _traceback_cases
  _total_cases += 1
  _header(label)
  async def _runner():
    try:
      result = await coro_fn()
      print(f'[no exception]  result = {result!r}')
    except Exception:
      tb_mod.print_exc()
  asyncio.run(_runner())
  print()


# ===========================================================================
# SECTION 1 – Basic sync
# ===========================================================================

def section_basic_sync():
  _header('SECTION 1 — Basic sync')

  run_case(
    'CASE 1: Simple chain — Chain(1).then(lambda x: 1/0).run()',
    lambda: Chain(1).then(lambda x: 1 / 0).run(),
  )

  run_case(
    'CASE 2: Callable root that raises — Chain(lambda: 1/0).run()',
    lambda: Chain(lambda: 1 / 0).run(),
  )

  run_case(
    'CASE 3: Error in do() — Chain(1).do(lambda x: 1/0).run()',
    lambda: Chain(1).do(lambda x: 1 / 0).run(),
  )

  run_case(
    'CASE 4: Error deep in chain (4 steps, fails at step 3)',
    lambda: (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: x * 2)
      .then(lambda x: 1 / 0)
      .then(str)
      .run()
    ),
  )

  def step_a(x):
    return x + 1

  def step_b(x):
    raise ValueError('boom')

  def step_c(x):
    return x * 2

  run_case(
    'CASE 5: Named functions — step_a → step_b(raises) → step_c',
    lambda: Chain(1).then(step_a).then(step_b).then(step_c).run(),
  )


# ===========================================================================
# SECTION 2 – Basic async
# ===========================================================================

async def _bad_async(x):
  raise RuntimeError('async boom')

async def _async_identity(x):
  return x

async def _async_root_raises():
  raise TypeError('root fail')


def section_basic_async():
  _header('SECTION 2 — Basic async')

  run_async_case(
    'CASE 6: Async function that raises',
    lambda: Chain(1).then(_bad_async).run(),
  )

  run_async_case(
    'CASE 7: sync → async → error',
    lambda: Chain(1).then(lambda x: x + 1).then(_async_identity).then(_bad_async).run(),
  )

  run_async_case(
    'CASE 8: Async root that raises',
    lambda: Chain(_async_root_raises).run(),
  )


# ===========================================================================
# SECTION 3 – Nested chains
# ===========================================================================

def section_nested_chains():
  _header('SECTION 3 — Nested chains')

  run_case(
    'CASE 9: Inner chain raises — Chain(1).then(Chain().then(lambda: 1/0))',
    lambda: Chain(1).then(Chain().then(lambda x: 1 / 0)).run(),
  )

  run_case(
    'CASE 10: Doubly nested chain raises',
    lambda: Chain(1).then(
      Chain().then(
        Chain().then(lambda x: 1 / 0)
      )
    ).run(),
  )

  def inner_step(x):
    return x * 10

  def outer_step(x):
    raise OverflowError('outer overflow')

  def deep_fail(x):
    raise AttributeError('deep fail')

  run_case(
    'CASE 11: Nested chains with named functions in both levels',
    lambda: Chain(1).then(
      Chain().then(inner_step).then(deep_fail)
    ).then(outer_step).run(),
  )


# ===========================================================================
# SECTION 4 – except_ handler
# ===========================================================================

def section_except_handler():
  _header('SECTION 4 — except_ handler')

  run_case_caught(
    'CASE 12: except_ handles the exception (no traceback expected)',
    lambda: (
      Chain(1)
      .then(lambda x: 1 / 0)
      .except_(lambda e: f'caught: {e}')
      .run()
    ),
  )

  run_case(
    'CASE 13: except_ itself raises (chained exception expected)',
    lambda: (
      Chain(1)
      .then(lambda x: 1 / 0)
      .except_(lambda e: 1 / 0)
      .run()
    ),
  )

  run_case(
    'CASE 14: except_ with non-matching exception type (ZeroDivisionError vs TypeError)',
    lambda: (
      Chain(1)
      .then(lambda x: 1 / 0)
      .except_(lambda e: 'caught', exceptions=TypeError)
      .run()
    ),
  )


# ===========================================================================
# SECTION 5 – finally_ handler
# ===========================================================================

def section_finally_handler():
  _header('SECTION 5 — finally_ handler')

  run_case(
    'CASE 15: finally_ runs even when main chain raises (finally prints)',
    lambda: (
      Chain(1)
      .then(lambda x: 1 / 0)
      .finally_(lambda x: print(f'  [finally] root_value={x!r}'))
      .run()
    ),
  )

  run_case(
    'CASE 16: finally_ itself raises (nested exception)',
    lambda: (
      Chain(1)
      .then(lambda x: 1 / 0)
      .finally_(lambda x: 1 / 0)
      .run()
    ),
  )


# ===========================================================================
# SECTION 6 – foreach / foreach_do
# ===========================================================================

def section_foreach():
  _header('SECTION 6 — foreach / foreach_do')

  run_case(
    'CASE 17: foreach callback always raises — Chain([1,2,3]).foreach(lambda x: 1/0)',
    lambda: Chain([1, 2, 3]).foreach(lambda x: 1 / 0).run(),
  )

  run_case(
    'CASE 18: foreach fails on specific item — Chain([1,2,0]).foreach(lambda x: 10//x)',
    lambda: Chain([1, 2, 0]).foreach(lambda x: 10 // x).run(),
  )

  run_case(
    'CASE 19: foreach_do callback raises',
    lambda: Chain([1, 2, 3]).foreach_do(lambda x: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 7 – filter
# ===========================================================================

def section_filter():
  _header('SECTION 7 — filter')

  run_case(
    'CASE 20: Error in filter predicate',
    lambda: Chain([1, 2, 3]).filter(lambda x: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 8 – with_ / with_do
# ===========================================================================

class _SimpleCM:
  """A simple context manager that yields a string."""
  def __enter__(self):
    return 'ctx_val'

  def __exit__(self, *args):
    return False


class _BadEnterCM:
  """A context manager whose __enter__ raises."""
  def __enter__(self):
    raise RuntimeError('__enter__ failed')

  def __exit__(self, *args):
    return False


def section_with():
  _header('SECTION 8 — with_ / with_do')

  run_case(
    'CASE 21: Error inside with_ body',
    lambda: Chain(_SimpleCM()).with_(lambda ctx: 1 / 0).run(),
  )

  run_case(
    'CASE 22: Error in __enter__',
    lambda: Chain(_BadEnterCM()).with_(lambda ctx: ctx).run(),
  )

  run_case(
    'CASE 23: with_do error — body raises, original value not kept',
    lambda: Chain(_SimpleCM()).with_do(lambda ctx: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 9 – gather
# ===========================================================================

def section_gather():
  _header('SECTION 9 — gather')

  run_case(
    'CASE 24: One gathered function raises (sync)',
    lambda: Chain(1).gather(
      lambda x: x + 1,
      lambda x: 1 / 0,
      lambda x: x * 3,
    ).run(),
  )

  run_async_case(
    'CASE 25: One gathered function raises (async)',
    lambda: Chain(1).gather(
      lambda x: x + 1,
      lambda x: 1 / 0,
      lambda x: x * 3,
    ).run(),
  )


# ===========================================================================
# SECTION 10 – Frozen chains
# ===========================================================================

def section_frozen():
  _header('SECTION 10 — Frozen chains')

  def _run_frozen_sync():
    fc = Chain().then(lambda x: 1 / 0).freeze()
    return fc(1)

  run_case('CASE 26: Frozen chain that raises (sync)', _run_frozen_sync)

  def _run_frozen_deep():
    def step1(x):
      return x + 1
    def step2(x):
      return x * 2
    def step3(x):
      raise StopIteration('frozen stop')
    fc = Chain().then(step1).then(step2).then(step3).freeze()
    return fc(5)

  run_case('CASE 27: Frozen chain — named steps, raises at step 3', _run_frozen_deep)


# ===========================================================================
# SECTION 11 – Deep call stacks
# ===========================================================================

def section_deep():
  _header('SECTION 11 — Deep call stacks')

  def _build_deep_chain():
    c = Chain(1)
    for _ in range(9):
      c = c.then(lambda x: x + 1)
    c = c.then(lambda x: 1 / 0)
    return c.run()

  run_case('CASE 28: 10 steps, last one raises', _build_deep_chain)

  def _build_triple_nested():
    return Chain(1).then(
      Chain().then(
        Chain().then(
          Chain().then(lambda x: 1 / 0)
        )
      )
    ).run()

  run_case('CASE 29: 3-level nested chain, innermost raises', _build_triple_nested)


# ===========================================================================
# SECTION 12 – Sync/async combination matrix
# ===========================================================================

async def _fail_sync_wrapped(x):
  # wraps a sync raiser in an async fn for matrix testing
  raise ValueError('sync-flavour error in async step')


async def _passthrough_async(x):
  return x


def _fail_sync(x):
  raise ValueError('sync error in sync step')


def section_matrix():
  _header('SECTION 12 — Sync/async combination matrix')

  run_case(
    'CASE 30: sync → sync → error',
    lambda: Chain(1).then(lambda x: x + 1).then(_fail_sync).run(),
  )

  run_async_case(
    'CASE 31: sync → async → error',
    lambda: Chain(1).then(lambda x: x + 1).then(_fail_sync_wrapped).run(),
  )

  run_async_case(
    'CASE 32: async → sync → error',
    lambda: Chain(1).then(_passthrough_async).then(_fail_sync).run(),
  )

  run_async_case(
    'CASE 33: async → async → error',
    lambda: Chain(1).then(_passthrough_async).then(_bad_async).run(),
  )


# ===========================================================================
# SECTION 13 – with args / kwargs / ellipsis
# ===========================================================================

def _raiser_extra_arg(first_arg, extra):
  raise ValueError(f'first_arg={first_arg!r}, extra={extra!r}')


def _raiser_kwonly(**kw):
  raise ValueError(f'kwargs={kw!r}')


def _raiser_no_args():
  raise ValueError('called with no args (ellipsis)')


def section_args():
  _header('SECTION 13 — with args / kwargs / ellipsis')

  run_case(
    'CASE 34: Error with explicit positional extra arg',
    lambda: Chain(1).then(_raiser_extra_arg, 1, 'extra').run(),
  )

  run_case(
    'CASE 35: Error with kwargs only',
    lambda: Chain(1).then(_raiser_kwonly, key='val').run(),
  )

  run_case(
    'CASE 36: Error with ellipsis — callable invoked with no args',
    lambda: Chain(1).then(_raiser_no_args, ...).run(),
  )


# ===========================================================================
# SECTION 14 – return_ / break_
# ===========================================================================

def section_control_flow():
  _header('SECTION 14 — return_ / break_')

  # return_ is an expected early-exit mechanism; it should NOT raise.
  global _total_cases, _caught_by_except_cases
  _total_cases += 1
  _caught_by_except_cases += 1
  _header('CASE 37: Chain.return_() — early exit, result expected (no traceback)')
  try:
    result = Chain(1).then(lambda x: Chain.return_(x + 99)).run()
    print(f'[early return]  result = {result!r}')
  except Exception:
    tb_mod.print_exc()
  print()

  run_case(
    'CASE 38: Chain.break_() outside foreach — raises QuentException',
    lambda: Chain(1).then(lambda x: Chain.break_()).run(),
  )


# ===========================================================================
# SECTION 15 – Chained exceptions (raise ... from ...)
# ===========================================================================

def _raise_from(x):
  try:
    raise TypeError('original cause')
  except TypeError as e:
    raise ValueError('new top-level error') from e


def section_chained_exc():
  _header('SECTION 15 — Chained exceptions (raise from)')

  run_case(
    'CASE 39: raise ValueError("new") from TypeError("original") inside chain',
    lambda: Chain(1).then(_raise_from).run(),
  )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
  section_basic_sync()
  section_basic_async()
  section_nested_chains()
  section_except_handler()
  section_finally_handler()
  section_foreach()
  section_filter()
  section_with()
  section_gather()
  section_frozen()
  section_deep()
  section_matrix()
  section_args()
  section_control_flow()
  section_chained_exc()

  # Summary
  bar = '═' * 60
  print(f'\n{bar}')
  print('  SUMMARY')
  print(bar)
  print(f'  Ran           : {_total_cases} cases')
  print(f'  Had traceback : {_traceback_cases}')
  print(f'  Caught/silent : {_caught_by_except_cases}')
  print(bar)
  print()


if __name__ == '__main__':
  main()
