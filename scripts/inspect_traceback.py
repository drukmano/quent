"""Visual inspection of quent exception traceback formatting.

Run with:  python3 inspect_traceback.py
"""

from __future__ import annotations

import asyncio
import functools
import sys
import traceback as tb_mod

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from quent import Q

# ---------------------------------------------------------------------------
# Case counters
# ---------------------------------------------------------------------------
_total_cases = 0
_traceback_cases = 0
_caught_by_except_cases = 0


def _header(label: str) -> None:
  width = max(60, len(label) + 6)
  bar = '=' * width
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


def run_async_case_caught(label: str, coro_fn) -> None:
  """Run an async case expected to be handled by except_, printing the result."""
  global _total_cases, _caught_by_except_cases
  _total_cases += 1
  _caught_by_except_cases += 1
  _header(label)

  async def _runner():
    try:
      result = await coro_fn()
      print(f'[caught by except_]  result = {result!r}')
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
    'CASE 1: Simple pipeline — Q(1).then(lambda x: 1/0).run()',
    lambda: Q(1).then(lambda x: 1 / 0).run(),
  )

  run_case(
    'CASE 2: Callable root that raises — Q(lambda: 1/0).run()',
    lambda: Q(lambda: 1 / 0).run(),
  )

  run_case(
    'CASE 3: Error in do() — Q(1).do(lambda x: 1/0).run()',
    lambda: Q(1).do(lambda x: 1 / 0).run(),
  )

  run_case(
    'CASE 4: Error deep in pipeline (4 steps, fails at step 3)',
    lambda: Q(1).then(lambda x: x + 1).then(lambda x: x * 2).then(lambda x: 1 / 0).then(str).run(),
  )

  def step_a(x):
    return x + 1

  def step_b(x):
    raise ValueError('boom')

  def step_c(x):
    return x * 2

  run_case(
    'CASE 5: Named functions — step_a → step_b(raises) → step_c',
    lambda: Q(1).then(step_a).then(step_b).then(step_c).run(),
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
    lambda: Q(1).then(_bad_async).run(),
  )

  run_async_case(
    'CASE 7: sync → async → error',
    lambda: Q(1).then(lambda x: x + 1).then(_async_identity).then(_bad_async).run(),
  )

  run_async_case(
    'CASE 8: Async root that raises',
    lambda: Q(_async_root_raises).run(),
  )


# ===========================================================================
# SECTION 3 – Nested pipelines
# ===========================================================================


def section_nested_pipelines():
  _header('SECTION 3 — Nested pipelines')

  run_case(
    'CASE 9: Inner pipeline raises — Q(1).then(Q().then(lambda: 1/0))',
    lambda: Q(1).then(Q().then(lambda x: 1 / 0)).run(),
  )

  run_case(
    'CASE 10: Doubly nested pipeline raises',
    lambda: Q(1).then(Q().then(Q().then(lambda x: 1 / 0))).run(),
  )

  def inner_step(x):
    return x * 10

  def outer_step(x):
    raise OverflowError('outer overflow')

  def deep_fail(x):
    raise AttributeError('deep fail')

  run_case(
    'CASE 11: Nested pipelines with named functions in both levels',
    lambda: Q(1).then(Q().then(inner_step).then(deep_fail)).then(outer_step).run(),
  )


# ===========================================================================
# SECTION 4 – except_ handler
# ===========================================================================


def section_except_handler():
  _header('SECTION 4 — except_ handler')

  run_case_caught(
    'CASE 12: except_ handles the exception (no traceback expected)',
    lambda: Q(1).then(lambda x: 1 / 0).except_(lambda e: f'caught: {e}').run(),
  )

  run_case(
    'CASE 13: except_ itself raises (chained exception expected)',
    lambda: Q(1).then(lambda x: 1 / 0).except_(lambda e: 1 / 0).run(),
  )

  run_case(
    'CASE 14: except_ with non-matching exception type (ZeroDivisionError vs TypeError)',
    lambda: Q(1).then(lambda x: 1 / 0).except_(lambda e: 'caught', exceptions=TypeError).run(),
  )


# ===========================================================================
# SECTION 5 – finally_ handler
# ===========================================================================


def section_finally_handler():
  _header('SECTION 5 — finally_ handler')

  run_case(
    'CASE 15: finally_ runs even when main pipeline raises (finally prints)',
    lambda: Q(1).then(lambda x: 1 / 0).finally_(lambda x: print(f'  [finally] root_value={x!r}')).run(),
  )

  run_case(
    'CASE 16: finally_ itself raises (nested exception)',
    lambda: Q(1).then(lambda x: 1 / 0).finally_(lambda x: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 6 – map / foreach_do
# ===========================================================================


def section_map():
  _header('SECTION 6 — map / foreach_do')

  run_case(
    'CASE 17: map callback always raises — Q([1,2,3]).foreach(lambda x: 1/0)',
    lambda: Q([1, 2, 3]).foreach(lambda x: 1 / 0).run(),
  )

  run_case(
    'CASE 18: map fails on specific item — Q([1,2,0]).foreach(lambda x: 10//x)',
    lambda: Q([1, 2, 0]).foreach(lambda x: 10 // x).run(),
  )

  run_case(
    'CASE 19: foreach_do callback raises',
    lambda: Q([1, 2, 3]).foreach_do(lambda x: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 7 – foreach predicate errors (formerly filter, which was removed)
# ===========================================================================


def section_filter():
  _header('SECTION 7 — foreach predicate errors')

  run_case(
    'CASE 20: Error in foreach callback',
    lambda: Q([1, 2, 3]).foreach(lambda x: 1 / 0).run(),
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
    lambda: Q(_SimpleCM()).with_(lambda ctx: 1 / 0).run(),
  )

  run_case(
    'CASE 22: Error in __enter__',
    lambda: Q(_BadEnterCM()).with_(lambda ctx: ctx).run(),
  )

  run_case(
    'CASE 23: with_do error — body raises, original value not kept',
    lambda: Q(_SimpleCM()).with_do(lambda ctx: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 9 – gather
# ===========================================================================


def section_gather():
  _header('SECTION 9 — gather')

  run_case(
    'CASE 24: One gathered function raises (sync)',
    lambda: (
      Q(1)
      .gather(
        lambda x: x + 1,
        lambda x: 1 / 0,
        lambda x: x * 3,
      )
      .run()
    ),
  )

  run_async_case(
    'CASE 25: One gathered function raises (async)',
    lambda: (
      Q(1)
      .gather(
        lambda x: x + 1,
        lambda x: 1 / 0,
        lambda x: x * 3,
      )
      .run()
    ),
  )


# ===========================================================================
# SECTION 10 – Reusable pipelines
# ===========================================================================


def section_reusable():
  _header('SECTION 10 — Reusable pipelines')

  def _run_reusable_sync():
    q = Q().then(lambda x: 1 / 0)
    return q(1)

  run_case('CASE 26: Reusable pipeline that raises (sync)', _run_reusable_sync)

  def _run_reusable_deep():
    def step1(x):
      return x + 1

    def step2(x):
      return x * 2

    def step3(x):
      raise StopIteration('reusable stop')

    q = Q().then(step1).then(step2).then(step3)
    return q(5)

  run_case('CASE 27: Reusable pipeline — named steps, raises at step 3', _run_reusable_deep)


# ===========================================================================
# SECTION 11 – Deep call stacks
# ===========================================================================


def section_deep():
  _header('SECTION 11 — Deep call stacks')

  def _build_deep_pipeline():
    q = Q(1)
    for _ in range(9):
      q = q.then(lambda x: x + 1)
    q = q.then(lambda x: 1 / 0)
    return q.run()

  run_case('CASE 28: 10 steps, last one raises', _build_deep_pipeline)

  def _build_triple_nested():
    return Q(1).then(Q().then(Q().then(Q().then(lambda x: 1 / 0)))).run()

  run_case('CASE 29: 3-level nested pipeline, innermost raises', _build_triple_nested)


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
    lambda: Q(1).then(lambda x: x + 1).then(_fail_sync).run(),
  )

  run_async_case(
    'CASE 31: sync → async → error',
    lambda: Q(1).then(lambda x: x + 1).then(_fail_sync_wrapped).run(),
  )

  run_async_case(
    'CASE 32: async → sync → error',
    lambda: Q(1).then(_passthrough_async).then(_fail_sync).run(),
  )

  run_async_case(
    'CASE 33: async → async → error',
    lambda: Q(1).then(_passthrough_async).then(_bad_async).run(),
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
    lambda: Q(1).then(_raiser_extra_arg, 1, 'extra').run(),
  )

  run_case(
    'CASE 35: Error with kwargs only',
    lambda: Q(1).then(_raiser_kwonly, key='val').run(),
  )

  run_case(
    'CASE 36: Error with ellipsis — callable invoked with no args',
    lambda: Q(1).then(_raiser_no_args, ...).run(),
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
  _header('CASE 37: Q.return_() — early exit, result expected (no traceback)')
  try:
    result = Q(1).then(lambda x: Q.return_(x + 99)).run()
    print(f'[early return]  result = {result!r}')
  except Exception:
    tb_mod.print_exc()
  print()

  run_case(
    'CASE 38: Q.break_() outside map — raises QuentException',
    lambda: Q(1).then(lambda x: Q.break_()).run(),
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
    'CASE 39: raise ValueError("new") from TypeError("original") inside pipeline',
    lambda: Q(1).then(_raise_from).run(),
  )


# ===========================================================================
# SECTION 16 – Nested pipeline with except_ handler
# ===========================================================================


def _inner_raiser(x):
  raise ValueError('inner boom')


def section_nested_except():
  _header('SECTION 16 — Nested pipeline with except_ handler')

  # CASE 40: Inner pipeline raises, outer pipeline has except_ that catches
  run_case_caught(
    'CASE 40: Inner pipeline raises, outer except_ catches',
    lambda: Q(1).then(Q().then(_inner_raiser)).except_(lambda e: f'outer caught: {e}').run(),
  )

  # CASE 41: Inner pipeline raises, outer except_ with wrong exception type
  run_case(
    'CASE 41: Inner pipeline raises, outer except_ wrong type (catches TypeError, gets ValueError)',
    lambda: Q(1).then(Q().then(_inner_raiser)).except_(lambda e: f'caught: {e}', exceptions=TypeError).run(),
  )

  # CASE 42: Inner pipeline has its own except_ that catches, outer continues
  run_case_caught(
    'CASE 42: Inner except_ catches, outer pipeline continues (no traceback)',
    lambda: (
      Q(1)
      .then(Q().then(_inner_raiser).except_(lambda e: f'inner caught: {e}'))
      .then(lambda x: f'outer got: {x}')
      .except_(lambda e: f'outer caught: {e}')
      .run()
    ),
  )


# ===========================================================================
# SECTION 17 – Nested pipeline with finally_ handler
# ===========================================================================

_finally_tracker_17 = []


def section_nested_finally():
  _header('SECTION 17 — Nested pipeline with finally_ handler')

  _finally_tracker_17.clear()

  # CASE 43: Outer pipeline has finally_, inner pipeline raises
  run_case(
    'CASE 43: Outer finally_ runs when inner pipeline raises',
    lambda: (
      Q(1)
      .then(Q().then(lambda x: 1 / 0))
      .finally_(lambda x: _finally_tracker_17.append('outer_finally_ran'))
      .run()
    ),
  )
  print(f'  [tracker] _finally_tracker_17 = {_finally_tracker_17}')

  _finally_tracker_17.clear()

  # CASE 44: Both inner and outer have finally_
  def _case44():
    inner = Q().then(lambda x: 1 / 0).finally_(lambda x: _finally_tracker_17.append('inner_finally'))
    return Q(1).then(inner).finally_(lambda x: _finally_tracker_17.append('outer_finally')).run()

  run_case(
    'CASE 44: Both inner and outer have finally_, inner raises',
    _case44,
  )
  print(f'  [tracker] _finally_tracker_17 = {_finally_tracker_17}')


# ===========================================================================
# SECTION 18 – Pipeline with both except_ and finally_
# ===========================================================================

_finally_tracker_18 = []


def section_except_and_finally():
  _header('SECTION 18 — Pipeline with both except_ and finally_')

  _finally_tracker_18.clear()

  # CASE 45: except_ catches + finally_ runs
  run_case_caught(
    'CASE 45: except_ catches + finally_ runs (no traceback)',
    lambda: (
      Q(1)
      .then(lambda x: 1 / 0)
      .except_(lambda e: f'caught: {e}')
      .finally_(lambda x: _finally_tracker_18.append('finally_45'))
      .run()
    ),
  )
  print(f'  [tracker] _finally_tracker_18 = {_finally_tracker_18}')

  _finally_tracker_18.clear()

  # CASE 46: except_ doesn't catch (wrong type) + finally_ runs
  run_case(
    'CASE 46: except_ wrong type + finally_ runs (traceback shown, finally_ ran)',
    lambda: (
      Q(1)
      .then(lambda x: 1 / 0)
      .except_(lambda e: 'caught', exceptions=TypeError)
      .finally_(lambda x: _finally_tracker_18.append('finally_46'))
      .run()
    ),
  )
  print(f'  [tracker] _finally_tracker_18 = {_finally_tracker_18}')

  _finally_tracker_18.clear()

  # CASE 47: except_ raises + finally_ runs
  def _except_reraises(e):
    raise RuntimeError('except_ re-raised')

  def _case47():
    return (
      Q(1)
      .then(lambda x: 1 / 0)
      .except_(_except_reraises)
      .finally_(lambda x: _finally_tracker_18.append('finally_47'))
      .run()
    )

  run_case(
    'CASE 47: except_ raises new error + finally_ runs (chained traceback)',
    _case47,
  )
  print(f'  [tracker] _finally_tracker_18 = {_finally_tracker_18}')


# ===========================================================================
# SECTION 19 – Multiple chained steps with args and kwargs combinations
# ===========================================================================


def _step_with_args_kwargs(x, extra1, extra2, key=None):
  raise ValueError(f'x={x!r}, extra1={extra1!r}, extra2={extra2!r}, key={key!r}')


def _do_with_args(x, extra, key=None):
  raise ValueError(f'do: x={x!r}, extra={extra!r}, key={key!r}')


def section_args_kwargs():
  _header('SECTION 19 — Multiple chained steps with args and kwargs combinations')

  # CASE 48: .then(fn, arg1, arg2, key=val) that raises
  run_case(
    'CASE 48: .then(fn, arg1, arg2, key=val) raises',
    lambda: Q(1).then(_step_with_args_kwargs, 'a1', 'a2', key='kv').run(),
  )

  # CASE 49: .do(fn, arg1, key=val) that raises
  run_case(
    'CASE 49: .do(fn, arg1, key=val) raises',
    lambda: Q(1).do(_do_with_args, 'extra', key='kval').run(),
  )

  # CASE 50: Nested pipeline as a .then() step with args passed to a step that raises
  def _inner_raises(x):
    raise RuntimeError(f'inner got: {x!r}')

  run_case(
    'CASE 50: Nested pipeline step with inner raises',
    lambda: Q(1).then(Q().then(lambda x: x + 1).then(_inner_raises)).run(),
  )


# ===========================================================================
# SECTION 20 – Async nested pipelines
# ===========================================================================


async def _async_inner_raiser(x):
  raise ValueError('async inner boom')


async def _async_step_ok(x):
  return x + 100


def section_async_nested():
  _header('SECTION 20 — Async nested pipelines')

  # CASE 51: Async inner pipeline raises inside sync outer pipeline
  run_async_case(
    'CASE 51: Async inner pipeline raises inside sync outer pipeline',
    lambda: Q(1).then(Q().then(_async_inner_raiser)).run(),
  )

  # CASE 52: Sync inner pipeline raises inside async outer pipeline
  run_async_case(
    'CASE 52: Sync inner pipeline raises inside async outer (outer has async step before)',
    lambda: Q(1).then(_async_step_ok).then(Q().then(_inner_raiser)).run(),
  )

  # CASE 53: Double-nested async: outer async -> inner async -> innermost raises
  run_async_case(
    'CASE 53: Double-nested async — outer async -> inner async -> innermost raises',
    lambda: (
      Q(1).then(_async_step_ok).then(Q().then(_async_step_ok).then(Q().then(_async_inner_raiser))).run()
    ),
  )


# ===========================================================================
# SECTION 21 – Reusable pipeline variations
# ===========================================================================


def section_reusable_variations():
  _header('SECTION 21 — Reusable pipeline variations')

  # CASE 54: Reusable pipeline with except_ handler that catches
  def _case54():
    q = Q().then(lambda x: 1 / 0).except_(lambda e: f'reusable caught: {e}')
    return q(1)

  run_case_caught(
    'CASE 54: Reusable pipeline with except_ that catches',
    _case54,
  )

  # CASE 55: Reusable pipeline with nested pipeline that raises
  def _case55():
    q = Q().then(Q().then(lambda x: 1 / 0))
    return q(1)

  run_case(
    'CASE 55: Reusable pipeline with nested pipeline that raises',
    _case55,
  )

  # CASE 56: Reusable pipeline called multiple times — 2nd call same formatting
  def _case56():
    q = Q().then(lambda x: 1 / 0)
    results = []
    for i in range(2):
      try:
        q(i)
      except Exception:
        results.append(f'call {i + 1} raised')
        tb_mod.print_exc()
        print()
    return results

  global _total_cases, _traceback_cases
  _total_cases += 1
  _header('CASE 56: Reusable pipeline called twice — both calls show formatting')
  try:
    result = _case56()
    print(f'[no exception]  result = {result!r}')
  except Exception:
    _traceback_cases += 1
    tb_mod.print_exc()
  print()


# ===========================================================================
# SECTION 22 – Edge cases
# ===========================================================================


def section_edge_cases():
  _header('SECTION 22 — Edge cases')

  # CASE 57: Empty pipeline (no root, no links)
  run_case(
    'CASE 57: Empty pipeline — Q().run() (should return None, no exception)',
    lambda: Q().run(),
  )

  # CASE 58: Pipeline with only root, no links
  run_case(
    'CASE 58: Pipeline with only root, no links — Q(42).run() (should return 42)',
    lambda: Q(42).run(),
  )

  # CASE 59: Pipeline root is another pipeline
  run_case(
    'CASE 59: Pipeline root is another pipeline — Q(Q(1).then(lambda x: 1/0)).run()',
    lambda: Q(Q(1).then(lambda x: 1 / 0)).run(),
  )

  # CASE 60: Very long pipeline (20 steps) — verify formatting with many steps
  def _case60():
    q = Q(1)
    for i in range(19):
      q = q.then(lambda x, _i=i: x + 1)
    q = q.then(lambda x: 1 / 0)
    return q.run()

  run_case(
    'CASE 60: Very long pipeline (20 steps), last one raises',
    _case60,
  )

  # CASE 61: Pipeline with functools.partial as a step that raises
  def _partial_target(a, b, c):
    raise ValueError(f'partial: a={a!r}, b={b!r}, c={c!r}')

  run_case(
    'CASE 61: Pipeline with functools.partial as a step that raises',
    lambda: Q(1).then(functools.partial(_partial_target, 'a_val', 'b_val')).run(),
  )

  # CASE 62: Pipeline with a class (constructor) as a step
  run_case(
    'CASE 62: Pipeline with class (int) as step, then lambda raises',
    lambda: Q(1).then(int).then(lambda x: 1 / 0).run(),
  )


# ===========================================================================
# SECTION 23 – Mixed operations pipeline
# ===========================================================================


def section_mixed_ops():
  _header('SECTION 23 — Mixed operations pipeline')

  # CASE 63: Complex pipeline mixing then/do/foreach/with_ — error deep
  def _fail_in_foreach(x):
    raise RuntimeError(f'foreach callback failed on {x!r}')

  run_case(
    'CASE 63: then -> do -> foreach -> foreach pipeline, error in second foreach (deep)',
    lambda: (
      Q([1, 2, 3])
      .then(lambda x: [v * 10 for v in x])
      .do(lambda x: None)
      .foreach(lambda x: x + 1)
      .foreach(_fail_in_foreach)
      .run()
    ),
  )

  # CASE 64: Pipeline with then -> foreach -> foreach, error in second foreach
  run_case(
    'CASE 64: then -> foreach -> foreach pipeline, error in second foreach',
    lambda: Q([1, 2, 0]).then(lambda x: x).foreach(lambda x: x * 2).foreach(lambda x: 10 / x).run(),
  )


# ===========================================================================
# SECTION 24 – Async except/finally handlers
# ===========================================================================


async def _async_except_handler(e):
  return f'async caught: {e}'


async def _async_finally_handler(x):
  _async_finally_tracker.append('async_finally_ran')


_async_finally_tracker = []


def section_async_handlers():
  _header('SECTION 24 — Async except/finally handlers')

  # CASE 65: Async except_ handler that catches
  run_async_case_caught(
    'CASE 65: Async except_ handler that catches',
    lambda: Q(1).then(lambda x: 1 / 0).except_(_async_except_handler).run(),
  )

  # CASE 66: Async finally_ handler + sync pipeline raises
  _async_finally_tracker.clear()

  run_async_case(
    'CASE 66: Async finally_ handler + sync pipeline raises',
    lambda: Q(1).then(_async_step_ok).then(lambda x: 1 / 0).finally_(_async_finally_handler).run(),
  )
  print(f'  [tracker] _async_finally_tracker = {_async_finally_tracker}')


# ===========================================================================
# SECTION 25 – Decorator pipelines
# ===========================================================================


def section_decorator():
  _header('SECTION 25 — Decorator pipelines')

  # CASE 67: Decorated function — Q.as_decorator() wrapping a failing fn
  @Q().then(lambda x: x + 1).then(lambda x: 1 / 0).as_decorator()
  def decorated_simple(x):
    return x

  run_case(
    'CASE 67: Decorated function — q.as_decorator() wrapping, inner link raises',
    lambda: decorated_simple(42),
  )

  # CASE 68: Decorator with named functions mid-pipeline
  def add_one(x):
    return x + 1

  def fail_step(x):
    raise ValueError('decorator pipeline failed')

  @Q().then(add_one).then(fail_step).then(str).as_decorator()
  def decorated_mid_fail(x):
    return x

  run_case(
    'CASE 68: Decorator — named functions, fail in the middle of pipeline',
    lambda: decorated_mid_fail(10),
  )


# ===========================================================================
# SECTION 26 – iterate / iterate_do
# ===========================================================================


def section_iterate():
  _header('SECTION 26 — iterate / iterate_do')

  def _bad_iterate_fn(x):
    if x == 3:
      raise ValueError(f'iterate failed on {x}')
    return x * 10

  # CASE 69: Sync iterate fn raises
  run_case(
    'CASE 69: iterate — sync fn raises on item 3',
    lambda: list(Q(range(5)).iterate(_bad_iterate_fn)),
  )

  # CASE 70: iterate_do fn raises
  def _bad_iterate_do_fn(x):
    if x == 2:
      raise ValueError(f'iterate_do failed on {x}')

  run_case(
    'CASE 70: iterate_do — sync fn raises on item 2',
    lambda: list(Q(range(5)).iterate_do(_bad_iterate_do_fn)),
  )

  # CASE 71: Async iterate fn raises
  async def _async_bad_iterate_fn(x):
    if x == 3:
      raise ValueError(f'async iterate failed on {x}')
    return x * 10

  async def _consume_async_iterate():
    result = []
    async for item in Q(range(5)).iterate(_async_bad_iterate_fn):
      result.append(item)
    return result

  run_async_case(
    'CASE 71: iterate — async fn raises on item 3',
    _consume_async_iterate,
  )


# ===========================================================================
# SECTION 27 – Async context managers
# ===========================================================================


class _AsyncSimpleCM:
  async def __aenter__(self):
    return 'async_ctx_val'

  async def __aexit__(self, *args):
    return False


async def _async_body_raises(ctx):
  raise ValueError(f'async body failed with ctx={ctx!r}')


def section_async_cm():
  _header('SECTION 27 — Async context managers')

  # CASE 72: AsyncCM body raises
  run_async_case(
    'CASE 72: Async context manager — body raises',
    lambda: Q(_AsyncSimpleCM()).with_(_async_body_raises).run(),
  )


# ===========================================================================
# SECTION 28 – Async filter/gather
# ===========================================================================


class _AsyncRange:
  def __init__(self, n):
    self.n = n

  def __aiter__(self):
    self._i = 0
    return self

  async def __anext__(self):
    if self._i >= self.n:
      raise StopAsyncIteration
    val = self._i
    self._i += 1
    return val


async def _async_bad_pred(x):
  if x == 3:
    raise ValueError(f'async filter failed on {x}')
  return x % 2 == 0


async def _async_gather_raises(x):
  raise ValueError('async gather fn failed')


async def _async_gather_ok(x):
  return x + 100


def section_async_filter_gather():
  _header('SECTION 28 — Async filter/gather')

  # CASE 73: Async foreach callback raises
  run_async_case(
    'CASE 73: Async foreach — callback raises on item 3',
    lambda: Q(_AsyncRange(5)).foreach(_async_bad_pred).run(),
  )

  # CASE 74: Async gather fn raises
  run_async_case(
    'CASE 74: Async gather — one async fn raises',
    lambda: Q(1).gather(_async_gather_ok, _async_gather_raises, _async_gather_ok).run(),
  )


# ===========================================================================
# SECTION 29 – Callable objects
# ===========================================================================


class _CallableObj:
  def __call__(self, x):
    return x + 1

  def __repr__(self):
    return '<CallableObj>'


class _CallableObjRaises:
  def __call__(self, x):
    raise ValueError('callable object raised')

  def __repr__(self):
    return '<CallableObjRaises>'


class _AsyncCallableObj:
  async def __call__(self, x):
    raise ValueError('async callable object raised')

  def __repr__(self):
    return '<AsyncCallableObj>'


def section_callable_objects():
  _header('SECTION 29 — Callable objects')

  # CASE 75: Callable object in pipeline, then raise_fn
  def _raise(x):
    raise ValueError('after callable obj')

  run_case(
    'CASE 75: CallableObj in pipeline, then raise_fn',
    lambda: Q(1).then(_CallableObj()).then(_raise).run(),
  )

  # CASE 76: Callable object that raises
  run_case(
    'CASE 76: CallableObj that raises',
    lambda: Q(1).then(_CallableObjRaises()).run(),
  )

  # CASE 77: Async callable object that raises
  run_async_case(
    'CASE 77: AsyncCallableObj that raises',
    lambda: Q(1).then(_AsyncCallableObj()).run(),
  )


# ===========================================================================
# SECTION 30 – sys.excepthook demo
# ===========================================================================


def section_excepthook():
  _header('SECTION 30 — sys.excepthook demo')

  global _total_cases
  _total_cases += 1

  _header('CASE 78: sys.excepthook display for a mid-pipeline failure')
  print('(Simulating what an uncaught exception would show via sys.excepthook)\n')
  try:
    Q(1).then(lambda v: v + 1).then(lambda v: 1 / 0).run()
  except Exception:
    exc_type, exc_value, exc_tb = sys.exc_info()
    sys.excepthook(exc_type, exc_value, exc_tb)
  print()


# ===========================================================================
# MAIN
# ===========================================================================


def main():
  section_basic_sync()
  section_basic_async()
  section_nested_pipelines()
  section_except_handler()
  section_finally_handler()
  section_map()
  section_filter()
  section_with()
  section_gather()
  section_reusable()
  section_deep()
  section_matrix()
  section_args()
  section_control_flow()
  section_chained_exc()
  section_nested_except()
  section_nested_finally()
  section_except_and_finally()
  section_args_kwargs()
  section_async_nested()
  section_reusable_variations()
  section_edge_cases()
  section_mixed_ops()
  section_async_handlers()
  section_decorator()
  section_iterate()
  section_async_cm()
  section_async_filter_gather()
  section_callable_objects()
  section_excepthook()

  # Summary
  bar = '=' * 60
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
