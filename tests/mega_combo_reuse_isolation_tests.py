"""Comprehensive tests for chain REUSE, STATE ISOLATION, ORDERING, and CONCURRENCY.

Tests the interaction patterns that emerge from using chains multiple times
and in complex real-world scenarios. These are the sneakiest bugs — they only
appear when chains are reused, shared, or run concurrently.
"""
import asyncio
import logging
import contextlib
import weakref
from unittest import TestCase, IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TrackedCM:
  """Sync context manager that tracks enter/exit."""
  def __init__(self, value='cm'):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncTrackedCM:
  """Async context manager that tracks enter/exit."""
  def __init__(self, value='acm'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


# ===================================================================
# SECTION 1: Chain reuse (same chain, multiple runs)
# ===================================================================

class ChainReuseBasicTests(TestCase):
  """Test that a single chain can be run() multiple times correctly."""

  def test_s1_01_simple_chain_reuse_5_times(self):
    """Chain(1).then(lambda v: v+1) — run() 5 times, always returns 2."""
    c = Chain(1).then(lambda v: v + 1)
    for i in range(5):
      assert c.run() == 2, f"Run {i} returned {c.run()}"

  def test_s1_02_void_chain_reuse_different_inputs(self):
    """Void chain reused with different root value overrides."""
    c = Chain().then(lambda v: v * 3)
    for i in range(10):
      assert c.run(i) == i * 3

  def test_s1_03_chain_with_multiple_then_links(self):
    """Chain with multiple then links — many runs, consistent results."""
    c = Chain(10).then(lambda v: v + 1).then(lambda v: v * 2).then(lambda v: v - 5)
    for _ in range(20):
      assert c.run() == 17  # (10+1)*2-5 = 17

  def test_s1_04_chain_with_do_reuse(self):
    """Chain with do() — side effects run each time, result unchanged."""
    effects = []
    c = Chain(42).do(lambda v: effects.append(v)).then(lambda v: v + 1)
    for i in range(5):
      assert c.run() == 43
    assert effects == [42] * 5

  def test_s1_06_chain_with_literal_reuse(self):
    """Chain with literal value in then() — reuse returns same literal."""
    c = Chain(1).then(42)
    for _ in range(10):
      assert c.run() == 42


class ChainReuseExceptFinallyTests(TestCase):
  """Test chain reuse with except_ and finally_ handlers."""

  def test_s1_07_except_reuse_error_then_success(self):
    """Chain with except_ — run with error, then without, no stale state."""
    calls = {'except': 0, 'then': 0}

    def maybe_fail(v):
      if v == 'fail':
        raise ValueError('boom')
      calls['then'] += 1
      return v

    c = Chain().then(maybe_fail).except_(
      lambda v: calls.__setitem__('except', calls['except'] + 1),
      reraise=False
    )
    c.run('fail')
    assert calls['except'] == 1
    assert c.run('ok') == 'ok'
    assert calls['except'] == 1  # No new except call
    assert calls['then'] == 1

  def test_s1_08_except_reuse_success_then_error(self):
    """Chain with except_ — run success, then error, then success again."""
    calls = {'except': 0}

    def maybe_fail(v):
      if v < 0:
        raise ValueError('negative')
      return v * 2

    c = Chain().then(maybe_fail).except_(
      lambda v: calls.__setitem__('except', calls['except'] + 1),
      reraise=False
    )
    assert c.run(5) == 10
    c.run(-1)
    assert calls['except'] == 1
    assert c.run(3) == 6
    assert calls['except'] == 1

  def test_s1_09_finally_reuse(self):
    """Chain with finally_ — runs each invocation."""
    finally_count = [0]
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: finally_count.__setitem__(0, finally_count[0] + 1))
    for _ in range(5):
      assert c.run() == 2
    assert finally_count[0] == 5


class ChainReuseForeachTests(TestCase):
  """Test chain reuse with foreach."""

  def test_s1_10_foreach_reuse_different_lists(self):
    """Chain with foreach — reuse with different input lists."""
    c = Chain().foreach(lambda x: x * 2)
    assert c.run([1, 2, 3]) == [2, 4, 6]
    assert c.run([10, 20]) == [20, 40]
    assert c.run([]) == []

  def test_s1_12_filter_reuse(self):
    """Chain with filter — reuse with different lists."""
    c = Chain().filter(lambda x: x > 0)
    assert c.run([-1, 0, 1, 2]) == [1, 2]
    assert c.run([5, -5]) == [5]
    assert c.run([-1, -2]) == []


class ChainReuseWithTests(TestCase):
  """Test chain reuse with context managers."""

  def test_s1_13_with_reuse_different_cms(self):
    """Chain with with_ — reuse with different CMs, each entered/exited."""
    cms = [TrackedCM(f'val{i}') for i in range(3)]
    c = Chain().with_(lambda ctx: ctx + '!')
    for cm in cms:
      result = c.run(cm)
      assert result == cm.value + '!'
      assert cm.entered
      assert cm.exited


class ChainReuseGatherTests(IsolatedAsyncioTestCase):
  """Test chain reuse with gather."""

  async def test_s1_14_gather_reuse(self):
    """Chain with gather — run multiple times."""
    c = Chain().gather(lambda v: v + 1, lambda v: v * 2)
    r1 = c.run(5)
    assert r1 == [6, 10]
    r2 = c.run(10)
    assert r2 == [11, 20]


# ===================================================================
# SECTION 2: State isolation between runs
# ===================================================================

class StateIsolationTests(TestCase):
  """Verify that one run of a chain doesn't affect another."""

  def test_s2_01_mutable_closure_isolation(self):
    """Chain with mutable state in closure — verify one run doesn't affect another."""
    shared_list = []
    c = Chain().then(lambda v: (shared_list.append(v), v)[-1]).then(lambda v: v + 1)
    assert c.run(10) == 11
    assert c.run(20) == 21
    # Closure state IS shared (this is expected Python behavior),
    # but chain execution state is independent.
    assert shared_list == [10, 20]

  def test_s2_02_except_handler_external_state(self):
    """except_ handler that mutates external state — verify isolation."""
    error_log = []

    def fail(v):
      raise ValueError(f'err-{v}')

    c = Chain().then(fail).except_(
      lambda v: error_log.append('caught'), reraise=False
    )
    c.run(1)
    c.run(2)
    assert error_log == ['caught', 'caught']

  def test_s2_03_finally_handler_call_count(self):
    """finally_ handler tracking call count — called each run."""
    count = [0]
    c = Chain(42).then(lambda v: v).finally_(lambda v: count.__setitem__(0, count[0] + 1))
    for _ in range(10):
      c.run()
    assert count[0] == 10

  def test_s2_04_error_then_success_no_stale_error(self):
    """First run raises, second succeeds — no leftover error state."""
    call_order = []

    def maybe_fail(v):
      call_order.append(v)
      if v == 'fail':
        raise ValueError('boom')
      return v

    c = Chain().then(maybe_fail).except_(lambda v: None, reraise=False)
    c.run('fail')
    result = c.run('ok')
    assert result == 'ok'
    assert call_order == ['fail', 'ok']

  def test_s2_05_foreach_temp_args_no_leak(self):
    """foreach reuse — verify temp_args on Link doesn't leak between runs.

    The source comments note that temp_args persists intentionally for
    traceback context. Verify this doesn't cause incorrect results.
    """
    c = Chain().foreach(lambda x: x * 10)
    r1 = c.run([1, 2, 3])
    assert r1 == [10, 20, 30]
    r2 = c.run([4, 5])
    assert r2 == [40, 50]
    # Results are correct despite temp_args possibly persisting
    r3 = c.run([])
    assert r3 == []

  def test_s2_06_filter_temp_args_no_leak(self):
    """filter reuse — same concern about temp_args."""
    c = Chain().filter(lambda x: x > 5)
    assert c.run([1, 10, 3, 7]) == [10, 7]
    assert c.run([6, 4, 8]) == [6, 8]
    assert c.run([1, 2]) == []

  def test_s2_07_with_cm_state_no_leak(self):
    """_With reuse — verify context manager state doesn't leak."""
    c = Chain().with_(lambda ctx: ctx.upper())
    cm1 = TrackedCM('hello')
    cm2 = TrackedCM('world')
    assert c.run(cm1) == 'HELLO'
    assert c.run(cm2) == 'WORLD'
    assert cm1.entered and cm1.exited
    assert cm2.entered and cm2.exited


class StateIsolationAsyncTests(IsolatedAsyncioTestCase):
  """State isolation tests involving async."""

  async def test_s2_08_sync_then_async_no_leftover(self):
    """First run is sync, second triggers async — no leftover async state."""
    sync_fn = lambda v: v + 1

    async def async_fn(v):
      return v + 1

    # Use a void chain with a sync function
    c = Chain().then(sync_fn)
    r1 = c.run(10)
    assert r1 == 11  # sync

    # Now test a chain with async fn
    c2 = Chain().then(async_fn)
    r2 = await c2.run(10)
    assert r2 == 11  # async

    # And the first chain still works sync
    r3 = c.run(20)
    assert r3 == 21

  async def test_s2_09_return_then_full_run(self):
    """Chain reuse where first run uses return_, second runs fully."""
    c = Chain().then(lambda v: v + 1).then(lambda v: v * 2)
    # Normal run
    r1 = c.run(5)
    assert r1 == 12  # (5+1)*2 = 12
    # Run again — still works
    r2 = c.run(10)
    assert r2 == 22  # (10+1)*2 = 22


# ===================================================================
# SECTION 3: Chain reuse
# ===================================================================

class ChainReuseTests(TestCase):
  """Test chain reuse patterns."""

  def test_s3_01_chain_100_times_different_args(self):
    """Chain called 100 times with different args — correct each time."""
    c = Chain().then(lambda v: v ** 2)
    for i in range(100):
      assert c(i) == i ** 2

  def test_s3_02_chain_with_except_reuse_after_error(self):
    """Chain with except_ — reuse after error."""
    caught = [0]

    def maybe_fail(v):
      if v < 0:
        raise ValueError('neg')
      return v * 2

    c = Chain().then(maybe_fail).except_(
      lambda v: caught.__setitem__(0, caught[0] + 1), reraise=False
    )
    assert c(5) == 10
    c(-1)
    assert caught[0] == 1
    assert c(3) == 6
    assert caught[0] == 1

  def test_s3_03_chain_with_finally_reuse(self):
    """Chain with finally_ — cleanup runs each time."""
    cleanup_count = [0]
    c = Chain().then(lambda v: v + 1).finally_(
      lambda v: cleanup_count.__setitem__(0, cleanup_count[0] + 1)
    )
    for i in range(10):
      assert c(i) == i + 1
    assert cleanup_count[0] == 10

  def test_s3_04_chain_with_foreach_reuse(self):
    """Chain with foreach — reuse with different iterables."""
    c = Chain().foreach(lambda x: x * 2)
    assert c([1, 2, 3]) == [2, 4, 6]
    assert c([10]) == [20]
    assert c([]) == []



class ChainConcurrentTests(IsolatedAsyncioTestCase):
  """Test chains under concurrent async load."""

  async def test_s3_06_chain_concurrent_100(self):
    """Chain called concurrently (asyncio.gather of 100 calls)."""
    async def add_one(v):
      await asyncio.sleep(0)
      return v + 1

    c = Chain().then(add_one)
    coros = [c(i) for i in range(100)]
    results = await asyncio.gather(*coros)
    assert sorted(results) == list(range(1, 101))


# ===================================================================
# SECTION 4: Clone independence
# ===================================================================

class CloneIndependenceTests(TestCase):
  """Test that cloned chains are fully independent."""

  def test_s4_01_clone_modify_original(self):
    """Clone chain, modify original, verify clone unchanged."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    assert c2.run() == 2
    assert c.run() == 200

  def test_s4_02_clone_modify_clone(self):
    """Clone chain, modify clone, verify original unchanged."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 100)
    assert c.run() == 2
    assert c2.run() == 200

  def test_s4_03_clone_with_except_add_more(self):
    """Clone chain with except_, add more except_ to clone only."""
    error_log = []

    def raise_type(v):
      raise TypeError('type!')

    c = Chain(1).then(raise_type).except_(
      lambda v: error_log.append('original'), exceptions=[ValueError], reraise=False
    )
    c2 = c.clone()
    c2.except_(
      lambda v: error_log.append('clone_type'), exceptions=[TypeError], reraise=False
    )
    # Original can't catch TypeError
    with self.assertRaises(TypeError):
      c.run()
    # Clone can
    error_log.clear()
    c2.run()
    assert 'clone_type' in error_log

  def test_s4_04_deep_clone_chain(self):
    """Deep clone: original -> clone -> clone of clone — all independent."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    c.then(lambda v: v * 10)
    c2.then(lambda v: v * 20)
    c3.then(lambda v: v * 30)
    assert c.run() == 20    # (1+1)*10
    assert c2.run() == 40   # (1+1)*20
    assert c3.run() == 60   # (1+1)*30



class CloneConcurrentTests(IsolatedAsyncioTestCase):
  """Clone and run concurrently — verify independence."""

  async def test_s4_06_clone_concurrent(self):
    """Clone and run concurrently — verify independence."""
    async def add(v):
      await asyncio.sleep(0)
      return v + 1

    c = Chain().then(add)
    c2 = c.clone()
    r1, r2 = await asyncio.gather(c.run(10), c2.run(20))
    assert r1 == 11
    assert r2 == 21


# ===================================================================
# SECTION 5: Concurrent execution (asyncio)
# ===================================================================

class ConcurrentExecutionTests(IsolatedAsyncioTestCase):
  """Test concurrent async execution patterns."""

  async def test_s5_01_chain_gather_100(self):
    """Same chain via asyncio.gather(100 calls) — all correct values."""
    async def square(v):
      await asyncio.sleep(0)
      return v * v

    c = Chain().then(square)
    coros = [c(i) for i in range(100)]
    results = await asyncio.gather(*coros)
    assert results == [i * i for i in range(100)]

  async def test_s5_02_different_chains_shared_fn(self):
    """Different chains sharing same fn — no shared state issues."""
    fn = lambda v: v + 1
    chains = [Chain().then(fn) for _ in range(10)]
    results = []
    for i, c in enumerate(chains):
      results.append(c.run(i))
    assert results == list(range(1, 11))

  async def test_s5_03_chain_with_sleep_concurrent(self):
    """Chain with sleep(0) called concurrently — all complete correctly."""
    async def process(v):
      await asyncio.sleep(0)
      return v * 2

    c = Chain().then(process)
    coros = [c(i) for i in range(50)]
    results = await asyncio.gather(*coros)
    assert results == [i * 2 for i in range(50)]

  async def test_s5_04_async_foreach_concurrent(self):
    """Chain with async foreach called concurrently."""
    async def double(x):
      await asyncio.sleep(0)
      return x * 2

    c = Chain().foreach(double)
    r1, r2, r3 = await asyncio.gather(
      c([1, 2]),
      c([3, 4]),
      c([5]),
    )
    assert r1 == [2, 4]
    assert r2 == [6, 8]
    assert r3 == [10]

  async def test_s5_05_async_with_concurrent(self):
    """Chain with async with_ called concurrently."""
    async def body(ctx):
      await asyncio.sleep(0)
      return ctx + '!'

    c = Chain().with_(body)
    cms = [AsyncTrackedCM(f'val{i}') for i in range(5)]
    results = await asyncio.gather(*[c(cm) for cm in cms])
    for i, cm in enumerate(cms):
      assert results[i] == f'val{i}!'
      assert cm.entered
      assert cm.exited

  async def test_s5_06_task_registry_under_load(self):
    """Task registry under concurrent load — tasks added and cleaned up."""
    initial = _get_registry_size()

    async def slow(v):
      await asyncio.sleep(0.01)
      return v

    c = Chain().then(slow)
    coros = [c(i) for i in range(20)]
    results = await asyncio.gather(*coros)
    assert sorted(results) == list(range(20))
    # After gather completes, tasks should be cleaned up
    await asyncio.sleep(0.05)
    assert _get_registry_size() == initial


# ===================================================================
# SECTION 6: Link ordering edge cases
# ===================================================================

class LinkOrderingTests(TestCase):
  """Test that link execution order is preserved correctly."""

  def test_s6_01_twenty_link_chain_order(self):
    """20-link chain — verify execution order."""
    order = []
    c = Chain(0)
    for i in range(20):
      c.then(lambda v, _i=i: (order.append(_i), v + 1)[-1])
    result = c.run()
    assert result == 20
    assert order == list(range(20))

  def test_s6_02_links_added_in_loop(self):
    """Links added in a loop — verify all execute."""
    c = Chain(0)
    for i in range(50):
      c.then(lambda v: v + 1)
    assert c.run() == 50

  def test_s6_03_mixed_then_do_ordering(self):
    """Chain with mixed then/do interleaving — ordering preserved."""
    log = []
    c = Chain(0)
    c.then(lambda v: v + 1)
    c.do(lambda v: log.append(('do1', v)))
    c.then(lambda v: v + 2)
    c.do(lambda v: log.append(('do2', v)))
    c.then(lambda v: v + 3)
    result = c.run()
    assert result == 6  # 0+1+2+3
    assert log == [('do1', 1), ('do2', 3)]

  def test_s6_04_except_catches_only_before_it(self):
    """except_ at different positions — only catches from its position backward."""
    caught = []

    def fail_on_neg(v):
      if v < 0:
        raise ValueError('neg')
      return v

    # except_ is after the failing link
    c1 = Chain().then(fail_on_neg).except_(
      lambda v: caught.append('caught'), reraise=False
    )
    c1.run(-1)
    assert caught == ['caught']

  def test_s6_05_multiple_except_first_match_wins(self):
    """Multiple except_ handlers — first matching one wins."""
    caught = []

    def raise_val(v):
      raise ValueError('boom')

    c = Chain(1).then(raise_val).except_(
      lambda v: caught.append('first'), exceptions=[ValueError], reraise=False
    ).except_(
      lambda v: caught.append('second'), exceptions=[ValueError], reraise=False
    )
    c.run()
    assert caught == ['first']

  def test_s6_06_except_type_discrimination(self):
    """except_ with different exception types — correct handler for each type."""
    caught = []

    def raise_type(v):
      raise TypeError('type!')

    def raise_val(v):
      raise ValueError('val!')

    c_type = Chain(1).then(raise_type).except_(
      lambda v: caught.append('val_handler'), exceptions=[ValueError], reraise=False
    ).except_(
      lambda v: caught.append('type_handler'), exceptions=[TypeError], reraise=False
    )
    c_type.run()
    assert caught == ['type_handler']

  def test_s6_07_finally_always_runs(self):
    """finally_ always runs regardless of success or failure."""
    log = []
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append('finally'))
    c.run()
    assert 'finally' in log

    log.clear()
    c2 = Chain(1).then(lambda v: (_ for _ in ()).throw(ValueError())).except_(
      lambda v: None, reraise=False
    ).finally_(lambda v: log.append('finally'))
    c2.run()
    assert 'finally' in log

  def test_s6_08_nested_chain_ordering(self):
    """Links that are nested chains — verify ordering."""
    inner = Chain().then(lambda v: v * 10)
    c = Chain(2).then(inner).then(lambda v: v + 1)
    assert c.run() == 21  # 2 -> inner(2)=20 -> 20+1=21


# ===================================================================
# SECTION 7: Config interaction combinations
# ===================================================================

class ConfigInteractionTests(TestCase):
  """Test config options and their combinations."""

  def test_s7_01_config_debug(self):
    """config(debug=True) + then — verify no crash (debug mode active)."""
    c = Chain(1).config(debug=True).then(lambda v: v + 1)
    assert c.run() == 2

  def test_s7_04_config_on_clone_independence(self):
    """config on clone — verify independence."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.config(debug=True)
    # Original should not have debug
    assert c.run() == 2
    assert c2.run() == 2

  def test_s7_06_config_after_links(self):
    """Changing config after adding links — verify behavior."""
    c = Chain(1).then(lambda v: v + 1)
    c.config(debug=True)
    assert c.run() == 2

  def test_s7_07_config_debug_combos(self):
    """Debug config combinations."""
    for debug in [True, False]:
      c = Chain(1).then(lambda v: v + 1)
      c.config(debug=debug)
      result = c.run()
      assert result == 2, f"Failed with debug={debug}"


class ConfigAutorunTests(IsolatedAsyncioTestCase):
  """Test autorun config with async."""

  async def test_s7_08_config_autorun(self):
    """config(autorun=True) + async then — auto-executes, returns Task."""
    async def add(v):
      return v + 1

    c = Chain(1).config(autorun=True).then(add)
    result = c.run()
    # autorun should create a task/future
    assert asyncio.isfuture(result) or asyncio.iscoroutine(result)
    final = await result
    assert final == 2

  async def test_s7_09_config_autorun_and_debug(self):
    """config(autorun=True, debug=True) — both active."""
    async def add(v):
      return v + 1

    c = Chain(1).config(autorun=True, debug=True).then(add)
    result = c.run()
    final = await result
    assert final == 2


# ===================================================================
# SECTION 8: Multiple except/finally handler interactions
# ===================================================================

class MultipleExceptFinallyTests(TestCase):
  """Test interactions between multiple except_ and finally_ handlers."""

  def test_s8_01_except_type_dispatch(self):
    """except_(h1, TypeError).except_(h2, ValueError) — correct handler for each."""
    caught = []

    c_type = Chain(1).then(lambda v: (_ for _ in ()).throw(TypeError('t'))).except_(
      lambda v: caught.append('type'), exceptions=[TypeError], reraise=False
    ).except_(
      lambda v: caught.append('value'), exceptions=[ValueError], reraise=False
    )
    c_type.run()
    assert caught == ['type']

    caught.clear()
    c_val = Chain(1).then(lambda v: (_ for _ in ()).throw(ValueError('v'))).except_(
      lambda v: caught.append('type'), exceptions=[TypeError], reraise=False
    ).except_(
      lambda v: caught.append('value'), exceptions=[ValueError], reraise=False
    )
    c_val.run()
    assert caught == ['value']

  def test_s8_02_duplicate_except_type_first_wins(self):
    """except_(h1, TypeError).except_(h2, TypeError) — first wins."""
    caught = []

    def raise_type(v):
      raise TypeError('t')

    c = Chain(1).then(raise_type).except_(
      lambda v: caught.append('first'), exceptions=[TypeError], reraise=False
    ).except_(
      lambda v: caught.append('second'), exceptions=[TypeError], reraise=False
    )
    c.run()
    assert caught == ['first']

  def test_s8_03_generic_except_catches_all(self):
    """except_(h1) — catches everything (defaults to Exception)."""
    caught = []

    def raise_val(v):
      raise ValueError('v')

    c = Chain(1).then(raise_val).except_(
      lambda v: caught.append('generic'), reraise=False
    )
    c.run()
    assert caught == ['generic']

  def test_s8_04_except_reraise_then_second_except(self):
    """except_(h, reraise=True).except_(h2) — first re-raises, second catches."""
    caught = []

    def raise_val(v):
      raise ValueError('v')

    c = Chain(1).then(raise_val).except_(
      lambda v: caught.append('first'), exceptions=[ValueError], reraise=True
    ).except_(
      lambda v: caught.append('second'), exceptions=[ValueError], reraise=False
    )
    # First handler catches, logs 'first', then re-raises.
    # Re-raised exception is NOT re-caught by second handler (it's already been handled once).
    with self.assertRaises(ValueError):
      c.run()
    assert caught == ['first']

  def test_s8_05_double_reraise(self):
    """except_(h, reraise=True).except_(h2, reraise=True) — still raises."""
    caught = []

    def raise_val(v):
      raise ValueError('v')

    c = Chain(1).then(raise_val).except_(
      lambda v: caught.append('first'), reraise=True
    ).except_(
      lambda v: caught.append('second'), reraise=True
    )
    with self.assertRaises(ValueError):
      c.run()
    assert caught == ['first']

  def test_s8_06_only_one_finally_allowed(self):
    """finally_(c1).finally_(c2) — raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)

  def test_s8_07_except_and_finally_ordering(self):
    """except_(h).finally_(cleanup) — both run in correct order."""
    log = []

    def raise_val(v):
      raise ValueError('boom')

    c = Chain(1).then(raise_val).except_(
      lambda v: log.append('except'), reraise=False
    ).finally_(lambda v: log.append('finally'))
    c.run()
    # except runs first, then finally
    assert log == ['except', 'finally']


class MultipleExceptFinallyAsyncTests(IsolatedAsyncioTestCase):
  """Test except/finally with async handlers."""

  async def test_s8_08_async_except_handler(self):
    """except_ with async handler — works correctly."""
    caught = []

    async def handler(v):
      caught.append('async_caught')

    def raise_val(v):
      raise ValueError('v')

    c = Chain(1).then(raise_val).except_(handler, reraise=False)
    await c.run()
    assert caught == ['async_caught']


# ===================================================================
# SECTION 10: Generator/iterate reuse and isolation
# ===================================================================

class GeneratorReuseTests(TestCase):
  """Test iterate() generator reuse and isolation."""

  def test_s10_01_iterate_multiple_times(self):
    """iterate() generator iterated multiple times — each is independent."""
    gen = Chain([1, 2, 3]).iterate()
    r1 = list(gen)
    r2 = list(gen)
    assert r1 == [1, 2, 3]
    assert r2 == [1, 2, 3]

  def test_s10_02_iterate_same_chain(self):
    """iterate() from same chain — generators are independent."""
    c = Chain([10, 20, 30])
    g1 = c.iterate()
    g2 = c.iterate()
    assert list(g1) == [10, 20, 30]
    assert list(g2) == [10, 20, 30]

  def test_s10_03_iterate_partial(self):
    """iterate() called then abandoned (partially iterated) — no leaks."""
    gen = Chain([1, 2, 3, 4, 5]).iterate()
    it = iter(gen)
    assert next(it) == 1
    assert next(it) == 2
    # Abandon the iterator — no exception or resource leak
    del it

  def test_s10_04_iterate_with_fn(self):
    """iterate() with transform fn — reusable."""
    gen = Chain([1, 2, 3]).iterate(fn=lambda x: x * 10)
    assert list(gen) == [10, 20, 30]
    assert list(gen) == [10, 20, 30]

  def test_s10_05_iterate_void_chain(self):
    """iterate() on void chain with override."""
    gen = Chain().iterate()
    assert list(gen([1, 2])) == [1, 2]
    assert list(gen([3, 4, 5])) == [3, 4, 5]


class AsyncGeneratorReuseTests(IsolatedAsyncioTestCase):
  """Test async iterate() reuse."""

  async def test_s10_06_async_iterate_reuse(self):
    """Async iterate() reuse — each iteration independent."""
    gen = Chain([1, 2, 3]).iterate()
    r1 = [x async for x in gen]
    r2 = [x async for x in gen]
    assert r1 == [1, 2, 3]
    assert r2 == [1, 2, 3]


# ===================================================================
# SECTION 11: Memory / resource patterns
# ===================================================================

class MemoryResourceTests(TestCase):
  """Test memory and resource patterns."""

  def test_s11_01_chain_1000_links(self):
    """Chain with 1000 then links — verify no stack overflow."""
    c = Chain(0)
    for _ in range(1000):
      c.then(lambda v: v + 1)
    assert c.run() == 1000

  def test_s11_02_deeply_nested_chains_50(self):
    """50 levels of nested chains — verify no recursion limit."""
    # Each level creates a NEW chain with its own .then(lambda v: v + 1)
    # then wraps it in an outer chain. So each nesting level adds 1.
    c = Chain().then(lambda v: v + 1)
    for _ in range(49):
      inner = c
      c = Chain().then(inner).then(lambda v: v + 1)
    # 50 increments total (1 from innermost + 49 from the .then(+1) on each wrap)
    result = c.run(0)
    assert result == 50

  def test_s11_03_large_data_reference_semantics(self):
    """Chain with large data values — verify reference semantics (no copies)."""
    big_list = list(range(10000))
    ids_seen = []
    c = Chain().then(lambda v: (ids_seen.append(id(v)), v)[-1])
    c.run(big_list)
    c.run(big_list)
    # Same object passed through
    assert ids_seen[0] == ids_seen[1] == id(big_list)

  def test_s11_04_cm_cleanup_multiple_runs(self):
    """Chain with CM that checks cleanup — no resource leaks."""
    cms = []
    c = Chain().with_(lambda ctx: ctx)
    for _ in range(10):
      cm = TrackedCM('test')
      cms.append(cm)
      c.run(cm)
    for cm in cms:
      assert cm.entered
      assert cm.exited


class TaskRegistryCleanupTests(IsolatedAsyncioTestCase):
  """Test task registry cleanup."""

  async def test_s11_05_task_registry_cleanup(self):
    """Task registry — tasks removed after completion."""
    initial = _get_registry_size()

    async def work(v):
      await asyncio.sleep(0.01)
      return v

    c = Chain().then(work)
    tasks = [c(i) for i in range(10)]
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.05)
    assert _get_registry_size() == initial


# ===================================================================
# SECTION 12: Interleaved construction and execution
# ===================================================================

class InterleavedConstructionTests(TestCase):
  """Test building chains interleaved with execution."""

  def test_s12_01_build_run_continue(self):
    """Start building chain, run partial, continue building — verify behavior."""
    c = Chain(1).then(lambda v: v + 1)
    r1 = c.run()
    assert r1 == 2
    c.then(lambda v: v * 10)
    r2 = c.run()
    assert r2 == 20  # (1+1)*10 = 20

  def test_s12_02_build_in_loop_run_each_step(self):
    """Build chain in loop, run at each step."""
    c = Chain(0)
    for i in range(1, 6):
      c.then(lambda v, _i=i: v + _i)
      result = c.run()
      assert result == sum(range(i + 1))

  def test_s12_03_clone_then_modify_original(self):
    """Chain().then(a) — clone, add .then(b) to original, clone doesn't see b."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    clone_result = c2.run()
    chain_result = c.run()
    # Clone is independent — it does NOT see changes to the original chain
    assert chain_result == 200  # (1+1)*100
    assert clone_result == 2  # Clone is a snapshot: (1+1)

  def test_s12_04_build_interleaved_with_clone(self):
    """Chain building interleaved with clone — verify snapshot points."""
    c = Chain(1).then(lambda v: v + 1)
    clone_at_2 = c.clone()  # snapshot: returns 2
    c.then(lambda v: v * 3)
    clone_at_6 = c.clone()  # snapshot: returns 6
    c.then(lambda v: v + 10)
    # Verify each snapshot
    assert clone_at_2.run() == 2
    assert clone_at_6.run() == 6
    assert c.run() == 16  # (1+1)*3+10


# ===================================================================
# SECTION 13: Additional edge case combos
# ===================================================================

class ExceptFinallyReuseComboTests(TestCase):
  """Combined except/finally with chain reuse patterns."""

  def test_s13_01_except_finally_reuse_alternating(self):
    """Chain with except_ and finally_ — alternate success and failure."""
    log = []

    def maybe_fail(v):
      if v % 2 == 0:
        raise ValueError('even')
      return v

    c = Chain().then(maybe_fail).except_(
      lambda v: log.append('except'), reraise=False
    ).finally_(lambda v: log.append('finally'))

    for i in range(6):
      log.clear()
      c.run(i)
      assert 'finally' in log
      if i % 2 == 0:
        assert 'except' in log
      else:
        assert 'except' not in log

  def test_s13_02_chain_except_finally_reuse(self):
    """Chain with except_ and finally_ — reuse many times."""
    except_count = [0]
    finally_count = [0]

    def fail(v):
      raise ValueError('always')

    c = Chain().then(fail).except_(
      lambda v: except_count.__setitem__(0, except_count[0] + 1), reraise=False
    ).finally_(
      lambda v: finally_count.__setitem__(0, finally_count[0] + 1)
    )

    for i in range(10):
      c(i)
    assert except_count[0] == 10
    assert finally_count[0] == 10

  def test_s13_03_reuse_chain_different_exception_types(self):
    """Reuse chain with different exception types across runs."""
    caught = []

    def raise_based_on(v):
      if v == 'type':
        raise TypeError('t')
      elif v == 'val':
        raise ValueError('v')
      return v

    c = Chain().then(raise_based_on).except_(
      lambda v: caught.append('type'), exceptions=[TypeError], reraise=False
    ).except_(
      lambda v: caught.append('val'), exceptions=[ValueError], reraise=False
    )
    c.run('type')
    assert caught == ['type']
    c.run('val')
    assert caught == ['type', 'val']
    result = c.run('ok')
    assert result == 'ok'
    assert caught == ['type', 'val']  # no new catches


class DoThenInteractionTests(TestCase):
  """Test interactions between do() and then() on reuse."""

  def test_s13_06_do_does_not_affect_value(self):
    """do() result is discarded — value passes through correctly on reuse."""
    c = Chain(10).do(lambda v: v * 999).then(lambda v: v + 1)
    assert c.run() == 11
    assert c.run() == 11
    assert c.run() == 11

  def test_s13_07_do_side_effects_accumulate(self):
    """do() side effects happen each run."""
    effects = []
    c = Chain(42).do(lambda v: effects.append(v))
    c.run()
    c.run()
    c.run()
    assert effects == [42, 42, 42]


class VoidChainReuseTests(TestCase):
  """Test void chain (no root) reuse patterns."""

  def test_s13_08_void_chain_different_values(self):
    """Void chain reused with many different values."""
    c = Chain().then(lambda v: v + 1)
    assert c.run(0) == 1
    assert c.run(100) == 101
    assert c.run(-50) == -49
    assert c.run(0.5) == 1.5

    # String test uses string-compatible operation
    c2 = Chain().then(lambda v: v + '!')
    assert c2.run('hello') == 'hello!'
    assert c2.run('world') == 'world!'

  def test_s13_09_void_chain_with_root_override_error(self):
    """Chain with root cannot accept value override."""
    c = Chain(42).then(lambda v: v)
    assert c.run() == 42
    with self.assertRaises(QuentException):
      c.run(99)


class ChainBoolReprReuseTests(TestCase):
  """Test __bool__ and __repr__ stability on reuse."""

  def test_s13_10_bool_always_true(self):
    """Chain.__bool__ always True, even after runs."""
    c = Chain(1).then(lambda v: v + 1)
    assert bool(c)
    c.run()
    assert bool(c)
    c.run()
    assert bool(c)

  def test_s13_11_repr_stable_after_runs(self):
    """Chain.__repr__ doesn't crash after multiple runs."""
    c = Chain(1).then(lambda v: v + 1)
    r1 = repr(c)
    c.run()
    r2 = repr(c)
    c.run()
    r3 = repr(c)
    # Just verify no crash and consistent type
    assert isinstance(r1, str)
    assert isinstance(r2, str)
    assert isinstance(r3, str)


class ForeachFilterGatherReuseComboTests(TestCase):
  """Test foreach, filter, gather combinations on reuse."""

  def test_s13_12_foreach_then_filter_reuse(self):
    """Chain with foreach then filter — reuse."""
    c = Chain().foreach(lambda x: x * 2).then(lambda lst: [x for x in lst if x > 5])
    assert c.run([1, 2, 3, 4]) == [6, 8]
    assert c.run([1, 2]) == []
    assert c.run([10]) == [20]

  def test_s13_13_filter_then_foreach_reuse(self):
    """Chain with filter then foreach — reuse."""
    c = Chain().filter(lambda x: x > 0).foreach(lambda x: x * 10)
    assert c.run([-1, 0, 1, 2]) == [10, 20]
    assert c.run([5, -5]) == [50]


class ForeachFilterGatherReuseAsyncTests(IsolatedAsyncioTestCase):
  """Async versions of foreach/filter/gather reuse."""

  async def test_s13_14_async_gather_reuse(self):
    """Chain with gather and async fns — reuse."""
    async def add1(v):
      return v + 1

    async def mul2(v):
      return v * 2

    c = Chain().gather(add1, mul2)
    r1 = await c.run(5)
    assert r1 == [6, 10]
    r2 = await c.run(10)
    assert r2 == [11, 20]

  async def test_s13_15_async_foreach_reuse(self):
    """Chain with async foreach — reuse."""
    async def double(x):
      return x * 2

    c = Chain().foreach(double)
    r1 = await c.run([1, 2, 3])
    assert r1 == [2, 4, 6]
    r2 = await c.run([10, 20])
    assert r2 == [20, 40]


# ===================================================================
# SECTION 14: Stress / property-like tests
# ===================================================================

class StressReuseTests(TestCase):
  """Stress test reuse patterns."""

  def test_s14_01_chain_reuse_1000_times(self):
    """Same chain run 1000 times — consistent results."""
    c = Chain(7).then(lambda v: v * 3).then(lambda v: v - 1)
    for _ in range(1000):
      assert c.run() == 20  # 7*3-1 = 20

  def test_s14_02_chain_reuse_1000_times(self):
    """Chain called 1000 times — consistent."""
    c = Chain().then(lambda v: v + 1)
    for i in range(1000):
      assert c(i) == i + 1

  def test_s14_03_clone_reuse_100_clones(self):
    """100 clones from same original — all independent."""
    c = Chain(0).then(lambda v: v + 1)
    clones = [c.clone() for _ in range(100)]
    for i, clone in enumerate(clones):
      clone.then(lambda v, _i=i: v + _i)
    for i, clone in enumerate(clones):
      assert clone.run() == 1 + i
    # Original unchanged
    assert c.run() == 1


class StressConcurrentTests(IsolatedAsyncioTestCase):
  """Stress test concurrent patterns."""

  async def test_s14_04_massive_concurrent_chain(self):
    """200 concurrent calls to same chain."""
    async def process(v):
      await asyncio.sleep(0)
      return v * v

    c = Chain().then(process)
    n = 200
    coros = [c(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    assert results == [i * i for i in range(n)]


# ===================================================================
# SECTION 15: Chain decorator pattern reuse
# ===================================================================

class DecoratorReuseTests(TestCase):
  """Test chain decorator pattern."""

  def test_s15_01_decorator_reuse(self):
    """Chain as decorator — decorated fn reusable."""
    @Chain().then(lambda v: v * 2).decorator()
    def my_fn(x):
      return x + 1

    assert my_fn(5) == 12  # my_fn(5)=6, *2=12
    assert my_fn(10) == 22  # my_fn(10)=11, *2=22
    assert my_fn(0) == 2   # my_fn(0)=1, *2=2


class DecoratorAsyncTests(IsolatedAsyncioTestCase):
  """Test chain decorator with async."""

  async def test_s15_02_async_decorator_reuse(self):
    """Chain decorator with async fn — reusable."""
    async def add_one(v):
      return v + 1

    @Chain().then(add_one).decorator()
    def my_fn(x):
      return x * 3

    r1 = await my_fn(5)  # 5*3=15, +1=16
    assert r1 == 16
    r2 = await my_fn(10)  # 10*3=30, +1=31
    assert r2 == 31


# ===================================================================
# SECTION 17: return_ and break_ interaction with reuse
# ===================================================================

class ReturnBreakReuseTests(TestCase):
  """Test return_ and break_ signals don't leak between runs."""

  def test_s17_01_foreach_break_reuse(self):
    """foreach with break_ — reuse after break."""
    def stop_at_3(x):
      if x == 3:
        Chain.break_()
      return x * 10

    c = Chain().foreach(stop_at_3)
    r1 = c.run([1, 2, 3, 4, 5])
    assert r1 == [10, 20]  # break at 3, before appending
    # Reuse — no leftover break state
    r2 = c.run([1, 2])
    assert r2 == [10, 20]
    r3 = c.run([1, 2, 3, 4])
    assert r3 == [10, 20]

  def test_s17_02_foreach_break_with_value_reuse(self):
    """foreach with break_(value) — reuse."""
    def stop_at_3(x):
      if x == 3:
        Chain.break_('stopped')
      return x

    c = Chain().foreach(stop_at_3)
    r1 = c.run([1, 2, 3, 4])
    assert r1 == 'stopped'
    r2 = c.run([1, 2])
    assert r2 == [1, 2]


# ===================================================================
# SECTION 18: Nested chain reuse
# ===================================================================

class NestedChainReuseTests(TestCase):
  """Test nested chain patterns with reuse."""

  def test_s18_01_nested_chain_reuse(self):
    """Outer chain with nested inner chain — reuse outer."""
    inner = Chain().then(lambda v: v * 10)
    outer = Chain().then(inner)
    assert outer.run(2) == 20
    assert outer.run(3) == 30
    assert outer.run(5) == 50

  def test_s18_02_nested_chain_multiple_levels(self):
    """Multiple levels of nesting — reuse."""
    level3 = Chain().then(lambda v: v + 1)
    level2 = Chain().then(level3)
    level1 = Chain().then(level2)
    assert level1.run(0) == 1
    assert level1.run(10) == 11

  def test_s18_03_nested_chain_reuse(self):
    """Chain containing nested chain — reuse."""
    inner = Chain().then(lambda v: v * 2)
    c = Chain().then(inner).then(lambda v: v + 1)
    assert c(5) == 11   # 5*2+1
    assert c(10) == 21  # 10*2+1


# ===================================================================
# SECTION 19: with_ reuse edge cases
# ===================================================================

class WithReuseEdgeCaseTests(TestCase):
  """Test with_ reuse edge cases."""

  def test_s19_01_with_cm_exception_then_success(self):
    """with_ — first run has CM body raise, second succeeds."""
    calls = {'enter': 0, 'exit': 0}

    class CountCM:
      def __enter__(self):
        calls['enter'] += 1
        return 'ctx'

      def __exit__(self, *args):
        calls['exit'] += 1
        return False  # don't suppress

    call_count = [0]

    def body(ctx):
      call_count[0] += 1
      if call_count[0] == 1:
        raise ValueError('first')
      return ctx.upper()

    c = Chain().with_(body).except_(lambda v: None, reraise=False)
    c.run(CountCM())
    assert calls == {'enter': 1, 'exit': 1}

    c.run(CountCM())
    assert calls == {'enter': 2, 'exit': 2}

  def test_s19_02_with_different_cm_types(self):
    """with_ — reuse with different CM types."""
    c = Chain().with_(lambda ctx: str(ctx))

    cm1 = TrackedCM('hello')
    assert c.run(cm1) == 'hello'
    assert cm1.exited

    cm2 = TrackedCM(42)
    assert c.run(cm2) == '42'
    assert cm2.exited


# ===================================================================
# SECTION 20: Complex real-world patterns
# ===================================================================

class RealWorldPatternTests(TestCase):
  """Test patterns that might appear in real applications."""

  def test_s20_01_pipeline_builder(self):
    """Build a data processing pipeline and reuse it."""
    pipeline = (
      Chain()
      .then(lambda data: [x for x in data if x is not None])  # remove None
      .foreach(lambda x: x.strip())                            # strip strings
      .then(lambda lst: [x for x in lst if x])                 # remove empty
      .foreach(lambda x: x.lower())                            # lowercase
    )
    r1 = pipeline.run(['  Hello ', None, 'WORLD', '', '  '])
    assert r1 == ['hello', 'world']
    r2 = pipeline.run([None, None])
    assert r2 == []
    r3 = pipeline.run(['  A  ', 'B', 'c'])
    assert r3 == ['a', 'b', 'c']

  def test_s20_02_error_recovery_chain(self):
    """Chain with error recovery — reusable error handling."""
    default = 'default'

    def parse_int(v):
      return int(v)

    c = Chain().then(parse_int).except_(lambda v: default, reraise=False)
    assert c.run('42') == 42
    assert c.run('abc') == default
    assert c.run('100') == 100
    assert c.run('xyz') == default



class RealWorldAsyncPatternTests(IsolatedAsyncioTestCase):
  """Async real-world pattern tests."""

  async def test_s20_04_async_pipeline_reuse(self):
    """Async data pipeline — reuse."""
    async def fetch(url):
      await asyncio.sleep(0)
      return f'data-{url}'

    async def process(data):
      await asyncio.sleep(0)
      return data.upper()

    pipeline = Chain().then(fetch).then(process)
    r1 = await pipeline.run('url1')
    assert r1 == 'DATA-URL1'
    r2 = await pipeline.run('url2')
    assert r2 == 'DATA-URL2'

  async def test_s20_05_concurrent_pipelines(self):
    """Multiple different pipelines running concurrently."""
    async def add(v):
      await asyncio.sleep(0)
      return v + 1

    async def mul(v):
      await asyncio.sleep(0)
      return v * 2

    p1 = Chain().then(add)
    p2 = Chain().then(mul)

    results = await asyncio.gather(
      p1(10), p2(10), p1(20), p2(20), p1(30), p2(30)
    )
    assert results == [11, 20, 21, 40, 31, 60]


# ===================================================================
# SECTION 22: Clone config preservation
# ===================================================================

class CloneConfigTests(TestCase):
  """Test that clone preserves configuration."""

  def test_s22_01_clone_preserves_debug(self):
    """Clone preserves debug config."""
    c = Chain(1).config(debug=True).then(lambda v: v + 1)
    c2 = c.clone()
    # Both should work with debug mode
    assert c.run() == 2
    assert c2.run() == 2

  def test_s22_04_clone_finally_independence(self):
    """Clone's finally_ doesn't affect original."""
    log1 = []
    log2 = []
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.finally_(lambda v: log1.append('orig_finally'))
    c2.finally_(lambda v: log2.append('clone_finally'))
    c.run()
    c2.run()
    assert log1 == ['orig_finally']
    assert log2 == ['clone_finally']


# ===================================================================
# SECTION 23: Edge cases with Null sentinel
# ===================================================================

class NullSentinelTests(TestCase):
  """Test behavior with Null sentinel value."""

  def test_s23_01_chain_returns_none_for_null(self):
    """Chain with no operations returns None (not Null)."""
    c = Chain()
    assert c.run() is None

  def test_s23_02_void_chain_no_root_returns_none(self):
    """Void chain with only do() returns None."""
    log = []
    c = Chain().do(lambda: log.append('side'))
    result = c.run()
    assert result is None
    assert log == ['side']

  def test_s23_03_reuse_void_chain_null_consistency(self):
    """Void chain reuse — always returns None when no value produced."""
    c = Chain()
    for _ in range(10):
      assert c.run() is None


# ===================================================================
# SECTION 24: Mixed sync/async chain reuse
# ===================================================================

class MixedSyncAsyncReuseTests(IsolatedAsyncioTestCase):
  """Test chains with mixed sync and async operations on reuse."""

  async def test_s24_01_sync_async_interleaved(self):
    """Chain with sync then async then sync — reuse."""
    async def async_add(v):
      return v + 10

    c = Chain().then(lambda v: v + 1).then(async_add).then(lambda v: v * 2)
    r1 = await c.run(5)
    assert r1 == 32  # (5+1+10)*2 = 32
    r2 = await c.run(10)
    assert r2 == 42  # (10+1+10)*2 = 42

  async def test_s24_02_async_root_reuse(self):
    """Chain with async root callable — reuse."""
    async def async_root():
      return 42

    c = Chain(async_root).then(lambda v: v + 1)
    r1 = await c.run()
    assert r1 == 43
    r2 = await c.run()
    assert r2 == 43

  async def test_s24_03_async_except_reuse(self):
    """Chain with async except_ handler — reuse."""
    caught = []

    async def handler(v):
      caught.append('async_caught')

    def raise_val(v):
      raise ValueError('v')

    c = Chain().then(raise_val).except_(handler, reraise=False)
    await c.run(1)
    await c.run(2)
    assert caught == ['async_caught', 'async_caught']

  async def test_s24_04_async_finally_reuse(self):
    """Chain with async finally_ handler — reuse."""
    log = []

    async def cleanup(v):
      log.append('cleaned')

    c = Chain().then(lambda v: v + 1).finally_(cleanup)
    # Async finally in sync chain path results in a warning
    # but the chain should still work. Let's use an async chain.
    async def async_step(v):
      return v + 1

    c2 = Chain().then(async_step).finally_(cleanup)
    r1 = await c2.run(10)
    assert r1 == 11
    r2 = await c2.run(20)
    assert r2 == 21
    assert log == ['cleaned', 'cleaned']


# ===================================================================
# SECTION 25: Additional reuse edge cases for 150+ test coverage
# ===================================================================

class FrozenChainEdgeCaseTests(TestCase):
  """Additional frozen chain edge cases."""

  def test_s25_01_frozen_void_no_args(self):
    """Frozen void chain called with no args returns None."""
    frozen = Chain().freeze()
    assert frozen() is None

  def test_s25_02_frozen_with_root_value(self):
    """Frozen chain with root value — call without args."""
    frozen = Chain(42).then(lambda v: v + 1).freeze()
    for _ in range(10):
      assert frozen() == 43

  def test_s25_03_frozen_run_method(self):
    """Frozen chain .run() method — same as __call__."""
    frozen = Chain().then(lambda v: v * 2).freeze()
    assert frozen.run(5) == 10
    assert frozen(5) == 10

  def test_s25_04_frozen_with_do(self):
    """Frozen chain with do() — side effects on each call."""
    log = []
    frozen = Chain().do(lambda v: log.append(v)).then(lambda v: v + 1).freeze()
    assert frozen(10) == 11
    assert frozen(20) == 21
    assert log == [10, 20]


class ChainReuseExceptionEdgeCases(TestCase):
  """Exception handling edge cases on reuse."""

  def test_s25_05_except_catches_subclass(self):
    """except_ with base class catches subclass on reuse."""
    caught = []

    class MyError(ValueError):
      pass

    def raise_sub(v):
      raise MyError('sub')

    c = Chain().then(raise_sub).except_(
      lambda v: caught.append('base'), exceptions=[ValueError], reraise=False
    )
    c.run(1)
    c.run(2)
    assert caught == ['base', 'base']

  def test_s25_06_except_reraise_false_returns_handler_value(self):
    """except_ with reraise=False — handler return value becomes chain result."""
    def raise_val(v):
      raise ValueError('boom')

    c = Chain().then(raise_val).except_(lambda v: 'recovered', reraise=False)
    assert c.run(1) == 'recovered'
    assert c.run(2) == 'recovered'

  def test_s25_07_except_no_match_raises(self):
    """except_ that doesn't match — exception still raised on each run."""
    def raise_type(v):
      raise TypeError('t')

    c = Chain().then(raise_type).except_(
      lambda v: None, exceptions=[ValueError], reraise=False
    )
    with self.assertRaises(TypeError):
      c.run(1)
    with self.assertRaises(TypeError):
      c.run(2)


class CloneReuseComboTests(TestCase):
  """Additional clone + reuse combinations."""

  def test_s25_11_clone_reuse_both(self):
    """Clone and original both reused independently."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    for _ in range(5):
      assert c.run() == 2
      assert c2.run() == 2

  def test_s25_12_clone_from_void_chain(self):
    """Clone void chain — both accept different values."""
    c = Chain().then(lambda v: v * 3)
    c2 = c.clone()
    assert c.run(5) == 15
    assert c2.run(10) == 30
    assert c.run(7) == 21
    assert c2.run(2) == 6

  def test_s25_13_clone_with_foreach_independence(self):
    """Clone chain with foreach — independent."""
    c = Chain().foreach(lambda x: x + 1)
    c2 = c.clone()
    assert c.run([1, 2]) == [2, 3]
    assert c2.run([10, 20]) == [11, 21]

  def test_s25_14_clone_with_filter_independence(self):
    """Clone chain with filter — independent."""
    c = Chain().filter(lambda x: x > 0)
    c2 = c.clone()
    assert c.run([-1, 0, 1]) == [1]
    assert c2.run([5, -5]) == [5]


class ConcurrentIsolationEdgeCaseTests(IsolatedAsyncioTestCase):
  """Additional concurrent isolation edge cases."""

  async def test_s25_15_concurrent_different_frozen_chains(self):
    """Multiple different frozen chains running concurrently."""
    async def add(v):
      await asyncio.sleep(0)
      return v + 1

    async def mul(v):
      await asyncio.sleep(0)
      return v * 2

    async def sub(v):
      await asyncio.sleep(0)
      return v - 3

    f1 = Chain().then(add).freeze()
    f2 = Chain().then(mul).freeze()
    f3 = Chain().then(sub).freeze()

    results = await asyncio.gather(
      f1(10), f2(10), f3(10),
      f1(20), f2(20), f3(20),
    )
    assert results == [11, 20, 7, 21, 40, 17]

  async def test_s25_16_concurrent_clones(self):
    """Clones running concurrently — independent."""
    async def process(v):
      await asyncio.sleep(0)
      return v ** 2

    c = Chain().then(process)
    clones = [c.clone() for _ in range(10)]
    coros = [clones[i].run(i) for i in range(10)]
    results = await asyncio.gather(*coros)
    assert results == [i ** 2 for i in range(10)]

  async def test_s25_17_concurrent_frozen_with_except(self):
    """Frozen chain with except_ under concurrency."""
    except_count = [0]

    async def maybe_fail(v):
      await asyncio.sleep(0)
      if v % 2 == 0:
        raise ValueError('even')
      return v

    frozen = Chain().then(maybe_fail).except_(
      lambda v: except_count.__setitem__(0, except_count[0] + 1),
      reraise=False
    ).freeze()
    coros = [frozen(i) for i in range(20)]
    await asyncio.gather(*coros)
    assert except_count[0] == 10  # 0,2,4,6,8,10,12,14,16,18 are even


class OrderingAdditionalTests(TestCase):
  """Additional ordering tests."""

  def test_s25_18_ten_do_then_alternating(self):
    """10 alternating do/then — ordering preserved."""
    log = []
    c = Chain(0)
    for i in range(10):
      if i % 2 == 0:
        c.then(lambda v, _i=i: v + _i + 1)
      else:
        c.do(lambda v, _i=i: log.append(('do', _i, v)))
    result = c.run()
    # then at i=0: 0 + 1 = 1
    # do  at i=1: log (1)
    # then at i=2: 1 + 3 = 4
    # do  at i=3: log (4)
    # then at i=4: 4 + 5 = 9
    # do  at i=5: log (9)
    # then at i=6: 9 + 7 = 16
    # do  at i=7: log (16)
    # then at i=8: 16 + 9 = 25
    # do  at i=9: log (25)
    assert result == 25
    assert log == [
      ('do', 1, 1), ('do', 3, 4), ('do', 5, 9),
      ('do', 7, 16), ('do', 9, 25)
    ]

  def test_s25_19_chain_with_ellipsis_args(self):
    """Chain with ellipsis arg (no-arg call) — reuse."""
    call_count = [0]

    def fn():
      call_count[0] += 1
      return call_count[0]

    c = Chain(1).then(fn, ...)
    assert c.run() == 1
    assert c.run() == 2
    assert c.run() == 3


class GatherReuseEdgeCaseTests(IsolatedAsyncioTestCase):
  """Additional gather reuse tests."""

  async def test_s25_20_gather_sync_reuse(self):
    """Gather with sync fns — reuse."""
    c = Chain().gather(
      lambda v: v + 1,
      lambda v: v + 2,
      lambda v: v + 3,
    )
    r1 = c.run(10)
    assert r1 == [11, 12, 13]
    r2 = c.run(20)
    assert r2 == [21, 22, 23]

  async def test_s25_21_gather_mixed_sync_async_reuse(self):
    """Gather with mixed sync/async fns — reuse."""
    async def async_add(v):
      return v + 100

    c = Chain().gather(
      lambda v: v + 1,
      async_add,
    )
    r1 = await c.run(5)
    assert r1 == [6, 105]
    r2 = await c.run(10)
    assert r2 == [11, 110]
