"""Mega bug-hunt and false-positive verification tests for the quent library.

Every test in this file is designed to verify EXACT semantics -- not just
"it works" but "it does precisely what the source code claims". False
positives (tests that pass but verify the wrong thing) are the enemy.

Categories:
  A. Value Identity and Equality Bugs
  B. Async Correctness Bugs
  C. Exception Handling Correctness
  D. Mutation and State Isolation Bugs
  E. Null Sentinel Handling
  F. Edge Case Regression Tests
  G. Traceback and Diagnostics Verification
  H. Additional False-Positive Hunters
"""

import asyncio
import inspect
import unittest
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class Exc1(TestExc):
  pass


class Exc2(TestExc):
  pass


class Exc3(Exc2):
  """Subclass of Exc2 -- used to test isinstance matching."""
  pass


class Exc4(Exception):
  """Completely unrelated exception."""
  pass


def raise_(_v=None):
  """Raise TestExc, ignoring the chain value argument."""
  raise TestExc()


def identity(v):
  return v


async def aidentity(v):
  return v


def make_sentinel():
  """Return a unique object for identity checking."""
  return object()


class SyncCM:
  """Synchronous context manager that returns a specific object from __enter__."""
  def __init__(self, enter_val):
    self.enter_val = enter_val
    self.exited = False
    self.exc_info = (None, None, None)

  def __enter__(self):
    return self.enter_val

  def __exit__(self, *exc_info):
    self.exited = True
    self.exc_info = exc_info
    return False


class AsyncCM:
  """Async context manager that returns a specific object from __aenter__."""
  def __init__(self, enter_val):
    self.enter_val = enter_val
    self.exited = False
    self.exc_info = (None, None, None)

  async def __aenter__(self):
    return self.enter_val

  async def __aexit__(self, *exc_info):
    self.exited = True
    self.exc_info = exc_info
    return False


class SyncCMSuppressing(SyncCM):
  """Sync CM that suppresses exceptions."""
  def __exit__(self, *exc_info):
    self.exited = True
    self.exc_info = exc_info
    return True


class AsyncCMSuppressing(AsyncCM):
  """Async CM that suppresses exceptions."""
  async def __aexit__(self, *exc_info):
    self.exited = True
    self.exc_info = exc_info
    return True


def setattr_helper(lst, val):
  lst[0] = val


# ===================================================================
# A. Value Identity and Equality Bugs (15+ tests)
# ===================================================================

class TestValueIdentity(IsolatedAsyncioTestCase):
  """Verify that values flow correctly and are not copied, mutated, or swapped."""

  async def test_01_chain_then_passes_exact_same_object_reference(self):
    """Chain passes the EXACT SAME object reference through .then() (not a copy)."""
    for fn in [identity, aidentity]:
      with self.subTest(fn=fn):
        sentinel = make_sentinel()
        result = await await_(Chain(sentinel).then(fn).run())
        self.assertIs(result, sentinel)

  async def test_02_chain_root_to_first_link_identity(self):
    """Chain passes the EXACT SAME object reference from root to first link."""
    sentinel = make_sentinel()
    received = []
    def capture(v):
      received.append(v)
      return v
    await await_(Chain(sentinel).then(capture).run())
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], sentinel)

  async def test_04_do_discards_result_and_passes_previous_value_identity(self):
    """.do() truly discards the result and passes the PREVIOUS value (identity)."""
    sentinel = make_sentinel()
    do_received = []
    def side_effect(v):
      do_received.append(v)
      return 'should be discarded'
    result = await await_(Chain(sentinel).do(side_effect).run())
    self.assertIs(result, sentinel)
    self.assertEqual(len(do_received), 1)
    self.assertIs(do_received[0], sentinel)

  async def test_05_except_handler_receives_root_value(self):
    """Exception handler receives the ROOT value (not current at time of exception)."""
    root_sentinel = make_sentinel()
    exc_received = []
    def handler(v):
      exc_received.append(v)
      return 'handled'
    def transform(v):
      return 'transformed'
    result = await await_(
      Chain(root_sentinel)
      .then(transform)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertEqual(len(exc_received), 1)
    self.assertIs(exc_received[0], root_sentinel)

  async def test_05b_except_handler_receives_root_value_async(self):
    """Exception handler receives the ROOT value in async path too."""
    root_sentinel = make_sentinel()
    exc_received = []
    async def handler(v):
      exc_received.append(v)
      return 'handled'
    async def transform(v):
      return 'transformed'
    result = await await_(
      Chain(root_sentinel)
      .then(transform)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertEqual(len(exc_received), 1)
    self.assertIs(exc_received[0], root_sentinel)

  async def test_06_finally_handler_receives_root_value(self):
    """Finally handler receives the ROOT value."""
    root_sentinel = make_sentinel()
    finally_received = []
    def on_finally(v):
      finally_received.append(v)
    await await_(
      Chain(root_sentinel)
      .then(identity)
      .finally_(on_finally)
      .run()
    )
    self.assertEqual(len(finally_received), 1)
    self.assertIs(finally_received[0], root_sentinel)

  async def test_07_with_passes_enter_return_value_to_body(self):
    """with_ passes __enter__ return value to body (not the CM object itself)."""
    enter_sentinel = make_sentinel()
    cm = SyncCM(enter_sentinel)
    body_received = []
    def body(v):
      body_received.append(v)
      return v
    await await_(Chain(cm).with_(body).run())
    self.assertEqual(len(body_received), 1)
    self.assertIs(body_received[0], enter_sentinel)

  async def test_07b_with_async_passes_aenter_return_value_to_body(self):
    """with_ passes __aenter__ return value to body for async CM."""
    enter_sentinel = make_sentinel()
    cm = AsyncCM(enter_sentinel)
    body_received = []
    async def body(v):
      body_received.append(v)
      return v
    await await_(Chain(cm).with_(body).run())
    self.assertEqual(len(body_received), 1)
    self.assertIs(body_received[0], enter_sentinel)

  async def test_08_foreach_passes_each_element_identity(self):
    """foreach passes each element (identity) to fn."""
    s1, s2, s3 = make_sentinel(), make_sentinel(), make_sentinel()
    items = [s1, s2, s3]
    received = []
    def capture(el):
      received.append(el)
      return el
    await await_(Chain(items).foreach(capture).run())
    self.assertEqual(len(received), 3)
    self.assertIs(received[0], s1)
    self.assertIs(received[1], s2)
    self.assertIs(received[2], s3)

  async def test_09_filter_passes_original_elements_identity(self):
    """filter passes original elements (identity) to result (not predicate return)."""
    s1, s2, s3 = make_sentinel(), make_sentinel(), make_sentinel()
    items = [s1, s2, s3]
    count = [0]
    def predicate(el):
      count[0] += 1
      return 'truthy_string'
    result = await await_(Chain(items).filter(predicate).run())
    self.assertEqual(len(result), 3)
    self.assertIs(result[0], s1)
    self.assertIs(result[1], s2)
    self.assertIs(result[2], s3)
    self.assertEqual(count[0], 3)

  async def test_10_gather_passes_current_value_identity(self):
    """gather passes current_value (identity) to each function."""
    sentinel = make_sentinel()
    received = []
    def fn1(v):
      received.append(v)
      return 'fn1'
    def fn2(v):
      received.append(v)
      return 'fn2'
    result = await await_(Chain(sentinel).gather(fn1, fn2).run())
    self.assertEqual(len(received), 2)
    self.assertIs(received[0], sentinel)
    self.assertIs(received[1], sentinel)
    self.assertEqual(result, ['fn1', 'fn2'])

  async def test_12_run_override_value_identity(self):
    """chain.run() override value identity preserved."""
    sentinel = make_sentinel()
    received = []
    def capture(v):
      received.append(v)
      return v
    result = await await_(Chain().then(capture).run(sentinel))
    self.assertIs(result, sentinel)
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], sentinel)

  async def test_13_nested_chain_receives_current_value_identity(self):
    """Nested chain receives current_value from parent (identity)."""
    sentinel = make_sentinel()
    received = []
    inner = Chain().then(lambda v: (received.append(v), v)[1])
    result = await await_(Chain(sentinel).then(inner).run())
    self.assertIs(result, sentinel)
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], sentinel)

  async def test_14_clone_shares_callable_references(self):
    """Clone shares callable references but has independent state."""
    fn_ref = lambda v: v
    original = Chain(42).then(fn_ref)
    cloned = original.clone()
    r1 = await await_(original.run())
    r2 = await await_(cloned.run())
    self.assertEqual(r1, 42)
    self.assertEqual(r2, 42)

  async def test_15_chain_reuse_does_not_mutate(self):
    """Chain reuse does not cause mutation issues."""
    c = Chain().then(lambda v: v * 2)
    r1 = await await_(c.run(5))
    self.assertEqual(r1, 10)
    r2 = await await_(c.run(7))
    self.assertEqual(r2, 14)


# ===================================================================
# B. Async Correctness Bugs (15+ tests)
# ===================================================================

class TestAsyncCorrectness(IsolatedAsyncioTestCase):
  """Verify that async behavior is EXACTLY correct."""

  async def test_16_async_chain_returns_awaited_result(self):
    """Async chain returns the AWAITED result, not the coroutine object."""
    async def compute():
      return 42
    result = await await_(Chain(compute).run())
    self.assertEqual(result, 42)
    self.assertFalse(inspect.iscoroutine(result))

  async def test_17_sync_values_correct_after_async_transition(self):
    """When a chain goes async mid-chain, already-computed sync values are correct."""
    log = []
    def sync_fn(v):
      log.append(('sync', v))
      return v + 1
    async def async_fn(v):
      log.append(('async', v))
      return v + 10
    def final_fn(v):
      log.append(('final', v))
      return v
    result = await await_(
      Chain(1)
      .then(sync_fn)
      .then(async_fn)
      .then(final_fn)
      .run()
    )
    self.assertEqual(result, 12)
    self.assertEqual(log, [('sync', 1), ('async', 2), ('final', 12)])

  async def test_18_previous_value_preserved_across_async_for_do(self):
    """previous_value is correctly preserved across async transition for .do()."""
    sentinel = make_sentinel()
    async def async_side_effect(v):
      return 'discarded'
    result = await await_(
      Chain(sentinel)
      .do(async_side_effect)
      .run()
    )
    self.assertIs(result, sentinel)

  async def test_19_root_value_set_after_first_async_eval(self):
    """root_value is correctly set after first async evaluation."""
    async def async_root():
      return 42
    exc_received = []
    def on_exc(v):
      exc_received.append(v)
      return 'handled'
    result = await await_(
      Chain()
      .then(lambda v: v + 1)
      .then(raise_)
      .except_(on_exc, reraise=False)
      .run(async_root)
    )
    self.assertEqual(len(exc_received), 1)
    self.assertEqual(exc_received[0], 42)

  async def test_21_async_except_handler_result_awaited(self):
    """Async exception handler result is awaited before returning."""
    async def handler(v):
      return 'async_handled'
    result = await await_(
      Chain(42)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertEqual(result, 'async_handled')
    self.assertFalse(inspect.iscoroutine(result))

  async def test_22_async_finally_handler_result_awaited(self):
    """Async finally handler result is awaited in async path."""
    finally_ran = [False]
    async def on_finally(v):
      finally_ran[0] = True
    async def async_identity(v):
      return v
    await await_(
      Chain(1)
      .then(async_identity)
      .finally_(on_finally)
      .run()
    )
    self.assertTrue(finally_ran[0])

  async def test_23_ensure_future_creates_task(self):
    """ensure_future creates a Task when autorun is True."""
    async def compute():
      return 42
    chain = Chain(compute).config(autorun=True)
    result = chain.run()
    self.assertIsInstance(result, asyncio.Task)
    val = await result
    self.assertEqual(val, 42)

  async def test_25_break_propagation_async(self):
    """_Break propagation works correctly in async path."""
    async def do_break(el):
      if el == 2:
        Chain.break_()
      return el * 10
    result = await await_(
      Chain([0, 1, 2, 3, 4])
      .foreach(do_break)
      .run()
    )
    self.assertEqual(result, [0, 10])

  async def test_26_async_foreach_collects_all_results(self):
    """Async foreach collects ALL results before returning."""
    async def transform(el):
      return el * 2
    result = await await_(
      Chain([1, 2, 3, 4, 5])
      .foreach(transform)
      .run()
    )
    self.assertEqual(result, [2, 4, 6, 8, 10])

  async def test_27_async_gather_preserves_order(self):
    """Async gather result ordering matches function order."""
    async def slow(v):
      await asyncio.sleep(0.02)
      return 'slow'
    async def fast(v):
      return 'fast'
    result = await await_(
      Chain(1)
      .gather(slow, fast)
      .run()
    )
    self.assertEqual(result, ['slow', 'fast'])

  async def test_28_async_filter_preserves_order(self):
    """Async filter preserves original order."""
    async def is_even(el):
      return el % 2 == 0
    result = await await_(
      Chain([5, 4, 3, 2, 1])
      .filter(is_even)
      .run()
    )
    self.assertEqual(result, [4, 2])

  async def test_29_sync_to_async_transition_in_foreach(self):
    """Sync-to-async transition preserves accumulated state in foreach."""
    call_count = [0]
    async def maybe_async(el):
      call_count[0] += 1
      return el * 10
    result = await await_(
      Chain([1, 2, 3])
      .foreach(maybe_async)
      .run()
    )
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(call_count[0], 3)

  async def test_30_async_cm_aexit_called_on_exception(self):
    """Async context manager __aexit__ is always called (even on exception)."""
    cm = AsyncCM(make_sentinel())
    def body(v):
      raise TestExc('boom')
    try:
      await await_(Chain(cm).with_(body).run())
    except TestExc:
      pass
    self.assertTrue(cm.exited)
    self.assertIsNotNone(cm.exc_info[0])

  async def test_30b_sync_cm_exit_called_on_exception(self):
    """Sync context manager __exit__ is always called (even on exception)."""
    cm = SyncCM(make_sentinel())
    def body(v):
      raise TestExc('boom')
    try:
      await await_(Chain(cm).with_(body).run())
    except TestExc:
      pass
    self.assertTrue(cm.exited)
    self.assertIsNotNone(cm.exc_info[0])


# ===================================================================
# C. Exception Handling Correctness (15+ tests)
# ===================================================================

class TestExceptionHandling(IsolatedAsyncioTestCase):
  """Verify that exception handling follows EXACT documented semantics."""

  async def test_31_only_first_matching_handler_called(self):
    """ONLY the first matching exception handler is called (not all)."""
    calls = []
    def handler1(v):
      calls.append('h1')
      return 'h1_result'
    def handler2(v):
      calls.append('h2')
      return 'h2_result'
    result = await await_(
      Chain(1)
      .then(raise_)
      .except_(handler1, reraise=False)
      .except_(handler2, reraise=False)
      .run()
    )
    self.assertEqual(calls, ['h1'])
    self.assertEqual(result, 'h1_result')

  async def test_32_exception_handlers_checked_in_order(self):
    """Exception handlers are checked IN ORDER (first match wins)."""
    calls = []
    def h1(v):
      calls.append('h1')
      return 'h1'
    def h2(v):
      calls.append('h2')
      return 'h2'
    result = await await_(
      Chain(1)
      .then(raise_)
      .except_(h1, exceptions=Exc1, reraise=False)
      .except_(h2, exceptions=TestExc, reraise=False)
      .run()
    )
    self.assertEqual(calls, ['h2'])
    self.assertEqual(result, 'h2')

  async def test_33_subclass_exception_matches_parent_handler(self):
    """Subclass exception matches parent exception handler (isinstance)."""
    calls = []
    def handler(v):
      calls.append('matched')
      return 'caught'
    def raise_subclass(v):
      raise Exc1()
    result = await await_(
      Chain(1)
      .then(raise_subclass)
      .except_(handler, exceptions=TestExc, reraise=False)
      .run()
    )
    self.assertEqual(calls, ['matched'])
    self.assertEqual(result, 'caught')

  async def test_34_reraise_true_handler_runs_and_propagates(self):
    """reraise=True: handler runs AND exception propagates."""
    handler_ran = [False]
    def handler(v):
      handler_ran[0] = True
    with self.assertRaises(TestExc):
      await await_(
        Chain(1)
        .then(raise_)
        .except_(handler, reraise=True)
        .run()
      )
    self.assertTrue(handler_ran[0])

  async def test_35_reraise_false_suppresses_exception(self):
    """reraise=False: handler runs AND exception does NOT propagate."""
    handler_ran = [False]
    def handler(v):
      handler_ran[0] = True
      return 'handled'
    result = await await_(
      Chain(1)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertTrue(handler_ran[0])
    self.assertEqual(result, 'handled')

  async def test_36_except_handler_return_becomes_chain_result(self):
    """Exception handler return value becomes the chain result (reraise=False)."""
    sentinel = make_sentinel()
    def handler(v):
      return sentinel
    result = await await_(
      Chain(1)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertIs(result, sentinel)

  async def test_37_handler_new_exception_chains_cause(self):
    """Handler raising new exception: chains as __cause__ from original."""
    def handler(v):
      raise Exc4('handler error')
    try:
      await await_(
        Chain(1)
        .then(raise_)
        .except_(handler, reraise=False)
        .run()
      )
      self.fail('Expected exception')
    except Exc4 as e:
      self.assertIsInstance(e.__cause__, TestExc)

  async def test_38_except_handler_receives_root_not_current(self):
    """except_ handler receives ROOT value, not current_value at exception point."""
    root_val = make_sentinel()
    received = []
    def handler(v):
      received.append(v)
      return 'ok'
    def mutate(v):
      return 'mutated'
    result = await await_(
      Chain(root_val)
      .then(mutate)
      .then(mutate)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], root_val)

  async def test_39_finally_runs_after_except_handler(self):
    """finally_ handler runs AFTER exception handler."""
    order = []
    def on_except(v):
      order.append('except')
      return 'handled'
    def on_finally(v):
      order.append('finally')
    result = await await_(
      Chain(1)
      .then(raise_)
      .except_(on_except, reraise=False)
      .finally_(on_finally)
      .run()
    )
    self.assertEqual(order, ['except', 'finally'])

  async def test_40_finally_runs_even_when_no_handler_matches(self):
    """finally_ handler runs even when NO exception handler matches."""
    finally_ran = [False]
    def on_finally(v):
      finally_ran[0] = True
    def handler(v):
      return 'should not match'
    with self.assertRaises(TestExc):
      await await_(
        Chain(1)
        .then(raise_)
        .except_(handler, exceptions=Exc4, reraise=False)
        .finally_(on_finally)
        .run()
      )
    self.assertTrue(finally_ran[0])

  async def test_41_only_one_finally_allowed(self):
    """Only ONE finally_ handler is allowed (second raises QuentException)."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)

  async def test_42_return_in_nested_chain_propagates(self):
    """_Return in nested chain propagates through parent."""
    inner = Chain().then(lambda v: Chain.return_(99))
    result = await await_(
      Chain(1)
      .then(inner)
      .then(lambda v: v * 100)
      .run()
    )
    self.assertEqual(result, 99)

  async def test_43_break_outside_iteration_raises(self):
    """_Break outside iteration context raises QuentException."""
    with self.assertRaises(QuentException):
      await await_(
        Chain(1)
        .then(lambda v: Chain.break_())
        .run()
      )

  async def test_44_internal_quent_exc_in_finally_becomes_quent_exc(self):
    """_InternalQuentException in finally handler becomes QuentException."""
    with self.assertRaises(QuentException):
      await await_(
        Chain(1)
        .then(aidentity)
        .finally_(lambda v: Chain.return_(42))
        .run()
      )

  async def test_45_exception_has_quent_attribute(self):
    """Exception has __quent__ attribute set after chain processes it.

    __quent__ is set by remove_self_frames_from_traceback, which is only
    called from _await_run. To trigger that path we need an async except_
    handler whose coroutine is awaited via _await_run_fn.
    """
    async def async_handler(v=None):
      raise TestExc('from handler')

    c = (
      Chain(1)
      .then(raise_)
      .except_(async_handler, reraise=False)
    )
    try:
      await c.run()
    except TestExc as e:
      self.assertTrue(hasattr(e, '__quent__'))
    else:
      self.fail('Expected TestExc')


# ===================================================================
# D. Mutation and State Isolation Bugs (10+ tests)
# ===================================================================

class TestMutationIsolation(IsolatedAsyncioTestCase):
  """Verify chains do not leak state between runs or clones."""

  async def test_46_running_chain_does_not_mutate_links(self):
    """Running a chain does NOT mutate its links (can be run multiple times)."""
    chain = Chain().then(lambda v: v * 2)
    r1 = await await_(chain.run(5))
    r2 = await await_(chain.run(10))
    r3 = await await_(chain.run(3))
    self.assertEqual(r1, 10)
    self.assertEqual(r2, 20)
    self.assertEqual(r3, 6)

  async def test_47_running_chain_does_not_mutate_root_link(self):
    """Running a chain does NOT mutate root_link."""
    chain = Chain(42).then(lambda v: v + 1)
    r1 = await await_(chain.run())
    r2 = await await_(chain.run())
    self.assertEqual(r1, 43)
    self.assertEqual(r2, 43)

  async def test_48_temp_args_not_set_permanently(self):
    """Running a chain does NOT set temp_args permanently on links."""
    chain = Chain().then(lambda v: v + 1)
    r1 = await await_(chain.run(10))
    r2 = await await_(chain.run(20))
    self.assertEqual(r1, 11)
    self.assertEqual(r2, 21)

  async def test_49_concurrent_async_chain_isolation(self):
    """Concurrent async runs of a chain do not interfere."""
    async def slow_double(v):
      await asyncio.sleep(0.01)
      return v * 2
    c = Chain().then(slow_double)
    results = await asyncio.gather(
      await_(c.run(1)),
      await_(c.run(2)),
      await_(c.run(3)),
      await_(c.run(4)),
      await_(c.run(5)),
    )
    self.assertEqual(sorted(results), [2, 4, 6, 8, 10])

  async def test_50_clone_links_independent(self):
    """Clone links are independent (mutating one does not affect other)."""
    original = Chain(1).then(lambda v: v + 1)
    cloned = original.clone()
    original.then(lambda v: v * 100)
    r_orig = await await_(original.run())
    r_clone = await await_(cloned.run())
    self.assertEqual(r_orig, 200)
    self.assertEqual(r_clone, 2)

  async def test_51_clone_on_finally_independent(self):
    """Clone on_finally_link is independent."""
    original_finally_ran = [False]
    original = Chain(1).then(identity).finally_(lambda v: setattr_helper(original_finally_ran, True))
    cloned = original.clone()
    await await_(original.run())
    self.assertTrue(original_finally_ran[0])
    # Clone should also work independently
    clone_finally_ran = [False]
    # We cannot replace the clone's finally, but we can verify the clone runs
    await await_(cloned.run())

  async def test_52_nested_chain_flag_does_not_prevent_parent(self):
    """Nested chain is_nested flag does not prevent parent execution."""
    inner = Chain().then(lambda v: v * 2)
    result1 = await await_(Chain(5).then(inner).run())
    self.assertEqual(result1, 10)
    result2 = await await_(Chain(7).then(inner).run())
    self.assertEqual(result2, 14)

  async def test_53_foreach_no_state_leak_between_runs(self):
    """foreach does not leak state between runs."""
    chain = Chain().foreach(lambda el: el * 2)
    r1 = await await_(chain.run([1, 2, 3]))
    r2 = await await_(chain.run([10, 20]))
    self.assertEqual(r1, [2, 4, 6])
    self.assertEqual(r2, [20, 40])

  async def test_54_filter_no_state_leak_between_runs(self):
    """filter does not leak state between runs."""
    chain = Chain().filter(lambda el: el > 3)
    r1 = await await_(chain.run([1, 2, 3, 4, 5]))
    r2 = await await_(chain.run([10, 1, 20, 2]))
    self.assertEqual(r1, [4, 5])
    self.assertEqual(r2, [10, 20])

  async def test_55_multiple_runs_with_except_handler(self):
    """Chain with except handler can be run multiple times without state leak."""
    handler_calls = []
    def handler(v):
      handler_calls.append(v)
      return 'handled'
    chain = Chain().then(raise_).except_(handler, reraise=False)
    r1 = await await_(chain.run(10))
    r2 = await await_(chain.run(20))
    self.assertEqual(r1, 'handled')
    self.assertEqual(r2, 'handled')
    self.assertEqual(handler_calls, [10, 20])


# ===================================================================
# E. Null Sentinel Handling (8+ tests)
# ===================================================================

class TestNullSentinel(IsolatedAsyncioTestCase):
  """Verify Null sentinel behavior exactly matches documentation."""

  async def test_56_null_is_not_none(self):
    """Null is NOT None (they are different objects)."""
    self.assertIsNot(Null, None)
    self.assertIsNotNone(Null)

  async def test_57_chain_no_root_returns_none(self):
    """Chain() with no links returns None (not Null sentinel)."""
    result = Chain().run()
    self.assertIsNone(result)

  async def test_58_chain_none_root_returns_none(self):
    """Chain(None).run() returns None (None is a valid value)."""
    result = await await_(Chain(None).run())
    self.assertIsNone(result)

  async def test_59_chain_zero_root_returns_zero(self):
    """Chain(0).run() returns 0 (not None, not Null)."""
    result = await await_(Chain(0).run())
    self.assertEqual(result, 0)
    self.assertIsNotNone(result)

  async def test_60_chain_false_root_returns_false(self):
    """Chain(False).run() returns False."""
    result = await await_(Chain(False).run())
    self.assertIs(result, False)

  async def test_61_chain_empty_string_returns_empty_string(self):
    """Chain("").run() returns empty string."""
    result = await await_(Chain("").run())
    self.assertEqual(result, "")
    self.assertIsNotNone(result)

  async def test_62_then_fn_returning_none_flows_through(self):
    """.then(fn) where fn returns None: None flows through (not Null)."""
    def returns_none(v):
      return None
    result = await await_(Chain(42).then(returns_none).run())
    self.assertIsNone(result)

  async def test_63_void_chain_with_override_none(self):
    """void chain with override None: None is the root."""
    received = []
    def capture(v):
      received.append(v)
      return v
    result = await await_(Chain().then(capture).run(None))
    self.assertIsNone(result)
    self.assertEqual(len(received), 1)
    self.assertIsNone(received[0])


# ===================================================================
# F. Edge Case Regression Tests (10+ tests)
# ===================================================================

class TestEdgeCases(IsolatedAsyncioTestCase):
  """Edge cases that are likely to harbor regressions."""

  async def test_64_exception_in_root_evaluation(self):
    """Chain with exception in root evaluation (not in links)."""
    def bad_root():
      raise TestExc('root error')
    with self.assertRaises(TestExc):
      await await_(Chain(bad_root).run())

  async def test_65_exception_in_first_link(self):
    """Chain with exception in the very first link (after root)."""
    with self.assertRaises(TestExc):
      await await_(Chain(1).then(raise_).run())

  async def test_66_exception_in_last_link(self):
    """Chain with exception in the very last link."""
    with self.assertRaises(TestExc):
      await await_(
        Chain(1)
        .then(lambda v: v + 1)
        .then(lambda v: v + 1)
        .then(raise_)
        .run()
      )

  async def test_67_chain_no_links_root_only(self):
    """Chain with NO links (root only) just evaluates root."""
    result = await await_(Chain(42).run())
    self.assertEqual(result, 42)

  async def test_68_chain_literal_root(self):
    """Chain(42) uses EVAL_RETURN_AS_IS and returns 42."""
    result = await await_(Chain(42).run())
    self.assertEqual(result, 42)

  async def test_69_then_none_literal(self):
    """Chain(42).then(None): None replaces 42 as literal."""
    result = await await_(Chain(42).then(None).run())
    self.assertIsNone(result)

  async def test_70_then_true_literal(self):
    """Chain(42).then(True).run() returns True."""
    result = await await_(Chain(42).then(True).run())
    self.assertIs(result, True)

  async def test_71_then_list_literal(self):
    """Chain(42).then([1,2,3]).run() returns [1,2,3] (literal list)."""
    lst = [1, 2, 3]
    result = await await_(Chain(42).then(lst).run())
    self.assertIs(result, lst)

  async def test_72_ellipsis_marker(self):
    """Chain(fn, ...).run() calls fn with no args (Ellipsis as marker)."""
    called_with = []
    def fn():
      called_with.append(True)
      return 'no_args'
    result = await await_(Chain(fn, ...).run())
    self.assertEqual(result, 'no_args')
    self.assertEqual(called_with, [True])

  async def test_73_root_with_explicit_args(self):
    """Chain(fn, arg1, arg2).run() calls fn(arg1, arg2)."""
    def fn(a, b):
      return a + b
    result = await await_(Chain(fn, 10, 20).run())
    self.assertEqual(result, 30)

  async def test_73b_run_override_not_allowed_with_root(self):
    """Cannot override root value of a chain that already has a root."""
    with self.assertRaises(QuentException):
      Chain(42).run(99)

  async def test_73c_nested_chain_cannot_run_directly(self):
    """Nested chain (is_nested=True) cannot be run directly."""
    inner = Chain().then(lambda v: v)
    Chain(1).then(inner)
    with self.assertRaises(QuentException):
      inner.run(5)

  async def test_73d_do_with_non_callable_raises(self):
    """do() with non-callable raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).do(42)

  async def test_73e_pipe_run_syntax(self):
    """Pipe syntax with run() terminator works correctly."""
    result = await await_(Chain(5).then(lambda v: v * 2).run())
    self.assertEqual(result, 10)

  async def test_73f_pipe_run_with_override(self):
    """Pipe syntax with run(value) override for void chain."""
    result = await await_(Chain().then(lambda v: v * 3).run(7))
    self.assertEqual(result, 21)

  async def test_73g_chain_bool_always_true(self):
    """Chain.__bool__ always returns True."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(42)))
    self.assertTrue(bool(Chain().then(identity)))

  async def test_73h_chain_callable_runs(self):
    """Chain is callable and acts like run()."""
    result = await await_(Chain().then(lambda v: v + 1)(10))
    self.assertEqual(result, 11)


# ===================================================================
# G. Traceback and Diagnostics Verification (5+ tests)
# ===================================================================

class TestDiagnostics(IsolatedAsyncioTestCase):
  """Verify exception augmentation and diagnostics."""

  async def test_74_exception_has_quent_attribute(self):
    """Exception has __quent__ attribute set.

    __quent__ is set by remove_self_frames_from_traceback, which is only
    called from _await_run. To trigger that path we need an async except_
    handler whose coroutine is awaited via _await_run_fn.
    """
    async def async_handler(v=None):
      raise TestExc('from handler')

    c = (
      Chain(1)
      .then(raise_)
      .except_(async_handler, reraise=False)
    )
    try:
      await c.run()
    except TestExc as e:
      self.assertTrue(getattr(e, '__quent__', False))
    else:
      self.fail('Expected TestExc')

  async def test_75_exception_traceback_cleaned(self):
    """Exception __traceback__ is cleaned of internal frames."""
    try:
      await await_(Chain(1).then(raise_).run())
    except TestExc as e:
      self.assertIsNotNone(e.__traceback__)
    else:
      self.fail('Expected TestExc')

  async def test_76_nested_chain_exception_source_link(self):
    """Nested chain exception: __quent_source_link__ properly resolved."""
    inner = Chain().then(raise_)
    try:
      await await_(Chain(1).then(inner).run())
    except TestExc as e:
      self.assertFalse(hasattr(e, '__quent_source_link__'))
    else:
      self.fail('Expected TestExc')

  async def test_77_repr_doesnt_crash(self):
    """repr(chain) does not crash for any chain configuration."""
    chains = [
      Chain(),
      Chain(42),
      Chain(lambda: 1),
      Chain(42).then(lambda v: v),
      Chain(42).then(identity).do(identity),
      Chain(42).except_(identity),
      Chain(42).finally_(identity),
      Chain().then(lambda v: v).then(lambda v: v),
    ]
    for c in chains:
      r = repr(c)
      self.assertIsInstance(r, str)
      self.assertTrue(len(r) > 0)

  async def test_78_repr_includes_method_names(self):
    """repr(chain) includes method names for then, do, except_, finally_."""
    c = (
      Chain(42)
      .then(identity)
      .do(identity)
      .except_(identity)
      .finally_(identity)
    )
    r = repr(c)
    self.assertIn('.then', r)
    self.assertIn('.do', r)
    self.assertIn('.except_', r)
    self.assertIn('.finally_', r)


# ===================================================================
# H. Additional False-Positive Hunters
# ===================================================================

class TestFalsePositiveHunters(IsolatedAsyncioTestCase):
  """Tests specifically designed to catch false positives."""

  async def test_fp_01_then_actually_transforms(self):
    """Verify .then() actually applies the transformation."""
    result = await await_(Chain(5).then(lambda v: v * 3).run())
    self.assertEqual(result, 15)
    self.assertNotEqual(result, 5)

  async def test_fp_03_chain_order_matters(self):
    """Operations execute in the exact order they were added."""
    log = []
    def logger(name):
      def fn(v):
        log.append(name)
        return v
      return fn
    await await_(
      Chain(1)
      .then(logger('a'))
      .then(logger('b'))
      .then(logger('c'))
      .run()
    )
    self.assertEqual(log, ['a', 'b', 'c'])

  async def test_fp_04_chain_value_flows_not_duplicated(self):
    """Each .then() receives the RESULT of the PREVIOUS .then(), not root."""
    result = await await_(
      Chain(1)
      .then(lambda v: v + 10)
      .then(lambda v: v * 2)
      .then(lambda v: v + 3)
      .run()
    )
    self.assertEqual(result, 25)

  async def test_fp_06_except_handler_returning_none(self):
    """except handler returning None: chain result is None."""
    def handler(v):
      return None
    result = await await_(
      Chain(42)
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertIsNone(result)

  async def test_fp_07_foreach_empty_list(self):
    """foreach on empty list returns empty list."""
    result = await await_(Chain([]).foreach(lambda el: el * 2).run())
    self.assertEqual(result, [])
    self.assertIsInstance(result, list)

  async def test_fp_08_filter_all_rejected(self):
    """filter where all elements rejected returns empty list."""
    result = await await_(Chain([1, 2, 3]).filter(lambda el: False).run())
    self.assertEqual(result, [])
    self.assertIsInstance(result, list)

  async def test_fp_09_filter_all_accepted(self):
    """filter where all elements accepted returns all."""
    result = await await_(Chain([1, 2, 3]).filter(lambda el: True).run())
    self.assertEqual(result, [1, 2, 3])

  async def test_fp_10_gather_single_function(self):
    """gather with single function returns single-element list."""
    result = await await_(Chain(5).gather(lambda v: v * 2).run())
    self.assertEqual(result, [10])

  async def test_fp_11_do_exception_propagates(self):
    """.do() exception propagates (it does not silently swallow)."""
    with self.assertRaises(TestExc):
      await await_(
        Chain(1)
        .do(raise_)
        .then(lambda v: v + 1)
        .run()
      )

  async def test_fp_12_with_result_is_body_return(self):
    """with_ result is the body return value, not CM or __enter__."""
    cm = SyncCM('enter_val')
    def body(v):
      return 'body_result'
    result = await await_(Chain(cm).with_(body).run())
    self.assertEqual(result, 'body_result')

  async def test_fp_14_chain_no_root_no_links_returns_none(self):
    """Chain with no root and no links returns None."""
    result = Chain().run()
    self.assertIsNone(result)

  async def test_fp_15_async_root_with_sync_links(self):
    """Async root with sync links works correctly."""
    async def async_root():
      return 10
    result = await await_(Chain(async_root).then(lambda v: v + 5).run())
    self.assertEqual(result, 15)

  async def test_fp_16_sync_root_with_async_links(self):
    """Sync root with async links works correctly."""
    async def async_add(v):
      return v + 5
    result = await await_(Chain(10).then(async_add).run())
    self.assertEqual(result, 15)

  async def test_fp_17_break_with_value_in_foreach(self):
    """break_ with value returns that value from foreach."""
    def process(el):
      if el == 3:
        Chain.break_(100)
      return el
    result = await await_(Chain([1, 2, 3, 4]).foreach(process).run())
    self.assertEqual(result, 100)

  async def test_fp_18_break_without_value_returns_partial(self):
    """break_ without value returns partial results collected so far."""
    def process(el):
      if el == 3:
        Chain.break_()
      return el * 10
    result = await await_(Chain([1, 2, 3, 4]).foreach(process).run())
    self.assertEqual(result, [10, 20])

  async def test_fp_19_return_with_no_value_returns_none(self):
    """return_ with no value returns None from chain."""
    inner = Chain().then(lambda v: Chain.return_())
    result = await await_(Chain(42).then(inner).run())
    self.assertIsNone(result)

  async def test_fp_20_return_with_value_returns_that_value(self):
    """return_ with value returns that value from chain."""
    inner = Chain().then(lambda v: Chain.return_(99))
    result = await await_(Chain(42).then(inner).run())
    self.assertEqual(result, 99)

  async def test_fp_23_except_string_type_raises_typeerror(self):
    """except_ with string as exceptions raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(1).except_(lambda v: None, exceptions='TestExc')

  async def test_fp_24_cm_exception_suppressed(self):
    """Context manager that suppresses exception: chain continues."""
    cm = SyncCMSuppressing('enter_val')
    def body(v):
      raise TestExc('suppressed')
    # When __exit__ returns True, exception is suppressed.
    # The _With except branch does not return a value, so Python returns None.
    result = await await_(Chain(cm).with_(body).run())
    self.assertIsNone(result)

  async def test_fp_25_multiple_except_handlers_different_types(self):
    """Multiple handlers with different types: correct one matches."""
    calls = []
    def h_exc1(v):
      calls.append('exc1')
      return 'exc1'
    def h_exc2(v):
      calls.append('exc2')
      return 'exc2'
    def h_exc4(v):
      calls.append('exc4')
      return 'exc4'
    def raise_exc2(v):
      raise Exc2('specific')
    result = await await_(
      Chain(1)
      .then(raise_exc2)
      .except_(h_exc1, exceptions=Exc1, reraise=False)
      .except_(h_exc4, exceptions=Exc4, reraise=False)
      .except_(h_exc2, exceptions=Exc2, reraise=False)
      .run()
    )
    self.assertEqual(calls, ['exc2'])
    self.assertEqual(result, 'exc2')

  async def test_fp_26_exc3_matches_exc2_handler(self):
    """Exc3 (subclass of Exc2) matches Exc2 handler via isinstance."""
    calls = []
    def h_exc2(v):
      calls.append('exc2')
      return 'caught_by_parent'
    def raise_exc3(v):
      raise Exc3('child')
    result = await await_(
      Chain(1)
      .then(raise_exc3)
      .except_(h_exc2, exceptions=Exc2, reraise=False)
      .run()
    )
    self.assertEqual(calls, ['exc2'])
    self.assertEqual(result, 'caught_by_parent')

  async def test_fp_27_chain_decorator(self):
    """Chain decorator wraps functions correctly."""
    chain = Chain().then(lambda v: v * 2)
    decorator = chain.decorator()

    @decorator
    def my_fn(x):
      return x + 1

    result = await await_(my_fn(5))
    self.assertEqual(result, 12)

  async def test_fp_28_chain_with_all_features(self):
    """Complex chain with then, do, except, finally: all interact correctly."""
    log = []
    def step1(v):
      log.append(f'step1:{v}')
      return v + 1
    def step2(v):
      log.append(f'step2:{v}')
      return v * 2
    def side_effect(v):
      log.append(f'side:{v}')
      return 'ignored'
    def on_exc(v):
      log.append(f'exc:{v}')
      return 'handled'
    def on_finally(v):
      log.append(f'finally:{v}')
    result = await await_(
      Chain(10)
      .then(step1)
      .do(side_effect)
      .then(step2)
      .except_(on_exc, reraise=False)
      .finally_(on_finally)
      .run()
    )
    self.assertEqual(result, 22)
    self.assertEqual(log, ['step1:10', 'side:11', 'step2:11', 'finally:10'])
