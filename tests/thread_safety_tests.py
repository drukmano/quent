# SPDX-License-Identifier: MIT
"""Thread safety tests for concurrent execution of shared Chain instances.

Tests validate SPEC §3 guarantee: a fully constructed chain is safe to execute
concurrently from multiple threads, because execution uses only function-local
state and never mutates the list structure.

Also covers SPEC §10.1 (clone independence) and SPEC §10.2 (decorator thread safety),
and SPEC §14.4 (on_step under concurrency).
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest import TestCase

from quent import Chain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THREAD_COUNT = 30
STRESS_THREAD_COUNT = 50
ITERATIONS = 100


# ---------------------------------------------------------------------------
# Helper: run chain from N threads, collect results
# ---------------------------------------------------------------------------


def _run_concurrent(chain: Chain, n: int, run_value=None) -> list:
  """Execute chain.run() from n threads simultaneously, return all results."""
  barrier = threading.Barrier(n)
  results = [None] * n
  errors: list[Exception] = []
  lock = threading.Lock()

  def _worker(idx: int) -> None:
    barrier.wait()  # synchronize start to maximize contention
    try:
      if run_value is None:
        results[idx] = chain.run()
      else:
        results[idx] = chain.run(run_value)
    except Exception as exc:
      with lock:
        errors.append(exc)

  threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  if errors:
    raise errors[0]
  return results


def _run_concurrent_varying(chain: Chain, values: list) -> list:
  """Execute chain.run(v) for each v in values, one thread per value."""
  n = len(values)
  barrier = threading.Barrier(n)
  results = [None] * n
  errors: list[Exception] = []
  lock = threading.Lock()

  def _worker(idx: int, value) -> None:
    barrier.wait()
    try:
      results[idx] = chain.run(value)
    except Exception as exc:
      with lock:
        errors.append(exc)

  threads = [threading.Thread(target=_worker, args=(i, values[i])) for i in range(n)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

  if errors:
    raise errors[0]
  return results


# ---------------------------------------------------------------------------
# §3: Basic concurrent run — then-chain
# ---------------------------------------------------------------------------


class BasicConcurrentRunTest(TestCase):
  """SPEC §3: A fully constructed chain is safe to execute concurrently from multiple threads."""

  def test_simple_then_chain_concurrent(self):
    """Shared then-chain produces same result from all threads simultaneously."""
    chain = Chain(10).then(lambda x: x + 1).then(lambda x: x * 2)
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [22] * THREAD_COUNT)

  def test_multi_step_then_chain_concurrent(self):
    """Multi-step pipeline produces consistent results under concurrent execution."""
    chain = Chain(1).then(lambda x: x + 1).then(lambda x: x * 3).then(lambda x: x - 2).then(lambda x: x + 10)
    # 1 -> 2 -> 6 -> 4 -> 14
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [14] * THREAD_COUNT)

  def test_run_value_shared_chain(self):
    """chain.run(v) from multiple threads: same chain, same value, same result."""
    chain = Chain().then(lambda x: x * 2).then(lambda x: x + 1)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=5)
    # 5 -> 10 -> 11
    self.assertEqual(results, [11] * THREAD_COUNT)


# ---------------------------------------------------------------------------
# §3: Various chain shapes — concurrent correctness
# ---------------------------------------------------------------------------


class ChainShapeConcurrentTest(TestCase):
  """SPEC §3: All chain operation types are safe under concurrent execution."""

  def test_foreach_concurrent_outer(self):
    """Chain with foreach() is safe to run from multiple threads concurrently."""
    chain = Chain([1, 2, 3, 4]).foreach(lambda x: x * 2)
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [[2, 4, 6, 8]] * THREAD_COUNT)

  def test_foreach_with_inner_concurrency(self):
    """foreach(concurrency=N) inside concurrent thread execution stays correct."""
    chain = Chain([1, 2, 3, 4]).foreach(lambda x: x + 10, concurrency=4)
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [[11, 12, 13, 14]] * THREAD_COUNT)

  def test_foreach_do_concurrent_outer(self):
    """Chain with foreach_do() is safe to run from multiple threads concurrently."""
    chain = Chain([10, 20, 30]).foreach_do(lambda x: x + 1)
    # foreach_do keeps original elements
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [[10, 20, 30]] * THREAD_COUNT)

  def test_foreach_do_with_inner_concurrency(self):
    """foreach_do(concurrency=N) inside concurrent thread execution stays correct."""
    chain = Chain([1, 2, 3]).foreach_do(lambda x: x * 100, concurrency=3)
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [[1, 2, 3]] * THREAD_COUNT)

  def test_gather_concurrent_outer(self):
    """Chain with gather() is safe to run from multiple threads concurrently."""
    chain = Chain(5).gather(lambda x: x + 1, lambda x: x * 2, lambda x: x - 1)
    results = _run_concurrent(chain, THREAD_COUNT)
    self.assertEqual(results, [(6, 10, 4)] * THREAD_COUNT)

  def test_nested_chain_concurrent(self):
    """Chain with a nested Chain step is safe under concurrent execution."""
    inner = Chain().then(lambda x: x + 100)
    outer = Chain(7).then(inner)
    results = _run_concurrent(outer, THREAD_COUNT)
    self.assertEqual(results, [107] * THREAD_COUNT)

  def test_except_handler_concurrent(self):
    """Chain with except_() handler executes correctly from multiple threads."""
    chain = Chain().then(lambda x: 1 / x).except_(lambda exc: -1)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=0)
    # ZeroDivisionError -> handler returns -1
    self.assertEqual(results, [-1] * THREAD_COUNT)

  def test_except_handler_no_error_concurrent(self):
    """Chain with except_() that never fires is safe under concurrent execution."""
    chain = Chain().then(lambda x: x * 3).except_(lambda exc: -999)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=4)
    self.assertEqual(results, [12] * THREAD_COUNT)

  def test_finally_handler_concurrent(self):
    """Chain with finally_() handler executes correctly from multiple threads."""
    counter = [0]
    lock = threading.Lock()

    def _cleanup(rv):
      with lock:
        counter[0] += 1

    chain = Chain().then(lambda x: x + 1).finally_(_cleanup)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=10)
    self.assertEqual(results, [11] * THREAD_COUNT)
    self.assertEqual(counter[0], THREAD_COUNT)

  def test_if_conditional_concurrent(self):
    """Chain with if_().then() conditional is safe under concurrent execution."""
    chain = Chain().if_(lambda x: x > 0).then(lambda x: x * 10).else_(lambda x: -x)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=5)
    self.assertEqual(results, [50] * THREAD_COUNT)

  def test_if_else_branch_concurrent(self):
    """Chain with if_().then().else_() else branch is safe under concurrent execution."""
    chain = Chain().if_(lambda x: x > 0).then(lambda x: x * 10).else_(lambda x: -x)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=-3)
    self.assertEqual(results, [3] * THREAD_COUNT)

  def test_with_context_manager_concurrent(self):
    """Chain with with_() context manager step is safe under concurrent execution."""

    @contextlib.contextmanager
    def _make_ctx(value):
      yield value * 2

    chain = Chain().then(_make_ctx).with_(lambda ctx: ctx + 1)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=5)
    # value=5 -> ctx=10 -> ctx+1=11
    self.assertEqual(results, [11] * THREAD_COUNT)

  def test_do_side_effect_concurrent(self):
    """Chain with do() side-effect step is safe under concurrent execution."""
    side_effects = []
    lock = threading.Lock()

    def _record(x):
      with lock:
        side_effects.append(x)

    chain = Chain().do(_record).then(lambda x: x + 1)
    results = _run_concurrent(chain, THREAD_COUNT, run_value=7)
    self.assertEqual(results, [8] * THREAD_COUNT)
    self.assertEqual(len(side_effects), THREAD_COUNT)
    self.assertTrue(all(v == 7 for v in side_effects))


# ---------------------------------------------------------------------------
# §3: Run-time value variation — per-thread input isolation
# ---------------------------------------------------------------------------


class PerThreadValueIsolationTest(TestCase):
  """SPEC §3: function-local state ensures each thread's value is independent."""

  def test_different_inputs_produce_different_outputs(self):
    """Each thread passes a different value; each gets the correct result for its input."""
    chain = Chain().then(lambda x: x * x)
    values = list(range(THREAD_COUNT))
    results = _run_concurrent_varying(chain, values)
    expected = [v * v for v in values]
    self.assertEqual(results, expected)

  def test_multi_step_per_thread_isolation(self):
    """Multi-step chain with per-thread input produces isolated correct results."""
    chain = Chain().then(lambda x: x + 1).then(lambda x: x * 2).then(str)
    values = list(range(THREAD_COUNT))
    results = _run_concurrent_varying(chain, values)
    expected = [str((v + 1) * 2) for v in values]
    self.assertEqual(results, expected)

  def test_foreach_per_thread_isolation(self):
    """foreach() with per-thread input values produces isolated correct results."""
    chain = Chain().foreach(lambda x: x + 100)
    values = [[i, i + 1, i + 2] for i in range(THREAD_COUNT)]
    results = _run_concurrent_varying(chain, values)
    expected = [[i + 100, i + 101, i + 102] for i in range(THREAD_COUNT)]
    self.assertEqual(results, expected)

  def test_gather_per_thread_isolation(self):
    """gather() with per-thread input values produces isolated correct results."""
    chain = Chain().gather(lambda x: x + 1, lambda x: x * 2)
    values = list(range(THREAD_COUNT))
    results = _run_concurrent_varying(chain, values)
    expected = [(v + 1, v * 2) for v in values]
    self.assertEqual(results, expected)

  def test_conditional_per_thread_isolation(self):
    """Conditional chain routes each thread correctly based on its own input."""
    chain = Chain().if_(lambda x: x % 2 == 0).then(lambda x: x * 10).else_(lambda x: x * 100)
    values = list(range(THREAD_COUNT))
    results = _run_concurrent_varying(chain, values)
    expected = [v * 10 if v % 2 == 0 else v * 100 for v in values]
    self.assertEqual(results, expected)

  def test_except_per_thread_isolation(self):
    """except_() handler: threads that error get handler result, others get normal result."""
    # Chain that divides by input: 0 -> error -> handler, nonzero -> result
    chain = Chain().then(lambda x: 100 // x).except_(lambda exc: -1)
    # Use values 0..N-1; thread with value 0 gets -1, rest get 100//v
    values = list(range(THREAD_COUNT))
    results = _run_concurrent_varying(chain, values)
    expected = [-1 if v == 0 else 100 // v for v in values]
    self.assertEqual(results, expected)


# ---------------------------------------------------------------------------
# §14.4: on_step callback under concurrency
# ---------------------------------------------------------------------------


class OnStepConcurrentTest(TestCase):
  """SPEC §14.4: on_step is class-level; set before concurrent execution begins."""

  def setUp(self):
    self._log: list[tuple] = []
    self._lock = threading.Lock()
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    with self._lock:
      self._log.append((step_name, result, elapsed_ns))

  def test_on_step_fires_for_all_threads(self):
    """on_step callback fires for every step from every thread; no callbacks lost."""
    chain = Chain(5).then(lambda x: x + 1)
    # Each chain.run() fires 2 callbacks: 'root' and 'then'
    _run_concurrent(chain, THREAD_COUNT)
    with self._lock:
      total = len(self._log)
    self.assertEqual(total, THREAD_COUNT * 2)

  def test_on_step_thread_safe_collection(self):
    """Thread-safe collection via lock captures callbacks without corruption."""
    chain = Chain(1).then(lambda x: x + 1).then(lambda x: x * 3)
    # 3 steps per run: root, then, then
    _run_concurrent(chain, THREAD_COUNT)
    with self._lock:
      steps = [entry[0] for entry in self._log]
    root_count = steps.count('root')
    then_count = steps.count('then')
    self.assertEqual(root_count, THREAD_COUNT)
    self.assertEqual(then_count, THREAD_COUNT * 2)

  def test_on_step_results_are_correct(self):
    """on_step results recorded under concurrency match expected computed values."""
    chain = Chain(10).then(lambda x: x + 5)
    # root result = 10, then result = 15
    _run_concurrent(chain, THREAD_COUNT)
    with self._lock:
      root_results = {entry[1] for entry in self._log if entry[0] == 'root'}
      then_results = {entry[1] for entry in self._log if entry[0] == 'then'}
    self.assertEqual(root_results, {10})
    self.assertEqual(then_results, {15})

  def test_on_step_elapsed_ns_nonnegative(self):
    """elapsed_ns values recorded by concurrent on_step callbacks are non-negative ints."""
    chain = Chain(3).then(lambda x: x)
    _run_concurrent(chain, THREAD_COUNT)
    with self._lock:
      for _step, _result, elapsed_ns in self._log:
        self.assertIsInstance(elapsed_ns, int)
        self.assertGreaterEqual(elapsed_ns, 0)


# ---------------------------------------------------------------------------
# §10.1: Clone independence under concurrency
# ---------------------------------------------------------------------------


class CloneIndependenceConcurrentTest(TestCase):
  """SPEC §10.1: clone() produces an independent linked list; concurrent runs don't cross-contaminate."""

  def test_original_and_clone_run_concurrently(self):
    """Original and clone can execute simultaneously from different threads without interference."""
    original = Chain(100).then(lambda x: x + 1)
    clone = original.clone()

    original_results = []
    clone_results = []
    orig_errors = []
    clone_errors = []

    def _run_original():
      try:
        original_results.append(original.run())
      except Exception as e:
        orig_errors.append(e)

    def _run_clone():
      try:
        clone_results.append(clone.run())
      except Exception as e:
        clone_errors.append(e)

    threads = [threading.Thread(target=_run_original) for _ in range(THREAD_COUNT)] + [
      threading.Thread(target=_run_clone) for _ in range(THREAD_COUNT)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(orig_errors)
    self.assertFalse(clone_errors)
    self.assertEqual(original_results, [101] * THREAD_COUNT)
    self.assertEqual(clone_results, [101] * THREAD_COUNT)

  def test_clone_independence_different_steps(self):
    """Clone extended with extra steps produces different result from original; both correct under concurrency."""
    base = Chain(10).then(lambda x: x * 2)
    extended = base.clone().then(lambda x: x + 100)

    base_results = _run_concurrent(base, THREAD_COUNT)
    extended_results = _run_concurrent(extended, THREAD_COUNT)

    self.assertEqual(base_results, [20] * THREAD_COUNT)
    self.assertEqual(extended_results, [120] * THREAD_COUNT)

  def test_multiple_clones_concurrent(self):
    """Multiple independent clones all run correctly under concurrent execution."""
    original = Chain(5).then(lambda x: x + 1)
    clones = [original.clone() for _ in range(5)]

    all_results = []
    lock = threading.Lock()
    errors = []

    def _run_clone(c):
      try:
        r = c.run()
        with lock:
          all_results.append(r)
      except Exception as e:
        with lock:
          errors.append(e)

    threads = [threading.Thread(target=_run_clone, args=(c,)) for c in clones for _ in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors)
    self.assertEqual(all_results, [6] * 50)

  def test_clone_nested_chain_independence(self):
    """Cloned chains with nested Chain steps maintain independent execution."""
    inner = Chain().then(lambda x: x + 50)
    original = Chain(1).then(inner)
    clone = original.clone()

    orig_results = _run_concurrent(original, THREAD_COUNT)
    clone_results = _run_concurrent(clone, THREAD_COUNT)

    self.assertEqual(orig_results, [51] * THREAD_COUNT)
    self.assertEqual(clone_results, [51] * THREAD_COUNT)


# ---------------------------------------------------------------------------
# §10.2: Decorator thread safety
# ---------------------------------------------------------------------------


class DecoratorConcurrentTest(TestCase):
  """SPEC §10.2: decorated function is callable from many threads concurrently."""

  def test_decorator_concurrent_calls(self):
    """Decorated function called from many threads produces correct results for each."""
    process = Chain().then(lambda x: x * 3).then(lambda x: x + 1).decorator()

    @process
    def compute(n: int) -> int:
      return n

    barrier = threading.Barrier(THREAD_COUNT)
    results = [None] * THREAD_COUNT
    errors = []
    lock = threading.Lock()

    def _worker(idx):
      barrier.wait()
      try:
        results[idx] = compute(idx)
      except Exception as e:
        with lock:
          errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors)
    # compute(n): n -> n*3 -> n*3+1
    expected = [i * 3 + 1 for i in range(THREAD_COUNT)]
    self.assertEqual(results, expected)

  def test_decorator_same_instance_concurrent(self):
    """Single decorated function instance handles concurrent calls without state corruption."""
    add_ten = Chain().then(lambda x: x + 10).decorator()

    @add_ten
    def get_value(n: int) -> int:
      return n

    values = list(range(THREAD_COUNT))
    results = [None] * THREAD_COUNT
    barrier = threading.Barrier(THREAD_COUNT)
    errors = []
    lock = threading.Lock()

    def _worker(idx):
      barrier.wait()
      try:
        results[idx] = get_value(values[idx])
      except Exception as e:
        with lock:
          errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors)
    expected = [v + 10 for v in values]
    self.assertEqual(results, expected)

  def test_decorator_error_handling_concurrent(self):
    """Decorated function with error in chain handles exceptions per-thread independently."""
    handle_div = Chain().then(lambda x: 100 // x).except_(lambda exc: -1).decorator()

    @handle_div
    def reciprocal(n: int) -> int:
      return n

    values = list(range(THREAD_COUNT))
    results = [None] * THREAD_COUNT
    barrier = threading.Barrier(THREAD_COUNT)
    errors = []
    lock = threading.Lock()

    def _worker(idx):
      barrier.wait()
      try:
        results[idx] = reciprocal(values[idx])
      except Exception as e:
        with lock:
          errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors)
    # reciprocal(0) -> ZeroDivisionError -> -1; rest -> 100 // v
    expected = [-1 if v == 0 else 100 // v for v in values]
    self.assertEqual(results, expected)


# ---------------------------------------------------------------------------
# Stress tests — higher thread counts and repeated iterations
# ---------------------------------------------------------------------------


class StressConcurrentTest(TestCase):
  """Stress tests: high thread counts and repeated iterations to surface races."""

  def test_stress_simple_chain(self):
    """STRESS: simple then-chain under 50 threads x 100 iterations produces no errors."""
    chain = Chain().then(lambda x: x + 1).then(lambda x: x * 2)
    errors = []
    lock = threading.Lock()

    def _worker():
      for _ in range(ITERATIONS):
        try:
          result = chain.run(5)
          if result != 12:
            with lock:
              errors.append(AssertionError(f'expected 12, got {result}'))
        except Exception as e:
          with lock:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(STRESS_THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors, f'Stress test failures: {errors[:5]}')

  def test_stress_foreach_chain(self):
    """STRESS: foreach() chain under 50 threads x 100 iterations produces no errors."""
    chain = Chain().foreach(lambda x: x * 2)
    errors = []
    lock = threading.Lock()

    def _worker():
      for _ in range(ITERATIONS):
        try:
          result = chain.run([1, 2, 3])
          if result != [2, 4, 6]:
            with lock:
              errors.append(AssertionError(f'expected [2,4,6], got {result}'))
        except Exception as e:
          with lock:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(STRESS_THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors, f'Stress test failures: {errors[:5]}')

  def test_stress_gather_chain(self):
    """STRESS: gather() chain under 50 threads x 100 iterations produces no errors."""
    chain = Chain().gather(lambda x: x + 1, lambda x: x - 1)
    errors = []
    lock = threading.Lock()

    def _worker():
      for _ in range(ITERATIONS):
        try:
          result = chain.run(10)
          if result != (11, 9):
            with lock:
              errors.append(AssertionError(f'expected (11, 9), got {result}'))
        except Exception as e:
          with lock:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(STRESS_THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors, f'Stress test failures: {errors[:5]}')

  def test_stress_except_chain(self):
    """STRESS: except_() chain under 50 threads x 100 iterations produces no errors."""

    def _safe_div(x):
      if x == 0:
        raise ValueError('zero')
      return 1 // x

    chain = Chain().then(_safe_div).except_(lambda exc: -1)
    errors = []
    lock = threading.Lock()

    def _worker():
      for _ in range(ITERATIONS):
        try:
          r1 = chain.run(2)
          r2 = chain.run(0)
          if r1 != 0 or r2 != -1:
            with lock:
              errors.append(AssertionError(f'expected 0/-1, got {r1}/{r2}'))
        except Exception as e:
          with lock:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(STRESS_THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors, f'Stress test failures: {errors[:5]}')

  def test_stress_nested_chain(self):
    """STRESS: nested Chain steps under 50 threads x 100 iterations produce no errors."""
    inner = Chain().then(lambda x: x + 10)
    outer = Chain().then(inner).then(lambda x: x * 2)
    errors = []
    lock = threading.Lock()

    def _worker():
      for _ in range(ITERATIONS):
        try:
          result = outer.run(5)
          if result != 30:
            with lock:
              errors.append(AssertionError(f'expected 30, got {result}'))
        except Exception as e:
          with lock:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(STRESS_THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors, f'Stress test failures: {errors[:5]}')

  def test_stress_conditional_chain(self):
    """STRESS: if_/else_ conditional chain under 50 threads x 100 iterations produces no errors."""
    chain = Chain().if_(lambda x: x > 0).then(lambda x: x * 2).else_(lambda x: 0)
    errors = []
    lock = threading.Lock()

    def _worker():
      for _ in range(ITERATIONS):
        try:
          r_pos = chain.run(5)
          r_neg = chain.run(-3)
          if r_pos != 10 or r_neg != 0:
            with lock:
              errors.append(AssertionError(f'expected 10/0, got {r_pos}/{r_neg}'))
        except Exception as e:
          with lock:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(STRESS_THREAD_COUNT)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertFalse(errors, f'Stress test failures: {errors[:5]}')

  def test_stress_threadpoolexecutor(self):
    """STRESS: chain.run() via ThreadPoolExecutor.submit (no Barrier) under high load."""
    chain = Chain().then(lambda x: x + 1).then(lambda x: x * 3)
    futures = []
    with ThreadPoolExecutor(max_workers=STRESS_THREAD_COUNT) as pool:
      for i in range(STRESS_THREAD_COUNT * ITERATIONS):
        futures.append(pool.submit(chain.run, i % 10))

    errors = []
    for fut in as_completed(futures):
      try:
        fut.result()
      except Exception as e:
        errors.append(e)

    self.assertFalse(errors, f'ThreadPoolExecutor stress failures: {errors[:5]}')
