"""
Testing quent pipelines with unittest.

This recipe demonstrates how to unit test quent pipelines using unittest and
unittest.mock. It covers:

  - Testing sync pipeline happy paths and error paths
  - Testing async pipelines with AsyncMock
  - Verifying .do() side-effect behaviour (callable called, pipeline value unchanged)
  - Testing .except_() -- handler's return value replaces pipeline result
  - Testing .finally_() -- runs on success and failure, result discarded
  - Cloning pipelines to verify execution independence
  - Using .as_decorator() to wrap functions with pipeline behaviour
  - Using on_step for instrumentation-based testing
  - Testing control flow (Q.return_, Q.break_)
  - Testing the bridge contract (same pipeline, sync vs async callables)

Key testing insights:

1. Inject mock callables directly into the pipeline rather than patching
   module-level names with 'patch'. This keeps tests explicit and avoids
   fragile string targets.

2. Always use MagicMock(spec=fn) or create_autospec(fn) when passing mocks
   into pipelines. Plain MagicMock() has a _quent_is_q attribute auto-created
   by MagicMock's attribute machinery, which causes quent to treat the mock as
   a nested pipeline (duck-typing: getattr(v, '_quent_is_q', False)). The
   spec= parameter restricts attribute access to the real function's interface.

3. Pipeline input convention: pass the initial value to .run(v), not as the
   pipeline root. Use Q().then(step1).then(step2).run(input) so that
   .run(input) threads the input into the first step as its argument.

Run with:
    python -m unittest examples.testing_chains
    # or:
    python examples/testing_chains.py
"""

from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, create_autospec

from quent import Q, QuentException

# ---------------------------------------------------------------------------
# Application under test
# ---------------------------------------------------------------------------
# A minimal user-management pipeline. Each function is a plain callable so
# tests can inject mocks for any step without patching module-level names.


def fetch_user(user_id: int) -> dict:
  """Simulate fetching a user record from a database (will be mocked)."""
  return {'id': user_id, 'name': 'Alice', 'email': 'alice@example.com'}


def validate_user(user: dict) -> dict:
  """Check required fields; raise ValueError if the record is incomplete."""
  required = ('id', 'name', 'email')
  missing = [f for f in required if not user.get(f)]
  if missing:
    raise ValueError(f'missing fields: {missing}')
  return user


def enrich_user(user: dict) -> dict:
  """Add computed fields to a validated user record."""
  return {**user, 'display_name': user['name'].upper(), 'active': True}


def save_user(user: dict) -> dict:
  """Simulate persisting a user record (will be mocked)."""
  return user


def build_user_pipeline() -> Q:
  """Return a reusable pipeline: fetch -> validate -> enrich -> save, with error handler.

  The value passed to .run(user_id) flows into fetch_user as its first argument.
  """

  def handle_validation_error(ei) -> dict:
    # except_ default calling convention: fn(QuentExcInfo(exc, root_value)).
    # ei.root_value is the initial value passed to .run() -- the user_id here.
    return {'error': str(ei.exc), 'user_id': ei.root_value}

  return (
    Q()
    .then(fetch_user)
    .then(validate_user)
    .then(enrich_user)
    .then(save_user)
    .except_(handle_validation_error, exceptions=ValueError)
  )


async def async_fetch_user(user_id: int) -> dict:
  """Async version of fetch_user (will be mocked)."""
  return {'id': user_id, 'name': 'Alice', 'email': 'alice@example.com'}


async def async_save_user(user: dict) -> dict:
  """Async version of save_user (will be mocked)."""
  return user


# ---------------------------------------------------------------------------
# TestQSync -- sync pipeline tests
# ---------------------------------------------------------------------------


class TestQSync(unittest.TestCase):
  """Sync pipeline tests using MagicMock for injectable dependencies."""

  def test_happy_path(self):
    """Mock fetch and save; assert save receives enriched user and result matches."""
    # Use spec= so quent's duck-typing check (_quent_is_q) returns False,
    # not a truthy MagicMock attribute.
    mock_fetch = MagicMock(
      spec=fetch_user,
      return_value={'id': 1, 'name': 'Bob', 'email': 'bob@example.com'},
    )
    mock_save = MagicMock(spec=save_user, side_effect=lambda user: user)

    result = (
      Q()
      .then(mock_fetch)
      .then(validate_user)
      .then(enrich_user)
      .then(mock_save)
      .run(1)
    )

    # fetch was called with the user_id threaded from .run(1).
    mock_fetch.assert_called_once_with(1)

    # save receives the enriched record produced by enrich_user.
    expected_saved = {
      'id': 1, 'name': 'Bob', 'email': 'bob@example.com',
      'display_name': 'BOB', 'active': True,
    }
    mock_save.assert_called_once_with(expected_saved)

    # Pipeline returns the last step's result.
    self.assertEqual(result, expected_saved)

  def test_validation_failure_with_except(self):
    """When fetch returns an incomplete record, except_ handler returns error dict."""
    # Missing 'email' field -- validate_user will raise ValueError.
    mock_fetch = MagicMock(spec=fetch_user, return_value={'id': 2, 'name': 'Charlie'})
    mock_save = MagicMock(spec=save_user)

    result = (
      Q()
      .then(mock_fetch)
      .then(validate_user)
      .then(enrich_user)
      .then(mock_save)
      # except_ default convention: fn(exc, root_value) -- root_value is .run() arg.
      .except_(lambda ei: {'error': str(ei.exc), 'user_id': ei.root_value}, exceptions=ValueError)
      .run(2)
    )

    # save must NOT have been called -- pipeline aborted at validate_user.
    mock_save.assert_not_called()

    # except_ handler's return value replaces the pipeline result (reraise=False default).
    self.assertIn('error', result)
    self.assertEqual(result['user_id'], 2)

  def test_do_side_effect(self):
    """.do() steps execute their callable but discard the result."""
    fetched = {'id': 3, 'name': 'Diana', 'email': 'diana@example.com'}
    mock_fetch = MagicMock(spec=fetch_user, return_value=fetched)
    # tracker is a side-effect; its return value is irrelevant to the pipeline.
    tracker = MagicMock(spec=save_user, return_value='discarded')

    result = (
      Q()
      .then(mock_fetch)
      .then(validate_user)
      .do(tracker)        # side-effect: called with validated user; return discarded
      .then(enrich_user)  # still receives the same validated dict
      .run(3)
    )

    # tracker was invoked exactly once with the validated user dict.
    tracker.assert_called_once_with(fetched)

    # pipeline continued through enrich_user as if .do() was not there.
    self.assertEqual(result['display_name'], 'DIANA')
    self.assertTrue(result['active'])

  def test_except_with_explicit_args(self):
    """except_(handler, arg) -- explicit args mean handler receives those args, not QuentExcInfo."""
    mock_fetch = MagicMock(spec=fetch_user, side_effect=ConnectionError('network down'))

    result = (
      Q()
      .then(mock_fetch)
      .then(validate_user)
      # Explicit args: handler receives those args instead of QuentExcInfo.
      .except_(
        lambda code: {'error': code, 'user_id': 99},
        'ConnectionError',
        exceptions=ConnectionError,
      )
      .run(99)
    )

    mock_fetch.assert_called_once_with(99)
    self.assertEqual(result['error'], 'ConnectionError')
    self.assertEqual(result['user_id'], 99)

  def test_finally_always_runs(self):
    """finally_() runs on both success and failure paths; its return is discarded."""
    finally_tracker = MagicMock(spec=save_user)

    # Success path.
    Q(42).then(lambda x: x * 2).finally_(finally_tracker).run()
    # finally_ follows standard calling convention: receives root value.
    finally_tracker.assert_called_once_with(42)

    finally_tracker.reset_mock()

    # Failure path: the step raises, but finally_ still runs.
    def failing_step(x):
      raise RuntimeError('boom')

    with self.assertRaises(RuntimeError):
      Q(42).then(failing_step).finally_(finally_tracker).run()
    finally_tracker.assert_called_once_with(42)

  def test_clone_independence(self):
    """Cloned pipelines are independent -- running one does not affect the other."""
    enrichment_calls: list[int] = []

    def recording_enrich(user: dict) -> dict:
      enrichment_calls.append(user['id'])
      return enrich_user(user)

    base = Q().then(validate_user).then(recording_enrich)
    clone_a = base.clone()
    clone_b = base.clone()

    user_eve = {'id': 10, 'name': 'Eve', 'email': 'eve@example.com'}
    user_frank = {'id': 20, 'name': 'Frank', 'email': 'frank@example.com'}

    result_a = clone_a.run(user_eve)
    result_b = clone_b.run(user_frank)

    self.assertEqual(result_a['id'], 10)
    self.assertEqual(result_b['id'], 20)
    self.assertIn(10, enrichment_calls)
    self.assertIn(20, enrichment_calls)

    # Clones are distinct objects.
    self.assertIsNot(clone_a, clone_b)
    self.assertIsNot(clone_a, base)

    # Extending a clone does not affect the base or other clones.
    extended = clone_a.clone().then(lambda u: {**u, 'extended': True})
    result_ext = extended.run(user_eve)
    self.assertTrue(result_ext['extended'])
    result_a_again = clone_a.run(user_eve)
    self.assertNotIn('extended', result_a_again)


# ---------------------------------------------------------------------------
# TestQAsync -- async pipeline tests
# ---------------------------------------------------------------------------


class TestQAsync(IsolatedAsyncioTestCase):
  """Async pipeline tests using AsyncMock for injectable dependencies."""

  async def test_async_happy_path(self):
    """Mock async fetch and save; verify enriched user flows through correctly."""
    user_data = {'id': 5, 'name': 'Grace', 'email': 'grace@example.com'}
    mock_fetch = AsyncMock(spec=async_fetch_user, return_value=user_data)
    mock_save = AsyncMock(spec=async_save_user, side_effect=lambda user: user)

    # .run() returns a coroutine when async steps are present -- must await.
    result = await (
      Q()
      .then(mock_fetch)
      .then(validate_user)
      .then(enrich_user)
      .then(mock_save)
      .run(5)
    )

    mock_fetch.assert_called_once_with(5)
    expected = {**user_data, 'display_name': 'GRACE', 'active': True}
    mock_save.assert_called_once_with(expected)
    self.assertEqual(result, expected)

  async def test_async_error_handling(self):
    """When async fetch raises, except_ handler runs and its return value is the result."""
    mock_fetch = AsyncMock(spec=async_fetch_user, side_effect=ValueError('async db error'))

    result = await (
      Q()
      .then(mock_fetch)
      .then(validate_user)
      .then(enrich_user)
      .except_(lambda ei: {'error': str(ei.exc), 'user_id': ei.root_value}, exceptions=ValueError)
      .run(7)
    )

    mock_fetch.assert_called_once_with(7)
    self.assertEqual(result['error'], 'async db error')
    self.assertEqual(result['user_id'], 7)

  async def test_mixed_sync_async(self):
    """Pipeline transparently bridges async fetch -> sync validate/enrich -> async save."""
    user_data = {'id': 9, 'name': 'Heidi', 'email': 'heidi@example.com'}
    mock_async_fetch = AsyncMock(spec=async_fetch_user, return_value=user_data)
    mock_async_save = AsyncMock(spec=async_save_user, side_effect=lambda user: user)

    result = await (
      Q()
      .then(mock_async_fetch)  # async -- triggers async transition
      .then(validate_user)     # sync -- runs inside async continuation
      .then(enrich_user)       # sync
      .then(mock_async_save)   # async again
      .run(9)
    )

    mock_async_fetch.assert_called_once_with(9)
    mock_async_save.assert_called_once()
    self.assertEqual(result['display_name'], 'HEIDI')
    self.assertTrue(result['active'])


# ---------------------------------------------------------------------------
# TestControlFlow -- Q.return_ and Q.break_
# ---------------------------------------------------------------------------


class TestControlFlow(unittest.TestCase):
  """Tests for control flow signals."""

  def test_early_return(self):
    """Q.return_() exits the pipeline and produces the return value."""
    result = (
      Q(10)
      .then(lambda x: Q.return_(x * 2) if x > 5 else x)
      .then(lambda x: x + 100)  # skipped by return_
      .run()
    )
    self.assertEqual(result, 20)

  def test_early_return_no_value(self):
    """Q.return_() with no value produces None."""
    result = (
      Q(10)
      .then(lambda x: Q.return_())
      .then(lambda x: x + 100)
      .run()
    )
    self.assertIsNone(result)

  def test_break_in_map(self):
    """Q.break_() stops iteration; partial results are returned."""
    result = (
      Q([1, 2, 3, 4, 5])
      .foreach(lambda x: Q.break_() if x == 3 else x * 2)
      .run()
    )
    # x=1 -> 2, x=2 -> 4, x=3 -> break (no value), partial results [2, 4]
    self.assertEqual(result, [2, 4])

  def test_break_with_value(self):
    """Q.break_(value) appends break value to partial results."""
    result = (
      Q([1, 2, 3, 4, 5])
      .foreach(lambda x: Q.break_(x * 10) if x == 3 else x * 2)
      .run()
    )
    # x=3 -> break with value 30 (appended to [2, 4])
    self.assertEqual(result, [2, 4, 30])

  def test_break_outside_iteration_raises(self):
    """Q.break_() outside foreach/foreach_do raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).then(lambda x: Q.break_()).run()


# ---------------------------------------------------------------------------
# TestDecorator -- Q.as_decorator()
# ---------------------------------------------------------------------------


class TestDecorator(unittest.TestCase):
  """Tests for Q.as_decorator() -- wrapping functions with pipeline behaviour."""

  def test_decorator_wraps_function(self):
    """Decorated function's return value feeds into the pipeline as initial value."""
    enrichment_tracker = create_autospec(enrich_user, side_effect=lambda user: {**user, 'enriched': True})

    @Q().then(enrichment_tracker).as_decorator()
    def get_user(user_id: int) -> dict:
      return {'id': user_id, 'name': 'Ivan', 'email': 'ivan@example.com'}

    result = get_user(42)

    enrichment_tracker.assert_called_once_with({'id': 42, 'name': 'Ivan', 'email': 'ivan@example.com'})
    self.assertTrue(result['enriched'])
    self.assertEqual(result['id'], 42)

  def test_decorator_preserves_error_handling(self):
    """except_ and finally_ still execute correctly in decorator mode."""
    finally_tracker = MagicMock(spec=save_user)
    except_tracker = MagicMock(spec=save_user, return_value={'error': 'caught'})

    @(
      Q()
      .then(validate_user)
      .except_(except_tracker, exceptions=ValueError)
      .finally_(finally_tracker)
      .as_decorator()
    )
    def get_incomplete_user(user_id: int) -> dict:
      return {'id': user_id, 'name': 'Judy'}  # missing 'email'

    result = get_incomplete_user(55)

    # except_ default convention: fn(QuentExcInfo(exc, root_value)).
    except_tracker.assert_called_once()
    ei_arg = except_tracker.call_args.args[0]
    self.assertIsInstance(ei_arg.exc, ValueError)
    self.assertEqual(ei_arg.root_value, {'id': 55, 'name': 'Judy'})

    # finally_ always runs; receives root_value; return discarded.
    finally_tracker.assert_called_once_with({'id': 55, 'name': 'Judy'})

    # Pipeline result is except_ handler's return value.
    self.assertEqual(result, {'error': 'caught'})


# ---------------------------------------------------------------------------
# TestOnStep -- instrumentation callback
# ---------------------------------------------------------------------------


class TestOnStep(unittest.TestCase):
  """Tests for Q.on_step instrumentation."""

  def setUp(self):
    """Record the original on_step value to restore after each test."""
    self._original_on_step = Q.on_step

  def tearDown(self):
    """Restore original on_step to prevent cross-test contamination."""
    Q.on_step = self._original_on_step

  def test_on_step_records_steps(self):
    """on_step receives (q, step_name, input_value, result, elapsed_ns) for each step."""
    steps_log: list[tuple[str, object]] = []

    def recorder(q, step_name, input_value, result, elapsed_ns):
      steps_log.append((step_name, result))

    Q.on_step = recorder

    Q(10).then(lambda x: x * 2).do(lambda x: None).run()

    # Expect: root, then, do
    step_names = [name for name, _ in steps_log]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)
    self.assertIn('do', step_names)

    # Root step should report value 10.
    root_entry = next((name, val) for name, val in steps_log if name == 'root')
    self.assertEqual(root_entry[1], 10)

    # then step should report value 20.
    then_entry = next((name, val) for name, val in steps_log if name == 'then')
    self.assertEqual(then_entry[1], 20)

  def test_on_step_disabled_by_default(self):
    """When on_step is None (default), no instrumentation overhead."""
    # Just verify the pipeline runs without error when on_step is None.
    Q.on_step = None
    result = Q(5).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)

  def test_on_step_elapsed_ns(self):
    """elapsed_ns is a non-negative integer for each step."""
    elapsed_values: list[int] = []

    def recorder(q, step_name, input_value, result, elapsed_ns):
      elapsed_values.append(elapsed_ns)

    Q.on_step = recorder
    Q(1).then(lambda x: x + 1).run()

    for ns in elapsed_values:
      self.assertIsInstance(ns, int)
      self.assertGreaterEqual(ns, 0)


# ---------------------------------------------------------------------------
# TestBridgeContract -- same pipeline, sync vs async callables
# ---------------------------------------------------------------------------


class TestBridgeContract(IsolatedAsyncioTestCase):
  """Verify the bridge contract: swapping sync/async callables produces same result."""

  async def test_sync_async_equivalence(self):
    """The same pipeline with sync vs async fetch/save produces equal results."""
    user_data = {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}

    # Sync variant.
    sync_result = (
      Q()
      .then(lambda uid: user_data)
      .then(validate_user)
      .then(enrich_user)
      .then(lambda u: u)
      .run(1)
    )

    # Async variant: functionally equivalent callables.
    async def async_fetch(uid):
      return user_data

    async def async_save(u):
      return u

    async_result = await (
      Q()
      .then(async_fetch)
      .then(validate_user)
      .then(enrich_user)
      .then(async_save)
      .run(1)
    )

    # Bridge contract: results must be identical.
    self.assertEqual(sync_result, async_result)


# ---------------------------------------------------------------------------
# TestPatterns -- common testing patterns with quent
# ---------------------------------------------------------------------------


class TestPatterns(unittest.TestCase):
  """Miscellaneous patterns for testing pipelines built with quent."""

  def test_gather_results(self):
    """gather() returns a tuple; access results by index."""
    result = (
      Q(10)
      .gather(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x ** 2,
      )
      .then(lambda r: {'sum': r[0], 'double': r[1], 'square': r[2]})
      .run()
    )

    self.assertEqual(result, {'sum': 11, 'double': 20, 'square': 100})

  def test_map_and_filter(self):
    """map() transforms each element; list comprehension filters the result."""
    result = (
      Q([1, 2, 3, 4, 5, 6])
      .foreach(lambda x: x * 10)
      .then(lambda xs: [x for x in xs if x > 30])
      .run()
    )

    self.assertEqual(result, [40, 50, 60])

  def test_if_else(self):
    """if_/else_ conditional branching."""
    q = (
      Q()
      .if_(lambda x: x > 0).then(lambda x: f'positive: {x}')
      .else_(lambda x: f'non-positive: {x}')
    )

    self.assertEqual(q.run(5), 'positive: 5')
    self.assertEqual(q.run(-3), 'non-positive: -3')
    self.assertEqual(q.run(0), 'non-positive: 0')

  def test_foreach_do_preserves_elements(self):
    """foreach_do() calls fn for side-effects but collects original elements."""
    side_effects: list[int] = []

    result = (
      Q([1, 2, 3])
      .foreach_do(lambda x: side_effects.append(x * 10))
      .run()
    )

    # foreach_do collects original elements, not fn's return values.
    self.assertEqual(result, [1, 2, 3])
    # Side-effects were executed.
    self.assertEqual(side_effects, [10, 20, 30])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == '__main__':
  unittest.main()
