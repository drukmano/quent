"""Tests for the set_initial_values flag in _run() and _run_async().

Covers root_value / current_value initialization, ignore_result (do()) semantics,
callable root evaluation, falsy values, frozen chains, empty chains, multi-run
state isolation, and async transitions.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null
from quent._chain import _FrozenChain
from helpers import async_fn, async_identity, make_tracker, make_async_tracker


# ---------------------------------------------------------------------------
# Sync path: set_initial_values in _run()
# ---------------------------------------------------------------------------


class TestSetInitialValuesSync(unittest.TestCase):

  def test_root_link_sets_root_value(self):
    """Chain(42) sets root_value to 42; finally_ receives it."""
    tracker = make_tracker()
    Chain(42).finally_(tracker).run()
    self.assertEqual(len(tracker.calls), 1)
    self.assertEqual(tracker.calls[0], ((42,), {}))

  def test_run_value_sets_root_value(self):
    """Chain().then(...).finally_(tracker).run(42) — run value becomes root."""
    tracker = make_tracker()
    Chain().then(lambda x: x + 1).finally_(tracker).run(42)
    self.assertEqual(len(tracker.calls), 1)
    self.assertEqual(tracker.calls[0], ((42,), {}))

  def test_callable_root_evaluated(self):
    """Chain(lambda: 5) evaluates the callable; root_value = 5."""
    tracker = make_tracker()
    result = Chain(lambda: 5).finally_(tracker).run()
    self.assertEqual(result, 5)
    self.assertEqual(tracker.calls[0], ((5,), {}))

  def test_do_first_link_preserves_initial(self):
    """do() as first link after root must NOT change current_value."""
    result = Chain(5).do(lambda x: 999).then(lambda x: x).run()
    self.assertEqual(result, 5)

  def test_multiple_do_preserve(self):
    """Several consecutive do() links must not alter current_value."""
    calls = []
    fn1 = lambda x: calls.append(('fn1', x)) or 'ignored1'
    fn2 = lambda x: calls.append(('fn2', x)) or 'ignored2'
    fn3 = lambda x: calls.append(('fn3', x)) or 'ignored3'
    result = Chain(5).do(fn1).do(fn2).do(fn3).run()
    self.assertEqual(result, 5)
    # Each do() receives current_value = 5
    self.assertEqual(calls, [('fn1', 5), ('fn2', 5), ('fn3', 5)])

  def test_run_value_callable(self):
    """run(lambda: 10) evaluates the callable, result feeds into then()."""
    result = Chain().then(lambda x: x + 1).run(lambda: 10)
    self.assertEqual(result, 11)

  def test_no_root_no_run_first_then_is_root(self):
    """Chain().then(lambda: 42).then(lambda x: x+1).run() — first then sets root."""
    result = Chain().then(lambda: 42).then(lambda x: x + 1).run()
    self.assertEqual(result, 43)

  def test_empty_chain_returns_none(self):
    """Chain().run() with no root and no links returns None."""
    result = Chain().run()
    self.assertIsNone(result)

  # -- Falsy initial values --

  def test_root_value_zero(self):
    """Chain(0) — 0 is falsy but valid; root_value = 0."""
    tracker = make_tracker()
    result = Chain(0).finally_(tracker).run()
    self.assertEqual(result, 0)
    self.assertEqual(tracker.calls[0], ((0,), {}))

  def test_root_value_false(self):
    """Chain(False) — False is falsy but valid."""
    tracker = make_tracker()
    result = Chain(False).finally_(tracker).run()
    self.assertIs(result, False)
    self.assertIs(tracker.calls[0][0][0], False)

  def test_root_value_none(self):
    """Chain(None) — None is a valid chain value, distinct from Null."""
    tracker = make_tracker()
    result = Chain(None).finally_(tracker).run()
    # None is not callable, so it's returned as-is.
    self.assertIsNone(result)
    self.assertIsNone(tracker.calls[0][0][0])

  def test_root_value_empty_string(self):
    """Chain('') — empty string is falsy but valid."""
    tracker = make_tracker()
    result = Chain('').finally_(tracker).run()
    self.assertEqual(result, '')
    self.assertEqual(tracker.calls[0][0][0], '')

  def test_root_value_empty_list(self):
    """Chain([]) — empty list is falsy but valid."""
    tracker = make_tracker()
    result = Chain([]).finally_(tracker).run()
    self.assertEqual(result, [])
    self.assertEqual(tracker.calls[0][0][0], [])

  def test_run_value_zero(self):
    """run(0) — 0 is valid; flows into the chain."""
    tracker = make_tracker()
    result = Chain().then(lambda x: x + 1).finally_(tracker).run(0)
    self.assertEqual(result, 1)
    self.assertEqual(tracker.calls[0], ((0,), {}))

  # -- do() as only link --

  def test_do_only_link_with_root(self):
    """Chain(5).do(fn).run() — do discards its result; chain returns root."""
    calls = []
    result = Chain(5).do(lambda x: calls.append(x)).run()
    self.assertEqual(result, 5)
    self.assertEqual(calls, [5])

  def test_do_only_link_no_root(self):
    """Chain().do(lambda: 'side').run(10) — do discards; returns 10."""
    calls = []
    result = Chain().do(lambda x: calls.append(x) or 'ignored').run(10)
    self.assertEqual(result, 10)
    self.assertEqual(calls, [10])

  # -- finally_ with no other links --

  def test_only_finally_with_root(self):
    """Chain(42).finally_(tracker).run() — handler receives root_value=42."""
    tracker = make_tracker()
    result = Chain(42).finally_(tracker).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker.calls[0], ((42,), {}))

  def test_only_finally_no_root(self):
    """Chain().finally_(tracker).run() — root_value is Null; handler called with Null semantics."""
    tracker = make_tracker()
    result = Chain().finally_(tracker).run()
    self.assertIsNone(result)
    # root_value is Null, so _evaluate_value treats tracker as callable with no current_value;
    # tracker is called with no args if root_value is Null.
    self.assertEqual(len(tracker.calls), 1)
    self.assertEqual(tracker.calls[0], ((), {}))

  # -- Multiple runs on same chain (state isolation) --

  def test_multiple_runs_no_state_leak(self):
    """Running the same chain multiple times must not leak state."""
    c = Chain().then(lambda x: x * 2)
    self.assertEqual(c.run(3), 6)
    self.assertEqual(c.run(5), 10)
    self.assertEqual(c.run(0), 0)

  def test_multiple_runs_root_value_isolated(self):
    """root_value must be computed fresh on each run()."""
    tracker = make_tracker()
    c = Chain(lambda: 42).finally_(tracker)
    c.run()
    c.run()
    self.assertEqual(len(tracker.calls), 2)
    self.assertEqual(tracker.calls[0], ((42,), {}))
    self.assertEqual(tracker.calls[1], ((42,), {}))

  def test_run_with_different_values_root_isolated(self):
    """Each run(v) sets root_value independently."""
    collected = []
    c = Chain().then(lambda x: x + 1).finally_(lambda rv: collected.append(rv))
    c.run(10)
    c.run(20)
    self.assertEqual(collected, [10, 20])

  # -- Root value and current_value divergence --

  def test_root_value_not_affected_by_then(self):
    """root_value stays as the first evaluated result, even as current_value changes."""
    tracker = make_tracker()
    result = (
      Chain(1)
      .then(lambda x: x + 10)
      .then(lambda x: x + 100)
      .finally_(tracker)
      .run()
    )
    self.assertEqual(result, 111)
    # root_value = 1 (first evaluation of root_link)
    self.assertEqual(tracker.calls[0], ((1,), {}))

  def test_root_value_with_run_arg_and_root_link(self):
    """run(v) overrides root_link: root_value = evaluated run value."""
    tracker = make_tracker()
    # Chain(100) has root_link=100, but run(5) creates a temp link for 5
    # that splices before first_link, bypassing root_link.
    result = Chain(100).then(lambda x: x + 1).finally_(tracker).run(5)
    # The run value 5 becomes the root; root_link (100) is bypassed.
    self.assertEqual(result, 6)
    self.assertEqual(tracker.calls[0], ((5,), {}))

  # -- Frozen chains and initial values --

  def test_frozen_chain_root_value(self):
    """Frozen chain preserves root_value semantics."""
    tracker = make_tracker()
    frozen = Chain(42).then(lambda x: x + 1).finally_(tracker).freeze()
    result = frozen.run()
    self.assertEqual(result, 43)
    self.assertEqual(tracker.calls[0], ((42,), {}))

  def test_frozen_chain_run_value(self):
    """Frozen chain with run(v) uses v as root."""
    tracker = make_tracker()
    frozen = Chain().then(lambda x: x * 3).finally_(tracker).freeze()
    result = frozen.run(7)
    self.assertEqual(result, 21)
    self.assertEqual(tracker.calls[0], ((7,), {}))

  def test_frozen_chain_multiple_runs_isolated(self):
    """Frozen chain does not leak state across runs."""
    tracker = make_tracker()
    frozen = Chain().then(lambda x: x + 1).finally_(tracker).freeze()
    self.assertEqual(frozen.run(1), 2)
    self.assertEqual(frozen.run(10), 11)
    self.assertEqual(tracker.calls[0], ((1,), {}))
    self.assertEqual(tracker.calls[1], ((10,), {}))

  # -- Edge: do() between then() links --

  def test_do_between_thens(self):
    """do() in the middle of a chain must not disrupt value flow."""
    side = []
    result = (
      Chain(1)
      .then(lambda x: x + 1)     # 2
      .do(lambda x: side.append(x))
      .then(lambda x: x + 10)    # 12
      .run()
    )
    self.assertEqual(result, 12)
    self.assertEqual(side, [2])

  # -- Edge: then() returns non-callable (plain value) --

  def test_then_plain_value_replaces_current(self):
    """then(42) uses 42 as a plain value (not callable), replacing current_value."""
    result = Chain(1).then(42).run()
    self.assertEqual(result, 42)

  # -- has_root_value logic --

  def test_has_root_value_false_no_root_no_run(self):
    """No root_link and no run value: has_root_value is False, link starts at first_link."""
    result = Chain().then(lambda: 99).run()
    self.assertEqual(result, 99)

  def test_has_root_value_true_with_root_link(self):
    """Chain(v) sets has_root_value=True via root_link."""
    tracker = make_tracker()
    Chain(7).finally_(tracker).run()
    self.assertEqual(tracker.calls[0], ((7,), {}))


# ---------------------------------------------------------------------------
# Async path: set_initial_values in _run_async()
# ---------------------------------------------------------------------------


class TestSetInitialValuesAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_root_sets_root_value(self):
    """Async root callable — root_value captured after await."""
    tracker = make_tracker()
    async def async_root():
      return 42

    result = await Chain(async_root).finally_(tracker).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker.calls[0], ((42,), {}))

  async def test_async_do_preserves(self):
    """do() in async path must not change current_value."""
    side = []

    async def async_side(x):
      side.append(x)
      return 'ignored'

    result = await Chain(5).do(async_side).then(lambda x: x).run()
    self.assertEqual(result, 5)
    self.assertEqual(side, [5])

  async def test_async_transition_preserves_root_value(self):
    """root_value set before async transition is preserved for finally_."""
    tracker = make_tracker()
    result = await (
      Chain(10)
      .then(lambda x: x + 1)     # 11, sync
      .then(async_fn)             # awaitable -> async transition, 12
      .then(lambda x: x + 1)     # 13
      .finally_(tracker)
      .run()
    )
    self.assertEqual(result, 13)
    # root_value = 10 (set during sync phase before async transition)
    self.assertEqual(tracker.calls[0], ((10,), {}))

  async def test_async_run_value_callable(self):
    """run(async_callable) — awaitable run value sets root after await."""
    tracker = make_tracker()

    async def async_start():
      return 7

    result = await Chain().then(lambda x: x + 1).finally_(tracker).run(async_start)
    self.assertEqual(result, 8)
    self.assertEqual(tracker.calls[0], ((7,), {}))

  async def test_async_root_value_with_falsy(self):
    """Async root returning 0 (falsy) must still set root_value = 0."""
    tracker = make_tracker()

    async def async_zero():
      return 0

    result = await Chain(async_zero).then(lambda x: x + 1).finally_(tracker).run()
    self.assertEqual(result, 1)
    self.assertEqual(tracker.calls[0], ((0,), {}))

  async def test_async_multiple_do_preserve(self):
    """Multiple async do() links must not change current_value."""
    calls = []

    async def s1(x):
      calls.append(('s1', x))
      return 'nope1'

    async def s2(x):
      calls.append(('s2', x))
      return 'nope2'

    result = await Chain(5).do(s1).do(s2).then(lambda x: x).run()
    self.assertEqual(result, 5)
    self.assertEqual(calls, [('s1', 5), ('s2', 5)])

  async def test_async_empty_chain_returns_none(self):
    """Chain().run() with async step that evaluates to Null path."""
    # An empty chain has no links, so it returns None synchronously.
    result = Chain().run()
    self.assertIsNone(result)

  async def test_async_root_value_not_affected_by_then(self):
    """root_value stays as first evaluation even through async steps."""
    tracker = make_tracker()
    result = await (
      Chain(1)
      .then(async_fn)          # 2
      .then(lambda x: x + 10) # 12
      .finally_(tracker)
      .run()
    )
    self.assertEqual(result, 12)
    self.assertEqual(tracker.calls[0], ((1,), {}))

  async def test_async_do_between_thens(self):
    """do() between async then() links preserves pipeline value."""
    side = []

    async def async_side_effect(x):
      side.append(x)
      return 'discarded'

    result = await (
      Chain(1)
      .then(async_fn)                 # 2
      .do(async_side_effect)          # side effect, discarded
      .then(lambda x: x + 100)       # 102
      .run()
    )
    self.assertEqual(result, 102)
    self.assertEqual(side, [2])

  async def test_async_frozen_chain_root_value(self):
    """Frozen chain with async steps preserves root_value."""
    tracker = make_tracker()
    frozen = Chain(5).then(async_fn).then(lambda x: x * 2).finally_(tracker).freeze()
    result = await frozen.run()
    # 5 -> async_fn(5)=6 -> *2=12
    self.assertEqual(result, 12)
    self.assertEqual(tracker.calls[0], ((5,), {}))

  async def test_async_frozen_multiple_runs(self):
    """Frozen chain with async steps does not leak state."""
    tracker = make_tracker()
    frozen = Chain().then(async_fn).finally_(tracker).freeze()
    r1 = await frozen.run(1)  # async_fn(1) = 2
    r2 = await frozen.run(10) # async_fn(10) = 11
    self.assertEqual(r1, 2)
    self.assertEqual(r2, 11)
    self.assertEqual(tracker.calls[0], ((1,), {}))
    self.assertEqual(tracker.calls[1], ((10,), {}))

  async def test_async_root_value_with_except(self):
    """root_value set even when async chain raises and is caught by except_."""
    tracker = make_tracker()

    async def fail(x):
      raise ValueError('boom')

    result = await (
      Chain(7)
      .then(fail)
      .except_(lambda rv, e: 'caught')
      .finally_(tracker)
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker.calls[0], ((7,), {}))


if __name__ == '__main__':
  unittest.main()
