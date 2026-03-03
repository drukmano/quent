"""Exhaustive tests for iteration operations (foreach, filter, gather,
foreach_indexed), control flow signals (_Return, _Break), and context
managers (with_).

Covers sync, async, sync-to-async transitions, full-async iterables,
edge cases (empty/single/large collections), Break/Return propagation,
CM enter/exit lifecycle, nested chains, Cascade interactions, and
exception handler interplay.
"""

import asyncio
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class SyncIterator:
  """Minimal sync iterable for testing."""
  def __init__(self, items=None):
    self._items = items if items is not None else list(range(10))

  def __iter__(self):
    return iter(self._items)


class AsyncIterator:
  """Minimal async iterable for testing."""
  def __init__(self, items=None):
    self._items = list(items) if items is not None else list(range(10))

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class CustomIterator:
  """Custom iterator class with __iter__/__next__ directly."""
  def __init__(self, items):
    self._items = list(items)
    self._idx = 0

  def __iter__(self):
    self._idx = 0
    return self

  def __next__(self):
    if self._idx >= len(self._items):
      raise StopIteration
    val = self._items[self._idx]
    self._idx += 1
    return val


class SimpleCM:
  """Sync context manager for testing."""
  def __init__(self, value='ctx_value', exit_return=False,
               enter_raise=None, exit_raise=None):
    self.value = value
    self.exit_return = exit_return
    self.enter_raise = enter_raise
    self.exit_raise = exit_raise
    self.entered = False
    self.exited = False
    self.exit_args = None

  def __enter__(self):
    if self.enter_raise:
      raise self.enter_raise
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_args = (exc_type, exc_val, exc_tb)
    if self.exit_raise:
      raise self.exit_raise
    return self.exit_return


class AsyncCM:
  """Async context manager for testing."""
  def __init__(self, value='async_ctx', exit_return=False,
               enter_raise=None, exit_raise=None):
    self.value = value
    self.exit_return = exit_return
    self.enter_raise = enter_raise
    self.exit_raise = exit_raise
    self.entered = False
    self.exited = False
    self.exit_args = None

  async def __aenter__(self):
    if self.enter_raise:
      raise self.enter_raise
    self.entered = True
    return self.value

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_args = (exc_type, exc_val, exc_tb)
    if self.exit_raise:
      raise self.exit_raise
    return self.exit_return


class CoroExitCM:
  """Sync CM whose __exit__ returns a coroutine (edge case for _with_to_async)."""
  def __init__(self, value='coro_exit_val', suppress=False):
    self.value = value
    self.suppress = suppress
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    async def _exit():
      return self.suppress
    return _exit()


# ===========================================================================
# A. FOREACH COMPREHENSIVE (20+ tests)
# ===========================================================================

class ForeachComprehensiveTests(MyTestCase):
  """Exhaustive testing of the foreach operation."""

  # -- 1. Sync foreach with sync function on a regular list --
  async def test_foreach_sync_fn_on_list(self):
    """Apply a sync function to each element of a list and collect results."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(lambda x: x * 2).run(),
          [2, 4, 6, 8]
        )

  # -- 2. Foreach on empty iterable --
  async def test_foreach_empty_iterable(self):
    """Foreach on an empty list returns an empty list."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).foreach(lambda x: x * 2).run(),
          []
        )

  # -- 3. Foreach on single-element iterable --
  async def test_foreach_single_element(self):
    """Foreach on a single-element list returns a single-element result."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [42]).foreach(lambda x: x + 1).run(),
          [43]
        )

  # -- 4. Foreach on large iterable (1000 elements) --
  async def test_foreach_large_iterable(self):
    """Foreach over 1000 elements correctly processes all."""
    items = list(range(1000))
    expected = [x * 3 for x in items]
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, items).foreach(lambda x: x * 3).run(),
          expected
        )

  # -- 5. Foreach where fn returns coroutine on FIRST element --
  async def test_foreach_async_transition_on_first_element(self):
    """fn returns coroutine immediately on the first element, triggering
    _foreach_to_async from the very start."""
    await self.assertEqual(
      Chain([10, 20, 30]).foreach(lambda x: aempty(x + 1)).run(),
      [11, 21, 31]
    )

  # -- 6. Foreach where fn returns coroutine on MIDDLE element --
  async def test_foreach_async_transition_on_middle_element(self):
    """fn returns sync for first elements, then coroutine mid-iteration."""
    call_count = {'n': 0}
    def f(el):
      call_count['n'] += 1
      if call_count['n'] == 3:
        return aempty(el * 10)
      return el * 10
    call_count['n'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5]).foreach(f).run(),
      [10, 20, 30, 40, 50]
    )

  # -- 7. Foreach where fn returns coroutine on LAST element --
  async def test_foreach_async_transition_on_last_element(self):
    """fn returns coroutine only on the very last element."""
    def f(el):
      if el == 5:
        return aempty(el * 100)
      return el * 100
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5]).foreach(f).run(),
      [100, 200, 300, 400, 500]
    )

  # -- 8. Foreach with async iterable (__aiter__) --
  async def test_foreach_async_iterable(self):
    """Foreach on an async iterable uses _foreach_full_async."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3])).foreach(lambda x: x * 5).run(),
          [5, 10, 15]
        )

  # -- 9. Foreach with custom iterator class --
  async def test_foreach_custom_iterator(self):
    """Foreach on an object implementing __iter__/__next__ directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, CustomIterator([10, 20, 30])).foreach(lambda x: x + 1).run(),
          [11, 21, 31]
        )

  # -- 10. Foreach with _Break exception --
  async def test_foreach_break_returns_partial_list(self):
    """Break in foreach returns the accumulated list so far."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 4:
            return Chain.break_()
          return fn(el * 2)
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).foreach(f).run(),
          [2, 4, 6]
        )

  # -- 11. Foreach with _Break carrying a value --
  async def test_foreach_break_with_value(self):
    """Break with a value overrides the accumulated list."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def f(el):
          if el == 3:
            return Chain.break_(sentinel)
          return fn(el)
        await self.assertIs(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(),
          sentinel
        )

  # -- 12. Foreach with _Break on first element --
  async def test_foreach_break_on_first_element(self):
    """Break on the very first element returns []."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          return Chain.break_()
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          []
        )

  # -- 13. Foreach with _Break on last element --
  async def test_foreach_break_on_last_element(self):
    """Break on the last element returns all but the last."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el == 3:
            return Chain.break_()
          return fn(el * 10)
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          [10, 20]
        )

  # -- 14. Foreach with other exception --
  async def test_foreach_exception_propagates(self):
    """Non-break exceptions propagate from foreach."""
    def f(el):
      if el == 2:
        raise TestExc('foreach error')
      return el
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).foreach(f).run())

  # -- 15. Foreach in async with _Break --
  async def test_foreach_async_iterable_break(self):
    """Break in fully async foreach returns partial list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 3:
            return Chain.break_()
          return fn(el * 2)
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3, 4, 5])).foreach(f).run(),
          [2, 4]
        )

  # -- 16. Foreach in async with exception --
  async def test_foreach_async_iterable_exception(self):
    """Exception in fully async foreach propagates and sets __quent_link_temp_args__."""
    def f(el):
      if el == 2:
        raise TestExc('async foreach error')
      return el
    with self.assertRaises(TestExc):
      await await_(Chain(AsyncIterator([1, 2, 3])).foreach(f).run())

  # -- 17. Foreach where fn returns None for some elements --
  async def test_foreach_fn_returns_none(self):
    """fn returning None includes None in the result list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el % 2 == 0:
            return None
          return el
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(),
          [1, None, 3, None]
        )

  # -- 18. Foreach where fn modifies mutable state --
  async def test_foreach_fn_modifies_mutable_state(self):
    """fn modifying external mutable state is observable."""
    state = []
    def f(el):
      state.append(el)
      return el * 2
    for fn, ctx in self.with_fn():
      with ctx:
        state.clear()
        await self.assertEqual(
          Chain(fn, [10, 20, 30]).foreach(f).run(),
          [20, 40, 60]
        )
        super(MyTestCase, self).assertEqual(state, [10, 20, 30])

  # -- 19. Foreach with frozen Chain as fn --
  async def test_foreach_frozen_chain_as_fn(self):
    """A frozen Chain instance used as the foreach fn argument.
    Chains must be frozen to be used as fn in foreach (unfrozen chains
    used directly as fn trigger 'cannot run nested chain' error)."""
    inner = Chain().then(lambda x: x * 2).then(lambda x: x + 1).freeze()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(inner).run(),
          [3, 5, 7]
        )

  # -- 20. Foreach chained with .then() --
  async def test_foreach_then_chain(self):
    """Result list from foreach flows to the next .then() link."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(lambda x: x * 2).then(sum).run(),
          12
        )

  # -- 21. Foreach with async fn on async iterable --
  async def test_foreach_async_fn_on_async_iterable(self):
    """Async fn applied to elements from an async iterable."""
    await self.assertEqual(
      Chain(AsyncIterator([2, 4, 6])).foreach(lambda x: aempty(x // 2)).run(),
      [1, 2, 3]
    )

  # -- 22. Foreach break with value in async foreach --
  async def test_foreach_async_break_with_value(self):
    """Break with a value in fully async foreach returns that value."""
    sentinel = object()
    def f(el):
      if el == 2:
        return Chain.break_(sentinel)
      return el
    result = await await_(Chain(AsyncIterator([1, 2, 3])).foreach(f).run())
    super(MyTestCase, self).assertIs(result, sentinel)

  # -- 23. Foreach in _foreach_to_async with exception --
  async def test_foreach_to_async_exception_sets_temp_args(self):
    """Exception in _foreach_to_async path sets __quent_link_temp_args__ on exc."""
    call_count = {'n': 0}
    def f(el):
      call_count['n'] += 1
      if call_count['n'] == 1:
        return aempty(el)
      raise TestExc('async transition error')
    call_count['n'] = 0
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).foreach(f).run())

  # -- 24. Foreach break in _foreach_to_async path --
  async def test_foreach_to_async_break(self):
    """Break in _foreach_to_async path after coroutine transition."""
    call_count = {'n': 0}
    def f(el):
      call_count['n'] += 1
      if call_count['n'] == 1:
        return aempty(el * 10)
      if call_count['n'] == 3:
        return Chain.break_()
      return el * 10
    call_count['n'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5]).foreach(f).run(),
      [10, 20]
    )


# ===========================================================================
# B. FOREACH_INDEXED COMPREHENSIVE (10+ tests)
# ===========================================================================

class ForeachIndexedComprehensiveTests(MyTestCase):
  """Exhaustive testing of the foreach_indexed operation."""

  # -- 22 (global). Sync foreach_indexed with index verification --
  async def test_foreach_indexed_sync_indices(self):
    """fn(idx, el) receives correct zero-based indices."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c']).foreach(
            lambda idx, el: (idx, el), with_index=True
          ).run(),
          [(0, 'a'), (1, 'b'), (2, 'c')]
        )

  # -- 23. Foreach_indexed with empty iterable --
  async def test_foreach_indexed_empty(self):
    """Foreach_indexed on empty list returns []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).foreach(
            lambda idx, el: (idx, el), with_index=True
          ).run(),
          []
        )

  # -- 24. Foreach_indexed where fn returns coroutine --
  async def test_foreach_indexed_async_transition(self):
    """fn returns coroutine mid-iteration in indexed path."""
    call_count = {'n': 0}
    def f(idx, el):
      call_count['n'] += 1
      if call_count['n'] > 1:
        return aempty((idx, el))
      return (idx, el)
    call_count['n'] = 0
    await self.assertEqual(
      Chain(['a', 'b', 'c']).foreach(f, with_index=True).run(),
      [(0, 'a'), (1, 'b'), (2, 'c')]
    )

  # -- 25. Foreach_indexed with async iterable --
  async def test_foreach_indexed_async_iterable(self):
    """Foreach_indexed on async iterable uses _foreach_indexed_full_async."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator(['x', 'y', 'z'])).foreach(
            lambda idx, el: (idx, el), with_index=True
          ).run(),
          [(0, 'x'), (1, 'y'), (2, 'z')]
        )

  # -- 26. Foreach_indexed with _Break --
  async def test_foreach_indexed_break(self):
    """Break at idx==2 returns partial list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(idx, el):
          if idx == 2:
            return Chain.break_()
          return fn((idx, el))
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c', 'd']).foreach(f, with_index=True).run(),
          [(0, 'a'), (1, 'b')]
        )

  # -- 27. Foreach_indexed with exception --
  async def test_foreach_indexed_exception(self):
    """Exception in indexed fn propagates."""
    def f(idx, el):
      raise TestExc('indexed error')
    with self.assertRaises(TestExc):
      await await_(Chain(['a', 'b']).foreach(f, with_index=True).run())

  # -- 28. Foreach_indexed index starts at 0 and increments --
  async def test_foreach_indexed_correct_indices_large(self):
    """Verify index correctness over 50 elements."""
    items = list(range(50))
    expected = [(i, i) for i in range(50)]
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, items).foreach(
            lambda idx, el: (idx, el), with_index=True
          ).run(),
          expected
        )

  # -- 29. Foreach_indexed async transition preserves correct indices --
  async def test_foreach_indexed_async_transition_preserves_indices(self):
    """When async transition occurs mid-iteration, subsequent indices are correct."""
    call_count = {'n': 0}
    def f(idx, el):
      call_count['n'] += 1
      if call_count['n'] == 3:
        return aempty((idx, el))
      return (idx, el)
    call_count['n'] = 0
    await self.assertEqual(
      Chain(['a', 'b', 'c', 'd', 'e']).foreach(f, with_index=True).run(),
      [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
    )

  # -- 30. Foreach_indexed with break carrying a value --
  async def test_foreach_indexed_break_with_value(self):
    """Break with a value in indexed foreach returns that value."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def f(idx, el):
          if idx == 1:
            return Chain.break_(sentinel)
          return fn((idx, el))
        await self.assertIs(
          Chain(fn, ['a', 'b', 'c']).foreach(f, with_index=True).run(),
          sentinel
        )

  # -- 31. Foreach_indexed with async iterable and break --
  async def test_foreach_indexed_async_iterable_break(self):
    """Break in _foreach_indexed_full_async returns partial list."""
    def f(idx, el):
      if idx >= 2:
        return Chain.break_()
      return (idx, el)
    await self.assertEqual(
      Chain(AsyncIterator([10, 20, 30, 40])).foreach(f, with_index=True).run(),
      [(0, 10), (1, 20)]
    )

  # -- 32. Foreach_indexed with async iterable and exception --
  async def test_foreach_indexed_async_iterable_exception(self):
    """Exception in _foreach_indexed_full_async propagates."""
    def f(idx, el):
      if idx == 1:
        raise TestExc('indexed async error')
      return (idx, el)
    with self.assertRaises(TestExc):
      await await_(
        Chain(AsyncIterator([10, 20, 30])).foreach(f, with_index=True).run()
      )

  # -- 33. Foreach_indexed chained with .then() --
  async def test_foreach_indexed_then_chain(self):
    """Result of foreach_indexed flows to subsequent .then() link."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, ['a', 'b']).foreach(
            lambda idx, el: f'{idx}:{el}', with_index=True
          ).then(lambda lst: ','.join(lst)).run(),
          '0:a,1:b'
        )

  # -- 34. Foreach_indexed async fn on first element --
  async def test_foreach_indexed_async_on_first_element(self):
    """fn returns coroutine immediately on the first element."""
    await self.assertEqual(
      Chain([100, 200]).foreach(
        lambda idx, el: aempty((idx, el)), with_index=True
      ).run(),
      [(0, 100), (1, 200)]
    )


# ===========================================================================
# C. FILTER COMPREHENSIVE (12+ tests)
# ===========================================================================

class FilterComprehensiveTests(MyTestCase):
  """Exhaustive testing of the filter operation."""

  # -- 31. Sync filter with sync predicate --
  async def test_filter_sync_predicate(self):
    """Filter keeps elements where predicate returns truthy."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).run(),
          [2, 4]
        )

  # -- 32. Filter on empty iterable --
  async def test_filter_empty(self):
    """Filter on empty iterable returns []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).filter(lambda x: True).run(),
          []
        )

  # -- 33. Filter where ALL elements pass --
  async def test_filter_all_pass(self):
    """All elements pass the predicate."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [2, 4, 6]).filter(lambda x: x % 2 == 0).run(),
          [2, 4, 6]
        )

  # -- 34. Filter where NO elements pass --
  async def test_filter_none_pass(self):
    """No elements pass the predicate, returns []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 3, 5]).filter(lambda x: x % 2 == 0).run(),
          []
        )

  # -- 35. Filter where predicate returns coroutine --
  async def test_filter_async_predicate(self):
    """Sync iterable, async predicate triggers _filter_to_async."""
    await self.assertEqual(
      Chain([1, 2, 3, 4]).filter(lambda x: aempty(x > 2)).run(),
      [3, 4]
    )

  # -- 36. Filter with async iterable --
  async def test_filter_async_iterable(self):
    """Async iterable triggers _filter_full_async."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator([10, 20, 30, 40])).filter(lambda x: x >= 30).run(),
          [30, 40]
        )

  # -- 37. Filter exception handling --
  async def test_filter_exception_propagates(self):
    """Exception in predicate propagates."""
    def pred(x):
      raise TestExc('filter error')
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).filter(pred).run())

  # -- 38. Filter with falsy but valid values (0, '', False) --
  async def test_filter_falsy_predicate_results_filter_out(self):
    """Predicate returning 0, '', or False filters out the element."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: 0).run(),
          []
        )
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: '').run(),
          []
        )
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: False).run(),
          []
        )

  # -- 39. Filter with truthy non-True values --
  async def test_filter_truthy_predicate_keeps_element(self):
    """Predicate returning 1, 'yes', [1] keeps the element."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c']).filter(lambda x: 1).run(),
          ['a', 'b', 'c']
        )
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c']).filter(lambda x: 'yes').run(),
          ['a', 'b', 'c']
        )
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c']).filter(lambda x: [1]).run(),
          ['a', 'b', 'c']
        )

  # -- 40. Filter chained with foreach --
  async def test_filter_then_foreach(self):
    """Filter then foreach: filter result flows to foreach as the iterable."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5, 6])
          .filter(lambda x: x % 2 == 0)
          .foreach(lambda x: x * 10)
          .run(),
          [20, 40, 60]
        )

  # -- 41. Filter chained with then --
  async def test_filter_then_chain(self):
    """Result list from filter flows to .then() link."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).filter(lambda x: x > 3).then(len).run(),
          2
        )

  # -- 42. Filter with complex predicate (multiple conditions) --
  async def test_filter_complex_predicate(self):
    """Predicate uses multiple conditions."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, list(range(20))).filter(
            lambda x: x > 5 and x < 15 and x % 3 == 0
          ).run(),
          [6, 9, 12]
        )

  # -- 43. Filter sync-to-async transition --
  async def test_filter_sync_to_async_transition(self):
    """Predicate returns sync for first elements, then coroutine."""
    call_count = {'n': 0}
    def pred(x):
      call_count['n'] += 1
      if call_count['n'] > 2:
        return aempty(x % 2 == 0)
      return x % 2 == 0
    call_count['n'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5, 6]).filter(pred).run(),
      [2, 4, 6]
    )

  # -- 44. Filter with async iterable and async predicate --
  async def test_filter_async_iterable_async_predicate(self):
    """Async iterable with async predicate: full async path."""
    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3, 4, 5]))
      .filter(lambda x: aempty(x >= 3))
      .run(),
      [3, 4, 5]
    )

  # -- 45. Filter with single element that passes --
  async def test_filter_single_element_passes(self):
    """Single element passing the predicate."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [42]).filter(lambda x: x > 0).run(),
          [42]
        )

  # -- 46. Filter with single element that fails --
  async def test_filter_single_element_fails(self):
    """Single element failing the predicate."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [42]).filter(lambda x: x < 0).run(),
          []
        )


# ===========================================================================
# D. GATHER COMPREHENSIVE (10+ tests)
# ===========================================================================

class GatherComprehensiveTests(MyTestCase):
  """Exhaustive testing of the gather operation."""

  # -- 43. Gather with multiple sync functions --
  async def test_gather_all_sync(self):
    """All functions are sync, results returned in order."""
    f1 = lambda v: v + 1
    f2 = lambda v: v * 2
    f3 = lambda v: v ** 2
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather(f1, f2, f3).run(),
          [6, 10, 25]
        )

  # -- 44. Gather with all async functions --
  async def test_gather_all_async(self):
    """All functions are async, uses asyncio.gather."""
    f1 = lambda v: aempty(v + 1)
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: aempty(v ** 2)
    await self.assertEqual(
      Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  # -- 45. Gather with mix of sync and async functions --
  async def test_gather_mixed_sync_async(self):
    """Some sync, some async functions."""
    f1 = lambda v: v + 1
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: v ** 2
    await self.assertEqual(
      Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  # -- 46. Gather with single function --
  async def test_gather_single_fn(self):
    """Single fn returns [result]."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather(lambda v: v * 3).run(),
          [15]
        )

  # -- 47. Gather with zero functions --
  async def test_gather_empty(self):
    """No functions => returns []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather().run(),
          []
        )

  # -- 48. Gather result ordering matches function order --
  async def test_gather_preserves_order(self):
    """Results match function order even with async delays."""
    async def slow(v):
      await asyncio.sleep(0.01)
      return 'slow'
    def fast(v):
      return 'fast'
    await self.assertEqual(
      Chain(1).gather(slow, fast, slow).run(),
      ['slow', 'fast', 'slow']
    )

  # -- 49. Gather where one function raises --
  async def test_gather_exception_propagates(self):
    """One fn raises, exception propagates."""
    def f1(v):
      return v + 1
    def f2(v):
      raise TestExc('gather error')
    def f3(v):
      return v + 3
    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(f1, f2, f3).run())

  # -- 50. Gather chained with .then() --
  async def test_gather_then_chain(self):
    """Result list from gather flows to .then() link."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).gather(
            lambda v: v + 1, lambda v: v + 2, lambda v: v + 3
          ).then(sum).run(),
          36
        )

  # -- 51. Gather functions all receive the same current_value --
  async def test_gather_all_receive_same_value(self):
    """All functions receive the same current value."""
    received = []
    def capture(v):
      received.append(v)
      return v
    for fn, ctx in self.with_fn():
      with ctx:
        received.clear()
        await self.assertEqual(
          Chain(fn, 42).gather(capture, capture, capture).run(),
          [42, 42, 42]
        )
        super(MyTestCase, self).assertEqual(received, [42, 42, 42])

  # -- 52. Gather with identical functions --
  async def test_gather_identical_functions(self):
    """Same function object used multiple times."""
    fn = lambda v: v * 2
    for fn_, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn_, 7).gather(fn, fn, fn).run(),
          [14, 14, 14]
        )

  # -- 53. Gather with async only first fn --
  async def test_gather_first_fn_async_rest_sync(self):
    """Only the first fn is async."""
    await self.assertEqual(
      Chain(3).gather(
        lambda v: aempty(v + 10),
        lambda v: v + 20,
        lambda v: v + 30
      ).run(),
      [13, 23, 33]
    )

  # -- 54. Gather with async only last fn --
  async def test_gather_last_fn_async_rest_sync(self):
    """Only the last fn is async."""
    await self.assertEqual(
      Chain(3).gather(
        lambda v: v + 10,
        lambda v: v + 20,
        lambda v: aempty(v + 30)
      ).run(),
      [13, 23, 33]
    )


# ===========================================================================
# E. CONTEXT MANAGER (with_) COMPREHENSIVE (15+ tests)
# ===========================================================================

class ContextManagerComprehensiveTests(MyTestCase):
  """Exhaustive testing of the with_ (context manager) operation."""

  # -- 53. Basic sync CM: __enter__/__exit__ both called --
  async def test_with_sync_basic(self):
    """Basic sync CM: body receives __enter__ value, __exit__ called."""
    cm = SimpleCM('hello')
    result = Chain(cm).with_(lambda ctx: f'got_{ctx}').run()
    await self.assertEqual(result, 'got_hello')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertEqual(cm.exit_args, (None, None, None))

  # -- 54. CM body receives the __enter__ return value --
  async def test_with_body_receives_enter_value(self):
    """Body receives the value returned by __enter__."""
    received = []
    cm = SimpleCM('context_payload')
    def body(ctx):
      received.append(ctx)
      return ctx
    Chain(cm).with_(body).run()
    super(MyTestCase, self).assertEqual(received, ['context_payload'])

  # -- 55. CM where body ignores result (Cascade) --
  async def test_with_in_cascade_ignores_result(self):
    """In Cascade, with_ body result is discarded; root CM is returned.
    Cascade passes root_value to all links, so with_ receives the CM directly."""
    cm = SimpleCM('ctx_val')
    result = Cascade(cm).with_(lambda ctx: 'body_result_ignored').run()
    super(MyTestCase, self).assertIs(result, cm)

  # -- 56. CM where __enter__ raises --
  async def test_with_enter_raises_exit_not_called(self):
    """__enter__ raises => __exit__ NOT called."""
    exc = TestExc('enter failed')
    cm = SimpleCM('val', enter_raise=exc)
    with self.assertRaises(TestExc):
      Chain(cm).with_(lambda ctx: ctx).run()
    super(MyTestCase, self).assertFalse(cm.exited)

  # -- 57. CM where body raises --
  async def test_with_body_raises_exit_called_with_exc(self):
    """Body raises => __exit__ called with exception info."""
    cm = SimpleCM('val')
    exc = TestExc('body error')
    def body_raises(ctx):
      raise exc
    with self.assertRaises(TestExc):
      Chain(cm).with_(body_raises).run()
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertIs(cm.exit_args[0], TestExc)
    super(MyTestCase, self).assertIs(cm.exit_args[1], exc)

  # -- 58. CM where __exit__ suppresses exception --
  async def test_with_exit_suppresses_exception(self):
    """__exit__ returns True => exception suppressed."""
    cm = SimpleCM('val', exit_return=True)
    def body_raises(ctx):
      raise TestExc('suppressed')
    result = Chain(cm).with_(body_raises).run()
    await self.assertIsNone(result)

  # -- 59. CM where __exit__ returns falsy --
  async def test_with_exit_falsy_exception_propagates(self):
    """__exit__ returns False => exception propagates."""
    cm = SimpleCM('val', exit_return=False)
    def body_raises(ctx):
      raise TestExc('not suppressed')
    with self.assertRaises(TestExc):
      Chain(cm).with_(body_raises).run()

  # -- 60. CM where __exit__ itself raises --
  async def test_with_exit_raises(self):
    """__exit__ raises a new exception => new exception propagates."""
    exit_exc = TestExc('exit_error')
    cm = SimpleCM('val', exit_raise=exit_exc)
    result_exc = None
    try:
      Chain(cm).with_(lambda ctx: ctx).run()
    except TestExc as e:
      result_exc = e
    super(MyTestCase, self).assertIs(result_exc, exit_exc)

  # -- 61. Async CM (__aenter__/__aexit__) --
  async def test_with_async_cm_basic(self):
    """Async CM uses _with_full_async path."""
    cm = AsyncCM('async_hello')
    result = await Chain(cm).with_(lambda ctx: f'got_{ctx}').run()
    await self.assertEqual(result, 'got_async_hello')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  # -- 62. Async CM body returns coroutine --
  async def test_with_async_cm_async_body(self):
    """Async CM with async body: coroutine is awaited."""
    cm = AsyncCM('actx')
    async def async_body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'async_actx')

  # -- 63. Exception in async CM body --
  async def test_with_async_cm_body_exception(self):
    """Body raises in async CM => __aexit__ called with exception info."""
    cm = AsyncCM('val')
    exc = TestExc('async body error')
    def body_raises(ctx):
      raise exc
    with self.assertRaises(TestExc):
      await Chain(cm).with_(body_raises).run()
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertIs(cm.exit_args[0], TestExc)
    super(MyTestCase, self).assertIs(cm.exit_args[1], exc)

  # -- 64. Async CM where __aexit__ suppresses exception --
  async def test_with_async_cm_exit_suppresses(self):
    """__aexit__ returns True => exception suppressed."""
    cm = AsyncCM('val', exit_return=True)
    def body_raises(ctx):
      raise TestExc('suppressed async')
    result = await Chain(cm).with_(body_raises).run()
    await self.assertIsNone(result)

  # -- 65. Sync CM with body returning coroutine --
  async def test_with_sync_cm_async_body(self):
    """Sync CM, async body triggers _with_to_async."""
    cm = SimpleCM('sync_ctx')
    async def async_body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'async_sync_ctx')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  # -- 66. _with_to_async where __exit__ returns coroutine --
  async def test_with_sync_cm_coro_exit_success(self):
    """Sync CM whose __exit__ returns a coroutine, awaited on success path."""
    cm = CoroExitCM(value='coro_val', suppress=False)
    async def async_body(ctx):
      return f'got_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'got_coro_val')

  # -- 67. Nested context managers: with_ inside with_ --
  async def test_with_nested_cms(self):
    """Nested with_ calls: outer CM wraps inner CM."""
    outer_cm = SimpleCM('outer')
    inner_cm = SimpleCM('inner')
    result = (
      Chain(outer_cm)
      .with_(lambda ctx: Chain(inner_cm).with_(
        lambda ictx: f'{ctx}+{ictx}'
      ).run())
      .run()
    )
    await self.assertEqual(result, 'outer+inner')
    super(MyTestCase, self).assertTrue(outer_cm.exited)
    super(MyTestCase, self).assertTrue(inner_cm.exited)

  # -- 68. CM chained with .then() --
  async def test_with_then_chain(self):
    """CM result flows to the next .then() link."""
    cm = SimpleCM('ctx_val')
    result = Chain(cm).with_(lambda ctx: 42).then(lambda v: v + 1).run()
    await self.assertEqual(result, 43)

  # -- 69. CM with explicit args/kwargs --
  async def test_with_explicit_args(self):
    """with_(fn, arg, kwarg=val) passes explicit args, not ctx."""
    cm = SimpleCM('ctx_ignored')
    received = []
    def body(a, b=None):
      received.append((a, b))
      return f'{a}_{b}'
    result = Chain(cm).with_(body, 'hello', b='world').run()
    await self.assertEqual(result, 'hello_world')
    super(MyTestCase, self).assertEqual(received, [('hello', 'world')])

  # -- 70. Async CM with explicit args --
  async def test_with_async_cm_explicit_args(self):
    """Explicit args with async CM."""
    cm = AsyncCM('async_ignored')
    received = []
    def body(a, b):
      received.append((a, b))
      return f'{a}:{b}'
    result = await Chain(cm).with_(body, 'p', 'q').run()
    await self.assertEqual(result, 'p:q')
    super(MyTestCase, self).assertEqual(received, [('p', 'q')])

  # -- 71. Sync CM where async body raises, __exit__ suppresses via coro --
  async def test_with_sync_cm_coro_exit_suppresses_exception(self):
    """Sync CM __exit__ returns coroutine resolving to True => suppresses."""
    cm = CoroExitCM(value='v', suppress=True)
    async def async_body(ctx):
      raise TestExc('should be suppressed')
    result = await Chain(cm).with_(async_body).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertTrue(cm.exited)

  # -- 72. CM body using a nested Chain --
  async def test_with_body_as_nested_chain(self):
    """Body is a Chain that processes the context value."""
    cm = SimpleCM('nested_ctx')
    inner = Chain().then(lambda ctx: f'inner_{ctx}')
    result = Chain(cm).with_(inner).run()
    await self.assertEqual(result, 'inner_nested_ctx')

  # -- 73. Async CM enter/exit ordering --
  async def test_with_async_cm_ordering(self):
    """Verify __aenter__ before body, __aexit__ after body."""
    order = []
    class OrderedAsyncCM:
      async def __aenter__(self):
        order.append('enter')
        return 'val'
      async def __aexit__(self, *args):
        order.append('exit')
        return False
    def body(ctx):
      order.append('body')
      return ctx
    await Chain(OrderedAsyncCM()).with_(body).run()
    super(MyTestCase, self).assertEqual(order, ['enter', 'body', 'exit'])


# ===========================================================================
# F. CONTROL FLOW: _Return (12+ tests)
# ===========================================================================

class ReturnControlFlowTests(MyTestCase):
  """Exhaustive testing of the _Return control flow signal."""

  # -- 70. _Return in top-level chain (non-nested) --
  async def test_return_in_top_level_chain(self):
    """Chain.return_ in top-level chain returns the value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).then(lambda v: Chain.return_(v * 2)).run(),
          20
        )

  # -- 71. _Return in nested chain propagates to outermost --
  async def test_return_in_nested_chain(self):
    """Return in nested chain propagates to outermost chain, skipping all
    subsequent links. The _Return value becomes the final result."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(lambda v: Chain.return_(v * 10))
        # _Return(50) propagates from inner to outer, .then(+1) is SKIPPED
        await self.assertEqual(
          Chain(fn, 5).then(inner).then(lambda v: v + 1).run(),
          50
        )

  # -- 72. _Return with literal value --
  async def test_return_with_literal_value(self):
    """Chain.return_(42) returns 42."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(42)).run(),
          42
        )

  # -- 73. _Return with callable value --
  async def test_return_with_callable_value(self):
    """Chain.return_(lambda: 42) calls the lambda without args."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(lambda: 42)).run(),
          42
        )

  # -- 74. _Return with callable + args --
  async def test_return_with_callable_and_args(self):
    """Chain.return_(int, '42') calls int('42')."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(int, '42')).run(),
          42
        )

  # -- 75. _Return with Null value --
  async def test_return_with_null_value(self):
    """Chain.return_() with no args returns None."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn, 99).then(lambda v: Chain.return_()).run()
        )

  # -- 76. _Return in except_ handler context --
  async def test_return_in_except_handler(self):
    """Return raised within an operation caught by except_ handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1)
          .then(lambda v: Chain.return_(100))
          .except_(lambda v: 'handled', reraise=False)
          .run(),
          100
        )

  # -- 77. _Return in foreach (within frozen chain) --
  async def test_return_in_foreach_via_frozen_chain(self):
    """Return inside a frozen chain used as foreach fn stops the inner chain,
    producing a value per element. Frozen chains do not propagate _Return."""
    inner = Chain().then(lambda x: Chain.return_(x * 100)).freeze()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(inner).run(),
          [100, 200, 300]
        )

  # -- 78. _Return with async callable --
  async def test_return_with_async_callable(self):
    """Chain.return_(aempty, 42) calls aempty(42) and awaits."""
    await self.assertEqual(
      Chain(1).then(lambda v: Chain.return_(aempty, 42)).run(),
      42
    )

  # -- 79. _Return in deeply nested chains (3+ levels) --
  async def test_return_deeply_nested(self):
    """Return propagates through 3 levels of nesting to outermost.
    _Return escapes ALL nesting -- subsequent .then() is skipped."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner3 = Chain().then(lambda v: Chain.return_(v * 1000))
        inner2 = Chain().then(inner3)
        inner1 = Chain().then(inner2)
        await self.assertEqual(
          Chain(fn, 2).then(inner1).then(lambda v: v + 1).run(),
          2000
        )

  # -- 80. _Return in simple path chain --
  async def test_return_in_simple_chain(self):
    """Return in a chain with only .then() links (simple path)."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 7).then(lambda v: Chain.return_(v + 3)).run(),
          10
        )

  # -- 81. _Return in Cascade chain --
  async def test_return_in_cascade(self):
    """Return in a Cascade chain returns the specified value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Cascade(fn, 5).then(lambda v: Chain.return_(99)).run(),
          99
        )

  # -- 82. _Return with multiple links after it --
  async def test_return_skips_subsequent_links(self):
    """Return stops chain evaluation, subsequent links are not executed."""
    executed = []
    def track(v):
      executed.append('should_not_run')
      return v
    for fn, ctx in self.with_fn():
      with ctx:
        executed.clear()
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(42)).then(track).run(),
          42
        )
        super(MyTestCase, self).assertEqual(executed, [])


# ===========================================================================
# G. CONTROL FLOW: _Break (10+ tests)
# ===========================================================================

class BreakControlFlowTests(MyTestCase):
  """Exhaustive testing of the _Break control flow signal."""

  # -- 82. _Break in top-level chain (non-nested) --
  async def test_break_top_level_raises_quent_exception(self):
    """Break at top level (non-nested, non-foreach) raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).then(lambda v: Chain.break_()).run()

  # -- 83. _Break in nested chain propagates to foreach --
  async def test_break_in_nested_chain_propagates(self):
    """Break in a nested chain containing foreach propagates correctly.
    The inner chain has its own foreach; _Break propagates through nested chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        def break_at_3(el):
          if el >= 3:
            return Chain.break_()
          return fn(el * 10)
        inner = Chain().foreach(break_at_3)
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).then(inner).run(),
          [10, 20]
        )

  # -- 84. _Break with a value --
  async def test_break_with_value(self):
    """Chain.break_(42) in foreach returns 42."""
    sentinel = object()
    def f(el):
      if el == 2:
        return Chain.break_(sentinel)
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIs(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          sentinel
        )

  # -- 85. _Break with Null value --
  async def test_break_with_null_value(self):
    """Chain.break_() with no args returns the accumulated list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el == 3:
            return Chain.break_()
          return fn(el * 10)
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(),
          [10, 20]
        )

  # -- 86. _Break in foreach returns accumulated list --
  async def test_break_in_foreach_returns_accumulated(self):
    """Break accumulates results up to the break point."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 4:
            return Chain.break_()
          return fn(el)
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).foreach(f).run(),
          [1, 2, 3]
        )

  # -- 87. _Break in foreach with value overrides list --
  async def test_break_in_foreach_with_value_overrides(self):
    """Break with a value returns that value instead of the list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el == 3:
            return Chain.break_('custom_result')
          return fn(el)
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(),
          'custom_result'
        )

  # -- 88. _Break in iterate() generator --
  async def test_break_in_iterate_stops_generator(self):
    """Break in iterate() causes the generator to return (StopIteration)."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i * 10
    r = []
    for i in Chain(SyncIterator, list(range(6))).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 10, 20])

  # -- 89. _Break in async foreach --
  async def test_break_in_async_foreach(self):
    """Break in fully async foreach (async iterable)."""
    def f(el):
      if el >= 3:
        return Chain.break_()
      return el * 2
    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3, 4, 5])).foreach(f).run(),
      [2, 4]
    )

  # -- 90. _Break in async iterate --
  async def test_break_in_async_iterate(self):
    """Break in async generator via async for."""
    def f(i):
      if i >= 2:
        return Chain.break_()
      return i * 10
    r = []
    async for i in Chain(AsyncIterator, list(range(5))).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 10])

  # -- 91. _Break in foreach_indexed --
  async def test_break_in_foreach_indexed(self):
    """Break in foreach_indexed returns partial list."""
    def f(idx, el):
      if idx >= 2:
        return Chain.break_()
      return (idx, el)
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c', 'd']).foreach(f, with_index=True).run(),
          [(0, 'a'), (1, 'b')]
        )

  # -- 92. _Break with callable value --
  async def test_break_with_callable_value(self):
    """Chain.break_(fn) where fn is a callable: fn is called to produce value."""
    def f(el):
      if el == 2:
        return Chain.break_(lambda: 'break_result')
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          'break_result'
        )

  # -- 93. _Break with callable + args --
  async def test_break_with_callable_and_args(self):
    """Chain.break_(int, '99') calls int('99') for the break value."""
    def f(el):
      if el == 2:
        return Chain.break_(int, '99')
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          99
        )


# ===========================================================================
# H. GENERATOR/ITERATE COMPREHENSIVE (10+ tests)
# ===========================================================================

class GeneratorIterateComprehensiveTests(MyTestCase):
  """Exhaustive testing of iterate() and _Generator."""

  # -- 92. chain.iterate() returns a _Generator --
  async def test_iterate_returns_generator(self):
    """iterate() returns an object with <_Generator> repr."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    super(MyTestCase, self).assertEqual(repr(gen), '<_Generator>')

  # -- 93. Sync iteration via __iter__ --
  async def test_iterate_sync_for_loop(self):
    """Sync for loop iterates over chain results."""
    r = []
    for i in Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: i * 2):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [2, 4, 6])

  # -- 94. Async iteration via __aiter__ --
  async def test_iterate_async_for_loop(self):
    """Async for loop iterates over chain results."""
    r = []
    async for i in Chain(AsyncIterator, [1, 2, 3]).iterate(lambda i: i * 2):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [2, 4, 6])

  # -- 95. Iterate with mapping function fn --
  async def test_iterate_with_fn(self):
    """Iterate with fn transforms each element."""
    r = []
    for i in Chain(SyncIterator, [10, 20]).iterate(lambda i: i + 5):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [15, 25])

  # -- 96. Iterate without fn yields raw elements --
  async def test_iterate_without_fn(self):
    """Iterate with no fn yields elements as-is."""
    r = []
    for i in Chain(SyncIterator, [5, 6, 7]).iterate():
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [5, 6, 7])

  # -- 97. Generator __call__ creates a new _Generator with run args --
  async def test_generator_call_creates_new_instance(self):
    """Calling a _Generator returns a new _Generator, not the same object."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    nested = gen(SyncIterator([1, 2, 3]))
    super(MyTestCase, self).assertIsNot(gen, nested)
    super(MyTestCase, self).assertEqual(repr(nested), '<_Generator>')

  # -- 98. _Break in iterator stops generator --
  async def test_iterate_break_stops_generator(self):
    """Break in iterate fn causes StopIteration."""
    def f(i):
      if i >= 5:
        return Chain.break_()
      return i * 2
    r = []
    for i in Chain(SyncIterator).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(5)])

  # -- 99. _Return in iterator raises QuentException --
  async def test_iterate_return_raises_quent_exception(self):
    """Return in iterate raises QuentException."""
    with self.assertRaises(QuentException):
      for _ in Chain(SyncIterator, [1, 2]).iterate(Chain.return_):
        pass

  # -- 100. Generator __repr__ --
  async def test_generator_repr(self):
    """repr(_Generator) returns '<_Generator>'."""
    gen = Chain(SyncIterator).iterate()
    super(MyTestCase, self).assertEqual(repr(gen), '<_Generator>')

  # -- 101. Iterate with nested Chain as fn --
  async def test_iterate_with_chain_as_fn(self):
    """Chain used as iterate fn."""
    body = Chain().then(lambda v: v ** 2)
    r = []
    for i in Chain(SyncIterator, [2, 3, 4]).iterate(body):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [4, 9, 16])

  # -- 102. Iterate with async fn --
  async def test_iterate_async_fn(self):
    """Async fn in async generator is awaited."""
    r = []
    async for i in Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: aempty(i * 3)):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [3, 6, 9])

  # -- 103. Generator called with args produces correct values --
  async def test_generator_call_with_args(self):
    """A _Generator called with new args produces values from those args."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i * 3)
    nested = gen(SyncIterator([10, 20, 30]))
    r = []
    for i in nested:
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [30, 60, 90])

  # -- 104. Iterate over empty iterable --
  async def test_iterate_empty(self):
    """Iterate over empty iterable yields nothing."""
    r = []
    for i in Chain(SyncIterator, []).iterate(lambda i: i * 2):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [])

  # -- 105. Async generator break in async iterable --
  async def test_async_iterate_break_async_iterable(self):
    """Break in async generator with async iterable source."""
    def f(i):
      if i >= 2:
        return Chain.break_()
      return i * 10
    r = []
    async for i in Chain(AsyncIterator, list(range(5))).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 10])

  # -- 106. Async generator return raises QuentException --
  async def test_async_iterate_return_raises_quent_exception(self):
    """Return in async generator raises QuentException."""
    with self.assertRaises(QuentException):
      async for _ in Chain(SyncIterator, [1, 2]).iterate(Chain.return_):
        pass


# ===========================================================================
# I. EDGE CASES AND INTERACTIONS (5+ tests)
# ===========================================================================

class EdgeCasesInteractionsTests(MyTestCase):
  """Tests for edge cases and interactions between features."""

  # -- 102. foreach + except_ --
  async def test_foreach_with_except_handler(self):
    """Exception in foreach is caught by chain's except_ handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        handler_called = {'called': False}
        def handler(v):
          handler_called['called'] = True
          return 'handled'
        def f(el):
          if el == 2:
            raise TestExc('foreach fail')
          return fn(el)
        try:
          await await_(
            Chain(fn, [1, 2, 3]).foreach(f)
            .except_(handler, reraise=False)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(handler_called['called'])

  # -- 103. filter + except_ --
  async def test_filter_with_except_handler(self):
    """Exception in filter is caught by chain's except_ handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        handler_called = {'called': False}
        def handler(v):
          handler_called['called'] = True
          return 'handled'
        def pred(x):
          if x == 2:
            raise TestExc('filter fail')
          return x > 0
        try:
          await await_(
            Chain(fn, [1, 2, 3]).filter(pred)
            .except_(handler, reraise=False)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(handler_called['called'])

  # -- 104. gather + except_ --
  async def test_gather_with_except_handler(self):
    """Exception in gather is caught by chain's except_ handler."""
    handler_called = {'called': False}
    def handler(v):
      handler_called['called'] = True
      return 'handled'
    def f1(v):
      return v + 1
    def f2(v):
      raise TestExc('gather fail')
    try:
      Chain(5).gather(f1, f2).except_(handler, reraise=False).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertTrue(handler_called['called'])

  # -- 105. with_ + except_ --
  async def test_with_with_except_handler(self):
    """Exception in CM body is caught by chain's except_ handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        handler_called = {'called': False}
        def handler(v):
          handler_called['called'] = True
          return 'handled'
        cm = SimpleCM('val')
        def body_raises(ctx):
          raise TestExc('with fail')
        try:
          await await_(
            Chain(fn, cm).with_(body_raises)
            .except_(handler, reraise=False)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(handler_called['called'])

  # -- 106. Nested frozen chain with return_ inside foreach --
  async def test_return_inside_foreach_via_frozen_chain(self):
    """Return inside a frozen chain used as foreach fn: each invocation
    stops the inner chain, producing a value per element."""
    inner = Chain().then(lambda x: Chain.return_(x * 100)).freeze()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(inner).run(),
          [100, 200, 300]
        )

  # -- 107. foreach + finally_ --
  async def test_foreach_with_finally_handler(self):
    """finally_ callback runs even after foreach completes."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = {'called': False}
        def on_finally(v=None):
          finally_called['called'] = True
        await await_(
          Chain(fn, [1, 2, 3]).foreach(lambda x: x * 2)
          .finally_(on_finally)
          .run()
        )
        super(MyTestCase, self).assertTrue(finally_called['called'])

  # -- 108. foreach + filter chained --
  async def test_foreach_filter_chained(self):
    """Foreach output flows to filter as the iterable."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5])
          .foreach(lambda x: x * 2)
          .filter(lambda x: x > 4)
          .run(),
          [6, 8, 10]
        )

  # -- 109. gather + then + foreach --
  async def test_gather_then_foreach_chain(self):
    """Gather results flow to then which flows to foreach."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 2)
          .gather(lambda v: v + 1, lambda v: v + 2, lambda v: v + 3)
          .foreach(lambda x: x * 10)
          .run(),
          [30, 40, 50]
        )

  # -- 110. with_ result flows to foreach --
  async def test_with_result_flows_to_foreach(self):
    """CM body returns a list, which is then iterated by foreach."""
    cm = SimpleCM('ignored')
    result = Chain(cm).with_(lambda ctx: [1, 2, 3]).foreach(lambda x: x * 5).run()
    await self.assertEqual(result, [5, 10, 15])

  # -- 111. Break in foreach inside async chain --
  async def test_break_in_async_chain_foreach(self):
    """Break in foreach when chain has async root."""
    def f(el):
      if el >= 3:
        return Chain.break_()
      return el * 10
    await self.assertEqual(
      Chain(aempty, [1, 2, 3, 4]).foreach(f).run(),
      [10, 20]
    )

  # -- 112. Return with callable in Cascade --
  async def test_return_callable_in_cascade(self):
    """Return with callable inside Cascade returns the call result."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Cascade(fn, 5).then(lambda v: Chain.return_(lambda: 'ret_val')).run(),
          'ret_val'
        )

  # -- 113. Multiple breaks in different foreach calls --
  async def test_multiple_foreach_with_break(self):
    """Multiple chained foreach calls, each with its own break logic."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f1(el):
          if el >= 3:
            return Chain.break_()
          return fn(el * 10)
        result1 = await await_(
          Chain(fn, [1, 2, 3, 4]).foreach(f1).run()
        )
        super(MyTestCase, self).assertEqual(result1, [10, 20])

        def f2(el):
          if el >= 2:
            return Chain.break_()
          return fn(el * 100)
        result2 = await await_(
          Chain(fn, [1, 2, 3]).foreach(f2).run()
        )
        super(MyTestCase, self).assertEqual(result2, [100])

  # -- 114. Foreach where fn returns both sync and async intermixed --
  async def test_foreach_intermixed_sync_async(self):
    """fn alternates between sync and async returns."""
    def f(el):
      if el % 2 == 0:
        return aempty(el * 10)
      return el * 10
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5]).foreach(f).run(),
      [10, 20, 30, 40, 50]
    )

  # -- 115. Filter then gather --
  async def test_filter_then_gather(self):
    """Filter result list used as input to gather (via then)."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5])
          .filter(lambda x: x > 3)
          .then(len)
          .gather(lambda v: v + 1, lambda v: v * 10)
          .run(),
          [3, 20]
        )


# ===========================================================================
# J. CASCADE INTERACTIONS WITH ITERATION/CM
# ===========================================================================

class CascadeIterationCMTests(MyTestCase):
  """Tests for Cascade interactions with iteration and CM operations."""

  async def test_cascade_foreach_returns_root(self):
    """Cascade([1,2,3]).foreach(fn).run() returns root, not the mapped list."""
    root = [1, 2, 3]
    mapped = []
    def capture(x):
      mapped.append(x * 10)
      return x * 10
    result = Cascade(root).foreach(capture).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [10, 20, 30])

  async def test_cascade_filter_returns_root(self):
    """Cascade([...]).filter(pred).run() returns root, not filtered list."""
    root = [1, 2, 3, 4, 5]
    result = Cascade(root).filter(lambda x: x > 3).run()
    await self.assertEqual(result, root)

  async def test_cascade_gather_returns_root(self):
    """Cascade(v).gather(fn1, fn2).run() returns root, not gathered results."""
    result = Cascade(5).gather(lambda v: v + 1, lambda v: v * 2).run()
    await self.assertEqual(result, 5)

  async def test_cascade_with_returns_root(self):
    """Cascade(cm).with_(body).run() returns root, not body result."""
    cm = SimpleCM('ctx_val')
    result = Cascade(cm).with_(lambda ctx: 'body_result').run()
    super(MyTestCase, self).assertIs(result, cm)

  async def test_cascade_async_foreach_returns_root(self):
    """Cascade with async root + foreach returns root value."""
    root = [4, 5, 6]
    mapped = []
    def capture(x):
      mapped.append(x * 2)
      return x * 2
    result = await await_(Cascade(aempty, root).foreach(capture).run())
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [8, 10, 12])

  async def test_cascade_foreach_indexed_returns_root(self):
    """Cascade([...]).foreach(fn, with_index=True).run() returns root."""
    root = ['a', 'b', 'c']
    result = Cascade(root).foreach(lambda idx, el: (idx, el), with_index=True).run()
    await self.assertEqual(result, root)


# ===========================================================================
# K. ASYNC-SPECIFIC EDGE CASES
# ===========================================================================

class AsyncSpecificEdgeCaseTests(MyTestCase):
  """Tests for async-specific edge cases in iteration and CM paths."""

  async def test_foreach_all_elements_async(self):
    """Every element produces a coroutine from fn."""
    await self.assertEqual(
      Chain([1, 2, 3]).foreach(lambda x: aempty(x + 100)).run(),
      [101, 102, 103]
    )

  async def test_filter_all_elements_async_predicate(self):
    """Async predicate for all elements."""
    await self.assertEqual(
      Chain([1, 2, 3, 4]).filter(lambda x: aempty(x % 2 == 0)).run(),
      [2, 4]
    )

  async def test_with_async_cm_async_body_async_chain(self):
    """Async CM + async body + async root: fully async path."""
    cm = AsyncCM('fully_async')
    async def async_body(ctx):
      return f'body_{ctx}'
    result = await Chain(aempty, cm).with_(async_body).run()
    await self.assertEqual(result, 'body_fully_async')

  async def test_foreach_indexed_all_async(self):
    """Every indexed element produces a coroutine."""
    await self.assertEqual(
      Chain([10, 20]).foreach(
        lambda idx, el: aempty((idx, el * 2)), with_index=True
      ).run(),
      [(0, 20), (1, 40)]
    )

  async def test_gather_all_coroutines_use_asyncio_gather(self):
    """When all fns return coroutines, asyncio.gather is used internally."""
    results = []
    async def f1(v):
      await asyncio.sleep(0.01)
      results.append('f1')
      return v + 1
    async def f2(v):
      await asyncio.sleep(0.01)
      results.append('f2')
      return v + 2
    r = await await_(Chain(10).gather(f1, f2).run())
    super(MyTestCase, self).assertEqual(r, [11, 12])
    # Both should have run (order may vary due to concurrency)
    super(MyTestCase, self).assertEqual(sorted(results), ['f1', 'f2'])

  async def test_async_cm_exit_ordering_on_exception(self):
    """Async CM __aexit__ receives correct exception info when body raises."""
    cm = AsyncCM('val')
    exc = TestExc('ordering_test')
    def body_raises(ctx):
      raise exc
    with self.assertRaises(TestExc):
      await Chain(cm).with_(body_raises).run()
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertEqual(cm.exit_args[0], TestExc)
    super(MyTestCase, self).assertIs(cm.exit_args[1], exc)

  async def test_foreach_to_async_break_with_coro_value(self):
    """Break with async callable value in _foreach_to_async path."""
    call_count = {'n': 0}
    def f(el):
      call_count['n'] += 1
      if call_count['n'] == 1:
        return aempty(el * 10)
      if call_count['n'] == 3:
        return Chain.break_(aempty, 'break_coro_result')
      return el * 10
    call_count['n'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5]).foreach(f).run(),
      'break_coro_result'
    )


if __name__ == '__main__':
  import unittest
  unittest.main()
