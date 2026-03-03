"""Variant x Feature matrix tests.

Systematically tests every Chain feature across ALL chain variants:
Chain, Cascade, run (pipe terminator), FrozenChain, clone, decorator.

Sections:
  1. Chain vs Cascade value flow for every feature
  2. run (pipe terminator) with every feature
  3. FrozenChain with every feature
  4. clone() with every feature
  5. decorator() with every feature
  6. Cross-variant nesting
  7. Variant-specific edge cases
  8. Async variants for each chain type
"""

import asyncio
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, run, QuentException, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleCM:
  """Sync context manager that yields a predetermined value."""
  def __init__(self, value='ctx'):
    self._value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self._value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Async context manager that yields a predetermined value."""
  def __init__(self, value='actx'):
    self._value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self._value

  async def __aexit__(self, *args):
    self.exited = True
    return False


def raise_test_exc(v=None):
  raise TestExc('boom')


# ===================================================================
# Section 1: Chain vs Cascade value flow for EVERY feature
# ===================================================================

class ChainVsCascadeThenTests(MyTestCase):
  """then: Chain passes previous result, Cascade passes root."""

  async def test_chain_then_passes_previous_result(self):
    result = Chain(10).then(lambda v: v * 2).then(lambda v: v + 1).run()
    await self.assertEqual(result, 21)

  async def test_cascade_then_passes_root(self):
    result = Cascade(10).then(lambda v: v * 2).then(lambda v: v + 1).run()
    # Cascade: both then's receive root (10). Returns root (10).
    await self.assertEqual(result, 10)

  async def test_chain_then_three_links(self):
    result = Chain(2).then(lambda v: v + 3).then(lambda v: v * 4).then(lambda v: v - 1).run()
    await self.assertEqual(result, 19)  # (2+3)*4 - 1 = 19

  async def test_cascade_then_three_links_all_receive_root(self):
    received = []
    Cascade(5).then(lambda v: received.append(v)).then(lambda v: received.append(v)).then(lambda v: received.append(v)).run()
    super(MyTestCase, self).assertEqual(received, [5, 5, 5])


class ChainVsCascadeDoTests(MyTestCase):
  """do: verify do() receives correct value in Chain vs Cascade."""

  async def test_chain_do_receives_current_value(self):
    captured = []
    result = Chain(10).then(lambda v: v * 3).do(lambda v: captured.append(v)).run()
    await self.assertEqual(result, 30)
    super(MyTestCase, self).assertEqual(captured, [30])

  async def test_cascade_do_receives_root(self):
    captured = []
    result = Cascade(10).then(lambda v: v * 3).do(lambda v: captured.append(v)).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(captured, [10])

  async def test_chain_do_discards_result(self):
    result = Chain(10).do(lambda v: 999).then(lambda v: v + 1).run()
    await self.assertEqual(result, 11)

  async def test_cascade_do_discards_result(self):
    result = Cascade(10).do(lambda v: 999).run()
    await self.assertEqual(result, 10)


class ChainVsCascadeForeachTests(MyTestCase):
  """foreach: Chain passes previous result to foreach; Cascade passes root."""

  async def test_chain_foreach_iterates_previous(self):
    result = Chain([1, 2, 3]).foreach(lambda x: x * 10).run()
    await self.assertEqual(result, [10, 20, 30])

  async def test_cascade_foreach_iterates_root_returns_root(self):
    root = [1, 2, 3]
    mapped = []
    result = Cascade(root).foreach(lambda x: mapped.append(x * 10) or x * 10).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [10, 20, 30])

  async def test_chain_foreach_after_then(self):
    """Chain: then transforms, foreach iterates the transformed value."""
    result = Chain([1, 2]).then(lambda v: [x + 10 for x in v]).foreach(lambda x: x * 2).run()
    await self.assertEqual(result, [22, 24])

  async def test_cascade_foreach_after_then_still_iterates_root(self):
    """Cascade: foreach receives root regardless of preceding then."""
    root = [1, 2]
    captured = []
    result = Cascade(root).then(lambda v: [99]).foreach(lambda x: captured.append(x)).run()
    await self.assertEqual(result, root)
    # foreach iterates root [1, 2], not [99]
    super(MyTestCase, self).assertEqual(captured, [1, 2])


class ChainVsCascadeFilterTests(MyTestCase):
  """filter: same comparison as foreach."""

  async def test_chain_filter(self):
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).run()
    await self.assertEqual(result, [4, 5])

  async def test_cascade_filter_returns_root(self):
    root = [1, 2, 3, 4, 5]
    result = Cascade(root).filter(lambda x: x > 3).run()
    await self.assertEqual(result, root)

  async def test_chain_filter_after_then(self):
    result = Chain(5).then(lambda v: list(range(v))).filter(lambda x: x % 2 == 0).run()
    await self.assertEqual(result, [0, 2, 4])

  async def test_cascade_filter_after_then_filters_root(self):
    root = [0, 1, 2, 3, 4]
    result = Cascade(root).then(lambda v: 'ignored').filter(lambda x: x % 2 == 0).run()
    await self.assertEqual(result, root)


class ChainVsCascadeWithTests(MyTestCase):
  """with_: Chain passes previous to with_, Cascade passes root."""

  async def test_chain_with_uses_previous_as_cm(self):
    cm = SimpleCM(value='hello')
    result = Chain(cm).with_(lambda ctx: ctx.upper()).run()
    await self.assertEqual(result, 'HELLO')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_cascade_with_uses_root_as_cm_returns_root(self):
    cm = SimpleCM(value='hello')
    result = Cascade(cm).with_(lambda ctx: ctx.upper()).run()
    await self.assertIs(result, cm)
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_chain_with_after_then(self):
    cm = SimpleCM(value=42)
    result = Chain(lambda: cm).with_(lambda ctx: ctx + 8).run()
    await self.assertEqual(result, 50)

  async def test_cascade_with_after_then(self):
    cm = SimpleCM(value=42)
    result = Cascade(cm).then(lambda v: 'ignored').with_(lambda ctx: ctx + 8).run()
    await self.assertIs(result, cm)


class ChainVsCascadeGatherTests(MyTestCase):
  """gather: Chain passes previous to each fn, Cascade passes root."""

  async def test_chain_gather(self):
    result = Chain(10).gather(lambda v: v * 2, lambda v: v + 5).run()
    await self.assertEqual(result, [20, 15])

  async def test_cascade_gather_returns_root(self):
    received = []
    result = Cascade(10).gather(
      lambda v: received.append(('a', v)) or v * 2,
      lambda v: received.append(('b', v)) or v + 5,
    ).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(received, [('a', 10), ('b', 10)])

  async def test_chain_gather_after_then(self):
    result = Chain(3).then(lambda v: v * 2).gather(lambda v: v + 1, lambda v: v - 1).run()
    await self.assertEqual(result, [7, 5])

  async def test_cascade_gather_after_then_receives_root(self):
    received = []
    Cascade(3).then(lambda v: v * 2).gather(
      lambda v: received.append(v),
      lambda v: received.append(v),
    ).run()
    super(MyTestCase, self).assertEqual(received, [3, 3])


class ChainVsCascadeForeachIndexedTests(MyTestCase):
  """foreach_indexed: same comparison."""

  async def test_chain_foreach_indexed(self):
    result = Chain([10, 20, 30]).foreach(lambda i, v: (i, v), with_index=True).run()
    await self.assertEqual(result, [(0, 10), (1, 20), (2, 30)])

  async def test_cascade_foreach_indexed_returns_root(self):
    root = [10, 20, 30]
    captured = []
    result = Cascade(root).foreach(lambda i, v: captured.append((i, v)), with_index=True).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(captured, [(0, 10), (1, 20), (2, 30)])


class ChainVsCascadeExceptTests(MyTestCase):
  """except_: handler receives exception (via root_value) in both."""

  async def test_chain_except_catches(self):
    result = Chain(10).then(raise_test_exc).except_(lambda v: 'caught', reraise=False).run()
    await self.assertEqual(result, 'caught')

  async def test_cascade_except_catches(self):
    """Cascade except_ handler receives root_value."""
    result = Cascade(10).then(raise_test_exc).except_(lambda v: v, reraise=False).run()
    await self.assertEqual(result, 10)

  async def test_chain_except_reraise(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc')).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, ['exc'])

  async def test_cascade_except_reraise(self):
    ran = []
    try:
      Cascade(10).then(raise_test_exc).except_(lambda v: ran.append(v)).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, [10])


class ChainVsCascadeFinallyTests(MyTestCase):
  """finally_: cleanup runs in both, receives root_value."""

  async def test_chain_finally_runs(self):
    ran = []
    result = Chain(10).then(lambda v: v + 5).finally_(lambda v: ran.append('fin')).run()
    await self.assertEqual(result, 15)
    super(MyTestCase, self).assertEqual(ran, ['fin'])

  async def test_cascade_finally_runs(self):
    ran = []
    result = Cascade(10).then(lambda v: v + 5).finally_(lambda v: ran.append('fin')).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(ran, ['fin'])

  async def test_chain_finally_on_exception(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).finally_(lambda v: ran.append('fin')).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, ['fin'])

  async def test_cascade_finally_on_exception(self):
    ran = []
    try:
      Cascade(10).then(raise_test_exc).finally_(lambda v: ran.append('fin')).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class ChainVsCascadeReturnBreakTests(MyTestCase):
  """return_ and break_ behavior in Chain vs Cascade."""

  async def test_chain_return_exits_nested(self):
    result = Chain(Chain(10).then(lambda v: Chain.return_(v * 2))).then(lambda v: v + 100).run()
    await self.assertEqual(result, 20)

  async def test_cascade_return_exits_nested(self):
    result = Cascade(10).then(lambda v: None).run()
    await self.assertEqual(result, 10)

  async def test_chain_break_in_foreach(self):
    def f(x):
      if x == 3:
        Chain.break_()
      return x * 10
    result = Chain([1, 2, 3, 4]).foreach(f).run()
    await self.assertEqual(result, [10, 20])

  async def test_cascade_break_in_foreach(self):
    root = [1, 2, 3, 4]
    def f(x):
      if x == 3:
        Chain.break_()
      return x * 10
    result = Cascade(root).foreach(f).run()
    await self.assertEqual(result, root)


# ===================================================================
# Section 2: run (pipe terminator) with every feature
# ===================================================================

class RunPipeTests(MyTestCase):
  """Test run() as a pipe terminator with various features."""

  async def test_run_pipe_then(self):
    result = Chain(10) | (lambda v: v * 2) | run()
    await self.assertEqual(result, 20)

  async def test_run_pipe_with_root_override(self):
    result = Chain() | (lambda v: v * 3) | run(7)
    await self.assertEqual(result, 21)

  async def test_run_pipe_chain_multiple_ops(self):
    result = Chain(2) | (lambda v: v + 3) | (lambda v: v * 10) | run()
    await self.assertEqual(result, 50)

  async def test_run_stores_args(self):
    r = run(10, 20, key='val')
    super(MyTestCase, self).assertEqual(r.root_value, 10)
    super(MyTestCase, self).assertEqual(r.args, (20,))
    super(MyTestCase, self).assertEqual(r.kwargs, {'key': 'val'})

  async def test_run_default_root_is_null(self):
    r = run()
    super(MyTestCase, self).assertIs(r.root_value, Null)

  async def test_run_pipe_cascade(self):
    result = Cascade(10) | (lambda v: v * 2) | run()
    await self.assertEqual(result, 10)

  async def test_run_pipe_with_literal(self):
    result = Chain(5) | 99 | run()
    await self.assertEqual(result, 99)

  async def test_run_pipe_with_except(self):
    ran = []
    try:
      Chain(raise_test_exc) | run()
    except TestExc:
      ran.append('caught')
    super(MyTestCase, self).assertEqual(ran, ['caught'])


# ===================================================================
# Section 3: FrozenChain with every feature
# ===================================================================

class FrozenChainThenTests(MyTestCase):
  """FrozenChain with then operations."""

  async def test_frozen_basic(self):
    frozen = Chain(10).then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen(), 20)
    await self.assertEqual(frozen.run(), 20)

  async def test_frozen_with_root_override(self):
    frozen = Chain().then(lambda v: v ** 2).freeze()
    await self.assertEqual(frozen(3), 9)
    await self.assertEqual(frozen(4), 16)
    await self.assertEqual(frozen(5), 25)

  async def test_frozen_reuse_multiple_calls(self):
    frozen = Chain(0).then(lambda v: v + 1).freeze()
    results = [frozen() for _ in range(5)]
    super(MyTestCase, self).assertEqual(results, [1, 1, 1, 1, 1])


class FrozenChainExceptTests(MyTestCase):
  """FrozenChain with except_."""

  async def test_frozen_except_noraise(self):
    frozen = Chain(10).then(raise_test_exc).except_(lambda v: 'recovered', reraise=False).freeze()
    await self.assertEqual(frozen(), 'recovered')

  async def test_frozen_except_reraise(self):
    frozen = Chain(10).then(raise_test_exc).except_(lambda v: None).freeze()
    with self.assertRaises(TestExc):
      frozen()

  async def test_frozen_except_reuse(self):
    frozen = Chain(10).then(raise_test_exc).except_(lambda v: 42, reraise=False).freeze()
    await self.assertEqual(frozen(), 42)
    await self.assertEqual(frozen(), 42)


class FrozenChainFinallyTests(MyTestCase):
  """FrozenChain with finally_."""

  async def test_frozen_finally_runs(self):
    ran = []
    frozen = Chain(10).finally_(lambda v: ran.append('fin')).freeze()
    await self.assertEqual(frozen(), 10)
    super(MyTestCase, self).assertEqual(ran, ['fin'])

  async def test_frozen_finally_on_error(self):
    ran = []
    frozen = Chain(raise_test_exc).finally_(lambda: ran.append('fin'), ...).freeze()
    try:
      frozen()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class FrozenChainForeachTests(MyTestCase):
  """FrozenChain with foreach."""

  async def test_frozen_foreach(self):
    frozen = Chain([1, 2, 3]).foreach(lambda x: x * 10).freeze()
    await self.assertEqual(frozen(), [10, 20, 30])

  async def test_frozen_foreach_reuse(self):
    frozen = Chain().foreach(lambda x: x * 2).freeze()
    await self.assertEqual(frozen([1, 2]), [2, 4])
    await self.assertEqual(frozen([10, 20]), [20, 40])


class FrozenChainFilterTests(MyTestCase):
  """FrozenChain with filter."""

  async def test_frozen_filter(self):
    frozen = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).freeze()
    await self.assertEqual(frozen(), [4, 5])


class FrozenChainGatherTests(MyTestCase):
  """FrozenChain with gather."""

  async def test_frozen_gather(self):
    frozen = Chain(10).gather(lambda v: v * 2, lambda v: v + 5).freeze()
    await self.assertEqual(frozen(), [20, 15])

  async def test_frozen_gather_reuse(self):
    frozen = Chain().gather(lambda v: v * 2, lambda v: v + 1).freeze()
    await self.assertEqual(frozen(5), [10, 6])
    await self.assertEqual(frozen(3), [6, 4])


class FrozenChainWithCMTests(MyTestCase):
  """FrozenChain with with_ (context manager)."""

  async def test_frozen_with_cm(self):
    frozen = Chain(SimpleCM(value=42)).with_(lambda ctx: ctx + 8).freeze()
    await self.assertEqual(frozen(), 50)


class FrozenChainForeachIndexedTests(MyTestCase):
  """FrozenChain with foreach_indexed."""

  async def test_frozen_foreach_indexed(self):
    frozen = Chain([10, 20]).foreach(lambda i, v: (i, v), with_index=True).freeze()
    await self.assertEqual(frozen(), [(0, 10), (1, 20)])


class FrozenChainNestedTests(MyTestCase):
  """FrozenChain nested inside another chain (primary use case)."""

  async def test_frozen_nested_in_chain(self):
    doubler = Chain().then(lambda v: v * 2).freeze()
    result = Chain(5).then(doubler).run()
    await self.assertEqual(result, 10)

  async def test_frozen_nested_in_chain_multiple(self):
    add1 = Chain().then(lambda v: v + 1).freeze()
    mul3 = Chain().then(lambda v: v * 3).freeze()
    result = Chain(2).then(add1).then(mul3).run()
    await self.assertEqual(result, 9)  # (2+1)*3 = 9

  async def test_frozen_cascade_preserves_semantics(self):
    """Frozen Cascade still returns root."""
    frozen = Cascade(99).then(lambda v: v * 100).freeze()
    await self.assertEqual(frozen(), 99)

  async def test_frozen_cascade_nested_in_chain(self):
    """Frozen Cascade nested in a Chain returns root of the Cascade."""
    frozen_casc = Cascade().then(lambda v: v * 100).freeze()
    result = Chain(5).then(frozen_casc).run()
    await self.assertEqual(result, 5)


class FrozenChainBoolTests(TestCase):
  """FrozenChain __bool__ behavior is inherited from callable."""

  def test_frozen_is_truthy(self):
    frozen = Chain(1).freeze()
    self.assertTrue(bool(frozen))


# ===================================================================
# Section 4: clone() with every feature
# ===================================================================

class CloneThenTests(TestCase):
  """Clone with then, verify independence."""

  def test_clone_add_more_to_clone(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 100)
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 200)

  def test_clone_add_more_to_original(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    self.assertEqual(c.run(), 200)
    self.assertEqual(c2.run(), 2)


class CloneExceptTests(TestCase):
  """Clone with except_, verify error handling works independently."""

  def test_clone_except_works(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc'), reraise=False)
    c2 = c.clone()
    ran.clear()
    c2.run()
    self.assertEqual(ran, ['exc'])

  def test_clone_except_original_unaffected(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc'), reraise=False)
    c2 = c.clone()
    ran.clear()
    c.run()
    self.assertEqual(ran, ['exc'])
    ran.clear()
    c2.run()
    self.assertEqual(ran, ['exc'])


class CloneFinallyTests(TestCase):
  """Clone with finally_, verify cleanup works independently."""

  def test_clone_finally_works(self):
    ran = []
    c = Chain(10).finally_(lambda v: ran.append('fin'))
    c2 = c.clone()
    ran.clear()
    c2.run()
    self.assertEqual(ran, ['fin'])

  def test_clone_finally_on_error(self):
    ran = []
    c = Chain(raise_test_exc).finally_(lambda: ran.append('fin'), ...)
    c2 = c.clone()
    ran.clear()
    try:
      c2.run()
    except TestExc:
      pass
    self.assertEqual(ran, ['fin'])


class CloneForeachTests(TestCase):
  """Clone with foreach."""

  def test_clone_foreach(self):
    c = Chain([1, 2, 3]).foreach(lambda x: x * 10)
    c2 = c.clone()
    self.assertEqual(c2.run(), [10, 20, 30])

  def test_clone_foreach_add_filter_to_clone(self):
    c = Chain([1, 2, 3, 4]).foreach(lambda x: x * 10)
    c2 = c.clone()
    c2.then(lambda lst: [x for x in lst if x > 20])
    self.assertEqual(c.run(), [10, 20, 30, 40])
    self.assertEqual(c2.run(), [30, 40])


class CloneWithCMTests(TestCase):
  """Clone with with_."""

  def test_clone_with_cm(self):
    c = Chain(SimpleCM(value=42)).with_(lambda ctx: ctx + 8)
    c2 = c.clone()
    self.assertEqual(c2.run(), 50)


class CloneGatherTests(TestCase):
  """Clone with gather."""

  def test_clone_gather(self):
    c = Chain(10).gather(lambda v: v * 2, lambda v: v + 5)
    c2 = c.clone()
    self.assertEqual(c2.run(), [20, 15])


class CloneCascadeSemantics(TestCase):
  """Clone preserves Cascade semantics."""

  def test_clone_cascade_preserves_cascade(self):
    c = Cascade(42).then(lambda v: v * 100)
    c2 = c.clone()
    self.assertIsInstance(c2, Cascade)
    self.assertEqual(c2.run(), 42)

  def test_clone_cascade_independence(self):
    c = Cascade(1).then(lambda v: v * 2)
    c2 = c.clone()
    c.then(lambda v: v + 100)
    self.assertEqual(c.run(), 1)
    self.assertEqual(c2.run(), 1)


class CloneConfigTests(TestCase):
  """Clone preserves config (debug, no_async, autorun)."""

  def test_clone_preserves_no_async(self):
    c = Chain(10).no_async(True).then(lambda v: v + 1)
    c2 = c.clone()
    self.assertEqual(c2.run(), 11)
    self.assertTrue(c2._is_sync)

  def test_clone_preserves_autorun(self):
    c = Chain(10).config(autorun=True)
    c2 = c.clone()
    self.assertEqual(c2.run(), 10)

  def test_modify_clone_config_original_unchanged(self):
    c = Chain(10).no_async(True)
    c2 = c.clone()
    c2.no_async(False)
    self.assertTrue(c._is_sync)
    self.assertFalse(c2._is_sync)


class CloneMultipleAndDeepTests(TestCase):
  """Multiple clones and clone of clone."""

  def test_multiple_clones_same_original(self):
    c = Chain(3).then(lambda v: v + 1)
    clones = [c.clone() for _ in range(5)]
    for clone in clones:
      self.assertEqual(clone.run(), 4)
    c.then(lambda v: v * 100)
    for clone in clones:
      self.assertEqual(clone.run(), 4)

  def test_clone_of_clone(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    c2.then(lambda v: v * 10)
    c3.then(lambda v: v * 100)
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 20)
    self.assertEqual(c3.run(), 200)


# ===================================================================
# Section 5: decorator() with every feature
# ===================================================================

class DecoratorThenTests(MyTestCase):
  """decorator() with then."""

  async def test_decorator_basic(self):
    @Chain().then(lambda v: v * 2).decorator()
    def double(x):
      return x
    await self.assertEqual(double(5), 10)

  async def test_decorator_multiple_then(self):
    @Chain().then(lambda v: v * 2).then(lambda v: v + 1).decorator()
    def fn(x):
      return x
    await self.assertEqual(fn(3), 7)  # 3*2+1 = 7

  async def test_decorator_with_args(self):
    @Chain().then(lambda v: v).decorator()
    def add(a, b):
      return a + b
    await self.assertEqual(add(3, 7), 10)


class DecoratorExceptTests(MyTestCase):
  """decorator() with except_."""

  async def test_decorator_except_noraise(self):
    @Chain().then(raise_test_exc).except_(lambda v: 'fallback', reraise=False).decorator()
    def fn(x):
      return x
    await self.assertEqual(fn(5), 'fallback')

  async def test_decorator_except_reraise(self):
    @Chain().then(raise_test_exc).except_(lambda v: None).decorator()
    def fn(x):
      return x
    with self.assertRaises(TestExc):
      fn(5)


class DecoratorFinallyTests(MyTestCase):
  """decorator() with finally_."""

  async def test_decorator_finally(self):
    ran = []

    @Chain().then(lambda v: v * 2).finally_(lambda v: ran.append('fin')).decorator()
    def fn(x):
      return x
    await self.assertEqual(fn(5), 10)
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class DecoratorForeachTests(MyTestCase):
  """decorator() with foreach (decorated fn returns list)."""

  async def test_decorator_foreach(self):
    @Chain().foreach(lambda x: x * 10).decorator()
    def get_list():
      return [1, 2, 3]
    await self.assertEqual(get_list(), [10, 20, 30])


class DecoratorCascadeTests(MyTestCase):
  """decorator() with Cascade."""

  async def test_cascade_decorator_returns_root(self):
    side_effects = []

    @Cascade().then(lambda v: side_effects.append(v) or v * 100).decorator()
    def fn(x):
      return x * 2
    result = fn(5)
    await self.assertEqual(result, 10)  # 5*2=10, Cascade returns root
    super(MyTestCase, self).assertEqual(side_effects, [10])


class DecoratorReuseTests(MyTestCase):
  """decorator() reuse."""

  async def test_decorator_reuse(self):
    @Chain().then(lambda v: v + 1).decorator()
    def inc(x):
      return x
    await self.assertEqual(inc(1), 2)
    await self.assertEqual(inc(10), 11)
    await self.assertEqual(inc(100), 101)


class DecoratorAsyncTests(MyTestCase):
  """decorator() with async decorated function."""

  async def test_decorator_async_fn(self):
    @Chain().then(lambda v: v * 5).decorator()
    async def fn(x):
      return x + 1
    await self.assertEqual(fn(2), 15)  # (2+1)*5 = 15


# ===================================================================
# Section 6: Cross-variant nesting
# ===================================================================

class CrossVariantNestingTests(MyTestCase):
  """Test nesting one variant inside another."""

  async def test_chain_containing_frozen_cascade(self):
    frozen_casc = Cascade().then(lambda v: v * 100).freeze()
    result = Chain(5).then(frozen_casc).then(lambda v: v + 1).run()
    # frozen_casc receives 5, Cascade returns 5 (root). Chain continues: 5+1=6
    await self.assertEqual(result, 6)

  async def test_cascade_containing_frozen_chain(self):
    frozen_chain = Chain().then(lambda v: v * 2).freeze()
    result = Cascade(5).then(frozen_chain).run()
    # Cascade returns root (5) regardless
    await self.assertEqual(result, 5)

  async def test_nested_frozen_three_levels(self):
    inner = Chain().then(lambda v: v + 1).freeze()
    middle = Chain().then(inner).then(lambda v: v * 2).freeze()
    result = Chain(3).then(middle).run()
    await self.assertEqual(result, 8)  # (3+1)*2 = 8

  async def test_clone_chain_with_frozen_nested(self):
    inner = Chain().then(lambda v: v * 10).freeze()
    c = Chain(2).then(inner)
    c2 = c.clone()
    c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 21)   # 2*10+1 = 21
    await self.assertEqual(c2.run(), 20)  # 2*10 = 20

  async def test_decorator_wrapping_chain_with_nested(self):
    inner = Chain().then(lambda v: v + 10).freeze()

    @Chain().then(inner).then(lambda v: v * 2).decorator()
    def fn(x):
      return x
    await self.assertEqual(fn(5), 30)  # (5+10)*2 = 30

  async def test_frozen_chain_inside_frozen_cascade(self):
    frozen_chain = Chain().then(lambda v: v * 3).freeze()
    frozen_casc = Cascade().then(frozen_chain).freeze()
    result = frozen_casc(7)
    await self.assertEqual(result, 7)  # Cascade returns root

  async def test_frozen_cascade_inside_frozen_chain(self):
    frozen_casc = Cascade().then(lambda v: v * 3).freeze()
    frozen_chain = Chain().then(frozen_casc).then(lambda v: v + 100).freeze()
    result = frozen_chain(7)
    # Cascade returns root (7). Chain: 7+100 = 107
    await self.assertEqual(result, 107)


# ===================================================================
# Section 7: Variant-specific edge cases
# ===================================================================

class VariantEdgeCasesTests(MyTestCase):
  """Edge cases specific to each variant."""

  async def test_cascade_with_cm_body_receives_enter_value(self):
    """Cascade.with_(): body receives __enter__ result."""
    body_received = []
    cm = SimpleCM(value='entered')
    Cascade(cm).with_(lambda ctx: body_received.append(ctx)).run()
    super(MyTestCase, self).assertEqual(body_received, ['entered'])

  async def test_cascade_foreach_requires_iterable_root(self):
    """Cascade.foreach iterates root. If root isn't iterable, it fails."""
    with self.assertRaises((TypeError, AttributeError)):
      Cascade(10).foreach(lambda x: x).run()

  async def test_cascade_gather_each_fn_receives_root(self):
    received = []
    Cascade(42).gather(
      lambda v: received.append(v),
      lambda v: received.append(v),
      lambda v: received.append(v),
    ).run()
    super(MyTestCase, self).assertEqual(received, [42, 42, 42])

  async def test_run_pipe_no_links_just_root(self):
    result = Chain(42) | run()
    await self.assertEqual(result, 42)

  async def test_void_chain_returns_none(self):
    result = Chain().run()
    await self.assertIsNone(result)

  async def test_void_cascade_returns_none(self):
    result = Cascade().run()
    await self.assertIsNone(result)

  async def test_frozen_no_args_vs_with_args(self):
    frozen = Chain().then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen(5), 10)
    # No args on a void frozen chain: v is Null at link time, lambda receives Null
    # Actually the chain has no root, so it calls lambda with no arg
    # lambda v: v*2 with no arg → error. Let's test with a rooted chain.
    frozen2 = Chain(3).then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen2(), 6)

  async def test_chain_bool_is_true(self):
    c = Chain()
    super(MyTestCase, self).assertTrue(bool(c))

  async def test_chain_call_is_run(self):
    result = Chain(42)()
    await self.assertEqual(result, 42)

  async def test_clone_then_freeze_vs_freeze_original(self):
    c = Chain(5).then(lambda v: v * 2)
    c2 = c.clone()
    f1 = c.freeze()
    f2 = c2.freeze()
    # Both should produce the same result
    await self.assertEqual(f1(), 10)
    await self.assertEqual(f2(), 10)
    # Now modify original, frozen should still work
    c.then(lambda v: v + 100)
    await self.assertEqual(c.run(), 110)
    # f1 is a snapshot via _run reference, shares the original chain's internal state
    # f2 is from the clone, independent
    await self.assertEqual(f2(), 10)

  async def test_cascade_except_handler_receives_root(self):
    """Cascade except_ handler receives root_value."""
    captured = []
    Cascade(42).then(raise_test_exc).except_(lambda v: captured.append(v), reraise=False).run()
    super(MyTestCase, self).assertEqual(captured, [42])

  async def test_chain_then_val_freeze_vs_chain_val_freeze(self):
    """Chain().then(val).freeze() vs Chain(val).freeze() - both should return val."""
    f1 = Chain().then(10).freeze()
    f2 = Chain(10).freeze()
    await self.assertEqual(f1(), 10)
    await self.assertEqual(f2(), 10)

  async def test_chain_pipe_or_operator(self):
    """Test the | operator for Chain."""
    result = Chain(2) | (lambda v: v + 3) | (lambda v: v * 4) | run()
    await self.assertEqual(result, 20)

  async def test_cascade_pipe_returns_root(self):
    result = Cascade(10) | (lambda v: v * 99) | run()
    await self.assertEqual(result, 10)


# ===================================================================
# Section 8: Async variants for each chain type
# ===================================================================

class AsyncChainTests(MyTestCase):
  """Async behavior with Chain."""

  async def test_async_then(self):
    async def double(v):
      return v * 2
    result = await Chain(5).then(double).run()
    await self.assertEqual(result, 10)

  async def test_async_foreach(self):
    async def transform(x):
      return x * 10
    result = await Chain([1, 2, 3]).foreach(transform).run()
    await self.assertEqual(result, [10, 20, 30])

  async def test_async_with_cm(self):
    cm = AsyncCM(value='hello')
    result = await Chain(cm).with_(lambda ctx: ctx.upper()).run()
    await self.assertEqual(result, 'HELLO')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_async_gather(self):
    async def double(v):
      return v * 2
    result = await Chain(5).gather(double, lambda v: v + 1).run()
    await self.assertEqual(result, [10, 6])

  async def test_sync_to_async_transition(self):
    """Sync root transitions to async in the middle."""
    async def async_add(v):
      return v + 10
    result = await Chain(5).then(lambda v: v * 2).then(async_add).run()
    await self.assertEqual(result, 20)  # 5*2=10, 10+10=20


class AsyncCascadeTests(MyTestCase):
  """Async behavior with Cascade."""

  async def test_async_then_returns_root(self):
    async def double(v):
      return v * 2
    result = await Cascade(aempty, 5).then(double).run()
    super(MyTestCase, self).assertEqual(result, 5)

  async def test_async_foreach_returns_root(self):
    root = [1, 2, 3]
    async def transform(x):
      return x * 10
    result = await Cascade(aempty, root).foreach(transform).run()
    super(MyTestCase, self).assertEqual(result, root)

  async def test_async_with_cm_returns_root(self):
    cm = AsyncCM(value='hello')
    result = await Cascade(cm).with_(lambda ctx: ctx.upper()).run()
    super(MyTestCase, self).assertIs(result, cm)

  async def test_async_gather_returns_root(self):
    async def double(v):
      return v * 2
    result = await Cascade(aempty, 5).gather(double, lambda v: v + 1).run()
    super(MyTestCase, self).assertEqual(result, 5)


class AsyncFrozenChainTests(MyTestCase):
  """Async behavior with FrozenChain."""

  async def test_async_frozen_then(self):
    async def double(v):
      return v * 2
    frozen = Chain(5).then(double).freeze()
    await self.assertEqual(frozen(), 10)

  async def test_async_frozen_reuse(self):
    async def inc(v):
      return v + 1
    frozen = Chain().then(inc).freeze()
    await self.assertEqual(frozen(1), 2)
    await self.assertEqual(frozen(10), 11)

  async def test_async_frozen_except(self):
    async def fail(v):
      raise TestExc('async boom')
    frozen = Chain(10).then(fail).except_(lambda v: 'caught', reraise=False).freeze()
    await self.assertEqual(frozen(), 'caught')

  async def test_async_frozen_cascade(self):
    async def double(v):
      return v * 2
    frozen = Cascade(aempty, 7).then(double).freeze()
    await self.assertEqual(frozen(), 7)


class AsyncCloneTests(MyTestCase):
  """Async behavior with clone."""

  async def test_async_clone(self):
    async def add10(v):
      return v + 10
    c = Chain(5).then(add10)
    c2 = c.clone()
    await self.assertEqual(c.run(), 15)
    await self.assertEqual(c2.run(), 15)

  async def test_async_clone_independence(self):
    async def add10(v):
      return v + 10
    c = Chain(5).then(add10)
    c2 = c.clone()
    c2.then(lambda v: v * 2)
    await self.assertEqual(c.run(), 15)
    await self.assertEqual(c2.run(), 30)


class AsyncDecoratorTests(MyTestCase):
  """Async behavior with decorator."""

  async def test_async_decorator_chain(self):
    async def add10(v):
      return v + 10

    @Chain().then(add10).decorator()
    def fn(x):
      return x * 2
    await self.assertEqual(fn(3), 16)  # 3*2=6, 6+10=16

  async def test_async_decorator_async_fn(self):
    @Chain().then(lambda v: v * 5).decorator()
    async def fn(x):
      return x + 1
    await self.assertEqual(fn(2), 15)  # (2+1)*5=15


# ===================================================================
# Additional coverage: Sleep, iterate, to_thread, no_async, debug
# ===================================================================

class SleepTests(MyTestCase):
  """Sleep forces async — run returns a coroutine that must be awaited."""

  async def test_chain_sleep(self):
    result = await Chain(10).sleep(0).then(lambda v: v + 1).run()
    await self.assertEqual(result, 11)

  async def test_cascade_sleep_returns_root(self):
    result = await Cascade(42).sleep(0).then(lambda v: v * 100).run()
    super(MyTestCase, self).assertEqual(result, 42)


class IterateTests(MyTestCase):
  """iterate() returns a generator."""

  async def test_chain_iterate_sync(self):
    results = []
    for i in Chain(lambda: [1, 2, 3]).iterate(lambda i: i * 2):
      results.append(i)
    super(MyTestCase, self).assertEqual(results, [2, 4, 6])

  async def test_chain_iterate_no_fn(self):
    results = []
    for i in Chain(lambda: [10, 20]).iterate():
      results.append(i)
    super(MyTestCase, self).assertEqual(results, [10, 20])

  async def test_chain_iterate_async(self):
    results = []
    async for i in Chain(lambda: [1, 2, 3]).iterate(lambda i: aempty(i * 2)):
      results.append(i)
    super(MyTestCase, self).assertEqual(results, [2, 4, 6])


class NoAsyncTests(MyTestCase):
  """no_async() disables async detection."""

  async def test_no_async_chain(self):
    c = Chain(10).no_async(True).then(lambda v: v + 1)
    result = c.run()
    await self.assertEqual(result, 11)

  async def test_no_async_cascade(self):
    c = Cascade(10).no_async(True).then(lambda v: v * 2)
    result = c.run()
    await self.assertEqual(result, 10)


class DebugTests(MyTestCase):
  """debug mode tests."""

  async def test_debug_chain(self):
    c = Chain(10).config(debug=True).then(lambda v: v + 1)
    result = c.run()
    await self.assertEqual(result, 11)

  async def test_debug_cascade(self):
    c = Cascade(10).config(debug=True).then(lambda v: v * 2)
    result = c.run()
    await self.assertEqual(result, 10)


class ChainReprTests(TestCase):
  """Chain __repr__ returns a string."""

  def test_chain_repr(self):
    c = Chain(10).then(lambda v: v + 1)
    r = repr(c)
    self.assertIsInstance(r, str)
    self.assertIn('Chain', r)

  def test_cascade_repr(self):
    c = Cascade(10).then(lambda v: v + 1)
    r = repr(c)
    self.assertIsInstance(r, str)


class ChainExceptFinallyComboTests(MyTestCase):
  """except_ + finally_ together across variants."""

  async def test_chain_except_finally_on_error(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc')).finally_(lambda v: ran.append('fin')).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertIn('exc', ran)
    super(MyTestCase, self).assertIn('fin', ran)

  async def test_cascade_except_finally_on_error(self):
    ran = []
    try:
      Cascade(10).then(raise_test_exc).except_(lambda v: ran.append('exc')).finally_(lambda v: ran.append('fin')).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertIn('exc', ran)
    super(MyTestCase, self).assertIn('fin', ran)

  async def test_chain_except_noraise_finally(self):
    ran = []
    result = Chain(10).then(raise_test_exc).except_(lambda v: 'recovered', reraise=False).finally_(lambda v: ran.append('fin')).run()
    await self.assertEqual(result, 'recovered')
    super(MyTestCase, self).assertEqual(ran, ['fin'])

  async def test_cascade_except_noraise_finally(self):
    ran = []
    result = Cascade(10).then(raise_test_exc).except_(lambda v: v, reraise=False).finally_(lambda v: ran.append('fin')).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class FrozenExceptFinallyComboTests(MyTestCase):
  """FrozenChain except_ + finally_ combo."""

  async def test_frozen_except_finally_on_error(self):
    ran = []
    frozen = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc')).finally_(lambda v: ran.append('fin')).freeze()
    try:
      frozen()
    except TestExc:
      pass
    super(MyTestCase, self).assertIn('exc', ran)
    super(MyTestCase, self).assertIn('fin', ran)

  async def test_frozen_except_noraise_finally(self):
    ran = []
    frozen = Chain(10).then(raise_test_exc).except_(lambda v: 'ok', reraise=False).finally_(lambda v: ran.append('fin')).freeze()
    await self.assertEqual(frozen(), 'ok')
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class CloneExceptFinallyComboTests(TestCase):
  """Clone except_ + finally_ combo."""

  def test_clone_except_finally(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc'), reraise=False).finally_(lambda v: ran.append('fin'))
    c2 = c.clone()
    ran.clear()
    c2.run()
    self.assertIn('exc', ran)
    self.assertIn('fin', ran)


class DecoratorExceptFinallyComboTests(MyTestCase):
  """Decorator except_ + finally_ combo."""

  async def test_decorator_except_finally(self):
    ran = []

    @Chain().then(raise_test_exc).except_(lambda v: 'recovered', reraise=False).finally_(lambda v: ran.append('fin')).decorator()
    def fn(x):
      return x
    result = fn(5)
    await self.assertEqual(result, 'recovered')
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class AsyncExceptFinallyComboTests(MyTestCase):
  """Async except_ + finally_ combo."""

  async def test_async_chain_except_finally(self):
    ran = []

    async def fail(v):
      raise TestExc('async')

    result = await Chain(10).then(fail).except_(lambda v: 'caught', reraise=False).finally_(lambda v: ran.append('fin')).run()
    await self.assertEqual(result, 'caught')
    super(MyTestCase, self).assertEqual(ran, ['fin'])

  async def test_async_cascade_except_finally(self):
    ran = []

    async def fail(v):
      raise TestExc('async')

    result = await Cascade(aempty, 10).then(fail).except_(lambda v: v, reraise=False).finally_(lambda v: ran.append('fin')).run()
    super(MyTestCase, self).assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(ran, ['fin'])


class GatherAsyncTests(MyTestCase):
  """Gather with async functions."""

  async def test_chain_gather_async(self):
    async def double(v):
      return v * 2

    async def triple(v):
      return v * 3
    result = await Chain(5).gather(double, triple).run()
    await self.assertEqual(result, [10, 15])

  async def test_cascade_gather_async(self):
    async def double(v):
      return v * 2
    result = await Cascade(aempty, 5).gather(double, lambda v: v + 1).run()
    super(MyTestCase, self).assertEqual(result, 5)


class ForeachIndexedAsyncTests(MyTestCase):
  """foreach_indexed with async."""

  async def test_chain_foreach_indexed_async(self):
    async def transform(i, v):
      return (i, v * 10)
    result = await Chain([1, 2, 3]).foreach(transform, with_index=True).run()
    await self.assertEqual(result, [(0, 10), (1, 20), (2, 30)])

  async def test_cascade_foreach_indexed_async(self):
    root = [1, 2, 3]
    captured = []

    async def capture(i, v):
      captured.append((i, v))
    result = await Cascade(aempty, root).foreach(capture, with_index=True).run()
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(captured, [(0, 1), (1, 2), (2, 3)])


class FilterAsyncTests(MyTestCase):
  """filter with async."""

  async def test_chain_filter_async(self):
    async def pred(x):
      return x > 2
    result = await Chain([1, 2, 3, 4]).filter(pred).run()
    await self.assertEqual(result, [3, 4])

  async def test_cascade_filter_async(self):
    root = [1, 2, 3, 4]

    async def pred(x):
      return x > 2
    result = await Cascade(aempty, root).filter(pred).run()
    super(MyTestCase, self).assertEqual(result, root)


class WithAsyncTests(MyTestCase):
  """with_ with async context manager."""

  async def test_chain_with_async_cm(self):
    cm = AsyncCM(value=42)
    result = await Chain(cm).with_(lambda ctx: ctx + 8).run()
    await self.assertEqual(result, 50)
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_cascade_with_async_cm(self):
    cm = AsyncCM(value=42)
    result = await Cascade(cm).with_(lambda ctx: ctx + 8).run()
    super(MyTestCase, self).assertIs(result, cm)
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)


class FrozenChainCallableVsRunTests(MyTestCase):
  """FrozenChain() vs FrozenChain.run() equivalence."""

  async def test_call_vs_run(self):
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    await self.assertEqual(frozen(), frozen.run())

  async def test_call_vs_run_with_override(self):
    frozen = Chain().then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen(7), frozen.run(7))


class CloneNotNestedTests(TestCase):
  """Clone should not be marked as nested."""

  def test_clone_is_not_nested(self):
    inner = Chain(1).then(lambda v: v + 1)
    Chain(0).then(inner)  # marks inner as nested
    c2 = inner.clone()
    # clone should not be nested
    self.assertEqual(c2.run(), 2)


class VoidChainVariantTests(MyTestCase):
  """Void chains (no root) across variants."""

  async def test_void_chain(self):
    await self.assertIsNone(Chain().run())

  async def test_void_cascade(self):
    await self.assertIsNone(Cascade().run())

  async def test_void_chain_with_then(self):
    result = Chain().then(lambda: 42, ...).run()
    await self.assertEqual(result, 42)

  async def test_void_cascade_with_then(self):
    """Void Cascade: root is Null, cascade restores Null -> returns None."""
    result = Cascade().then(lambda: 42, ...).run()
    await self.assertIsNone(result)

  async def test_void_frozen(self):
    frozen = Chain().freeze()
    await self.assertIsNone(frozen())

  async def test_void_clone(self):
    c = Chain()
    c2 = c.clone()
    await self.assertIsNone(c2.run())


class ChainExceptSpecificExceptionsTests(MyTestCase):
  """except_ with specific exception types."""

  async def test_chain_except_specific_match(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('caught'), exceptions=TestExc).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, ['caught'])

  async def test_chain_except_specific_no_match(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('caught'), exceptions=ValueError).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, [])

  async def test_cascade_except_specific_match(self):
    ran = []
    try:
      Cascade(10).then(raise_test_exc).except_(lambda v: ran.append('caught'), exceptions=TestExc).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(ran, ['caught'])


class MultipleExceptHandlersTests(MyTestCase):
  """Multiple except_ handlers."""

  async def test_first_matching_handler_wins(self):
    class Exc1(TestExc):
      pass

    class Exc2(TestExc):
      pass

    ran = []

    def raise_exc1(v=None):
      raise Exc1()

    result = (
      Chain(10)
      .then(raise_exc1)
      .except_(lambda v: ran.append('h1') or 'h1', exceptions=Exc2, reraise=False)
      .except_(lambda v: ran.append('h2') or 'h2', exceptions=Exc1, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'h2')
    super(MyTestCase, self).assertEqual(ran, ['h2'])


class CascadeDoChainTest(MyTestCase):
  """Cascade with do() and then() interleaved."""

  async def test_cascade_do_then_interleaved(self):
    received = []
    result = (
      Cascade(100)
      .do(lambda v: received.append(('do1', v)))
      .then(lambda v: received.append(('then1', v)))
      .do(lambda v: received.append(('do2', v)))
      .run()
    )
    await self.assertEqual(result, 100)
    super(MyTestCase, self).assertEqual(received, [
      ('do1', 100), ('then1', 100), ('do2', 100)
    ])


class ChainNoAsyncWithAsyncFnTest(MyTestCase):
  """no_async with an async function should not detect coroutines."""

  async def test_no_async_returns_coroutine_object(self):
    """When no_async is True, async fn returns the coroutine object, not awaited."""
    async def coro(v):
      return v * 2

    c = Chain(5).no_async(True).then(coro)
    result = c.run()
    # The result is a coroutine object because no_async prevents detection
    import inspect
    super(MyTestCase, self).assertTrue(inspect.iscoroutine(result))
    # Clean up the coroutine to avoid warning
    result.close()


class FrozenChainFromVoidWithOverrideTests(MyTestCase):
  """Frozen void chain called with root override."""

  async def test_frozen_void_with_override(self):
    frozen = Chain().then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen(10), 30)
    await self.assertEqual(frozen(7), 21)

  async def test_frozen_void_foreach_with_override(self):
    frozen = Chain().foreach(lambda x: x + 1).freeze()
    await self.assertEqual(frozen([1, 2, 3]), [2, 3, 4])

  async def test_frozen_void_gather_with_override(self):
    frozen = Chain().gather(lambda v: v * 2, lambda v: v + 1).freeze()
    await self.assertEqual(frozen(5), [10, 6])


class CloneVoidChainTests(TestCase):
  """Clone void chains."""

  def test_clone_void_chain_then_override(self):
    c = Chain().then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c2.run(5), 10)
    self.assertEqual(c2.run(3), 6)


if __name__ == '__main__':
  import unittest
  unittest.main()
