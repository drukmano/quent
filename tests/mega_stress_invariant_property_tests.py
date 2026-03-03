"""Mega stress, invariant, and property-based tests for quent.

Tests mathematical invariants, stress scenarios (chain length, iteration size,
concurrency), and randomized/parameterized patterns across sync and async paths.
No external test libraries (hypothesis etc.) are used.
"""
import asyncio
import gc
import weakref

from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Ref:
  """Weak-referenceable wrapper for Cython objects."""
  __slots__ = ('obj', '__weakref__')
  def __init__(self, obj):
    self.obj = obj


def _sync_add1(v):
  return v + 1


async def _async_add1(v):
  return v + 1


def _sync_double(v):
  return v * 2


async def _async_double(v):
  return v * 2


def _sync_square(v):
  return v ** 2


async def _async_square(v):
  return v ** 2


def _sync_negate(v):
  return -v


async def _async_negate(v):
  return -v


def _sync_identity(v):
  return v


async def _async_identity(v):
  return v


# ===================================================================
# A. Chain Equivalence Invariants (15+ tests)
# ===================================================================

class ChainEquivalenceInvariantsTests(MyTestCase):
  """Verify mathematical properties that MUST hold for any input."""

  # ------------------------------------------------------------------
  # 1. Chain(x).then(f).run() == f(x)
  # ------------------------------------------------------------------
  async def test_chain_then_equals_direct_call_sync(self):
    for x in [0, 1, -1, 42, 100]:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(_sync_add1).run())
        super(MyTestCase, self).assertEqual(result, _sync_add1(x))

  async def test_chain_then_equals_direct_call_async(self):
    for x in [0, 1, -1, 42, 100]:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(_async_add1).run())
        expected = await _async_add1(x)
        super(MyTestCase, self).assertEqual(result, expected)

  # ------------------------------------------------------------------
  # 2. Chain(x).then(f).then(g).run() == g(f(x)) -- composition
  # ------------------------------------------------------------------
  async def test_two_level_composition_sync(self):
    for x in [0, 1, 5, -3, 10]:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(_sync_add1).then(_sync_double).run())
        super(MyTestCase, self).assertEqual(result, _sync_double(_sync_add1(x)))

  async def test_two_level_composition_async(self):
    for x in [0, 1, 5, -3, 10]:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(_async_add1).then(_async_double).run())
        expected = await _async_double(await _async_add1(x))
        super(MyTestCase, self).assertEqual(result, expected)

  async def test_two_level_composition_mixed(self):
    for x in [0, 1, 5, -3, 10]:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(_sync_add1).then(_async_double).run())
        expected = await _async_double(_sync_add1(x))
        super(MyTestCase, self).assertEqual(result, expected)

  # ------------------------------------------------------------------
  # 3. h(g(f(x))) -- 3-level composition
  # ------------------------------------------------------------------
  async def test_three_level_composition_sync(self):
    for x in [0, 1, 2, -2, 7]:
      with self.subTest(x=x):
        result = await await_(
          Chain(x).then(_sync_add1).then(_sync_double).then(_sync_square).run()
        )
        super(MyTestCase, self).assertEqual(result, _sync_square(_sync_double(_sync_add1(x))))

  async def test_three_level_composition_async(self):
    for x in [0, 1, 2, -2, 7]:
      with self.subTest(x=x):
        result = await await_(
          Chain(x).then(_async_add1).then(_async_double).then(_async_square).run()
        )
        expected = await _async_square(await _async_double(await _async_add1(x)))
        super(MyTestCase, self).assertEqual(result, expected)

  async def test_three_level_composition_mixed(self):
    for x in [0, 1, 2, -2, 7]:
      with self.subTest(x=x):
        result = await await_(
          Chain(x).then(_sync_add1).then(_async_double).then(_sync_square).run()
        )
        expected = _sync_square(await _async_double(_sync_add1(x)))
        super(MyTestCase, self).assertEqual(result, expected)

  # ------------------------------------------------------------------
  # 4. Chain(x).run() == x for any literal x -- identity
  # ------------------------------------------------------------------
  async def test_chain_identity_various_literals(self):
    for x in [None, 0, False, '', [], {}, 42, 'hello', [1, 2, 3], (1,), {1: 2}, 0.5, True]:
      with self.subTest(x=x):
        await self.assertEqual(Chain(x).run(), x)

  # ------------------------------------------------------------------
  # 5. Chain().run(x) == Chain(x).run() -- override equivalence
  # ------------------------------------------------------------------
  async def test_override_equivalence_sync(self):
    for x in [0, 1, 42, 'abc', [1, 2], None, False]:
      with self.subTest(x=x):
        r1 = await await_(Chain().run(x))
        r2 = await await_(Chain(x).run())
        super(MyTestCase, self).assertEqual(r1, r2)

  async def test_override_equivalence_with_then_sync(self):
    for x in [0, 1, 5, -3]:
      with self.subTest(x=x):
        r1 = await await_(Chain().then(_sync_add1).run(x))
        r2 = await await_(Chain(x).then(_sync_add1).run())
        super(MyTestCase, self).assertEqual(r1, r2)

  async def test_override_equivalence_with_then_async(self):
    for x in [0, 1, 5, -3]:
      with self.subTest(x=x):
        r1 = await await_(Chain().then(_async_add1).run(x))
        r2 = await await_(Chain(x).then(_async_add1).run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 6. Pipe equivalence: Chain(x).then(f).run() == (Chain(x) | f | run())
  # ------------------------------------------------------------------
  async def test_pipe_equivalence_sync(self):
    for x in [0, 1, 10, -5]:
      with self.subTest(x=x):
        r1 = await await_(Chain(x).then(_sync_double).run())
        r2 = await await_(Chain(x) | _sync_double | run())
        super(MyTestCase, self).assertEqual(r1, r2)

  async def test_pipe_equivalence_async(self):
    for x in [0, 1, 10, -5]:
      with self.subTest(x=x):
        r1 = await await_(Chain(x).then(_async_double).run())
        r2 = await await_(Chain(x) | _async_double | run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 7. Clone equivalence: Chain(x).clone().run() == Chain(x).run()
  # ------------------------------------------------------------------
  async def test_clone_equivalence_various(self):
    for x in [0, 1, -1, 42, 'hello', [1, 2], None, False]:
      with self.subTest(x=x):
        c = Chain(x)
        r1 = await await_(c.clone().run())
        r2 = await await_(c.clone().run())
        super(MyTestCase, self).assertEqual(r1, r2)

  async def test_clone_equivalence_with_links(self):
    for x in [0, 1, 5, 10]:
      with self.subTest(x=x):
        c = Chain(x).then(_sync_add1).then(_sync_double)
        r1 = await await_(c.clone().run())
        r2 = await await_(c.clone().run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 8. Freeze equivalence: Chain(x).freeze().run() == Chain(x).run()
  # ------------------------------------------------------------------
  async def test_freeze_equivalence_sync(self):
    for x in [0, 1, 42, -5]:
      with self.subTest(x=x):
        c = Chain(x)
        r1 = await await_(c.clone().run())
        r2 = await await_(c.freeze().run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 9. Freeze call equivalence: Chain(x).freeze()() == Chain(x).run()
  # ------------------------------------------------------------------
  async def test_freeze_call_equivalence(self):
    for x in [0, 1, 42, -5]:
      with self.subTest(x=x):
        c = Chain(x)
        r1 = await await_(c.clone().run())
        r2 = await await_(c.freeze()())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 10. Cascade(x).then(f).run() == x -- cascade returns root
  # ------------------------------------------------------------------
  async def test_cascade_returns_root_sync(self):
    for x in [0, 1, 42, 'hello', [1, 2]]:
      with self.subTest(x=x):
        result = await await_(Cascade(x).then(_sync_double).run())
        super(MyTestCase, self).assertEqual(result, x)

  async def test_cascade_returns_root_async(self):
    for x in [0, 1, 42, 'hello']:
      with self.subTest(x=x):
        result = await await_(Cascade(x).then(_async_identity).run())
        super(MyTestCase, self).assertEqual(result, x)

  # ------------------------------------------------------------------
  # 11. Chain(x).do(f).run() == x -- do discards
  # ------------------------------------------------------------------
  async def test_do_discards_result_sync(self):
    side_effects = []
    for x in [0, 1, 42, 'hello']:
      with self.subTest(x=x):
        side_effects.clear()
        result = await await_(Chain(x).do(lambda v: side_effects.append(v)).run())
        super(MyTestCase, self).assertEqual(result, x)
        super(MyTestCase, self).assertEqual(side_effects, [x])

  async def test_do_discards_result_async(self):
    side_effects = []
    async def async_side(v):
      side_effects.append(v)
      return 'should_be_discarded'
    for x in [0, 1, 42]:
      with self.subTest(x=x):
        side_effects.clear()
        result = await await_(Chain(x).do(async_side).run())
        super(MyTestCase, self).assertEqual(result, x)
        super(MyTestCase, self).assertEqual(side_effects, [x])

  # ------------------------------------------------------------------
  # 12. Chain(x).then(f).do(g).run() == f(x) -- do after then discards g
  # ------------------------------------------------------------------
  async def test_do_after_then_discards(self):
    side_effects = []
    for x in [0, 1, 5]:
      with self.subTest(x=x):
        side_effects.clear()
        result = await await_(
          Chain(x).then(_sync_add1).do(lambda v: side_effects.append(v)).run()
        )
        super(MyTestCase, self).assertEqual(result, _sync_add1(x))
        super(MyTestCase, self).assertEqual(side_effects, [_sync_add1(x)])

  # ------------------------------------------------------------------
  # 13. Chain(x).then(lambda v: v).run() == x -- identity function
  # ------------------------------------------------------------------
  async def test_identity_function_sync(self):
    for x in [None, 0, False, '', 42, 'hello', [], {}]:
      with self.subTest(x=x):
        await self.assertEqual(Chain(x).then(lambda v: v).run(), x)

  async def test_identity_function_async(self):
    for x in [None, 0, False, '', 42, 'hello']:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(_async_identity).run())
        super(MyTestCase, self).assertEqual(result, x)

  # ------------------------------------------------------------------
  # 14. Associativity: Chain(x).then(f).then(g).run() == Chain(f(x)).then(g).run()
  # ------------------------------------------------------------------
  async def test_associativity_sync(self):
    for x in [0, 1, 5, 10, -3]:
      with self.subTest(x=x):
        r1 = await await_(Chain(x).then(_sync_add1).then(_sync_double).run())
        r2 = await await_(Chain(_sync_add1(x)).then(_sync_double).run())
        super(MyTestCase, self).assertEqual(r1, r2)

  async def test_associativity_async(self):
    for x in [0, 1, 5, 10, -3]:
      with self.subTest(x=x):
        r1 = await await_(Chain(x).then(_async_add1).then(_async_double).run())
        intermediate = await _async_add1(x)
        r2 = await await_(Chain(intermediate).then(_async_double).run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 15. Sync + async combined invariants
  # ------------------------------------------------------------------
  async def test_all_invariants_both_paths(self):
    """Verify key invariants for both sync and async function variants."""
    for fn_add, fn_dbl, fn_id, label in [
      (_sync_add1, _sync_double, _sync_identity, 'sync'),
      (_async_add1, _async_double, _async_identity, 'async'),
    ]:
      with self.subTest(path=label):
        for x in [0, 1, 5, -2]:
          # Identity
          r = await await_(Chain(x).then(fn_id).run())
          super(MyTestCase, self).assertEqual(r, x)

          # Composition
          r = await await_(Chain(x).then(fn_add).then(fn_dbl).run())
          intermediate = await await_(fn_add(x))
          expected = await await_(fn_dbl(intermediate))
          super(MyTestCase, self).assertEqual(r, expected)

          # Cascade invariant
          r = await await_(Cascade(x).then(fn_dbl).run())
          super(MyTestCase, self).assertEqual(r, x)


# ===================================================================
# B. Cascade Invariants (8+ tests)
# ===================================================================

class CascadeInvariantsTests(MyTestCase):
  """Verify that Cascade always passes root value and returns root value."""

  # ------------------------------------------------------------------
  # 16. Every link in Cascade receives the same root value
  # ------------------------------------------------------------------
  async def test_every_link_receives_root_sync(self):
    received = []
    root = 42
    Cascade(root) \
      .then(lambda v: received.append(v)) \
      .then(lambda v: received.append(v)) \
      .then(lambda v: received.append(v)) \
      .run()
    super(MyTestCase, self).assertEqual(received, [root, root, root])

  async def test_every_link_receives_root_async(self):
    received = []
    root = 42
    async def capture(v):
      received.append(v)
    await await_(
      Cascade(root).then(capture).then(capture).then(capture).run()
    )
    super(MyTestCase, self).assertEqual(received, [root, root, root])

  # ------------------------------------------------------------------
  # 17. Cascade always returns root value regardless of what links return
  # ------------------------------------------------------------------
  async def test_cascade_returns_root_regardless(self):
    for root in [1, 'hello', [1, 2], None]:
      with self.subTest(root=root):
        result = await await_(
          Cascade(root)
            .then(lambda v: 'totally_different')
            .then(lambda v: 99999)
            .run()
        )
        super(MyTestCase, self).assertEqual(result, root)

  # ------------------------------------------------------------------
  # 18. Cascade with override: Cascade().run(x) -- all links get x
  # ------------------------------------------------------------------
  async def test_cascade_override_all_links_get_x(self):
    received = []
    x = 77
    await await_(
      Cascade()
        .then(lambda v: received.append(v))
        .then(lambda v: received.append(v))
        .run(x)
    )
    super(MyTestCase, self).assertEqual(received, [x, x])

  # ------------------------------------------------------------------
  # 19. Cascade with N links: each one gets root, final result is root
  # ------------------------------------------------------------------
  async def test_cascade_n_links_returns_root(self):
    for n in [0, 1, 5, 20]:
      with self.subTest(n=n):
        root = 42
        c = Cascade(root)
        for _ in range(n):
          c = c.then(lambda v: v * 100)
        result = await await_(c.run())
        super(MyTestCase, self).assertEqual(result, root)

  # ------------------------------------------------------------------
  # 20. Cascade with .do() still receives root
  # ------------------------------------------------------------------
  async def test_cascade_do_receives_root(self):
    received = []
    root = 55
    Cascade(root).do(lambda v: received.append(v)).run()
    super(MyTestCase, self).assertEqual(received, [root])

  # ------------------------------------------------------------------
  # 21. Cascade with mix of .then() and .do() -- all get root
  # ------------------------------------------------------------------
  async def test_cascade_mixed_then_do_all_get_root(self):
    received_then = []
    received_do = []
    root = 33
    Cascade(root) \
      .then(lambda v: received_then.append(v)) \
      .do(lambda v: received_do.append(v)) \
      .then(lambda v: received_then.append(v)) \
      .do(lambda v: received_do.append(v)) \
      .run()
    super(MyTestCase, self).assertEqual(received_then, [root, root])
    super(MyTestCase, self).assertEqual(received_do, [root, root])

  # ------------------------------------------------------------------
  # 22. Cascade async path preserves root value invariant
  # ------------------------------------------------------------------
  async def test_cascade_async_preserves_root(self):
    received = []
    root = 'async_root'
    async def capture(v):
      received.append(v)
      return 'should_be_ignored'
    result = await await_(
      Cascade(root).then(capture).then(capture).run()
    )
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(received, [root, root])

  # ------------------------------------------------------------------
  # 23. Cascade invariant holds for 0, 1, 5, 20 links
  # ------------------------------------------------------------------
  async def test_cascade_invariant_various_link_counts(self):
    for n in [0, 1, 5, 20]:
      with self.subTest(n=n):
        root = 100
        c = Cascade(root)
        received = []
        for _ in range(n):
          c = c.then(lambda v: received.append(v))
        result = await await_(c.run())
        super(MyTestCase, self).assertEqual(result, root)
        super(MyTestCase, self).assertEqual(len(received), n)
        for v in received:
          super(MyTestCase, self).assertEqual(v, root)


# ===================================================================
# C. Exception Handling Invariants (10+ tests)
# ===================================================================

class ExceptionHandlingInvariantsTests(MyTestCase):
  """Verify except_ and finally_ invariants."""

  # ------------------------------------------------------------------
  # 24. except_ handler is called exactly ONCE
  # ------------------------------------------------------------------
  async def test_except_called_exactly_once_sync(self):
    call_count = [0]
    def handler(v=None):
      call_count[0] += 1
    try:
      Chain().then(lambda: (_ for _ in ()).throw(TestExc())).except_(handler, reraise=True).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(call_count[0], 1)

  async def test_except_called_exactly_once_sync_simple(self):
    call_count = [0]
    def handler(v=None):
      call_count[0] += 1
    def raiser():
      raise TestExc()
    try:
      Chain(raiser).except_(handler, reraise=True).run()
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(call_count[0], 1)

  async def test_except_called_exactly_once_async(self):
    call_count = [0]
    async def handler(v=None):
      call_count[0] += 1
    async def raiser():
      raise TestExc()
    try:
      await await_(Chain(raiser).except_(handler, reraise=True).run())
    except TestExc:
      pass
    super(MyTestCase, self).assertEqual(call_count[0], 1)

  # ------------------------------------------------------------------
  # 25. except_ with reraise=True: exception still propagates
  # ------------------------------------------------------------------
  async def test_except_reraise_true_propagates(self):
    handler_called = [False]
    def handler(v=None):
      handler_called[0] = True
    def raiser():
      raise TestExc('should propagate')
    with self.assertRaises(TestExc):
      Chain(raiser).except_(handler, reraise=True).run()
    super(MyTestCase, self).assertTrue(handler_called[0])

  async def test_except_reraise_true_propagates_async(self):
    handler_called = [False]
    async def handler(v=None):
      handler_called[0] = True
    async def raiser():
      raise TestExc('should propagate')
    with self.assertRaises(TestExc):
      await await_(Chain(raiser).except_(handler, reraise=True).run())
    super(MyTestCase, self).assertTrue(handler_called[0])

  # ------------------------------------------------------------------
  # 26. except_ with reraise=False: exception does NOT propagate
  # ------------------------------------------------------------------
  async def test_except_reraise_false_no_propagation(self):
    handler_called = [False]
    def handler(v=None):
      handler_called[0] = True
    def raiser():
      raise TestExc()
    # Should not raise
    result = Chain(raiser).except_(handler, reraise=False).run()
    super(MyTestCase, self).assertTrue(handler_called[0])
    super(MyTestCase, self).assertIsNone(result)

  async def test_except_reraise_false_no_propagation_async(self):
    handler_called = [False]
    async def handler(v=None):
      handler_called[0] = True
    async def raiser():
      raise TestExc()
    result = await await_(Chain(raiser).except_(handler, reraise=False).run())
    super(MyTestCase, self).assertTrue(handler_called[0])
    super(MyTestCase, self).assertIsNone(result)

  # ------------------------------------------------------------------
  # 27. Multiple except_ handlers: ONLY the first matching one runs
  # ------------------------------------------------------------------
  async def test_multiple_except_only_first_matching_runs(self):
    called = {'first': False, 'second': False}
    def first_handler(v=None):
      called['first'] = True
    def second_handler(v=None):
      called['second'] = True
    def raiser():
      raise TestExc()
    try:
      Chain(raiser) \
        .except_(first_handler, exceptions=TestExc, reraise=True) \
        .except_(second_handler, exceptions=TestExc, reraise=True) \
        .run()
    except TestExc:
      pass
    super(MyTestCase, self).assertTrue(called['first'])
    super(MyTestCase, self).assertFalse(called['second'])

  # ------------------------------------------------------------------
  # 28. finally_ handler ALWAYS runs (success path)
  # ------------------------------------------------------------------
  async def test_finally_runs_on_success_sync(self):
    finally_called = [False]
    def on_finally(v=None):
      finally_called[0] = True
    Chain(42).then(lambda v: v + 1).finally_(on_finally).run()
    super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_finally_runs_on_success_async(self):
    finally_called = [False]
    async def on_finally(v=None):
      finally_called[0] = True
    await await_(Chain(42).then(_async_add1).finally_(on_finally).run())
    super(MyTestCase, self).assertTrue(finally_called[0])

  # ------------------------------------------------------------------
  # 29. finally_ handler ALWAYS runs (exception path)
  # ------------------------------------------------------------------
  async def test_finally_runs_on_exception_sync(self):
    finally_called = [False]
    def on_finally(v=None):
      finally_called[0] = True
    def raiser():
      raise TestExc()
    with self.assertRaises(TestExc):
      Chain(raiser).finally_(on_finally).run()
    super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_finally_runs_on_exception_async(self):
    finally_called = [False]
    async def on_finally(v=None):
      finally_called[0] = True
    async def raiser():
      raise TestExc()
    with self.assertRaises(TestExc):
      await await_(Chain(raiser).finally_(on_finally).run())
    super(MyTestCase, self).assertTrue(finally_called[0])

  # ------------------------------------------------------------------
  # 30. finally_ ALWAYS runs (exception caught and not reraised path)
  # ------------------------------------------------------------------
  async def test_finally_runs_on_caught_exception_sync(self):
    finally_called = [False]
    handler_called = [False]
    def on_finally(v=None):
      finally_called[0] = True
    def on_except(v=None):
      handler_called[0] = True
    def raiser():
      raise TestExc()
    Chain(raiser).except_(on_except, reraise=False).finally_(on_finally).run()
    super(MyTestCase, self).assertTrue(handler_called[0])
    super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_finally_runs_on_caught_exception_async(self):
    finally_called = [False]
    handler_called = [False]
    async def on_finally(v=None):
      finally_called[0] = True
    async def on_except(v=None):
      handler_called[0] = True
    async def raiser():
      raise TestExc()
    await await_(
      Chain(raiser).except_(on_except, reraise=False).finally_(on_finally).run()
    )
    super(MyTestCase, self).assertTrue(handler_called[0])
    super(MyTestCase, self).assertTrue(finally_called[0])

  # ------------------------------------------------------------------
  # 31. except_ does not swallow exceptions it doesn't match
  # ------------------------------------------------------------------
  async def test_except_does_not_swallow_unmatched(self):
    handler_called = [False]
    def handler(v=None):
      handler_called[0] = True

    class SpecificExc(Exception):
      pass

    def raiser():
      raise TestExc()

    with self.assertRaises(TestExc):
      Chain(raiser).except_(handler, exceptions=SpecificExc, reraise=False).run()
    super(MyTestCase, self).assertFalse(handler_called[0])

  # ------------------------------------------------------------------
  # 32. except_ handler receives root value, not intermediate value
  # ------------------------------------------------------------------
  async def test_except_receives_root_value(self):
    received = [None]
    root = 'the_root'
    def handler(v=None):
      received[0] = v
    def raiser(v):
      raise TestExc()
    Chain(root).then(raiser).except_(handler, reraise=False).run()
    super(MyTestCase, self).assertEqual(received[0], root)

  async def test_except_receives_root_value_async(self):
    received = [None]
    root = 'the_root'
    async def handler(v=None):
      received[0] = v
    async def raiser(v):
      raise TestExc()
    await await_(
      Chain(root).then(raiser).except_(handler, reraise=False).run()
    )
    super(MyTestCase, self).assertEqual(received[0], root)

  # ------------------------------------------------------------------
  # 33. Exception type matching uses isinstance (subclasses match)
  # ------------------------------------------------------------------
  async def test_exception_subclass_matching(self):
    class ParentExc(Exception):
      pass
    class ChildExc(ParentExc):
      pass
    handler_called = [False]
    def handler(v=None):
      handler_called[0] = True
    def raiser():
      raise ChildExc()
    # Handler for ParentExc should catch ChildExc
    Chain(raiser).except_(handler, exceptions=ParentExc, reraise=False).run()
    super(MyTestCase, self).assertTrue(handler_called[0])


# ===================================================================
# D. Iteration Invariants (10+ tests)
# ===================================================================

class IterationInvariantsTests(MyTestCase):
  """Verify foreach, filter, gather invariants."""

  # ------------------------------------------------------------------
  # 34. foreach preserves element count
  # ------------------------------------------------------------------
  async def test_foreach_preserves_count(self):
    for items in [[], [1], [1, 2, 3], list(range(50))]:
      with self.subTest(len=len(items)):
        result = await await_(Chain(items).foreach(lambda x: x * 2).run())
        super(MyTestCase, self).assertEqual(len(result), len(items))

  # ------------------------------------------------------------------
  # 35. foreach with identity function
  # ------------------------------------------------------------------
  async def test_foreach_identity(self):
    for items in [[], [1], [1, 2, 3], list(range(20))]:
      with self.subTest(items=items):
        result = await await_(Chain(items).foreach(lambda x: x).run())
        super(MyTestCase, self).assertEqual(result, items)

  # ------------------------------------------------------------------
  # 36. filter preserves order (subsequence of input)
  # ------------------------------------------------------------------
  async def test_filter_preserves_order(self):
    items = [5, 3, 8, 1, 9, 2, 7, 4, 6]
    result = await await_(Chain(items).filter(lambda x: x > 4).run())
    # Verify it is a subsequence preserving order
    it = iter(items)
    for r in result:
      found = False
      for item in it:
        if item == r:
          found = True
          break
      super(MyTestCase, self).assertTrue(found, f'{r} not found in order')

  # ------------------------------------------------------------------
  # 37. filter with always-true predicate: result == input
  # ------------------------------------------------------------------
  async def test_filter_always_true(self):
    for items in [[], [1], [1, 2, 3], list(range(20))]:
      with self.subTest(items=items):
        result = await await_(Chain(items).filter(lambda x: True).run())
        super(MyTestCase, self).assertEqual(result, items)

  # ------------------------------------------------------------------
  # 38. filter with always-false predicate: result == []
  # ------------------------------------------------------------------
  async def test_filter_always_false(self):
    for items in [[], [1], [1, 2, 3], list(range(20))]:
      with self.subTest(items=items):
        result = await await_(Chain(items).filter(lambda x: False).run())
        super(MyTestCase, self).assertEqual(result, [])

  # ------------------------------------------------------------------
  # 39. gather result length == number of functions
  # ------------------------------------------------------------------
  async def test_gather_result_length(self):
    for n in [0, 1, 3, 5, 10]:
      with self.subTest(n=n):
        fns = [lambda v, i=i: v + i for i in range(n)]
        result = await await_(Chain(10).gather(*fns).run())
        super(MyTestCase, self).assertEqual(len(result), n)

  # ------------------------------------------------------------------
  # 40. gather preserves function order
  # ------------------------------------------------------------------
  async def test_gather_preserves_order(self):
    fns = [
      lambda v: v * 1,
      lambda v: v * 2,
      lambda v: v * 3,
      lambda v: v * 4,
      lambda v: v * 5,
    ]
    result = await await_(Chain(10).gather(*fns).run())
    super(MyTestCase, self).assertEqual(result, [10, 20, 30, 40, 50])

  async def test_gather_preserves_order_async(self):
    async def f1(v): return v * 1
    async def f2(v): return v * 2
    async def f3(v): return v * 3
    result = await await_(Chain(10).gather(f1, f2, f3).run())
    super(MyTestCase, self).assertEqual(result, [10, 20, 30])

  # ------------------------------------------------------------------
  # 41. foreach_indexed indices are 0, 1, 2, ... N-1
  # ------------------------------------------------------------------
  async def test_foreach_indexed_indices(self):
    items = ['a', 'b', 'c', 'd', 'e']
    result = await await_(Chain(items).foreach(lambda i, el: (i, el), with_index=True).run())
    for idx, (i, el) in enumerate(result):
      super(MyTestCase, self).assertEqual(i, idx)
      super(MyTestCase, self).assertEqual(el, items[idx])

  async def test_foreach_indexed_indices_async(self):
    items = ['a', 'b', 'c', 'd', 'e']
    async def indexed_fn(i, el):
      return (i, el)
    result = await await_(Chain(items).foreach(indexed_fn, with_index=True).run())
    for idx, (i, el) in enumerate(result):
      super(MyTestCase, self).assertEqual(i, idx)
      super(MyTestCase, self).assertEqual(el, items[idx])

  # ------------------------------------------------------------------
  # 42. foreach on empty iterable -> []
  # ------------------------------------------------------------------
  async def test_foreach_empty_iterable(self):
    result = await await_(Chain([]).foreach(lambda x: x * 2).run())
    super(MyTestCase, self).assertEqual(result, [])

  # ------------------------------------------------------------------
  # 43. filter on empty iterable -> []
  # ------------------------------------------------------------------
  async def test_filter_empty_iterable(self):
    result = await await_(Chain([]).filter(lambda x: True).run())
    super(MyTestCase, self).assertEqual(result, [])


# ===================================================================
# E. Stress Tests -- Chain Length (8+ tests)
# ===================================================================

class StressChainLengthTests(MyTestCase):
  """Verify chains of various lengths work correctly."""

  # ------------------------------------------------------------------
  # 44. Chain with 10 then-links
  # ------------------------------------------------------------------
  async def test_chain_10_links(self):
    c = Chain(0)
    for _ in range(10):
      c = c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 10)

  # ------------------------------------------------------------------
  # 45. Chain with 100 then-links
  # ------------------------------------------------------------------
  async def test_chain_100_links(self):
    c = Chain(0)
    for _ in range(100):
      c = c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 100)

  # ------------------------------------------------------------------
  # 46. Chain with 500 then-links
  # ------------------------------------------------------------------
  async def test_chain_500_links(self):
    c = Chain(0)
    for _ in range(500):
      c = c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 500)

  # ------------------------------------------------------------------
  # 47. Chain with 1000 then-links -- no crash
  # ------------------------------------------------------------------
  async def test_chain_1000_links(self):
    c = Chain(0)
    for _ in range(1000):
      c = c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 1000)

  # ------------------------------------------------------------------
  # 48. Chain with 10 nested chains (1 level each)
  # ------------------------------------------------------------------
  async def test_10_nested_chains_1_level(self):
    inner = Chain(1)
    for _ in range(9):
      inner = Chain().then(inner)
    await self.assertEqual(inner.run(), 1)

  # ------------------------------------------------------------------
  # 49. Chain with 5 levels of nesting
  # ------------------------------------------------------------------
  async def test_5_levels_nesting(self):
    inner = Chain().then(lambda v: v + 1)
    for _ in range(4):
      prev = inner
      inner = Chain().then(prev)
    result = await await_(inner.run(0))
    super(MyTestCase, self).assertEqual(result, 1)

  # ------------------------------------------------------------------
  # 50. Chain with 10 levels of nesting
  # ------------------------------------------------------------------
  async def test_10_levels_nesting(self):
    inner = Chain().then(lambda v: v + 1)
    for _ in range(9):
      prev = inner
      inner = Chain().then(prev)
    result = await await_(inner.run(0))
    super(MyTestCase, self).assertEqual(result, 1)

  # ------------------------------------------------------------------
  # 51. Chain with 20 levels of nesting
  # ------------------------------------------------------------------
  async def test_20_levels_nesting(self):
    inner = Chain().then(lambda v: v + 1)
    for _ in range(19):
      prev = inner
      inner = Chain().then(prev)
    result = await await_(inner.run(0))
    super(MyTestCase, self).assertEqual(result, 1)

  # ------------------------------------------------------------------
  # Extra: Chain with mixed sync/async links
  # ------------------------------------------------------------------
  async def test_chain_100_mixed_sync_async(self):
    c = Chain(0)
    for i in range(100):
      if i % 5 == 0:
        c = c.then(_async_add1)
      else:
        c = c.then(lambda v: v + 1)
    result = await await_(c.run())
    super(MyTestCase, self).assertEqual(result, 100)


# ===================================================================
# F. Stress Tests -- Iteration Size (6+ tests)
# ===================================================================

class StressIterationSizeTests(MyTestCase):
  """Verify iteration operations scale to large inputs."""

  # ------------------------------------------------------------------
  # 52. foreach on 100 elements
  # ------------------------------------------------------------------
  async def test_foreach_100_elements(self):
    items = list(range(100))
    result = await await_(Chain(items).foreach(lambda x: x * 2).run())
    super(MyTestCase, self).assertEqual(len(result), 100)
    super(MyTestCase, self).assertEqual(result, [x * 2 for x in items])

  # ------------------------------------------------------------------
  # 53. foreach on 1000 elements
  # ------------------------------------------------------------------
  async def test_foreach_1000_elements(self):
    items = list(range(1000))
    result = await await_(Chain(items).foreach(lambda x: x + 1).run())
    super(MyTestCase, self).assertEqual(len(result), 1000)
    super(MyTestCase, self).assertEqual(result[0], 1)
    super(MyTestCase, self).assertEqual(result[999], 1000)

  # ------------------------------------------------------------------
  # 54. foreach on 10000 elements
  # ------------------------------------------------------------------
  async def test_foreach_10000_elements(self):
    items = list(range(10000))
    result = await await_(Chain(items).foreach(lambda x: x).run())
    super(MyTestCase, self).assertEqual(len(result), 10000)
    super(MyTestCase, self).assertEqual(result, items)

  # ------------------------------------------------------------------
  # 55. filter on 1000 elements
  # ------------------------------------------------------------------
  async def test_filter_1000_elements(self):
    items = list(range(1000))
    result = await await_(Chain(items).filter(lambda x: x % 2 == 0).run())
    super(MyTestCase, self).assertEqual(len(result), 500)
    super(MyTestCase, self).assertEqual(result, [x for x in items if x % 2 == 0])

  # ------------------------------------------------------------------
  # 56. gather with 50 functions
  # ------------------------------------------------------------------
  async def test_gather_50_functions(self):
    fns = [lambda v, i=i: v + i for i in range(50)]
    result = await await_(Chain(0).gather(*fns).run())
    super(MyTestCase, self).assertEqual(len(result), 50)
    super(MyTestCase, self).assertEqual(result, list(range(50)))

  # ------------------------------------------------------------------
  # 57. gather with 100 functions
  # ------------------------------------------------------------------
  async def test_gather_100_functions(self):
    fns = [lambda v, i=i: v * i for i in range(100)]
    result = await await_(Chain(1).gather(*fns).run())
    super(MyTestCase, self).assertEqual(len(result), 100)
    for i in range(100):
      super(MyTestCase, self).assertEqual(result[i], i)

  # ------------------------------------------------------------------
  # Extra: filter on 10000 elements
  # ------------------------------------------------------------------
  async def test_filter_10000_elements(self):
    items = list(range(10000))
    result = await await_(Chain(items).filter(lambda x: x % 3 == 0).run())
    expected = [x for x in items if x % 3 == 0]
    super(MyTestCase, self).assertEqual(result, expected)


# ===================================================================
# G. Stress Tests -- Concurrent Execution (8+ tests)
# ===================================================================

class StressConcurrentExecutionTests(MyTestCase):
  """Verify concurrent async chain executions produce correct results."""

  # ------------------------------------------------------------------
  # 58. 10 concurrent async chain executions
  # ------------------------------------------------------------------
  async def test_10_concurrent_async_chains(self):
    async def work(v):
      return v * 2
    tasks = [asyncio.ensure_future(await_(Chain(i).then(work).run())) for i in range(10)]
    results = await asyncio.gather(*tasks)
    for i in range(10):
      super(MyTestCase, self).assertEqual(results[i], i * 2)

  # ------------------------------------------------------------------
  # 59. 50 concurrent async chain executions
  # ------------------------------------------------------------------
  async def test_50_concurrent_async_chains(self):
    async def work(v):
      return v + 100
    tasks = [asyncio.ensure_future(await_(Chain(i).then(work).run())) for i in range(50)]
    results = await asyncio.gather(*tasks)
    for i in range(50):
      super(MyTestCase, self).assertEqual(results[i], i + 100)

  # ------------------------------------------------------------------
  # 60. 100 concurrent async chain executions
  # ------------------------------------------------------------------
  async def test_100_concurrent_async_chains(self):
    async def work(v):
      return v ** 2
    tasks = [asyncio.ensure_future(await_(Chain(i).then(work).run())) for i in range(100)]
    results = await asyncio.gather(*tasks)
    for i in range(100):
      super(MyTestCase, self).assertEqual(results[i], i ** 2)

  # ------------------------------------------------------------------
  # 61. Concurrent FrozenChain executions (50 concurrent)
  # ------------------------------------------------------------------
  async def test_concurrent_frozen_chain_50(self):
    frozen = Chain().then(lambda v: v * 3).freeze()
    tasks = [asyncio.ensure_future(await_(frozen.run(i))) for i in range(50)]
    results = await asyncio.gather(*tasks)
    for i in range(50):
      super(MyTestCase, self).assertEqual(results[i], i * 3)

  # ------------------------------------------------------------------
  # 62. Concurrent clone executions (50 concurrent)
  # ------------------------------------------------------------------
  async def test_concurrent_clone_executions_50(self):
    async def work(v):
      return v + 10
    base = Chain().then(work)
    clones = [base.clone() for _ in range(50)]
    tasks = [asyncio.ensure_future(await_(clones[i].run(i))) for i in range(50)]
    results = await asyncio.gather(*tasks)
    for i in range(50):
      super(MyTestCase, self).assertEqual(results[i], i + 10)

  # ------------------------------------------------------------------
  # 63. Concurrent foreach executions (20 concurrent)
  # ------------------------------------------------------------------
  async def test_concurrent_foreach_20(self):
    async def double(x):
      return x * 2
    tasks = []
    for i in range(20):
      items = list(range(i * 10, (i + 1) * 10))
      tasks.append(asyncio.ensure_future(await_(Chain(items).foreach(double).run())))
    results = await asyncio.gather(*tasks)
    for i in range(20):
      items = list(range(i * 10, (i + 1) * 10))
      expected = [x * 2 for x in items]
      super(MyTestCase, self).assertEqual(results[i], expected)

  # ------------------------------------------------------------------
  # 64. Rapid sequential chain creation and execution (1000 chains)
  # ------------------------------------------------------------------
  async def test_rapid_sequential_1000_chains(self):
    results = []
    for i in range(1000):
      r = await await_(Chain(i).then(lambda v: v + 1).run())
      results.append(r)
    super(MyTestCase, self).assertEqual(len(results), 1000)
    for i in range(1000):
      super(MyTestCase, self).assertEqual(results[i], i + 1)

  # ------------------------------------------------------------------
  # 65. Concurrent chains with exception handlers
  # ------------------------------------------------------------------
  async def test_concurrent_chains_with_exception_handlers(self):
    caught_count = [0]
    async def maybe_raise(v):
      if v % 2 == 0:
        raise TestExc()
      return v
    def handler(v=None):
      caught_count[0] += 1
      return -1

    tasks = []
    for i in range(20):
      chain = Chain(i).then(maybe_raise).except_(handler, reraise=False)
      tasks.append(asyncio.ensure_future(await_(chain.run())))
    results = await asyncio.gather(*tasks)
    even_count = sum(1 for i in range(20) if i % 2 == 0)
    super(MyTestCase, self).assertEqual(caught_count[0], even_count)
    for i in range(20):
      if i % 2 == 0:
        super(MyTestCase, self).assertEqual(results[i], -1)
      else:
        super(MyTestCase, self).assertEqual(results[i], i)


# ===================================================================
# H. Stress Tests -- Memory and Cleanup (5+ tests)
# ===================================================================

class StressMemoryCleanupTests(MyTestCase):
  """Verify memory and resource cleanup under stress."""

  # ------------------------------------------------------------------
  # 66. Create and run 1000 chains -> verify no leak via registry size
  # ------------------------------------------------------------------
  async def test_1000_chains_registry_cleanup(self):
    initial_size = _get_registry_size()
    for i in range(1000):
      await await_(Chain(i).then(lambda v: v + 1).run())
    # Allow event loop to process done callbacks
    await asyncio.sleep(0.01)
    final_size = _get_registry_size()
    # Registry size should not grow unboundedly
    super(MyTestCase, self).assertLessEqual(final_size - initial_size, 100)

  # ------------------------------------------------------------------
  # 67. Create and run chains with ensure_future -> verify cleanup
  # ------------------------------------------------------------------
  async def test_ensure_future_cleanup(self):
    initial_size = _get_registry_size()
    tasks = []
    for i in range(100):
      async def work(v):
        return v * 2
      t = asyncio.ensure_future(await_(Chain(i).then(work).run()))
      tasks.append(t)
    await asyncio.gather(*tasks)
    await asyncio.sleep(0.01)
    final_size = _get_registry_size()
    super(MyTestCase, self).assertLessEqual(final_size - initial_size, 50)

  # ------------------------------------------------------------------
  # 68. Create many clones -> verify independence
  # ------------------------------------------------------------------
  async def test_many_clones_independence(self):
    base = Chain(0).then(lambda v: v + 1).then(lambda v: v * 2)
    clones = [base.clone() for _ in range(500)]
    # Modify the original
    base.then(lambda v: v + 1000)
    # All clones should produce the same, unmodified result
    for i, c in enumerate(clones):
      result = await await_(c.run())
      super(MyTestCase, self).assertEqual(result, 2, f'Clone {i} produced {result}')

  # ------------------------------------------------------------------
  # 69. Large foreach result collection (10000 elements)
  # ------------------------------------------------------------------
  async def test_large_foreach_result(self):
    items = list(range(10000))
    result = await await_(Chain(items).foreach(lambda x: x * 3).run())
    super(MyTestCase, self).assertEqual(len(result), 10000)
    super(MyTestCase, self).assertEqual(result[0], 0)
    super(MyTestCase, self).assertEqual(result[9999], 9999 * 3)

  # ------------------------------------------------------------------
  # 70. Deep nesting with large intermediate values
  # ------------------------------------------------------------------
  async def test_deep_nesting_large_intermediates(self):
    # Build a chain that creates a large list from an int, then measures its length.
    # Nesting means: 100 -> [0..99] -> len=100 at each level.
    inner = Chain().then(lambda v: list(range(v))).then(lambda v: len(v))
    for _ in range(4):
      prev = inner
      inner = Chain().then(prev).then(lambda v: list(range(v))).then(lambda v: len(v))
    result = await await_(inner.run(100))
    # Each level: int -> list(range(int)) -> len() -> int, always 100
    super(MyTestCase, self).assertEqual(result, 100)

  # ------------------------------------------------------------------
  # Extra: weakref cleanup of wrapper objects
  # ------------------------------------------------------------------
  async def test_weakref_cleanup(self):
    for _ in range(100):
      wrapper = _Ref(Chain(42).then(lambda v: v + 1))
      ref = weakref.ref(wrapper)
      result = wrapper.obj.run()
      super(MyTestCase, self).assertEqual(result, 43)
      del wrapper
    gc.collect()
    # The last ref should be None
    super(MyTestCase, self).assertIsNone(ref())


# ===================================================================
# I. Randomized/Parameterized Testing (10+ tests)
# ===================================================================

class RandomizedParameterizedTests(MyTestCase):
  """Test invariants across many different values using loops."""

  # ------------------------------------------------------------------
  # 71. Chain(x).then(double).run() == x*2 for various x
  # ------------------------------------------------------------------
  async def test_chain_double_invariant(self):
    for x in [0, 1, -1, 100, 0.5, -0.5, 10**6]:
      with self.subTest(x=x):
        result = await await_(Chain(x).then(lambda v: v * 2).run())
        super(MyTestCase, self).assertEqual(result, x * 2)

  # ------------------------------------------------------------------
  # 72. Chain(x).run() == x for various x
  # ------------------------------------------------------------------
  async def test_chain_identity_various(self):
    for x in [None, 0, False, '', [], {}, 42, 'hello', [1, 2, 3]]:
      with self.subTest(x=x):
        await self.assertEqual(Chain(x).run(), x)

  # ------------------------------------------------------------------
  # 73. Chain(items).filter(> 0).run() for various lists
  # ------------------------------------------------------------------
  async def test_filter_positive_various_lists(self):
    test_cases = [
      [],
      [1, 2, 3],
      [-1, 0, 1],
      [-5, -3, -1],
      [0, 0, 0],
      list(range(-50, 50)),
    ]
    for items in test_cases:
      with self.subTest(items=items):
        result = await await_(Chain(items).filter(lambda x: x > 0).run())
        expected = [x for x in items if x > 0]
        super(MyTestCase, self).assertEqual(result, expected)

  # ------------------------------------------------------------------
  # 74. Chain(items).foreach(square).run() for various lists
  # ------------------------------------------------------------------
  async def test_foreach_square_various_lists(self):
    test_cases = [
      [],
      [1],
      [1, 2, 3],
      list(range(20)),
      [-5, -3, 0, 3, 5],
    ]
    for items in test_cases:
      with self.subTest(items=items):
        result = await await_(Chain(items).foreach(lambda x: x ** 2).run())
        expected = [x ** 2 for x in items]
        super(MyTestCase, self).assertEqual(result, expected)

  # ------------------------------------------------------------------
  # 75. Cascade(x).then(double).run() == x for various x
  # ------------------------------------------------------------------
  async def test_cascade_double_returns_root_various(self):
    for x in [0, 1, -1, 42, 'hello', [1, 2], None, False, 3.14]:
      with self.subTest(x=x):
        result = await await_(Cascade(x).then(lambda v: 'ignored').run())
        super(MyTestCase, self).assertEqual(result, x)

  # ------------------------------------------------------------------
  # 76. Chain(x).clone().run() == Chain(x).run() for various x
  # ------------------------------------------------------------------
  async def test_clone_equivalence_various(self):
    for x in [0, 1, -1, 42, 'hello', [1, 2], None, False]:
      with self.subTest(x=x):
        c = Chain(x)
        r1 = await await_(c.clone().run())
        r2 = await await_(c.clone().run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 77. Chain(x).freeze().run() == Chain(x).run() for various x
  # ------------------------------------------------------------------
  async def test_freeze_equivalence_various(self):
    for x in [0, 1, -1, 42, 'hello', [1, 2], None, False]:
      with self.subTest(x=x):
        c = Chain(x)
        r1 = await await_(c.clone().run())
        r2 = await await_(c.freeze().run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 78. Chain().run(x) == Chain(x).run() for various x
  # ------------------------------------------------------------------
  async def test_override_equivalence_various(self):
    for x in [0, 1, -1, 42, 'hello', [1, 2], None, False]:
      with self.subTest(x=x):
        r1 = await await_(Chain().run(x))
        r2 = await await_(Chain(x).run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 79. Pipe equivalence for various chains
  # ------------------------------------------------------------------
  async def test_pipe_equivalence_various(self):
    for x in [0, 1, 5, 10, -3]:
      with self.subTest(x=x):
        r1 = await await_(Chain(x).then(lambda v: v + 1).run())
        r2 = await await_(Chain(x) | (lambda v: v + 1) | run())
        super(MyTestCase, self).assertEqual(r1, r2)

  # ------------------------------------------------------------------
  # 80. Exception invariants with various exception types
  # ------------------------------------------------------------------
  async def test_exception_invariants_various_types(self):
    exc_types = [ValueError, TypeError, RuntimeError, KeyError, IndexError]
    for exc_type in exc_types:
      with self.subTest(exc_type=exc_type.__name__):
        handler_called = [False]
        def handler(v=None):
          handler_called[0] = True
        def raiser():
          raise exc_type('test')
        Chain(raiser).except_(handler, exceptions=exc_type, reraise=False).run()
        super(MyTestCase, self).assertTrue(handler_called[0])


# ===================================================================
# J. Freelist Stress (4+ tests)
# ===================================================================

class FreelistStressTests(MyTestCase):
  """Exercise the Cython freelist allocations (Chain: 32, Link: 64)."""

  # ------------------------------------------------------------------
  # 81. Create and destroy 100 chains rapidly (freelist of size 32)
  # ------------------------------------------------------------------
  async def test_create_destroy_100_chains(self):
    for i in range(100):
      c = Chain(i).then(lambda v: v + 1)
      result = await await_(c.run())
      super(MyTestCase, self).assertEqual(result, i + 1)
      del c

  # ------------------------------------------------------------------
  # 82. Create and destroy 200 links rapidly (freelist of size 64)
  # ------------------------------------------------------------------
  async def test_create_destroy_200_links_via_chains(self):
    for i in range(200):
      c = Chain(i).then(lambda v: v + 1).then(lambda v: v * 2)
      result = await await_(c.run())
      super(MyTestCase, self).assertEqual(result, (i + 1) * 2)
      del c

  # ------------------------------------------------------------------
  # 83. Create chains, run them, create more -> freelist reuse
  # ------------------------------------------------------------------
  async def test_freelist_reuse_pattern(self):
    # Phase 1: create and destroy to fill freelist
    for i in range(50):
      c = Chain(i).then(lambda v: v + 1)
      await await_(c.run())
      del c
    # Phase 2: create more to reuse freelist entries
    for i in range(50):
      c = Chain(i).then(lambda v: v * 2)
      result = await await_(c.run())
      super(MyTestCase, self).assertEqual(result, i * 2)
      del c

  # ------------------------------------------------------------------
  # 84. Concurrent chain creation and destruction
  # ------------------------------------------------------------------
  async def test_concurrent_chain_creation_destruction(self):
    async def create_and_run(i):
      c = Chain(i).then(lambda v: v + 1)
      result = await await_(c.run())
      return result
    tasks = [asyncio.ensure_future(create_and_run(i)) for i in range(100)]
    results = await asyncio.gather(*tasks)
    for i in range(100):
      super(MyTestCase, self).assertEqual(results[i], i + 1)

  # ------------------------------------------------------------------
  # Extra: Cascade freelist stress
  # ------------------------------------------------------------------
  async def test_cascade_freelist_stress(self):
    for i in range(100):
      c = Cascade(i).then(lambda v: v * 100)
      result = await await_(c.run())
      super(MyTestCase, self).assertEqual(result, i)
      del c


# ===================================================================
# K. Boolean/Type Invariants (4+ tests)
# ===================================================================

class BooleanTypeInvariantsTests(MyTestCase):
  """Verify type-level invariants about chains."""

  # ------------------------------------------------------------------
  # 85. bool(Chain()) is always True
  # ------------------------------------------------------------------
  async def test_bool_always_true(self):
    chains = [
      Chain(),
      Chain(42),
      Chain(None),
      Chain(0),
      Chain(False),
      Chain(''),
      Chain([]),
      Chain().then(lambda v: v),
      Cascade(),
      Cascade(42),
    ]
    for c in chains:
      with self.subTest(chain=repr(c)):
        super(MyTestCase, self).assertTrue(bool(c))

  # ------------------------------------------------------------------
  # 86. repr(Chain()) never raises
  # ------------------------------------------------------------------
  async def test_repr_never_raises(self):
    chains = [
      Chain(),
      Chain(42),
      Chain(None),
      Chain(0).then(lambda v: v + 1),
      Chain('hello').then(lambda v: v.upper()).then(lambda v: v * 2),
      Cascade(),
      Cascade(42).then(lambda v: v * 2),
      Chain().then(lambda v: v).do(lambda v: None),
    ]
    for c in chains:
      with self.subTest(chain='chain'):
        # Should not raise any exception
        r = repr(c)
        super(MyTestCase, self).assertIsInstance(r, str)
        super(MyTestCase, self).assertGreater(len(r), 0)

  # ------------------------------------------------------------------
  # 87. chain.run() always returns a value or raises (never hangs)
  # ------------------------------------------------------------------
  async def test_run_returns_or_raises(self):
    # Test chains that return values
    chains_with_expected = [
      (Chain(), None),
      (Chain(42), 42),
      (Chain(0), 0),
      (Chain(None), None),
      (Chain(False), False),
      (Chain('hello'), 'hello'),
    ]
    for c, expected in chains_with_expected:
      with self.subTest(expected=expected):
        result = await await_(c.run())
        super(MyTestCase, self).assertEqual(result, expected)

    # Test chain that raises
    def raiser():
      raise TestExc()
    with self.assertRaises(TestExc):
      Chain(raiser).run()

  # ------------------------------------------------------------------
  # 88. Chain methods always return self (fluent API)
  # ------------------------------------------------------------------
  async def test_fluent_api_returns_self(self):
    c = Chain(42)
    # then
    result = c.then(lambda v: v + 1)
    super(MyTestCase, self).assertIs(result, c)
    # do
    result = c.do(lambda v: None)
    super(MyTestCase, self).assertIs(result, c)
    # except_
    result = c.except_(lambda v: None, reraise=False)
    super(MyTestCase, self).assertIs(result, c)
    # finally_
    result = c.finally_(lambda v: None)
    super(MyTestCase, self).assertIs(result, c)
    # foreach
    c2 = Chain([1, 2, 3])
    result = c2.foreach(lambda x: x)
    super(MyTestCase, self).assertIs(result, c2)
    # filter
    c3 = Chain([1, 2, 3])
    result = c3.filter(lambda x: True)
    super(MyTestCase, self).assertIs(result, c3)
    # gather
    c4 = Chain(1)
    result = c4.gather(lambda v: v)
    super(MyTestCase, self).assertIs(result, c4)
    # with_
    c5 = Chain(1)
    result = c5.with_(lambda v: v)
    super(MyTestCase, self).assertIs(result, c5)
    # sleep
    c6 = Chain(1)
    result = c6.sleep(0.001)
    super(MyTestCase, self).assertIs(result, c6)
    # config
    c7 = Chain(1)
    result = c7.config(debug=False)
    super(MyTestCase, self).assertIs(result, c7)
    # no_async
    c8 = Chain(1)
    result = c8.no_async()
    super(MyTestCase, self).assertIs(result, c8)
    # to_thread
    c9 = Chain(1)
    result = c9.to_thread(lambda v: v)
    super(MyTestCase, self).assertIs(result, c9)

  # ------------------------------------------------------------------
  # Extra: Cascade methods return self
  # ------------------------------------------------------------------
  async def test_cascade_fluent_api_returns_self(self):
    c = Cascade(42)
    result = c.then(lambda v: v + 1)
    super(MyTestCase, self).assertIs(result, c)
    result = c.do(lambda v: None)
    super(MyTestCase, self).assertIs(result, c)

  # ------------------------------------------------------------------
  # Extra: Chain with Null sentinel
  # ------------------------------------------------------------------
  async def test_null_sentinel_not_returned(self):
    """Null should never leak to user code. Chain().run() returns None, not Null."""
    result = Chain().run()
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertIsNot(result, Null)

  # ------------------------------------------------------------------
  # Extra: void chain returns None
  # ------------------------------------------------------------------
  async def test_void_chain_returns_none(self):
    result = await await_(Chain().then(lambda: 42).run())
    # void chain (no root) with a then that takes no args
    super(MyTestCase, self).assertEqual(result, 42)

  # ------------------------------------------------------------------
  # Extra: Chain(x).run() type preservation
  # ------------------------------------------------------------------
  async def test_type_preservation(self):
    test_values = [42, 'hello', 3.14, True, False, [1, 2], {'a': 1}, (1, 2)]
    for v in test_values:
      with self.subTest(v=v):
        result = await await_(Chain(v).run())
        super(MyTestCase, self).assertEqual(type(result), type(v))


# ===================================================================
# Additional invariant: do + Cascade interaction
# ===================================================================

class CascadeDoInteractionTests(MyTestCase):
  """Extra tests combining Cascade with do in various configurations."""

  async def test_cascade_do_then_returns_root(self):
    received_do = []
    received_then = []
    root = 99
    result = await await_(
      Cascade(root)
        .do(lambda v: received_do.append(v))
        .then(lambda v: received_then.append(v))
        .do(lambda v: received_do.append(v))
        .run()
    )
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(received_do, [root, root])
    super(MyTestCase, self).assertEqual(received_then, [root])

  async def test_cascade_multiple_do_all_receive_root(self):
    received = []
    root = 'root_val'
    result = await await_(
      Cascade(root)
        .do(lambda v: received.append(('do1', v)))
        .do(lambda v: received.append(('do2', v)))
        .do(lambda v: received.append(('do3', v)))
        .run()
    )
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(len(received), 3)
    for tag, v in received:
      super(MyTestCase, self).assertEqual(v, root)


# ===================================================================
# Additional: Complex chain pattern invariants
# ===================================================================

class ComplexChainPatternTests(MyTestCase):
  """Test complex combinations of chain features."""

  async def test_chain_then_foreach_filter_composition(self):
    """Chain -> transform -> foreach -> filter pipeline."""
    items = list(range(20))
    result = await await_(
      Chain(items)
        .foreach(lambda x: x * 2)
        .filter(lambda x: x > 10)
        .run()
    )
    expected = [x * 2 for x in items if x * 2 > 10]
    super(MyTestCase, self).assertEqual(result, expected)

  async def test_chain_filter_foreach_composition(self):
    """Chain -> filter -> foreach pipeline."""
    items = list(range(20))
    result = await await_(
      Chain(items)
        .filter(lambda x: x % 2 == 0)
        .foreach(lambda x: x * 10)
        .run()
    )
    expected = [x * 10 for x in items if x % 2 == 0]
    super(MyTestCase, self).assertEqual(result, expected)

  async def test_chain_with_do_does_not_affect_pipeline(self):
    """do() inserted between operations should not affect the pipeline result."""
    side_effects = []
    result = await await_(
      Chain(5)
        .then(lambda v: v + 1)
        .do(lambda v: side_effects.append(v))
        .then(lambda v: v * 2)
        .do(lambda v: side_effects.append(v))
        .run()
    )
    super(MyTestCase, self).assertEqual(result, 12)
    super(MyTestCase, self).assertEqual(side_effects, [6, 12])

  async def test_frozen_chain_reuse_many_times(self):
    """A frozen chain can be called many times with different inputs."""
    frozen = Chain().then(lambda v: v ** 2).then(lambda v: v + 1).freeze()
    for x in range(50):
      result = await await_(frozen(x))
      super(MyTestCase, self).assertEqual(result, x ** 2 + 1)

  async def test_clone_then_modify_independence(self):
    """Cloning then modifying original does not affect clone."""
    base = Chain(0).then(lambda v: v + 1)
    clone = base.clone()
    base.then(lambda v: v * 1000)
    r_clone = await await_(clone.run())
    r_base = await await_(base.run())
    super(MyTestCase, self).assertEqual(r_clone, 1)
    super(MyTestCase, self).assertEqual(r_base, 1000)

  async def test_gather_with_mixed_sync_async(self):
    """Gather with a mix of sync and async functions."""
    async def async_fn(v):
      return v * 10
    def sync_fn(v):
      return v * 20
    result = await await_(Chain(5).gather(async_fn, sync_fn, async_fn).run())
    super(MyTestCase, self).assertEqual(result, [50, 100, 50])

  async def test_chain_of_chains(self):
    """Chain containing nested Chain objects."""
    inner = Chain().then(lambda v: v + 10)
    outer = Chain(5).then(inner)
    result = await await_(outer.run())
    super(MyTestCase, self).assertEqual(result, 15)

  async def test_cascade_with_nested_chain(self):
    """Cascade containing a nested Chain -- result is still root."""
    inner = Chain().then(lambda v: v * 100)
    root = 7
    result = await await_(Cascade(root).then(inner).run())
    super(MyTestCase, self).assertEqual(result, root)
