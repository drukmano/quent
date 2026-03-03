import io
import logging
from unittest import TestCase

from tests.utils import empty, aempty, await_, MyTestCase
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# RootlessCloneWalkTests
# ---------------------------------------------------------------------------
class RootlessCloneWalkTests(TestCase):
  """Clone of rootless chains exercising the walk loop in clone() lines 518-521.

  The walk loop in clone() finds current_link for rootless chains:
    walk = new_chain.first_link
    while walk.next_link is not None:   # line 519
      walk = walk.next_link             # line 520
    new_chain.current_link = walk if walk is not new_chain.first_link else None

  - 1 link: loop body never executes; current_link = None
  - 2 links: loop body executes once; current_link = link2
  - 3+ links: loop body executes multiple times (line 520 hit >1x)

  Rootless chains with no root override call the first link with no arguments
  (evaluate_value sees current_value=Null and calls link.v()). Use no-arg
  lambdas for the first link, or run with a root override.
  """

  def test_rootless_clone_1_link(self):
    """Rootless chain with 1 link: walk loop exits immediately, current_link=None."""
    c = Chain().then(lambda: 'one')
    c2 = c.clone()
    self.assertEqual(c2.run(), 'one')
    # Appending to clone should work (current_link=None means first_link is head)
    c2.then(lambda v: v + '_extra')
    self.assertEqual(c2.run(), 'one_extra')

  def test_rootless_clone_2_links(self):
    """Rootless chain with 2 links: walk loop body executes once."""
    c = Chain().then(lambda: 10).then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c2.run(), 20)
    # Verify clone independence: appending to original doesn't affect clone
    c.then(lambda v: v + 1000)
    self.assertEqual(c2.run(), 20)

  def test_rootless_clone_3_links(self):
    """Rootless chain with 3 links: walk loop body executes twice (line 520 hit >1x)."""
    c = Chain().then(lambda: 5).then(lambda v: v * 2).then(lambda v: v + 1)
    c2 = c.clone()
    self.assertEqual(c2.run(), 11)
    # Verify clone independence
    c.then(lambda v: v * 100)
    self.assertEqual(c2.run(), 11)

  def test_rootless_clone_4_links(self):
    """Rootless chain with 4 links: walk loop body executes 3 times."""
    c = (
      Chain()
      .then(lambda: 1)
      .then(lambda v: v + 1)
      .then(lambda v: v * 3)
      .then(lambda v: v - 2)
    )
    c2 = c.clone()
    self.assertEqual(c2.run(), 4)  # 1 -> 2 -> 6 -> 4

  def test_rootless_clone_append_after_clone(self):
    """Appending to clone after cloning a 3-link rootless chain works correctly."""
    c = Chain().then(lambda: 1).then(lambda v: v + 1).then(lambda v: v * 10)
    c2 = c.clone()
    c2.then(lambda v: v + 5)
    self.assertEqual(c2.run(), 25)  # 1 -> 2 -> 20 -> 25
    self.assertEqual(c.run(), 20)   # original unchanged

  def test_rootless_clone_no_links(self):
    """Rootless chain with no links: first_link is None, clone branch at line 522-524."""
    c = Chain()
    c2 = c.clone()
    self.assertIsNone(c2.run())

  def test_rootless_clone_with_root_override(self):
    """Rootless clone with 3 links, run with a root value override."""
    c = Chain().then(lambda v: v * 2).then(lambda v: v + 1).then(lambda v: v * 3)
    c2 = c.clone()
    self.assertEqual(c2.run(5), 33)  # 5 -> 10 -> 11 -> 33


# ---------------------------------------------------------------------------
# BreakNonNestedTests
# ---------------------------------------------------------------------------
class BreakNonNestedTests(MyTestCase):
  """_Break in non-nested context raises QuentException.

  These tests cover the except _Break clauses in:
  - _run (line 182-185): non-simple sync path
  - _run_async (line 299-302): non-simple async path
  - _run_simple (line 411-414): simple sync path
  - _run_async_simple (line 459-462): simple async path
  """

  async def test_break_non_nested_nonsimple_sync(self):
    """_Break in non-nested, non-simple sync path (_run lines 182-185).

    Using .do() makes the chain non-simple. Chain.break_ raises _Break.
    _run catches it, sees is_nested=False, raises QuentException.
    """
    with self.assertRaises(QuentException) as cm:
      Chain(1).do(lambda v: None).then(Chain.break_).run()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_non_nested_nonsimple_async(self):
    """_Break in non-nested, non-simple async path (_run_async lines 299-302).

    Using .do(aempty) forces async transition. Then Chain.break_ raises _Break
    in _run_async. It catches it, sees is_nested=False, raises QuentException.
    """
    with self.assertRaises(QuentException) as cm:
      await await_(Chain(1).do(aempty).then(Chain.break_).run())
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_non_nested_simple_sync(self):
    """_Break in non-nested, simple sync path (_run_simple lines 411-414).

    Chain().then(Chain.break_) is simple. _run delegates to _run_simple.
    _Break is caught, is_nested=False, raises QuentException.
    """
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_).run()

  async def test_break_non_nested_simple_async(self):
    """_Break in non-nested, simple async path (_run_async_simple lines 459-462).

    Chain(aempty).then(Chain.break_) is simple. The async root triggers
    _run_async_simple. _Break is caught, is_nested=False, raises QuentException.
    """
    with self.assertRaises(QuentException):
      await await_(Chain(aempty).then(Chain.break_).run())


# ---------------------------------------------------------------------------
# BreakNestedPropagationTests
# ---------------------------------------------------------------------------
class BreakNestedPropagationTests(MyTestCase):
  """_Break in nested chains: re-raised if is_nested=True.

  When a nested chain catches _Break and is_nested=True, it re-raises
  so the parent chain can handle it. The parent chain then catches _Break
  in its own except clause. If the parent is not nested, it raises
  QuentException (proving the _Break propagated through the nested chain).

  Covers:
  - _run_simple line 412-413: nested simple sync re-raise
  - _run_async_simple line 460-461: nested simple async re-raise
  - _run line 182-184: nested non-simple sync re-raise
  - _run_async line 299-301: nested non-simple async re-raise
  """

  async def test_break_nested_simple_sync_propagates(self):
    """_Break re-raised from nested simple chain (_run_simple line 412-413).

    Inner chain is simple (only .then links). .then(inner) sets is_nested=True.
    evaluate_value calls inner._run(current_value, None, None, True).
    Inner's _run_simple catches _Break, sees is_nested=True, re-raises.
    Outer's _run catches _Break, not nested, raises QuentException.
    """
    inner = Chain().then(Chain.break_)
    with self.assertRaises(QuentException) as cm:
      Chain(1).then(inner).run()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_nested_simple_async_propagates(self):
    """_Break re-raised from nested simple async chain (_run_async_simple line 460-461).

    Inner chain: Chain().then(aempty).then(Chain.break_). The rootless inner
    chain receives the outer value as root override, then aempty triggers
    _run_async_simple. _Break caught, is_nested=True, re-raises.
    Outer chain catches _Break and raises QuentException.

    Note: the inner chain must be rootless so it can accept the outer chain's
    value as a root override. Using Chain(aempty) would conflict because it
    already has a root value.
    """
    inner = Chain().then(aempty).then(Chain.break_)
    with self.assertRaises(QuentException) as cm:
      await await_(Chain(1).then(inner).run())
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_nested_nonsimple_sync_propagates(self):
    """_Break re-raised from nested non-simple sync chain (_run lines 182-184).

    Inner chain uses .do() to be non-simple. is_nested=True.
    Inner's _run catches _Break, sees is_nested=True, re-raises.
    """
    inner = Chain().do(empty).then(Chain.break_)
    with self.assertRaises(QuentException) as cm:
      Chain(1).then(inner).run()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_nested_nonsimple_async_propagates(self):
    """_Break re-raised from nested non-simple async chain (_run_async lines 299-301).

    Inner chain uses .do(aempty) to force non-simple + async.
    Inner's _run_async catches _Break, is_nested=True, re-raises.
    """
    inner = Chain().do(aempty).then(Chain.break_)
    with self.assertRaises(QuentException) as cm:
      await await_(Chain(1).then(inner).run())
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_nested_propagates_through_foreach(self):
    """_Break from nested chain inside foreach stops iteration.

    When Chain.break_ is called directly in the foreach lambda (not via
    a nested chain), _Foreach catches _Break and returns partial results.
    """
    await self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(
        lambda x: Chain.break_('stopped') if x == 3 else x
      ).run(),
      'stopped'
    )


# ---------------------------------------------------------------------------
# RootOverrideTests
# ---------------------------------------------------------------------------
class RootOverrideTests(MyTestCase):
  """Root override error paths and temp_root_link creation.

  Covers:
  - _run lines 116-118: root override with existing root (non-simple)
  - _run_simple lines 353-355: root override with existing root (simple)
  - _run lines 122-126: temp_root_link creation (non-simple)
  """

  async def test_root_override_error_nonsimple_sync(self):
    """Cannot override root of a chain that already has a root (_run line 117-118).

    .do() makes it non-simple so it goes through _run (not _run_simple).
    """
    with self.assertRaises(QuentException) as cm:
      Chain(1).do(lambda v: None).then(lambda v: v).run(2)
    super(MyTestCase, self).assertIn('override', str(cm.exception).lower())

  async def test_root_override_error_simple_sync(self):
    """Cannot override root in simple path (_run_simple line 353-355)."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).then(lambda v: v).run(2)
    super(MyTestCase, self).assertIn('override', str(cm.exception).lower())

  async def test_temp_root_link_nonsimple(self):
    """Void chain with root override in non-simple path (_run lines 122-126).

    Chain().do(lambda v: None).then(lambda v: v * 2).run(5):
    - is_root_value_override=True, has_root_value=False
    - Creates temp_root_link, sets link=temp_root_link, link.next_link=first_link
    - Evaluates temp root (5), then evaluates the chain links
    """
    result = Chain().do(lambda v: None).then(lambda v: v * 2).run(5)
    await self.assertEqual(result, 10)

  async def test_temp_root_link_nonsimple_with_args(self):
    """Void chain root override with callable + args in non-simple path."""
    result = Chain().do(lambda v: None).then(lambda v: v + 1).run(lambda a, b: a + b, 3, 7)
    await self.assertEqual(result, 11)  # root = 3+7=10, then 10+1=11

  async def test_temp_root_link_nonsimple_async(self):
    """Void chain root override in non-simple async path."""
    result = await await_(
      Chain().do(aempty).then(lambda v: v * 3).run(5)
    )
    await self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# DebugLazyInitTests
# ---------------------------------------------------------------------------
class DebugLazyInitTests(MyTestCase):
  """Debug link_results lazy initialization.

  In _run, link_results starts as None. When _debug=True:
  - Root evaluation initializes it at lines 138-140
  - Link evaluation checks/initializes at lines 164-165

  For a rootless chain with debug=True going through _run (non-simple via
  .do()), the first link evaluation hits the lazy init at lines 164-165
  because no root was evaluated to initialize it earlier.

  Note: debug=True forces the non-simple _run path (line 104 condition).
  """

  async def test_debug_lazy_init_rootless_nonsimple_sync(self):
    """link_results lazy init at _run line 164-165 for rootless non-simple chain.

    Chain().do(empty).then(lambda: 42).config(debug=True).run()
    - No root => link_results stays None after root section
    - do(empty) gets called with Null (no-arg call via evaluate_value)
    - First link evaluation with debug: link_results is None => lazy init
    """
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      result = Chain().do(empty).then(lambda: 42).config(debug=True).run()
      await self.assertEqual(result, 42)
      log_output = stream.getvalue()
      # Debug should have logged link evaluation results
      super(MyTestCase, self).assertTrue(len(log_output) > 0)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  async def test_debug_lazy_init_rootless_nonsimple_async(self):
    """link_results lazy init at _run_async line 255-256 or 279-280.

    Chain().do(aempty).then(lambda: 99).config(debug=True).run()
    - do(aempty) returns coroutine, triggers _run_async transition
    - In _run_async, link_results is None when first link evaluated
    - Lazy init happens at lines 255-256 or 279-280

    Note: _run_async does not call _logger.debug(), so we cannot assert
    on log output for the async path. The lazy init is still exercised
    (link_results dict is created and populated) — verified by the chain
    producing the correct result without errors.
    """
    result = await await_(
      Chain().do(aempty).then(lambda: 99).config(debug=True).run()
    )
    await self.assertEqual(result, 99)

  async def test_debug_with_root_initializes_link_results(self):
    """With a root, link_results is initialized at root eval (_run line 138-140).

    Subsequent link evaluations at line 164 see link_results is not None.
    """
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      result = Chain(10).do(lambda v: None).then(lambda v: v + 5).config(debug=True).run()
      await self.assertEqual(result, 15)
      log_output = stream.getvalue()
      # Should log root value
      super(MyTestCase, self).assertIn('10', log_output)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)


# ---------------------------------------------------------------------------
# ReturnBreakClassmethodTests
# ---------------------------------------------------------------------------
class ReturnBreakClassmethodTests(MyTestCase):
  """Tests for Chain.return_() and Chain.break_() classmethods.

  Covers lines 629-637 of _chain_core.pxi.
  """

  async def test_return_with_value(self):
    """Chain.return_(value) exits the chain and returns the value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(v * 10)).then(lambda v: v * 100).run(),
          10
        )

  async def test_return_without_value(self):
    """Chain.return_() with no value returns None."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn, 1).then(lambda v: Chain.return_()).then(lambda v: v * 100).run()
        )

  async def test_return_with_callable_value(self):
    """Chain.return_(callable, args) evaluates the callable."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(lambda a, b: a + b, 3, 7)).run(),
          10
        )

  async def test_break_with_value_in_foreach(self):
    """Chain.break_(value) in foreach returns that value as the foreach result."""
    sentinel = object()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIs(
          Chain(fn, [1, 2, 3]).foreach(
            lambda x: Chain.break_(sentinel) if x == 2 else x
          ).run(),
          sentinel
        )

  async def test_break_without_value_in_foreach(self):
    """Chain.break_() with no value in foreach returns the accumulated list."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(
            lambda x: Chain.break_() if x == 3 else x
          ).run(),
          [1, 2]
        )

  async def test_break_with_callable_value_in_foreach(self):
    """Chain.break_(callable, args) evaluates the callable as the break value."""
    await self.assertEqual(
      Chain([1, 2, 3]).foreach(
        lambda x: Chain.break_(lambda a, b: a + b, 10, 20) if x == 2 else x
      ).run(),
      30
    )

  async def test_return_classmethod_on_cascade(self):
    """Cascade.return_() works the same as Chain.return_() (inherited classmethod)."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(lambda v: Cascade.return_(v * 2)).then(lambda v: 999).run(),
          10
        )

  async def test_break_classmethod_on_cascade(self):
    """Cascade.break_() works the same as Chain.break_() (inherited classmethod)."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(
            lambda x: Cascade.break_('done') if x == 2 else x
          ).run(),
          'done'
        )


# ---------------------------------------------------------------------------
# NestedDirectRunTests
# ---------------------------------------------------------------------------
class NestedDirectRunTests(MyTestCase):
  """Nested chain direct run prevention.

  When a chain is used as a nested sub-chain (is_nested=True), calling
  .run() directly raises QuentException. This is tested for both the
  simple path (_run_simple line 344-345) and the non-simple path
  (_run line 102-103).
  """

  async def test_nested_direct_run_simple_path(self):
    """Directly running a nested simple chain raises QuentException (_run_simple line 344-345).

    inner is simple (only .then links). After being nested in outer,
    inner.is_nested=True. inner.run() -> _run -> _run_simple (since simple),
    which checks invoked_by_parent_chain=False and is_nested=True -> raises.
    """
    inner = Chain().then(lambda v: v * 2)
    _outer = Chain(1).then(inner)  # This sets inner.is_nested = True
    with self.assertRaises(QuentException) as cm:
      inner.run(5)
    super(MyTestCase, self).assertIn('nested', str(cm.exception).lower())

  async def test_nested_direct_run_nonsimple_path(self):
    """Directly running a nested non-simple chain raises QuentException (_run line 102-103).

    inner uses .do() so it's non-simple. After nesting, inner.is_nested=True.
    inner.run() -> _run checks invoked_by_parent_chain=False and is_nested=True -> raises.
    """
    inner = Chain().do(empty).then(lambda v: v * 2)
    _outer = Chain(1).then(inner)  # sets inner.is_nested = True
    with self.assertRaises(QuentException) as cm:
      inner.run(5)
    super(MyTestCase, self).assertIn('nested', str(cm.exception).lower())

  async def test_nested_direct_call_raises(self):
    """Directly calling a nested chain via __call__ also raises."""
    inner = Chain().then(lambda v: v + 1)
    _outer = Chain(1).then(inner)
    with self.assertRaises(QuentException):
      inner(5)

  async def test_clone_of_nested_can_run_directly(self):
    """Cloning a nested chain resets is_nested, allowing direct run."""
    inner = Chain().then(lambda v: v * 2)
    _outer = Chain(1).then(inner)
    # inner is now nested and cannot be run directly
    with self.assertRaises(QuentException):
      inner.run(5)
    # clone resets is_nested
    inner_clone = inner.clone()
    await self.assertEqual(inner_clone.run(5), 10)
