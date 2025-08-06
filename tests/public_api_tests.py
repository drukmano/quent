import asyncio
import inspect
from unittest import IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_, TestExc
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException, run


class AsyncIter:
  def __init__(self, items):
    self.items = items
    self.index = 0

  def __aiter__(self):
    self.index = 0
    return self

  async def __anext__(self):
    if self.index >= len(self.items):
      raise StopAsyncIteration
    val = self.items[self.index]
    self.index += 1
    return val


class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr, msg=None):
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr, msg=None):
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first, second, msg=None):
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj, msg=None):
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1, expr2, msg=None):
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1, expr2, msg=None):
    return super().assertIsNot(await await_(expr1), expr2, msg)


# ---------------------------------------------------------------------------
# Class 1: SuppressTests
# ---------------------------------------------------------------------------
class SuppressTests(MyTestCase):

  async def test_suppress_default(self):
    def raise_val(v=None):
      raise ValueError("boom")
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn, 1).then(raise_val).suppress().run()
        )

  async def test_suppress_not_base_exception(self):
    class MyBaseExc(BaseException):
      pass
    def raise_base(v=None):
      raise MyBaseExc("base")
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(MyBaseExc):
          await await_(
            Chain(fn, 1).then(raise_base).suppress().run()
          )

  async def test_suppress_specific_exceptions(self):
    def raise_val(v=None):
      raise ValueError("boom")
    def raise_type(v=None):
      raise TypeError("boom")
    for fn, ctx in self.with_fn():
      with ctx:
        # Suppresses ValueError
        await self.assertIsNone(
          Chain(fn, 1).then(raise_val).suppress(ValueError).run()
        )
        # Does NOT suppress TypeError when only ValueError is specified
        with self.assertRaises(TypeError):
          await await_(
            Chain(fn, 1).then(raise_type).suppress(ValueError).run()
          )

  async def test_suppress_no_error(self):
    obj_ = object()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIs(
          Chain(fn, obj_).suppress().run(), obj_
        )

  async def test_suppress_async(self):
    async def raise_value_error(v=None):
      raise ValueError("async boom")
    await self.assertIsNone(
      Chain(1).then(raise_value_error).suppress().run()
    )

  async def test_suppress_multiple_exceptions(self):
    def raise_value(v=None):
      raise ValueError("v")
    def raise_type(v=None):
      raise TypeError("t")
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn, 1).then(raise_value).suppress(ValueError, TypeError).run()
        )
        await self.assertIsNone(
          Chain(fn, 1).then(raise_type).suppress(ValueError, TypeError).run()
        )


# ---------------------------------------------------------------------------
# Class 2: OnSuccessTests
# ---------------------------------------------------------------------------
class OnSuccessTests(MyTestCase):

  async def test_on_success_called(self):
    called = [False]
    def cb(v):
      called[0] = True
    for fn, ctx in self.with_fn():
      with ctx:
        called[0] = False
        await await_(Chain(fn, 42).on_success(cb).run())
        super(MyTestCase, self).assertTrue(called[0])

  async def test_on_success_receives_value(self):
    received = [None]
    def cb(v):
      received[0] = v
    obj_ = object()
    for fn, ctx in self.with_fn():
      with ctx:
        received[0] = None
        await await_(Chain(fn, obj_).on_success(cb).run())
        super(MyTestCase, self).assertIs(received[0], obj_)

  async def test_on_success_return_ignored(self):
    obj_ = object()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIs(
          Chain(fn, obj_).on_success(lambda v: "ignored_return").run(), obj_
        )

  async def test_on_success_not_called_on_exception(self):
    called = [False]
    def cb(v):
      called[0] = True
    for fn, ctx in self.with_fn():
      with ctx:
        called[0] = False
        def raiser(v=None):
          raise ValueError("fail")
        await await_(
          Chain(fn).then(raiser).except_(lambda v: None, reraise=False).on_success(cb).run()
        )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_on_success_duplicate_raises(self):
    with self.assertRaises(QuentException):
      Chain().on_success(lambda v: v).on_success(lambda v: v)

  async def test_on_success_async_callback(self):
    called = [False]
    async def cb(v):
      called[0] = True
    for fn, ctx in self.with_fn():
      with ctx:
        called[0] = False
        await await_(Chain(fn, 10).on_success(cb).run())
        super(MyTestCase, self).assertTrue(called[0])

  async def test_on_success_with_finally(self):
    order = []
    def on_success_cb(v):
      order.append('success')
    def on_finally_cb(v):
      order.append('finally')
    for fn, ctx in self.with_fn():
      with ctx:
        order.clear()
        await await_(
          Chain(fn, 1).on_success(on_success_cb).finally_(on_finally_cb).run()
        )
        super(MyTestCase, self).assertEqual(order, ['success', 'finally'])

  async def test_on_success_with_cascade(self):
    received = [None]
    def cb(v):
      received[0] = v
    obj_ = object()
    for fn, ctx in self.with_fn():
      with ctx:
        received[0] = None
        await await_(
          Cascade(obj_).then(fn).then(lambda v: "other").on_success(cb).run()
        )
        super(MyTestCase, self).assertIs(received[0], obj_)


# ---------------------------------------------------------------------------
# Class 3: FilterTests
# ---------------------------------------------------------------------------
class FilterTests(MyTestCase):

  async def test_filter_sync(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).run(),
          [2, 4]
        )

  async def test_filter_async_iterable(self):
    result = await await_(
      Chain(AsyncIter([1, 2, 3, 4, 5])).filter(lambda x: x > 3).run()
    )
    super(MyTestCase, self).assertEqual(result, [4, 5])

  async def test_filter_async_predicate(self):
    async def is_even(x):
      return x % 2 == 0
    result = await await_(
      Chain([10, 11, 12, 13, 14]).filter(is_even).run()
    )
    super(MyTestCase, self).assertEqual(result, [10, 12, 14])

  async def test_filter_empty(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).filter(lambda x: True).run(),
          []
        )

  async def test_filter_all_pass(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: True).run(),
          [1, 2, 3]
        )

  async def test_filter_none_pass(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: False).run(),
          []
        )


# ---------------------------------------------------------------------------
# Class 4: ReduceTests
# ---------------------------------------------------------------------------
class ReduceTests(MyTestCase):

  async def test_reduce_with_initial(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).reduce(lambda acc, x: acc + x, 10).run(),
          16
        )

  async def test_reduce_without_initial(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).reduce(lambda acc, x: acc + x).run(),
          10
        )

  async def test_reduce_empty_raises(self):
    with self.assertRaises(TypeError):
      await await_(
        Chain([]).reduce(lambda acc, x: acc + x).run()
      )

  async def test_reduce_single_element(self):
    called = [False]
    def reducer(acc, x):
      called[0] = True
      return acc + x
    for fn, ctx in self.with_fn():
      with ctx:
        called[0] = False
        await self.assertEqual(
          Chain(fn, [42]).reduce(reducer).run(),
          42
        )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_reduce_with_initial_empty(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).reduce(lambda acc, x: acc + x, 99).run(),
          99
        )

  async def test_reduce_async_iterable(self):
    result = await await_(
      Chain(AsyncIter([1, 2, 3, 4])).reduce(lambda acc, x: acc + x, 0).run()
    )
    super(MyTestCase, self).assertEqual(result, 10)

  async def test_reduce_async_reducer(self):
    async def async_add(acc, x):
      return acc + x
    result = await await_(
      Chain([5, 10, 15]).reduce(async_add, 0).run()
    )
    super(MyTestCase, self).assertEqual(result, 30)


# ---------------------------------------------------------------------------
# Class 5: GatherTests
# ---------------------------------------------------------------------------
class GatherTests(MyTestCase):

  async def test_gather_all_sync(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather(
            lambda v: v + 1,
            lambda v: v * 2,
            lambda v: v - 3,
          ).run(),
          [6, 10, 2]
        )

  async def test_gather_all_async(self):
    async def add1(v):
      return v + 1
    async def mul2(v):
      return v * 2
    async def sub3(v):
      return v - 3
    result = await await_(
      Chain(5).gather(add1, mul2, sub3).run()
    )
    super(MyTestCase, self).assertEqual(result, [6, 10, 2])

  async def test_gather_mixed(self):
    async def async_mul(v):
      return v * 10
    result = await await_(
      Chain(3).gather(
        lambda v: v + 1,
        async_mul,
      ).run()
    )
    super(MyTestCase, self).assertEqual(result, [4, 30])

  async def test_gather_single_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 7).gather(lambda v: v * 3).run(),
          [21]
        )

  async def test_gather_receives_value(self):
    obj_ = object()
    received = []
    def collector(v):
      received.append(v)
      return v
    for fn, ctx in self.with_fn():
      with ctx:
        received.clear()
        await await_(
          Chain(fn, obj_).gather(collector, collector).run()
        )
        super(MyTestCase, self).assertEqual(len(received), 2)
        super(MyTestCase, self).assertIs(received[0], obj_)
        super(MyTestCase, self).assertIs(received[1], obj_)


# ---------------------------------------------------------------------------
# Class 6: PipeMethodTests
# ---------------------------------------------------------------------------
class PipeMethodTests(MyTestCase):

  async def test_pipe_callable(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).pipe(lambda v: v * 2).run(),
          10
        )

  async def test_pipe_chain(self):
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(fn).then(lambda v: v + 100)
        await self.assertEqual(
          Chain(fn, 5).pipe(inner).run(),
          105
        )

  async def test_pipe_returns_self(self):
    c = Chain(10)
    result = c.pipe(lambda v: v)
    super(MyTestCase, self).assertIs(result, c)


# ---------------------------------------------------------------------------
# Class 7: ComposeTests
# ---------------------------------------------------------------------------
class ComposeTests(MyTestCase):

  async def test_compose_callables(self):
    composed = Chain.compose(
      lambda v: v + 1,
      lambda v: v * 3,
    )
    await self.assertEqual(composed.run(2), 9)

  async def test_compose_chains(self):
    c1 = Chain().then(lambda v: v + 10)
    c2 = Chain().then(lambda v: v * 2)
    composed = Chain.compose(c1, c2)
    await self.assertEqual(composed.run(5), 30)

  async def test_compose_empty(self):
    composed = Chain.compose()
    await self.assertIsNone(composed.run())

  async def test_compose_single(self):
    composed = Chain.compose(lambda v: v ** 2)
    await self.assertEqual(composed.run(4), 16)

  async def test_compose_returns_chain(self):
    composed = Chain.compose(lambda v: v)
    super(MyTestCase, self).assertIsInstance(composed, Chain)


# ---------------------------------------------------------------------------
# Class 8: IfRaiseElseRaiseTests
# ---------------------------------------------------------------------------
class IfRaiseElseRaiseTests(MyTestCase):

  async def test_if_raise_truthy(self):
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, True).if_raise(ValueError("truthy")).run()
          )

  async def test_if_raise_falsy(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertFalse(
          Chain(fn, False).if_raise(ValueError("should not raise")).run()
        )

  async def test_else_raise_falsy(self):
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(RuntimeError):
          await await_(
            Chain(fn, False).if_(lambda v: "yes").else_raise(RuntimeError("falsy path")).run()
          )

  async def test_if_not_raise(self):
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, False).if_not_raise(ValueError("falsy")).run()
          )
        # Truthy value should NOT raise
        await self.assertTrue(
          Chain(fn, True).if_not_raise(ValueError("should not raise")).run()
        )

  async def test_combined_pattern(self):
    for fn, ctx in self.with_fn():
      with ctx:
        # Truthy path triggers if_raise
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, True).if_raise(ValueError("if_true")).run()
          )
        # Falsy path triggers else_raise
        with self.assertRaises(RuntimeError):
          await await_(
            Chain(fn, False).if_(lambda v: "yes").else_raise(RuntimeError("else_path")).run()
          )


# ---------------------------------------------------------------------------
# Class 9: ReprTests
# ---------------------------------------------------------------------------
class ReprTests(MyTestCase):

  async def test_repr_empty_chain(self):
    c = Chain()
    r = repr(c)
    super(MyTestCase, self).assertTrue(r.startswith('Chain()'))

  async def test_repr_with_root(self):
    def my_root_fn():
      pass
    c = Chain(my_root_fn)
    r = repr(c)
    super(MyTestCase, self).assertIn('my_root_fn', r)

  async def test_repr_with_operations(self):
    c = Chain(10).then(lambda v: v + 1).then(lambda v: v * 2)
    r = repr(c)
    super(MyTestCase, self).assertIn('.then(', r)

  async def test_repr_cascade(self):
    c = Cascade()
    r = repr(c)
    super(MyTestCase, self).assertTrue(r.startswith('Cascade'))

  async def test_repr_chain_attr(self):
    c = ChainAttr()
    r = repr(c)
    super(MyTestCase, self).assertTrue(r.startswith('ChainAttr'))


# ---------------------------------------------------------------------------
# Class 10: ForeachIndexedTests
# ---------------------------------------------------------------------------
class ForeachIndexedTests(MyTestCase):

  async def test_foreach_indexed_sync(self):
    results = []
    def collect(idx, el):
      results.append((idx, el))
      return (idx, el)
    for fn, ctx in self.with_fn():
      with ctx:
        results.clear()
        await await_(
          Chain(fn, [10, 20, 30]).foreach(collect, with_index=True).run()
        )
        super(MyTestCase, self).assertEqual(results, [(0, 10), (1, 20), (2, 30)])

  async def test_foreach_indexed_async(self):
    results = []
    def collect(idx, el):
      results.append((idx, el))
      return (idx, el)
    results.clear()
    result = await await_(
      Chain(AsyncIter([10, 20, 30])).foreach(collect, with_index=True).run()
    )
    super(MyTestCase, self).assertEqual(results, [(0, 10), (1, 20), (2, 30)])

  async def test_foreach_indexed_values_correct(self):
    indices = []
    def track_index(idx, el):
      indices.append(idx)
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        indices.clear()
        await await_(
          Chain(fn, ['a', 'b', 'c', 'd', 'e']).foreach(track_index, with_index=True).run()
        )
        super(MyTestCase, self).assertEqual(indices, [0, 1, 2, 3, 4])

  async def test_foreach_indexed_break(self):
    def stop_at_2(idx, el):
      if idx == 2:
        return Chain.break_()
      return el * 10
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).foreach(stop_at_2, with_index=True).run(),
          [10, 20]
        )

  async def test_foreach_indexed_exception(self):
    def raiser(idx, el):
      if el == 3:
        raise ValueError("bad element")
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, [1, 2, 3]).foreach(raiser, with_index=True).run()
          )

  async def test_foreach_do_indexed(self):
    side_effects = []
    def track(idx, el):
      side_effects.append((idx, el))
      return "ignored"
    for fn, ctx in self.with_fn():
      with ctx:
        side_effects.clear()
        await self.assertEqual(
          Chain(fn, [10, 20, 30]).foreach_do(track, with_index=True).run(),
          [10, 20, 30]
        )
        super(MyTestCase, self).assertEqual(side_effects, [(0, 10), (1, 20), (2, 30)])

  async def test_foreach_do_indexed_async(self):
    side_effects = []
    async def track(idx, el):
      side_effects.append((idx, el))
      return "ignored"
    side_effects.clear()
    result = await await_(
      Chain(AsyncIter([100, 200])).foreach_do(track, with_index=True).run()
    )
    super(MyTestCase, self).assertEqual(result, [100, 200])
    super(MyTestCase, self).assertEqual(side_effects, [(0, 100), (1, 200)])
