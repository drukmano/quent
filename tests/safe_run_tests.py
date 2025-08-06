import unittest
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from quent import Chain, Cascade


class SafeRunBasicTests(unittest.TestCase):
  def test_safe_run_returns_same_result_as_run(self):
    c = Chain(5).then(lambda v: v * 2)
    self.assertEqual(c.safe_run(), c.run())

  def test_safe_run_with_root_override(self):
    c = Chain().then(lambda v: v + 1)
    self.assertEqual(c.safe_run(10), 11)

  def test_safe_run_preserves_original_chain(self):
    c = Chain(1).then(lambda v: v + 1)
    c.safe_run()
    c.safe_run()
    self.assertEqual(c.run(), 2)

  def test_safe_run_cascade(self):
    results = []
    c = Cascade(10).do(lambda v: results.append(v))
    c.safe_run()
    self.assertEqual(results, [10])

  def test_safe_run_with_then_chain(self):
    c = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3)
    self.assertEqual(c.safe_run(), 6)

  def test_safe_run_void_chain(self):
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c.safe_run(5), 10)


class SafeRunThreadTests(unittest.TestCase):
  def test_concurrent_safe_run_threads(self):
    c = Chain(1).then(lambda v: v + 1)
    results = []
    errors = []

    def worker():
      try:
        results.append(c.safe_run())
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(len(errors), 0, f"Errors: {errors}")
    self.assertEqual(results, [2] * 10)

  def test_concurrent_safe_run_threadpool(self):
    c = Chain(1).then(lambda v: v * 3)
    with ThreadPoolExecutor(max_workers=5) as pool:
      futures = [pool.submit(c.safe_run) for _ in range(20)]
      results = [f.result() for f in as_completed(futures)]
    self.assertEqual(sorted(results), [3] * 20)

  def test_concurrent_with_stateful_chain(self):
    counter = {'value': 0}
    lock = threading.Lock()

    def increment(v):
      with lock:
        counter['value'] += 1
      return v + counter['value']

    c = Chain(0).then(increment)
    with ThreadPoolExecutor(max_workers=4) as pool:
      futures = [pool.submit(c.safe_run) for _ in range(10)]
      results = [f.result() for f in futures]
    self.assertEqual(len(results), 10)


class SafeRunAsyncTests(unittest.IsolatedAsyncioTestCase):
  async def test_concurrent_safe_run_async(self):
    c = Chain(1).then(lambda v: v + 1)
    tasks = [asyncio.to_thread(c.safe_run) for _ in range(10)]
    results = await asyncio.gather(*tasks)
    self.assertEqual(list(results), [2] * 10)

  async def test_safe_run_with_async_chain(self):
    async def double(v):
      return v * 2
    c = Chain(5).then(double)
    result = await c.safe_run()
    self.assertEqual(result, 10)


class SafeRunErrorTests(unittest.TestCase):
  def test_safe_run_propagates_exceptions(self):
    def fail(v):
      raise ValueError("test error")
    c = Chain(1).then(fail)
    with self.assertRaises(ValueError):
      c.safe_run()

  def test_safe_run_with_except_handler(self):
    c = Chain(1).then(lambda v: 1 / 0).except_(lambda v: "handled", reraise=False)
    self.assertEqual(c.safe_run(), "handled")

  def test_safe_run_with_finally(self):
    log = []
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append("done"))
    self.assertEqual(c.safe_run(), 2)
    self.assertEqual(log, ["done"])


if __name__ == '__main__':
  unittest.main()
