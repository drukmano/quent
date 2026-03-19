---
title: "Async Handling -- Transparent Sync/Async Bridging"
description: "How quent transparently bridges sync and async Python code. Covers automatic async detection, mixed pipelines, the three-tier execution pattern, and async transitions."
tags:
  - async
  - sync
  - bridging
  - asyncio
  - coroutine
search:
  boost: 7
---

# Async Handling

quent's defining feature is transparent sync/async bridging. A single pipeline definition works for both sync and async callables -- zero ceremony, zero code duplication. This page explains how the mechanism works, what your code needs to account for, and where the edge cases are.

---

## The Bridge Contract

The bridge contract is the central guarantee of quent:

> **Given a pipeline of N steps, replacing any step's callable with a functionally equivalent callable of the opposite sync/async kind produces the same observable result.**

"Functionally equivalent" means: given the same input, the sync callable returns value `V` and the async callable returns a coroutine that resolves to the same value `V`.

This guarantee holds for all pipeline operations -- steps, side-effects, iteration, gathering, context managers, conditionals, error handlers, and cleanup handlers.

---

## The Two-Tier Execution Model

Every pipeline execution follows the same two-tier process:

**Tier 1 -- Sync fast path:** Execution always starts synchronously. The engine walks the pipeline's linked list, evaluating each step. After each step, the result is checked with a fast custom awaitable check (~10x faster than `inspect.isawaitable()`).

**Tier 2 -- Async continuation:** On the first awaitable result, the engine immediately hands off to an async continuation. The continuation receives the pending awaitable and all accumulated state (current value, root value, position in the linked list). It awaits the result, then continues walking the remaining steps in async mode.

This means:

1. A pipeline where every step is synchronous executes **entirely synchronously**. No event loop is created, no coroutines are allocated, no async machinery is touched.
2. A pipeline where any step returns an awaitable transparently transitions to async at that point.
3. The transition point can be anywhere -- first step, middle step, last step.
4. Once async, always async. There is no transition back to sync.

!!! important
    The async transition is **one-way**. Once a pipeline transitions to async mode, all remaining steps execute inside the async continuation. Even if subsequent steps return plain (non-awaitable) values, the pipeline stays in async mode because the overall execution is already inside a coroutine.

---

## How Detection Works

After each step, quent runs a fast custom awaitable check on the result. This check is an optimized replacement for `inspect.isawaitable()`, approximately **10x faster** (~30ns vs ~380ns).

It handles all three awaitable types:

1. **Native coroutines** (`CoroutineType`) -- the most common async case
2. **Generator-based coroutines** decorated with `@types.coroutine` -- have the `CO_ITERABLE_COROUTINE` flag
3. **Objects with `__await__`** -- `Future`, `Task`, custom awaitables

The check short-circuits on the first `isinstance` check for the common sync case, making the overhead negligible for sync-only pipelines.

---

## The Async Transition

### A Concrete Walkthrough

Consider a pipeline with three steps, where the middle step is async:

```python
import asyncio
from quent import Q

def double(x):
  return x * 2

async def fetch_factor(x):
  await asyncio.sleep(0.01)
  return x + 10

def to_string(x):
  return f"result: {x}"

pipeline = (
  Q()
  .then(double)
  .then(fetch_factor)
  .then(to_string)
)
```

When you call `pipeline.run(5)`:

| Step | What Happens | Mode |
|------|-------------|------|
| `double(5)` | Returns `10`. Not awaitable. | sync |
| `fetch_factor(10)` | Returns a coroutine. Awaitable detected. | sync -> **async transition** |
| `to_string(20)` | Async continuation awaits the coroutine (gets `20`), then calls `to_string(20)`. Returns `"result: 20"`. | async |

`pipeline.run(5)` returns a coroutine. The caller must await it:

```python
result = await pipeline.run(5)
# result = "result: 20"
```

---

## run() Return Type

The return type of `.run()` depends entirely on what happens during execution:

=== "All Sync"

    When every step returns a non-awaitable value, `.run()` returns the final value directly. No event loop is needed.

    ```python
    from quent import Q

    q = Q(5).then(lambda x: x + 1).then(lambda x: x * 2)
    result = q.run()
    # result = 12  (a plain int, not a coroutine)
    ```

=== "Async Transition"

    When any step returns an awaitable, `.run()` returns a coroutine. The caller must await it.

    ```python
    import asyncio
    from quent import Q

    async def async_double(x):
      return x * 2

    q = Q(5).then(async_double)
    result = await q.run()
    # result = 10
    ```

---

## Mixing Sync and Async Steps

The real power of quent shows when your pipeline mixes sync and async steps:

```python
import asyncio
from quent import Q

def validate(data):
  if not data:
    raise ValueError("empty data")
  return data

async def fetch_from_api(query):
  await asyncio.sleep(0.01)
  return {"query": query, "results": [1, 2, 3]}

def extract_results(response):
  return response["results"]

async def save_to_db(results):
  await asyncio.sleep(0.01)
  return len(results)

pipeline = (
  Q()
  .then(validate)
  .then(fetch_from_api)
  .then(extract_results)
  .then(save_to_db)
)
```

The execution proceeds as follows:

1. `validate` is sync -- runs in sync mode, returns data.
2. `fetch_from_api` is async -- triggers the transition. The async continuation takes over.
3. `extract_results` is sync -- runs inside the async continuation. Its non-awaitable return value is used directly.
4. `save_to_db` is async -- the async continuation awaits its coroutine.

```python
async def main():
  count = await pipeline.run("search term")
  # count = 3

asyncio.run(main())
```

The same pipeline definition would also work if you swapped in sync versions of `fetch_from_api` and `save_to_db` -- in that case, `pipeline.run("search term")` would return `3` directly without needing `await`.

---

## Three-Tier Iteration

Every iteration operation in quent (`.foreach()`, `.foreach_do()`) implements three execution tiers internally. This is what makes them work transparently with any combination of sync and async inputs and callbacks.

### Tier 1: Sync Fast Path

Everything is synchronous. The operation runs a plain loop, collects results, and returns. Zero async overhead.

```python
from quent import Q

result = Q([1, 2, 3]).foreach(lambda x: x * 2).run()
# result = [2, 4, 6]
# No coroutines created, no event loop involved
```

### Tier 2: Mid-Operation Async Transition

The operation starts synchronously, but discovers an awaitable partway through. It hands off the **live iterator state** to an async function that picks up exactly where the sync path left off. No items are re-processed.

```python
import asyncio
from quent import Q

async def async_transform(x):
  await asyncio.sleep(0.001)
  return x * 10

result = await Q([1, 2, 3]).foreach(async_transform).run()
# result = [10, 20, 30]
```

Internally, the sync loop calls `async_transform(1)` and gets back a coroutine. It hands the iterator, the partial results, and the pending awaitable to an async continuation. That function awaits the coroutine, then continues iterating in async mode.

### Tier 3: Full Async Path

The input is an async iterable (has `__aiter__` but not `__iter__`). The entire operation runs async from the start.

```python
import asyncio
from quent import Q

async def async_range(n):
  for i in range(n):
    await asyncio.sleep(0.001)
    yield i

result = await (
  Q(async_range, 5)
  .foreach(lambda x: x * 2)
  .run()
)
# result = [0, 2, 4, 6, 8]
```

### Tier Summary

| Tier | Input | Callback | Example |
|------|-------|----------|---------|
| 1 -- Sync fast path | sync iterable | sync fn | `Q([1,2,3]).foreach(str)` |
| 2 -- Mid-op transition | sync iterable | async fn | `Q([1,2,3]).foreach(async_fetch)` |
| 3 -- Full async | async iterable | sync or async fn | `Q(aiter).foreach(process)` |

This pattern applies to every operation:

- **`.foreach()` / `.foreach_do()`** -- three tiers over iterables
- **`.with_()` / `.with_do()`** -- three tiers for context managers
- **`.if_()` / `.else_()`** -- sync predicate, async predicate, and async branch evaluation

---

## Async Iterables and Async Context Managers

### Async Iterables

`.foreach()` and `.foreach_do()` support async iterables (`__aiter__`):

```python
async def stream_records():
  async for record in db.stream():
    yield record

result = await Q(stream_records).foreach(process).run()
```

When an object implements both `__iter__` and `__aiter__`, the async protocol is preferred if an async event loop is running (asyncio, trio, or curio).

### Async Context Managers

`.with_()` and `.with_do()` support async context managers (`__aenter__` / `__aexit__`):

```python
result = await (
  Q(aiohttp.ClientSession)
  .with_(lambda session: session.get('https://example.com'))
  .run()
)
```

Dual-protocol context managers (supporting both sync and async) use the async protocol when an async event loop is running (asyncio, trio, or curio).

---

## Async Finally in Sync Pipelines

There is one edge case where the sync and async worlds collide: when a **sync pipeline's** `finally_()` handler returns a coroutine.

### When It Happens

If the pipeline execution is synchronous (no async transition in the main pipeline), but the finally handler is an async function:

```python
import asyncio
from quent import Q

async def async_cleanup(root_value):
  await notify_service(root_value)

q = (
  Q()
  .then(process)      # sync
  .then(validate)     # sync
  .finally_(async_cleanup)
)

result = await q.run(data)
# async_cleanup returns a coroutine -- quent performs an async transition.
# run() returns a coroutine instead of a plain value.
```

### What quent Does

When a sync pipeline's finally handler returns a coroutine, the engine performs an **async transition**: `run()` returns a coroutine instead of a plain value. When the caller awaits this coroutine, the finally handler's coroutine is awaited first, and then the pipeline's result is returned (success path) or the active exception is re-raised (failure path). The pipeline result flows through the async wrapper -- nothing is discarded.

### except_() Handler Edge Cases

The same async transition model applies to `except_()` handlers:

- **`except_()` with `reraise=True`:** When the handler returns a coroutine in a sync pipeline, `run()` returns a coroutine. The caller awaits it, the handler completes, and the original exception is re-raised. This ensures reliable completion of async side-effects.

- **`except_()` with `reraise=False`:** When the handler returns a coroutine, this is a normal async transition -- the coroutine becomes the pipeline's result. The caller awaits it to get the handler's resolved value.

---

## Performance: Zero Async Overhead for Sync Pipelines

quent is designed so that fully synchronous pipelines have **zero async overhead**:

- **No event loop interaction** -- no `asyncio` import at evaluation time for sync pipelines.
- **No coroutine creation** -- no `async def` functions are called on the sync path.
- **Fast awaitable check** -- ~30ns per step, O(1). Short-circuits on the first `isinstance` check for the common sync case.
- **Sync path is a plain `while` loop** -- calling functions and checking results. No async machinery.
- **One-way transition** -- the async transition happens at most once per pipeline execution. Once the async continuation takes over, it processes all remaining steps without checking whether to go back to sync.

For detailed benchmark numbers -- per-step overhead, async transition cost, I/O-bound comparisons, and reproducible scripts -- see the **[Performance & Benchmarks](performance.md)** guide.

---

## Practical Patterns

### Library Code That Supports Both Callers

```python
from quent import Q

def process_order(order_service):
  """Works with both sync and async order services."""
  return (
    Q(order_service.fetch_order)
    .then(order_service.validate)
    .then(order_service.apply_discounts)
    .then(order_service.save)
    .run()
  )

# Sync caller
total = process_order(sync_order_service)

# Async caller
total = await process_order(async_order_service)
```

### Incremental Async Migration

Replace functions one at a time. The pipeline code stays identical:

```python
from quent import Q

pipeline = (
  Q()
  .then(fetch_user)    # was sync, now async -- no pipeline changes
  .then(validate_user) # still sync
  .then(enrich_user)   # still sync
  .then(save_user)     # was sync, now async -- no pipeline changes
)

# Before migration: returns value directly
# After migration: returns coroutine (caller adds await)
result = pipeline.run(user_id)
```

### Concurrent Execution with gather

`.gather()` runs multiple functions concurrently. The first function is probed to detect sync vs async. If it returns an awaitable, all functions run as async tasks concurrently. All functions must be consistently sync or async:

```python
import asyncio
from quent import Q

async def check_inventory(product):
  await asyncio.sleep(0.1)
  return {"in_stock": True}

async def get_pricing(product):
  await asyncio.sleep(0.1)
  return {"price": 29.99}

def get_metadata(product):
  return {"category": "electronics"}

pipeline = (
  Q()
  .gather(check_inventory, get_pricing, get_metadata)
)

async def main():
  results = await pipeline.run({"id": "PROD-1"})
  # results = (
  #   {"in_stock": True},
  #   {"price": 29.99},
  #   {"category": "electronics"},
  # )

asyncio.run(main())
```

Results are returned in the same order as the functions were passed, regardless of completion order.

### Async Context Managers

`.with_()` works transparently with async context managers:

```python
import aiohttp
from quent import Q

async def create_session():
  return aiohttp.ClientSession()

async def fetch_data(session):
  async with session.get("https://api.example.com/data") as resp:
    return await resp.json()

pipeline = (
  Q(create_session)
  .with_(fetch_data)
)

async def main():
  data = await pipeline.run()
  print(data)

asyncio.run(main())
```

---

## Further Reading

- **[Getting Started](../getting-started.md)** -- installation, first pipeline, calling conventions
- **[Pipelines Guide](pipelines.md)** -- pipeline building, context managers, conditionals, control flow
- **[Error Handling](error-handling.md)** -- `except_()`, `finally_()`, and how they interact with async
- **[Reuse and Patterns](reuse.md)** -- cloning, decorators, and composition
