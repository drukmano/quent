---
title: "Getting Started with Quent"
description: "Learn how to install quent and build your first Python pipeline chain. Covers value flow, calling conventions, collection operations, and running chains sync or async."
tags:
  - tutorial
  - getting started
  - installation
  - pipeline
search:
  boost: 8
---

# Getting Started

## Installation

Install quent from PyPI:

```bash
pip install quent
```

quent requires **Python 3.10** or later and has **zero runtime dependencies** -- it is pure Python.

## Your First Chain

A `Chain` is a sequential pipeline. You add steps with `.then()`, and execute with `.run()`:

```python
from quent import Chain

result = (
  Chain(5)
  .then(lambda x: x * 2)   # 5 * 2 = 10
  .then(lambda x: x + 3)   # 10 + 3 = 13
  .run()
)
# result = 13
```

Here is what happens:

1. `Chain(5)` creates a chain with root value `5`.
2. The first `.then()` receives `5`, returns `10`. This **replaces** the current value.
3. The second `.then()` receives `10`, returns `13`.
4. `.run()` executes the chain and returns the final value.

Every `.then()` call appends a step to the pipeline. The result of each step becomes the input to the next.

---

## Key Concepts

### The Pipeline Model

A chain is a sequential pipeline modeled as a singly-linked list. Steps are appended in O(1) time. Execution walks head-to-tail, threading a **current value** through each step.

- **Build time:** You construct the pipeline by calling `.then()`, `.do()`, `.foreach()`, etc. Each call appends a step.
- **Run time:** `.run()` walks the pipeline, evaluating each step in order.

Building is mutable (appending changes the chain). Execution is immutable (the pipeline structure is never modified during execution). A fully constructed chain can be executed concurrently from multiple threads.

### Value Threading

The pipeline threads a single value from step to step:

```
Chain(root)          root is evaluated -> current_value = result
  .then(f)           f(current_value) -> current_value = result
  .do(g)             g(current_value) -> result discarded, current_value unchanged
  .then(h)           h(current_value) -> current_value = result
  .run()             returns current_value
```

When the pipeline completes with no value ever having been produced, the result is `None`.

### Sync/Async Transparency

This is quent's defining feature. A single chain definition works for both sync and async callables:

```python
from quent import Chain

pipeline = (
  Chain()
  .then(fetch)
  .then(validate)
  .then(transform)
  .then(save)
)
```

Whether `fetch`, `validate`, `transform`, or `save` are sync or async -- or any mix of both -- the same pipeline works:

```python
# All sync -- returns a plain value
result = pipeline.run(data)

# Some or all async -- returns a coroutine
result = await pipeline.run(data)
```

Execution always starts synchronously. On the first awaitable result, the engine transitions to async and stays async. A fully sync pipeline has zero async overhead.

!!! tip
    For a deep dive into how the sync/async bridge works, see [Async Handling](guide/async.md).

---

## Value Flow: then vs do

The two most common methods are `.then()` and `.do()`. They differ in one critical way:

- **`.then(fn)`** -- fn's result **replaces** the current value.
- **`.do(fn)`** -- fn runs as a **side-effect**. Its result is **discarded**, and the current value passes through unchanged.

```python
from quent import Chain

result = (
  Chain(10)
  .then(lambda x: x * 2)  # receives 10, returns 20 -> current value is now 20
  .do(print)               # receives 20, prints it, result discarded -> current value stays 20
  .then(lambda x: x + 1)  # receives 20, returns 21 -> current value is now 21
  .run()
)
# prints: 20
# result = 21
```

Use `.then()` when you want to transform the value. Use `.do()` when you want to observe or log it without changing it.

!!! note
    `.do()` enforces that its argument is callable at build time. Passing a non-callable (like a string or integer) raises `TypeError`. This prevents bugs where a literal value is accidentally used as a side-effect -- it would silently do nothing.

---

## Working with Callables

quent works with **any Python callable**: functions, lambdas, methods, classes, objects with `__call__`, coroutine functions -- anything Python considers callable.

```python
from quent import Chain

# Regular functions
def double(x):
  return x * 2

# Lambdas
Chain(5).then(lambda x: x + 1).run()  # 6

# Built-in functions
Chain(5).then(str).run()  # '5'

# Classes (calling a class creates an instance)
Chain(42).then(str).run()  # '42'

# Bound methods
Chain('  hello  ').then(str.strip).run()  # 'hello'
```

### Non-callable Values

`.then()` also accepts non-callable values. A non-callable value simply replaces the current pipeline value:

```python
Chain(5).then(lambda x: x * 2).then(42).run()
# 42 -- the literal replaces the current value
```

This is useful for injecting constant values into a pipeline.

---

## Calling Conventions

!!! important
    Calling conventions are the most important concept in quent. They determine how each step receives its arguments. Understanding these rules makes everything else straightforward.

When a step's callable is invoked, quent decides what arguments to pass based on two rules, checked in priority order. The first matching rule wins.

### Rule 1: Explicit Args/Kwargs

When positional arguments or keyword arguments are provided at registration time, the callable receives **only those arguments**. The current pipeline value is **not** passed:

```python
from quent import Chain

def greet(name, greeting="Hello"):
  return f"{greeting}, {name}!"

result = Chain(42).then(greet, "World").run()
# calls greet("World") -> "Hello, World!"
# the current value (42) is NOT passed
```

### Rule 2: Default

When no explicit arguments are provided:

- **Callable, current value exists:** `fn(current_value)`
- **Callable, no current value:** `fn()` (called with no arguments)
- **Not callable:** The value is returned as-is

```python
from quent import Chain

Chain(5).then(str).run()          # str(5) -> '5'
Chain().then(dict).run()          # dict() -> {}
Chain(5).then(42).run()           # 42 (non-callable, returned as-is)
```

### Nested Chains

When the step's value is itself a `Chain`, the nested chain is executed with the current value as its input:

```python
from quent import Chain

inner = Chain().then(lambda x: x * 2).then(lambda x: x + 1)

result = Chain(5).then(inner).run()
# inner receives 5, runs its steps: 5 * 2 = 10, 10 + 1 = 11
# result = 11
```

Control flow signals (`return_()`, `break_()`) propagate through nested chains to the outer chain.

### Summary Table

| Priority | Rule | Trigger | Invocation |
|----------|------|---------|------------|
| 1 | Explicit Args | Args/kwargs provided | `fn(*args, **kwargs)` |
| 2 | Default | None of the above | `fn(cv)`, `fn()`, or `v` as-is |

!!! tip
    When in doubt: explicit arguments **replace** the current value, they do not **extend** it. If you need both the current value and extra arguments, use a lambda: `.then(lambda x: fn(x, extra_arg))`.

---

## Collection Operations

quent provides two methods for working with iterables:

```python
from quent import Chain

# .foreach() -- transform each element, collect results
result = Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
# result = [2, 4, 6]

# .foreach_do() -- side-effect on each element, keep originals
result = Chain([1, 2, 3]).foreach_do(print).run()
# prints: 1, 2, 3 (each on its own line)
# result = [1, 2, 3]
```

These follow the same pattern as `then` vs `do`:

- `.foreach(fn)` collects fn's return values (like `.then()` -- result matters).
- `.foreach_do(fn)` discards fn's return values (like `.do()` -- side-effect only).

To filter elements, use `.then()` with a list comprehension:

```python
from quent import Chain

result = (
  Chain([1, 2, 3, 4, 5, 6])
  .then(lambda xs: [x for x in xs if x % 2 == 0])  # [2, 4, 6]
  .foreach(lambda x: x ** 2)                         # [4, 16, 36]
  .then(sum)                                          # 56
  .run()
)
# result = 56
```

Both work transparently with both sync and async callables.

!!! note
    `.foreach()` and `.foreach_do()` both require their argument to be callable, just like `.do()`. Passing a non-callable raises `TypeError`.

---

## Running Chains

### .run() and \_\_call\_\_

`.run()` executes the chain and returns the result. Calling the chain directly does the same thing:

```python
from quent import Chain

chain = Chain(5).then(lambda x: x * 2)

# These are equivalent
result = chain.run()
result = chain()
```

### Injecting a Run-Time Value

`.run(value)` injects a value that **replaces** the build-time root value for that execution:

```python
from quent import Chain

double = Chain().then(lambda x: x * 2)

result = double.run(5)   # 10
result = double.run(100) # 200
```

This makes chains reusable -- define the pipeline once, run it with different inputs.

When both a root value and a run value exist, the run value wins:

```python
from quent import Chain

# Run value (C) replaces root value (A)
Chain('A').then(str.upper).run('hello')  # 'HELLO', not 'A'
```

### Return Type

The return type of `.run()` depends on what happens during execution:

- If all steps are synchronous, `.run()` returns the final value directly.
- If any step returns an awaitable, `.run()` returns a coroutine that must be awaited.

```python
from quent import Chain

chain = Chain().then(process).then(save)

# If process and save are sync functions:
result = chain.run(data)        # returns the value directly

# If either is async:
result = await chain.run(data)  # returns a coroutine, so await it
```

---

## None vs No Value

`None` is a valid pipeline value in quent. This is distinct from having **no value** at all:

```python
from quent import Chain

# Chain with root value None -- None flows through the pipeline
chain = Chain(None)

# Chain with no root value -- the pipeline starts empty
chain = Chain()
```

The difference matters for calling conventions:

```python
# fn(None) -- current value is None
Chain(None).then(fn).run()

# fn() -- no current value exists
Chain().then(fn).run()
```

quent uses an internal `Null` sentinel to distinguish these cases. It is not part of the public API; you do not need to interact with it directly.

---

## Quick Sync/Async Demo

Here is a complete example showing the same pipeline working with both sync and async callables:

```python
from quent import Chain

# Define the pipeline once
pipeline = (
  Chain()
  .then(validate)
  .then(transform)
  .do(log)
  .then(save)
)

# Sync usage -- all callables are sync
result = pipeline.run(data)

# Async usage -- some callables are async
result = await pipeline.run(data)
```

No `async def` variant. No `if asyncio.iscoroutinefunction(...)` checks. No code duplication.

### Before and After

Without quent, supporting both sync and async callers means writing the same logic twice:

```python
# Without quent: two functions, same logic
def process_sync(data):
  data = validate(data)
  data = transform(data)
  save(data)
  return data

async def process_async(data):
  data = await validate(data)
  data = await transform(data)
  await save(data)
  return data
```

With quent, there is only one:

```python
# With quent: one definition, both worlds
def process(data):
  return (
    Chain(data)
    .then(validate)
    .then(transform)
    .do(save)
    .run()
  )
```

If any step happens to be async, the chain handles the transition automatically. The caller `await`s if needed. No duplication.

---

## Reusing Chains

Chains are mutable -- calling `.then()` modifies the original. Use `.clone()` to create independent copies:

```python
from quent import Chain

base = Chain().then(validate).then(normalize)
for_api = base.clone().then(to_json)
for_db = base.clone().then(to_sql)
```

For more on reuse patterns, see [Reuse and Patterns](guide/reuse.md).

---

## Next Steps

Now that you understand the basics, explore the rest of the documentation:

- **[Why Quent](why-quent.md)** -- understand the problem quent solves and when to use it
- **[Chains](guide/chains.md)** -- comprehensive guide to pipeline building, context managers, conditionals, and control flow
- **[Async Handling](guide/async.md)** -- deep dive into sync/async bridging, the two-tier execution model, and async transitions
- **[Error Handling](guide/error-handling.md)** -- exception handlers, cleanup, and enhanced tracebacks
- **[Reuse and Patterns](guide/reuse.md)** -- cloning, nesting, decorators, and composition patterns
