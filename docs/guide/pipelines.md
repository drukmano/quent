---
title: "Pipelines Guide -- Building Pipelines with Quent"
description: "Complete guide to building pipelines with quent's Q class. Covers then, do, map, foreach_do, gather, context managers, conditionals, iteration, and control flow."
tags:
  - guide
  - pipeline
  - tutorial
search:
  boost: 5
---

# Building Pipelines with Q

The `Q` class is the core of quent. It is a sequential pipeline of operations that transparently bridges synchronous and asynchronous execution. You define your pipeline once -- using `.then()`, `.do()`, and the other methods documented here -- and it works with sync callables, async callables, or any mix of both.

This page covers every pipeline-building method. For async-specific behavior, see [Async Handling](async.md). For error handling and recovery, see [Error Handling](error-handling.md). For patterns like cloning, nesting, and decorators, see [Reuse and Patterns](reuse.md).

---

## The Pipeline Model

A pipeline is a singly-linked list of steps. Building appends nodes to the tail in O(1) time. Execution walks head-to-tail, threading a **current value** through each step.

- **Build-time mutation:** Every call to `.then()`, `.do()`, `.foreach()`, etc. appends a new node. Building is not thread-safe -- pipelines must be fully constructed before being shared across threads.
- **Run-time immutability:** Execution walks the list from head to tail, never mutating the list structure. A fully constructed pipeline is safe to execute concurrently from multiple threads.

This separation is the foundation of the thread-safety model.

---

## Constructor: `Q(v=<no value>, *args, **kwargs)`

A pipeline is created by calling the `Q` constructor. The root value seeds the pipeline.

### Forms

```python
from quent import Q

# No root value -- the pipeline starts empty.
q = Q()

# Non-callable root -- used as-is.
q = Q(42)
q = Q([1, 2, 3])
q = Q(None)  # None is a valid root value

# Callable root -- called when the pipeline runs.
q = Q(fetch_data)

# Callable root with arguments -- called with these args when the pipeline runs.
q = Q(fetch_data, user_id, max_results=30)
```

### Root Value Semantics

The root value is evaluated when the pipeline runs, not when it is constructed:

- **Callable root:** Called (with optional args/kwargs) when `run()` is invoked. The return value becomes the first current value.
- **Non-callable root:** Used as-is. Providing args/kwargs to a non-callable raises `TypeError` at build time.
- **No root (`Q()`):** The pipeline starts with no value. The first step determines the initial current value.

### Root Value vs Run Value

There are two ways to provide the initial value:

- **At build time:** `Q(v)` sets a root value.
- **At run time:** `q.run(v)` injects a run value.

When both exist, the **run value wins** and the build-time root is ignored entirely:

```python
from quent import Q

q = Q('build_time').then(str.upper)

q.run()           # 'BUILD_TIME' -- root value used
q.run('run_time') # 'RUN_TIME'   -- run value replaces root
```

The root value (once evaluated) is also captured as the **root value for error handlers**: `except_()` and `finally_()` handlers receive the root value, not the current pipeline value at the point of failure. This is by design -- the root value represents "what this pipeline was invoked with."

---

## Running a Pipeline

### `run(v=Null, *args, **kwargs)`

Execute the pipeline and return the final value:

```python
result = Q(42).then(lambda x: x * 2).run()
# result = 84
```

Pass a value to `.run()` to inject it as the initial input:

```python
q = Q().then(lambda x: x * 2)
result = q.run(10)   # 20
result = q.run(100)  # 200
```

If `v` is callable, it is called with `(*args, **kwargs)` and the result becomes the initial value. If `v` is not callable and args/kwargs are provided, `TypeError` is raised.

**Return type:** Either a plain value (all sync) or a coroutine (async transition occurred).

### `__call__(v=Null, *args, **kwargs)`

Alias for `run()`. Enables pipelines as first-class callables:

```python
q = Q().then(lambda x: x * 2)
result = q(10)  # same as q.run(10)
```

### `debug(v=Null, *args, **kwargs)`

Execute the pipeline with step-level instrumentation and return a `DebugResult` capturing the execution trace. The original pipeline is **not modified** -- `debug()` clones the pipeline internally and runs the clone.

```python
from quent import Q

result = Q(5).then(lambda x: x * 2).then(str).debug()
# result.value    -> '10'
# result.steps    -> list of StepRecord objects
# result.elapsed_ns -> total nanoseconds

result.print_trace()  # prints a formatted table to stderr
```

**`DebugResult` fields:**

| Field / Property | Type | Description |
|---|---|---|
| `value` | `T` | The pipeline's final result (same as `run()` would return) |
| `steps` | `list[StepRecord]` | Ordered list of step records |
| `elapsed_ns` | `int` | Total wall-clock nanoseconds |
| `succeeded` | `bool` | `True` if all steps completed without error |
| `failed` | `bool` | `True` if any step raised an exception |
| `print_trace(file=None)` | method | Prints a formatted trace table to `file` (default: `sys.stderr`) |

Each `StepRecord` in `.steps` is a frozen dataclass with: `step_name`, `input_value`, `result`, `elapsed_ns`, `exception`, and an `ok` property.

**Return type:**

- Fully synchronous pipeline: returns a `DebugResult` directly.
- Pipeline that transitions to async: returns a coroutine that resolves to a `DebugResult`.

```python
# Async pipeline
debug_result = await Q(fetch_data).then(process).debug()
debug_result.print_trace()
```

**Exception behavior:** If the pipeline raises during `debug()`, the exception propagates normally -- `debug()` does not suppress errors. Steps that executed before the failure are captured in the `DebugResult`'s `.steps` list (accessible if you catch the exception and inspect the debug pipeline's state).

!!! note
    `DebugResult` and `StepRecord` are not exported from `quent.__all__`. They are accessible as return types from `debug()` or from `quent._debug`.

---

## then and do

These are the two fundamental pipeline operations.

### `.then(v, *args, **kwargs)` -- Transform the Value

Append a step whose result **replaces** the current pipeline value:

```python
result = (
  Q(5)
  .then(lambda x: x * 2)   # current value: 10
  .then(lambda x: x + 1)   # current value: 11
  .then(str)                # current value: '11'
  .run()
)
# result = '11'
```

When `v` is not callable, it replaces the current value directly:

```python
result = Q(5).then(lambda x: x * 2).then('override').run()
# result = 'override'
```

!!! note
    Providing args/kwargs to a non-callable value raises `TypeError` at build time.

### `.do(fn, *args, **kwargs)` -- Side Effects

Append a side-effect step. `fn` is called, but its return value is **discarded** -- the current pipeline value passes through unchanged:

```python
result = (
  Q(42)
  .then(lambda x: x * 2)  # current value: 84
  .do(print)               # prints 84, current value still 84
  .then(str)               # current value: '84'
  .run()
)
# stdout: 84
# result = '84'
```

If `fn` returns an awaitable, it is still awaited (to complete the side-effect), but its resolved value is discarded.

!!! warning
    `.do()` **requires** a callable. Passing a non-callable raises `TypeError` at build time. This prevents bugs where a literal value is accidentally used as a side-effect -- it would silently do nothing.

---

## Calling Conventions

The calling conventions define **exactly how a step's callable is invoked**. They are the most important contract in quent -- every pipeline step, every operation, and every handler invocation goes through these rules.

There are **2 rules** for standard pipeline steps, applied in priority order. The first matching rule wins.

### Rule 1: Explicit Args/Kwargs

**Trigger:** Positional arguments or keyword arguments were provided at registration time.

**Behavior:** The callable is invoked with **only the explicit arguments**. The current pipeline value is **not** passed.

```python
from quent import Q

def format_number(currency, decimals=2):
  ...

Q(5).then(format_number, 'USD', decimals=2).run()
# calls: format_number('USD', decimals=2) -- the 5 is NOT passed
```

**Constraints:**

- The step must be callable. Providing arguments to a non-callable raises `TypeError` at build time.

!!! note "Design note"
    There is no built-in way to pass both the current value AND explicit arguments in a single `.then()` call. This is intentional. Use a lambda: `.then(lambda x: fn(x, extra_arg))`.

### Rule 2: Default Passthrough

**Trigger:** No explicit arguments were provided.

**Behavior depends on callability:**

- **Callable, current value exists:** `fn(current_value)`
- **Callable, no current value:** `fn()` (called with no arguments)
- **Not callable:** The value itself is returned as-is

```python
from quent import Q

Q(5).then(str).run()          # str(5) -> '5'
Q().then(dict).run()          # dict() -> {}
Q(5).then(42).run()           # 42 (non-callable, replaces value)
```

### Nested Pipelines

When the step's value is itself a `Q` instance, the nested pipeline is executed with the current value passed as its input. Control flow signals (`return_()`, `break_()`) propagate from the nested pipeline to the outer pipeline.

```python
from quent import Q

inner = Q().then(lambda x: x * 2).then(lambda x: x + 1)

# inner receives current value (5), runs its steps
result = Q(5).then(inner).run()
# 5 -> 10 -> 11
# result = 11
```

**Edge cases:**

- Nested pipelines have a depth limit (default: 50) to prevent unbounded recursion. Exceeding the limit raises `QuentException`.
- When a pipeline is used as a step in another pipeline, control flow signals propagate through to the outer pipeline.
- When a pipeline is executed directly via `.run()`, escaped control flow signals are caught and wrapped in `QuentException`.

### Summary Table

| Priority | Rule | Trigger | Invocation |
|----------|------|---------|------------|
| 1 | Explicit args | Args/kwargs provided | `fn(*args, **kwargs)` |
| 2 | Default | None of the above | `fn(cv)`, `fn()`, or `v` as-is |

---

## foreach and foreach_do

These methods operate on the **elements** of the current pipeline value (which must be iterable).

### `.foreach(fn, *, concurrency=None, executor=None)` -- Transform Each Element

Apply `fn` to each element and collect the results into a list:

```python
from quent import Q

result = Q([1, 2, 3]).foreach(lambda x: x ** 2).run()
# result = [1, 4, 9]
```

The list of results replaces the current pipeline value.

### `.foreach_do(fn, *, concurrency=None, executor=None)` -- Side-Effect Per Element

Apply `fn` to each element as a side-effect. The **original elements** (not fn's return values) are collected:

```python
from quent import Q

result = Q([1, 2, 3]).foreach_do(print).run()
# stdout: 1, 2, 3 (one per line)
# result = [1, 2, 3]  (original elements)
```

### Chaining Collection Operations

These return `self`, so you can compose them:

```python
result = (
  Q([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  .then(lambda xs: [x for x in xs if x % 2 == 0])  # [2, 4, 6, 8, 10]
  .foreach(lambda x: x ** 2)                         # [4, 16, 36, 64, 100]
  .then(sum)                                          # 220
  .run()
)
```

### Async Support

All three methods support async transparently:

- If `fn` returns an awaitable for any element, the operation transitions to async and awaits it. Subsequent elements continue in async mode.
- Supports both sync iterables (`__iter__`) and async iterables (`__aiter__`). When both protocols are present, the async protocol is preferred if an async event loop is running (asyncio, trio, or curio).

```python
async def fetch_details(user_id):
  ...

result = await Q(user_ids).foreach(fetch_details).run()
```

### Concurrent Execution

Pass a `concurrency` parameter to process elements in parallel:

```python
from quent import Q

# Process up to 5 elements concurrently
result = await Q(urls).foreach(fetch, concurrency=5).run()
```

| Mode | Mechanism |
|------|-----------|
| Sync | `ThreadPoolExecutor(max_workers=concurrency)` |
| Async | `asyncio.Semaphore(concurrency)` |

Without `concurrency`, elements are processed sequentially.

!!! note
    Both methods require `fn` to be callable. Passing a non-callable raises `TypeError`.

!!! warning
    When using `concurrency`, the entire input iterable is eagerly materialized into a list before processing begins. Do not use with infinite or very large iterables.

### Error Behavior

**Sequential:** Exceptions propagate immediately, stopping iteration at the failing element.

**Concurrent:** When a single worker fails, that exception propagates directly. When multiple workers fail, exceptions are wrapped in an `ExceptionGroup`. Control flow signals take priority: `return_()` > `break_()` > regular exceptions.

### break_() in Iteration

`Q.break_()` stops iteration early:

```python
from quent import Q

def process(item):
  if item == 'STOP':
    return Q.break_()  # stop, return results so far
  return item.upper()

result = Q(['a', 'b', 'STOP', 'c']).foreach(process).run()
# result = ['A', 'B']
```

With a value, `break_()` **appends** the value to the results collected so far:

```python
def process(item):
  if item < 0:
    return Q.break_('found_negative')
  return item * 2

result = Q([1, 2, -1, 3]).foreach(process).run()
# result = [2, 4, 'found_negative']
```

---

## gather -- Concurrent Fan-Out

`.gather(*fns, concurrency=-1, executor=None)` runs multiple functions on the current pipeline value concurrently. Results are returned as a **tuple** in the same positional order as `fns`.

```python
from quent import Q

results = Q(data).gather(validate, enrich, score).run()
# results = (validate_result, enrich_result, score_result)
```

### Always Concurrent

Gather is **always concurrent** -- there is no sequential fallback:

- **Sync path:** Uses a `ThreadPoolExecutor`. The first function is probed to detect sync vs async.
- **Async path:** Uses semaphore-limited async tasks (`TaskGroup` on Python 3.11+, `asyncio.gather` on 3.10).

Mixed sync/async is not supported within a single `gather()` -- all functions must be consistently sync or async.

!!! note
    At least one function must be provided. Passing zero functions raises `QuentException`. A single function still returns a one-element tuple: `(result,)`.

### Concurrency Limit

```python
# Limit to 2 concurrent executions
Q(data).gather(fn_a, fn_b, fn_c, fn_d, concurrency=2).run()
```

Without `concurrency`, all functions run concurrently with no limit.

### Error Behavior

- **Single failure:** The exception propagates directly (not wrapped).
- **Multiple failures:** Regular exceptions are wrapped in an `ExceptionGroup`.
- **Control flow:** `Q.return_()` takes absolute priority. `Q.break_()` is not allowed in `gather()` -- it raises `QuentException`.

### After gather: Accessing Results

The result of `gather()` is a tuple. Access individual results by index, or destructure in a lambda:

```python
from quent import Q

result = (
  Q(user_id)
  .gather(
    lambda uid: fetch_profile(uid),
    lambda uid: fetch_settings(uid),
    lambda uid: compute_score(uid),
  )
  .then(lambda results: {
    'profile': results[0],
    'settings': results[1],
    'score': results[2],
  })
  .run()
)
```

---

## with\_ and with\_do -- Context Managers

These methods enter the current pipeline value as a context manager and call your function with the context value.

### `.with_(fn, *args, **kwargs)`

Enter the current value as a context manager, call `fn` with the context value (the result of `__enter__`), and replace the pipeline value with fn's result:

```python
from quent import Q

content = (
  Q('data.txt')
  .then(open)
  .with_(lambda f: f.read())
  .run()
)
# Equivalent to:
# with open('data.txt') as f:
#   content = f.read()
```

The context manager is properly exited regardless of whether `fn` succeeds or fails.

### `.with_do(fn, *args, **kwargs)`

Same as `.with_()`, but fn's result is **discarded**. The original pipeline value (the context manager object itself, **before** entering) passes through:

```python
from quent import Q

result = (
  Q(db.connect)
  .with_do(lambda conn: conn.execute('INSERT INTO ...'))
  .run()
)
# result is the connection object, not the execute result
```

### Sync and Async Context Managers

Both methods work transparently with sync and async context managers:

- **Sync CM:** `__enter__()` / `__exit__()` used.
- **Async CM:** `__aenter__()` / `__aexit__()` used.
- **Dual-protocol:** When an async event loop is running (asyncio, trio, or curio), the async protocol is preferred. Otherwise, the sync protocol is used.

If `fn` returns an awaitable inside a sync context manager, the operation transitions to async seamlessly.

### Exception Suppression

If `fn` raises and `__exit__` returns a truthy value (suppressing the exception), the pipeline continues. For `.with_()`, the current value becomes `None`. For `.with_do()`, the original pipeline value passes through.

### Control Flow Signals

If `fn` raises a control flow signal (`return_()` or `break_()`), `__exit__` is called with no exception info (clean exit), and the signal propagates to the outer pipeline.

---

## if\_ and else\_ -- Conditional Logic

`.if_()` begins a conditional branch. It must be followed immediately by `.then()` or `.do()`, which registers the truthy branch. If the predicate is truthy, that branch is evaluated and its result replaces the current value. If falsy, the value passes through unchanged (or the `else_()` branch runs, if registered).

### `.if_(predicate=None).then(v, *args, **kwargs)`

```python
from quent import Q

result = (
  Q(value)
  .if_(lambda x: x > 0).then(process_positive)
  .run()
)
# If value > 0: result = process_positive(value)
# If value <= 0: result = value (passes through)
```

The `.then()` (or `.do()`) after `.if_()` becomes the truthy branch rather than a regular pipeline step. Standard calling conventions apply: pass args and kwargs directly to `.then()`:

```python
.if_(lambda x: x > 0).then(transform, arg1, arg2, key='value')
```

### `.else_(v, *args, **kwargs)`

Register an alternative branch for the immediately preceding `.if_().then()` (or `.if_().do()`):

```python
result = (
  Q(value)
  .if_(lambda x: x > 0).then(process_positive)
  .else_(process_negative)
  .run()
)
```

!!! warning
    `.else_()` must follow **immediately** after the `.then()` or `.do()` that follows `.if_()` -- no other operations in between. Otherwise, `QuentException` is raised. Only one `.else_()` per `.if_()`.

### `.else_do(fn, *args, **kwargs)` -- Side-Effect Else Branch

Register an else branch whose result is **discarded** (the current pipeline value passes through unchanged):

```python
result = (
  Q(-5)
  .if_(lambda x: x > 0).then(str)
  .else_do(print)
  .run()
)
# prints: -5 (side-effect from else_do)
# result = -5 (current value passes through)
```

`.else_do()` follows the same positioning rules as `.else_()` -- it must immediately follow the `.then()` or `.do()` after `.if_()`.

### Without a Predicate

When `predicate` is omitted, the truthiness of the current pipeline value itself is used:

```python
result = (
  Q(user_or_none)
  .if_().then(process_user)
  .else_(lambda _: 'no user found')
  .run()
)
# If user_or_none is truthy: result = process_user(user_or_none)
# If user_or_none is falsy:  result = 'no user found'
```

When the pipeline has no current value (internal Null sentinel), the predicate evaluates to falsy.

### Literal Predicates

When `predicate` is a non-callable value, its truthiness is used directly:

```python
# Truthy literal -- branch always runs
.if_(True).then(transform)

# Falsy literal -- branch never runs (value passes through)
.if_(None).then(transform)
```

### The Truthy Branch

The value passed to `.then()` can be a callable, a non-callable value (used as-is), or a nested Q pipeline:

```python
# Callable -- receives current value
.if_(predicate).then(transform)

# Non-callable -- used as-is
.if_(predicate).then('default_value')

# Nested pipeline
.if_(predicate).then(Q().then(validate).then(process))

# Callable with explicit args (Rule 1 -- current value not passed)
.if_(predicate).then(transform, arg1, arg2, key='value')
```

### Predicate Semantics

Predicates use the standard 2-rule calling convention. If args/kwargs are provided to `.if_()`, they are forwarded to the predicate callable.

!!! note
    When a predicate is a nested `Q` instance, `return_()` inside the predicate pipeline propagates to the outer pipeline (early exit is valid from a predicate). `break_()` inside a predicate pipeline raises `QuentException` -- predicates are not iteration contexts.

### Async Predicates and Branches

Both the predicate and the then/else callables can be sync or async. If either returns an awaitable, the operation transitions to async.

---

## while\_ -- Loops

`.while_()` begins a loop operation. It must be followed immediately by `.then()` or `.do()`, which becomes the loop body. The loop repeatedly evaluates the body while the predicate is truthy. When the predicate becomes falsy (or `break_()` is raised), the loop exits and the current value continues down the pipeline.

### `.while_(predicate=None, /, *args, **kwargs).then(v, *args, **kwargs)`

```python
from quent import Q

# Decrement until zero -- predicate tests truthiness of current value
result = Q(10).while_().then(lambda x: x - 1).run()
# Iteration: 10 → 9 → 8 → ... → 1 → 0 (0 is falsy, loop stops)
# result = 0

# Predicate callable -- loop while value exceeds threshold
result = Q(100).while_(lambda x: x > 1).then(lambda x: x // 2).run()
# Iteration: 100 → 50 → 25 → 12 → 6 → 3 → 1 (1 is not > 1, loop stops)
# result = 1
```

### Predicate Behavior

The predicate follows the same semantics as `.if_()`:

- **`None` (default):** The truthiness of the current loop value is used. `None` and `0` and empty containers are falsy.
- **Callable:** Invoked with the current loop value. Standard calling conventions apply -- pass args/kwargs to `.while_()` to forward them (Rule 1: current value not passed).
- **Literal value:** Its truthiness is used directly (`True` means loop forever, `None` means never loop).

### Body Modes: `.then()` vs `.do()`

The step immediately after `while_()` is absorbed as the loop body, not added as a regular pipeline step:

- **`.then(fn)`:** `fn`'s result feeds back as the loop value for the next iteration. When the loop exits, the final loop value replaces the current pipeline value.
- **`.do(fn)`:** `fn` runs for side effects; its return value is discarded. The loop value is **not changed** each iteration.

```python
# .then() -- value transforms each iteration
result = Q(1).while_(lambda x: x < 128).then(lambda x: x * 2).run()
# result = 128

# .do() -- value is unchanged, loop runs forever without break_()
results = []
Q(5).while_(lambda x: x > 0).do(results.append)
# Infinite loop! .do() never changes 5, so the predicate always sees 5.
# Always use break_() or use .then() to change the loop value.
```

!!! warning
    When using `.do()` with `while_()`, the loop value never changes. If the predicate tests the loop value (including the default `None` predicate), this creates an **infinite loop**. Use `break_()` to exit, or use `.then()` to transform the loop value.

### Exiting Early with `break_()`

`Q.break_()` exits the loop immediately:

```python
from quent import Q

# Break with a value -- the break value becomes the loop result
result = Q(1).while_(True).then(lambda x: Q.break_(x) if x >= 100 else x * 2).run()
# Iteration: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128 (128 >= 100, break with 128)
# result = 128

# Break without a value -- result is the current loop value at break time
result = Q(1).while_(True).then(lambda x: Q.break_() if x >= 100 else x * 2).run()
# result = 128 (same -- the current loop value when break_() was raised)
```

`break_()` can be raised from either the body or the predicate. This differs from `foreach` break semantics -- `while_` preserves the **current loop value** (or the break value), while `foreach` preserves **partial results** collected so far.

### Nesting Conditionals in the Loop Body

To use `.if_()` inside a loop body, wrap it in a nested pipeline:

```python
from quent import Q

result = (
  Q(data)
  .while_(has_more)
  .then(Q().if_(is_valid).then(process).else_(skip))
  .run()
)
```

!!! note
    `.if_()` cannot be used as the truthy branch of a `while_()` directly -- use a nested pipeline instead. Nesting also applies to nested while loops. Calling `while_()` while another `while_()` is pending raises `QuentException`.

### Async Support

Both the predicate and body can be sync or async. The loop follows the standard two-tier bridge: starts sync, transitions to async on the first awaitable, stays async for remaining iterations.

```python
# Async predicate and async body -- run() returns a coroutine
result = await Q(resource).while_(async_check_available).then(async_process).run()
```

### Error Behavior

Exceptions from the predicate or body propagate immediately through the pipeline's error handling. The while loop terminates on the first exception. The pipeline's `except_()` and `finally_()` handlers apply normally.

---

## drive\_gen -- Generator Driving

`.drive_gen(fn)` drives a sync or async generator bidirectionally using Python's generator send protocol. The step function `fn` processes each yielded value and its return is sent back into the generator. When the generator stops, the last `fn` result becomes the pipeline value.

### Basic Usage

```python
from quent import Q

def gen():
  x = yield 1        # yield 1 to driver
  x = yield x + 1    # yield (sent_value + 1) to driver
  x = yield x + 1    # yield (sent_value + 1) to driver

result = Q(gen()).drive_gen(lambda x: x * 2).run()
# Flow: yield 1 → fn(1)=2 → send 2 → yield 3 → fn(3)=6 → send 6 → yield 7 → fn(7)=14 → StopIteration
# result = 14
```

The current pipeline value must be a sync generator, async generator, or a callable that produces one. If it is callable (but not already a generator), it is called first to obtain the generator.

### The Send Protocol

Execution follows this cycle:

1. **Get first value:** `next(gen)` (sync) or `await gen.__anext__()` (async).
2. **Drive loop:** Call `result = fn(yielded_value)`. If `result` is awaitable, await it.
3. **Send back:** `gen.send(result)` (sync) or `await gen.asend(result)` (async).
4. **Repeat** until `StopIteration` / `StopAsyncIteration`.
5. **Cleanup (always):** `gen.close()` / `await gen.aclose()` on all exit paths.

The **return value** of the operation is the last value returned by `fn`. If the generator yields nothing at all, the pipeline value becomes `None` (fn was never called). The generator's own return value (`StopIteration.value`) is ignored.

### Non-Standard Calling Convention

`fn` is always called as `fn(yielded_value)` -- the standard 2-rule calling convention does **not** apply here. The yielded value is always passed directly; args/kwargs dispatch is not available.

```python
# fn always receives the yielded value directly
Q(gen()).drive_gen(process).run()     # calls: process(yielded_value)
Q(gen()).drive_gen(lambda x: x).run() # same: lambda receives yielded value
```

### Sync and Async Generators

```python
# Sync generator, sync step function -- fully sync
Q(gen()).drive_gen(step_fn).run()

# Async generator, sync step function -- fully async (must await)
result = await Q(async_gen()).drive_gen(process_request).run()

# Sync generator, async step function -- mid-transition
# The generator stays sync, but fn's awaitables are awaited
await Q(auth_flow(req)).drive_gen(async_send).run()

# Callable that produces a generator
Q(lambda: gen()).drive_gen(step_fn).run()
```

The mid-transition case (sync generator + async `fn`) is the primary motivating use case -- matching patterns like httpx's auth flow where the generator is sync but the handler may be async.

### Error Semantics

- **Exception from `fn`:** Propagates out of `drive_gen`. The generator is closed in cleanup. The exception is **not** injected into the generator (no `gen.throw()`).
- **Exception from `gen.send()`:** Propagates out of `drive_gen`. The generator is closed in cleanup.
- **Control flow signals** (`return_()`, `break_()`): Propagate unchanged to the enclosing pipeline. The generator is closed in cleanup.

Cleanup via `gen.close()` / `gen.aclose()` is guaranteed on all exit paths.

### Composability

`.drive_gen()` is a standard pipeline step -- chain `.then()`, `.do()`, `.except_()`, and `.finally_()` after it:

```python
# Post-processing after the generator finishes
Q(gen()).drive_gen(step_fn).then(validate).do(log).run()

# Error handling and cleanup
Q(gen()).drive_gen(step_fn).except_(handle_error).finally_(cleanup).run()
```

---

## iterate -- Lazy Iteration

`.iterate()` and `.iterate_do()` return a dual sync/async iterator over the pipeline's output.

### `.iterate(fn=None)`

Returns a `QuentIterator` object. The pipeline is executed when iteration begins (not when `iterate()` is called). If `fn` is provided, each element is transformed by `fn` before yielding:

```python
from quent import Q

# Without fn -- yields raw elements
for item in Q(fetch_all_users).iterate():
  process(item)

# With fn -- yields fn(element)
for name in Q(fetch_all_users).iterate(lambda u: u.name):
  print(name)
```

### `.iterate_do(fn=None)`

Same as `.iterate()`, but fn's return values are discarded. The original elements are yielded:

```python
for user in Q(fetch_all_users).iterate_do(log_user):
  # log_user(user) is called for side-effects
  # the original user object is yielded
  process(user)
```

### Sync and Async Iteration

The returned `QuentIterator` supports both `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops):

```python
# Sync iteration
for item in Q(get_items).iterate():
  handle(item)

# Async iteration
async for item in Q(get_items).iterate():
  await handle(item)
```

!!! warning
    If you use `for` (sync iteration) but the pipeline or the `fn` callable returns a coroutine, a `TypeError` is raised. Use `async for` in that case.

### Reusable Iterators

Calling the `QuentIterator` object with arguments creates a new iterator bound to those arguments:

```python
gen = Q(fetch_page).iterate()

# Iterate with different inputs
for item in gen(page=1):
  ...
for item in gen(page=2):
  ...
```

Each call returns a fresh iterator instance. The original iterator's configuration (fn, side-effect mode) is preserved; only the run arguments change.

### Control Flow in Iteration

- **`return_(v)`**: Yields the return value (if provided) and stops iteration. Previously yielded values are preserved.
- **`break_(v)`**: Stops iteration. If a value is provided, it is yielded before stopping. If no value, iteration stops immediately.

!!! note
    This differs from `foreach()`/`foreach_do()` where `break_(v)` **appends** the value to the collected results. In `iterate()`, the break value is yielded as one additional item before stopping.

### Error Handling Scope

The pipeline's `except_()` and `finally_()` handlers apply to the pipeline execution that produces the iterable. Exceptions from the iteration callback `fn` during iteration are NOT covered by the pipeline's handlers -- they propagate directly to the caller.

### Buffered Iteration

`.buffer(n)` attaches a backpressure-aware bounded buffer between the pipeline's iterable output and the iteration consumer. The producer runs ahead by up to `n` items while the consumer processes them, decoupling the two sides.

```python
from quent import Q

# Sync: producer runs in a background thread, consumer reads with backpressure
for item in Q(produce).buffer(10).iterate():
  process(item)

# With a transformation function
for item in Q(produce).buffer(5).iterate(transform):
  consume(item)

# Async: producer runs as a background asyncio.Task
async for item in Q(async_produce).buffer(10).iterate():
  await process(item)
```

`buffer()` is a **pipeline modifier, not a pipeline step** -- it does not add a link to the pipeline. It only takes effect when consumed via an iteration terminal (`iterate()`, `iterate_do()`). Using `buffer()` without an iteration terminal and calling `run()` instead raises `QuentException`.

**Backpressure:** When the buffer is full, the producer blocks. When the buffer is empty, the consumer blocks. This prevents unbounded memory growth with fast producers.

| Mode | Mechanism |
|------|-----------|
| Sync (`for`) | Background daemon thread + `queue.Queue(maxsize=n)` |
| Async (`async for`) | Background `asyncio.Task` + `asyncio.Queue(maxsize=n)` |

**FIFO ordering** is guaranteed -- items are delivered in the same order they were produced.

**Error behavior:** If the producer raises an exception, it is propagated to the consumer at the next `get()`. If the consumer exits early (`break`, `GeneratorExit`), the producer is signaled to stop and cleanup is guaranteed.

!!! note
    `n` must be a positive integer. `bool` values, `0`, and negative values raise `ValueError` or `TypeError`. The buffer size is preserved across `clone()` and when calling a `QuentIterator` with new arguments.

---

## flat\_iterate and flat\_iterate\_do -- Flatmap Iteration

`.flat_iterate()` and `.flat_iterate_do()` are flatmap iteration terminals. Each source element is expanded into sub-items before yielding.

### `.flat_iterate(fn=None, *, flush=None)`

Each element of the pipeline's iterable result is either iterated directly (when `fn` is `None`, flattening one level of nesting) or transformed by `fn` into a sub-iterable whose items are individually yielded.

```python
from quent import Q

# Flatten one level of nesting (fn=None)
for item in Q([[1, 2], [3, 4]]).flat_iterate():
  print(item)  # 1, 2, 3, 4

# Transform each element into a sub-iterable
for word in Q(['hello world', 'foo bar']).flat_iterate(str.split):
  print(word)  # hello, world, foo, bar
```

The optional `flush` callable is invoked once after the source iterable is fully consumed. It must return an iterable; each item is yielded into the stream. This is useful for emitting buffered or remaining items after the source ends.

### `.flat_iterate_do(fn=None, *, flush=None)`

Like `flat_iterate()`, but `fn` runs as a side-effect — its returned iterable is consumed (executing side-effects) but the items are not yielded. The original source elements are yielded instead.

```python
for item in Q([[1, 2], [3]]).flat_iterate_do(lambda sub: [print(x) for x in sub]):
  print('source:', item)
```

Both methods return a `QuentIterator` supporting `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops). All iteration behavior — sync/async, control flow, deferred `finally_()`, and `buffer()` — matches `iterate()`.

---

## Deferred with\_ in Iteration

When `.with_(fn)` or `.with_do(fn)` is the **last pipeline step** before an iteration terminal (`.iterate()`, `.iterate_do()`, `.flat_iterate()`, `.flat_iterate_do()`), context manager entry is **deferred** to iteration time.

Without deferral, `.with_()` would enter the context manager during the pipeline's `run()` phase and exit it before iteration begins — the resource would be closed before any items are consumed. Deferral keeps the context manager open throughout iteration:

```python
from quent import Q

# File stays open for entire iteration, then closes
for line in Q(open, 'data.txt').with_(lambda f: f).iterate(str.strip):
  process(line)

# Async context manager with deferred cleanup
async for row in Q(db.connect).with_(lambda conn: conn.cursor()).iterate():
  process(row)
```

The context manager is exited in the generator's `finally:` block, guaranteeing cleanup on all exit paths (normal exhaustion, `break`, exceptions, generator `.close()`).

**When both `buffer()` and a deferred `with_` are active,** the buffer wraps the iterable after the context manager is entered and the inner function produces the iterable.

---

## Named Pipelines

`.name(label)` assigns a user-provided label for traceback identification. The label appears in pipeline visualizations (`Q[label](root)`), exception notes, and `repr(q)`. It has no effect on execution semantics:

```python
from quent import Q

Q(fetch).name('auth_pipeline').then(validate).run()
```

---

## Context API

The Context API provides named storage scoped to the execution context. It is useful when non-adjacent steps need to share data without altering the pipeline's value flow.

Storage is backed by a `ContextVar`-based dictionary. Each `set()` creates a new dict (copy-on-write), ensuring concurrent workers are properly isolated.

### Instance methods (pipeline steps)

```python
q.set(key)           # store current pipeline value under key
q.set(key, value)    # store explicit value under key
q.get(key)           # retrieve value -- replaces current pipeline value
q.get(key, default)  # retrieve with fallback
```

```python
from quent import Q

result = (
  Q(fetch_user)
  .set('user')                        # store current value (the user) in context
  .set('source', 'api')              # store explicit value 'api' under 'source'
  .then(validate_permissions)         # transform continues with original user
  .get('user')                        # retrieve original user
  .then(format_response)
  .run(user_id)
)
```

### Class methods (immediate)

```python
Q.set(key, value)         # store value immediately (not a pipeline step)
Q.get(key)                # retrieve value immediately
Q.get(key, default)       # retrieve with fallback
```

!!! note "Dual dispatch"
    `q.set(...)` / `q.get(...)` (instance access) append a pipeline step. `Q.set(...)` / `Q.get(...)` (class access) operate on context immediately via descriptor dispatch.

---

## from\_steps -- Dynamic Pipeline Construction

`Q.from_steps(*steps)` constructs a pipeline from a sequence of steps, each appended via `.then()`:

```python
from quent import Q

# Variadic form
pipeline = Q.from_steps(validate, normalize, str.upper)
pipeline.run('  hello  ')

# List form -- useful for dynamic construction
steps = [validate, normalize, str.upper]
pipeline = Q.from_steps(steps)

# From a plugin registry
plugins = load_plugins()
pipeline = Q.from_steps([p.transform for p in plugins])
```

`Q.from_steps(a, b, c)` is equivalent to `Q().then(a).then(b).then(c)`. When a single `list` or `tuple` is passed, it is unpacked as the step sequence.

---

## Control Flow

quent provides two class methods for non-local control flow. These work by raising internal `BaseException` signals that the pipeline catches and handles. They bypass `except Exception` clauses.

### `Q.return_(v=<no value>, *args, **kwargs)` -- Early Exit

Exit the pipeline early, returning `v` as the pipeline's result:

```python
from quent import Q

def process(data):
  if data is None:
    return Q.return_('default')
  return data.upper()

result = Q(None).then(process).then(further_processing).run()
# result = 'default'  (further_processing is never called)
```

**Value semantics:**

- **No value:** `Q.return_()` -- the pipeline returns `None`.
- **Non-callable value:** `Q.return_(42)` -- the pipeline returns `42`.
- **Callable value:** `Q.return_(fn, *args, **kwargs)` -- the callable is invoked when the signal is caught. Its return value becomes the pipeline's result.

!!! tip "Idiomatic usage with `return`"
    `Q.return_()` works by raising an internal `BaseException` subclass — the signal propagates regardless of whether `return` is present. However, it is idiomatically written with `return` to satisfy type checkers and signal intent to readers:

    ```python
    # Idiomatic:
    return Q.return_(value)

    # Also valid -- the exception-based mechanism does not require the return
    # keyword for propagation, but linters may flag subsequent code as unreachable
    Q.return_(value)
    ```

### `Q.break_(v=<no value>, *args, **kwargs)` -- Break from a Loop or Iteration

Break out of a `.foreach()`, `.foreach_do()`, `.while_()`, `.iterate()`, `.iterate_do()`, `.flat_iterate()`, or `.flat_iterate_do()` context:

```python
from quent import Q

result = Q([1, 2, 3, 4, 5]).foreach(
  lambda x: Q.break_(x * 10) if x == 3 else x * 2
).run()
# result = [2, 4, 30]
# x=1 -> 2, x=2 -> 4, x=3 -> break with value 30 (appended to [2, 4])
```

Without a value, the partial results collected so far are returned:

```python
result = Q([1, 2, 3, 4, 5]).foreach(
  lambda x: Q.break_() if x == 3 else x * 2
).run()
# result = [2, 4]
```

!!! warning
    `Q.break_()` is only valid inside `.foreach()`, `.foreach_do()`, `.while_()`, or an iteration context (`.iterate()`, `.iterate_do()`, `.flat_iterate()`, `.flat_iterate_do()`). Using it elsewhere raises `QuentException`. Using it inside `gather()` also raises `QuentException`.

**`break_()` in `while_()` vs `foreach()`:** The semantics differ. In `while_()`, the break value (or current loop value if no value given) becomes the loop's result directly. In `foreach()`, the break value is **appended** to the partial results list collected so far.

### Nested Pipeline Propagation

Control flow signals propagate through nested pipelines:

```python
from quent import Q

validate = (
  Q()
  .if_(lambda x: not x.get('valid')).then(lambda _: Q.return_({'error': 'invalid'}))
  .then(check_permissions)
)

pipeline = (
  Q()
  .then(validate)    # return_() propagates through to outer pipeline
  .then(transform)   # skipped if validate returned early
  .then(save)        # skipped if validate returned early
)

result = pipeline.run({'valid': False})
# result = {'error': 'invalid'}
```

### Restrictions

- **In except/finally handlers:** `return_()` and `break_()` raise `QuentException`. Control flow signals are not allowed in error or cleanup handlers.
- **Signal escape:** If a signal escapes past `run()`, it is caught and wrapped in `QuentException`.

---

## Nesting Pipelines

A `Q` instance can be used as a step inside another pipeline. The inner pipeline executes as a single atomic step:

```python
from quent import Q

validate = Q().then(check_schema).then(check_permissions)
transform = Q().then(normalize).then(enrich)

pipeline = (
  Q(request)
  .then(validate)   # runs the entire validate pipeline as one step
  .then(transform)  # runs the entire transform pipeline as one step
  .then(save)
  .run()
)
```

When a pipeline is nested:

- The current pipeline value is passed as input to the inner pipeline.
- The inner pipeline's final result becomes the outer pipeline's current value.
- Control flow signals propagate through.
- The inner pipeline's `except_()` and `finally_()` handlers apply only to that pipeline's execution.

See [Reuse and Patterns](reuse.md) for more on composition with nested pipelines.

---

## Method Chaining Patterns

Every builder method returns `self`, enabling fluent pipelines:

```python
from quent import Q

# Vertical style (recommended for complex pipelines)
result = (
  Q(request)
  .then(authenticate)
  .then(authorize)
  .do(log_access)
  .then(process)
  .then(save)
  .except_(handle_error)
  .finally_(cleanup)
  .run()
)

# Inline style (for simple pipelines)
result = Q(5).then(lambda x: x * 2).then(str).run()
```

---

## Summary

| Method | Receives current value? | Result replaces value? | Requires callable? |
|---|---|---|---|
| `.then(v)` | Yes (if callable, no explicit args) | Yes | No |
| `.do(fn)` | Yes (if no explicit args) | No (discarded) | Yes |
| `.foreach(fn)` | Each element | Yes (list) | Yes |
| `.foreach_do(fn)` | Each element | Yes (list of originals) | Yes |
| `.gather(*fns)` | Each fn gets current value | Yes (tuple) | Yes (all fns) |
| `.with_(fn)` | Context value (`__enter__` result) | Yes | Yes |
| `.with_do(fn)` | Context value (`__enter__` result) | No (original CM passes through) | Yes |
| `.if_(pred).then(fn)` | Predicate and fn get current value | Yes (if branch runs) | Predicate: Yes or None |
| `.else_(v)` | Current value | Yes | No |
| `.while_(pred).then(fn)` | Predicate and fn get loop value | Yes (final loop value) | Predicate: Yes or None |
| `.while_(pred).do(fn)` | Predicate and fn get loop value | No (loop value unchanged) | Predicate: Yes or None |
| `.drive_gen(fn)` | fn receives each yielded value | Yes (last fn result) | Yes |
| `.flat_iterate(fn)` | Each sub-item from fn(element) | N/A — yields sub-items | No |
| `.flat_iterate_do(fn)` | Each source element | N/A — yields source elements | No |
| `.buffer(n)` | N/A (pipeline modifier) | N/A | N/A |
| `.name(label)` | N/A (metadata only) | N/A | N/A |
| `.debug(v)` | N/A (execution method) | N/A — returns `DebugResult` | N/A |
