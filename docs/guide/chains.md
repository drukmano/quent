---
title: "Chain Guide -- Building Pipelines with Quent"
description: "Complete guide to building pipelines with quent's Chain class. Covers then, do, map, foreach_do, gather, context managers, conditionals, iteration, and control flow."
tags:
  - guide
  - chain
  - pipeline
  - tutorial
search:
  boost: 5
---

# Building Pipelines with Chain

The `Chain` class is the core of quent. It is a sequential pipeline of operations that transparently bridges synchronous and asynchronous execution. You define your pipeline once -- using `.then()`, `.do()`, and the other methods documented here -- and it works with sync callables, async callables, or any mix of both.

This page covers every pipeline-building method. For async-specific behavior, see [Async Handling](async.md). For error handling and recovery, see [Error Handling](error-handling.md). For patterns like cloning, nesting, and decorators, see [Reuse and Patterns](reuse.md).

---

## The Pipeline Model

A chain is a singly-linked list of steps. Building appends nodes to the tail in O(1) time. Execution walks head-to-tail, threading a **current value** through each step.

- **Build-time mutation:** Every call to `.then()`, `.do()`, `.foreach()`, etc. appends a new node. Building is not thread-safe -- chains must be fully constructed before being shared across threads.
- **Run-time immutability:** Execution walks the list from head to tail, never mutating the list structure. A fully constructed chain is safe to execute concurrently from multiple threads.

This separation is the foundation of the thread-safety model.

---

## Constructor: `Chain(v=Null, *args, **kwargs)`

A chain is created by calling the `Chain` constructor. The root value seeds the pipeline.

### Forms

```python
from quent import Chain

# No root value -- the chain starts empty.
chain = Chain()

# Non-callable root -- used as-is.
chain = Chain(42)
chain = Chain([1, 2, 3])
chain = Chain(None)  # None is a valid root value

# Callable root -- called when the chain runs.
chain = Chain(fetch_data)

# Callable root with arguments -- called with these args when the chain runs.
chain = Chain(fetch_data, user_id, max_results=30)
```

### Root Value Semantics

The root value is evaluated when the chain runs, not when it is constructed:

- **Callable root:** Called (with optional args/kwargs) when `run()` is invoked. The return value becomes the first current value.
- **Non-callable root:** Used as-is. Providing args/kwargs to a non-callable raises `TypeError` at build time.
- **No root (`Chain()`):** The pipeline starts with no value. The first step determines the initial current value.

### Root Value vs Run Value

There are two ways to provide the initial value:

- **At build time:** `Chain(v)` sets a root value.
- **At run time:** `chain.run(v)` injects a run value.

When both exist, the **run value wins** and the build-time root is ignored entirely:

```python
from quent import Chain

chain = Chain('build_time').then(str.upper)

chain.run()           # 'BUILD_TIME' -- root value used
chain.run('run_time') # 'RUN_TIME'   -- run value replaces root
```

The root value (once evaluated) is also captured as the **root value for error handlers**: `except_()` and `finally_()` handlers receive the root value, not the current pipeline value at the point of failure. This is by design -- the root value represents "what this chain was invoked with."

---

## Running a Chain

### `run(v=Null, *args, **kwargs)`

Execute the pipeline and return the final value:

```python
result = Chain(42).then(lambda x: x * 2).run()
# result = 84
```

Pass a value to `.run()` to inject it as the initial input:

```python
chain = Chain().then(lambda x: x * 2)
result = chain.run(10)   # 20
result = chain.run(100)  # 200
```

If `v` is callable, it is called with `(*args, **kwargs)` and the result becomes the initial value. If `v` is not callable and args/kwargs are provided, `TypeError` is raised.

**Return type:** Either a plain value (all sync) or a coroutine (async transition occurred).

### `__call__(v=Null, *args, **kwargs)`

Alias for `run()`. Enables chains as first-class callables:

```python
chain = Chain().then(lambda x: x * 2)
result = chain(10)  # same as chain.run(10)
```

---

## then and do

These are the two fundamental pipeline operations.

### `.then(v, *args, **kwargs)` -- Transform the Value

Append a step whose result **replaces** the current pipeline value:

```python
result = (
  Chain(5)
  .then(lambda x: x * 2)   # current value: 10
  .then(lambda x: x + 1)   # current value: 11
  .then(str)                # current value: '11'
  .run()
)
# result = '11'
```

When `v` is not callable, it replaces the current value directly:

```python
result = Chain(5).then(lambda x: x * 2).then('override').run()
# result = 'override'
```

!!! note
    Providing args/kwargs to a non-callable value raises `TypeError` at build time.

### `.do(fn, *args, **kwargs)` -- Side Effects

Append a side-effect step. `fn` is called, but its return value is **discarded** -- the current pipeline value passes through unchanged:

```python
result = (
  Chain(42)
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
from quent import Chain

def format_number(currency, decimals=2):
  ...

Chain(5).then(format_number, 'USD', decimals=2).run()
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
from quent import Chain

Chain(5).then(str).run()          # str(5) -> '5'
Chain().then(dict).run()          # dict() -> {}
Chain(5).then(42).run()           # 42 (non-callable, replaces value)
```

### Nested Chains

When the step's value is itself a `Chain` instance, the nested chain is executed with the current value passed as its input. Control flow signals (`return_()`, `break_()`) propagate from the nested chain to the outer chain.

```python
from quent import Chain

inner = Chain().then(lambda x: x * 2).then(lambda x: x + 1)

# inner receives current value (5), runs its steps
result = Chain(5).then(inner).run()
# 5 -> 10 -> 11
# result = 11
```

**Edge cases:**

- Nested chains have a depth limit (default: 50) to prevent unbounded recursion. Exceeding the limit raises `QuentException`.
- When a chain is used as a step in another chain, control flow signals propagate through to the outer chain.
- When a chain is executed directly via `.run()`, escaped control flow signals are caught and wrapped in `QuentException`.

### Summary Table

| Priority | Rule | Trigger | Invocation |
|----------|------|---------|------------|
| 1 | Explicit args | Args/kwargs provided | `fn(*args, **kwargs)` |
| 2 | Default | None of the above | `fn(cv)`, `fn()`, or `v` as-is |

---

## foreach and foreach_do

These methods operate on the **elements** of the current pipeline value (which must be iterable).

### `.foreach(fn, *, concurrency=None)` -- Transform Each Element

Apply `fn` to each element and collect the results into a list:

```python
from quent import Chain

result = Chain([1, 2, 3]).foreach(lambda x: x ** 2).run()
# result = [1, 4, 9]
```

The list of results replaces the current pipeline value.

### `.foreach_do(fn, *, concurrency=None)` -- Side-Effect Per Element

Apply `fn` to each element as a side-effect. The **original elements** (not fn's return values) are collected:

```python
from quent import Chain

result = Chain([1, 2, 3]).foreach_do(print).run()
# stdout: 1, 2, 3 (one per line)
# result = [1, 2, 3]  (original elements)
```

### Chaining Collection Operations

These return `self`, so you can compose them:

```python
result = (
  Chain([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  .then(lambda xs: [x for x in xs if x % 2 == 0])  # [2, 4, 6, 8, 10]
  .foreach(lambda x: x ** 2)                         # [4, 16, 36, 64, 100]
  .then(sum)                                          # 220
  .run()
)
```

### Async Support

All three methods support async transparently:

- If `fn` returns an awaitable for any element, the operation transitions to async and awaits it. Subsequent elements continue in async mode.
- Supports both sync iterables (`__iter__`) and async iterables (`__aiter__`). When both protocols are present, the async protocol is preferred if an event loop is running.

```python
async def fetch_details(user_id):
  ...

result = await Chain(user_ids).foreach(fetch_details).run()
```

### Concurrent Execution

Pass a `concurrency` parameter to process elements in parallel:

```python
from quent import Chain

# Process up to 5 elements concurrently
result = await Chain(urls).foreach(fetch, concurrency=5).run()
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

`Chain.break_()` stops iteration early:

```python
from quent import Chain

def process(item):
  if item == 'STOP':
    return Chain.break_()  # stop, return results so far
  return item.upper()

result = Chain(['a', 'b', 'STOP', 'c']).foreach(process).run()
# result = ['A', 'B']
```

With a value, `break_()` **appends** the value to the results collected so far:

```python
def process(item):
  if item < 0:
    return Chain.break_('found_negative')
  return item * 2

result = Chain([1, 2, -1, 3]).foreach(process).run()
# result = [2, 4, 'found_negative']
```

---

## gather -- Concurrent Fan-Out

`.gather(*fns, concurrency=-1, executor=None)` runs multiple functions on the current pipeline value concurrently. Results are returned as a **tuple** in the same positional order as `fns`.

```python
from quent import Chain

results = Chain(data).gather(validate, enrich, score).run()
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
Chain(data).gather(fn_a, fn_b, fn_c, fn_d, concurrency=2).run()
```

Without `concurrency`, all functions run concurrently with no limit.

### Error Behavior

- **Single failure:** The exception propagates directly (not wrapped).
- **Multiple failures:** Regular exceptions are wrapped in an `ExceptionGroup`.
- **Control flow:** `Chain.return_()` takes absolute priority. `Chain.break_()` is not allowed in `gather()` -- it raises `QuentException`.

### After gather: Accessing Results

The result of `gather()` is a tuple. Access individual results by index, or destructure in a lambda:

```python
from quent import Chain

result = (
  Chain(user_id)
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
from quent import Chain

content = (
  Chain('data.txt')
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
from quent import Chain

result = (
  Chain(db.connect)
  .with_do(lambda conn: conn.execute('INSERT INTO ...'))
  .run()
)
# result is the connection object, not the execute result
```

### Sync and Async Context Managers

Both methods work transparently with sync and async context managers:

- **Sync CM:** `__enter__()` / `__exit__()` used.
- **Async CM:** `__aenter__()` / `__aexit__()` used.
- **Dual-protocol:** When an event loop is running, the async protocol is preferred. Otherwise, the sync protocol is used.

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
from quent import Chain

result = (
  Chain(value)
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
  Chain(value)
  .if_(lambda x: x > 0).then(process_positive)
  .else_(process_negative)
  .run()
)
```

!!! warning
    `.else_()` must follow **immediately** after the `.then()` or `.do()` that follows `.if_()` -- no other operations in between. Otherwise, `QuentException` is raised. Only one `.else_()` per `.if_()`.

### Without a Predicate

When `predicate` is omitted, the truthiness of the current pipeline value itself is used:

```python
result = (
  Chain(user_or_none)
  .if_().then(process_user)
  .else_(lambda _: 'no user found')
  .run()
)
# If user_or_none is truthy: result = process_user(user_or_none)
# If user_or_none is falsy:  result = 'no user found'
```

When the chain has no current value (internal Null sentinel), the predicate evaluates to falsy.

### Literal Predicates

When `predicate` is a non-callable value, its truthiness is used directly:

```python
# Truthy literal -- branch always runs
.if_(True).then(transform)

# Falsy literal -- branch never runs (value passes through)
.if_(None).then(transform)
```

### The Truthy Branch

The value passed to `.then()` can be a callable, a non-callable value (used as-is), or a nested Chain:

```python
# Callable -- receives current value
.if_(predicate).then(transform)

# Non-callable -- used as-is
.if_(predicate).then('default_value')

# Nested chain
.if_(predicate).then(Chain().then(validate).then(process))

# Callable with explicit args (Rule 4 -- current value not passed)
.if_(predicate).then(transform, arg1, arg2, key='value')
```

### Predicate Semantics

Predicates use the standard 2-rule calling convention. If args/kwargs are provided to `.if_()`, they are forwarded to the predicate callable.

!!! note
    Nested Chain predicates go through `Chain.run()`, so control flow signals (`return_()`, `break_()`) inside predicates are caught and wrapped in `QuentException`. Predicates should not alter control flow.

### Async Predicates and Branches

Both the predicate and the then/else callables can be sync or async. If either returns an awaitable, the operation transitions to async.

---

## iterate -- Lazy Iteration

`.iterate()` and `.iterate_do()` return a dual sync/async iterator over the chain's output.

### `.iterate(fn=None)`

Returns a `ChainIterator` object. The chain is executed when iteration begins (not when `iterate()` is called). If `fn` is provided, each element is transformed by `fn` before yielding:

```python
from quent import Chain

# Without fn -- yields raw elements
for item in Chain(fetch_all_users).iterate():
  process(item)

# With fn -- yields fn(element)
for name in Chain(fetch_all_users).iterate(lambda u: u.name):
  print(name)
```

### `.iterate_do(fn=None)`

Same as `.iterate()`, but fn's return values are discarded. The original elements are yielded:

```python
for user in Chain(fetch_all_users).iterate_do(log_user):
  # log_user(user) is called for side-effects
  # the original user object is yielded
  process(user)
```

### Sync and Async Iteration

The returned `ChainIterator` supports both `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops):

```python
# Sync iteration
for item in Chain(get_items).iterate():
  handle(item)

# Async iteration
async for item in Chain(get_items).iterate():
  await handle(item)
```

!!! warning
    If you use `for` (sync iteration) but the chain or the `fn` callable returns a coroutine, a `TypeError` is raised. Use `async for` in that case.

### Reusable Iterators

Calling the `ChainIterator` object with arguments creates a new iterator bound to those arguments:

```python
gen = Chain(fetch_page).iterate()

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

The chain's `except_()` and `finally_()` handlers apply to the chain execution that produces the iterable. Exceptions from the iteration callback `fn` during iteration are NOT covered by the chain's handlers -- they propagate directly to the caller.

---

## Control Flow

quent provides two class methods for non-local control flow. These work by raising internal `BaseException` signals that the chain catches and handles. They bypass `except Exception` clauses.

### `Chain.return_(v=<no value>, *args, **kwargs)` -- Early Exit

Exit the chain early, returning `v` as the chain's result:

```python
from quent import Chain

def process(data):
  if data is None:
    return Chain.return_('default')
  return data.upper()

result = Chain(None).then(process).then(further_processing).run()
# result = 'default'  (further_processing is never called)
```

**Value semantics:**

- **No value:** `Chain.return_()` -- the chain returns `None`.
- **Non-callable value:** `Chain.return_(42)` -- the chain returns `42`.
- **Callable value:** `Chain.return_(fn, *args, **kwargs)` -- the callable is invoked when the signal is caught. Its return value becomes the chain's result.

!!! warning "Must use `return`"
    `Chain.return_()` raises an internal signal. You **must** use it with `return` so the signal propagates up the call stack:

    ```python
    # Correct:
    return Chain.return_(value)

    # Wrong -- the signal is raised immediately; if your function catches
    # BaseException, the chain will not see it
    Chain.return_(value)
    ```

### `Chain.break_(v=<no value>, *args, **kwargs)` -- Break from Iteration

Break out of a `.foreach()` or `.foreach_do()` iteration:

```python
from quent import Chain

result = Chain([1, 2, 3, 4, 5]).foreach(
  lambda x: Chain.break_(x * 10) if x == 3 else x * 2
).run()
# result = [2, 4, 30]
# x=1 -> 2, x=2 -> 4, x=3 -> break with value 30 (appended to [2, 4])
```

Without a value, the partial results collected so far are returned:

```python
result = Chain([1, 2, 3, 4, 5]).foreach(
  lambda x: Chain.break_() if x == 3 else x * 2
).run()
# result = [2, 4]
```

!!! warning
    `Chain.break_()` is only valid inside `.foreach()` or `.foreach_do()`. Using it elsewhere raises `QuentException`. Using it inside `gather()` also raises `QuentException`.

### Nested Chain Propagation

Control flow signals propagate through nested chains:

```python
from quent import Chain

validate = (
  Chain()
  .if_(lambda x: not x.get('valid')).then(lambda _: Chain.return_({'error': 'invalid'}))
  .then(check_permissions)
)

pipeline = (
  Chain()
  .then(validate)    # return_() propagates through to outer chain
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

## Nesting Chains

A `Chain` can be used as a step inside another chain. The inner chain executes as a single atomic step:

```python
from quent import Chain

validate = Chain().then(check_schema).then(check_permissions)
transform = Chain().then(normalize).then(enrich)

pipeline = (
  Chain(request)
  .then(validate)   # runs the entire validate chain as one step
  .then(transform)  # runs the entire transform chain as one step
  .then(save)
  .run()
)
```

When a chain is nested:

- The current pipeline value is passed as input to the inner chain.
- The inner chain's final result becomes the outer chain's current value.
- Control flow signals propagate through.
- The inner chain's `except_()` and `finally_()` handlers apply only to the inner chain's execution.

See [Reuse and Patterns](reuse.md) for more on composition with nested chains.

---

## Method Chaining Patterns

Every builder method returns `self`, enabling fluent chains:

```python
from quent import Chain

# Vertical style (recommended for complex chains)
result = (
  Chain(request)
  .then(authenticate)
  .then(authorize)
  .do(log_access)
  .then(process)
  .then(save)
  .except_(handle_error)
  .finally_(cleanup)
  .run()
)

# Inline style (for simple chains)
result = Chain(5).then(lambda x: x * 2).then(str).run()
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
