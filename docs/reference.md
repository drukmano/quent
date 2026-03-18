---
title: "API Reference — Chain, Null, QuentException"
description: "Complete API reference for quent: Chain class with all pipeline methods, Null sentinel, QuentException, calling conventions, and configuration options."
tags:
  - api
  - reference
  - chain
  - methods
search:
  boost: 8
---

# API Reference

All public exports are available from the top-level package:

```python
from quent import (
  Chain, ChainExcInfo, ChainIterator, QuentException,
  __version__,
)
```

---

## Chain

```python
class Chain(v=<no value>, /, *args, **kwargs)
```

A sequential pipeline that transparently bridges synchronous and asynchronous operations. Steps are appended with fluent methods (`.then()`, `.do()`, etc.) and the chain is executed with `.run()`.

### Constructor

```python
Chain(v=<no value>, /, *args, **kwargs)
```

Create a new chain with an optional root value or callable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Root value or callable. Defaults to no value (the internal `Null` sentinel, distinct from `None`). |
| `*args` | `Any` | Positional arguments passed to `v` when it is callable. |
| `**kwargs` | `Any` | Keyword arguments passed to `v` when it is callable. |

**Returns:** `Chain` instance.

**Raises:** `TypeError` if `v` is not callable but `args` or `kwargs` are provided.

```python
# Chain with a literal root value
Chain(5)

# Chain with a callable root — called at run time
Chain(fetch_data, user_id)

# Chain with no root value (value injected at run time)
Chain()
```

!!! note "Chain(None) vs Chain()"
    `Chain(None)` creates a chain with root value `None`. `Chain()` creates a chain with no root value -- the two are distinct. When the current value is `None`, a callable step receives `fn(None)`. When there is no value (internal `Null`), a callable step receives `fn()` with zero arguments. See [Null Sentinel](#null-sentinel).

---

### Pipeline Building Methods

All pipeline building methods return `self` (`Chain`), enabling fluent chaining. These methods record intent at build time -- actual execution happens during `.run()`.

#### then

```python
chain.then(v, /, *args, **kwargs) -> Chain
```

Append a step whose result **replaces** the current pipeline value. `v` can be any value -- if callable, it is called according to the [calling conventions](#calling-conventions); if not callable, it is used as-is.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Value or callable to evaluate. |
| `*args` | `Any` | Positional arguments for `v`. |
| `**kwargs` | `Any` | Keyword arguments for `v`. |

**Returns:** `self` (`Chain`).

**Raises:** `TypeError` if `v` is not callable but `args` or `kwargs` are provided.

```python
Chain(5).then(lambda x: x * 2).run()
# 10

# Non-callable: replaces the current value with a literal
Chain(5).then('hello').run()
# 'hello'

# With explicit arguments (current value is NOT passed)
Chain(5).then(pow, 2, 10).run()
# pow(2, 10) = 1024
```

!!! tip
    Because `then` accepts any value, you can use it to inject constants into the pipeline: `.then(42)` replaces the current value with `42`.

---

#### do

```python
chain.do(fn, /, *args, **kwargs) -> Chain
```

Append a side-effect step. `fn` **must** be callable (raises `TypeError` otherwise). The result of `fn` is **discarded** -- the current pipeline value passes through unchanged. If `fn` returns an awaitable, it is still awaited (to complete the side-effect), but the resolved value is discarded.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Callable to execute as a side-effect. |
| `*args` | `Any` | Positional arguments for `fn`. |
| `**kwargs` | `Any` | Keyword arguments for `fn`. |

**Returns:** `self` (`Chain`).

**Raises:** `TypeError` if `fn` is not callable.

```python
Chain(5).do(print).then(lambda x: x * 2).run()
# prints: 5
# result: 10
```

!!! info "Why do() requires a callable"
    A non-callable `.do(42)` would evaluate `42` as a literal, discard it, and pass the current value through -- indistinguishable from not having the step at all. Requiring callability catches this mistake at build time.

---

#### foreach

```python
chain.foreach(fn, /, *, concurrency=None, executor=None) -> Chain
```

Apply `fn` to each element of the current iterable value. Collects results into a list that replaces the current value. `fn` must be callable.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | *(required)* | Callable applied to each element. |
| `concurrency` | `int \| None` | `None` | Maximum concurrent executions. `None` = sequential. See [Concurrent Execution](#concurrent-execution). |
| `executor` | `Executor \| None` | `None` | Optional executor for sync concurrent execution. When provided, used instead of creating a new ThreadPoolExecutor (caller manages lifecycle). |

**Returns:** `self` (`Chain`). The current value becomes `list[result]`.

**Raises:**

- `TypeError` if `fn` is not callable.
- `TypeError` if `concurrency` is not an integer (including `bool`).
- `ValueError` if `concurrency` is less than 1 (excluding `-1` for unbounded).

```python
Chain([1, 2, 3]).foreach(lambda x: x ** 2).run()
# [1, 4, 9]

# With concurrency: process up to 4 items concurrently
Chain(urls).foreach(fetch, concurrency=4).run()
```

Supports both sync iterables (`__iter__`) and async iterables (`__aiter__`). When both protocols are present, the async protocol is preferred if an async event loop is running (asyncio, trio, or curio). Supports mid-iteration async transition -- if `fn` returns an awaitable for any element, the operation transitions to async automatically.

Use `Chain.break_()` inside `fn` to stop iteration early. Without a value, partial results collected so far are returned. With a value, it is appended to the partial results.

---

#### foreach_do

```python
chain.foreach_do(fn, /, *, concurrency=None, executor=None) -> Chain
```

Apply `fn` as a side-effect to each element. The **original** elements are collected into a list (`fn`'s return values are discarded).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | *(required)* | Side-effect callable applied to each element. |
| `concurrency` | `int \| None` | `None` | Maximum concurrent executions. `None` = sequential. See [Concurrent Execution](#concurrent-execution). |
| `executor` | `Executor \| None` | `None` | Optional executor for sync concurrent execution. When provided, used instead of creating a new ThreadPoolExecutor (caller manages lifecycle). |

**Returns:** `self` (`Chain`). The current value becomes a list of the original elements.

**Raises:** Same as `foreach()`.

```python
Chain([1, 2, 3]).foreach_do(print).run()
# prints: 1, 2, 3
# result: [1, 2, 3]
```

All execution mechanics (concurrent, error, break behavior) are identical to `foreach()`.

---

#### gather

```python
chain.gather(*fns, concurrency=-1, executor=None) -> Chain
```

Run multiple functions on the current pipeline value **concurrently**. Returns a tuple of results in the **same order** as `fns`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*fns` | `Callable` | *(required)* | One or more callables. Each receives the current value. |
| `concurrency` | `int` | `-1` | Maximum simultaneous executions. `-1` = unbounded (all run concurrently). See [Concurrent Execution](#concurrent-execution). |
| `executor` | `Executor \| None` | `None` | Optional executor for sync concurrent execution. When provided, used instead of creating a new ThreadPoolExecutor (caller manages lifecycle). |

**Returns:** `self` (`Chain`). The current value becomes a `tuple` of results.

**Raises:**

- `QuentException` if zero functions are provided.
- `TypeError` if any fn is not callable.
- `TypeError` if `concurrency` is not an integer (including `bool`).
- `ValueError` if `concurrency` is less than 1 (excluding `-1` for unbounded).

```python
Chain(10).gather(
  lambda x: x + 1,
  lambda x: x * 2,
  lambda x: x ** 2,
).run()
# (11, 20, 100)
```

!!! warning "Gather is always concurrent"
    Unlike `foreach()`/`foreach_do()` which default to sequential, `gather()` is **always** concurrent. In sync mode it uses `ThreadPoolExecutor`; in async mode it uses `TaskGroup` (3.11+) or `asyncio.gather` (3.10). There is no sequential fallback.

!!! warning "ExceptionGroup"
    When multiple gathered functions raise concurrently, the exceptions are wrapped in an `ExceptionGroup`. A single failure propagates directly (no wrapping). `Chain.break_()` is not allowed inside `gather()` -- it raises `QuentException`.

---

#### with\_

```python
chain.with_(fn, /, *args, **kwargs) -> Chain
```

Enter the current pipeline value as a context manager, call `fn` with the context value (the result of `__enter__` or `__aenter__`). `fn`'s result **replaces** the current pipeline value. Works with both sync and async context managers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Callable to evaluate with the context value. Must be callable. |
| `*args` | `Any` | Positional arguments for `fn`. |
| `**kwargs` | `Any` | Keyword arguments for `fn`. |

**Returns:** `self` (`Chain`).

**Raises:** `TypeError` if `fn` is not callable.

```python
Chain(open('data.txt')).with_(lambda f: f.read()).run()
# reads file contents; file is properly closed
```

**Exception suppression:** If `fn` raises and `__exit__` returns a truthy value, the pipeline continues with `None` as the current value.

**Dual-protocol objects:** When the current value supports both sync and async context manager protocols and an async event loop is running (asyncio, trio, or curio), the async protocol is preferred.

**Control flow signals:** If `fn` raises `return_()` or `break_()`, `__exit__` is called with no exception info (clean exit), and the signal propagates.

---

#### with\_do

```python
chain.with_do(fn, /, *args, **kwargs) -> Chain
```

Same as `with_()` but `fn`'s result is **discarded**. The pipeline value remains the original value (the context manager object itself, not the `__enter__` result).

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Callable to execute as a side-effect inside the context. |
| `*args` | `Any` | Positional arguments for `fn`. |
| `**kwargs` | `Any` | Keyword arguments for `fn`. |

**Returns:** `self` (`Chain`).

**Raises:** `TypeError` if `fn` is not callable.

If an exception is suppressed by `__exit__`, the original pipeline value passes through (not `None` -- unlike `with_`).

---

#### if\_

```python
chain.if_(predicate=None, /, *args, **kwargs) -> Chain
```

Begin a conditional branch. Must be followed by `.then()` or `.do()`, which registers the truthy branch. When `predicate` is `None`, the truthiness of the **current pipeline value** is used. When `predicate` is callable, it is invoked per the standard 2-rule calling convention. When `predicate` is a non-callable value, its truthiness is used directly.

| Parameter | Type | Description |
|-----------|------|-------------|
| `predicate` | `Any \| None` | Callable, literal value, or `None` (uses current value truthiness). |
| `*args` | `Any` | Positional arguments forwarded to the predicate callable. |
| `**kwargs` | `Any` | Keyword arguments forwarded to the predicate callable. |

**Returns:** `self` (`Chain`). The next `.then()` or `.do()` becomes the truthy branch.

**Raises:** `QuentException` if `if_()` is called while another `if_()` is already pending.

Call `.then(fn, *args, **kwargs)` or `.do(fn, *args, **kwargs)` after `if_()` to specify the truthy branch:

```python
Chain(data).if_(lambda x: len(x) > 0).then(process).run()

# with arguments passed to the branch function
Chain(data).if_(predicate).then(save, 'backup', compress=True).run()

# else branch
Chain(-5).if_(lambda x: x > 0).then(str).else_(abs).run()  # 5
```

!!! note "Predicate semantics"
    When `predicate` is `None` and the chain has no current value (Null), the predicate evaluates to falsy.

---

#### else\_

```python
chain.else_(v, /, *args, **kwargs) -> Chain
```

Register an else branch for the **immediately preceding** `if_()`. Evaluated when `if_()`'s predicate was falsy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Value or callable for the else branch. |
| `*args` | `Any` | Positional arguments for `v`. |
| `**kwargs` | `Any` | Keyword arguments for `v`. |

**Returns:** `self` (`Chain`).

**Raises:** `QuentException` if not immediately preceded by `if_()`, or if a second `else_()` is registered on the same `if_()`.

```python
Chain(value).if_(
  lambda x: x > 0
).then(
  process_positive
).else_(
  process_negative
).run()
```

!!! warning
    `else_()` must be called **immediately** after `.then()`/`.do()` on an `if_()` (or after `if_()` itself if no branch is given) with no other operations in between. Any intervening operation raises `QuentException`.

---

#### name

```python
chain.name(label, /) -> Chain
```

Assign a user-provided label for traceback identification. The label appears in chain visualizations (`Chain[label](root)`), exception notes, and `repr(chain)`. No effect on execution semantics -- purely for debuggability.

| Parameter | Type | Description |
|-----------|------|-------------|
| `label` | `str` | A short descriptive string identifying this chain. |

**Returns:** `self` (`Chain`).

```python
Chain(fetch).name('auth_pipeline').then(validate).run()
```

---

### Concurrent Execution

The `concurrency` parameter is available on `.foreach()`, `.foreach_do()`, and `.gather()`.

| `concurrency` | Sync mode | Async mode |
|---------------|-----------|------------|
| `None` (default) | Sequential (foreach/foreach_do). All concurrent (gather). | Sequential (foreach/foreach_do). All concurrent (gather). |
| `-1` (unbounded) | `ThreadPoolExecutor` with one worker per item/fn | All tasks launched concurrently (no semaphore) |
| Positive integer | `ThreadPoolExecutor(max_workers=concurrency)` | `asyncio.Semaphore(concurrency)` to limit concurrent tasks |

**Validation:** Must be a positive integer (`>= 1`), `-1` (unbounded), or `None`. Booleans are rejected (`TypeError`). Values less than `1` (excluding `-1`) raise `ValueError`.

**Sync/async detection:** The first item/function is probed. If it returns an awaitable, async path is used. If not, sync path. Mixed sync/async within a single concurrent operation raises `TypeError`.

**Concurrent iteration materializes eagerly:** The entire input iterable is converted to a list before processing begins. Do not use with infinite or very large iterables.

!!! note "Unbounded concurrency"
    Use `concurrency=-1` for unbounded concurrent execution. The effective concurrency equals the number of items at runtime.

---

### Error Handling Methods

#### except\_

```python
chain.except_(fn, /, *args, exceptions=None, reraise=False, **kwargs) -> Chain
```

Register an exception handler. Only **one** `except_` per chain.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | *(required)* | Handler callable. |
| `*args` | `Any` | | Positional arguments for `fn`. |
| `exceptions` | `type \| Iterable[type] \| None` | `None` | Exception types to catch. Defaults to `(Exception,)` when `None`. |
| `reraise` | `bool` | `False` | When `True`, re-raise the original exception after running the handler. |
| `**kwargs` | `Any` | | Keyword arguments for `fn`. |

**Returns:** `self` (`Chain`).

**Raises:**

- `TypeError` if `fn` is not callable.
- `QuentException` if `except_` is already registered.
- `QuentException` if `exceptions` is an empty iterable.
- `TypeError` if any value in `exceptions` is not a `BaseException` subclass.
- `TypeError` if a string is passed as `exceptions` (common mistake: `"ValueError"` instead of `ValueError`).

**Handler calling convention (uses the standard 2-rule convention):**

The handler receives a `ChainExcInfo(exc, root_value)` as its current value. The standard 2-rule dispatch applies:

| Registration | Handler invocation |
|-------------|-------------------|
| `except_(handler)` | `handler(ChainExcInfo(exc, root_value))` |
| `except_(handler, arg1, arg2)` | `handler(arg1, arg2)` -- ChainExcInfo NOT passed |
| `except_(handler, key=val)` | `handler(key=val)` -- ChainExcInfo NOT passed |

Access the exception via `.exc` and the root value via `.root_value` on the `ChainExcInfo` NamedTuple.

The `root_value` is normalized to `None` when the chain has no root value.

**`reraise=False` (default):** Handler's return value replaces the chain's result. Exception is consumed.

**`reraise=True`:** Handler runs for side-effects only, then the original exception is re-raised. Handler's return value is discarded.

**Handler failure with `reraise=True`:** If the handler raises an `Exception`, the handler's exception is discarded (with a `RuntimeWarning`), and the original exception is re-raised. If the handler raises a `BaseException` (e.g., `KeyboardInterrupt`), it propagates naturally.

**Handler failure with `reraise=False`:** The handler's exception propagates. The original chain exception is set as `__cause__` via `raise handler_exc from original_exc`.

```python
# Swallow exception, return fallback
Chain(fetch_data).except_(lambda ei: 'fallback').run()

# Log and re-raise -- ei is ChainExcInfo(exc, root_value)
Chain(fetch_data).except_(
  lambda ei: logger.error('Failed on %s: %s', ei.root_value, ei.exc),
  reraise=True,
).run()

# Catch specific exception types
Chain(fetch_data).except_(
  handle_error,
  exceptions=ConnectionError,
).run()
```

!!! warning "BaseException subclasses"
    A `RuntimeWarning` is emitted if you configure `except_()` to catch `BaseException` subclasses that are not `Exception` subclasses (e.g., `KeyboardInterrupt`, `SystemExit`).

!!! warning "Control flow in except handlers"
    Using `Chain.return_()` or `Chain.break_()` inside an except handler raises `QuentException`. Control flow signals are not allowed in error handlers.

---

#### finally\_

```python
chain.finally_(fn, /, *args, **kwargs) -> Chain
```

Register a cleanup handler. Only **one** `finally_` per chain. Always runs regardless of success or failure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Cleanup callable. |
| `*args` | `Any` | Positional arguments for `fn`. |
| `**kwargs` | `Any` | Keyword arguments for `fn`. |

**Returns:** `self` (`Chain`).

**Raises:**

- `TypeError` if `fn` is not callable.
- `QuentException` if `finally_` is already registered.

!!! important "Finally handler semantics"
    - Receives the chain's **root value** (normalized to `None` if absent), not the current pipeline value.
    - Follows the **standard** calling conventions (not the except handler convention).
    - Return value is **always discarded**.
    - If the handler raises while an exception is active, the handler's exception **replaces** the original (preserved as `__context__`).
    - Control flow signals (`return_()`, `break_()`) are not allowed -- raises `QuentException`.

```python
Chain(acquire_resource).then(process).finally_(release_resource).run()
```

**Async finally in sync chains:** When a sync chain's finally handler returns a coroutine, the engine performs an async transition: `run()` returns a coroutine instead of a plain value. When the caller awaits this coroutine, the finally handler's coroutine is awaited first, and then the chain's result is returned (or the active exception is re-raised).

---

### Execution Order

The full error handling flow:

1. Pipeline steps execute sequentially.
2. If a step raises matching `exceptions`:
    - Except handler runs (if registered).
    - `reraise=False`: handler's return value becomes the result; finally runs in success context.
    - `reraise=True`: original exception re-raised; finally runs in failure context.
3. If a step raises a non-matching exception: exception propagates; finally runs in failure context.
4. On success: finally runs in success context.
5. Finally always runs last.

---

### Execution Methods

#### run

```python
chain.run(v=<no value>, /, *args, **kwargs) -> Any
```

Execute the chain and return the final result.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Optional value injected into the chain. Overrides the root value if both are present. |
| `*args` | `Any` | Positional arguments for `v` if callable. |
| `**kwargs` | `Any` | Keyword arguments for `v` if callable. |

**Returns:** The final pipeline value. Returns a **coroutine** if any step returned an awaitable (caller must `await` it). Returns `None` if no value was produced.

**Raises:**

- `TypeError` if `v` is not callable but `args`/`kwargs` are provided.
- `QuentException` if a control flow signal escapes the pipeline.

```python
# Execute with the root value
Chain(5).then(lambda x: x * 2).run()
# 10

# Inject a value at run time (overrides root)
chain = Chain().then(lambda x: x * 2)
chain.run(5)   # 10
chain.run(10)  # 20
```

**Run value vs root value:** When both exist, the run value replaces the root entirely. `Chain(A).then(B).run(C)` is equivalent to `Chain(C).then(B).run()`.

**Root value capture:** The evaluated root/run value is captured as the "root value" for error handlers. `except_()` and `finally_()` receive this root value, not the current pipeline value at the point of failure.

---

#### \_\_call\_\_

```python
chain(v=<no value>, /, *args, **kwargs) -> Any
```

Alias for [`run()`](#run). A chain instance is directly callable:

```python
chain = Chain(5).then(lambda x: x * 2)
chain()  # 10
```

---

### Iteration Methods

#### iterate

```python
chain.iterate(fn=None) -> ChainIterator
```

Return a dual sync/async iterator over the chain's output. The chain is executed when iteration begins (not when `iterate()` is called). Each element of the iterable result is yielded.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable \| None` | Optional transform applied to each element. `None` = yield as-is. |

**Returns:** `ChainIterator` -- supports both `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops).

```python
# Sync iteration
for item in chain.iterate():
  process(item)

# Async iteration
async for item in chain.iterate():
  await process(item)

# With transform
for name in chain.iterate(lambda item: item['name']):
  print(name)
```

**Error behavior:**

- If `fn` returns an awaitable during sync iteration (`for`), a `TypeError` is raised directing the user to use `async for`.
- Exceptions from `fn` propagate directly to the caller at the iteration point -- they are NOT covered by the chain's `except_()` handlers.

**Control flow in iteration:**

- `return_(v)`: Yields the value (if provided) as a final item before stopping. Previously yielded items are preserved.
- `break_(v)`: Yields the value (if provided) before stopping. Without a value, stops immediately.

!!! note "Callable reuse"
    The returned iterator is callable. Calling it with arguments creates a new iterator with those arguments as the run-time parameters:

    ```python
    it = chain.iterate(fn)
    for item in it:          # runs chain with no arguments
      ...
    for item in it(value):   # runs chain with `value` as run value
      ...
    ```

---

#### iterate\_do

```python
chain.iterate_do(fn=None) -> ChainIterator
```

Like `iterate()` but `fn`'s return values are **discarded**. The original elements are yielded.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable \| None` | Optional side-effect callable. |

**Returns:** `ChainIterator`.

```python
for item in chain.iterate_do(print):
  process(item)
```

---

#### flat\_iterate

```python
chain.flat_iterate(fn=None, *, flush=None) -> ChainIterator
```

Return a dual sync/async flatmap iterator over the chain's output. Each element of the chain's iterable result is either iterated directly (when `fn` is `None`, flattening one level of nesting) or transformed by `fn` into a sub-iterable whose items are individually yielded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable \| None` | `None` | Optional callable that receives each element and returns an iterable. Each item from the returned iterable is yielded individually. When `None`, each source element is iterated directly (flattening one level). |
| `flush` | `Callable \| None` | `None` | Optional zero-argument callable invoked once after the source iterable is fully consumed. Must return an iterable; each item is yielded into the stream. Intended for emitting buffered or remaining items after the source ends (e.g., flushing a codec buffer). |

**Returns:** `ChainIterator` -- supports both `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops).

```python
# Flatten one level of nesting (fn=None)
for item in Chain([[1, 2], [3, 4]]).flat_iterate():
  print(item)  # 1, 2, 3, 4

# Transform each element into a sub-iterable
for word in Chain(['hello world', 'foo bar']).flat_iterate(str.split):
  print(word)  # hello, world, foo, bar

# With flush -- emit remaining buffered items after source ends
buffer = []
def chunk(item):
  buffer.append(item)
  if len(buffer) >= 3:
    result, buffer[:] = buffer[:], []
    return result
  return []

def flush_buffer():
  return buffer

for chunk in Chain(range(7)).flat_iterate(chunk, flush=flush_buffer):
  print(chunk)
```

All iteration behavior -- sync/async support, error handling, deferred `finally_()`, control flow, iterator reuse -- matches `iterate()`.

**Error behavior:**

- Exceptions from `fn` propagate to the caller at the iteration point, as with `iterate()`.
- If `flush()` raises, the exception propagates at the iteration point.
- If `fn` or `flush` returns an awaitable during sync iteration (`for`), a `TypeError` is raised directing the user to use `async for`.

---

#### flat\_iterate\_do

```python
chain.flat_iterate_do(fn=None, *, flush=None) -> ChainIterator
```

Like `flat_iterate()`, but `fn` runs as a side-effect -- its returned iterable is fully consumed (driving side-effects) but not yielded. The original source elements are yielded instead.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable \| None` | `None` | Optional side-effect callable. Its returned iterable is consumed but discarded. |
| `flush` | `Callable \| None` | `None` | Optional zero-argument callable. Output is yielded normally (not discarded). |

**Returns:** `ChainIterator`.

```python
for item in Chain([[1, 2], [3]]).flat_iterate_do(lambda sub: [print(x) for x in sub]):
  print('source:', item)
  # prints each sub-item via fn, then yields the original source element
```

When `fn` is `None`, behaves identically to `flat_iterate()` with no `fn` (flattens one level). The `flush` output is always yielded (the "do" discard semantic applies only to `fn`'s results).

---

#### Deferred with\_ in Iteration

When `.with_(fn)` or `.with_do(fn)` is the **last pipeline step** before an iteration terminal (`.iterate()`, `.iterate_do()`, `.flat_iterate()`, `.flat_iterate_do()`), context manager entry is **deferred** to iteration time. The context manager remains open for the entire duration of iteration and is exited when iteration ends.

!!! warning "Only `.with_(fn)` is supported"
    Bare `.with_()` (no argument) is prohibited. Only `.with_(fn)` with an explicit callable triggers deferred context manager wrapping.

**Why deferral?** Without deferral, `.with_()` would enter the CM during the chain's `run()` phase and exit it before iteration begins -- the resource would be closed before any items are consumed. Deferral keeps the CM open throughout iteration, matching the natural lifetime of `with` blocks in Python.

**Lifecycle:**

1. The chain runs normally, producing a value that must be a context manager.
2. At iteration start, the CM is entered via `__enter__()` (or `__aenter__()`).
3. If `.with_(fn)` was used: `fn` is invoked with the context value per the standard calling convention. The result becomes the iterable for iteration.
4. If `.with_do(fn)` was used: `fn` runs as a side-effect (result discarded); the CM object itself becomes the iterable (it must be iterable).
5. Iteration proceeds with the CM open.
6. The CM is exited in the generator's `finally:` block, guaranteeing cleanup on all exit paths (normal exhaustion, `break`, exceptions, generator `.close()`).

**CM exit semantics:**

- **Normal completion / source exhausted:** `__exit__(None, None, None)`.
- **`break`, `return_()`, `break_()` (control flow):** `__exit__(None, None, None)` -- control flow signals are not errors.
- **Generator `.close()` / `GeneratorExit`:** `__exit__(None, None, None)`.
- **Exception during iteration:** `__exit__(*sys.exc_info())` -- the CM receives the exception. If `__exit__` returns truthy, the exception is suppressed.

**Ordering with deferred `finally_()`:** When both a deferred `with_` and a deferred `finally_()` are active, the CM exits first, then the deferred `finally_()` runs. The deferred finally runs even if `__exit__` raises.

```python
# File stays open for entire iteration, then closes
for line in Chain(open, 'data.txt').with_(lambda f: f).iterate(str.strip):
  process(line)

# Async context manager with deferred cleanup
async for row in Chain(db.connect).with_(lambda conn: conn.cursor()).iterate():
  process(row)
```

---

### Class Methods

#### from\_steps

```python
Chain.from_steps(*steps) -> Chain
```

Construct a chain from a sequence of steps, each appended via `.then()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `*steps` | `Any` | Variadic positional arguments. Each becomes a `.then()` step. If a single argument is passed and it is a `list` or `tuple`, it is unpacked as the step sequence. |

**Returns:** A new `Chain` instance with no root value.

**Equivalence:** `Chain.from_steps(a, b, c)` is equivalent to `Chain().then(a).then(b).then(c)`.

Steps can be callables, literal values, or nested Chains -- anything `.then()` accepts. `Chain.from_steps()` with no arguments returns an empty chain (equivalent to `Chain()`).

```python
# Variadic form
pipeline = Chain.from_steps(validate, normalize, str.upper)
pipeline.run('  hello  ')

# List form -- useful for dynamic pipeline construction
steps = [validate, normalize, str.upper]
pipeline = Chain.from_steps(steps)
pipeline.run('  hello  ')

# Dynamic pipeline from a plugin registry
plugins = load_plugins()
pipeline = Chain.from_steps([p.transform for p in plugins])
```

---

### Reuse Methods

#### clone

```python
chain.clone() -> Chain
```

Create an independent copy of this chain.

**Returns:** A new `Chain` of the same type (subclass-safe).

**What is copied:**

- Pipeline structure (all step nodes) -- deep-copied. The clone has its own independent linked list.
- Nested chains within steps are **recursively cloned**.
- Conditional operations (`if_`/`else_`) are deep-copied.
- Error handler step nodes are cloned. Handler callables that are `Chain` instances are recursively cloned; non-chain callables are shared by reference.
- Keyword argument dictionaries are shallow-copied (mutable). Positional argument tuples are shared (immutable).

**What is shared by reference:**

- All callables (except Chain instances, which are always recursively cloned).
- Values and argument objects.
- Exception type tuples for `except_()`.

```python
base = Chain(fetch).then(validate)
branch_a = base.clone().then(transform_a)
branch_b = base.clone().then(transform_b)
```

Clones always behave as top-level chains, regardless of whether the original was nested.

---

#### decorator

```python
chain.decorator() -> Callable[..., Callable[..., Any]]
```

Wrap the chain as a function decorator. The decorated function's return value becomes the chain's input. The chain is cloned internally.

**Returns:** A decorator function.

```python
@Chain().then(lambda x: x.strip()).then(str.upper).decorator()
def get_name():
  return '  alice  '

get_name()  # 'ALICE'
```

The decorated function preserves its original signature via `functools.wraps`. Control flow signals that escape the decorated chain are caught and wrapped in `QuentException`.

---

### Control Flow (Class Methods)

#### return\_

```python
Chain.return_(v=<no value>, /, *args, **kwargs) -> NoReturn
```

Signal early termination of chain execution. The optional value becomes the chain's result.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Return value. When `Null`, the chain returns `None`. |
| `*args` | `Any` | Positional arguments for `v` if callable. |
| `**kwargs` | `Any` | Keyword arguments for `v` if callable. |

**Value semantics:**

- **No value:** `Chain.return_()` -- chain returns `None`.
- **Non-callable:** `Chain.return_(42)` -- chain returns `42`.
- **Callable:** `Chain.return_(fn, *args)` -- `fn` is called when the signal is caught; its return value becomes the result.

**Nested chain propagation:** Propagates up to the outermost chain. The nested chain does not catch the signal.

!!! warning "Must use `return`"
    `Chain.return_()` raises an internal exception. Always write `return Chain.return_(...)` so the signal propagates correctly and linters don't flag subsequent code as unreachable.

**Restrictions:** Raises `QuentException` if used in `except_()` or `finally_()` handlers.

---

#### break\_

```python
Chain.break_(v=<no value>, /, *args, **kwargs) -> NoReturn
```

Signal early termination of a `foreach()`/`foreach_do()` iteration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Break value. When provided, **appended** to partial results. When `Null`, partial results are returned as-is. |
| `*args` | `Any` | Positional arguments for `v` if callable. |
| `**kwargs` | `Any` | Keyword arguments for `v` if callable. |

**Raises:** `QuentException` if used outside of a `foreach`/`foreach_do` operation.

```python
Chain([1, 2, 3, 4, 5]).foreach(
  lambda x: Chain.break_() if x == 3 else x * 2
).run()
# [2, 4]  -- partial results up to the break

Chain([1, 2, 3, 4, 5]).foreach(
  lambda x: Chain.break_(x * 10) if x == 3 else x * 2
).run()
# [2, 4, 30]  -- break value appended to partial results
```

**Concurrent iteration break:** The break from the earliest input index wins. Results from later indices are discarded.

**Priority in concurrent operations:** `return_()` > `break_()` > regular exceptions.

**Restrictions:** Not allowed in `except_()`, `finally_()` handlers, or `gather()`.

---

### Instrumentation

#### on\_step

```python
Chain.on_step: ClassVar[Callable[[Chain, str, Any, Any, int], None] | None] = None
```

Class-level callback for chain execution instrumentation. Called after each pipeline step completes.

| Argument | Type | Description |
|----------|------|-------------|
| `chain` | `Chain` | The chain instance being executed. |
| `step_name` | `str` | Method name: `'root'`, `'then'`, `'do'`, `'foreach'`, `'foreach_do'`, `'gather'`, `'with_'`, `'with_do'`, `'if_'`, `'except_'`, or `'finally_'`. |
| `input_value` | `Any` | The input value the step received. |
| `result` | `Any` | The value produced by the step. |
| `elapsed_ns` | `int` | Wall-clock nanoseconds via `time.perf_counter_ns()`. |

**Zero overhead when disabled:** When `on_step` is `None` (default), no timing or callback dispatch occurs. The code path is short-circuited entirely.

**Error handling:** If the callback raises, it is logged at WARNING level and chain execution continues uninterrupted.

**Thread safety:** `on_step` is class-level. Set it before concurrent chain execution begins. Subclass overrides are respected.

```python
Chain.on_step = lambda chain, step, inp, result, ns: print(f'{step}: {ns/1e6:.1f}ms')
Chain(5).then(lambda x: x * 2).run()
Chain.on_step = None  # disable
```

---

### Context API

Pipeline steps are positional -- each step receives only the current value from the immediately preceding step. When non-adjacent steps need to share data (e.g., an early step produces a value that a later step needs, but intermediate transformations change the current value), the context API provides named storage scoped to the execution context, accessible from any step without altering the pipeline's value flow.

Storage is backed by a `ContextVar`-based dictionary. Each `set()` creates a new dict (copy-on-write), ensuring concurrent workers (via `foreach`/`gather` with `concurrency`) are properly isolated -- a worker's `set()` does not affect the parent or sibling workers.

#### set (instance -- pipeline step)

```python
chain.set(key: str) -> Chain
chain.set(key: str, value: Any) -> Chain
```

Append a pipeline step that stores a value under `key` in the execution context. The current pipeline value is **not changed** (like `.do()`).

| Form | Behavior |
|------|----------|
| `chain.set(key)` | Stores the current pipeline value under `key`. |
| `chain.set(key, value)` | Stores the explicit `value` under `key`. |

**Returns:** `self` (`Chain`).

```python
result = (
  Chain(fetch_user)
  .set('user')                        # store current value (the user) in context
  .set('source', 'api')              # store explicit value 'api' under 'source'
  .then(validate_permissions)         # transform continues with original user
  .get('user')                        # retrieve original user
  .then(format_response)
  .run(user_id)
)
```

#### set (class -- immediate)

```python
Chain.set(key: str, value: Any) -> None
```

Store a value in the execution context immediately. This is **not** a pipeline step -- it takes effect at the call site, not during `run()`.

```python
Chain.set('config', load_config())    # pre-populate context before running pipelines
```

#### get (instance -- pipeline step)

```python
chain.get(key: str) -> Chain
chain.get(key: str, default: Any) -> Chain
```

Append a pipeline step that retrieves the value stored under `key` from the execution context. The retrieved value **replaces** the current value (like `.then()`).

| Scenario | Behavior |
|----------|----------|
| Key found | Stored value becomes the new current value. |
| Key not found, no default | Raises `KeyError` at execution time. |
| Key not found, default provided | Default becomes the new current value. |

**Returns:** `self` (`Chain`).

```python
result = (
  Chain(fetch_user)
  .set('user')                        # store user in context
  .then(transform)                    # current value changes
  .get('user')                        # retrieve original user -- becomes current value
  .then(format_response)
  .run(user_id)
)
```

#### get (class -- immediate)

```python
Chain.get(key: str, default: Any = <missing>) -> Any
```

Retrieve a value from the execution context immediately. This is **not** a pipeline step.

- If `key` is found: returns the stored value.
- If `key` is not found and no `default`: raises `KeyError`.
- If `key` is not found and `default` provided: returns `default`.

```python
Chain.set('config', load_config())
result = (
  Chain(fetch_data)
  .then(lambda data: process(data, Chain.get('config')))
  .run()
)
```

!!! note "Dual dispatch"
    Both `set` and `get` are Python descriptors that dispatch differently based on instance vs. class access. Instance access (`chain.set(...)` / `chain.get(...)`) appends a pipeline step. Class access (`Chain.set(...)` / `Chain.get(...)`) operates on context immediately.

!!! warning "if_() constraint"
    `.set()` and `.get()` do not consume a pending `if_()`. Calling them while `if_()` is pending raises `QuentException`.

---

### Dunder Methods

#### \_\_bool\_\_

```python
chain.__bool__() -> bool
```

Always returns `True`. A chain instance is always truthy.

---

#### \_\_repr\_\_

```python
chain.__repr__() -> str
```

Returns the chain visualization (multiline, indented) — the same format used in traceback injection, but without the `<----` error marker. When `.name(label)` has been called, renders as `Chain[label](root)`. Respects `QUENT_TRACEBACK_VALUES=0`.

```python
repr(Chain(fetch_data).then(validate).do(log))
# Chain(fetch_data)
#     .then(validate)
#     .do(log)
```

---

#### Copying

`copy.copy()` and `copy.deepcopy()` are blocked on Chain objects (`TypeError`). A shallow or deep copy would produce a broken object with shared linked-list structure. Use [`.clone()`](#clone) to produce a correct independent copy.

Pickling is **not** blocked — most chain contents (lambdas, closures, bound methods) will naturally fail to pickle, but quent does not enforce this.

---

## ChainIterator

```python
from quent import ChainIterator
```

The iterator object returned by `chain.iterate()` and `chain.iterate_do()`. Supports both `__iter__` (sync) and `__aiter__` (async). Is callable -- calling with arguments creates a new iterator with those arguments as run-time parameters.

---

## ChainExcInfo

```python
from quent import ChainExcInfo
```

A `NamedTuple` with two fields passed to `except_()` handlers as their current value:

| Field | Type | Description |
|-------|------|-------------|
| `exc` | `BaseException` | The caught exception. |
| `root_value` | `Any` | The chain's evaluated root value, normalized to `None` when absent. |

```python
from quent import Chain, ChainExcInfo

def handle_error(ei: ChainExcInfo):
  print(f'Failed on {ei.root_value}: {ei.exc}')
  return 'fallback'

Chain(42).then(lambda x: 1/0).except_(handle_error).run()
# prints: Failed on 42: division by zero
# returns: 'fallback'
```

---

## Null Sentinel

`Null` is an internal sentinel used to distinguish "no value was provided" from `None`. It is never exposed to user code -- `run()` returns `None` (not `Null`) when no value is produced, and handlers always receive `None` when no root value exists. You may see `<Null>` mentioned in tracebacks or debug output; it indicates the absence of a value rather than a `None` value.

---

## QuentException

```python
from quent import QuentException
```

Subclass of `Exception` raised for quent API misuse. Never raised for errors in user code.

| Error | Cause |
|-------|-------|
| Duplicate handler | Second `except_()` or `finally_()` on same chain |
| Escaped signal | `_Return` or `_Break` escaped past `run()` |
| `break_()` outside iteration | Used outside `foreach`/`foreach_do` |
| `else_()` without `if_()` | No immediately preceding `if_()` |
| Pending `if_()` | `run()` called while `if_()` is pending (no `.then()`/`.do()` consumed it) |
| Control flow in handlers | `return_()`/`break_()` in `except_`/`finally_` |
| Nesting depth exceeded | Exceeded depth limit (default 50) |

---

## Calling Conventions

### Standard Calling Conventions (2 Rules)

Applied in priority order. First match wins.

| Priority | Rule | Trigger | Invocation |
|----------|------|---------|------------|
| 1 | **Explicit Args** | Args/kwargs provided at registration | `fn(*args, **kwargs)` -- current value NOT passed |
| 2 | **Default** | None of the above | Callable + value: `fn(cv)`. Callable + no value: `fn()`. Non-callable: returned as-is |

Nested chains are detected via duck-typing (`_quent_is_chain`) and execute with the current value as input.

**Constraints:**

- Providing args/kwargs to a non-callable raises `TypeError` at build time.

### Except Handler Calling Convention

Uses the **standard 2-rule convention**. The handler's current value is a `ChainExcInfo(exc, root_value)` NamedTuple:

| Registration | Handler invocation |
|-------------|-------------------|
| `except_(handler)` | `handler(ChainExcInfo(exc, root_value))` |
| `except_(handler, arg1, arg2)` | `handler(arg1, arg2)` -- ChainExcInfo NOT passed |
| `except_(handler, key=val)` | `handler(key=val)` -- ChainExcInfo NOT passed |

The `root_value` is normalized to `None` when the chain has no root value.

### Finally Handler Calling Convention

Follows the **standard** calling conventions (2 rules). Receives the root value (normalized to `None` if absent) as its current value. Return value is always discarded.

---

## Enhanced Tracebacks

### Chain Visualization

When an exception occurs during chain execution, a synthetic `<quent>` frame is injected into the traceback:

```
Traceback (most recent call last):
  File "<quent>", line 1, in
    Chain(fetch_data)
    .then(validate)
    .then(transform) <----
    .do(log)
ValueError: invalid data
```

The `<----` marker points to the step that raised the exception. Nested chains are rendered with increasing indentation.

### Internal Frame Cleaning

All quent-internal frames are removed from tracebacks. Only user code and the synthetic `<quent>` frame are shown. Cleaning applies recursively to `__cause__`, `__context__`, and `ExceptionGroup` sub-exceptions.

### Exception Notes (Python 3.11+)

A concise note is attached: `quent: exception at .then(validate) in Chain(fetch_data)`. Attached once per exception (idempotent).

### Hook Patching

Two patches installed at import time:

- `sys.excepthook` -- for uncaught exceptions.
- `traceback.TracebackException.__init__` -- for `logging`, `traceback.print_exc()`, etc.

### Visualization Limits

- Nesting depth: 50
- Links per level: 100
- Total length: 10,000 characters
- Total recursive calls: 500 per level

### Repr Sanitization (CWE-117)

All `repr()` output in visualizations is sanitized: ANSI escape sequences stripped, Unicode control characters stripped, length truncated to 200 characters.

---

## Alternative Event Loops (Trio & Curio)

quent supports **asyncio**, **trio**, and **curio** event loops. Async pipelines work transparently under any of these runtimes -- no configuration or adapter code is required.

### How It Works

Event loop detection uses `sys.modules` lookups to check whether a runtime's loop is active. This adds zero overhead when a library is not loaded (~50ns dict lookup returning `None`). Detection order:

1. **asyncio** -- checked first via the C-level `asyncio._get_running_loop()` for performance.
2. **trio** -- detected via `trio.lowlevel.current_trio_token()` (only probed if `trio.lowlevel` is in `sys.modules`).
3. **curio** -- detected via `curio.meta.curio_running()` (only probed if `curio.meta` is in `sys.modules`).

### Dual-Protocol Preference

When a pipeline value supports both sync and async protocols -- context managers (`__enter__`/`__exit__` vs `__aenter__`/`__aexit__`) or iterables (`__iter__` vs `__aiter__`) -- and any async event loop is running, the **async protocol** is preferred. This applies uniformly across asyncio, trio, and curio.

### Usage

No special API is needed. Use the same `Chain` API under any runtime:

```python
import trio
from quent import Chain

async def async_double(x):
  return x * 2

async def main():
  result = await Chain(5).then(async_double).run()
  print(result)  # 10

trio.run(main)
```

```python
import curio
from quent import Chain

async def main():
  result = await Chain(10).then(lambda x: x + 1).run()
  print(result)  # 11

curio.run(main)
```

Dual-protocol context managers and iterables automatically use the async protocol under trio and curio, just as they do under asyncio:

```python
import trio
from quent import Chain

# Dual-protocol CM uses __aenter__/__aexit__ under trio
async def main():
  result = await Chain(dual_protocol_cm).with_(lambda ctx: ctx).run()

trio.run(main)
```

### Differences from asyncio

- **Concurrency operations** (`gather`, `foreach` with `concurrency`): The async concurrent path uses `asyncio.Semaphore` and `asyncio.TaskGroup` (or `asyncio.gather` on 3.10). Under trio or curio, async steps still work correctly for sequential execution, but the concurrent async path relies on asyncio primitives. For concurrent async pipelines under trio or curio, ensure the steps are compatible with asyncio task scheduling.
- **Event loop detection** is lightweight and non-invasive -- it never imports trio or curio, only checks `sys.modules` for already-loaded modules.

---

## Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `QUENT_NO_TRACEBACK` | `1`, `true`, `yes` (case-insensitive) | Disable all traceback modifications. Must be set before import. |
| `QUENT_TRACEBACK_VALUES` | `0`, `false`, `no` (case-insensitive) | Suppress argument values in visualizations (show type placeholders instead). |

---

## Debug Logging

The `'quent'` logger emits debug-level messages at key execution points:

- Chain start/completion
- Step completion with result
- Async transition
- Step failure

Gated by `_log.isEnabledFor(DEBUG)` -- zero overhead when not at DEBUG level. Respects `QUENT_TRACEBACK_VALUES=0` for value suppression.

---

## Version

```python
from quent import __version__
```

Package version string via `importlib.metadata.version('quent')`.

---

## Complete Signatures

```python
# Constructor
Chain(v=<no value>, /, *args, **kwargs)

# Pipeline building
chain.then(v, /, *args, **kwargs) -> Chain
chain.do(fn, /, *args, **kwargs) -> Chain
chain.foreach(fn, /, *, concurrency=None, executor=None) -> Chain
chain.foreach_do(fn, /, *, concurrency=None, executor=None) -> Chain
chain.gather(*fns, concurrency=-1, executor=None) -> Chain
chain.with_(fn, /, *args, **kwargs) -> Chain
chain.with_do(fn, /, *args, **kwargs) -> Chain
chain.if_(predicate=None, /, *args, **kwargs) -> Chain  # follow with .then()/.do()
chain.else_(v, /, *args, **kwargs) -> Chain
chain.else_do(fn, /, *args, **kwargs) -> Chain
chain.name(label, /) -> Chain

# Error handling
chain.except_(fn, /, *args, exceptions=None, reraise=False, **kwargs) -> Chain
chain.finally_(fn, /, *args, **kwargs) -> Chain

# Execution
chain.run(v=<no value>, /, *args, **kwargs) -> Any
chain(v=<no value>, /, *args, **kwargs) -> Any  # alias for run()

# Iteration
chain.iterate(fn=None) -> ChainIterator
chain.iterate_do(fn=None) -> ChainIterator
chain.flat_iterate(fn=None, *, flush=None) -> ChainIterator
chain.flat_iterate_do(fn=None, *, flush=None) -> ChainIterator

# Context API (instance -- pipeline steps)
chain.set(key: str) -> Chain
chain.set(key: str, value: Any) -> Chain
chain.get(key: str) -> Chain
chain.get(key: str, default: Any) -> Chain

# Context API (class -- immediate)
Chain.set(key: str, value: Any) -> None
Chain.get(key: str, default: Any = <missing>) -> Any

# Reuse
chain.clone() -> Chain
chain.decorator() -> Callable
Chain.from_steps(*steps) -> Chain

# Control flow (class methods)
Chain.return_(v=<no value>, /, *args, **kwargs) -> NoReturn
Chain.break_(v=<no value>, /, *args, **kwargs) -> NoReturn

# Instrumentation (class attribute)
Chain.on_step: Callable[[Chain, str, Any, Any, int], None] | None = None
```
