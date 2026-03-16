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
class Chain(v=Null, /, *args, **kwargs)
```

A sequential pipeline that transparently bridges synchronous and asynchronous operations. Steps are appended with fluent methods (`.then()`, `.do()`, etc.) and the chain is executed with `.run()`.

### Constructor

```python
Chain(v=Null, /, *args, **kwargs)
```

Create a new chain with an optional root value or callable.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | `Any` | Root value or callable. Defaults to `Null` (no root value). |
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
- `ValueError` if `concurrency` is less than 1 or greater than 1024.

```python
Chain([1, 2, 3]).foreach(lambda x: x ** 2).run()
# [1, 4, 9]

# With concurrency: process up to 4 items concurrently
Chain(urls).foreach(fetch, concurrency=4).run()
```

Supports both sync iterables (`__iter__`) and async iterables (`__aiter__`). When both protocols are present, the async protocol is preferred if an event loop is running. Supports mid-iteration async transition -- if `fn` returns an awaitable for any element, the operation transitions to async automatically.

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
- `ValueError` if `concurrency` is less than 1 or greater than 1024.

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

**Dual-protocol objects:** When the current value supports both sync and async context manager protocols and an event loop is running, the async protocol is preferred.

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
| Positive integer | `ThreadPoolExecutor(max_workers=concurrency)` | `asyncio.Semaphore(concurrency)` to limit concurrent tasks |

**Validation:** Must be a positive integer (`>= 1`, `<= 1024`) or `None`. Booleans are rejected (`TypeError`). Out of range raises `ValueError`.

**Sync/async detection:** The first item/function is probed. If it returns an awaitable, async path is used. If not, sync path. Mixed sync/async within a single concurrent operation raises `TypeError`.

**Concurrent iteration materializes eagerly:** The entire input iterable is converted to a list before processing begins. Do not use with infinite or very large iterables.

!!! note "Upper bound"
    The hard upper bound is **1024**. This prevents accidental creation of unbounded thread pools or semaphore limits.

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

**Async finally in sync chains:** When a sync chain's finally handler returns a coroutine, it is scheduled as a fire-and-forget background task with a `RuntimeWarning`. If no event loop is running, a `QuentException` is raised.

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
chain.run(v=Null, /, *args, **kwargs) -> Any
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
chain(v=Null, /, *args, **kwargs) -> Any
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
Chain.return_(v=Null, /, *args, **kwargs) -> NoReturn
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
Chain.break_(v=Null, /, *args, **kwargs) -> NoReturn
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

Returns a string showing the chain structure:

```python
repr(Chain(fetch).then(validate).do(log))
# "Chain(fetch).then(...).do(...)"
```

---

#### Pickling

Chain objects **cannot be pickled**. Attempting `pickle.dumps(chain)` raises `TypeError`:

```
TypeError: Chain objects cannot be pickled. Chains contain arbitrary callables
whose execution during unpickling could lead to arbitrary code execution (CWE-502).
```

This affects `pickle.dumps`, `multiprocessing.Pool`, Celery task arguments, and pickle-based caches.

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

`Null` is an internal singleton sentinel meaning **"no value was provided."** Distinct from `None`, which is a valid pipeline value. It is not part of the public API.

| Property | Detail |
|----------|--------|
| `repr()` | `'<Null>'` |
| Immutable | `copy()` / `deepcopy()` return `self` |
| Unpicklable | `pickle.dumps(Null)` raises `TypeError` |
| Thread-safe | Created at module level |

**Critical distinction:**

```python
Chain(None)  # root value is None -- fn receives fn(None)
Chain()      # no root value -- fn receives fn() with zero args
```

**Effect on calling conventions:**

- Current value is Null + no explicit args: `fn()` (zero arguments)
- Current value is not Null (including `None`) + no explicit args: `fn(current_value)`
- Explicit args provided: `fn(*args, **kwargs)` regardless

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
Chain(v=Null, /, *args, **kwargs)

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
chain.run(v=Null, /, *args, **kwargs) -> Any
chain(v=Null, /, *args, **kwargs) -> Any  # alias for run()

# Iteration
chain.iterate(fn=None) -> ChainIterator
chain.iterate_do(fn=None) -> ChainIterator

# Reuse
chain.clone() -> Chain
chain.decorator() -> Callable

# Control flow (class methods)
Chain.return_(v=Null, /, *args, **kwargs) -> NoReturn
Chain.break_(v=Null, /, *args, **kwargs) -> NoReturn

# Instrumentation (class attribute)
Chain.on_step: Callable[[Chain, str, Any, Any, int], None] | None = None
```
