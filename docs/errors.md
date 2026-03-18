---
title: "Error Reference — QuentException Messages"
description: "Searchable index of every QuentException and TypeError raised by quent. Find any error by pasting its message text, then learn what triggers it and how to fix it."
tags:
  - errors
  - exceptions
  - troubleshooting
  - reference
search:
  boost: 9
---

# Error Reference

This page is a searchable index of every error quent raises. Paste the error message text into your browser's find function (`Ctrl+F` / `Cmd+F`) to locate it.

Errors are grouped by category. The [Troubleshooting](troubleshooting.md) page covers the most common mistakes with worked examples.

---

## Build-Time Validation

These errors are raised when constructing or configuring a chain — before any execution occurs. They indicate a misuse of the builder API.

---

### `<method>() requires a callable, got <type>`

**Exception type:** `TypeError`
**Source:** `quent/_validation.py` (line 23)

```
do() requires a callable, got int
except_() requires a callable, got str
finally_() requires a callable, got NoneType
```

**Trigger:** A non-callable was passed to a method that requires a callable. Affected methods: `.do()`, `.except_()`, `.finally_()`, `.foreach()`, `.foreach_do()`, `.gather()`, and others that enforce callable arguments.

**Fix:** Pass a callable (function, lambda, or any object with `__call__`). Use `.then()` instead of `.do()` when you want to pass a literal value — `.then()` accepts non-callables as replacement values.

---

### `Chain() keyword arguments require a root value as the first positional argument`

**Exception type:** `TypeError`
**Source:** `quent/_chain.py` (line 240)

**Trigger:** `kwargs` or `args` were passed to `Chain(...)` without a root value. Example: `Chain(key=val)` with no positional `v`.

**Fix:** Provide the callable or value as the first positional argument before any args/kwargs:

```python
Chain(my_function, key=val)  # correct
```

---

### `run() keyword arguments require a root value as the first positional argument`

**Exception type:** `TypeError`
**Source:** `quent/_chain.py` (line 955)

**Trigger:** `.run()` was called with `kwargs` but no root value argument.

**Fix:** Pass the root value as the first positional argument to `.run()`:

```python
chain.run(my_callable, key=val)  # correct
```

---

### `run() received arguments but v is not callable (got <type>)`

**Exception type:** `TypeError`
**Source:** `quent/_chain.py` (line 957–958)

**Trigger:** `.run(v, *args, **kwargs)` was called with extra args but `v` is not callable.

**Fix:** Only pass args/kwargs to `.run()` when `v` is callable. Use `.then(v)` if you just want to pass a literal root value.

---

### `gather() concurrency must be -1 or a positive integer, not None`

**Exception type:** `TypeError`
**Source:** `quent/_chain.py` (line 557)

**Trigger:** `.gather()` was called with `concurrency=None`. Unlike `.foreach()`, `.gather()` requires an explicit concurrency value — `None` is not accepted.

**Fix:** Use `concurrency=-1` for unbounded concurrent execution (the default), or a positive integer to cap parallelism:

```python
chain.gather(fn_a, fn_b, concurrency=-1)  # unbounded
chain.gather(fn_a, fn_b, concurrency=4)   # max 4 workers
```

---

### `<method>() concurrency must be a positive integer or -1 (unbounded), got <type>`

**Exception type:** `TypeError`
**Source:** `quent/_validation.py` (line 33)

```
foreach() concurrency must be a positive integer or -1 (unbounded), got bool
gather() concurrency must be a positive integer or -1 (unbounded), got float
```

**Trigger:** `concurrency` was passed as a non-integer type. `bool` is explicitly rejected even though `bool` is a subclass of `int` in Python.

**Fix:** Use a plain `int` literal: `concurrency=4` or `concurrency=-1`.

---

### `<method>() concurrency must be -1 (unbounded) or a positive integer, got <value>`

**Exception type:** `ValueError`
**Source:** `quent/_validation.py` (line 37)

```
foreach() concurrency must be -1 (unbounded) or a positive integer, got 0
gather() concurrency must be -1 (unbounded) or a positive integer, got -2
```

**Trigger:** `concurrency` is an integer but is zero or a negative integer other than `-1`.

**Fix:** Valid values: `-1` (unbounded) or any integer `>= 1`.

---

### `<method>() executor must be a concurrent.futures.Executor instance, got <type>`

**Exception type:** `TypeError`
**Source:** `quent/_validation.py` (line 44)

**Trigger:** The `executor` parameter was passed a value that is not a `concurrent.futures.Executor` instance.

**Fix:** Pass a `ThreadPoolExecutor` or other `Executor` subclass, or omit `executor` to use the default:

```python
from concurrent.futures import ThreadPoolExecutor
chain.foreach(fn, executor=ThreadPoolExecutor(max_workers=4))
```

---

### `<method>() requires at least one exception type when exceptions is provided.`

**Exception type:** `QuentException`
**Source:** `quent/_validation.py` (line 70)

```
except_() requires at least one exception type when exceptions is provided.
```

**Trigger:** An empty iterable was passed as the `exceptions` argument to `.except_()`.

**Fix:** Pass at least one exception type, or omit `exceptions` to catch `Exception` (the default):

```python
chain.except_(handler, exceptions=(ValueError, TypeError))
chain.except_(handler)  # catches Exception by default
```

---

### `<method>() expects exception types, not string '<value>'`

**Exception type:** `TypeError`
**Source:** `quent/_validation.py` (line 60)

```
except_() expects exception types, not string 'ValueError'
```

**Trigger:** A string was passed as an exception type. A common mistake is quoting the exception name.

**Fix:** Pass the class itself, not a string:

```python
chain.except_(handler, exceptions=ValueError)       # correct
chain.except_(handler, exceptions='ValueError')     # wrong — string
```

---

### `<method>() expects exception types (subclasses of BaseException), got <value>`

**Exception type:** `TypeError`
**Source:** `quent/_validation.py` (lines 64, 74, 77)

**Trigger:** A value in the `exceptions` argument is not a class or is not a subclass of `BaseException`.

**Fix:** Only pass actual exception classes:

```python
chain.except_(handler, exceptions=(ValueError, RuntimeError))
```

---

### `You can only register one 'except' callback.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 823)

**Trigger:** `.except_()` was called more than once on the same chain.

**Fix:** Each chain supports exactly one exception handler. Consolidate handlers into one, or use nested chains for per-step error handling. See [Error Handling](guide/error-handling.md) for the nested-chain pattern.

---

### `You can only register one 'finally' callback.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 875)

**Trigger:** `.finally_()` was called more than once on the same chain.

**Fix:** Consolidate into a single cleanup function, or restructure with nested chains.

---

### `if_() called while a previous if_() is still pending — add .then() or .do() first.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 650)

**Trigger:** A second `.if_()` was called before the previous `.if_()` was resolved with `.then()` or `.do()`.

**Fix:** Always complete each `if_/then` or `if_/do` block before starting another `.if_()`:

```python
chain.if_(pred_a).then(val_a).if_(pred_b).then(val_b)  # correct
chain.if_(pred_a).if_(pred_b)  # wrong — second if_() sees pending first
```

---

### `if_() received args/kwargs but no predicate — pass a callable or value as the first argument.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 653)

**Trigger:** `.if_()` was called with positional or keyword arguments but without a predicate (first argument).

**Fix:** The first argument to `.if_()` must be the predicate (callable or value). Args/kwargs are passed to that predicate when it is callable.

---

### `else_() / else_do() called while a previous if_() is still pending — add .then() or .do() first.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 682–685)

```
else_() called while a previous if_() is still pending — add .then() or .do() first.
Usage: chain.if_(pred).then(v).else_(alt)
```

**Trigger:** `.else_()` or `.else_do()` was called with an unconsumed `.if_()` pending (no intervening `.then()` or `.do()`).

**Fix:** Call `.then()` or `.do()` after `.if_()` before calling `.else_()`:

```python
chain.if_(pred).then(val_a).else_(val_b)  # correct
chain.if_(pred).else_(val_b)              # wrong — no .then() after if_()
```

---

### `else_() / else_do() requires a preceding if_() — the chain has no steps yet.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 688–692)

```
else_() requires a preceding if_() — the chain has no steps yet.
Usage: chain.if_(pred).then(v).else_(alt)
```

**Trigger:** `.else_()` or `.else_do()` was called on a chain that has no steps.

**Fix:** Build the full `if_/then/else_` block:

```python
chain.if_(pred).then(val_a).else_(val_b)
```

---

### `else_() / else_do() must follow immediately after if_().then() with no operations in between.`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 694–699)

```
else_() must follow immediately after if_().then() with no operations in between.
The last operation in this chain is not if_().
Usage: chain.if_(pred).then(v).else_(alt)
```

**Trigger:** `.else_()` or `.else_do()` was placed after a step that is not part of an `if_/then` block. Another method call was inserted between `.then()` and `.else_()`.

**Fix:** `.else_()` must directly follow the `if_/then` or `if_/do` block:

```python
# Wrong — .do(log) breaks the if/else association
chain.if_(pred).then(val_a).do(log).else_(val_b)

# Correct — else_() immediately follows then()
chain.if_(pred).then(val_a).else_(val_b).do(log)
```

---

### `else_() has already been registered for this if_() — only one else branch is allowed per if_().`

**Exception type:** `QuentException`
**Source:** `quent/_if_ops.py` (line 115–118)

**Trigger:** `.else_()` or `.else_do()` was called a second time for the same `.if_()` block.

**Fix:** Each `if_` block supports exactly one else branch.

---

### `<method>() called with a pending .if_() that was never consumed by .then() or .do().`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 889–893)

```
run() called with a pending .if_() that was never consumed by .then() or .do().
Usage: chain.if_(pred).then(v).run()
```

**Trigger:** A terminal operation (`.run()`, `.clone()`, `.decorator()`, `.iterate()`, etc.) was called while an `.if_()` was pending — no `.then()` or `.do()` followed the `.if_()`.

**Fix:** Always close an `.if_()` with `.then()` or `.do()` before executing or cloning the chain.

---

### `gather() requires at least one function.`

**Exception type:** `QuentException`
**Source:** `quent/_gather_ops.py` (line 156)

**Trigger:** `.gather()` was called with no function arguments.

**Fix:** Pass at least one callable to `.gather()`:

```python
chain.gather(fn_a)            # valid (single function)
chain.gather(fn_a, fn_b)      # valid
chain.gather()                # wrong — no functions
```

---

### `Arguments were provided but the value is not callable (got <type>)`

**Exception type:** `TypeError`
**Source:** `quent/_link.py` (line 77)

**Trigger:** A `Link` was constructed with `args` or `kwargs` but the stored value is not callable and not a chain. This is an internal invariant violation, typically surfaced when an API method receives explicit args alongside a non-callable value.

**Fix:** Only provide call arguments alongside callable values in pipeline steps.

---

## Runtime Control Flow Errors

These errors are raised during chain execution when control flow signals are used in unsupported contexts.

---

### `Chain.break_() cannot be used outside of a foreach/foreach_do iteration.`

**Exception type:** `QuentException`
**Source:** `quent/_engine.py` (lines 1018–1019, 1185–1186)

**Trigger:** `Chain.break_()` was used inside a `.then()`, `.do()`, or other non-iteration pipeline step. The `_Break` signal escaped the chain's link-walk loop with no iteration context to catch it.

**Fix:** `Chain.break_()` is only valid inside callbacks passed to `.foreach()` or `.foreach_do()`. Use `Chain.return_()` to exit the entire chain early from a non-iteration step:

```python
# Wrong — break_() in a .then() step
chain.then(lambda x: Chain.break_(x) if x > 3 else x)

# Correct — use return_() for early chain exit
chain.then(lambda x: Chain.return_(x) if x > 3 else x * 2)

# Correct — use break_() inside foreach
chain.foreach(lambda x: Chain.break_() if x > 3 else x * 2)
```

---

### `Using _Return inside except handlers is not allowed.`

**Exception type:** `QuentException`
**Source:** `quent/_engine.py` (line 74, via `_signal_in_handler_msg`)

```
Using _Return inside except handlers is not allowed.
Using _Break inside except handlers is not allowed.
```

**Trigger:** `Chain.return_()` or `Chain.break_()` was called inside an `.except_()` handler. Control flow signals cannot be raised inside error or cleanup handlers.

**Fix:** Return the desired value directly from the handler instead of using `Chain.return_()`:

```python
# Wrong
chain.except_(lambda ei: Chain.return_('fallback'))

# Correct — handler's return value replaces the chain result
chain.except_(lambda ei: 'fallback')
```

---

### `Using _Return inside finally handlers is not allowed.`

**Exception type:** `QuentException`
**Source:** `quent/_engine.py` (lines 155, 186, via `_signal_in_handler_msg`)

```
Using _Return inside finally handlers is not allowed.
Using _Break inside finally handlers is not allowed.
```

**Trigger:** `Chain.return_()` or `Chain.break_()` was called inside a `.finally_()` handler.

**Fix:** Remove control flow signals from `finally_()` handlers. The `finally_()` handler's return value is always discarded — use it only for side effects and cleanup. Control flow must happen in the main pipeline.

---

### `break_() signals are not allowed in gather operations.`

**Exception type:** `QuentException`
**Source:** `quent/_gather_ops.py` (lines 81, 272)

**Trigger:** `Chain.break_()` was raised inside a callable passed to `.gather()`. Breaking out of a concurrent gather is not defined.

**Fix:** Remove `Chain.break_()` from gather operations. Use conditional logic inside the callable to decide what to return, or use `.foreach()` with `break_()` for iterable processing:

```python
# Wrong — break_() in a gather callable
chain.gather(lambda x: Chain.break_() if x is None else x)

# Correct — return a sentinel value instead
chain.gather(lambda x: None if x is None else process(x))
```

---

### `break_() cannot be used inside an if_() predicate.`

**Exception type:** `QuentException`
**Source:** `quent/_if_ops.py` (lines 64, 87)

**Trigger:** `Chain.break_()` was raised inside a callable used as the predicate for `.if_()`.

**Fix:** Do not use `Chain.break_()` inside an `if_()` predicate. Evaluate conditions with regular Python logic or use `Chain.return_()` if early exit from the chain is needed.

---

### `Unknown control flow signal: <signal_type>`

**Exception type:** `QuentException`
**Source:** `quent/_iter_ops.py` (line 94)

**Trigger:** An unrecognized `_ControlFlowSignal` subclass was encountered during iteration. This indicates a bug — only `_Return` and `_Break` are defined control flow signals. If you see this error, it is likely caused by a custom subclass of an internal quent type.

**Fix:** Do not subclass quent's internal control flow types (`_Return`, `_Break`). Use `Chain.return_()` and `Chain.break_()` through the public API only.

---

### `A _Return signal escaped the chain via <method>().`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 883)

```
A _Return signal escaped the chain via run().
A _Break signal escaped the chain via run().
```

**Trigger:** A `_Return` or `_Break` signal propagated past the top-level chain boundary. This can happen if a control flow signal is raised in a context that the engine cannot intercept, such as a thread or task spawned outside of quent's execution.

**Fix:** Use `Chain.return_()` and `Chain.break_()` only inside callables that are executed directly by the chain's pipeline steps. Signals raised in background threads or tasks external to quent cannot be caught.

---

## Runtime Iteration Errors

These errors are raised when using `iterate()`, `iterate_do()`, or similar generator-based APIs.

---

### `Cannot use sync iteration on an async chain; use 'async for' instead`

**Exception type:** `TypeError`
**Source:** `quent/_generator.py` (line 173)

**Trigger:** A synchronous `for` loop was used on a `ChainIterator` that produced an async chain (i.e., the chain returned a coroutine when run). Sync iteration cannot consume coroutines.

**Fix:** Use `async for` instead of `for`:

```python
# Wrong — sync loop on an async chain
for item in chain.iterate():
  ...

# Correct
async for item in chain.iterate():
  ...
```

---

### `iterate() <step> value resolved to a coroutine. Use "async for" with __aiter__ instead of "for" with __iter__.`

**Exception type:** `TypeError`
**Source:** `quent/_generator.py` (line 68)

**Trigger:** During synchronous iteration, a step value (such as the initial chain result, the per-item callback result, or the flush result) resolved to a coroutine. This means the pipeline is async but the caller used sync iteration.

**Fix:** Switch to `async for` over `chain.iterate()`.

---

### `iterate() callback <fn> returned a coroutine. Use "async for" with __aiter__ instead of "for" with __iter__.`

**Exception type:** `TypeError`
**Source:** `quent/_generator.py` (line 206–210)

**Trigger:** The per-item callback passed to `iterate()` returned a coroutine during synchronous iteration.

**Fix:** Use `async for`, or use a synchronous callback.

---

### `iterate(): deferred with_ inner function returned a coroutine. Use 'async for' with __aiter__ instead.`

**Exception type:** `TypeError`
**Source:** `quent/_generator.py` (line 186)

**Trigger:** During synchronous iteration, the inner function of a deferred `with_()` step returned a coroutine.

**Fix:** Switch to `async for` over `chain.iterate()`.

---

### `iterate(): flush callable returned a coroutine. Use 'async for' with __aiter__ instead.`

**Exception type:** `TypeError`
**Source:** `quent/_generator.py` (line 234)

**Trigger:** The flush callable (used in flat iteration modes) returned a coroutine during synchronous iteration.

**Fix:** Switch to `async for`, or use a synchronous flush callable.

---

## Runtime Context Manager Errors

---

### `<type> object does not support the context manager protocol (__enter__/__exit__ or __aenter__/__aexit__). Ensure the pipeline value at this step is a context manager.`

**Exception type:** `TypeError`
**Source:** `quent/_with_ops.py` (line 211–216), `quent/_generator.py` (line 336–338)

**Trigger:** `.with_()` or `.with_do()` was reached in the pipeline but the current value at that step is not a context manager. The object lacks `__enter__`/`__exit__` (sync) and `__aenter__`/`__aexit__` (async).

**Fix:** Ensure the pipeline value at the `.with_()` step is a context manager. If the context manager is produced by a prior step, verify that step returns the right type:

```python
# The value flowing into .with_() must be a context manager
Chain(my_resource).with_(lambda ctx: process(ctx)).run()
# 'my_resource' must implement __enter__/__exit__
```

---

## Internal Invariant Violations

These errors indicate a bug in quent itself, not in user code. If you encounter them, please open an issue at [github.com/drukmano/quent](https://github.com/drukmano/quent).

---

### `_on_except_exceptions must be set when _on_except_link is set`

**Exception type:** `QuentException`
**Source:** `quent/_engine.py` (line 127)

**Trigger:** An internal invariant was violated: the except handler was registered without a corresponding exception type list. This should never happen in normal usage.

---

### `root_link must not have ignore_result=True`

**Exception type:** `QuentException`
**Source:** `quent/_engine.py` (lines 847, 929, 1115)

**Trigger:** An internal invariant was violated during engine initialization. The root link of a chain must always capture its result.

---

### `link-walk loop exited with link still set`

**Exception type:** `QuentException`
**Source:** `quent/_engine.py` (lines 999, 1166)

**Trigger:** The engine's link-walk loop exited unexpectedly with a link remaining. This is an internal assertion failure.

---

### `clone() missing slots: [<slots>]`

**Exception type:** `QuentException`
**Source:** `quent/_chain.py` (line 1161)

**Trigger:** `clone()` produced a new `Chain` object with one or more `__slots__` uninitialized. This is an internal consistency check that fires only in debug mode (`__debug__ = True`, i.e., not when running with `-O`).

---

## Non-`QuentException` Errors

These errors are raised as standard Python exceptions (`TypeError`, `ValueError`) from quent's public API and are listed here for completeness.

---

### `Chain objects cannot be copied with copy.copy()/copy.deepcopy(). Use Chain.clone() instead.`

**Exception type:** `TypeError`
**Source:** `quent/_types.py` (line 251)

**Trigger:** `copy.copy()` or `copy.deepcopy()` was called on a `Chain` instance. Shallow copies produce broken objects with shared linked-list nodes; deep copies are semantically undefined for objects containing arbitrary callables.

**Fix:** Use `.clone()` to produce a correct independent copy:

```python
base = Chain().then(validate).then(transform)
branch_a = base.clone().then(to_json)
branch_b = base.clone().then(to_record)
```

---

## See Also

- **[Troubleshooting](troubleshooting.md)** — Common mistakes with worked examples and fixes.
- **[Error Handling](guide/error-handling.md)** — `except_()`, `finally_()`, `reraise`, and control flow restrictions.
- **[API Reference](reference.md)** — Full method signatures and parameter descriptions.
