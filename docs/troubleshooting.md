---
title: "Troubleshooting — Common Errors and How to Fix Them"
description: "Solutions for common quent errors: forgetting return before Chain.return_(), break_() outside iteration, else_() without if_(), non-callable in do(), exception handling pitfalls, duplicate handler registration, pickling, concurrency validation, nesting depth, and task limits."
tags:
  - troubleshooting
  - errors
  - debugging
  - faq
search:
  boost: 6
---

# Troubleshooting

This page covers the most common mistakes when using quent, with exact error messages, explanations, and fixes.

---

## 1. Forgetting `return` Before `Chain.return_()`

### Symptom

Your linter reports unreachable code after `Chain.return_()`, or readers are confused about the control flow:

```python
def process(x):
  if x < 0:
    Chain.return_('negative')  # <-- missing `return`
    print('this looks unreachable')  # linter warning
  return x * 2
```

### What happens

`Chain.return_()` works by raising an internal `_Return` exception. Because it raises immediately, the call itself will exit the function -- so the chain **does** receive the early-return signal even without `return`. However, omitting `return` is misleading: it looks like `Chain.return_()` is a no-op side-effect call, and linters will flag subsequent code as unreachable.

### Fix

Always write `return Chain.return_(...)`:

```python
def process(x):
  if x < 0:
    return Chain.return_('negative')
  return x * 2
```

!!! tip
    Using `return` makes the control flow explicit -- both for human readers and for static analysis tools. The value returned by `Chain.return_()` is never actually used (it raises before returning), but the `return` statement communicates intent.

---

## 2. Using `Chain.break_()` Outside Iteration

### Error

```
quent.QuentException: Chain.break_() cannot be used outside of a foreach/foreach_do iteration.
```

### Cause

`Chain.break_()` is only valid inside callbacks passed to `.foreach()` or `.foreach_do()`. If you call it from a `.then()` or `.do()` step, the `_Break` signal has no iteration loop to catch it, so it escapes to `run()` and is wrapped in `QuentException`.

```python
from quent import Chain

# WRONG -- break_() in a .then() step
result = (
  Chain(5)
  .then(lambda x: Chain.break_(x) if x > 3 else x)
  .run()
)
# raises QuentException
```

### Fix

Use `Chain.return_()` to exit the entire chain early. Use `Chain.break_()` only inside iteration callbacks:

```python
from quent import Chain

# Use return_() to exit the chain early
result = (
  Chain(5)
  .then(lambda x: Chain.return_(x) if x > 3 else x * 2)
  .run()
)
# result == 5

# Use break_() inside .foreach()
result = (
  Chain([1, 2, 3, 4, 5])
  .foreach(lambda x: Chain.break_() if x > 3 else x * 2)
  .run()
)
# result == [2, 4, 6]
```

!!! warning
    `Chain.break_()` is also invalid inside `.except_()` and `.finally_()` handlers and inside `.gather()` operations.

---

## 3. Calling `.else_()` or `.else_do()` Without Preceding `.if_()`

### Error

If the chain has no steps at all:

```
quent.QuentException: else_() requires a preceding if_() — the chain has no steps yet.
Usage: chain.if_(predicate).then(handler).else_(alternative)
```

If the last step is not an `.if_()` or `.then()`/`.do()` following `.if_()`:

```
quent.QuentException: else_() must follow immediately after if_() with no operations
in between. The last operation in this chain is not if_().
Usage: chain.if_(predicate).then(handler).else_(alternative)
```

### Cause

`.else_()` and `.else_do()` must be chained **immediately** after `.if_().then()` or `.if_().do()`. Any intervening operation breaks the association:

```python
from quent import Chain

# WRONG -- .do(print) between .if_().then() and .else_()
result = (
  Chain(value)
  .if_(lambda x: x > 0).then(process_positive)
  .do(print)
  .else_(process_negative)  # raises QuentException
  .run()
)
```

### Fix

Ensure `.else_()` immediately follows `.if_()`:

```python
from quent import Chain

# Correct
result = (
  Chain(value)
  .if_(lambda x: x > 0).then(process_positive)
  .else_(process_negative)
  .do(print)  # side-effects go after the if/else block
  .run()
)
```

!!! tip
    Think of `.if_()` and `.else_()` as a single unit. Build the complete conditional block first, then add subsequent pipeline steps.

---

## 4. Passing a Non-Callable to `.do()`

### Error

```
TypeError: do() requires a callable, got <type_name>
```

### Cause

`.do()` enforces that its argument is callable. A non-callable `.do(42)` would silently do nothing (the value is discarded since `.do()` is a side-effect step), so this is caught at build time.

```python
from quent import Chain

# WRONG
result = Chain(42).do('not a function').run()  # raises TypeError
```

### Fix

Use `.then()` for literal values, `.do()` only for callables:

```python
from quent import Chain

# .then() for literal values
result = Chain(42).then('replacement').run()  # result == 'replacement'

# .do() for side-effects
result = Chain(42).do(print).run()  # prints 42, result == 42
```

!!! info "`.then()` vs `.do()` -- key difference"
    - `.then(v)` accepts **any** value. Non-callables replace the current pipeline value.
    - `.do(fn)` requires a **callable**. Its return value is always discarded.

---

## 5. Expecting `except_()` to Catch `KeyboardInterrupt`

### Symptom

You press Ctrl+C during chain execution and expect your handler to catch it, but it propagates instead.

### Cause

`.except_()` catches `Exception` by default, **not** `BaseException`. `KeyboardInterrupt` and `SystemExit` are `BaseException` subclasses that are not `Exception` subclasses.

### Fix

If you genuinely need to catch `BaseException` subclasses, specify them explicitly:

```python
from quent import Chain

result = (
  Chain(long_running_task)
  .except_(handle_error, exceptions=(Exception, KeyboardInterrupt))
  .run()
)
```

!!! danger "Think twice"
    Catching `KeyboardInterrupt` or `SystemExit` prevents users from terminating your program. quent emits a warning when you do this. In almost all cases, use `.finally_()` for cleanup instead -- finally handlers run even when these signals propagate.

```python
from quent import Chain

# Preferred: finally_() for cleanup, let interrupts propagate
result = (
  Chain(long_running_task)
  .except_(handle_recoverable_error)
  .finally_(lambda _: cleanup())
  .run()
)
```

---

## 6. Duplicate Handler Registration

### Error

```
quent.QuentException: You can only register one 'except' callback.
```

```
quent.QuentException: You can only register one 'finally' callback.
```

### Cause

Each chain supports at most one `except_()` and one `finally_()`. This is enforced at registration time.

### Fix

Consolidate into a single handler, or use nested chains for per-section error handling:

```python
from quent import Chain

# Consolidate
def combined_handler(exc):
  if isinstance(exc, ConnectionError):
    return handle_connection_error(exc)
  if isinstance(exc, ValueError):
    return handle_validation_error(exc)
  raise exc

chain = Chain(data).then(process).except_(combined_handler)

# Or use nested chains
fetch_chain = Chain().then(fetch_data).except_(handle_fetch_error)
process_chain = Chain().then(validate).then(transform).except_(handle_process_error)

pipeline = (
  Chain(url)
  .then(fetch_chain)
  .then(process_chain)
  .finally_(lambda _: cleanup())
  .run()
)
```

---

## 7. `TypeError` When Pickling a Chain

### Error

```
TypeError: Chain objects cannot be pickled. Chains contain arbitrary callables
whose execution during unpickling could lead to arbitrary code execution (CWE-502).
```

### Cause

Chain objects deliberately block pickling as a security measure. This affects `pickle.dumps`, `multiprocessing.Pool`, Celery task arguments, and pickle-based caches.

### Fix

Define chains at module level and reference by name:

```python
from quent import Chain

pipeline = Chain().then(validate).then(transform).then(save)

# Celery: reference the module-level chain
@celery_app.task
def process_order(order_id):
  return pipeline.run(order_id)
```

---

## 8. Concurrency Parameter Validation Errors

### Errors

```
TypeError: map() concurrency must be a positive integer, got bool
```

```
ValueError: map() concurrency must be >= 1, got 0
```

### Cause

The `concurrency` parameter has strict validation:

- Must be a positive integer, `-1` (unbounded), or `None`. Booleans are rejected.
- Must be `>= 1` or `-1`.

### Fix

```python
# WRONG
Chain(urls).foreach(fetch, concurrency=True)   # bool rejected
Chain(urls).foreach(fetch, concurrency=0)      # must be >= 1

# RIGHT
Chain(urls).foreach(fetch, concurrency=4)
Chain(urls).foreach(fetch, concurrency=-1)     # unbounded
```

---

## 9. Chain Nesting Depth Exceeded

### Error

```
quent.QuentException: Maximum chain nesting depth (50) exceeded.
```

### Cause

Nested chain execution is capped at depth 50 to prevent stack overflow from circular or deeply recursive compositions.

### Fix

Flatten your pipeline instead of nesting deeply:

```python
from quent import Chain

# Instead of deep nesting, compose steps in a single chain
chain = Chain(42)
for step in processing_steps:
  chain = chain.then(step)
result = chain.run()
```

---

## 10. Enhanced Traceback Opt-Out

### Symptom

quent's traceback modifications interfere with your debugger, CI system, or custom exception handler.

### Fix: Environment Variable

Set before importing quent:

```bash
export QUENT_NO_TRACEBACK=1
```

This disables all traceback modifications: visualization injection, frame cleaning, and hook patching.

### Suppressing Values in Tracebacks

To keep chain visualizations but hide sensitive data:

```bash
export QUENT_TRACEBACK_VALUES=0
```

This replaces argument values with type-name placeholders (e.g., `<str>` instead of `'secret_api_key'`).

---

## 11. `ExceptionGroup` from Concurrent Operations

### Symptom

Multiple concurrent workers (in `gather()`, or `map()`/`foreach_do()` with `concurrency`) fail simultaneously, producing an `ExceptionGroup` instead of a single exception.

### Cause

When multiple concurrent workers fail, all exceptions are wrapped in an `ExceptionGroup`. A single failure propagates directly (no wrapping).

### Fix

Handle `ExceptionGroup` using `except*` (Python 3.11+) or the `.subgroup()` / `.split()` methods:

```python
from quent import Chain

try:
  result = (
    Chain(urls)
    .foreach(fetch, concurrency=4)
    .run()
  )
except* ConnectionError as eg:
  print(f'{len(eg.exceptions)} connection errors')
except* ValueError as eg:
  print(f'{len(eg.exceptions)} value errors')
```

Or catch `ExceptionGroup` in the chain's `except_()`:

```python
result = (
  Chain(urls)
  .foreach(fetch, concurrency=4)
  .except_(lambda ei: handle_group(ei.exc) if isinstance(ei.exc, ExceptionGroup) else handle_single(ei.exc))
  .run()
)
```

---

## 12. Async Transition from Sync Chain Handlers

### Symptom

Your sync chain's `run()` returns a coroutine instead of a plain value.

### Cause

A sync chain's `finally_()` or `except_()` handler returned a coroutine. The engine performs an async transition: `run()` returns a coroutine instead of a plain value.

### Fix

Either:

1. Use `await chain.run()` in an async context so the handler is properly awaited.
2. Make the handler synchronous.

```python
# Option 1: await in async context
result = await chain.run()

# Option 2: sync handler
chain.finally_(lambda rv: sync_cleanup(rv))
```

---

## 13. Control Flow Signals in Handlers

### Error

```
quent.QuentException: Chain.return_() cannot be used inside except_() handler.
```

```
quent.QuentException: Chain.break_() cannot be used inside finally_() handler.
```

### Cause

`Chain.return_()` and `Chain.break_()` are not allowed inside `except_()` or `finally_()` handlers. Control flow signals must be used in the main pipeline.

### Fix

Move the control flow logic into the main pipeline using nested chains:

```python
from quent import Chain

# WRONG -- return_() in except handler
chain = Chain(data).then(process).except_(
  lambda exc: Chain.return_('fallback')  # raises QuentException
)

# RIGHT -- use the handler's return value
chain = Chain(data).then(process).except_(
  lambda exc: 'fallback'  # handler's return value replaces the result
)
```
