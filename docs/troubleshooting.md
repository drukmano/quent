---
title: "Troubleshooting — Common Errors and How to Fix Them"
description: "Solutions for common quent errors: forgetting return before Q.return_(), Q.return_() vs Q.exit_() scoping, break_() outside iteration, else_() without if_(), non-callable in do(), exception handling pitfalls, duplicate handler registration, copying, concurrency validation, nesting depth, and task limits."
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

## 1. Forgetting `return` Before `Q.return_()`

### Symptom

Your linter reports unreachable code after `Q.return_()`, or readers are confused about the control flow:

```python
def process(x):
  if x < 0:
    Q.return_('negative')  # <-- missing `return`
    print('this looks unreachable')  # linter warning
  return x * 2
```

### What happens

`Q.return_()` works by raising an internal `_Return` exception. Because it raises immediately, the call itself will exit the function -- so the pipeline **does** receive the early-return signal even without `return`. However, omitting `return` is misleading: it looks like `Q.return_()` is a no-op side-effect call, and linters will flag subsequent code as unreachable.

### Fix

Always write `return Q.return_(...)`:

```python
def process(x):
  if x < 0:
    return Q.return_('negative')
  return x * 2
```

!!! tip
    Using `return` makes the control flow explicit -- both for human readers and for static analysis tools. The value returned by `Q.return_()` is never actually used (it raises before returning), but the `return` statement communicates intent.

---

## 2. Using `Q.break_()` Outside Iteration

### Error

```
quent.QuentException: Q.break_() cannot be used outside of a loop or iteration context (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do, while_).
```

### Cause

`Q.break_()` propagates outward through `Q` boundaries, `if_()` predicates and branches, `with_`/`with_do` bodies, `drive_gen`, and other non-iteration positions, looking for the nearest **iteration scope** (`foreach`/`foreach_do`/`iterate`/`iterate_do`/`flat_iterate`/`flat_iterate_do`/`while_`). If none catches it, it surfaces as `QuentException` at the outermost `run()`.

```python
from quent import Q

# WRONG -- break_() in a .then() step (no iteration scope)
result = (
  Q(5)
  .then(lambda x: Q.break_(x) if x > 3 else x)
  .run()
)
# raises QuentException
```

### Fix

Pick the signal that matches your intent:

| You want | Use |
|---|---|
| Exit the current `Q` early (Python-`return`-style) | `Q.return_(v)` |
| Stop the nearest iteration / loop | `Q.break_(v)` -- but only from inside one |
| Exit the entire top-level pipeline from any depth | `Q.exit_(v)` |

```python
from quent import Q

# Use return_() to exit the current Q early
result = Q(5).then(lambda x: Q.return_(x) if x > 3 else x * 2).run()
# result == 5

# Use exit_() to exit the entire pipeline regardless of nesting
result = Q(5).then(lambda x: Q.exit_(x) if x > 3 else x * 2).run()
# result == 5

# Use break_() inside an iteration scope
result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_() if x > 3 else x * 2).run()
# result == [2, 4, 6]
```

!!! warning
    `Q.break_()` is also invalid inside `.except_()`/`.finally_()` handlers and inside `.gather()` workers. `Q.return_()` is also invalid in handlers (but **allowed** in `gather()`/`drive_gen` — see §13 below). `Q.exit_()` is always allowed.

!!! note "Changed in 7.0.0"
    The wording of this error changed; `Q.break_()` no longer raises a specific message from inside `if_()` predicates, `with_` bodies, `drive_gen`, or nested-`Q` steps -- it now propagates through them looking for an iteration scope.

---

## 3. Calling `.else_()` or `.else_do()` Without Preceding `.if_()`

### Error

If the pipeline has no steps at all:

```
quent.QuentException: else_() requires a preceding if_() — the pipeline has no steps yet.
Usage: q.if_(predicate).then(handler).else_(alternative)
```

If the last step is not an `.if_()` or `.then()`/`.do()` following `.if_()`:

```
quent.QuentException: else_() must follow immediately after if_() with no operations
in between. The last operation in this pipeline is not if_().
Usage: q.if_(predicate).then(handler).else_(alternative)
```

### Cause

`.else_()` and `.else_do()` must be chained **immediately** after `.if_().then()` or `.if_().do()`. Any intervening operation breaks the association:

```python
from quent import Q

# WRONG -- .do(print) between .if_().then() and .else_()
result = (
  Q(value)
  .if_(lambda x: x > 0).then(process_positive)
  .do(print)
  .else_(process_negative)  # raises QuentException
  .run()
)
```

### Fix

Ensure `.else_()` immediately follows `.if_()`:

```python
from quent import Q

# Correct
result = (
  Q(value)
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
from quent import Q

# WRONG
result = Q(42).do('not a function').run()  # raises TypeError
```

### Fix

Use `.then()` for literal values, `.do()` only for callables:

```python
from quent import Q

# .then() for literal values
result = Q(42).then('replacement').run()  # result == 'replacement'

# .do() for side-effects
result = Q(42).do(print).run()  # prints 42, result == 42
```

!!! info "`.then()` vs `.do()` -- key difference"
    - `.then(v)` accepts **any** value. Non-callables replace the current pipeline value.
    - `.do(fn)` requires a **callable**. Its return value is always discarded.

---

## 5. Expecting `except_()` to Catch `KeyboardInterrupt`

### Symptom

You press Ctrl+C during pipeline execution and expect your handler to catch it, but it propagates instead.

### Cause

`.except_()` catches `Exception` by default, **not** `BaseException`. `KeyboardInterrupt` and `SystemExit` are `BaseException` subclasses that are not `Exception` subclasses.

### Fix

If you genuinely need to catch `BaseException` subclasses, specify them explicitly:

```python
from quent import Q

result = (
  Q(long_running_task)
  .except_(handle_error, exceptions=(Exception, KeyboardInterrupt))
  .run()
)
```

!!! danger "Think twice"
    Catching `KeyboardInterrupt` or `SystemExit` prevents users from terminating your program. quent emits a warning when you do this. In almost all cases, use `.finally_()` for cleanup instead -- finally handlers run even when these signals propagate.

```python
from quent import Q

# Preferred: finally_() for cleanup, let interrupts propagate
result = (
  Q(long_running_task)
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

Each pipeline supports at most one `except_()` and one `finally_()`. This is enforced at registration time.

### Fix

Consolidate into a single handler, or use nested pipelines for per-section error handling:

```python
from quent import Q

# Consolidate
def combined_handler(exc):
  if isinstance(exc, ConnectionError):
    return handle_connection_error(exc)
  if isinstance(exc, ValueError):
    return handle_validation_error(exc)
  raise exc

q = Q(data).then(process).except_(combined_handler)

# Or use nested pipelines
fetch_q = Q().then(fetch_data).except_(handle_fetch_error)
process_q = Q().then(validate).then(transform).except_(handle_process_error)

pipeline = (
  Q(url)
  .then(fetch_q)
  .then(process_q)
  .finally_(lambda _: cleanup())
  .run()
)
```

---

## 7. `TypeError` When Copying a `Q` Instance

### Error

```
TypeError: Q objects cannot be copied with copy.copy()/copy.deepcopy(). Use Q.clone() instead.
```

### Cause

`copy.copy()` and `copy.deepcopy()` are blocked on `Q` instances. A shallow copy would produce a broken object with shared linked-list node references, leading to subtle corruption. A deep copy is semantically undefined for objects containing arbitrary callables.

### Fix

Use `.clone()` to produce a correct independent copy:

```python
from quent import Q

base = Q().then(validate).then(transform)
branch_a = base.clone().then(to_json)    # independent copy
branch_b = base.clone().then(to_record)  # independent copy
```

---

## 8. Concurrency Parameter Validation Errors

### Errors

```
TypeError: foreach() concurrency must be a positive integer or -1 (unbounded), got bool
```

```
ValueError: foreach() concurrency must be -1 (unbounded) or a positive integer, got 0
```

### Cause

The `concurrency` parameter has strict validation:

- Must be a positive integer, `-1` (unbounded), or `None`. Booleans are rejected.
- Must be `>= 1` or `-1`.

### Fix

```python
# WRONG
Q(urls).foreach(fetch, concurrency=True)   # bool rejected
Q(urls).foreach(fetch, concurrency=0)      # must be >= 1

# RIGHT
Q(urls).foreach(fetch, concurrency=4)
Q(urls).foreach(fetch, concurrency=-1)     # unbounded
```

---

## 9. Pipeline Nesting Depth Exceeded (Visualization Limit)

### Error

```
Q(...<truncated at depth 50>...)
```

### Cause

Nested pipeline *visualization* is truncated at depth 50. This is a rendering limit only — there is no execution depth limit. Deeply nested pipelines execute without restriction; only the traceback/repr visualization is truncated.

### Fix

If the truncated visualization is unhelpful, flatten your pipeline instead of nesting deeply:

```python
from quent import Q

# Instead of deep nesting, compose steps in a single pipeline
q = Q(42)
for step in processing_steps:
  q = q.then(step)
result = q.run()
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

To keep pipeline visualizations but hide sensitive data:

```bash
export QUENT_TRACEBACK_VALUES=0
```

This replaces argument values with type-name placeholders (e.g., `<str>` instead of `'secret_api_key'`).

---

## 11. `ExceptionGroup` from Concurrent Operations

### Symptom

Multiple concurrent workers (in `gather()`, or `foreach()`/`foreach_do()` with `concurrency`) fail simultaneously, producing an `ExceptionGroup` instead of a single exception.

### Cause

When multiple concurrent workers fail, all exceptions are wrapped in an `ExceptionGroup`. A single failure propagates directly (no wrapping).

### Fix

Handle `ExceptionGroup` using `except*` (Python 3.11+) or the `.subgroup()` / `.split()` methods:

```python
from quent import Q

try:
  result = (
    Q(urls)
    .foreach(fetch, concurrency=4)
    .run()
  )
except* ConnectionError as eg:
  print(f'{len(eg.exceptions)} connection errors')
except* ValueError as eg:
  print(f'{len(eg.exceptions)} value errors')
```

Or catch `ExceptionGroup` in the pipeline's `except_()`

```python
result = (
  Q(urls)
  .foreach(fetch, concurrency=4)
  .except_(lambda ei: handle_group(ei.exc) if isinstance(ei.exc, ExceptionGroup) else handle_single(ei.exc))
  .run()
)
```

---

## 12. Async Transition from Sync Pipeline Handlers

### Symptom

Your sync pipeline's `run()` returns a coroutine instead of a plain value.

### Cause

A sync pipeline's `finally_()` or `except_()` handler returned a coroutine. The engine performs an async transition: `run()` returns a coroutine instead of a plain value.

### Fix

Either:

1. Use `await q.run()` in an async context so the handler is properly awaited.
2. Make the handler synchronous.

```python
# Option 1: await in async context
result = await q.run()

# Option 2: sync handler
q.finally_(lambda rv: sync_cleanup(rv))
```

---

## 13. Control Flow Signals in Handlers

### Error

```
quent.QuentException: Using _Return inside except handlers is not allowed.
```

```
quent.QuentException: Using _Break inside finally handlers is not allowed.
```

### Cause

`Q.return_()` and `Q.break_()` are not allowed when raised **directly** inside `except_()` or `finally_()` handlers -- they would skip handler invariants and the "always runs" guarantee. `Q.exit_()` is **allowed** in both (it propagates unconditionally).

### Fix

Return the value directly, or use `Q.exit_()` if your intent is to terminate the entire pipeline from inside the handler:

```python
from quent import Q

# WRONG -- return_() in except handler
q = Q(data).then(process).except_(
  lambda exc: Q.return_('fallback')  # raises QuentException
)

# RIGHT -- use the handler's return value
q = Q(data).then(process).except_(
  lambda exc: 'fallback'  # handler's return value replaces the result
)

# RIGHT -- exit_() bypasses the trap when you actually want to abort
q = Q(data).then(process).except_(
  lambda exc: Q.exit_('aborted')  # value surfaces at outermost run()
)
```

### Nested-`Q` handler: `Q.return_()` is absorbed locally

When the handler is registered on a **nested** `Q`, that nested `Q` absorbs its own `Q.return_()` -- the handler "returns" with the signal's value and the outer pipeline continues:

```python
inner = Q().then(may_fail).except_(lambda ei: Q.return_('fallback'))
outer = Q(data).then(inner).then(next_step)
outer.run()
# inner's handler absorbs the signal; 'fallback' flows to next_step as inner's result.
# (Pre-7.0 this raised QuentException; the handler trap now applies only to direct invocation.)
```

---

## 14. `Q.return_()` Doesn't Exit the Entire Pipeline From a Nested `Q`

### Symptom

You have a nested `Q` (`outer.then(inner)`) and call `Q.return_()` from inside `inner`. You expected the whole pipeline to short-circuit but the outer pipeline keeps running with the returned value as `inner`'s result.

### Cause

Since 7.0.0, `Q.return_()` returns from the **current `Q` only** -- Python-`return`-style. Each `Q` boundary absorbs its own signal. Pre-7.0, `Q.return_()` propagated to the outermost `run()`.

### Fix

Use `Q.exit_()` for the old "exit the entire pipeline from arbitrary depth" semantics:

```python
inner = Q().then(lambda x: Q.exit_('STOP') if x < 0 else x).then(lambda x: x * 2)
outer = Q(-5).then(inner).then(lambda x: x + 100)
outer.run()
# 'STOP' -- exit_() bypasses inner's boundary AND outer's; absorbed at outermost run().
```

Same applies if you used `Q.return_()` inside a `gather()` worker or `drive_gen` `fn` expecting the whole pipeline to abort -- those positions now return from the worker / fn (per §7.4 carve-out). Switch to `Q.exit_()`.
