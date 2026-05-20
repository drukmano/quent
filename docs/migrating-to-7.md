---
title: "Migrating to 7.0 — Control-Flow Model Redesign"
description: "Migration guide for quent 7.0.0: Q.return_() is now scoped to the current Q, the new Q.exit_() exits the entire pipeline, and gather/drive_gen/handler carve-outs changed. Before/after examples and a decision table."
tags:
  - migration
  - upgrade
  - control-flow
  - 7.0
---

# Migrating to 7.0

7.0.0 redesigns the control-flow model around Python analogies:

| Signal | Python analogy | Scope |
|---|---|---|
| `Q.return_()` | `return` | exits the **current `Q`** only |
| `Q.break_()` | labeled `break` to nearest loop | propagates out to the nearest enclosing **iteration scope** |
| `Q.exit_()` | `sys.exit()` (new) | propagates through everything; absorbed only at the outermost `run()` |

Pre-7.0.0, `Q.return_()` always propagated to the **outermost** `run()`, even from inside nested pipelines. That is now the job of `Q.exit_()`.

If you never nest `Q` instances and never use `Q.return_()` inside `gather`/`drive_gen`/handler, you are unaffected.

---

## Are you affected?

You are affected if **any** of these are true:

1. You register a nested `Q` (e.g. `outer.then(inner)`) where `inner` calls `Q.return_()` expecting the **outer** pipeline to exit.
2. You call `Q.return_()` inside a `gather()` worker expecting the entire pipeline to exit.
3. You call `Q.return_()` inside a `drive_gen` `fn` expecting the entire pipeline to exit.
4. You rely on `Q.return_()`/`Q.break_()` raising `QuentException` from inside a nested-`Q`'s `except_`/`finally_` handler.
5. You rely on `Q.break_()` raising `QuentException` from inside an `if_()` predicate (it now propagates outward).
6. You rely on `Q.break_()` being trapped inside `with_`/`with_do`/`drive_gen`/nested-`Q` step boundaries (it now propagates through them).

Migration in every case: replace `Q.return_()` with `Q.exit_()` where the intent is "exit the entire pipeline".

---

## 1. `Q.return_()` is now scoped to the current `Q`

=== "Before (6.1.1)"

    ```python
    inner = Q().then(lambda x: Q.return_('STOP') if x < 0 else x).then(lambda x: x * 2)
    outer = Q(-5).then(inner).then(lambda x: x + 100)
    outer.run()
    # 'STOP'  -- return_() propagated to outer run(); the .then(x+100) was skipped
    ```

=== "After (7.0.0)"

    ```python
    outer.run()
    # 'STOP' + 100 = error (str + int)
    # inner returned 'STOP'; outer continued with 'STOP' as inner's result and ran .then(x+100)
    ```

**Migration** — use `Q.exit_()` to keep the old semantics:

```python
inner = Q().then(lambda x: Q.exit_('STOP') if x < 0 else x).then(lambda x: x * 2)
outer = Q(-5).then(inner).then(lambda x: x + 100)
outer.run()
# 'STOP'  -- exit_() propagates through inner's boundary AND outer's; absorbed at outermost run()
```

Or leave `return_()` in place if "return from the inner pipeline only" is what you want.

---

## 2. `Q.return_()` inside `gather()` worker

**Before** — a worker raising `Q.return_()` ended the **entire pipeline** with that value; sibling workers were cancelled / discarded.

**After** — the worker returns from itself; the value becomes that gather position's tuple element. Sibling workers run normally.

```python
Q(5).gather(
  lambda x: x * 2,
  lambda x: Q.return_('alt') if x > 0 else x,
  lambda x: x + 100,
).run()
# 6.1.1: 'alt'
# 7.0.0: (10, 'alt', 105)
```

**Migration** — `Q.exit_()` for the old "abort the pipeline" semantics; the gather is still cancelled, `finally_` handlers still run:

```python
.gather(..., lambda x: Q.exit_('alt') if x > 0 else x, ...)
```

---

## 3. `Q.return_()` inside `drive_gen` `fn`

**Before** — `Q.return_()` exited the pipeline.

**After** — `Q.return_()` returns from `fn`; the value becomes the CV (drive_gen's normal "last fn return → CV"). Subsequent steps run. The generator is closed in cleanup.

**Migration** — `Q.exit_()` for the old behavior.

---

## 4. `Q.return_()` / `Q.break_()` inside a nested `Q`'s `except_`/`finally_` handler

**Before** — a direct call to `Q.return_()` from inside a nested `Q`'s handler always raised `QuentException`.

**After**:

- If the handler is **registered as a step** (the nested `Q` owns the handler), the nested `Q` absorbs the signal locally — the handler's return value is `None` (return_) and the nested `Q` ends with the signal's value. The outer pipeline continues normally.
- If the handler is a **plain callable** passed somewhere else (not registered on a `Q`), `Q.return_()` still raises `QuentException` (the handler trap applies only to direct invocation in a handler position).

**Migration** — if you relied on the `QuentException`, switch to a regular `raise` from your handler. If you wanted to short-circuit the pipeline from the handler, use `Q.exit_()`.

---

## 5. `Q.break_()` inside `if_()` predicate

=== "Before (6.1.1)"

    ```python
    Q.break_()  # inside an if_() predicate
    # QuentException("break_() cannot be used inside an if_() predicate")
    ```

=== "After (7.0.0)"

    `Q.break_()` propagates outward through `if_()` toward the nearest enclosing iteration scope. If no enclosing iteration exists, it still wraps as `QuentException` at the outermost `run()` — but the message is the generic *"…outside of a loop or iteration context"* (not the if_-specific one).

**Migration** — no code change needed; semantics are stricter (propagation now works as documented). If you were catching the specific message text, update to the new one.

---

## 6. `Q.break_()` propagates through more boundaries

**Before** — `Q.break_()` was trapped by `if_()`, `with_`/`with_do`, `drive_gen`, and nested-`Q` registered as a step.

**After** — trapped only in `except_`/`finally_` handlers and `gather()` workers. Everywhere else it propagates outward toward the nearest enclosing iteration scope (CM `__exit__` is still called cleanly, generators still close).

!!! warning "Lambda wrapping breaks `Q.break_()` propagation"
    `.then(lambda cv: inner.run(cv))` makes `inner.run()` outermost from `inner`'s perspective — a `Q.break_()` escaping `inner` is wrapped as `QuentException` at the lambda's call and does not reach the outer pipeline's iteration. To preserve `Q.break_()` propagation across nesting, register the inner `Q` directly via `.then(inner)`.

---

## 7. New: `Q.exit_(v=<no value>, /, *args, **kwargs)`

Use when you need the old "exit the entire pipeline from arbitrary depth" semantics.

```python
Q.exit_()                       # pipeline result is None
Q.exit_(42)                     # pipeline result is 42
Q.exit_(fn)                     # fn() lazily, at the outermost run()'s catch frame
Q.exit_(fn, *args, **kwargs)    # fn(*args, **kwargs) lazily
```

Standard Python `try/finally` semantics during propagation: every `finally_()` runs, every CM's `__exit__` runs, every `drive_gen` generator closes, every concurrent task cancels and awaits. Resources release.

If the lazy callable form's `fn` raises a control-flow signal, it is wrapped in `QuentException` (signals inside lazy values are misuse — same rule as `Q.return_()`).

---

## Carve-out summary

The **only** places where the propagation model is overridden. `Q.exit_()` bypasses every entry — it propagates regardless.

| Scope | `Q.return_()` | `Q.break_()` | `Q.exit_()` |
|---|---|---|---|
| `except_` handler | `QuentException` | `QuentException` | Propagates |
| `finally_` handler | `QuentException` | `QuentException` | Propagates |
| `gather()` worker | **Returns from worker** | `QuentException` | Propagates (siblings cancelled) |
| `drive_gen` `fn` | **Returns from `fn`** → CV | Propagates outward | Propagates |
| Escaping outermost `run()` with no enclosing iteration (`break_` only) | n/a | `QuentException` | n/a (absorbed) |

Anywhere else — `if_`/`else_*` predicates and branches, `with_`/`with_do` bodies, `then`/`do` callables, `while_` predicate and body, iteration callbacks, nested-`Q` steps — signals propagate without trap.

---

## Bug fixes folded into 7.0.0

Most of these only matter if you hit them; included for completeness. Full list in the [Changelog](changelog.md).

- `except_(reraise=True)` async pipeline + sync/async handler — the original exception is now re-raised correctly (was: handler's exception leaked, `__context__` got overwritten by Python's auto-chaining).
- `finally_` raising during signal propagation now preserves the in-flight signal as `__context__` (was: lost).
- Concurrent `foreach`/`foreach_do` now logs a `RuntimeWarning` when `Q.return_()` wins over co-occurring regulars (matches `gather` behavior).
- `gather` exception triage: `_Break` at position N no longer prevents a `_Return` at position N+M from winning (priority scan now reads the full exception list).
- `Q.exit_()` in concurrent `foreach`/`foreach_do` no longer wrapped as `QuentException("Unknown control flow signal: _Exit")`.
- Sync pipeline + async `finally_` + absorbed `Q.return_()` — `_Return` no longer re-raised after finally.
- Lazy callable raising a signal is now wrapped as `QuentException` per spec (was: leaked raw signal).
- `Q.exit_()` in `iterate*`/`flat_iterate*` terminals now yields the value as one final item and stops, mirroring `Q.return_()` (was: leaked raw `_Exit`).
- Async `drive_gen` `_Return` raised on awaited fn result is now caught (was: leaked raw signal).
- Async `Q.exit_()` outermost absorption now wraps the coroutine for await-time absorption (was: `_Exit` escaped through await).

---

## Quick decision table

| You want… | Use |
|---|---|
| Exit the current `Q` early (Python-`return`-style) | `Q.return_(v)` |
| Stop the nearest iteration (`foreach`/`while_`/etc.) | `Q.break_(v)` |
| Exit the entire top-level pipeline from any depth | `Q.exit_(v)` |
| Old pre-7.0 `return_()` behavior in nested pipelines | `Q.exit_(v)` |
| Old pre-7.0 `return_()` behavior in `gather()` worker | `Q.exit_(v)` |
| Old pre-7.0 `return_()` behavior in `drive_gen` `fn` | `Q.exit_(v)` |

See the [API Reference — Control Flow](reference.md#control-flow-class-methods) for the full signal specification.
