# Quent API Plan

Quent is a chain execution engine with transparent sync/async handling. It eliminates code duplication for libraries that need to support both sync and async callers. This document defines the final target API after restructuring.

---

## Identity Test

**"Does this solve a sync/async bifurcation that can't be handled in a `.then()` or outside the chain?"**

Every kept method must pass this test or be essential chain infrastructure (core linking, error handling, reuse, control flow, execution).

---

## Methods to Keep

| Category | Method | Description | Rationale |
|----------|--------|-------------|-----------|
| Core | `then(v, *args, **kwargs)` | Append operation; result becomes current value. | Foundation of chain linking. |
| Core | `do(fn, *args, **kwargs)` | Append side-effect; result discarded, current value preserved. | Side-effects without disrupting the chain. |
| Bifurcation Solver | `sleep(seconds)` | Async-transparent sleep. `asyncio.sleep` in async, `time.sleep` in sync. | `time.sleep` vs `asyncio.sleep` is a direct bifurcation. |
| Bifurcation Solver | `with_(fn, *args, **kwargs)` | Execute fn inside current value as async-transparent context manager. | `with` vs `async with` is a direct bifurcation. |
| Bifurcation Solver | `gather(*fns)` | Execute multiple fns concurrently on current value. | `asyncio.gather` has no sync equivalent; gather provides one. |
| Bifurcation Solver | `foreach(fn)` | Async-transparent iteration over elements. | `for` vs `async for` is a direct bifurcation. |
| Bifurcation Solver | `filter(fn, *args, **kwargs)` | Async-transparent filtering. | Filtering with async predicates requires `async for`; bifurcation. |
| Bifurcation Solver | `iterate(fn=None)` | Yield chain results as sync or async generator. | `yield` vs `async yield` is a direct bifurcation. |
| Bifurcation Solver | `to_thread(fn, *args, **kwargs)` | Run sync fn in thread pool when in async context; call directly in sync. | `asyncio.to_thread` has no sync equivalent; same pattern as `sleep`. |
| Error Handling | `except_(fn, *args, exceptions=None, reraise=True, **kwargs)` | Register exception handler. | Caller can't try/except when they don't know if `run()` returns a value or a Task. |
| Error Handling | `finally_(fn, *args, **kwargs)` | Register cleanup callback (always runs). | Same as `except_`; cleanup must work regardless of sync/async execution path. |
| Control Flow | `Chain.return_(v=Null)` | Early exit from chain with optional value. | Avoids abusing exceptions for early chain exit. |
| Control Flow | `Chain.break_(v=Null)` | Exit foreach loop with optional value. | Avoids abusing exceptions for early loop exit. |
| Reuse | `clone()` | Deep copy for safe concurrent reuse. | Chain infrastructure for safe reuse. |
| Reuse | `freeze()` | Return a frozen (immutable) copy of the chain. | Chain infrastructure for safe sharing. |
| Reuse | `decorator()` | Return the chain as a decorator. | Integration infrastructure; frozen chain that wraps a function. |
| Configuration | `no_async(default=False)` | Skip `iscoro()` checks for known-sync chains. When `True`, the chain assumes all operations are synchronous. | Performance optimization for known-sync paths. |
| Configuration | `config(*, autorun=None, debug=None)` | Batch configuration of execution policy. | Replaces standalone `autorun()` and other config methods. |

### Execution

`run()`, `__call__`, `__or__` (pipe operator) — execute the chain.

---

## Removed

| Method(s) | Rationale |
|-----------|-----------|
| `root`, `root_do` | Value routing, not a sync/async bifurcation. |
| `condition`, `if_`, `else_` | Conditional logic; handle in a lambda or function passed to `.then()`. |
| `raise_`, `if_raise`, `else_raise` | Not a bifurcation; Python's `raise` works the same in sync/async. |
| `while_true` | Loop mechanics, not a sync/async bifurcation. |
| `reduce` | Collection utility; solvable with `.then(functools.reduce(...))`. |
| `eq`, `neq`, `is_`, `is_not`, `in_`, `not_in`, `isinstance_`, `not_` | Comparison operators; a lambda covers these trivially. |
| `or_` | Logical operator; `.then(lambda v: v or default)`. |
| `pipe` | Alias for `then`; adds confusion, zero unique value. |
| `if_not`, `if_not_raise` | Redundant with removed conditionals. |
| `on_success` | Structure success logic as the last `.then()` before `finally_`. |
| `suppress` | `.except_(lambda: None, exceptions=exc, reraise=False)`. |
| `foreach_do`, `with_do`, `iterate_do` | Redundant `_do` variants; use `do` after the operation. |
| `attr`, `attr_fn` | `.then(lambda v: getattr(v, name))`. |
| `autorun()` | Use `config(autorun=True)`. |
| `set_async()` | Replaced by `no_async()`. |
| `safe_run()` | Use `clone().run()`. |
| `compose` | Use `Chain(f1).then(f2)`. |
| `with_context`, `get_context` | Use `contextvars` directly. |

---

## Rejected Additions

| Method | Rationale |
|--------|-----------|
| `retry` | Not a bifurcation; use `tenacity` or similar. |
| `timeout` | Async-only (`asyncio.timeout`); not a true bifurcation since sync has no equivalent. |
| `unpack` | Lambda covers it: `.then(lambda v: fn(*v))`. |
| `race` | Violates transparency: deterministic in sync, non-deterministic in async. |

---

## Removed Classes

| Class | Rationale |
|-------|-----------|
| `ChainAttr` / `CascadeAttr` | `__getattr__` magic is not a sync/async bifurcation; use `.then(lambda v: getattr(v, name))`. |
| `FrozenChain` (as user-facing export) | Kept internally for `freeze()` but not a separate export. |

---

## Classes

| Class | Purpose |
|-------|---------|
| `Chain` | Each operation receives the result of the previous operation. |
| `Cascade` | Each operation receives the root value (fluent interface pattern). |

---

## Exports

`Chain`, `Cascade`, `run`, `QuentException`, `Null`

---

## Summary

- **Before:** ~40+ methods, redundant paths, unclear identity
- **After:** 17 methods, 2 class methods, 2 classes, zero redundancy
- Every method passes the identity test: it solves a sync/async bifurcation or is essential chain infrastructure.
