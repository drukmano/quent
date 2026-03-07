# Quent Comprehensive Audit Report

**Date:** 2026-03-07
**Scope:** Full codebase audit -- core execution engine, operations, traceback system, type stubs, Python 3.13-3.15 compatibility, and async best practices.
**Codebase:** 4 source modules in `quent/`, zero runtime dependencies.
**Audit methodology:** Six independent agents analyzed orthogonal dimensions. Findings deduplicated and synthesized below.

---

## 1. Executive Summary

Six independent audit agents analyzed the quent codebase across core execution, operations, traceback patching, type stubs, forward compatibility (Python 3.13-3.15), and async best practices. The library's core sync/async bridging mechanism is correct and well-designed. The async best practices audit and the Python forward-compatibility audit both returned clean results -- no blocking issues for Python through 3.15, and all async patterns follow established conventions.

However, the audit identified **1 critical issue**, **6 medium-severity issues**, and **6 low-severity issues** that are new since the previous audit. Several previously-reported issues have been fixed; a few remain.

**Issue Counts by Severity (NEW findings only):**

| Severity | Count | Categories |
|----------|-------|------------|
| CRITICAL | 1 | Silent data loss (StopIteration swallowed in async transition path) |
| MEDIUM | 6 | Behavioral asymmetry, leaked coroutines, missing stubs, missing public API, dual-protocol inconsistency, silent visualization failures |
| LOW | 6 | Non-coroutine awaitable edge case, sync finally masking, `_Null` documentation inaccuracy, stale docstrings, stub style inconsistency, gather internal inconsistency |

---

## 2. Status of Previous Audit Findings

### Fixed

| ID | Description | Status | Evidence |
|----|-------------|--------|----------|
| H2 | Fire-and-forget task exceptions silently lost | **FIXED** | `_task_done_callback` at `_core.py:151-162` now warns via `warnings.warn`. |
| H3 | `asyncio.gather` does not cancel siblings on failure | **FIXED** | `_to_async` in `_make_gather` at `_ops.py:417-422` now cancels sibling tasks. |
| M2 | `then()` accepts non-callable with args silently | **FIXED** | `_evaluate_value` at `_core.py:248-249` validates callability when args/kwargs present. |
| M4 | Sync iterator over async chain leaks coroutine | **FIXED** | `_sync_generator` at `_ops.py:161-164` detects and closes unawaited coroutines. |
| M6 | Three-tier sync/async pattern duplicated across `_ops.py` | **FIXED** | `_make_iter_op` at `_ops.py:281` consolidates map/foreach/filter. |
| M9 | `filter()` does not support `break_()` | **FIXED** | Unified `_make_iter_op` handles `_Break` for all iteration modes. |
| L1 | `_Null` singleton TOCTOU race | **FIXED** | Singleton created at module level via `object.__new__(_Null)`, no `__new__` override needed. |
| L10 | Dual-mode context managers always use async path | **PARTIALLY FIXED** | `_make_with` (line 110) now prefers `__enter__` (sync). However, `_make_iter_op` (line 352) still prefers `__aiter__` -- see new finding below. |
| L11 | Ellipsis convention silently drops extra args | **FIXED** | `_validate_ellipsis` function at `_core.py:42-44` now validates. |
| L12 | `_ControlFlowSignal.__init__` comment factually wrong | **FIXED** | Comment at `_core.py:58-59` now accurately describes the behavior. |

### Removed (feature no longer exists)

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| H4 | `freeze()` does not prevent post-freeze mutation | **REMOVED** | `freeze()` and `_FrozenChain` were removed entirely from the codebase. |
| M5 | Hook conflict with other libraries on disable | **N/A** | `enable_traceback_patching()` / `disable_traceback_patching()` were removed; hooks are always-on. |
| R5 | Flatten linked list to tuple in `freeze()` | **N/A** | Freeze removed. |

### Still Present

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| H1 | StopIteration silently swallowed in map/foreach/filter | **PARTIALLY FIXED** | Fixed in the sync path (`_iter_op`, lines 362-365), but **resurfaced** in the `_to_async` async-transition path. Elevated to CRITICAL below. |
| M1 | `else_()` / `if_()` coupling is fragile | **STILL PRESENT** | Same API design. Error message could be improved. |
| M3 | `except_()` can swallow `BaseException` subclasses | **STILL PRESENT** | Documentation concern. Default is safe (`Exception`), but API allows unsafe configurations. |
| M7 | Library is asyncio-only, incompatible with trio/anyio-trio | **STILL PRESENT** | No alternative async runtime support added. |
| M8 | `_run()` / `_run_async()` share duplicated logic | **PARTIALLY ADDRESSED** | Shared helpers extracted (`_init_attempt_links`, `_except_handler_body`, `_finally_handler_body`, `_stamp_exception_source`), but core loop logic still duplicated. |
| R4 | Store callability flag on Link | **NOT DONE** | `callable(v)` still called on every evaluation. |

---

## 3. Critical & High Severity Issues

### C1. StopIteration Silently Swallowed in `_to_async` Async-Transition Path

- **Severity:** CRITICAL
- **File:** `quent/_ops.py:312-329`
- **Description:** The `_to_async` function inside `_make_iter_op` uses a single `try/except` block wrapping BOTH `next(iterator)` (line 319) and `fn(item)` (line 320). The `except StopIteration` on line 323 catches `StopIteration` from either source indiscriminately. This is the exact same bug as the previous H1 finding, which was correctly fixed in the sync path (`_iter_op`, lines 362-365) by separating `next(it)` into its own `try/except StopIteration` block -- but the fix was not applied to `_to_async`.
- **Trigger:**
  ```python
  call_count = 0
  async def mixed_fn(x):
      return x * 10

  def fn(x):
      global call_count
      call_count += 1
      if call_count == 1:
          return mixed_fn(x)  # awaitable -> triggers _to_async handoff
      raise StopIteration('should propagate')

  result = asyncio.run(Chain([1, 2, 3]).map(fn).run())
  # BUG: returns [10] instead of raising StopIteration
  ```
- **Impact:** Silent data loss. The chain produces a partial result with no error signal. The bug only manifests when a sync-to-async transition occurs mid-iteration AND a subsequent callback raises `StopIteration`.
- **Suggested fix:** Separate `next(iterator)` from `fn(item)` in `_to_async`:
  ```python
  async def _to_async(iterator, item, result, lst, idx):
      try:
          while True:
              if isawaitable(result):
                  result = await result
              _collect(lst, item, result)
              idx += 1
              try:
                  item = next(iterator)
              except StopIteration:
                  return lst
              result = fn(item)
      except _Break as exc:
          return await _async_handle_break(exc, lst)
      except _ControlFlowSignal:
          raise
      except BaseException as exc:
          _set_link_temp_args(exc, link, item=item, index=idx)
          raise
  ```

---

## 4. Medium Severity Issues

### M1-NEW. `on_except_raise=True` Behavioral Asymmetry Between Sync and Async Paths

- **Severity:** MEDIUM
- **File:** `quent/_chain.py:296-299` (sync), `quent/_chain.py:395-396` (async)
- **Description:** When `raise_=True` is set on `except_()`, the sync and async paths handle an awaitable handler result differently. The sync path (`_chain.py:296-299`) **closes** (discards) an awaitable handler result and re-raises the original exception. The async path (`_chain.py:395-396`) **awaits** (executes) the handler result before re-raising the original exception. This means a sync chain silently drops handler side-effects that an async chain would execute.
- **Trigger:** `Chain(fn).except_(async_logging_handler, raise_=True)` run synchronously -- the logging coroutine is closed without execution, so the log entry is silently lost.
- **Impact:** Handler side-effects (logging, metrics, cleanup) are silently dropped in sync context when `raise_=True`.
- **Suggested fix:** Align behavior. Either: (a) use `_fire_and_forget` in the sync path to schedule the handler coroutine, or (b) document that `raise_=True` with async handlers requires async execution context.

### M2-NEW. Leaked Coroutine When `__exit__` Returns Awaitable During `_ControlFlowSignal`

- **Severity:** MEDIUM
- **File:** `quent/_ops.py:116-118`
- **Description:** In the sync path of `_with_op`, when the body raises `_ControlFlowSignal`, `__exit__(None, None, None)` is called but its return value is discarded without checking if it is an awaitable. If `__exit__` returns a coroutine, that coroutine is never awaited or closed, producing `RuntimeWarning: coroutine was never awaited`. Other paths in `_with_op` correctly handle awaitable `__exit__` returns.
- **Trigger:** A sync chain using `.with_()` where the context manager's `__exit__` returns a coroutine, and the body contains a `Chain.return_()` or `Chain.break_()` call.
- **Impact:** Resource leak (coroutine never awaited/closed), `RuntimeWarning` emitted.
- **Suggested fix:**
  ```python
  except _ControlFlowSignal:
      exit_result = current_value.__exit__(None, None, None)
      if isawaitable(exit_result) and hasattr(exit_result, 'close'):
          exit_result.close()
      raise
  ```

### M3-NEW. Inconsistent Dual-Protocol Precedence Between `_make_with` and `_make_iter_op`

- **Severity:** MEDIUM
- **File:** `quent/_ops.py:110` (`_make_with`), `quent/_ops.py:352` (`_make_iter_op`)
- **Description:** `_make_with` checks `__enter__` first (preferring sync). `_make_iter_op` checks `__aiter__` first (preferring async). This contradicts quent's core design principle of "start sync, transition to async only when forced." For a dual-protocol iterable (has both `__iter__` and `__aiter__`), the async path is always taken, returning a coroutine even in a purely sync context.
- **Trigger:** Any object implementing both `__iter__` and `__aiter__` used in `.map()`, `.foreach()`, or `.filter()` -- the async path is always chosen regardless of execution context.
- **Impact:** Unnecessary async transitions; sync callers receive a coroutine they must await.
- **Suggested fix:** Align `_make_iter_op` with `_make_with`: check `__iter__` first, falling back to `__aiter__` only when `__iter__` is absent.

### M4-NEW. `__version__: str` Missing from `__init__.pyi` Type Stub

- **Severity:** MEDIUM (HIGH for downstream type checking)
- **File:** `quent/__init__.pyi`
- **Description:** `__version__` is a public API export listed in `__all__` but has no declaration in `__init__.pyi`. Type checkers (mypy, pyright) cannot see the `__version__` export, causing false-positive errors in downstream code that imports it.
- **Trigger:** `from quent import __version__` in a type-checked codebase produces an "attribute not found" error.
- **Suggested fix:** Add `__version__: str` to `quent/__init__.pyi`.

### M5-NEW. Three Functions Missing from `_ops.pyi` Type Stubs

- **Severity:** MEDIUM
- **File:** `quent/_ops.pyi`
- **Description:** Three internal functions defined in `_ops.py` have no corresponding stub declarations in `_ops.pyi`:
  - `_handle_iterate_exc` (defined at `_ops.py:24`)
  - `_async_handle_break` (defined at `_ops.py:31`)
  - `_make_iter_op` (defined at `_ops.py:281`)
- **Impact:** Internal-only functions, but incomplete stubs can confuse static analysis tools and make the stubs unreliable for contributors.
- **Suggested fix:** Add stub declarations for all three functions.

### M6-NEW. Visualization Failure Silently Swallowed

- **Severity:** MEDIUM
- **File:** `quent/_traceback.py:130`
- **Description:** The `except Exception` block in the traceback modification path catches any error during chain visualization and falls through to just cleaning frames. No logging, no warning. This makes debugging visualization bugs extremely difficult -- the visualization silently disappears with no indication that an error occurred.
- **Impact:** Debugging/diagnostic quality degradation. If chain visualization breaks for a specific chain structure, users see a degraded traceback with no clue that visualization was attempted and failed.
- **Suggested fix:** Add `warnings.warn(f'quent: chain visualization failed: {exc!r}', RuntimeWarning, stacklevel=2)` in the `except` block.

---

## 5. Low Severity Issues

### L1-NEW. `_fire_and_forget` Calls `.close()` Without `hasattr` Check

- **Severity:** LOW
- **File:** `quent/_chain.py:63`
- **Description:** `_fire_and_forget` calls `result.close()` without verifying that the object has a `.close()` method. If `result` is a non-coroutine awaitable (e.g., `asyncio.Future`), it lacks `.close()` and raises `AttributeError`, masking the original `RuntimeError` from `_ensure_future`.
- **Trigger:** A chain handler returns an `asyncio.Future` (not a coroutine) when no event loop is running.
- **Suggested fix:** Guard with `if hasattr(result, 'close'): result.close()`.

### L2-NEW. Sync Finally's `_fire_and_forget` Can Replace Active Exception

- **Severity:** LOW
- **File:** `quent/_chain.py:310`
- **Description:** When the sync finally handler returns an awaitable and `_fire_and_forget` raises `QuentException` (no event loop), that exception propagates out of the `finally:` block, replacing whatever exception was actively propagating. The original error is masked.
- **Trigger:** Sync chain with async finally handler, no running event loop, and a prior exception in the chain.
- **Suggested fix:** Wrap the `_fire_and_forget` call in a `try/except` that logs the scheduling failure without masking the original exception.

### L3-NEW. `_Null` Singleton Claim in CLAUDE.md Does Not Match Implementation

- **Severity:** LOW (documentation accuracy)
- **File:** `quent/_core.py`
- **Description:** CLAUDE.md states `_Null` is "Singleton via `__new__`" with a `__new__` override, but there is no `__new__` override in the code. The singleton is created by convention at module level via `Null = object.__new__(_Null)`, and `_Null` is private (not exported). The singleton property holds in practice, but the documentation is inaccurate.
- **Suggested fix:** Correct CLAUDE.md to describe the actual singleton mechanism (module-level creation, private class).

### L4-NEW. `if_()` Stub Uses Literal `None` Defaults Instead of `...` Convention

- **Severity:** LOW
- **File:** `quent/_chain.pyi:93, 95, 96`
- **Description:** The `if_()` method in `_chain.pyi` uses literal `None` as default values for `predicate`, `args`, and `kwargs` parameters, while the rest of the stub file uses `...` (Ellipsis) for defaults. This is inconsistent and may confuse contributors.
- **Suggested fix:** Align with the `...` convention used elsewhere in the file.

### L5-NEW. Stale Docstring References Removed Classes

- **Severity:** LOW (documentation accuracy)
- **File:** `quent/_traceback.py:319`
- **Description:** The docstring in `_format_link` references `_Foreach` and `_Filter` classes that no longer exist in the codebase. These were replaced by the unified `_make_iter_op` approach.
- **Suggested fix:** Update the docstring to reference the current `_quent_op` attribute mechanism.

### L6-NEW. `_make_gather` Uses `asyncio.ensure_future` Instead of Project's `_create_task_fn`

- **Severity:** LOW (consistency)
- **File:** `quent/_ops.py:410`
- **Description:** `_make_gather` uses `asyncio.ensure_future` directly (line 410) instead of the project's `_create_task_fn` from `_core.py`. Gather tasks therefore do not benefit from `eager_start=True` on Python 3.14+. Functionally correct but inconsistent with the rest of the codebase.
- **Suggested fix:** Use `_create_task_fn` for consistency and to benefit from eager task start on 3.14+.

---

## 6. Correctness Verifications

The following aspects were explicitly audited and confirmed **correct**. These provide confidence in the core implementation.

### Sync-to-Async Transition
All state variables (current value, link pointer, exception handlers, retry attempt index) are correctly passed from `_run()` to `_run_async()` during mid-chain async transition. No data is lost during handoff.

### Retry Mechanism
Retry state is fully reset between attempts. `_ControlFlowSignal` exceptions are never retried -- they propagate immediately. When sync `_run()` discovers an awaitable mid-retry, it delegates to `_run_async()` with the current attempt index; the async path picks up the retry loop from that point. Backoff delay dispatch (`_get_retry_delay`) correctly handles all three backoff types (None, float, callable).

### Fire-and-Forget Task Registry
`_task_registry` uses strong references (prevents GC of in-flight tasks) and `threading.Lock` (correct -- `asyncio.Lock` would be wrong because done-callbacks run synchronously). Auto-cleanup via done callbacks prevents unbounded growth. Follows the pattern documented in official Python documentation.

### Coroutine Cleanup
On all error paths, created coroutines are properly closed via `.close()` to prevent `ResourceWarning`. This includes the gather setup path, the `_ensure_future` fallback, and mid-iteration async handoff failures.

### Context Manager Exit
`__exit__` / `__aexit__` is always called on both success and exception paths in `_make_with`. The `__exit__` return value (True suppresses exception) is correctly handled, including awaitable `__exit__` returns.

### `_Break` Handling
`_Break` signals are correctly caught and handled in both sync and async iteration paths (map/foreach/filter), including mid-iteration async transitions. Partial results are correctly accumulated on break.

### `_evaluate_value` Dispatch
All combinations of (is_chain/not, has_args/not, Ellipsis/not, callable/not, Null current_value/not) produce correct results. The calling convention documented in CLAUDE.md accurately reflects the implementation.

### `_Return`/`_Break` Signal Containment
`run()` and `decorator()` have defensive `_ControlFlowSignal` catches converting leaked signals to `QuentException`. Control flow signals are never caught by `except_()` handlers, never retried, and raise `QuentException` if used inside `except_()` or `finally_()` handlers.

### `inspect.isawaitable()` Usage
Correct and canonical for runtime async detection. Covers coroutines, coroutine-like objects, and objects with `__await__`. Still the recommended approach through Python 3.15.

### `exec()` for Synthetic Frames
The `exec()` hack in `_traceback.py` is safe: it only executes pre-compiled bytecode (`_RAISE_CODE`), not arbitrary strings. The same pattern is used by Jinja2 for template tracebacks.

### Traceback Patching Pattern
Patching `sys.excepthook` and `TracebackException.__init__` is a well-established pattern used by Sentry, Rich, and the `exceptiongroup` backport.

### Circular Exception Chain Handling
`_clean_chained_exceptions` uses a `seen` set keyed by `id()` to prevent infinite loops on circular exception chains.

### Null Sentinel Identity
All comparisons use `is Null`. Pickle roundtrip preserves identity via `__reduce__`. The singleton is established at module load time before any user code runs.

### Linked List Integrity
The append path in `_then()` correctly maintains the singly-linked list with O(1) tail pointer. No cycles are possible through the public API. No reference cycles exist (singly-linked, no back-pointers).

### `asyncio.create_task(eager_start=True)` on Python 3.14
Correctly implemented via `functools.partial` with version gating. The version check `>= (3, 14)` is accurate -- `asyncio.create_task()` only accepts `eager_start` since Python 3.14.

### Exception Chaining in except/finally Handlers
`_except_handler_body` correctly chains exceptions (`from exc`) and modifies tracebacks. The `_run_async` finally block manually sets `__context__` for cases where Python's automatic chaining does not apply in async code.

---

## 7. Type Stubs & API Consistency

### Confirmed Correct

All public-facing stub signatures in `_chain.pyi`, `_core.pyi`, and `__init__.pyi` match their implementations exactly, with the following exceptions noted below. All slot declarations match. Cross-module function signatures are consistent. No stale stubs reference removed features (aside from the items listed).

### Issues Found

| ID | Severity | File | Description |
|----|----------|------|-------------|
| M4-NEW | MEDIUM | `quent/__init__.pyi` | `__version__: str` missing. Public API export in `__all__` with no stub declaration. |
| M5-NEW | MEDIUM | `quent/_ops.pyi` | Three internal functions missing stubs: `_handle_iterate_exc`, `_async_handle_break`, `_make_iter_op`. |
| L4-NEW | LOW | `quent/_chain.pyi:93,95,96` | `if_()` uses literal `None` defaults instead of `...` convention used elsewhere. |

---

## 8. Python 3.13-3.15 Compatibility

The Python forward-compatibility audit returned **all clear**. No blocking issues for Python versions through 3.15.

| Concern | Status | Details |
|---------|--------|---------|
| `Null` singleton under free-threading (3.13t) | **SAFE** | Module-level creation is serialized by import lock, even in free-threaded builds. |
| `threading.Lock` under free-threading | **SAFE** | Works identically in free-threaded Python. |
| `sys.excepthook` assignment | **SAFE** | Module-level, under import lock. |
| `exec()` under free-threading | **SAFE** | Frame is thread-local. No issues. |
| `eager_start` version check `>= (3, 14)` | **CORRECT** | `asyncio.create_task()` only accepts `eager_start` since Python 3.14. |
| `traceback.TracebackException.__init__` signature | **STABLE** | Unchanged through 3.15. |
| `types.TracebackType` constructor | **STABLE** | Unchanged through 3.15. |
| `code.replace(co_name, co_qualname)` | **STABLE** | No deprecation planned. |
| `inspect.isawaitable()` | **CANONICAL** | Still the recommended approach. Faster than equivalent `isinstance` checks. |
| Python 3.15 deprecations | **NONE AFFECT QUENT** | No APIs used by quent are deprecated in 3.15. |

### Async Best Practices Verification

| Pattern | Status | Details |
|---------|--------|---------|
| `asyncio.gather` usage | **CORRECT** | Not deprecated. quent's manual sibling cancellation correctly approximates TaskGroup safety. |
| `asyncio.ensure_future` usage | **ACCEPTABLE** | Has conditional deprecation warning (only when no running event loop). Not fully deprecated. Correct for heterogeneous awaitables. |
| Fire-and-forget with `_task_registry` | **BEST PRACTICE** | Follows official Python documentation pattern exactly. |
| `warnings.warn(RuntimeWarning)` for task exceptions | **ACCEPTABLE** | CPython issue #104091 confirms no perfect solution exists. |
| Coroutine cleanup via `.close()` | **STANDARD** | Still the standard mechanism through Python 3.14+. |
| `asyncio.sleep` for retry backoff | **CORRECT** | Standard pattern. |

---

## 9. Refactoring Opportunities

### R1-NEW. Extract Shared Retry Loop Structure

`_run()` and `_run_async()` implement nearly identical retry loop structures. Extracting a shared template or helper that encapsulates the retry/backoff/attempt-counting logic would reduce duplication and prevent future divergence.

### R2-NEW. Replace Exception Attribute Pollution with WeakKeyDictionary

The codebase attaches multiple attributes directly to user exception objects: `__quent__`, `__quent_source_link__`, `__quent_link_temp_args__`, `__quent_gather_index__`, `__quent_gather_fn__`. A `WeakKeyDictionary` keyed by exception identity would avoid polluting user exception namespaces and prevent potential attribute name collisions with other libraries.

### R3-NEW. Cache Callability on Link Construction

`_evaluate_value` calls `callable(v)` on every evaluation. Since callability is determined at Link construction time and `v` is immutable after creation, storing an `is_callable` boolean on the Link would save a function call per link evaluation on the hot path. (This is the same as previous R4, restated for continuity.)

### R4-PREV. Separate Chain Visualization from Traceback Patching

`_traceback.py` conflates two responsibilities: building the chain visualization string, and patching Python's traceback machinery. Separating these would make the visualization logic reusable (e.g., for logging, debugging) without requiring the traceback hooks.

---

## 10. Product Ideas & Novel Suggestions

Evaluated against quent's identity as a **transparent sync/async bridge** (not a collections library, not a functional programming toolkit).

### High Fit (directly serves the sync/async bridging mission)

| Idea | Rationale |
|------|-----------|
| **`.tap(fn)`** | Like `.do()` but also receives the chain object for introspection/logging. Enables middleware-style observation without modifying the pipeline value. |
| **`.timeout(seconds)`** | Per-link or whole-chain timeout. A genuine async bridging concern -- sync code rarely needs timeouts, but async pipelines frequently do. |
| **`logging.getLogger('quent').debug(...)` instrumentation** | Add `_log.debug(...)` calls at each link evaluation in the execution engine. Users enable via standard `logging.getLogger('quent').setLevel(logging.DEBUG)`. No new API surface needed — follows the pattern used by requests, urllib3, asyncio, and SQLAlchemy. Always-present calls are free when debug level is disabled (logging short-circuits before string formatting). |
| **`.clone()`** | Deep-copy the linked list for fork-and-extend patterns. Enables safe reuse without mutation concerns. |
| **Opt-out traceback patching** | Since `enable_/disable_traceback_patching()` were removed, provide an environment variable (e.g., `QUENT_NO_TRACEBACK=1`) or a pre-import flag for users who need to suppress the patching. |
| **Full generic type safety (`Chain[T]`)** | See detailed breakdown below. Critical for adoption by large typed libraries (Redis, SQLAlchemy, etc.). |

#### Full Generic Type Safety: `Chain[T]`

**Why this is critical:** quent's adoption target includes large, heavily-typed libraries (Redis clients, SQLAlchemy, ORM layers, API frameworks). For these consumers, type safety is non-negotiable — their users expect full IDE autocomplete, type inference through pipelines, and zero `Any`-typed black holes. A chain that erases all type information is unusable in these codebases.

**What "world-class" means here:**

```python
Chain(42)                          # Chain[int]
  .then(str)                       # Chain[str]
  .then(len)                       # Chain[int]
  .do(print)                       # Chain[int]  (value unchanged)
  .map(float)                      # Chain[list[float]]
  .filter(lambda x: x > 1.0)      # Chain[list[float]]
  .run()                           # int -> str -> int -> int -> list[float] -> list[float]
```

Every `.then()` must flow the return type forward. Every `.do()` must preserve the current type. `.map()` must unwrap iterables and re-wrap in `list[R]`. `.run()` must return the final `T`. IDE autocomplete must work at every step.

**Scope of work (stubs only — zero runtime changes):**

| Method | Type signature needed |
|--------|-----------------------|
| `Chain.__init__(v: T)` | `Chain` becomes `Generic[T]` |
| `.then(Callable[[T], R])` | Returns `Chain[R]` |
| `.then(Callable[[T], Awaitable[R]])` | Also returns `Chain[R]` (async transparency) |
| `.then(R)` (non-callable) | Returns `Chain[R]` |
| `.do(Callable[[T], Any])` | Returns `Chain[T]` (value preserved) |
| `.map(Callable[[V], R])` | Where `T = Iterable[V]`, returns `Chain[list[R]]` |
| `.foreach(Callable[[V], Any])` | Where `T = Iterable[V]`, returns `Chain[list[V]]` |
| `.filter(Callable[[V], Any])` | Where `T = Iterable[V]`, returns `Chain[list[V]]` |
| `.gather(*fns)` | Needs variadic generics (PEP 646, Python 3.11+) or overloads for 1-10 fns |
| `.with_(Callable[[CM_T], R])` | Where `T` is a context manager yielding `CM_T`, returns `Chain[R]` |
| `.with_do(Callable[[CM_T], Any])` | Returns `Chain[T]` (value preserved) |
| `.if_(..., then=Callable[[T], R])` | Returns `Chain[T \| R]` (union — predicate may not fire) |
| `.else_(Callable[[T], R])` | Narrows the union from preceding `if_()` |
| `.except_(Callable[[T, BaseException], R])` | Returns `Chain[T \| R]` |
| `.run()` | Returns `T` |
| `.run(V)` | Overrides root — return type depends on chain steps |
| `.iterate()` | Returns `_Generator[V]` where `T = Iterable[V]` |

**Implementation approach:**
- Massive `@overload` stacks in `_chain.pyi` (the `.py` files don't change at all)
- PEP 646 `TypeVarTuple` for `.gather()` (Python 3.11+; use `typing_extensions` for 3.10)
- `ParamSpec` may be needed for decorator patterns
- Separate overloads for callable vs non-callable arguments to `.then()`
- The stub file will likely be 3-5x the size of `_chain.py` itself

**Precedent:** The `returns` library, `Expression`, and `pipe` all implement similar generic chain typing. SQLAlchemy's `Select[T]` type is analogous — a builder that threads types through method chains.

**Risk:** Stub maintenance burden. Every new chain method requires corresponding overloads. Mypy and pyright may behave differently on complex overload resolution. Requires thorough testing with both type checkers.

### Medium Fit (useful but risks scope creep)

| Idea | Rationale |
|------|-----------|
| **Chain composition via `>>` operator** | `chain1 >> chain2` concatenates chains. Pythonic but risks positioning quent as a functional programming library. |

### Low Fit (outside the sync/async bridging mission)

| Idea | Rationale |
|------|-----------|
| **`.log(level, message_fn)`** | Convenience wrapper around `.do()` for structured logging. Useful but trivially composable from existing primitives. |
| **ProcessPoolExecutor dispatch** | CPU-bound parallelism is a different domain entirely. |
| **Structured concurrency (TaskGroups)** | Would transform quent into an async framework, which is explicitly out of scope. |

---

## 11. Untested Edge Cases & Scenarios

The following scenarios are not covered by the current test suite and may reveal undocumented behavior or latent bugs.

### Chain Lifecycle Edge Cases

| # | Scenario | Risk |
|---|----------|------|
| 1 | Chain with only `except_()` and no steps -- what does `.run()` return? | Undefined behavior |
| 2 | Chain with only `finally_()` and no steps | Undefined behavior |
| 3 | Chain with only `retry()` and no steps | Undefined behavior |
| 4 | Empty chain `Chain().run()` -- returns `None` (is this documented?) | Undocumented contract |
| 5 | Very long chains (10000+ links) -- linked list traversal performance | Performance regression |
| 6 | Two chains sharing the same Link objects (mutation of one affects the other) | Aliasing bug |

### Control Flow Edge Cases

| # | Scenario | Risk |
|---|----------|------|
| 7 | `return_()` with a nested Chain as the value | Unclear evaluation semantics |
| 8 | `break_()` with an awaitable value | Unclear whether value is awaited |
| 9 | Recursive chain (chain's step references itself) | Stack overflow or infinite loop |
| 10 | `on_except_raise=True` with no except handler registered | Impossible state or silent no-op |

### Operation Edge Cases

| # | Scenario | Risk |
|---|----------|------|
| 11 | `gather()` with zero functions -- returns `[]` | Verify empty list behavior |
| 12 | `gather()` with a single function | Unnecessary gather overhead |
| 13 | `if_()` with `then` as a tuple of wrong length (0, 1, or 4+) | Missing validation |
| 14 | `do()` with a function that returns a coroutine (coroutine is discarded -- is it closed?) | Potential resource leak |
| 15 | `then()` with `...` (Ellipsis) as the value itself (not as an arg) | Ellipsis as pipeline value |
| 16 | `with_()` where the body function is a Chain | Nested chain in with\_ context |
| 17 | `iterate()` on a chain that returns a non-iterable | Error message quality |
| 18 | `iterate()` with `fn` that raises on the first element | Error propagation |

### Handler Edge Cases

| # | Scenario | Risk |
|---|----------|------|
| 19 | Nested chain where inner chain has `except_()` -- does outer chain's `except_()` also fire? | Double handling |
| 20 | `retry()` with `max_attempts=1` (effectively no retry) | Verify single-attempt semantics |
| 21 | `retry()` with backoff callable that returns negative values | Undefined behavior |
| 22 | `except_()` handler that returns `Null` sentinel | Null as chain result |
| 23 | `finally_()` handler that returns a value (should be discarded) | Verify discard semantics |
| 24 | Chain used as context manager argument to `with_()` of another chain | Deep nesting |

---

## 12. Documentation Drift

The following discrepancies exist between CLAUDE.md / code comments and the actual codebase.

| Location | Discrepancy | Severity |
|----------|-------------|----------|
| CLAUDE.md "Public API" table | Lists 6 exports in `__all__` including `enable_traceback_patching()` and `disable_traceback_patching()`. Actual `__all__` has 4 items -- these two functions were removed. | MEDIUM |
| CLAUDE.md "Reuse" section | Describes `_FrozenChain`, `.freeze()`, and `.decorator()`. `freeze()` and `_FrozenChain` were removed from the codebase. | MEDIUM |
| CLAUDE.md "Chain Class" section | Lists 10 `__slots__` attributes. Actual Chain has 11 slots -- `on_except_raise` is present but undocumented. | LOW |
| CLAUDE.md "Null" section | States `_Null` is "Singleton via `__new__`" with a `__new__` override. There is no `__new__` override; singleton is via module-level `object.__new__(_Null)`. | LOW |
| CLAUDE.md "Public API" | Does not document the `raise_` parameter on `except_()` or the `on_except_raise` attribute. | LOW |
| CLAUDE.md "Key Design Decisions" #8 | Describes frozen chain boundary semantics for a feature that no longer exists. | LOW |
| CLAUDE.md "Competitive Landscape" | Mentions "Freeze for concurrency" as a key differentiator. Feature was removed. | LOW |
| `quent/_traceback.py:319` | Docstring in `_format_link` references `_Foreach` and `_Filter` classes that no longer exist. | LOW |
| `quent/__init__.pyi` | Missing `__version__: str` declaration for a public `__all__` export. | MEDIUM |

---

## Appendix: Methodology

### Audit Agents

Six independent agents analyzed the codebase, each with a distinct focus:

| Agent | Focus Area |
|-------|------------|
| 1 | Core execution engine (`_chain.py`, `_core.py`) -- correctness, edge cases, behavioral invariants |
| 2 | Operations module (`_ops.py`) -- iteration, context managers, gather, if/else |
| 3 | Traceback system (`_traceback.py`) -- patching, visualization, frame cleaning |
| 4 | Type stubs & API consistency (`*.pyi` files, cross-module signatures) |
| 5 | Python 3.13-3.15 forward compatibility -- free-threading, deprecations, API stability |
| 6 | Async best practices -- patterns, conventions, ecosystem alignment |

### Deduplication

Multiple agents identified overlapping concerns. The following were deduplicated:
- StopIteration swallowing (Agents 1 and 2 identified different paths; merged into C1)
- Dual-protocol precedence inconsistency (Agents 2 and 4; merged into M3-NEW)
- `_Null` singleton documentation accuracy (Agents 1 and 4; merged into L3-NEW)
- Exception attribute pollution (Agents 1 and 2; merged into R2-NEW)

### Limitations

- No load testing or benchmarking was performed.
- No security audit was conducted (quent does not handle untrusted input by design).
- Python 3.13t (free-threaded) was not tested at runtime; compatibility assessment is based on specification analysis.
- The audit did not analyze downstream usage patterns or real-world adoption.
