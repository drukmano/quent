# Quent Comprehensive Audit Report

**Date:** 2026-03-07
**Scope:** Full codebase audit -- API design, async/concurrency, exception handling, memory/resources, test coverage, and competitive positioning.
**Codebase:** 5 source files in `quent/`, ~1300 statements, zero runtime dependencies.

---

## 1. Executive Summary

Eight independent audit agents analyzed the quent codebase across orthogonal dimensions. The library is in strong shape: 5,007 tests pass with 99% statement and 99% branch coverage. The core sync/async bridging mechanism is correct and well-designed. The codebase follows a consistent internal architecture with defensive error handling throughout.

That said, the audit identified **4 high-severity issues**, **9 medium-severity issues**, and **12 low-severity issues**. The high-severity items involve silent data loss (StopIteration swallowing, fire-and-forget exception loss, gather not cancelling siblings) and a freeze-mutation safety gap. None are exploitable security vulnerabilities; all are correctness and robustness concerns.

**Issue Counts by Severity:**

| Severity | Count | Category |
|----------|-------|----------|
| HIGH | 4 | Correctness, data loss, safety |
| MEDIUM | 9 | API design, compatibility, semantics |
| LOW | 12 | Edge cases, naming, future-proofing |

---

## 2. Critical & High Severity Issues

### H1. StopIteration Silently Swallowed in map/foreach/filter Operations

- **Severity:** HIGH
- **File:** `quent/_ops.py:334-348` (foreach sync), `quent/_ops.py:276-294` (foreach async), `quent/_ops.py:410-425` (filter sync), `quent/_ops.py:366-378` (filter async)
- **Description:** `_foreach_op` and `_filter_op` use `while True` / `next(it)` to iterate, catching `StopIteration` to end the loop. However, if the user's callback `fn(item)` itself raises `StopIteration` (e.g., by calling `next()` on an exhausted internal iterator), that exception is indistinguishable from the iterator's natural exhaustion. The callback's `StopIteration` is silently caught, terminating the operation early with a **partial result and no error**.
- **Trigger:**
  ```python
  def bad_fn(x):
    if x == 3: raise StopIteration()
    return x * 2
  Chain([1, 2, 3, 4, 5]).map(bad_fn).run()
  # Returns [2, 4] -- items 3, 4, 5 silently dropped!
  ```
- **Impact:** Silent data loss. The chain produces a partial result with no error signal.
- **Suggested Fix:** Separate `next(it)` from `fn(item)` with respect to `StopIteration` handling:
  ```python
  while True:
    try:
      item = next(it)
    except StopIteration:
      break
    result = fn(item)  # StopIteration here now propagates as an error
    ...
  ```

### H2. Fire-and-Forget Task Exceptions Are Silently Lost

- **Severity:** HIGH
- **File:** `quent/_chain.py:238-254`, `quent/_chain.py:260-277`
- **Description:** When an except/finally handler returns a coroutine from a synchronous execution path, it is scheduled as a fire-and-forget task via `ensure_future()`. A `RuntimeWarning` is issued, but if the resulting task raises an exception, that exception is silently discarded by asyncio's default task exception handling. There is no `add_done_callback` or similar mechanism to surface the error.
- **Trigger:** A synchronous chain with an `except_()` handler that calls an async function. The handler is scheduled but its failure is invisible.
- **Impact:** Silent data loss. Cleanup/recovery code fails with no observable signal.
- **Suggested Fix:** Attach a done-callback to the task that logs unhandled exceptions at WARNING or ERROR level. Consider exposing a configurable callback via a module-level hook (e.g., `quent.on_task_error`).

### H3. `asyncio.gather` in `_make_gather` Does Not Cancel Siblings on Failure

- **Severity:** HIGH
- **File:** `quent/_ops.py:441`
- **Description:** The `_to_async` helper calls `asyncio.gather(*coros)` without `return_exceptions=True` and without wrapping in a `TaskGroup` or manual cancellation. If one coroutine raises, the remaining coroutines are left running as orphaned tasks. These tasks may have side effects (database writes, API calls) that continue executing after the gather has already failed.
- **Trigger:** `Chain(v).gather(async_fn_that_fails, async_fn_with_side_effects).run()` -- the second function continues running after the first raises.
- **Impact:** Orphaned async tasks with uncontrolled side effects; resource leaks.
- **Suggested Fix:** Use `asyncio.TaskGroup` (Python 3.11+) with a fallback to manual cancellation on older versions. Alternatively, wrap with `return_exceptions=True` and re-raise the first exception after all tasks complete.

### H4. `freeze()` Does Not Prevent Post-Freeze Mutation

- **Severity:** HIGH
- **File:** `quent/_chain.py:608-617, 649-668`
- **Description:** `_FrozenChain` stores a direct reference to the original `Chain` object (`self._chain = chain`). After `freeze()`, the user retains the original `Chain` reference and can continue calling `.then()`, `.except_()`, etc. on it. These mutations affect the frozen chain because both share the same underlying object. The docstring says "The chain must not be modified after freezing" but this is not enforced.
- **Trigger:** `c = Chain(1).then(lambda x: x+1); fc = c.freeze(); c.then(lambda x: x*10); fc.run()` -- the frozen chain now includes the post-freeze step.
- **Impact:** Undefined behavior in concurrent scenarios; violates the documented contract of `_FrozenChain` being safe for concurrent use.
- **Suggested Fix:** Either (a) deep-copy the linked list at freeze time (copy `root_link`, `first_link`, `current_link`, `on_except_link`, `on_finally_link` and their `next_link` chains), or (b) set a flag on the original Chain that causes mutation methods to raise `QuentException`.

---

## 3. Medium Severity Issues

### M1. `else_()` / `if_()` Coupling Is Fragile

- **Severity:** MEDIUM
- **File:** `quent/_chain.py:596-606`
- **Description:** `else_()` requires that the immediately preceding link has `_quent_op == 'if'`. Inserting any operation between `if_()` and `else_()` (e.g., `.then(log)`) produces a confusing `QuentException('else_() can only be used immediately after if_()')` instead of a targeted diagnostic.
- **Trigger:** `Chain(v).if_(pred, fn).then(log).else_(other_fn)` -- the `.then(log)` breaks the coupling.
- **Suggested Fix:** Improve the error message to explain that no operations may appear between `if_()` and `else_()`, or consider a different API (e.g., `if_(pred, then_fn, else_fn)`).

### M2. `then()` Accepts Non-Callables with Extra Args Silently

- **Severity:** MEDIUM
- **File:** `quent/_chain.py:440-442`, `quent/_core.py:225-228`
- **Description:** `then(42, "extra")` is accepted at build time but fails at runtime with a confusing `TypeError` because `_evaluate_value` tries to call `42("extra")`. Unlike `do()`, which validates callability at build time (line 446), `then()` does not.
- **Trigger:** `Chain(v).then(42, "extra").run()` raises `TypeError: 'int' object is not callable` at runtime.
- **Suggested Fix:** Add a validation check in `then()`: if `args` or `kwargs` are provided, verify that `v` is callable and raise `TypeError` with a clear message otherwise.

### M3. `except_()` Can Silently Swallow `BaseException` Subclasses

- **Severity:** MEDIUM
- **File:** `quent/_chain.py:450-476`
- **Description:** `except_()` accepts `exceptions=` parameter with `BaseException` subclasses including `KeyboardInterrupt` and `SystemExit`. When `exceptions=(BaseException,)` is passed, the handler catches and potentially swallows keyboard interrupts and system exit signals. The default (`Exception`) is safe, but the API allows unsafe configurations without warning.
- **Trigger:** `Chain(v).except_(handler, exceptions=(BaseException,)).run()` swallows `KeyboardInterrupt`.
- **Suggested Fix:** Emit a warning when `exceptions` includes `KeyboardInterrupt`, `SystemExit`, or `GeneratorExit`. Or document this as an explicit footgun.

### M4. Sync Iterator Over Async Chain Leaks Coroutine

- **Severity:** MEDIUM
- **File:** `quent/_ops.py:142-143`
- **Description:** When `iterate()` is used on a chain that returns async results, the sync generator path calls `chain_run(*run_args)` which may return a coroutine. The `for item in` loop then raises `TypeError` and the coroutine is never awaited or closed, causing a `ResourceWarning`.
- **Trigger:** `for x in Chain(async_source).iterate():` where `async_source` returns a coroutine.
- **Suggested Fix:** Wrap the initial `chain_run()` call with a check for awaitables and raise a clear error (e.g., "Cannot use sync iteration on an async chain; use `async for`").

### M5. Hook Conflict with Other Libraries on Disable

- **Severity:** MEDIUM
- **File:** `quent/_traceback.py:412-432`
- **Description:** `disable_traceback_patching()` unconditionally restores `_original_excepthook` (captured at import time). If another library (Sentry, Rich, etc.) installed its hook *after* quent's import, calling `disable_traceback_patching()` destroys that library's hook. The enable/disable pair assumes no other library modifies these globals between calls.
- **Trigger:** `import quent; import sentry_sdk; sentry_sdk.init(); quent.disable_traceback_patching()` -- Sentry's hook is destroyed.
- **Suggested Fix:** On disable, check if `sys.excepthook` is still quent's hook before restoring. If it has been replaced by a third party, leave it alone (or warn).

### M6. Three-Tier Sync/Async Pattern Duplicated Across `_ops.py`

- **Severity:** MEDIUM (maintainability)
- **File:** `quent/_ops.py:263-359` (foreach), `quent/_ops.py:362-429` (filter), `quent/_ops.py:91-129` (with\_)
- **Description:** The three-tier pattern (sync fast path, mid-operation async transition, full async path) is duplicated across `_make_foreach`, `_make_filter`, and `_make_with`. Each implements the same structural pattern with minor variations. This makes the code harder to maintain and increases the surface area for divergent bugs.
- **Suggested Fix:** Extract a shared higher-order function or base pattern that encapsulates the three-tier handoff, parameterized by the operation-specific logic.

### M7. Library Is asyncio-Native, Incompatible with trio/anyio-trio

- **Severity:** MEDIUM (ecosystem)
- **Description:** All async machinery uses `asyncio.gather`, `asyncio.create_task`, `asyncio.ensure_future`, and `asyncio.get_event_loop`. Code running under trio or anyio's trio backend cannot use quent's async features. This limits adoption in projects that use alternative async runtimes.
- **Suggested Fix:** Consider anyio as an optional backend for the gather/task operations, or document the asyncio-only constraint prominently.

### M8. `_run()` and `_run_async()` Share Extensive Duplicated Logic

- **Severity:** MEDIUM (maintainability)
- **File:** `quent/_chain.py`
- **Description:** The sync `_run()` and async `_run_async()` methods duplicate the link traversal, exception handling, and finally logic. Changes to one must be mirrored in the other, creating a maintenance burden and risk of divergence.
- **Suggested Fix:** Refactor shared logic into helper methods, keeping only the sync/async dispatch as the distinguishing factor.

### M9. `filter()` Does Not Support `break_()`

- **Severity:** MEDIUM
- **File:** `quent/_ops.py:362-428`
- **Description:** `_make_foreach` catches `_Break` explicitly (lines 288, 316, 345) allowing `break_()` to terminate iteration early with a partial result via `_handle_break_exc`. `_make_filter` does NOT catch `_Break` -- it only catches the parent class `_ControlFlowSignal` (lines 379, 396, 421), which re-raises `_Break`. The `_Break` then propagates to the chain's main exception handler, which converts it to `QuentException('Chain.break_() cannot be used outside of a map/foreach iteration.')`. This is an API inconsistency -- `break_()` works in `map()`/`foreach()` but not `filter()`.
- **Trigger:**
  ```python
  Chain([1, 2, 3, 4, 5]).filter(
    lambda x: Chain.break_() if x > 3 else x > 1
  ).run()
  # Raises QuentException instead of returning [2, 3]
  ```
- **Suggested Fix:** Add `except _Break as exc:` handlers to `_filter_op`, its `_to_async`, and `_full_async`, mirroring the pattern in `_make_foreach`.

---

## 4. Low Severity Issues

### L1. `_Null` Singleton Has Theoretical TOCTOU Race on Free-Threaded Python

- **Severity:** LOW
- **File:** `quent/_core.py:14-23`
- **Description:** The `__new__` method checks `cls._instance is None` and then assigns. On CPython 3.13t (free-threaded, no GIL), two threads could both see `None` and create two instances. Practically harmless since `_Null` has no state, but violates the singleton invariant.
- **Suggested Fix:** No action needed for current Python versions. If free-threaded support becomes a goal, use a `threading.Lock` or module-level `Null = object.__new__(_Null)`.

### L2. Non-Atomic Enable/Disable Traceback Patching

- **Severity:** LOW
- **File:** `quent/_traceback.py:409-432`
- **Description:** `enable_traceback_patching()` and `disable_traceback_patching()` modify `_patching_enabled`, `sys.excepthook`, and `TracebackException.__init__` in separate statements without a lock. On GIL-free Python, another thread could observe a partially-patched state.
- **Suggested Fix:** Wrap in a `threading.Lock` if free-threaded Python support is a goal.

### L3. `id(link)` Key in `_set_link_temp_args` Risks Collision After GC

- **Severity:** LOW
- **File:** `quent/_core.py:120-122`
- **Description:** `__quent_link_temp_args__` is keyed by `id(link)`. If a link is garbage-collected and a new link is allocated at the same address, the key could collide. In practice this is harmless because temp args are transient (attached to an in-flight exception) and links are alive during chain execution.
- **Suggested Fix:** No action needed. The data is short-lived and the race is theoretical.

### L4. User Code Physically Inside `quent/` Directory Would Be Filtered from Tracebacks

- **Severity:** LOW
- **File:** `quent/_traceback.py:41-59`
- **Description:** `_clean_internal_frames` filters frames whose `co_filename` starts with `_quent_file` (the quent package directory). If a user places their own code inside the quent package directory, their frames would be stripped from tracebacks.
- **Suggested Fix:** No action needed. This is an extreme edge case (users should not put code in site-packages).

### L5. `_ControlFlowSignal.args` Leaks Internal Details If It Escapes

- **Severity:** LOW
- **File:** `quent/_core.py:47-64`
- **Description:** `_ControlFlowSignal` stores `value`, `args_`, and `kwargs_` as instance attributes. If a signal escapes the chain (which the guard at `_chain.py:435-438` prevents), these internal details would be visible. The guard makes this a defense-in-depth concern only.
- **Suggested Fix:** No action needed. The guard is already in place.

### L6. Synthetic Traceback Frame Creates Reference Cycle

- **Severity:** LOW
- **File:** `quent/_traceback.py:113-124`
- **Description:** `_modify_traceback` stores the exception in `globals_['__exc__']` and then uses `exec()` to create a frame whose globals dict references the exception. The exception's `__traceback__` references the frame, creating a cycle: `exc -> __traceback__ -> frame -> globals_ -> exc`. Python's cyclic GC handles this, but it delays collection.
- **Suggested Fix:** Set `globals_['__exc__'] = None` after extracting the traceback, or use `weakref`.

### L7. `__quent_gather_index__` / `__quent_gather_fn__` Never Deleted from Exceptions

- **Severity:** LOW
- **File:** `quent/_ops.py:461-463`
- **Description:** When a gather operation fails, diagnostic attributes are attached to the exception but never cleaned up. These leak internal details in the exception object if it is caught and inspected by user code.
- **Suggested Fix:** Clean up these attributes in the chain's exception handler after using them for diagnostics.

### L8. Inconsistent Closure Capture Timing Across Operations

- **Severity:** LOW
- **File:** `quent/_ops.py:265,364` vs `quent/_ops.py:99,481`
- **Description:** `_make_foreach` and `_make_filter` capture `fn` (the callable) at factory time (build-time binding). `_make_with` and `_make_if` read the callable from the link at runtime. This inconsistency is not a bug (the link is immutable after creation) but makes the code harder to reason about.
- **Suggested Fix:** Standardize on one approach for consistency.

### L9. Parameter Name `v` Is Opaque

- **Severity:** LOW
- **File:** `quent/_chain.py` (throughout), `quent/_core.py` (throughout)
- **Description:** The parameter name `v` is used pervasively for "value or callable" arguments. While terse, it gives no hint about the dual nature of the parameter (it can be a plain value OR a callable). This makes the code harder to read for new contributors.
- **Suggested Fix:** Consider `v_or_fn` or document the convention in a code comment at the top of the module.

### L10. Dual-Mode Context Managers Always Use Async Path

- **Severity:** LOW
- **File:** `quent/_ops.py:95`
- **Description:** `_with_op` checks `hasattr(current_value, '__aenter__')` before checking `__enter__`. Objects implementing both `__enter__` and `__aenter__` (common in libraries like `aiohttp`, `asyncpg`) always take the async path, returning a coroutine even in a purely sync execution context. If the caller does not await the result, `__aexit__` is never called.
- **Trigger:**
  ```python
  class DualCM:
    def __enter__(self): return 'sync'
    def __exit__(self, *a): return False
    async def __aenter__(self): return 'async'
    async def __aexit__(self, *a): return False

  Chain(DualCM()).with_(lambda ctx: 'body').run()
  # Returns a coroutine object instead of 'body'
  ```
- **Suggested Fix:** Consider checking for a running event loop or allowing the user to specify sync/async preference, or prefer `__enter__` when the chain is in sync mode.

### L11. Ellipsis Convention Silently Drops Extra Arguments

- **Severity:** LOW
- **File:** `quent/_core.py:88-89` (`_resolve_value`), `quent/_core.py:219-220, 225-226` (`_evaluate_value`)
- **Description:** When `args[0] is ...`, the function is called with zero arguments (`v()`). Any additional positional arguments after the Ellipsis are silently ignored without error or warning.
- **Trigger:** `Chain().then(fn, ..., 'extra_arg').run()` -- `'extra_arg'` is silently dropped.
- **Suggested Fix:** Validate that when Ellipsis is the first arg, no further args or kwargs are provided: `if args[0] is ... and (len(args) > 1 or kwargs): raise QuentException(...)`.

### L12. `_ControlFlowSignal.__init__` Comment Is Factually Wrong

- **Severity:** LOW (documentation only)
- **File:** `quent/_core.py:58-60`
- **Description:** The comment claims skipping `super().__init__()` avoids "the standard Exception args tuple." In reality, `BaseException.__new__` sets `self.args` from the constructor's positional arguments regardless of whether `__init__` calls `super()`. The `.args` tuple `(v, args, kwargs)` is always allocated. The optimization comment is misleading.
- **Suggested Fix:** Remove or correct the comment to reflect reality.

---

## 5. Correctness Verification

The following aspects were explicitly audited and found to be **correct**. These findings provide confidence in the core implementation.

### Sync-to-Async Transition
- All state variables (current value, link pointer, exception handlers) are correctly passed from `_run()` to `_run_async()` during mid-chain async transition. No data is lost during the handoff.

### Retry Mechanism
- Retry state (attempt count, delay) is fully reset between retries. No stale state accumulation across attempts.

### Frozen Chain Concurrency
- `_FrozenChain` instances are safe for concurrent use from multiple `asyncio.Task`s, provided the underlying chain is not mutated post-freeze (see H4 for the mutation concern).

### Coroutine Cleanup
- On all error paths, created coroutines are properly closed via `.close()` to prevent `ResourceWarning`. This includes the gather setup path (`_ops.py:458-460`), the ensure_future fallback (`_chain.py:244-246`), and mid-iteration async handoff failures.

### `_aiter_wrap` Correctness
- The async iterator wrapper correctly handles both `__aiter__` and `__anext__` protocols.

### Context Manager Exit
- `__exit__` / `__aexit__` is always called on both success and exception paths in `_make_with`, including when `_ControlFlowSignal` is raised.

### `_Break` Handling in Async Paths
- `_Break` signals are correctly caught and handled in both sync and async foreach/map iterations, including mid-iteration async transitions.

### Fire-and-Forget Task Registry
- `_task_registry` uses strong references (correct -- prevents GC of in-flight tasks). CPython issue #91887 confirms this is still the recommended pattern.
- Uses `threading.Lock` (correct -- `asyncio.Lock` would be wrong because done-callbacks run synchronously in the event loop thread).

### `inspect.isawaitable()` Usage
- Correct and canonical for runtime async detection. Covers coroutines, coroutine-like objects, and objects with `__await__`.

### `exec()` for Synthetic Frames
- The `exec()` hack in `_traceback.py` is safe: it only executes pre-compiled bytecode (`_RAISE_CODE`), not arbitrary strings. The same pattern is used by Jinja2 for template tracebacks.

### Traceback Patching Pattern
- Patching `sys.excepthook` and `TracebackException.__init__` is a well-established pattern used by Sentry, Rich, and the `exceptiongroup` backport.

### `types.TracebackType` Construction
- Stable public API since Python 3.7. No compatibility concerns.

### `asyncio.create_task(eager_start=True)` on Python 3.14
- Correctly implemented via `functools.partial` with version gating (`_core.py:126-128`).

### Circular Exception Chain Handling
- `_clean_chained_exceptions` uses a `seen` set keyed by `id()` to prevent infinite loops on circular exception chains (`_traceback.py:62+`).

### Memory Management
- Chain linked lists do NOT create reference cycles (singly-linked, no back-pointers).
- `_task_registry` does NOT create cycles (set of tasks, no circular references).
- Closure variables in `_ops.py` do NOT create cycles.
- Generator cleanup relies on standard Python GC (correct).

### `_evaluate_value` Dispatch
- All 9 combinations of (is_chain/not, has_args/not, Ellipsis/not, callable/not, Null current_value/not) produce correct results. Exhaustively tested.

### `_Return`/`_Break` Signal Containment
- `run()` (`_chain.py:434`) and `decorator()` (`_chain.py:420`) have defensive `_ControlFlowSignal` catches converting leaked signals to `QuentException`. No leak path exists through normal API use.

### Exception Chaining in except/finally Handlers
- `_except_handler_body` correctly chains exceptions (`from exc`) and modifies tracebacks. The `_run_async` finally block manually sets `__context__` (`_chain.py:402-403`) for cases where Python's automatic chaining doesn't apply in async code.

### Linked List Integrity
- The three-path append in `_then()` (`_chain.py:122-133`) correctly maintains the singly-linked list with O(1) tail pointer. No cycles are possible through the public API.

### `ignore_finally` Flag
- Set atomically with `_run_async` delegation (`_chain.py:176`). The finally handler runs exactly once -- in `_run_async`'s own finally block, not in `_run`'s.

### Null Sentinel Identity
- All comparisons use `is Null`. Pickle roundtrip preserves identity via `__reduce__`. The singleton is established at module load time before any user code runs.

---

## 6. Refactoring Opportunities

### R1. Consolidate Three-Tier Sync/Async Pattern

The foreach, filter, and with\_ operations each independently implement the same structural pattern: sync fast path, mid-operation async transition, and full async path. Extracting a shared abstraction would reduce ~200 lines of near-duplicate code in `_ops.py` and make the pattern easier to audit.

### R2. Unify `_run()` and `_run_async()` Shared Logic

The sync and async execution methods in `_chain.py` duplicate link traversal, exception handling, and finally logic. A refactor could extract the shared control flow into helper methods, with sync/async dispatch as the only varying axis.

### R3. Separate Chain Visualization from Traceback Patching

`_traceback.py` conflates two responsibilities: (1) building the chain visualization string, and (2) patching Python's traceback machinery. Separating these would make the visualization logic reusable (e.g., for logging, debugging) without requiring the traceback hooks.

### R4. Store Callability Flag on Link

`_evaluate_value` calls `callable(v)` on every evaluation. Since callability is known at Link construction time, storing a `is_callable` boolean on the Link would save a function call per link evaluation. Micro-optimization, but the hot path warrants it.

### R5. Flatten Linked List to Tuple in `freeze()`

`_FrozenChain` traverses a linked list on every `.run()`. Since frozen chains are immutable, the linked list could be flattened to a tuple at freeze time for cache-friendly sequential access.

---

## 7. Product Ideas & Feature Suggestions

Evaluated against quent's identity as a **transparent sync/async bridge** (not a collections library, not a functional programming toolkit).

### High Fit (directly serves the sync/async bridging mission)

| Idea | Rationale |
|------|-----------|
| **Debug/trace mode** | A `.debug()` or `Chain.trace(callback)` method that emits each link's input/output without modifying the pipeline. Essential for debugging opaque chains. Does not exist in any competitor. |
| **Timeout support** | `.timeout(seconds)` on individual links or the whole chain. A genuine async bridging concern -- sync code does not need timeouts, but async pipelines frequently do. |
| **Chain cloning** | `.clone()` that deep-copies the linked list, enabling safe fork-and-extend patterns. Solves the freeze-mutation problem (H4) as a side effect. |

### Medium Fit (useful but risks scope creep)

| Idea | Rationale |
|------|-----------|
| **Composition operators** (`\|`, `>>`) | `chain1 \| chain2` to compose chains. Pythonic but risks positioning quent as a functional programming library. |
| **Conditional chaining** `.when(pred)` | Sugar for common `if_()` patterns. Useful but can be built with existing `if_()`. |
| **Parallel map** `.pmap()` | `asyncio.gather`-based parallel mapping. Genuine async concern but overlaps with gather. |

### Low Fit (outside the sync/async bridging mission)

| Idea | Rationale |
|------|-----------|
| `Promise.race()` equivalent | Niche; better served by `asyncio.wait(return_when=FIRST_COMPLETED)`. |
| ProcessPoolExecutor dispatch | CPU-bound parallelism is a different domain. |
| Structured concurrency (TaskGroups) | Would make quent an async framework. |

---

## 8. Test Coverage Analysis

### Summary

| Metric | Value |
|--------|-------|
| Total tests | 5,007 |
| Passing | 5,005 |
| Skipped | 2 (version-gated) |
| Failing | 0 |
| Statement coverage | 99% (1,321 / 1,323 lines) |
| Branch coverage | 99% (452 / 466 branches) |

### Uncovered Lines

The 2 uncovered statements are version-gated code paths:
- `quent/_core.py:127` -- `eager_start=True` path (Python 3.14+ only)
- The corresponding else branch (Python < 3.14)

### Uncovered Branches (14 total)

| Category | Count | Notes |
|----------|-------|-------|
| Structurally unreachable (dead code by design) | 6 | Defensive guards that cannot fire under current architecture |
| Version-gated | 2 | Python 3.14+ features |
| Cold-path / diagnostic code | 6 | Error formatting, edge-case handlers |

### Assessment

The core execution engine (`_chain.py:_run`, `_chain.py:_run_async`, `_core.py:_evaluate_value`) has **100% statement coverage**. No high-severity coverage gaps exist. The uncovered branches are either structurally unreachable or version-gated -- neither category represents a testing deficiency.

---

## 9. Competitive Landscape

### Unique Position

Quent is the **only library** that combines pipeline/chain abstraction with transparent sync/async bridging. No other Python library does both.

### Closest Alternatives

| Library | Downloads | Relationship to quent |
|---------|-----------|----------------------|
| **asgiref** | ~2B | `sync_to_async` / `async_to_sync` wrappers. No chaining. |
| **anyio** | ~5.8B | Runtime-agnostic async primitives. No pipeline abstraction. |
| **unsync** | ~2M | Auto-bridging philosophy (closest to quent's), but rudimentary chaining. |
| **toolz** | ~1.5B | Functional pipelines (sync only). No async support. |
| **returns** | ~14M | Typed FP with monadic composition. Has async support but opinionated architecture. |

### Key Differentiators

1. **Transparent runtime detection** -- a single chain works for both sync and async callables without the user declaring which.
2. **Rich control flow** -- `if_/else_`, `return_()`, `break_()`, `retry()`, `except_/finally_` within the pipeline.
3. **Traceback enhancement** -- chain visualization injected into standard Python tracebacks.
4. **Zero dependencies** -- no runtime dependencies whatsoever.
5. **Freeze for concurrency** -- immutable chain snapshots safe for concurrent async use.

### Feature Gaps vs Ecosystem

- No timeout support (anyio, asyncio have this natively)
- No alternative async runtime support (trio, curio)
- No `Promise.race()` / `FIRST_COMPLETED` equivalent
- No structured concurrency integration (TaskGroups)

### Python 3.13/3.14 Considerations

- **Free-threading (3.13t):** L1 and L2 identify minor concerns. No blocking issues.
- **Eager task start (3.14):** Already implemented (`_core.py:126-128`).
- **Multiple interpreters:** Not relevant to quent's architecture.

---

## 10. Methodology

### Audit Agents

Eight independent agents analyzed the codebase, each with a distinct focus:

| Agent | Focus Area |
|-------|------------|
| 1 | Critical bugs, logic errors, correctness proofs |
| 2 | Async/concurrency correctness, sync-to-async transitions |
| 3 | Exception handling, traceback patching, error propagation |
| 4 | API design, developer experience, product ideas |
| 5 | Memory management, resource lifecycle |
| 6 | Python best practices research, pattern validation |
| 7 | Test coverage measurement and gap analysis |
| 8 | Competitive analysis and ecosystem positioning |

### Files Analyzed

| File | Lines | Role |
|------|-------|------|
| `quent/_chain.py` | ~668 | Chain class, execution engine, freeze |
| `quent/_core.py` | ~232 | Link, Null sentinel, value evaluation |
| `quent/_ops.py` | ~510 | foreach, filter, gather, with\_, if\_ operations |
| `quent/_traceback.py` | ~432 | Traceback patching, chain visualization |
| `quent/__init__.py` | -- | Public API surface |
| `quent/_chain.pyi` | -- | Type stubs |

### Deduplication

Multiple agents identified the same issues independently. The following findings were deduplicated:
- Reference cycle via `globals_['__exc__']` (Agents 3 and 5)
- `id(link)` key collision risk (Agents 3 and 5)
- Fire-and-forget task exception loss (Agents 2, 4)
- Freeze-mutation safety (Agents 1, 4)
- Gather sibling cancellation (Agents 1, 2)
- `_Null` singleton thread safety (Agents 1, 2)

### Limitations

- No load testing or benchmarking was performed.
- No security audit was conducted (quent does not handle untrusted input by design).
- Python 3.13t (free-threaded) was not tested at runtime; concerns are theoretical.
- The audit did not analyze downstream usage patterns or real-world adoption.
