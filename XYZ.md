# EXHAUSTIVE CODE REVIEW — `quent/` Library

**Scope:** All 5 source files in `quent/` (482 + 212 + 458 + 383 + 8 = 1,543 lines)
**Standard:** Life-and-death, $1,000,000 expert review
**Date:** 2026-03-06
**Reviewed by:** Claude Opus 4.6 orchestrator with 7 parallel deep-dive agents
**Files audited:** `_core.py`, `_chain.py`, `_ops.py`, `_traceback.py`, `__init__.py`

---

## Table of Contents

1. [Bugs & Correctness Issues](#i-bugs--correctness-issues)
   - [Critical](#critical-none-found)
   - [High Severity](#high-severity)
   - [Medium Severity](#medium-severity)
   - [Low Severity](#low-severity)
2. [Security](#ii-security)
3. [Thread Safety](#iii-thread-safety)
4. [Memory & Performance](#iv-memory--performance)
5. [API Design Issues & Suggestions](#v-api-design-issues--suggestions)
6. [Product Ideas & New Features](#vi-product-ideas--new-features)
7. [Refactoring Opportunities](#vii-refactoring-opportunities)
8. [Competitive Analysis](#viii-competitive-analysis)
9. [Summary](#ix-summary)
10. [Detailed Per-File Audit Reports](#x-detailed-per-file-audit-reports)

---

## I. BUGS & CORRECTNESS ISSUES

### CRITICAL: None Found

The core execution engine is sound. Sync/async bridging is watertight. All 8 initial-value combinations in `_run()` produce correct results. Exception handling is proper. The library is well-engineered.

---

### HIGH SEVERITY

#### H1. `_await_exit_suppress` loses original exception context when `__exit__` awaitable raises

- **File:** `_ops.py:76-79`
- **Description:** When a sync context manager's `__exit__` returns an awaitable that raises during body-exception handling, the new exception has no `__cause__` or `__context__` referencing the original body exception. Python's implicit chaining doesn't fire because `exc` is in a closure, not the active exception.
- **Impact:** Debugging becomes harder because the original body exception is silently discarded with no traceback chain.
- **Demonstrated:**
  ```
  # Standard Python: RuntimeError.__context__ = ZeroDivisionError
  # _await_exit_suppress: RuntimeError.__context__ = None, __cause__ = None
  ```
- **Fix:**
  ```python
  async def _await_exit_suppress(suppress, exc, outer_value):
      try:
        if await suppress:
          return outer_value if ignore_result else None
      except BaseException as exit_exc:
        raise exit_exc from exc
      raise exc
  ```
- **ASSUMPTION:** This is rare in practice (sync `__exit__` returning a coroutine that raises is exotic), but it is a semantic correctness deviation from the `with` statement protocol.

---

#### H2. `_sync_generator` yields unawaited coroutines when `fn` is async

- **File:** `_ops.py:128-161`
- **Description:** `_sync_generator` calls `fn(item)` and yields the result directly without checking `isawaitable(result)`. If a user creates an `iterate()` generator with an `async def` function and consumes it synchronously via `for item in g`, they receive raw coroutine objects instead of their resolved values. These coroutines are never awaited, generating `RuntimeWarning: coroutine was never awaited`.
- **Demonstrated:**
  ```python
  async def double(x): return x * 2
  g = Chain([1, 2, 3]).iterate(double)
  list(g)  # => [<coroutine>, <coroutine>, <coroutine>]
  ```
- **Impact:** Silent data corruption (coroutine objects where values were expected) if the user calls `list(g)` instead of `[x async for x in g]` when `fn` is async.
- **Mitigation:** The user should use `async for` instead of `for` when `fn` is async. The `_async_generator` path correctly awaits results. However, there is no guard, warning, or documentation preventing the footgun.
- **Fix:** Detect the first awaitable in `_sync_generator` and raise a clear error rather than silently yielding coroutines.
- **ASSUMPTION:** This is considered a user error, not a library bug, given the library's philosophy of transparent sync/async bridging.

---

#### H3. `_get_true_source_link` can infinite loop with cyclic chain references

- **File:** `_traceback.py:134-147`
- **Description:** The `while source_link is not None:` loop has no cycle-detection mechanism. If two chains have their `root_link` properties set such that `chain_a.root_link.v = chain_b` and `chain_b.root_link.v = chain_a`, this function enters an infinite loop that hangs the process.
- **Confirmed via testing:** A cyclic chain structure causes a timeout (infinite loop). The function never terminates.
- **Exploitability through public API:** LOW. The `root_link` is set only at `Chain.__init__` construction time and is never mutated by public methods. Creating a cycle requires direct slot assignment, which is internal API abuse. However, `root_link` is a public slot (not name-mangled), so a user could accidentally or intentionally trigger this.
- **Fix:**
  ```python
  def _get_true_source_link(source_link, root_link):
    seen = set()
    while source_link is not None and id(source_link) not in seen:
      seen.add(id(source_link))
      ...
  ```

---

#### H4. `_clean_chained_exceptions` unbounded recursion

- **File:** `_traceback.py:62-70`
- **Description:** The function is recursive, following both `__cause__` and `__context__` chains. With a linear exception chain of ~1000 entries, this hits Python's default recursion limit (1000). Confirmed: a chain of 2000 exceptions causes `RecursionError`.
- **Impact:** A `RecursionError` during traceback formatting would mask the original error. This is exception handling code crashing during exception handling — the worst kind of failure.
- **Fix:** Convert to an iterative approach using an explicit stack:
  ```python
  def _clean_chained_exceptions(exc, seen):
    stack = [exc]
    while stack:
      exc = stack.pop()
      if exc is None or id(exc) in seen:
        continue
      seen.add(id(exc))
      if exc.__traceback__ is not None:
        exc.__traceback__ = _clean_internal_frames(exc.__traceback__)
      stack.append(exc.__cause__)
      stack.append(exc.__context__)
  ```

---

#### H5. `_stringify_chain` / `_resolve_nested_chain` recursion overflow on deep nesting

- **File:** `_traceback.py:203-228` and `_traceback.py:231-282`
- **Description:** `_resolve_nested_chain` calls `_stringify_chain` which calls `_format_link` which calls `_resolve_nested_chain` — forming a recursive cycle for nested chains. With ~500 levels of chain nesting, this causes `RecursionError`. Confirmed: a 200-deep nested chain stringifies to a 243KB string successfully, but a 500-deep chain hits `RecursionError`.
- **Impact:** Also produces enormous `co_name` strings (1MB+ strings accepted by `code.replace()`, producing 1MB+ traceback output).
- **Fix:** Add a `max_depth` guard with fallback to truncated output, or convert to iterative traversal.

---

#### H6. `_modify_traceback` failure replaces the original exception

- **File:** `_chain.py:45` (inside `_except_handler_body`)
- **Description:** `_modify_traceback(exc, chain, link, root_link)` is called before the except handler runs. If it raises (e.g., from any of H3-H5), the formatting error replaces the application error entirely. The user's `except_()` handler is never invoked, and the original application error becomes hidden as `__context__` of the traceback-formatting error.
- **Verified empirically:** Injecting a broken `_modify_traceback` causes the formatting error to propagate instead of the actual `ZeroDivisionError`.
- **Fix:** Wrap `_modify_traceback` in try/except within `_except_handler_body`:
  ```python
  def _except_handler_body(exc, chain, link, root_link):
    try:
      _modify_traceback(exc, chain, link, root_link)
    except Exception:
      pass  # formatting failure must never mask application error
    ...
  ```

---

#### H7. `_task_registry` not thread-safe on free-threaded Python 3.13+

- **File:** `_core.py:125`
- **Description:** `_task_registry: set[asyncio.Task[Any]] = set()` is a plain `set`. Under the GIL (standard CPython), `set.add()` and `set.discard()` are effectively atomic at the bytecode level. On free-threaded Python 3.13+ (compiled with `--disable-gil`), CPython adds internal locks but the Python docs explicitly state: "this should be treated as a description of the current implementation, not a guarantee of current or future behavior."
- **Impact:** Multiple event loops in separate threads calling `_ensure_future` concurrently could corrupt the set's internal hash table.
- **ASSUMPTION:** The authors consider free-threaded Python as future scope. Under GIL-enabled builds, this is safe.

---

#### H8. Frozen chain `_Return`/`_Break` semantics silently differ from unfrozen nested chains — undocumented

- **File:** `_chain.py:462-475`
- **Description:** `_FrozenChain.run()` catches `_ControlFlowSignal`, so `Chain.return_()` inside a frozen sub-chain returns a value to the *next step* instead of exiting the *outer chain*.
- **Verified empirically:**
  - `Chain(5).then(unfrozen_inner).then(f).run()` — `_Return` inside `unfrozen_inner` propagates to the outer chain, skipping `f`. Result: the return value.
  - `Chain(5).then(frozen_inner).then(f).run()` — `_Return` inside `frozen_inner` is caught by `frozen_inner.run()`. The return value becomes the input to `f`. Result: `f(return_value)`.
- **Impact:** Semantically correct (frozen = opaque boundary) but undocumented. Users who nest frozen chains expecting the same behavior as unfrozen chains will be surprised.

---

### MEDIUM SEVERITY

#### M1. `_make_gather` cleanup closes any object with `.close()`, not just coroutines

- **File:** `_ops.py:444-449`
- **Description:** The cleanup loop in the `except BaseException` block calls `.close()` on any result that has a `close` attribute. This includes not just unawaited coroutines (the intended target) but also file objects, database connections, StringIO buffers, sockets, and any other object with a `.close()` method that a sync fn might return.
- **Demonstrated:**
  ```python
  buffer = io.StringIO('hello')
  Chain(5).gather(lambda x: buffer, lambda x: 1/0).run()
  # buffer is now closed, even though it was a valid sync return value
  ```
- **Fix:** Check specifically for coroutines:
  ```python
  from inspect import isawaitable
  for r in results:
      if isawaitable(r) and hasattr(r, 'close'):
          r.close()
  ```
  Or use `asyncio.iscoroutine(r)` for even tighter specificity.

---

#### M2. `asyncio.gather` without `return_exceptions=True` — partial results lost

- **File:** `_ops.py:430`
- **Description:** When one gathered coroutine raises, others continue running but their results are lost. There is no way for the user to access partial results. Additionally, `asyncio.gather` raises the first exception, but if multiple coroutines fail, subsequent exceptions are silently swallowed.
- **Assessment:** This is an intentional design choice consistent with how `asyncio.gather` is typically used. The alternative (`return_exceptions=True`) would change the return type semantics. Current behavior is fail-fast, which is the safer default.
- **Recommendation:** Document this behavior. Consider offering `gather_safe()` that uses `return_exceptions=True`.

---

#### M3. `_aiter_wrap` blocks event loop with synchronous iterators

- **File:** `_ops.py:164-167`
- **Description:** When an async generator wraps a sync iterator via `_aiter_wrap`, each call to `__anext__` blocks the event loop for the duration of `next(sync_iter)`. If the sync iterator performs I/O or computation, the entire event loop stalls.
- **Assessment:** Inherent limitation of bridging sync iterators into async context. No general-purpose solution without `asyncio.to_thread` or similar. Should be documented as a known limitation.

---

#### M4. `_quent_file` uses `abspath` instead of `realpath` — symlink-based filtering failure

- **File:** `_traceback.py:17`
- **Code:** `_quent_file: str = os.path.dirname(os.path.abspath(__file__)) + os.sep`
- **Description:** `os.path.abspath()` does NOT resolve symlinks. If quent is installed via a symlink (e.g., `pip install -e .` with a symlinked path, or a symlinked virtualenv), `_quent_file` will contain the symlink path, while `tb.tb_frame.f_code.co_filename` in other quent source files may contain the resolved real path (or vice versa). The `startswith(_quent_file)` check on line 51 would fail, and internal quent frames would leak into user tracebacks.
- **Confirmed:** On macOS, `/var` is a symlink to `/private/var`. When importing through a symlinked path, `abspath` and `realpath` return different results.
- **Fix:** `os.path.realpath(__file__)` instead of `os.path.abspath(__file__)`.

---

#### M5. `_get_obj_name` doesn't sanitize newlines in output

- **File:** `_traceback.py:172-187`
- **Description:** When `_get_obj_name` falls through to `repr(obj)`, the repr string may contain newlines or other control characters. This string is embedded into the chain visualization which becomes `co_name` in the traceback. The result is a malformed traceback display where the function name spans multiple lines.
- **Confirmed:** An object with `__repr__` returning `"evil\nline2\nline3"` produces a traceback where the chain visualization breaks across lines.
- **Fix:** Sanitize the return value: `return repr(obj).replace('\n', '\\n')` or truncate at the first newline.

---

#### M6. No error guard in `_quent_excepthook` / `_patched_te_init`

- **File:** `_traceback.py:349-382`
- **Description:** Neither `_quent_excepthook` nor `_patched_te_init` has a `try/except` guard around the `_clean_chained_exceptions` call. If `_clean_chained_exceptions` raises (e.g., from `RecursionError` per H4), the exception propagates through the hook.
- **Impact:** For `_quent_excepthook`, Python's fallback prints both the original exception and the hook error. For `_patched_te_init`, it crashes whatever code was trying to format the exception.
- **Fix:** Wrap the cleaning logic in try/except and fall through to the original behavior on failure:
  ```python
  def _quent_excepthook(exc_type, exc_value, exc_tb):
    try:
      if getattr(exc_value, '__quent__', False):
        _clean_chained_exceptions(exc_value, set())
        exc_tb = exc_value.__traceback__
    except Exception:
      pass
    _original_excepthook(exc_type, exc_value, exc_tb)
  ```

---

#### M7. Import-time side effects with no opt-out

- **File:** `_traceback.py:359,382` and `__init__.py:3`
- **Description:** Importing `quent` (even just `from quent import Null`) immediately patches `sys.excepthook` and `traceback.TracebackException.__init__`. There is no opt-out mechanism.
- **Impact:** Affects ALL exception display in the process. Can conflict with other libraries that also patch `sys.excepthook` (e.g., IPython, rich, sentry-sdk) depending on import order.
- **Details:**
  - If quent is imported BEFORE another hook library: quent's hook chains to the original default, the other library replaces quent's hook entirely — quent's cleaning is lost.
  - If quent is imported AFTER: quent chains to the other library's hook correctly.
- **Recommendation:** Consider an opt-out mechanism (e.g., `QUENT_NO_TRACEBACK_PATCH=1` env var, or `quent.disable_traceback_patching()`).

---

#### M8. `Link.__init__` crashes on objects whose `__getattr__` raises non-`AttributeError`

- **File:** `_core.py:181`
- **Code:** `self.is_chain = getattr(v, '_is_chain', False)`
- **Description:** `getattr(obj, name, default)` only swallows `AttributeError`. If a user object's `__getattr__` raises a different exception (e.g., `ValueError`, `RuntimeError`), `Link.__init__` will crash with that exception.
- **Verified empirically:**
  ```python
  class RaisesValueError:
      def __getattr__(self, name):
          raise ValueError(f'Cannot access {name}')

  Link(RaisesValueError())  # raises ValueError: Cannot access _is_chain
  ```
- **Impact:** Any user object with a broken `__getattr__` (proxy objects, ORMs with lazy loading, descriptors) will crash at chain construction time with a confusing non-quent exception.
- **ASSUMPTION:** Objects that raise non-`AttributeError` from `__getattr__` are violating the descriptor protocol convention and would cause similar problems in many Python libraries. Likely acceptable as a design trade-off.

---

#### M9. `Link.__init__` crashes when setting `is_nested = True` on frozen objects

- **File:** `_core.py:183`
- **Code:** `v.is_nested = True`
- **Description:** If `v` has `_is_chain = True` (detected by `getattr`), the code unconditionally sets `v.is_nested = True`. This raises `AttributeError` if `v` uses `__slots__` without an `is_nested` slot, or if `v` has a `__setattr__` that rejects writes.
- **Verified empirically:**
  ```python
  class Frozen:
      __slots__ = ()
      _is_chain = True

  Link(Frozen())  # AttributeError: 'Frozen' object has no attribute 'is_nested'
  ```
- **Assessment:** In practice, only `Chain` instances have `_is_chain = True`, and `Chain` defines `is_nested` in its `__slots__`. User objects mimicking `_is_chain` is an unsupported scenario. However, if any third-party code or future refactoring introduces `_is_chain` on a read-only object, this will crash.

---

#### M10. Sync chain returns unawaited coroutine with no warning

- **File:** `_chain.py:151`
- **Description:** When `_run` encounters an awaitable, it returns a coroutine from what appears to be a sync call. There's no warning emitted (unlike the except/finally handlers which warn at lines 199-204 and 223-228). Python will emit `RuntimeWarning: coroutine was never awaited` if the coroutine is GC'd, but the message points to quent internals, not user code.
- **Impact:** A user calling `result = chain.run()` might not realize they got a coroutine instead of a value.

---

### LOW SEVERITY

#### L1. `_resolve_value` silently ignores trailing args/kwargs when `args[0]` is Ellipsis

- **File:** `_core.py:80-81`
- **Description:** `_resolve_value(fn, (..., 2, 3), {'k': 'v'})` calls `fn()`, silently dropping `2, 3` and `k='v'`. Consistent with the Ellipsis convention ("call with no arguments"), but could surprise users who accidentally pass extra args alongside Ellipsis.

---

#### L2. `_handle_break_exc`/`_handle_return_exc` propagate exceptions from `_resolve_value`

- **File:** `_core.py:87-100`
- **Description:** Both functions call `_resolve_value(exc.value, exc.args_, exc.kwargs_)` without a try/except. If the callable stored in `exc.value` raises an exception when called, it propagates unhandled. This means `Chain.return_(bad_callable, ...)` or `Chain.break_(bad_callable, ...)` will crash with the callable's exception rather than a clear error.
- **Verified:**
  ```python
  _handle_break_exc(_Break(lambda: 1/0, (...,), None), 'fallback')
  # raises ZeroDivisionError
  ```
- **ASSUMPTION:** Intentional — the user provided the callable, and its exceptions should propagate naturally.

---

#### L3. `decorator()` captures mutable chain with no documentation warning

- **File:** `_chain.py:310-327`
- **Description:** The `decorator()` method creates a closure over `self`. If the user adds links after calling `.decorator()`, the decorated function executes with the modified chain. This is documented in tests as "undefined behavior by design."
- **Gap:** `freeze()` has an explicit docstring warning about post-freeze modification (lines 425-429). `decorator()` lacks an equivalent warning.

---

#### L4. `_make_filter` does not handle `_Break`

- **File:** `_ops.py:351-417`
- **Description:** Asymmetric with `_make_foreach`. `_make_foreach` catches `_Break` and returns partial results. `_make_filter` does NOT — `_Break` propagates up and becomes `QuentException('Chain.break_() cannot be used outside of a foreach iteration.')`.
- **Assessment:** Intentional. Filter is conceptually a transformation, not an iteration loop. `Chain.break_()` inside a filter predicate is a user error.

---

#### L5. `_make_gather` — no error reporting for which function failed

- **File:** `_ops.py:435-453`
- **Description:** When a gathered function raises, the exception doesn't indicate which function (by index or name) caused the failure. Unlike `_make_foreach` and `_make_filter` which attach `item` and `index` via `_set_link_temp_args`, `_gather_op` does not annotate the exception.

---

#### L6. `do()` type annotation says `Callable` but accepts any value at runtime

- **File:** `_chain.py:343`
- **Signature:** `def do(self, fn: Callable[..., Any], /) -> Chain`
- **Description:** At runtime, `do(42)` works fine: `_evaluate_value` sees `callable(42) == False` and returns `42` as-is; since `ignore_result=True`, the result is discarded. The type annotation `Callable[..., Any]` is misleading. Static type checkers (mypy) would flag `chain.do(42)` as an error, but it works at runtime. `then()` correctly uses `v: Any`.

---

#### L7. `from None` on `_ControlFlowSignal` catch leaks internal signals via `__context__`

- **File:** `_chain.py:51,64,172,270`
- **Description:** `raise QuentException(...) from None` sets `__cause__ = None` and `__suppress_context__ = True`, but Python still sets `__context__` to the active exception (the `_ControlFlowSignal`). The signal is NOT displayed in tracebacks (suppressed), but IS accessible via `exc.__context__`, which could leak internal implementation details to users who inspect exception chains.

---

#### L8. `_Null` is not a true singleton — `_Null()` creates additional instances

- **File:** `_core.py:12-30`
- **Description:** Nothing prevents `_Null()` from being called directly. A second instance would fail `is Null` checks. `_Null` is a private class (underscore prefix) not exported in `__all__`, so users would need to go out of their way. The pickle `__reduce__` correctly preserves singleton identity.

---

#### L9. No size/depth limit on chain visualization string

- **File:** `_traceback.py:97`
- **Description:** The `_stringify_chain` output is used as `co_name` without any length limit. Deep nesting (even below the recursion limit) produces visualization strings of hundreds of kilobytes (243KB at 200 levels). This becomes the "function name" in the traceback, producing unwieldy output.
- **Fix:** Add a maximum depth parameter with truncation fallback.

---

#### L10. `_ControlFlowSignal` skips `super().__init__()` — cosmetic issue

- **File:** `_core.py:50-56`
- **Description:** When `super().__init__()` is not called, `BaseException.__new__` still sets `.args` to all positional arguments. For `_Return(42, (1, 2), {'k': 'v'})`: `exc.args = (42, (1, 2), {'k': 'v'})`, `str(exc) = "(42, (1, 2), {'k': 'v'})"`. This does NOT break `except Exception` catching, `traceback` formatting, `logging.exception()`, or pickling.
- **Assessment:** Since these are internal-only signals always caught by the chain engine and never displayed to users, the misleading `.args` is cosmetic only.

---

#### L11. `_ensure_future` with `eager_start=True` creates briefly "done but registered" task

- **File:** `_core.py:131-137`
- **Description:** On Python 3.14, if the coroutine completes without blocking, the task is already done when `_task_registry.add(task)` is called. The done callback is scheduled via `loop.call_soon()`, not called immediately. The task sits in the registry until the next event loop iteration. Not a correctness bug — cleanup happens correctly.

---

#### L12. `_get_obj_name` uses `or` chaining which treats falsy `__name__` as absent

- **File:** `_traceback.py:177`
- **Code:** `name = getattr(obj, '__name__', None) or getattr(obj, '__qualname__', None)`
- **Description:** If `obj.__name__` is `''` (empty string) or any other falsy value, it is treated as absent and `__qualname__` is checked instead. Minor display issue.

---

#### L13. `_format_call_args` silently drops args after Ellipsis in display

- **File:** `_traceback.py:193-196`
- **Description:** When `args[0]` is `Ellipsis`, only `'...'` is appended to parts, and `args[1:]` are ignored. Matches the semantic convention but could be mildly confusing for `(..., extra1, extra2)`.

---

#### L14. `_Generator.__call__` shallow copy shares mutable chain reference

- **File:** `_ops.py:237-240`
- **Description:** The new `_Generator` shares the same `_chain_run` bound method (and thus the same `Chain` instance) as the original. If the underlying `Chain` is mutated between creating an iterator and consuming it, the generator sees the mutated chain.
- **Assessment:** Documented/expected behavior. `freeze()` exists for creating immutable snapshots.

---

#### L15. `QuentException` has `__slots__ = ()` — effectively a no-op

- **File:** `_core.py:36`
- **Description:** `QuentException` uses `__slots__ = ()`, but since `Exception` doesn't define `__slots__`, instances still get `__dict__` from `Exception`. The `__slots__` declaration is a no-op in terms of preventing attribute creation.

---

#### L16. `_evaluate_value` — no validation that `v._run` exists when `is_chain` is True

- **File:** `_core.py:198-203`
- **Description:** When `link.is_chain` is `True`, `_evaluate_value` calls `v._run(...)` without checking that `_run` exists. If a user object has `_is_chain = True` but no `_run` method, this crashes with `AttributeError`. Only `Chain` objects should have `_is_chain = True` — this is an internal protocol.

---

#### L17. `_get_obj_name` does not catch `BaseException` subclasses from `repr()`

- **File:** `_traceback.py:184-187`
- **Description:** `except Exception` won't catch `KeyboardInterrupt` or `SystemExit` from `repr(obj)`. An object with `@property __name__` that raises `KeyboardInterrupt` would escape through `_get_obj_name`. Arguably correct — `KeyboardInterrupt` should terminate the program.

---

## II. SECURITY

### exec() is SAFE

The `exec(code, globals_, {})` call at `_traceback.py:114` is safe because:

1. **The code is fixed:** Always the pre-compiled `'raise __exc__'` statement. Not user-controlled.
2. **The chain visualization string goes into `co_name`/`co_qualname`** — these are metadata strings displayed in tracebacks, NOT executable code. Verified: injecting `"import os; os.system('...')"` as `co_name` does NOT execute the payload.
3. **The `globals_` dict is freshly constructed** from trusted values. No user-controlled data enters the dict.
4. **`__builtins__` is auto-injected** by `exec()` but the code is fixed, so this has no impact.

### `__reduce__` pickle singleton is CORRECT

Returning `'Null'` from `__reduce__` resolves via `getattr(quent._core, 'Null')` during unpickling, preserving singleton identity across all pickle protocols (0-5). Verified empirically: `pickle.loads(pickle.dumps(Null)) is Null` returns `True`.

---

## III. THREAD SAFETY

### FrozenChain IS Thread-Safe — CONFIRMED

- `_run()` uses only local variables for traversal state
- Each concurrent call creates its own temporary `Link` when `has_run_value=True`
- No shared mutable state is touched during execution
- Exception handling writes to the EXCEPTION object (`exc.__quent_source_link__`, etc.), not to the chain
- Verified by existing tests in `concurrent_safety_tests.py`

### Chain is NOT Thread-Safe — CONFIRMED

- `_then()` mutates `self.current_link` and `self.first_link`
- Concurrent `_then()` calls from different threads could corrupt the linked list
- `_run()` reads attributes that could be mutated by concurrent `_then()` calls

### `_task_registry` — Conditionally Safe

- Safe under GIL (standard CPython through 3.14)
- `set.add()` and `set.discard()` are effectively atomic under the GIL
- On free-threaded Python 3.13+: CPython adds internal locks for sets, but Python docs say this is not guaranteed
- In practice, asyncio tasks are created within a single event loop thread, making true concurrent access rare

### `sys.excepthook` Patching

- Global state — patches apply process-wide
- Thread safety during import: patching happens at module import time, which is serialized by the import lock
- `threading.excepthook` (Python 3.8+) defaults to calling `sys.excepthook`, so quent's patch works transitively for thread exceptions
- `faulthandler` is unaffected (reads C-level frame structures directly)

---

## IV. MEMORY & PERFORMANCE

### No Reference Cycles in Link Linked Lists

Links form a singly-linked list (`link1.next_link -> link2.next_link -> ...`). No back-references exist. Standard reference counting handles cleanup without requiring GC cycle collection.

### `_task_registry` Cleanup is Correct

- `discard` callback fires via `call_soon` on the event loop
- If the callback raises, asyncio logs and continues (does not crash the loop, does not leak)
- `set.discard` never raises (no-op if element not present)

### `foreach`/`filter` Accumulate All Results in Memory

By design. Both `_make_foreach` and `_make_filter` collect all results into `lst` (a `list`). For very large iterables, this could be a concern. The lazy/streaming alternative is `Chain.iterate()` which returns a `_Generator` object.

### `_Generator` Does NOT Create Retention Cycles

`_Generator` holds `self._chain_run` (bound method of Chain) and `self._chain` (the Chain object). Chain has no back-reference to `_Generator`. One-way reference, not a cycle.

### `_ControlFlowSignal.__slots__` Interaction

`_ControlFlowSignal` declares `__slots__`, but since `Exception` (its base) doesn't use `__slots__`, instances still have `__dict__`. The named slots (`value`, `args_`, `kwargs_`) are stored in slots for performance, which is correct.

### Frozen Chain Retains Entire Chain Structure

`_FrozenChain.__init__` stores `self._chain = chain`. Multiple `freeze()` calls return different `_FrozenChain` objects that all reference the SAME underlying `Chain`. The original chain cannot be GC'd while any frozen reference exists.

### `original_value` Field Retains Strong References

`Link.original_value` is used only for traceback formatting but retains a strong reference to the original value. For typical use cases (tens of links), the memory overhead is negligible.

---

## V. API DESIGN ISSUES & SUGGESTIONS

### Naming Conventions

| Current | Suggested Alias | Rationale |
|---------|----------------|-----------|
| `.do()` | `.tap()` | Universal convention (RxJS, Elixir, pipe). More recognizable. |
| `.foreach()` | `.map()` | `foreach` implies side-effects in most languages. `map` is universal for "apply and collect." |

### Type Annotation Inconsistencies

| Method | Current Annotation | Runtime Behavior | Fix |
|--------|-------------------|-----------------|-----|
| `then()` | `v: Any` | Accepts any value | Correct |
| `do()` | `fn: Callable[..., Any]` | Accepts any value (non-callable is discarded) | Change to `Any` or document |
| `except_()` | `fn: Any` | Accepts any value | Add `Callable` or document |
| `finally_()` | `fn: Any` | Accepts any value | Add `Callable` or document |
| `foreach()` | `fn: Callable[[Any], Any]` | No runtime check | Correct but undocumented |
| `filter()` | `fn: Callable[[Any], Any]` | No runtime check | Correct but undocumented |
| `gather()` | `*fns: Callable[[Any], Any]` | No runtime check | Correct but undocumented |

### Missing Validation

- No eager callability check on `foreach()`, `filter()`, `gather()` — errors surface only at execution time
- No enforcement of the "don't modify after freeze/decorator" contract

### Empty Chain Behavior (Well-Defined)

- `Chain().run()` → returns `None`
- `Chain().then(42).run()` → returns `42`
- `Chain().except_(handler).run()` → returns `None` (no exception, handler not invoked)
- `Chain().finally_(handler).run()` → returns `None`, invokes handler with no args

### `Chain(lambda: 42).run(99)` Behavior

When `run(v)` is called on a chain with a `root_link`, the run value creates a temporary Link that replaces the root for that execution. `lambda: 42` is entirely bypassed. The chain is NOT mutated. This is by design but could surprise users.

---

## VI. PRODUCT IDEAS & NEW FEATURES

### Tier 1: High Impact, Moderate Effort

#### 1. Conditional Branching: `.if_()` / `.if_not()`

Pattern from: RxJS (`filter`), returns (`lash`), Elixir (`then` with guards)

```python
Chain(value)
  .then(process)
  .if_(lambda v: v > 0, lambda v: v * 2)        # apply only if predicate true
  .if_not(lambda v: v is None, lambda v: v.upper())  # apply only if predicate false
```

This is the single most requested feature across pipeline library issues. Currently users must embed conditionals inside lambdas, breaking readability.

#### 2. `.retry()` — Built-in Retry with Backoff

Pattern from: RxJS `retry`/`retryWhen`, Effect `Schedule`, tenacity

```python
Chain()
  .then(fetch_from_api)
  .retry(max_attempts=3, backoff=lambda n: 2**n, on=(ConnectionError, TimeoutError))
```

Most requested missing feature for async pipeline libraries. Users currently must wrap individual functions with tenacity, which breaks the fluent API.

#### 3. `.timeout()` — Per-Step or Per-Chain Timeout

Pattern from: RxJS `timeout`, Effect structured concurrency, asyncio

```python
Chain()
  .then(slow_operation).timeout(seconds=5.0)
  .then(fast_followup)
```

For async chains, wrap the awaitable in `asyncio.wait_for`. For sync, raise after deadline. Critical for production use.

#### 4. Type Stubs / mypy Support

Pattern from: returns (mypy plugin), Expression (native typing)

Even partial type annotations using `@overload` for common arities (2-7 steps) would dramatically improve IDE experience. The `returns` library proved this is the #1 adoption driver for serious users. At minimum, provide `.pyi` stubs for `Chain`, `then`, `run`.

---

### Tier 2: Medium Impact, Low Effort

#### 5. `.tap()` Alias for `.do()`

Universal naming convention. Consider also a `.log(label)` shorthand: `do(lambda v: logger.debug(f'{label}: {v}'))`.

#### 6. `.map()` Alias for `.foreach()`

Universal naming convention across every programming language.

#### 7. `.reduce()` / `.fold()` Terminal Operation

Pattern from: Rust `fold`/`reduce`, PyFunctional `reduce`, toolz

```python
Chain([1, 2, 3, 4])
  .reduce(lambda acc, x: acc + x, initial=0)  # returns 10
```

#### 8. `.flat_map()` — Map Then Flatten

Pattern from: Rust `flat_map`, PyFunctional `flat_map`, RxJS `mergeMap`

```python
Chain([[1, 2], [3, 4]])
  .flat_map(lambda x: [v * 2 for v in x])  # yields [2, 4, 6, 8]
```

#### 9. `.filter_map()` — Combined Filter+Map

Pattern from: Rust `filter_map`

```python
Chain([1, 2, 3, 4, 5])
  .filter_map(lambda x: x * 2 if x % 2 == 0 else None)  # yields [4, 8]
```

Return `None` to skip, any other value to keep. Eliminates common two-step filter-then-map.

#### 10. `.take(n)` / `.skip(n)` / `.take_while(pred)` / `.skip_while(pred)`

Pattern from: Rust iterators, pipe library, PyFunctional. Fundamental iteration control primitives.

#### 11. `|` Operator Support

Pattern from: pipe library, pipetools, RxJS

```python
Chain(data) | transform1 | transform2 | transform3
```

Implement `__or__` to accept callables and append them as `.then()` steps.

---

### Tier 3: Novel / Advanced Ideas

#### 12. `.scan(fn, initial)` — Stateful Accumulation Yielding Intermediates

Pattern from: Rust `scan`, RxJS `scan`, toolz `accumulate`

```python
Chain([1, 2, 3, 4])
  .scan(lambda acc, x: acc + x, initial=0)  # yields [1, 3, 6, 10]
```

Unlike `reduce` which produces a single final value, `scan` produces a stream of intermediate accumulations.

#### 13. Placeholder `X` Object for Lambda-Free Expressions

Pattern from: pipetools `X`, lodash/fp auto-curry

```python
Chain([1, 2, 3, 4])
  .filter(X % 2 == 0)    # instead of lambda x: x % 2 == 0
  .map(X * 10)            # instead of lambda x: x * 10
```

A proxy object that records operations and replays them when called.

#### 14. `.partition(pred)` — Split Into Two Collections

Pattern from: Rust `partition`, Haskell

```python
Chain([1, 2, 3, 4, 5])
  .partition(lambda x: x % 2 == 0)  # returns ([2, 4], [1, 3, 5])
```

#### 15. `.enumerate()` — Yield `(index, element)` Pairs

Pattern from: Rust, Python built-in

#### 16. `.distinct()` / `.batch(n)` / `.flatten()` / `.zip()`

Common collection operations available in every pipeline library.

#### 17. `.log(label)` Shorthand for Debug Logging

```python
Chain(data)
  .then(transform)
  .log('after transform')  # prints: after transform: <value>
  .then(next_step)
```

#### 18. Opt-Out for Traceback Patching

Environment variable `QUENT_NO_TRACEBACK_PATCH=1` or explicit `quent.disable_traceback_patching()`.

---

### What NOT to Add (Wrong Scope)

- **Result/Maybe containers** — That's `returns`/`Expression` territory. quent's strength is pipeline composition, not error-as-values
- **DAG-based dependency resolution** — That's `pipefunc` territory (scientific computing)
- **Multiprocessing parallelism** — That's `pypeln` territory
- **Serialization I/O** — That's `PyFunctional` territory

---

## VII. REFACTORING OPPORTUNITIES

### Priority 1: Safety-Critical Fixes

1. **Wrap `_modify_traceback` in try/except** in all callers (`_except_handler_body`, `_finally_handler_body`, generator error paths) — a formatting failure must NEVER mask an application error (H6)

2. **Convert `_clean_chained_exceptions` to iterative** — recursive traversal is a ticking time bomb for deep exception chains (H4)

3. **Add cycle/depth guards to `_get_true_source_link` and `_stringify_chain`** — prevents infinite loops and recursion overflow (H3, H5)

4. **Add error guards to `_quent_excepthook` and `_patched_te_init`** — a crash in exception formatting is worse than unformatted output (M6)

### Priority 2: Correctness Fixes

5. **Fix `_await_exit_suppress` exception chaining** — add `try/except` and `raise exit_exc from exc` (H1)

6. **Tighten `_make_gather` cleanup** — only close awaitables, not arbitrary `.close()` objects (M1)

7. **Use `os.path.realpath`** instead of `os.path.abspath` in `_traceback.py` — prevents symlink-related frame filtering failures (M4)

8. **Sanitize `_get_obj_name` output** — replace newlines in repr strings (M5)

### Priority 3: Quality Improvements

9. **Detect unawaited coroutines in `_sync_generator`** — raise a clear error instead of silently yielding coroutines (H2)

10. **Document frozen chain control flow semantics** — explicitly state that `return_()`/`break_()` don't propagate through frozen boundaries (H8)

11. **Add `decorator()` docstring warning** about post-decoration mutation (L3)

12. **Harmonize type annotations** — make `do()`, `except_()`, `finally_()` consistent with `then()` (L6)

---

## VIII. COMPETITIVE ANALYSIS

### Competitive Landscape

| Library | Stars | Downloads/mo | Async? | Key Strength |
|---------|-------|-------------|--------|-------------|
| toolz | 5.1k | ~40M | No | Ubiquitous functional utilities |
| returns | 4.2k | ~300k | Yes (containers) | Railway-oriented programming, mypy plugin |
| PyFunctional | 2.5k | ~300k | No | Scala-style sequence operations |
| pypeln | 1.6k | N/A | Yes | Multi-backend concurrent pipelines |
| Expression | 735 | ~40k | Yes (effects) | F#-inspired effects system |
| pipefunc | 455 | N/A | Yes | DAG-based scientific pipelines |
| pipe | N/A | ~100k | No | Infix `|` operator syntax |

### What quent Does Uniquely Well

1. **Transparent sync/async bridging** — no other library auto-detects awaitables and switches mid-chain
2. **Traceback enhancement** — chain visualization in standard Python tracebacks is unique
3. **Zero-ceremony decorator mode** — `Chain().then(validate).then(process).decorator()`
4. **`freeze()` for concurrent reuse** — explicit concurrency-safe snapshot
5. **Ellipsis convention** — `...` means "call with no arguments" — elegant and unique
6. **Non-local control flow** (`return_`, `break_`) — powerful pattern most libraries lack

### Feature Gap Matrix

| Feature | quent | returns | Expression | toolz | PyFunctional | pipe | RxJS | Effect |
|---------|-------|---------|------------|-------|-------------|------|------|--------|
| Fluent method chaining | YES | partial | YES | NO | YES | YES (infix) | YES | YES |
| Sync/async bridging | YES (auto) | YES (containers) | YES | NO | NO | NO | YES | YES |
| Error as values (Result) | NO | YES | YES | NO | NO | NO | YES | YES |
| Typed error channel | NO | YES | YES | NO | NO | NO | NO | YES |
| Dependency injection | NO | YES | NO | NO | NO | NO | NO | YES |
| Side-effect step (tap/do) | YES | NO | NO | NO | NO | YES (tee) | YES (tap) | YES |
| Retry/backoff | NO | NO | NO | NO | NO | NO | YES | YES |
| Timeout | NO | NO | YES (cancel) | NO | NO | NO | YES | YES |
| Conditional branching | NO | YES (lash) | YES (match) | NO | NO | NO | YES | YES |
| Parallel map/filter | NO | NO | NO | NO | YES (pseq) | NO | YES | YES |
| gather/fan-out | YES | NO | NO | YES (juxt) | NO | NO | YES | YES |
| Context manager | YES | YES (managed) | YES | NO | NO | NO | NO | YES |
| Freeze/reuse | YES | YES (pipe) | YES (pipe) | YES (compose) | NO | YES | NO | YES |
| Traceback enhancement | YES | NO | NO | NO | NO | NO | NO | NO |
| Zero dependencies | YES | YES | YES | YES | YES | YES | N/A | N/A |
| Decorator mode | YES | NO | NO | NO | NO | YES (@Pipe) | N/A | N/A |
| Pattern matching | NO | NO | YES | NO | NO | NO | NO | YES |

### Key Libraries Analyzed in Detail

#### dry-python/returns (4.2k stars, ~300k downloads/mo)

- Railway-oriented programming: two-track execution (success/failure)
- Container types: `Maybe`, `Result`, `IO`, `IOResult`, `Future`, `FutureResult`, `RequiresContext`
- `flow(instance, *functions)` (up to 21 steps), `pipe(*functions)` (up to 7 steps, deferred)
- Pointfree utilities: `bind`, `map_`, `alt`, `lash`
- Full mypy plugin for type inference across composition boundaries
- Higher Kinded Types emulation
- **Unique:** `lash()` (inverse of `bind`, operates on failure track), `managed()` resource protocol, `RequiresContext` for dependency injection

#### Expression (735 stars)

- F#-inspired: `Option`, `Result`/`Try`, `Block`, `Map`, `Sequence`, `TypedArray`
- **Effects system** using `yield`/`yield from` as syntactic sugar for monadic bind
- `CancellationToken` for cooperative cancellation
- `Disposable` for resource management
- Structural pattern matching integration (Python 3.10+)
- **Unique:** Effects system (zero-boilerplate early-exit composition), CancellationToken, immutable persistent data structures

#### toolz/cytoolz (5.1k stars, ~40M downloads/mo)

- `pipe(data, f, g, h)`, `compose(*fns)`, `curry`, `juxt(*fns)`
- `thread_first(val, *forms)` / `thread_last(val, *forms)` — Clojure-style threading macros
- Rich iterator/dict utilities
- cytoolz: Cython drop-in, 2-5x faster
- **Unique:** `curry`, `juxt`, `thread_first`/`thread_last`, `memoize`, `sliding_window`, `accumulate`

#### PyFunctional (2.5k stars, ~300k downloads/mo)

- `seq(1, 2, 3).map(fn).filter(fn).reduce(fn)` — Scala/Spark-inspired
- Dual API: Scala-style + LINQ-style
- Aggregate functions, set operations
- Multi-format I/O: CSV, JSON, JSONL, SQLite3, gzip/bz2/lzma
- `pseq(...)` — parallel execution via multiprocessing
- **Unique:** Built-in parallel execution, serialization/deserialization, aggregate functions

#### pipe (JulienPalard/Pipe)

- Infix `|` operator: `data | select(fn) | where(fn) | sort(fn) | take(n)`
- Lazy evaluation throughout (generators)
- ~30+ built-in pipes
- `@Pipe` decorator for custom pipes
- **Unique:** Infix syntax, `tee` for debugging, `dedup`/`uniq`, `batched(n)`, `take_while`/`skip_while`, `transpose`

### Cross-Language Insights

#### From JavaScript/TypeScript (lodash/fp, RxJS, fp-ts, Effect)

- **Dual APIs** (data-first and data-last) for both direct calls and composition
- **Effect's three type parameters**: `Effect<Success, Error, Requirements>` encoding return, errors, and dependencies
- **Structured concurrency**: parallel, racing, cancellation, timeouts, retry — all composable
- **`switchMap`** — cancel previous async operation on new value
- **`debounceTime`/`throttleTime`** — rate control in the pipeline
- **`retry(n)`/`retryWhen(notifier)`** — retry as a pipeline operator
- **`distinctUntilChanged`** — deduplication
- **`scan(accumulator, seed)`** — running accumulation

#### From Rust Iterator Patterns

- **`scan(state, fn)`** — stateful iteration yielding intermediates
- **`filter_map(fn)`** — combined filter+map (None = skip, Some(v) = keep)
- **`flat_map(fn)`** — map then flatten
- **`take_while(pred)` / `skip_while(pred)`**
- **`intersperse(sep)`** — insert separator
- **`peekable()`** — look-ahead
- **`partition(pred)`** — split by predicate
- **`fold(init, fn)` / `reduce(fn)`** — terminal accumulation
- **`any(pred)` / `all(pred)`** — short-circuit boolean checks
- **`cycle()`** — infinite repetition
- **`array_chunks(N)` / `map_windows(N, fn)`** — windowed operations

#### From Elixir Pipe Patterns

- **`|>` operator** — first argument threading
- **`tap(value, fn)`** — side effect, return original (= quent's `do`)
- **`then(value, fn)`** — pipe into anonymous function
- **Convention:** data always first argument, same type returned — everything is pipeable by default

---

## IX. SUMMARY

### Findings by Severity

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Bugs/Correctness | 0 | 8 | 10 | 17 |
| Security | 0 | 0 | 0 | 0 |
| **Total** | **0** | **8** | **10** | **17** |

### High-Severity Findings Quick Reference

| ID | File | Line(s) | Description |
|----|------|---------|-------------|
| H1 | `_ops.py` | 76-79 | `_await_exit_suppress` loses exception context |
| H2 | `_ops.py` | 128-161 | `_sync_generator` yields unawaited coroutines |
| H3 | `_traceback.py` | 134-147 | `_get_true_source_link` infinite loop on cycles |
| H4 | `_traceback.py` | 62-70 | `_clean_chained_exceptions` unbounded recursion |
| H5 | `_traceback.py` | 203-282 | `_stringify_chain` recursion overflow |
| H6 | `_chain.py` | 45 | `_modify_traceback` failure masks app errors |
| H7 | `_core.py` | 125 | `_task_registry` not thread-safe on free-threaded Python |
| H8 | `_chain.py` | 462-475 | Frozen chain control flow semantics undocumented |

### Overall Assessment

The library is well-engineered with no critical bugs. The core execution engine (sync/async bridging, linked list traversal, exception handling) is correct and thoroughly tested. The 8 HIGH findings are concentrated in two areas:

1. **Traceback formatting code** lacking defensive guards against recursion, cycles, and crashes (H3, H4, H5, H6)
2. **Edge cases in the ops layer** — `_await_exit_suppress` chain loss (H1), `_sync_generator` yielding unawaited coroutines (H2)

All HIGH findings are fixable without architectural changes. The product has strong competitive differentiation (transparent sync/async bridging, traceback enhancement, freeze/decorator patterns) but would benefit significantly from:

- Type stubs / mypy support
- Conditional branching (`.if_()`)
- Retry/timeout primitives
- Universal naming aliases (`.tap()`, `.map()`)

---

## X. DETAILED PER-FILE AUDIT REPORTS

### X.1 `_core.py` — Core Types, Evaluation Primitives, Async Helpers

**File:** `/Users/user/Documents/quent/quent/_core.py` (212 lines)

#### Architecture

- `_Null` / `Null` — Singleton sentinel for "no value provided", distinct from None
- `QuentException` — Public exception type
- `_ControlFlowSignal` / `_Return` / `_Break` — Internal control flow via exceptions
- `_resolve_value` — Resolves a value per calling conventions
- `_handle_break_exc` / `_handle_return_exc` — Handle control flow exceptions
- `Link` — Atomic operation node (linked list node)
- `_evaluate_value` — Core evaluation function dispatching based on link type
- `_ensure_future` / `_task_registry` — Fire-and-forget task scheduling

#### `_evaluate_value` Branch Analysis (All 8 Branches Verified Correct)

| Condition | Args | Action | Result |
|-----------|------|--------|--------|
| `is_chain` + `args[0] is ...` | `(...)` | `v._run(Null, None, None)` | Chain called with no args |
| `is_chain` + args/kwargs | Various | `v._run(args[0] or cv, args[1:], kwargs)` | Chain called with explicit args |
| `is_chain` + no args | None | `v._run(current_value, None, None)` | Chain called with pipeline value |
| Not chain + `args[0] is ...` | `(...)` | `v()` | Callable called with no args |
| Not chain + args/kwargs | Various | `v(*args, **kwargs)` | Callable called with explicit args |
| Not chain + callable + cv≠Null | None | `v(current_value)` | Callable called with pipeline value |
| Not chain + callable + cv=Null | None | `v()` | Callable called with no args |
| Not chain + not callable | None | `v` | Value passed through |

#### `_Null` Pickle Round-Trip Verified

```python
pickle.loads(pickle.dumps(Null)) is Null  # True for all protocols 0-5
```

`__reduce__` returns `'Null'`, which pickle resolves as `getattr(quent._core, 'Null')`.

#### `_ensure_future` Correctness

- `eager_start=True` on Python 3.14 confirmed correct (parameter added to `asyncio.create_task` in 3.14)
- Coroutine properly closed on `RuntimeError` (no running event loop)
- Done callback `_task_registry.discard` correctly auto-removes tasks
- If callback raises, asyncio logs and continues (does not crash loop, does not leak)

---

### X.2 `_chain.py` — Chain and FrozenChain Execution Engine

**File:** `/Users/user/Documents/quent/quent/_chain.py` (482 lines)

#### `_run()` Initial Value Logic — All 8 Combinations Verified Correct

| Case | `has_run_value` | `has_root_value` | First Link | Expected | Result |
|------|----------------|-----------------|------------|----------|--------|
| `Chain(42).run()` | F | T | None | 42 | CORRECT |
| `Chain().then(lambda: 99).run()` | F | F | Present | 99 | CORRECT |
| `Chain(100).then(lambda x: x+1).run(5)` | T | T | Present | 6 | CORRECT |
| `Chain().then(lambda x: x+1).run(5)` | T | T | Present | 6 | CORRECT |
| `Chain().run()` | F | F | None | None | CORRECT |
| `Chain(42).run()` (root only) | F | T | None | 42 | CORRECT |
| `Chain().do(lambda: 'side').run(10)` | T | T | Present | 10 | CORRECT |
| `Chain().do(lambda: 'side').run()` | F | F | Present | None | CORRECT |

#### Sync/Async Bridging Verification

1. When `isawaitable(result)` is True: `ignore_finally = True`, returns `_run_async(...)` coroutine
2. Caller gets a coroutine from what appears to be a sync function
3. `_run_async` has its own complete `try/except/finally` handling
4. Sync `_run`'s `except BaseException` never fires for async exceptions (coroutine doesn't execute until awaited)
5. The bridging is watertight

#### FrozenChain Thread Safety Verification

- All `self.*` accesses in `_run` are READ-ONLY
- Each call creates a NEW local `Link` when `has_run_value=True`
- No shared mutable state is touched during execution
- Exception handling writes to EXCEPTION objects, not to the chain
- Verified by `concurrent_safety_tests.py`

#### `_run_async` Finally Block — Double-Handling is NOT Redundant

Lines 298-304 cover the case where `_evaluate_value` in `_finally_handler_body` returns an awaitable that, when awaited, raises. `_finally_handler_body` itself is sync and cannot await, so it returns the awaitable. The inner try/except in `_run_async` handles exceptions from `await rv`.

#### `on_except_exceptions` Can Never Be `None` When `on_except_link` Is Not `None`

All code paths in `except_()` set `on_except_exceptions` to a non-None tuple before setting `on_except_link`. The invariant is maintained by construction.

---

### X.3 `_ops.py` — Chain Operations

**File:** `/Users/user/Documents/quent/quent/_ops.py` (458 lines)

#### `_make_with` Context Manager Protocol Compliance

| Scenario | Correct? | Notes |
|----------|----------|-------|
| `__enter__` raises | YES | `__exit__` not called (outside try block) |
| Body raises, `__exit__` returns True | YES | Exception suppressed |
| Body raises, `__exit__` raises | YES | `raise exit_exc from exc` |
| Body raises, `__exit__` returns awaitable | YES* | *See H1 for chain loss |
| `async with` + `_ControlFlowSignal` | YES | Stored, CM exits cleanly, then re-raised |
| `hasattr(__aenter__)` check | YES | Async path preferred for dual-protocol objects |

#### Generator Cleanup on Abandonment

When a sync/async generator is abandoned (not fully consumed), Python's GC calls `.close()`, which throws `GeneratorExit`. Since the generators use `try/except` with specific types (`_Break`, `_Return`), `GeneratorExit` propagates normally for proper finalization.

#### `_Generator` Reusability

`__iter__` and `__aiter__` create fresh generator instances on each call, so the same `_Generator` can be iterated multiple times.

#### `_make_foreach` Sync-to-Async Handoff Verification

The `_to_async` function receives a live iterator from the sync path. The ordering is correct:
1. Sync path: `item = next(it)`, `result = fn(item)`, detects `isawaitable(result)`
2. `_to_async` receives: iterator, current item, pending awaitable, accumulated list, current index
3. `_to_async` first awaits the pending result, appends to list, then continues with `next(iterator)`

#### `asyncio.gather` Behavior Documentation

When `return_exceptions=False` (default, used by quent):
- First exception propagates to caller
- Other tasks continue running (NOT cancelled)
- No way to access partial results
- For cancel-on-first-failure, use `asyncio.TaskGroup` (3.11+)

---

### X.4 `_traceback.py` — Traceback Rewriting

**File:** `/Users/user/Documents/quent/quent/_traceback.py` (383 lines)

#### `exec()` Security — SAFE

1. Code executed is always the fixed `'raise __exc__'` — not user-controlled
2. Chain visualization string goes into `co_name` (display metadata) — NOT executed
3. `globals_` dict freshly constructed from trusted values
4. Verified: `co_name = "import os; os.system('...')"` does NOT execute

#### `_clean_internal_frames` Algorithm

1. Collects frames in forward order, keeping `<quent>` synthetic frames and frames outside the quent package
2. Iterates in reverse to build new `TracebackType` objects with proper `tb_next` chaining
3. Result preserves original frame order — verified correct

#### `TracebackException.__init__` Patch Forward Compatibility

The `_patched_te_init` accepts `**kwargs` and passes them through. This correctly handles new keyword arguments added in Python 3.11 (`compact`), 3.13 (`save_exc_type`), and any future additions. Confirmed working on Python 3.14.

#### Subinterpreter Compatibility (Python 3.12+)

`sys.excepthook` and `TracebackException.__init__` patches are installed per-interpreter (subinterpreters have isolated `sys` modules). Correct behavior.

#### `__suppress_context__` — Correctly NOT Checked

`_clean_chained_exceptions` traverses both `__cause__` and `__context__` regardless of `__suppress_context__`. This is correct: `__suppress_context__` controls display, not existence. The chain still exists and should have its frames cleaned.

#### Performance of Patches — Negligible

Both hooks do a single `getattr(exc_value, '__quent__', False)` check. Measured at ~0 additional microseconds on a ~21µs `TracebackException.__init__` call.

---

### X.5 `__init__.py` — Package Init

**File:** `/Users/user/Documents/quent/quent/__init__.py` (8 lines)

#### Import Order

```python
from . import _traceback  # triggers hook installation
from ._chain import Chain
from ._core import Null, QuentException
```

- `_traceback` imported first to install hooks before any chain code runs
- No circular import issues: `_core.py` → `_traceback.py` → `_chain.py` → `__init__.py`
- `_traceback.py` uses `TYPE_CHECKING` guard for the `Chain` import

#### Public API

```python
__all__ = ['Chain', 'Null', 'QuentException']
```

Minimal, clean public surface. All internals are underscore-prefixed.

---

## XI. WEB RESEARCH FINDINGS

### Python 3.14 `asyncio.create_task(eager_start=True)`

- `eager_start` parameter added to `asyncio.Task` constructor in Python 3.12
- `asyncio.create_task()` gained `**kwargs` passthrough in Python 3.14
- When `True` and event loop is running: task starts executing immediately until first block
- If coroutine completes without blocking, skips event loop scheduling entirely
- Version check `sys.version_info >= (3, 14)` in quent is CORRECT

### `inspect.isawaitable()` — Stable Across 3.10-3.14

Checks (in order):
1. `isinstance(object, types.CoroutineType)`
2. `isinstance(object, types.GeneratorType)` with `CO_ITERABLE_COROUTINE` flag
3. `isinstance(object, collections.abc.Awaitable)`

Key edge cases:
- Async generators are NOT awaitable — `isawaitable(async_gen_object)` returns `False`
- The function itself is not awaitable — only the coroutine it returns
- No changes across Python 3.10-3.14

### `asyncio.Task.add_done_callback` Exception Behavior

- Exceptions caught by `Handle._run()` in `asyncio/events.py`
- All except `SystemExit`/`KeyboardInterrupt` are logged to stderr
- Event loop continues running
- Other callbacks on the same task still execute
- Task result unaffected

### `set` Thread Safety

- Under GIL: `set.add()`/`set.discard()` effectively atomic (implementation detail, not guaranteed)
- Free-threaded Python 3.13+: internal critical section locks added
- Python docs: "this should be treated as a description of the current implementation, not a guarantee"
- The `threadsafety.html` page (3.14) documents `list` and `dict` guarantees but NOT `set`

### Fire-and-Forget Task Pattern

The `_task_registry` set with `add_done_callback(registry.discard)` IS the official recommended pattern, appearing verbatim in Python's `asyncio.create_task()` documentation:

```python
background_tasks = set()
for i in range(10):
    task = asyncio.create_task(some_coro(param=i))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
```

### `BaseException.__init__` Skip — Safe

`BaseException.__new__()` sets `self.args` from constructor arguments BEFORE `__init__` is called. Skipping `super().__init__()` has no impact on:
- `str(exc)` / `repr(exc)` — work correctly
- `traceback.format_exception` — works correctly
- `logging.exception()` — works correctly
- Pickle round-trip — works correctly

### `types.TracebackType` Constructor

- Direct construction enabled in Python 3.7 (bpo-30579)
- Signature: `TracebackType(tb_next, tb_frame, tb_lasti, tb_lineno)` — unchanged across 3.7-3.14
- `tb_next` became writable in 3.7

### `code.replace()` Method

- Introduced in Python 3.8
- `co_qualname` parameter added in Python 3.11
- `_HAS_QUALNAME = sys.version_info >= (3, 11)` check is CORRECT

---

*End of review. All findings documented. Ready for implementation.*
