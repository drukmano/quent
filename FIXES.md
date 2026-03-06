# Quent Codebase Review

## CRITICAL — Bugs / Correctness

### 1. `_evaluate_value` drops kwargs for nested chains when args is empty
**`_core.py:191`**

When a link wraps a nested chain and has kwargs but no positional args, kwargs are silently discarded:

```python
if link.is_chain:
    if args and args[0] is ...:
      return v._run(Null, None, None)
    if args:                            # empty tuple -> falsy -> skip
      return v._run(args[0], args[1:], kwargs or {})
    return v._run(current_value, None, None)  # <- kwargs DROPPED
```

`Chain(5).then(nested_chain, k=99).run()` silently loses `k=99`. The non-chain path (`_core.py:195-196`) correctly handles kwargs-only via `if args or kwargs:`. The chain path does not.

**Fix:** Add a `kwargs` check: `if args or kwargs:` instead of `if args:` on line 189, with appropriate argument forwarding.

### 2. `set_initial_values` ignores `ignore_result`, leaking `do()` results
**`_chain.py:152-159`**

The one-shot initialization block sets `current_value = result` unconditionally when `current_value is Null` -- *before* the `ignore_result` check:

```python
if not set_initial_values:
    set_initial_values = True
    if has_root_value and root_value is Null:
        root_value = result
    if current_value is Null:
        current_value = result      # <- ignores link.ignore_result
if not link.ignore_result:
    current_value = result
```

`Chain().do(lambda: 'side_effect').run()` returns `'side_effect'` instead of `None`. The `do()` docstring says "The result is discarded" -- this edge case violates that contract when `do()` is the first link with no root value.

**Fix:** Guard the initial `current_value` assignment: `if current_value is Null and not link.ignore_result:`.

---

## MODERATE — Meaningful Improvements

### 3. `_FrozenChain` doesn't actually freeze anything
**`_chain.py:453-472`**

`freeze()` stores a reference to the *same* Chain -- no copy, no locking, no mutation prevention. Mutations after freeze are visible through the frozen wrapper. The docstring says "The chain must not be modified after freezing" and CLAUDE.md says "_FrozenChain is safe for concurrent use" -- both rely entirely on user discipline. Adversarial tests confirm mutation-after-freeze silently works.

**Suggestion:** Either (a) shallow-copy the chain's critical fields at freeze time, or (b) set a flag on the Chain that causes `.then()` / `.except_()` / `.finally_()` to raise after freeze.

### 4. `StopIteration` from user `fn` silently swallowed in `foreach`
**`_ops.py:316-342`**

`_foreach_op` uses `while True` / `next(it)` / `except StopIteration`. If the user's `fn(item)` raises `StopIteration` (e.g., from calling `next()` on an exhausted iterator), it's caught by the same handler, silently truncating results with no error. This is the exact footgun PEP 479 was created to prevent.

### 5. `StopIteration` behavior inconsistent between `foreach` and `iterate`
**`_ops.py:128-161, 252-348`**

In `foreach`: `StopIteration` from `fn` silently terminates the loop (Finding 4). In `iterate` (a generator function): PEP 479 converts `StopIteration` from `fn` into `RuntimeError`. The same function + data produces different behavior depending on which method was used.

**Suggestion:** Separate the `fn(item)` call from the `next(it)` call in exception handling to distinguish user-raised `StopIteration` from iterator exhaustion.

### 6. `_modify_traceback` uses `sys.exc_info()[1]` instead of its `exc` parameter
**`_traceback.py:105`**

The `exc` parameter is the exception to modify, but line 105 re-fetches the active exception via `sys.exc_info()[1]`. Currently all call sites invoke from inside the matching `except` block, so they're the same object. But this is fragile -- any future call site outside its `except` block would graft the traceback onto the wrong exception.

**Fix:** Replace `exc_value = sys.exc_info()[1]` with `exc_value = exc`. One-line change, eliminates a class of future bugs.

### 7. `_Null` sentinel breaks under pickle/copy/re-instantiation
**`_core.py:12-21`**

`pickle.loads(pickle.dumps(Null)) is Null` -> `False`. `copy.copy(Null) is Null` -> `False`. `_Null()` creates a second instance. All `is Null` checks throughout the codebase would fail with these copies.

**Fix:** Add `__reduce__`, `__copy__`, `__deepcopy__` methods returning the singleton.

### 8. `__quent_link_temp_args__` never cleaned from exceptions
**`_traceback.py:89-93`**

`_modify_traceback` deletes `__quent_source_link__` after processing (line 89) but never deletes `__quent_link_temp_args__`. This attribute holds references to user values (`current_value`, items, context manager objects) that are retained for the lifetime of the exception, delaying GC.

**Fix:** Add cleanup: `if hasattr(exc, '__quent_link_temp_args__'): del exc.__quent_link_temp_args__`

### 9. `_run_async` missing `__quent_source_link__` stamp
**`_chain.py:272-281`**

`_run` explicitly stamps `exc.__quent_source_link__ = link` (lines 178-179) *before* calling `_except_handler_body`. `_run_async` omits this -- the stamp happens later inside `_modify_traceback`. The ordering difference means the first-write-wins semantics diverge between sync and async paths, potentially causing incorrect link highlighting in async traceback visualization.

**Fix:** Add the same guard to `_run_async`'s `BaseException` handler.

### 10. `except_()` doesn't validate exception types at registration time
**`_chain.py:341-364`**

`except_(fn, exceptions=42)` is accepted at registration time (42 is not iterable, falls to `else: self.on_except_exceptions = (42,)`). At runtime, `isinstance(exc, (42,))` raises an opaque `TypeError`. Fail-fast with a clear message at registration would be much better.

**Fix:** Validate each element is a `type` and `issubclass(exc_type, BaseException)` in `except_()`.

### 11. No size limit on traceback visualization
**`_traceback.py:170-185, 95-97`**

`_get_obj_name` calls `repr(obj)` on arbitrary user objects with no truncation. An object with a massive `repr` produces a massive `co_name` in the synthetic code object. Deeply nested chains with many links amplify this.

**Suggestion:** Truncate `repr` output (e.g., 200 chars) and cap nesting depth in visualization.

### 12. `_ensure_future` doesn't close coroutine on `RuntimeError`
**`_core.py:119-125`**

If `_create_task_fn(coro)` raises `RuntimeError` (no event loop), the coroutine is not closed within `_ensure_future`. Call sites in `_chain.py` mitigate this, but any new call site would leak.

**Fix:** Add try/except within `_ensure_future` to close the coroutine on failure.

### 13. Three-tier sync/async duplication across `foreach`/`filter`
**`_ops.py:252-417`**

`_make_foreach` and `_make_filter` share nearly identical structure (sync fast path, `_to_async` handoff, `_full_async` path) with the same iteration pattern, `isawaitable` checks, and exception handling. Any bugfix must be applied in 6+ places. This is the biggest maintainability concern in the codebase.

---

## MINOR — Style / Polish

| # | File:Line | Finding |
|---|-----------|---------|
| 14 | `_core.py:41-43` | Comment about skipping `super().__init__()` claims to avoid args tuple overhead -- incorrect, `BaseException.__new__` builds it regardless |
| 15 | `_core.py:129-138` | Link docstring lists `temp_args` slot that doesn't exist |
| 16 | `_core.py:39` | `args_` naming confusable with `Exception.args` -- consider `call_args` |
| 17 | `_core.py:165` | `original_value: Any | None` is redundant (`Any` includes `None`) |
| 18 | `_chain.py:318,330` | Two TODOs asking "is that even possible?" -- analysis confirms these are unreachable under normal operation; guards are defensive safety nets |
| 19 | `_chain.py:308` | `decorator()` return type should be `Callable[[Callable[..., Any]], Callable[..., Any]]` not `Callable[..., Callable[..., Any]]` |
| 20 | `_chain.py:212` | `result` variable shadowed in `finally` block -- use `finally_result` |
| 21 | `_chain.py:172` | `_Break` error message exposes internal class name -- should say `Chain.break_() cannot be used outside of a foreach iteration` |
| 22 | `_chain.py:237` | `_run_async` parameter `root_link: Link | None = None` default is never used (always passed explicitly) |
| 23 | `_ops.py:91` | Dual-protocol objects (both `__enter__` and `__aenter__`) always take async path -- documented but could surprise users |
| 24 | `_ops.py:447-449` | Coroutine cleanup uses `hasattr(r, 'close')` which is too broad -- matches file objects, generators, etc. Use `asyncio.iscoroutine()` |
| 25 | `_ops.py:237-240` | `_Generator.__call__` returns new instance -- non-obvious, needs docstring |
| 26 | `_traceback.py:291` | `link` parameter reassigned to `link.original_value` -- confusing shadow |
| 27 | `_traceback.py:369` | `_patched_te_init` uses `exc_tb` parameter name vs original's `exc_traceback` |
| 28 | `_traceback.py:126-145` | `_get_true_source_link` can infinite-loop on circular chain references -- add a `seen` set |
| 29 | `_core.py:116` | `_task_registry` is not thread-safe for free-threaded Python (PEP 703) |

---

## IDEAS — Feature Enhancements

| Idea | Description |
|------|-------------|
| **`break_()` in filter** | `_make_filter` doesn't handle `_Break` -- users can't early-terminate filtering. Would parallel `foreach` behavior. |
| **Chain composition** | No way to concatenate chains (`chain_a.pipe(chain_b)` or `chain_a + chain_b`). Users must nest. |
| **`if_()` / conditional** | Branching requires lambdas. A `.if_(predicate, then_fn, else_fn)` would be more fluent. |
| **`reduce()` / `fold()`** | Natural companion to `foreach` and `filter` for accumulation. |
| **`retry()` / `timeout()`** | Common async pipeline patterns that fit naturally. |
| **Aliases** | `map` for `foreach`, `tap` for `do` -- improves discoverability for FP users. |
| **`Link.__repr__`** | Currently shows unhelpful `<Link object at 0x...>`. A simple repr showing `v`, args, kwargs would aid debugging. |
| **Chain introspection** | No programmatic way to inspect chain structure (list links, get length, check if frozen). |

---

## Top 5 Highest-Impact Recommendations

1. **Fix kwargs drop in `_evaluate_value`** (Critical #1) -- silent data loss for nested chains with keyword args
2. **Fix `set_initial_values` / `ignore_result`** (Critical #2) -- `do()` leaks results when it's the first link
3. **Replace `sys.exc_info()[1]` with `exc`** (Moderate #6) -- one-line fix, eliminates fragile indirection
4. **Validate exception types in `except_()`** (Moderate #10) -- fail-fast instead of opaque runtime error
5. **Enforce `freeze()` immutability** (Moderate #3) -- make the "safe for concurrent use" claim actually hold
