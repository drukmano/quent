# Thread Safety Analysis

## Verdict

**Frozen chains are safe for concurrent use — both across threads and asyncio tasks.** Computation results are always correct. There is one minor caveat around diagnostics.

## Safety Matrix

| Scenario | Correct Results? | Correct Tracebacks? |
|----------|:---:|:---:|
| Frozen chain, multiple threads (GIL) | **YES** | mostly* |
| Frozen chain, multiple asyncio tasks | **YES** | mostly* |
| Unfrozen chain, multiple threads | **NO** — don't do this | **NO** |
| Any chain, free-threaded Python 3.13+ | **UNSAFE** | **UNSAFE** |

## Why It's Safe (for computation)

Every piece of execution state is **per-call**:
- `_ExecCtx` — created fresh each `_run()` call
- `temp_root_link` — created fresh via `_make_temp_link()`
- `link_results` (debug mode) — local variable
- Local iteration state (`lst`, `el`, `result`, iterators) — all stack-local

All Chain and Link fields (`v`, `args`, `kwargs`, `eval_code`, `next_link`, etc.) are **read-only during execution** — they're set during construction and never touched again.

## The One Caveat: `link.temp_args` (diagnostics only)

`_Foreach`, `_Filter`, and `_With` all mutate `self.link.temp_args` during execution — a field on a **shared** Link object. This is used exclusively for traceback display ("which element was being processed when the error occurred"). Under concurrent execution, if two calls error simultaneously, one traceback might show the other's element.

**Impact: cosmetic only.** Computation results, exception types, and control flow are all correct. Only the "context" shown in error tracebacks may be wrong under concurrent error conditions.

### Affected code locations

- `_iteration.pxi` — `_Foreach.__call__` lines 55, 66
- `_iteration.pxi` — `_Filter.__call__` lines 168, 175
- `_control_flow.pxi` — `_With.__call__` line 81
- `_control_flow.pxi` — `_with_full_async` line 140

## Free-threaded Python 3.13+ (no GIL)

Without the GIL, the `temp_args` write becomes a potential torn pointer — this could cause memory corruption, not just wrong diagnostics. The `task_registry` global set (`_async_utils.pxi` line 22) also becomes unsafe under concurrent `set.add()`/`set.discard()` without synchronization.

### Required work for free-threaded support
- Add a lock around `task_registry` mutations, or switch to a thread-safe container
- Either make `temp_args` per-execution (store in `_ExecCtx` instead of on shared `Link`), or add per-field locking
- Audit any Cython cdef class attribute stores for atomicity guarantees

## Shared Mutable State Inventory

### Module-level globals (immutable after init — SAFE)
- `Null`, `EMPTY_TUPLE`, `EMPTY_DICT` — sentinels
- `_PyCoroType`, `_CyCoroType` — type caches
- Function aliases (`_async_foreach_fn`, etc.) — bound once
- `sys.excepthook` — set once at import

### Module-level globals (mutable — requires care)
- `task_registry` (set) — safe under GIL, unsafe without
- `_registry_warned` (bool) — benign race at worst

### Chain object fields — ALL read-only during execution
- `root_link`, `first_link`, `on_finally_link`, `current_link`
- `is_cascade`, `_autorun`, `is_nested`, `_debug`, `_is_simple`, `_is_sync`

### Link object fields — ALL read-only during execution EXCEPT `temp_args`
- Read-only: `v`, `original_value`, `exceptions`, `next_link`, `args`, `kwargs`, `is_with_root`, `ignore_result`, `is_chain`, `is_exception_handler`, `reraise`, `eval_code`, `fn_name`
- **Mutable during execution: `temp_args`** (diagnostics only)

### Per-execution state (SAFE — never shared)
- `_ExecCtx` instances — created fresh per `_run()` call
- `temp_root_link` — created fresh per `_run()` call
- `link_results` dict — local variable in `_run()`
- Iteration locals (`lst`, `el`, `result`, iterators)
- Exception attributes (`__quent_link_temp_args__`, `__quent_source_link__`, `__quent__`)

## Recommendations

1. **Current usage (CPython with GIL):** Frozen chains are safe to share across threads and asyncio tasks. Construct chains in a single thread, freeze them, then share freely.
2. **Do NOT share unfrozen Chain instances across threads.** Construction methods (`_then`, `config`, `except_`, `finally_`) mutate Chain fields without synchronization.
3. **For independent copies:** Use `chain.clone()` to create per-thread copies if needed.
4. **For true snapshots:** Use `chain.clone().freeze()` — `freeze()` alone shares the execution engine with the original chain.
