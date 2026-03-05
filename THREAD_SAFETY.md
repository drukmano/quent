# Thread Safety Analysis

## Verdict

**Frozen chains are safe for concurrent use -- both across threads and asyncio tasks.** Computation results are always correct. Diagnostics are also safe thanks to per-exception state storage.

## Safety Matrix

| Scenario | Correct Results? | Correct Tracebacks? |
|----------|:---:|:---:|
| Frozen chain, multiple threads (GIL) | **YES** | **YES** |
| Frozen chain, multiple asyncio tasks | **YES** | **YES** |
| Unfrozen chain, multiple threads | **NO** -- don't do this | **NO** |
| Any chain, free-threaded Python 3.13+ | see below | see below |

## Why It's Safe (for computation)

Every piece of execution state is **per-call**:
- Local variables in `_run()` / `_run_async()` (`_chain.py`): `link`, `root_link`, `current_value`, `root_value`, `has_run_value`, `has_root_value`, `ignore_finally`, `set_initial_values`
- Temporary `Link` created when `run(v)` is called with a value -- stack-local
- Local iteration state in `_ops.py` closures (`lst`, `item`, `result`, iterators) -- all stack-local

All Chain and Link fields are **read-only during execution** -- they are set during construction and never touched again.

## Diagnostics: `_set_link_temp_args` (safe)

In the pure Python implementation, `_set_link_temp_args()` (`_core.py`) writes diagnostic information to the **exception object** itself, as `exc.__quent_link_temp_args__`. Since each exception instance is unique per error, this is inherently per-call. There is no mutation of shared Link or Chain state during execution.

### How it works

When a `_foreach_op`, `_filter_op`, or `_with_op` closure (in `_ops.py`) catches an exception, it calls:

```python
_set_link_temp_args(exc, link, item)
```

This writes to `exc.__quent_link_temp_args__[id(link)]`, where `exc` is the per-call exception object. Multiple concurrent executions catching different exceptions write to different objects -- no contention.

### Affected code locations

- `_ops.py` -- `_foreach_op` sync/async paths (lines ~203, 227, 253)
- `_ops.py` -- `_filter_op` sync/async paths (lines ~281, 298, 320)
- `_ops.py` -- `_with_op` sync/async paths
- `_core.py` -- `_set_link_temp_args()` (line ~94)

## Free-threaded Python 3.13+ (no GIL)

Without the GIL, additional concerns arise:

- **Link/Chain fields** are still read-only during execution, so they remain safe as long as construction is single-threaded and a happens-before relationship exists before sharing.
- **`_task_registry`** (`_core.py`, line ~116) -- a module-level `set[asyncio.Task]` mutated by `_ensure_future()`. `set.add()` and `set.discard()` are not atomic without the GIL. This needs synchronization.
- **`_set_link_temp_args`** -- writes to the exception object. Since exceptions are per-call and not shared across threads, this is safe even without the GIL.

### Required work for free-threaded support
- Add a lock around `_task_registry` mutations, or switch to a thread-safe container
- Audit attribute stores on exception objects (`__quent_source_link__`, `__quent_link_temp_args__`, `__quent__`) for atomicity -- these are all per-exception-instance writes, but CPython's free-threaded mode may need care for dict-based attribute access

## Shared Mutable State Inventory

### Module-level globals (immutable after init -- SAFE)
- `Null` -- singleton sentinel (`_core.py`)
- `_create_task_fn` -- bound once at import (`_core.py`)
- `_RAISE_CODE`, `_HAS_QUALNAME`, `_TracebackType`, `_quent_file` -- set once at import (`_traceback.py`)
- `sys.excepthook` -- set once at import (`_traceback.py`)

### Module-level globals (mutable -- requires care)
- `_task_registry` (set) (`_core.py`, line ~116) -- safe under GIL, unsafe without

### Chain object fields -- ALL read-only during execution
- `current_link`, `first_link`, `is_nested`
- `on_except_exceptions`, `on_except_link`, `on_finally_link`
- `root_link`

### Link object fields -- ALL read-only during execution
- `v`, `next_link`, `ignore_result`, `args`, `kwargs`, `original_value`, `is_chain`

### Per-execution state (SAFE -- never shared)
- Local variables in `_run()`: `link`, `root_link`, `current_value`, `root_value`, `has_run_value`, `has_root_value`, `ignore_finally`, `set_initial_values`
- Local variables in `_run_async()`: same set, received as parameters
- Temporary `Link` for `run(v)` calls -- created on the stack
- Iteration locals in `_ops.py` closures (`lst`, `item`, `result`, iterators)
- Exception attributes (`__quent_link_temp_args__`, `__quent_source_link__`, `__quent__`) -- per-exception-instance

## Recommendations

1. **Current usage (CPython with GIL):** Frozen chains are safe to share across threads and asyncio tasks. Construct chains in a single thread, freeze them, then share freely.
2. **Do NOT share unfrozen Chain instances across threads.** Construction methods (`then`, `do`, `except_`, `finally_`, etc.) mutate Chain fields without synchronization.
3. **For reusable pipelines:** Use `chain.freeze()` to create an immutable `_FrozenChain` snapshot. Frozen chains are safe for concurrent use because all execution state is per-call.
