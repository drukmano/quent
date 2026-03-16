---
title: "Error Handling -- except, finally, Tracebacks"
description: "Handle errors in quent pipelines with except_ and finally_. Covers exception handlers, cleanup, the separate except calling convention, and enhanced tracebacks."
tags:
  - error handling
  - exception
  - traceback
search:
  boost: 5
---

# Error Handling

quent provides two mechanisms for dealing with errors in pipelines: **`.except_()`** for catching exceptions and **`.finally_()`** for cleanup. Each can be registered **at most once** per chain -- a second registration raises `QuentException`.

Both work transparently with sync and async callables. For per-step error handling, use nested chains -- each nested chain gets its own except/finally pair.

---

## Design: One Handler Per Chain

Chains support exactly one exception handler and one cleanup handler each. This constraint keeps the execution model simple and predictable: you always know which handler runs, in what order, and what happens to the exception.

For per-step error handling, compose nested chains:

```python
from quent import Chain

# Each nested chain gets its own except_
step_with_fallback = (
  Chain()
  .then(risky_operation)
  .except_(lambda ei: 'fallback')
)

pipeline = (
  Chain()
  .then(step_with_fallback)  # has its own error handling
  .then(next_step)
  .except_(handle_pipeline_error)  # separate handler for the outer chain
)
```

---

## except\_ -- Exception Handling

### Signature

```python
.except_(fn, /, *args, exceptions=None, reraise=False, **kwargs)
```

Register an exception handler for the chain. When an exception occurs during chain execution, the handler is called with the exception and (optionally) additional arguments.

### Basic Usage

```python
from quent import Chain, ChainExcInfo

def handle_error(ei: ChainExcInfo):
  print(f"Error: {ei.exc}")
  return "fallback value"

result = (
  Chain(data)
  .then(process)
  .then(save)
  .except_(handle_error)
  .run()
)
# If process() or save() raises, handle_error receives ChainExcInfo(exc, root_value).
# The handler's return value ("fallback value") becomes the chain's result.
```

### What except\_ Catches

By default, `.except_()` catches `Exception` -- not `BaseException`. This follows Python convention: system signals like `KeyboardInterrupt` and `SystemExit` propagate without interference.

```python
# Default: catches Exception and its subclasses
.except_(handler)

# Catch specific exception types
.except_(handler, exceptions=ValueError)

# Catch multiple exception types
.except_(handler, exceptions=(ConnectionError, TimeoutError))
```

The `exceptions` parameter accepts a single exception type or an iterable of types.

**Validation at registration time:**

- An empty iterable raises `QuentException`.
- Non-`BaseException` subclasses raise `TypeError`.
- String values raise `TypeError` (common mistake: passing `"ValueError"` instead of `ValueError`).

!!! warning
    If you configure `except_()` to catch a `BaseException` subclass that is not an `Exception` subclass (e.g., `KeyboardInterrupt`), quent emits a `RuntimeWarning` advising you to consider `Exception` instead. Catching system signals can suppress critical shutdown behavior.

---

## Except Handler Calling Convention

!!! important
    The `except_()` handler uses the **standard 2-rule calling convention**. Its current value is a `ChainExcInfo(exc, root_value)` NamedTuple.

### How It Works

The handler receives a `ChainExcInfo` as its current value. The standard 2-rule dispatch then applies:

| Registration | Handler invocation |
|-------------|-------------------|
| `except_(handler)` | `handler(ChainExcInfo(exc, root_value))` |
| `except_(handler, arg1, arg2)` | `handler(arg1, arg2)` -- ChainExcInfo NOT passed |
| `except_(handler, key=val)` | `handler(key=val)` -- ChainExcInfo NOT passed |

Access the exception via `.exc` and the root value via `.root_value` on the `ChainExcInfo` NamedTuple. The `root_value` is normalized to `None` if the chain was created with no root value.

```python
from quent import Chain, ChainExcInfo

def handle_error(ei: ChainExcInfo):
  print(f"Failed on input {ei.root_value}: {ei.exc}")
  return "default"

.except_(handle_error)
# calls: handle_error(ChainExcInfo(exc, root_value))
```

---

## Consumption vs Re-raise

The `reraise` parameter controls what happens after the handler runs:

### `reraise=False` (default) -- Consume the Exception

The handler's return value **replaces** the chain's result. The exception is consumed -- it does not propagate. The chain is considered to have succeeded.

```python
from quent import Chain

result = (
  Chain(42)
  .then(lambda x: x / 0)
  .except_(lambda ei: f"failed on {ei.root_value}: {ei.exc}")
  .run()
)
# No exception propagates
# result = the handler's return value
```

### `reraise=True` -- Re-raise After Handler

The handler runs for **side-effects only** (e.g., logging, alerting). After the handler completes, the **original exception** is re-raised. The handler's return value is ignored.

```python
from quent import Chain

def notify_admin(ei):
  send_alert(f"Pipeline failed: {ei.exc}")

result = (
  Chain(data)
  .then(process)
  .except_(notify_admin, reraise=True)
  .run()
)
# notify_admin is called with ChainExcInfo, then the original exception is re-raised.
```

---

## Handler Failure

### With `reraise=True`

When the handler itself raises while `reraise=True`:

- **`Exception` subclass:** The handler's exception is **discarded**. A `RuntimeWarning` is emitted, a note is attached to the original exception (Python 3.11+), and the **original exception** is re-raised. The caller always sees the original failure.
- **`BaseException` subclass** (e.g., `KeyboardInterrupt`): The handler's exception propagates naturally -- system signals are never suppressed.

### With `reraise=False`

When the handler raises while `reraise=False`, the handler's exception propagates. The original chain exception is set as `__cause__` (via `raise handler_exc from original_exc`).

---

## finally\_ -- Cleanup

### Signature

```python
.finally_(fn, /, *args, **kwargs)
```

Register a cleanup handler that **always runs**, whether the chain succeeds or fails.

### Key Behaviors

1. **Always runs** -- on both success and failure paths, matching Python's `try/finally` semantics.
2. **Receives the root value** -- the chain's original input, not the current intermediate value. Normalized to `None` if no root value exists.
3. **Return value is always discarded** -- the finally handler cannot alter the chain's result.
4. **Follows the standard calling convention** -- not the except handler's special convention.

```python
from quent import Chain

def cleanup(original_input):
  print(f"Cleaning up for input: {original_input}")
  release_resources()

result = (
  Chain(resource_id)
  .then(acquire_resource)
  .then(process_resource)
  .finally_(cleanup)
  .run()
)
# cleanup always runs, receiving resource_id (the root value).
# Its return value is discarded.
```

### Calling Conventions

The finally handler follows the **standard** calling conventions (2 rules), with the root value (normalized to `None` if absent) as the current value:

```python
# handler(root_value) -- default
.finally_(cleanup)

# handler(arg1) -- explicit args override root value
.finally_(cleanup, some_resource)
```

### Finally Handler Failure

- **Exception already active:** The finally handler's exception **replaces** the original exception (matching Python's `try/finally` behavior). The original is preserved as `__context__`.
- **Success path:** The finally handler's exception propagates as the chain's error.

### Async Finally in Sync Chains

When a sync chain's finally handler returns a coroutine, the coroutine is scheduled as a **fire-and-forget background task** with a `RuntimeWarning`. If no event loop is running, a `QuentException` is raised and the cleanup does not execute.

See [Async Handling -- Fire-and-Forget Tasks](async.md#fire-and-forget-tasks) for details.

---

## Control Flow Restrictions

Using `Chain.return_()` or `Chain.break_()` inside an `except_()` or `finally_()` handler raises `QuentException`:

```python
from quent import Chain

def bad_handler(ei):
  return Chain.return_("value")  # raises QuentException

Chain(data).then(process).except_(bad_handler).run()
# QuentException: Using control flow signals inside except handlers is not allowed.
```

Control flow signals are not allowed in error or cleanup handlers -- they must be used in the main pipeline.

---

## Execution Order

The full error handling flow:

```
chain executes
  |
  +--> step raises exception:
  |      |
  |      +--> exception matches `exceptions`?
  |      |      |
  |      |      +--> YES: except_ handler runs
  |      |      |      |
  |      |      |      +--> reraise=False: handler result = chain result
  |      |      |      |    finally_ runs in success context
  |      |      |      |
  |      |      |      +--> reraise=True: handler runs for side-effects
  |      |      |           original exception re-raised
  |      |      |           finally_ runs in failure context
  |      |      |
  |      |      +--> NO: exception propagates
  |      |           finally_ runs in failure context
  |      |
  +--> success:
         |
         +--> finally_ runs in success context
```

The order of registration (`.except_()` vs `.finally_()`) does not matter -- they always execute in the order above.

---

## ExceptionGroup from Concurrent Operations

When concurrent operations (`gather()`, or `foreach()`/`foreach_do()` with `concurrency`) encounter **multiple** failures, the exceptions are wrapped in an `ExceptionGroup`:

```python
try:
  await Chain(data).gather(validate, enrich, score).run()
except ExceptionGroup as eg:
  for exc in eg.exceptions:
    print(exc)
```

- **Python 3.11+:** Uses the builtin `ExceptionGroup`.
- **Python 3.10:** Uses a polyfill that implements `.exceptions`, `.subgroup()`, `.split()`, and `.derive()`.

When **one** function/worker fails, the exception propagates directly (not wrapped in an `ExceptionGroup`).

The `ExceptionGroup` message indicates the source and count:

- `"gather() encountered 3 exceptions"`
- `"foreach() encountered 2 exceptions"`

---

## Enhanced Tracebacks

When an exception occurs inside a chain, quent automatically injects a visualization into the traceback showing the full pipeline and marking the failing step.

### Example Output

```
Traceback (most recent call last):
  File "example.py", line 28, in <module>
    .run()
     ^^^^^
  File "<quent>", line 1, in
    Chain(fetch_data, 42) = {'id': 42, 'value': 100}
    .then(validate) <----
    .then(transform)
    .then(save)
  File "example.py", line 11, in validate
    raise ValueError("Value too large")
ValueError: Value too large
```

### What quent Does

1. **Chain visualization:** A synthetic `<quent>` frame is injected showing every step. The `<----` marker identifies the failing step.
2. **Frame cleaning:** Internal quent frames are removed. You see only your code and the `<quent>` visualization.
3. **Exception notes** (Python 3.11+): A one-line note is attached via `add_note()`, surviving traceback reformatting.
4. **Chained exceptions:** Frames are cleaned from `__cause__`, `__context__`, and `ExceptionGroup` sub-exceptions.

### Nested Chain Visualization

Nested chains are rendered with indentation:

```
Chain(fetch)
    .then(
        Chain(parse)
            .then(validate) <----
    )
    .do(log)
```

### First-Write-Wins Error Marker

The `<----` marker always points to the **innermost** (deepest) failing step. When an exception propagates through nested chains, only the original failure point is recorded.

### Controlling Tracebacks

#### `QUENT_NO_TRACEBACK=1`

Disables all traceback modifications. Must be set **before** importing quent:

```bash
QUENT_NO_TRACEBACK=1 python my_script.py
```

#### `QUENT_TRACEBACK_VALUES=0`

Suppresses argument values in visualizations while preserving step names and structure. Useful in production to prevent sensitive data from leaking into logs:

```bash
QUENT_TRACEBACK_VALUES=0 python my_script.py
```

#### `uninstall_hooks()`

Restores `sys.excepthook` and `TracebackException.__init__` to their pre-quent values at runtime:

```python
from quent import uninstall_hooks

uninstall_hooks()  # idempotent, thread-safe
```

---

## Nested Chains for Per-Step Error Handling

Since each chain allows only one `except_()`, use nested chains for granular error handling:

```python
from quent import Chain

# Step with its own error recovery
fetch_with_fallback = (
  Chain()
  .then(fetch_from_primary)
  .except_(lambda ei: fetch_from_backup())
)

# Step with its own error recovery
save_with_retry = (
  Chain()
  .then(save_to_database)
  .except_(lambda ei: retry_save(ei.root_value), reraise=True)
)

pipeline = (
  Chain()
  .then(fetch_with_fallback)
  .then(validate)
  .then(save_with_retry)
  .except_(handle_pipeline_error)  # catches anything not handled above
  .finally_(cleanup)
)
```

Each nested chain's handlers apply only to that nested chain's execution. Unhandled exceptions propagate to the outer chain.

---

## Practical Patterns

### Logging and Re-raising

```python
from quent import Chain

pipeline = (
  Chain()
  .then(fetch)
  .then(process)
  .except_(lambda ei: logger.error(f"Pipeline failed: {ei.exc}"), reraise=True)
  .finally_(lambda rv: logger.info(f"Pipeline completed for: {rv}"))
)
```

### Fallback Values

```python
from quent import Chain

pipeline = (
  Chain()
  .then(fetch_from_cache)
  .except_(lambda ei: fetch_from_database())
)
```

### Selective Exception Handling

```python
from quent import Chain

pipeline = (
  Chain()
  .then(fetch)
  .then(process)
  .except_(
    lambda ei: {'error': str(ei.exc), 'input': ei.root_value},
    exceptions=(ConnectionError, TimeoutError)
  )
  .finally_(cleanup)
)
# Only ConnectionError and TimeoutError are caught.
# All other exceptions propagate directly.
```

---

## Summary

| Feature | Limit | Default catches | Handler receives | Return value |
|---------|-------|-----------------|------------------|--------------|
| `.except_()` | 1 per chain | `Exception` | Exception as first arg, then handler-specific rules | Replaces chain result (or re-raises with `reraise=True`) |
| `.finally_()` | 1 per chain | N/A (always runs) | Root value (standard calling convention) | Always discarded |

---

## Next Steps

- **[Chains](chains.md)** -- pipeline building, context managers, conditionals, and control flow
- **[Async Handling](async.md)** -- sync/async bridging, fire-and-forget patterns
- **[Reuse and Patterns](reuse.md)** -- cloning, nesting, decorators, and composition
