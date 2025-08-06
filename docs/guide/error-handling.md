# Error Handling

Quent provides structured error handling within chains via `.except_()` and `.finally_()`, along with enhanced stack traces that show the full chain state when an error occurs.

## except_

The `.except_()` method registers an exception handler that is called when any operation in the chain raises an exception.

```python
from quent import Chain

result = (
  Chain(risky_operation)
  .then(process_result)
  .except_(handle_error)       # called on exception
  .run()
)
```

The handler function receives the exception as its argument.

### Parameters

```python
.except_(fn, *args, exceptions=None, reraise=True, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Handler function. Receives the exception as its first argument. |
| `*args` | `Any` | Additional arguments passed to the handler |
| `exceptions` | `tuple` or `type` or `None` | Exception types to catch. `None` catches all exceptions. |
| `reraise` | `bool` | Whether to re-raise the exception after handling. Default: `True`. |
| `**kwargs` | `Any` | Additional keyword arguments passed to the handler |

### Re-raising vs Swallowing

By default (`reraise=True`), the exception is re-raised after the handler runs. This is useful for logging or cleanup that should not suppress the error.

```python
# Log the error but still raise it
chain.except_(lambda e: logger.error(f"Failed: {e}"), reraise=True)
```

Set `reraise=False` to swallow the exception. The handler's return value becomes the chain's result.

```python
# Return a default value on failure
chain.except_(lambda e: default_value, reraise=False)
```

### Exception Type Filtering

Filter handlers by exception type so that different exceptions are handled differently:

```python
result = (
  Chain(risky_operation)
  .except_(
    handle_value_error,
    exceptions=ValueError,
    reraise=False  # swallow ValueError
  )
  .except_(
    handle_type_error,
    exceptions=TypeError
    # reraise=True by default -- re-raises after handling
  )
  .except_(
    handle_any  # catch-all for anything else
  )
  .run()
)
```

Handlers are checked in order. The first handler whose `exceptions` filter matches (or that has no filter) handles the exception.

### Multiple Handlers

You can register multiple `.except_()` handlers. They are evaluated in registration order:

```python
chain = (
  Chain(operation)
  .except_(handle_value_error, exceptions=ValueError)
  .except_(handle_type_error, exceptions=TypeError)
  .except_(handle_any)  # catch-all
)
```

## suppress

The `.suppress()` method is a shorthand for suppressing specific exception types without a handler function:

```python
from quent import Chain

# Suppress FileNotFoundError -- chain returns None if it occurs
result = Chain(read_file, path).suppress(FileNotFoundError).run()
```

## finally_

The `.finally_()` method registers a cleanup handler that **always runs**, regardless of whether the chain succeeded or raised an exception.

```python
result = (
  Chain(risky_operation)
  .then(process_result)
  .except_(handle_error)
  .finally_(cleanup, ...)    # always called
  .run()
)
```

### Parameters

```python
.finally_(fn, *args, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Cleanup function |
| `*args` | `Any` | Arguments passed to the cleanup function |
| `**kwargs` | `Any` | Keyword arguments passed to the cleanup function |

The cleanup function receives the current value by default. Use the ellipsis (`...`) as the first argument to call it with no arguments:

```python
# cleanup() called with no arguments
Chain(data).then(process).finally_(cleanup, ...)

# cleanup(current_value) called with the current chain value
Chain(data).then(process).finally_(cleanup)
```

## on_success

The `.on_success()` method registers a handler that runs only when the chain completes **without** raising an exception:

```python
result = (
  Chain(operation)
  .then(process)
  .on_success(notify_completion)
  .except_(handle_error)
  .run()
)
```

## Enhanced Stack Traces

When an error occurs inside a chain, Quent attaches a visualization of the chain state to the exception using Python's `add_note()` mechanism. This shows the entire chain structure, intermediate values, and the exact operation that failed.

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

The trace shows:

- **The entire chain structure** -- every operation in the chain is listed
- **Intermediate values** -- e.g., `= {'id': 42, 'value': 100}` shows the result of the root operation
- **The exact operation that failed** -- marked with `<----`
- **The original exception** and its source location

This makes debugging chains significantly easier because you can see:

1. What the chain was doing (the full pipeline)
2. What value was being processed (intermediate results)
3. Exactly which step failed

### Debug Mode

Enable debug mode to log chain execution details and record intermediate results on each link:

```python
import logging
logging.getLogger("quent").setLevel(logging.DEBUG)

Chain(operation).config(debug=True).run()
```

In debug mode, intermediate results are recorded on each link for inspection in stack traces. This adds some overhead but provides complete visibility into chain execution.

## Combining Error Handling with Resilience

Error handling composes naturally with [retry](resilience.md) and [timeout](resilience.md):

```python
result = (
  Chain(call_api, request)
  .retry(3, delay=0.5, exceptions=(ConnectionError,))
  .timeout(10.0)
  .then(parse_response)
  .except_(log_api_error, reraise=False)
  .finally_(cleanup_connection, ...)
  .run()
)
```

In this example:

1. `call_api` is retried up to 3 times on `ConnectionError`
2. The entire chain has a 10-second timeout
3. `parse_response` processes the successful result
4. If any unhandled exception occurs, `log_api_error` handles it and suppresses re-raising
5. `cleanup_connection` always runs, regardless of success or failure

## Further Reading

- [Resilience](resilience.md) -- Retry, timeout, and safe_run
- [Chains & Cascades](chains.md) -- Core chain operations
- [API Reference](../reference.md) -- Full method signatures
