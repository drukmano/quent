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
.except_(fn, /, *args, exceptions=None, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Handler function. Receives the exception as its first argument. |
| `*args` | `Any` | Additional arguments passed to the handler |
| `exceptions` | `type` or `Iterable[type]` or `None` | Exception types to catch. `None` catches all `Exception` subclasses. |
| `**kwargs` | `Any` | Additional keyword arguments passed to the handler |

When the `exceptions` filter matches (or is `None`), the handler runs and its return value becomes the chain's result -- the exception is swallowed. When the exception does not match the filter, the exception propagates normally and the handler is not called.

Only **one** `.except_()` handler is allowed per chain. Calling it a second time raises `QuentException`.

```python
# Return a default value on failure
result = (
  Chain(risky_operation)
  .except_(lambda e: default_value)
  .run()
)
```

### Exception Type Filtering

Filter the handler to specific exception types using the `exceptions` parameter:

```python
result = (
  Chain(risky_operation)
  .except_(
    handle_value_error,
    exceptions=ValueError,  # only catches ValueError
  )
  .run()
)
```

You can also pass multiple exception types:

```python
result = (
  Chain(risky_operation)
  .except_(
    handle_error,
    exceptions=(ValueError, TypeError),  # catches either
  )
  .run()
)
```

If the raised exception does not match the `exceptions` filter, it propagates as if no handler was registered.

## finally_

The `.finally_()` method registers a cleanup handler that **always runs**, regardless of whether the chain succeeded or raised an exception.

```python
result = (
  Chain(risky_operation)
  .then(process_result)
  .except_(handle_error)
  .finally_(cleanup)           # always called
  .run()
)
```

### Parameters

```python
.finally_(fn, /, *args, **kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Cleanup function |
| `*args` | `Any` | Arguments passed to the cleanup function |
| `**kwargs` | `Any` | Keyword arguments passed to the cleanup function |

The cleanup function receives the **root value** by default. Use the ellipsis (`...`) as the first argument to call it with no arguments:

```python
# cleanup(root_value) called with the root value
Chain(data).then(process).finally_(cleanup)

# cleanup() called with no arguments
Chain(data).then(process).finally_(cleanup, ...)
```

Only **one** `.finally_()` handler is allowed per chain. Calling it a second time raises `QuentException`.

## Enhanced Stack Traces

When an error occurs inside a chain, Quent enhances the exception traceback with a visualization of the chain state. This shows the entire chain structure and the exact operation that failed.

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

## Further Reading

- [Chains](chains.md) -- Core chain operations
- [API Reference](../reference.md) -- Full method signatures
