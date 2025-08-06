# Resilience

Quent includes built-in resilience features that compose directly within the chain. Instead of wrapping functions with external decorators, you add retry, timeout, and safe execution as chain operations.

## Retry

The `.retry()` method automatically retries failed operations with configurable count, delay, and exception filtering.

```python
from quent import Chain

result = (
  Chain(call_external_api, request)
  .retry(3, delay=1.0, exceptions=(ConnectionError, TimeoutError))
  .then(parse_response)
  .run()
)
```

### Parameters

```python
.retry(count, *, delay=0.0, exceptions=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `count` | `int` | Number of retry attempts |
| `delay` | `float` | Delay in seconds between attempts (default: `0.0`) |
| `exceptions` | `tuple` or `type` or `None` | Exception types to retry on. `None` retries on all exceptions. |

### ExceptionGroup on Exhaustion

When all retry attempts are exhausted, an `ExceptionGroup` is raised containing every individual failure:

```python
try:
  result = Chain(flaky_operation).retry(3).run()
except ExceptionGroup as eg:
  for exc in eg.exceptions:
    print(f"Attempt failed: {exc}")
```

This gives you full visibility into every failure that occurred, not just the last one.

### Filtering by Exception Type

You can restrict retries to specific exception types. If an exception of a different type is raised, it propagates immediately without consuming retry attempts.

```python
result = (
  Chain(call_api, request)
  .retry(3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
  .then(parse_response)
  .run()
)
# A ValueError would propagate immediately.
# Only ConnectionError and TimeoutError trigger retries.
```

### Delay Between Retries

The `delay` parameter sets a fixed delay in seconds between retry attempts.

```python
# Retry 3 times with 1 second between attempts
Chain(flaky_operation).retry(3, delay=1.0).run()
```

## Timeout

The `.timeout()` method enforces a time limit on async chain execution. If the chain does not complete within the specified duration, an `asyncio.TimeoutError` is raised.

```python
from quent import Chain

result = await (
  Chain(long_running_task)
  .timeout(5.0)  # seconds
  .run()
)
```

### Parameters

```python
.timeout(delay)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `delay` | `float` | Maximum execution time in seconds |

!!! note
    Timeout applies to async chain execution. If the chain is entirely synchronous, the timeout has no effect because synchronous code cannot be interrupted by the event loop.

## safe_run

The `.safe_run()` method executes a chain with automatic cloning, making it safe for concurrent use from multiple threads or coroutines.

```python
from quent import Chain

template = Chain().then(validate).then(transform).then(save)

# Thread-safe -- each call operates on an independent clone
result = template.safe_run(data)
```

### Parameters

```python
.safe_run(v=None, *args, **kwargs)
```

Same signature as `.run()`. The difference is that `.safe_run()` clones the chain before execution, so the original chain is never modified. This is essential when using a chain template from multiple threads or async tasks.

### Why Not Just Use `.run()`?

Chains are mutable during execution -- the root value and internal state change as operations execute. If two threads call `.run()` on the same chain simultaneously, they would interfere with each other. `.safe_run()` prevents this by cloning first.

```python
import threading
from quent import Chain

pipeline = Chain().then(validate).then(transform).then(save)

# WRONG -- concurrent .run() on the same chain is unsafe
# threading.Thread(target=pipeline.run, args=(data1,)).start()
# threading.Thread(target=pipeline.run, args=(data2,)).start()

# CORRECT -- safe_run clones before execution
threading.Thread(target=pipeline.safe_run, args=(data1,)).start()
threading.Thread(target=pipeline.safe_run, args=(data2,)).start()
```

## Combining Resilience Features

Retry, timeout, and error handling compose naturally within the same chain.

### Resilient API Client

```python
from quent import Chain

def api_request(method, url, **request_kwargs):
  return (
    Chain(http_client.request, method, url, **request_kwargs)
    .with_context(method=method, url=url)
    .retry(3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
    .timeout(10.0)
    .then(validate_response)
    .then(parse_json)
    .except_(log_api_error, reraise=False)
    .run()
  )

def validate_response(response):
  ctx = Chain.get_context()
  if response.status >= 400:
    raise ValueError(f"{ctx['method']} {ctx['url']} returned {response.status}")
  return response

async def main():
  user = await api_request("GET", "/users/1")
  updated = await api_request("PUT", "/users/1", json={"name": "Alice"})
```

This single function:

- Retries up to 3 times on connection or timeout errors, with 0.5s between attempts
- Enforces a 10-second overall timeout
- Validates the response status
- Parses the JSON body
- Logs errors without re-raising (returns `None` on failure)
- Works transparently with both sync and async HTTP clients
- Propagates context (method, URL) for use in error handlers

### Batch Configuration

Use `.config()` to set multiple resilience options at once:

```python
chain = (
  Chain(operation)
  .config(timeout=5.0, retry=3)
  .run()
)
```

The `retry` parameter in `.config()` accepts either an `int` (retry count) or a `dict` with keys matching the `.retry()` method parameters:

```python
chain = (
  Chain(operation)
  .config(retry={"count": 3, "delay": 1.0, "exceptions": (ConnectionError,)})
  .run()
)
```

## Further Reading

- [Error Handling](error-handling.md) -- Exception handlers and finally blocks
- [Context Propagation](context.md) -- Passing metadata through chains
- [Async Handling](async.md) -- How transparent async works
- [API Reference](../reference.md) -- Full method signatures
- [Comparisons: vs tenacity](../comparisons/vs-tenacity.md) -- Quent's in-chain retry vs tenacity's decorators
