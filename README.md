# Quent

A high-performance chain interface library for Python with transparent async/await handling.

[![PyPI version](https://img.shields.io/pypi/v/quent)](https://pypi.org/project/quent/)
[![Python version](https://img.shields.io/pypi/pyversions/quent)](https://pypi.org/project/quent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Key Features

- **Fluent Chain Interface** -- Build readable pipelines by chaining operations without intermediate variables or nested calls
- **Transparent Async Handling** -- Use the exact same API for synchronous and asynchronous code; Quent automatically detects coroutines and handles them
- **Pipe Operator Syntax** -- Write expressive pipelines with `Chain(f1) | f2 | run()`
- **Built-in Resilience** -- Retry failed operations, enforce timeouts on async chains, and use `safe_run` for thread-safe execution
- **High Performance** -- Cython-compiled core with C-level coroutine detection and eager task creation via `asyncio.create_task(eager_start=True)`
- **Exceptional Stack Traces** -- Chain visualization on exceptions using `add_note()`, showing the exact chain state and the operation that failed
- **Context Propagation** -- Pass metadata across chain execution boundaries using contextvars with `.with_context()` and `Chain.get_context()`

## Who Is This For?

- **Library and SDK authors** who need to support both sync and async callers without maintaining two codebases or resorting to code generation tools like unasync.
- **Backend and API developers** (especially FastAPI/Starlette users) who mix sync and async code at application boundaries and want a single pipeline that handles both transparently.
- **Developers from Elixir, F#, or Rust** who miss pipe operators and functional composition and want an async-aware equivalent in Python.
- **Teams building resilient services** who want retry, timeout, and error handling composed directly in the pipeline rather than scattered across decorators and wrappers.

## When to Use Quent

### Use Quent when...

- You need the same code path to work for both sync and async callers without separate implementations.
- You are building pipelines that require retry, timeout, or error handling composed inline rather than applied via external decorators.
- You want functional composition and chaining without relying on operator overloading hacks or metaprogramming.
- You want to eliminate `await` boilerplate in async chains -- Quent detects coroutines and transitions automatically.
- You need to wrap existing synchronous libraries for use in async contexts without rewriting them.

### Don't use Quent when...

- Your codebase is purely synchronous with no async plans -- standard function calls or [pipe](https://pypi.org/project/pipe/) will be simpler.
- You need a full workflow orchestration engine with scheduling, DAGs, and distributed execution -- use [Prefect](https://www.prefect.io/) or [Airflow](https://airflow.apache.org/).
- You want lazy evaluation or streaming pipelines -- Quent evaluates eagerly.
- You need distributed pipeline execution across multiple processes or machines.
- You need Python < 3.14 support.

## Installation

```bash
pip install quent
```

Requires Python 3.14 or later. No runtime dependencies.

## Quick Start

```python
from quent import Chain, run

# Fluent API
result = Chain(fetch_data, user_id).then(validate).then(transform).then(save).run()

# Pipe syntax
result = Chain(fetch_data, user_id) | validate | transform | save | run()
```

Both forms work identically whether the functions involved are synchronous or asynchronous. Quent detects coroutines at runtime and transitions the chain to async execution automatically.

## Why Quent?

### The Problem

Traditional Python code often suffers from poor readability when chaining operations:

```python
# Nested function calls -- hard to read
result = send_data(normalize(validate(fetch_data(id))))

# Intermediate variables -- verbose
data = fetch_data(id)
data = validate(data)
data = normalize(data)
result = send_data(data)

# Async makes it worse -- which calls need await?
data = await fetch_data(id)
data = validate(data)          # Is this async?
data = await normalize(data)   # What if it's sometimes async?
result = await send_data(data)
```

### The Solution

Quent provides a clean, chainable interface that works identically for sync and async code:

```python
from quent import Chain

# Clean, readable, works for both sync and async
result = Chain(fetch_data, id).then(validate).then(normalize).then(send_data).run()
```

When Quent encounters a coroutine during chain evaluation, it:

1. Wraps it in an `asyncio.Task` with `eager_start=True` for immediate execution
2. Continues the remaining chain evaluation within the task
3. Returns the task, which the caller can `await` if needed

The chain starts synchronous and only transitions to async at the exact point where an awaitable value is first encountered -- there is no upfront async/sync decision.

## Core Concepts

### Chain

The `Chain` class enables method chaining for sequential operations. Each operation receives the result of the previous one.

```python
from quent import Chain

result = (
  Chain(fetch_user, user_id)
  .then(validate_permissions)
  .then(apply_transformations)
  .then(save_to_database)
  .run()
)
```

#### Core Operations

| Method | Receives | Result used? | Description |
|--------|----------|-------------|-------------|
| `.then(v)` | current value | Yes | Add operation, pass result forward |
| `.do(fn)` | current value | No | Side effect, discard result |
| `.root()` | -- | Yes | Reset current value to root value |
| `.root(fn)` | root value | Yes | Transform root value, use as current |
| `.root_do(fn)` | root value | No | Side effect on root value, discard result |

```python
result = (
  Chain(fetch_data, id)
  .do(log_operation)                    # log but discard return value
  .then(transform)                      # receives fetch_data result
  .root(lambda original: log(original)) # receives the original fetch_data result
  .then(save)
  .run()
)
```

### Cascade

The `Cascade` class implements the fluent interface pattern. Every operation receives the root value, and the final result is always the root value.

```python
from quent import Cascade, CascadeAttr

# All operations receive the original data
processed_data = (
  Cascade(fetch_data, id)
  .then(send_to_backup)     # backup receives data
  .then(send_to_analytics)  # analytics receives data
  .then(log_operation)      # logger receives data
  .run()  # returns the original data
)

# Make any class fluent with CascadeAttr
result = CascadeAttr(list()).append(1).append(2).extend([3, 4]).run()
# Returns: [1, 2, 3, 4]
```

### Pipe Operator

The `|` operator provides an alternative syntax for building chains. Use the `run` class to terminate and execute a pipe chain.

```python
from quent import Chain, run

# These are equivalent
result = Chain(fetch_data, id).then(validate).then(transform).then(save).run()
result = Chain(fetch_data, id) | validate | transform | save | run()

# Pass arguments to run
result = Chain().then(validate).then(transform) | run(fetch_data, id)
```

### Transparent Async

Write your code once, use it with both sync and async data sources:

```python
def process_data(data_source):
  """Works with both sync and async data sources."""
  return (
    Chain(data_source.fetch)
    .then(validate)
    .then(transform)
    .then(data_source.save)
    .run()
  )

# Synchronous -- returns the result directly
result = process_data(sync_database)

# Asynchronous -- returns a Task that can be awaited
result = await process_data(async_database)
```

When Quent detects a coroutine during chain evaluation, it wraps it in an `asyncio.Task` with `eager_start=True`. This means sync-completing coroutines execute immediately without a round-trip through the event loop, yielding 2-5x faster async chains compared to standard task scheduling.

## Error Handling

### except_ and finally_

```python
result = (
  Chain(risky_operation)
  .then(process_result)
  .except_(handle_error)       # called on exception
  .finally_(cleanup, ...)      # always called (... means no args)
  .run()
)
```

The `except_` method accepts several options:

```python
# Filter by exception type
chain.except_(
  handle_value_error,
  exceptions=ValueError,
  reraise=False  # do not re-raise after handling
)

# Multiple handlers
chain.except_(
  handle_value_error,
  exceptions=ValueError
).except_(
  handle_type_error,
  exceptions=TypeError
).except_(
  handle_any  # catch-all
)
```

When `reraise=True` (the default), the exception is re-raised after the handler runs. Set `reraise=False` to swallow the exception and continue.

### Enhanced Stack Traces

When an error occurs inside a chain, Quent attaches a visualization of the chain state to the exception using `add_note()`. This shows the entire chain structure, intermediate values, and the exact operation that failed:

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
- The entire chain structure
- Intermediate values (e.g., `= {'id': 42, 'value': 100}`)
- The exact operation that failed, marked with `<----`
- The original exception and its source location

## Resilience

### Retry

Automatically retry failed operations with configurable count, delay, and exception filtering:

```python
result = (
  Chain(call_external_api, request)
  .retry(3, delay=1.0, exceptions=(ConnectionError, TimeoutError))
  .then(parse_response)
  .run()
)
```

When all retry attempts are exhausted, an `ExceptionGroup` is raised containing every individual failure:

```python
try:
  result = Chain(flaky_operation).retry(3).run()
except ExceptionGroup as eg:
  for exc in eg.exceptions:
    print(f"Attempt failed: {exc}")
```

### Timeout

Enforce a time limit on async chain execution:

```python
result = await (
  Chain(long_running_task)
  .timeout(5.0)  # seconds
  .run()
)
```

### safe_run

Execute a chain with automatic cloning, making it safe for concurrent use from multiple threads:

```python
template = Chain().then(validate).then(transform).then(save)

# Thread-safe -- each call operates on an independent clone
result = template.safe_run(data)
```

## Context Propagation

Pass metadata through the chain execution using contextvars. The context is isolated per chain run and works across async boundaries:

```python
result = (
  Chain(fetch_user, user_id)
  .with_context(request_id="abc-123", trace=True)
  .then(validate)
  .then(enrich)
  .run()
)

def enrich(user):
  ctx = Chain.get_context()
  request_id = ctx["request_id"]
  # ... use request_id for logging, tracing, etc.
  return user
```

## Flow Control

### Conditionals

| Method | Condition |
|--------|-----------|
| `.if_(v)` | Execute `v` if current value is truthy |
| `.else_(v)` | Execute `v` if current value is falsy |
| `.if_not(v)` | Execute `v` if current value is falsy |
| `.condition(fn)` | Execute next link only if `fn(current_value)` is truthy |
| `.not_()` | Negate current value |
| `.eq(value)` | `current_value == value` |
| `.neq(value)` | `current_value != value` |
| `.is_(value)` | `current_value is value` |
| `.is_not(value)` | `current_value is not value` |
| `.in_(value)` | `current_value in value` |
| `.not_in(value)` | `current_value not in value` |
| `.isinstance_(*types)` | `isinstance(current_value, types)` |
| `.or_(value)` | `current_value or value` |
| `.if_raise(exc)` | Raise `exc` if current value is truthy |
| `.else_raise(exc)` | Raise `exc` if current value is falsy |
| `.if_not_raise(exc)` | Raise `exc` if current value is falsy |

```python
# Conditional execution
result = (
  Chain(get_user)
  .then(lambda u: u.age)
  .then(lambda a: a >= 18)
  .if_(grant_access)
  .else_(deny_access)
  .run()
)

# Raise on condition
result = (
  Chain(get_config, key)
  .else_raise(ValueError("Config key not found"))
  .then(parse_config)
  .run()
)
```

### Loops

```python
# Iterate over items and process each one; result is passed forward
Chain(get_items).foreach(process_item).then(summarize).run()

# Iterate as a side effect (result discarded)
Chain(get_items).foreach_do(log_item).then(continue_processing).run()

# Produce an iterator from chain results
for item in Chain(get_items).iterate(transform):
  print(item)

# Async iteration
async for item in Chain(get_items_async).iterate(transform):
  print(item)

# Loop while a function returns truthy
Chain(initial_state).while_true(step_function).run()
```

### Flow Statements

`Chain.return_()` and `Chain.break_()` are class methods that control chain execution flow:

```python
# Return early from a chain
Chain(get_data).then(lambda v: Chain.return_(v) if v else None).then(transform).run()

# Break out of a foreach or while_true loop
Chain(get_items).foreach(lambda item: Chain.break_() if item is None else process(item)).run()
```

### Context Managers

```python
# Execute fn within a context manager; result is passed forward
Chain(acquire_lock).with_(perform_operation).run()

# Execute as a side effect within a context manager
Chain(open_file, path).with_do(write_header).then(continue_processing).run()
```

### Sleep and Raise

```python
# Insert a delay into the chain
Chain(start_job).then(submit).sleep(2.0).then(check_status).run()

# Unconditionally raise an exception
Chain(get_value).then(validate).raise_(RuntimeError("unexpected state")).run()
```

## Attribute Access

Access attributes and call methods on the current chain value:

```python
# Get an attribute
Chain(fetch_user).attr("name").then(print).run()

# Call a method
Chain(fetch_user).attr_fn("get_profile", detailed=True).then(process).run()
```

### ChainAttr and CascadeAttr

`ChainAttr` and `CascadeAttr` provide dynamic attribute access via `__getattr__`, allowing natural dot-notation on the chain itself:

```python
from quent import ChainAttr

result = (
  ChainAttr("hello world")
  .upper()        # calls str.upper()
  .split()        # calls str.split()
  .run()
)
# Returns: ['HELLO', 'WORLD']
```

```python
from quent import CascadeAttr

result = CascadeAttr(list()).append(1).append(2).extend([3, 4]).run()
# Returns: [1, 2, 3, 4]
```

## Chain Reuse and Immutability

### clone

Create a deep copy of a chain for independent reuse:

```python
base = Chain().then(validate).then(normalize)
chain_a = base.clone().then(save_to_db)
chain_b = base.clone().then(send_to_api)
```

### freeze and FrozenChain

Freeze a chain into an immutable, callable snapshot. A `FrozenChain` cannot be modified but can be called repeatedly:

```python
from quent import Chain

processor = Chain().then(validate).then(normalize).then(save).freeze()

# Use as a callable
for item in items:
  processor(item)
```

### decorator

Use a chain as a function decorator. The decorated function becomes the root value:

```python
@Chain().then(validate).then(normalize).then(save).decorator()
def process(data):
  return data
```

## Configuration

### config

Set multiple configuration options at once:

```python
chain = (
  Chain(operation)
  .config(autorun=True, debug=True, timeout=5.0, retry=3)
  .run()
)
```

The `retry` parameter accepts either an `int` (retry count) or a `dict` with keys matching the `.retry()` method parameters.

### autorun

When enabled, async chain results are automatically scheduled via `asyncio.create_task`:

```python
Chain(async_operation).autorun().run()  # Task is scheduled immediately
```

### Debug Mode

Enable debug mode to log chain execution details through the `quent` logger:

```python
import logging
logging.getLogger("quent").setLevel(logging.DEBUG)

Chain(operation).config(debug=True).run()
```

In debug mode, intermediate results are recorded on each link for inspection in stack traces.

## Special Values

### Ellipsis (`...`)

When passed as the first argument to a chain operation, the ellipsis signals that the function should be called with zero arguments, overriding the default behavior of passing the current value:

```python
# cleanup() is called with no arguments
Chain(data).then(process).finally_(cleanup, ...)

# compare with: cleanup(current_value)
Chain(data).then(process).finally_(cleanup)
```

### Null

The `Null` sentinel represents "no value" and is distinct from `None`. It is useful when `None` is a valid value in your domain:

```python
from quent import Chain, Null

# Check if a chain produced no value
result = chain.run()
if result is Null:
  print("No value was produced")

# Access via class method
sentinel = Chain.null()
```

## Real-World Examples

### Redis Pipeline (Sync/Async Transparent)

A single implementation that works for both `redis` and `redis.asyncio` without code duplication:

```python
def flush(self) -> Any | Coroutine:
  """Execute pipeline and return results.

  Returns a coroutine if using async Redis, otherwise returns
  the result directly. The caller knows which client they're
  using and can await if needed.
  """
  pipe = self.r.pipeline(transaction=self.transaction)
  self.apply_operations(pipe)

  return (
    Chain(pipe.execute, raise_on_error=True)
    .then(self.remove_ignored_commands)
    .finally_(pipe.reset, ...)  # always reset, even on error
    .run()
  )
```

### HTTP Client with Retry, Timeout, and Context

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

## How Quent Compares

| Feature | Quent | pipe | toolz | tenacity | unasync | asyncer |
|---------|-------|------|-------|----------|---------|---------|
| Sync/async transparency | Yes | No | No | No | Code generation | Runtime wrappers |
| Retry/timeout built-in | Yes (in-chain) | No | No | Retry only (decorators) | No | No |
| Cython performance | Yes | No | No | No | No | No |
| Fluent chaining API | Yes | Partial (`\|>` only) | Partial | No | No | No |
| Error handling (except/finally) | Yes | No | No | No | No | No |
| Context propagation | Yes | No | No | No | No | No |
| Zero dependencies | Yes | Yes | No | No | No | No |
| Reusable/frozen chains | Yes | No | No | No | No | No |

Quent is the only Python library that combines fluent chaining, transparent sync/async handling, and built-in resilience in a single package. Where other tools address one of these concerns -- pipe for composition, tenacity for retries, unasync for dual sync/async support -- Quent unifies them into a single Cython-compiled chain interface with negligible overhead over direct function calls.

## Performance

Quent is built for minimal overhead:

- **Cython Implementation** -- Core logic compiled to C extensions with optimization directives (`boundscheck=False`, `wraparound=False`)
- **C-level Coroutine Detection** -- Exact type identity checks instead of `isinstance` for coroutine detection
- **Eager Task Creation** -- `asyncio.create_task(eager_start=True)` allows sync-completing coroutines to execute immediately without event loop round-trips

### Benchmark Results

Average of 10 iterations of 100,000 loops each:

| Scenario | Time (seconds) |
|----------|---------------|
| Direct function call | 1.19 |
| With Quent chain | 1.20 |
| With Quent frozen chain | 1.06 |

Quent adds negligible overhead to direct function calls. Frozen chains can actually be faster due to pre-built link structures avoiding repeated chain construction.

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `Chain` | Core chain class; each operation receives the previous result |
| `Cascade` | Chain variant where every operation receives the root value |
| `ChainAttr` | Chain with dynamic attribute access via `__getattr__` |
| `CascadeAttr` | Cascade with dynamic attribute access via `__getattr__` |
| `FrozenChain` | Immutable callable chain snapshot created by `.freeze()` |
| `run` | Helper class for pipe operator termination |
| `QuentException` | Base exception class for Quent errors |
| `Null` | Sentinel value representing "no value" (distinct from `None`) |

### Core Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `Chain()` | `Chain(v=None, *args, **kwargs)` | Initialize chain with an optional root value |
| `.then()` | `.then(v, *args, **kwargs) -> Self` | Add operation; result becomes current value |
| `.do()` | `.do(fn, *args, **kwargs) -> Self` | Side effect; result is discarded |
| `.root()` | `.root() -> Self` | Reset current value to root value |
| `.root()` | `.root(fn, *args, **kwargs) -> Self` | Transform root value; result becomes current value |
| `.root_do()` | `.root_do(fn, *args, **kwargs) -> Self` | Side effect on root value; result is discarded |
| `.run()` | `.run(v=None, *args, **kwargs) -> ResultOrAwaitable` | Execute the chain |
| `.__call__()` | `(v=None, *args, **kwargs) -> ResultOrAwaitable` | Alias for `.run()` |
| `.__or__()` | `chain \| other -> Self` | Pipe operator; append operation to chain |

### Attribute Access

| Method | Signature | Description |
|--------|-----------|-------------|
| `.attr()` | `.attr(name) -> Self` | Get attribute `name` from current value |
| `.attr_fn()` | `.attr_fn(name, *args, **kwargs) -> Self` | Call method `name` on current value |

### Conditionals

| Method | Signature | Description |
|--------|-----------|-------------|
| `.if_()` | `.if_(v, *args, **kwargs) -> Self` | Execute `v` if current value is truthy |
| `.else_()` | `.else_(v, *args, **kwargs) -> Self` | Execute `v` if current value is falsy |
| `.if_not()` | `.if_not(v, *args, **kwargs) -> Self` | Execute `v` if current value is falsy |
| `.condition()` | `.condition(fn, *args, **kwargs) -> Self` | Execute next link only if `fn(current_value)` is truthy |
| `.not_()` | `.not_() -> Self` | Negate current value |
| `.eq()` | `.eq(value) -> Self` | `current_value == value` |
| `.neq()` | `.neq(value) -> Self` | `current_value != value` |
| `.is_()` | `.is_(value) -> Self` | `current_value is value` |
| `.is_not()` | `.is_not(value) -> Self` | `current_value is not value` |
| `.in_()` | `.in_(value) -> Self` | `current_value in value` |
| `.not_in()` | `.not_in(value) -> Self` | `current_value not in value` |
| `.isinstance_()` | `.isinstance_(*types) -> Self` | `isinstance(current_value, types)` |
| `.or_()` | `.or_(value) -> Self` | `current_value or value` |
| `.if_raise()` | `.if_raise(exc) -> Self` | Raise `exc` if current value is truthy |
| `.else_raise()` | `.else_raise(exc) -> Self` | Raise `exc` if current value is falsy |
| `.if_not_raise()` | `.if_not_raise(exc) -> Self` | Raise `exc` if current value is falsy |

### Loops and Iteration

| Method | Signature | Description |
|--------|-----------|-------------|
| `.foreach()` | `.foreach(fn) -> Self` | Apply `fn` to each item; result is passed forward |
| `.foreach_do()` | `.foreach_do(fn) -> Self` | Apply `fn` to each item as side effect |
| `.iterate()` | `.iterate(fn=None) -> Iterator \| AsyncIterator` | Yield chain results as an iterator |
| `.iterate_do()` | `.iterate_do(fn=None) -> Iterator \| AsyncIterator` | Yield chain results as side-effect iterator |
| `.while_true()` | `.while_true(fn, *args, **kwargs) -> Self` | Loop while `fn` returns truthy |

### Context Managers

| Method | Signature | Description |
|--------|-----------|-------------|
| `.with_()` | `.with_(fn, *args, **kwargs) -> Self` | Execute `fn` inside current value as context manager |
| `.with_do()` | `.with_do(fn, *args, **kwargs) -> Self` | Side-effect `fn` inside current value as context manager |

### Error Handling

| Method | Signature | Description |
|--------|-----------|-------------|
| `.except_()` | `.except_(fn, *args, exceptions=None, reraise=True, **kwargs) -> Self` | Register exception handler |
| `.finally_()` | `.finally_(fn, *args, **kwargs) -> Self` | Register cleanup handler (always runs) |

### Resilience

| Method | Signature | Description |
|--------|-----------|-------------|
| `.retry()` | `.retry(count, *, delay=0.0, exceptions=None) -> Self` | Retry on failure; raises `ExceptionGroup` on exhaustion |
| `.timeout()` | `.timeout(delay) -> Self` | Enforce async execution time limit |
| `.safe_run()` | `.safe_run(v=None, *args, **kwargs) -> ResultOrAwaitable` | Thread-safe execution via automatic clone |

### Configuration

| Method | Signature | Description |
|--------|-----------|-------------|
| `.config()` | `.config(*, autorun=None, debug=None, timeout=None, retry=None) -> Self` | Batch configuration |
| `.autorun()` | `.autorun(autorun=True) -> Self` | Auto-schedule async results as tasks |
| `.with_context()` | `.with_context(**context) -> Self` | Attach context metadata to chain |
| `Chain.get_context()` | `Chain.get_context() -> dict` | Retrieve current chain context (static method) |

### Chain Reuse

| Method | Signature | Description |
|--------|-----------|-------------|
| `.clone()` | `.clone() -> Self` | Deep-copy the chain for independent reuse |
| `.freeze()` | `.freeze() -> FrozenChain` | Create immutable callable snapshot |
| `.decorator()` | `.decorator() -> Callable[[FuncT], FuncT]` | Use chain as a function decorator |

### Flow Control (Class Methods)

| Method | Signature | Description |
|--------|-----------|-------------|
| `Chain.return_()` | `Chain.return_(v=None, *args, **kwargs) -> None` | Exit chain early with a value |
| `Chain.break_()` | `Chain.break_(v=None, *args, **kwargs) -> None` | Break out of `foreach` or `while_true` loop |
| `Chain.null()` | `Chain.null() -> Null` | Return the `Null` sentinel |

### Miscellaneous

| Method | Signature | Description |
|--------|-----------|-------------|
| `.sleep()` | `.sleep(delay) -> Self` | Insert an async delay into the chain |
| `.raise_()` | `.raise_(exc) -> Self` | Unconditionally raise an exception |

### Type Aliases

```python
type ResultOrAwaitable[T] = T | Awaitable[T]

type ChainLink[IN, OUT] = Callable[[IN], OUT]
type AnyLink[OUT, IN] = OUT | Callable[[IN], OUT]
type FuncT = Callable[..., Any]
```

`ResultOrAwaitable` is the return type of `.run()` and `.__call__()` -- it is either the result value directly or an `Awaitable` that resolves to it, depending on whether async operations were encountered.

## Requirements

- **Python** >= 3.14
- **Runtime dependencies**: none
- **Build dependencies**: Cython >= 3.2.4, setuptools, wheel

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, project structure, and contribution guidelines.

## License

MIT License -- see [LICENSE](LICENSE) for details.
