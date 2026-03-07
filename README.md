# Quent

A fluent chain/pipeline library for Python with transparent sync/async handling.

[![PyPI version](https://img.shields.io/pypi/v/quent)](https://pypi.org/project/quent/)
[![Python version](https://img.shields.io/pypi/pyversions/quent)](https://pypi.org/project/quent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Key Features

- **Fluent Chain Interface** -- Build readable pipelines by chaining operations without intermediate variables or nested calls
- **Transparent Async Handling** -- Use the exact same API for synchronous and asynchronous code; Quent automatically detects coroutines and handles them
- **Exceptional Stack Traces** -- Chain visualization on exceptions using `add_note()`, showing the exact chain state and the operation that failed
- **Lightweight** -- Pure Python, zero runtime dependencies

## Who Is This For?

- **Library and SDK authors** who need to support both sync and async callers without maintaining two codebases or resorting to code generation tools like unasync.
- **Backend and API developers** (especially FastAPI/Starlette users) who mix sync and async code at application boundaries and want a single pipeline that handles both transparently.

## When to Use Quent

### Use Quent when...

- You need the same code path to work for both sync and async callers without separate implementations.
- You want functional composition and chaining without relying on operator overloading hacks or metaprogramming.
- You want to eliminate `await` boilerplate in async chains -- Quent detects coroutines and transitions automatically.
- You need to wrap existing synchronous libraries for use in async contexts without rewriting them.

### Don't use Quent when...

- Your codebase is purely synchronous with no async plans -- standard function calls will be simpler.
- You need a full workflow orchestration engine with scheduling, DAGs, and distributed execution -- use [Prefect](https://www.prefect.io/) or [Airflow](https://airflow.apache.org/).
- You want lazy evaluation or streaming pipelines -- Quent evaluates eagerly.
- You need distributed pipeline execution across multiple processes or machines.
- You need Python < 3.10 support.

## Installation

```bash
pip install quent
```

Requires Python 3.10 or later. No runtime dependencies.

## Quick Start

```python
from quent import Chain

result = Chain(fetch_data, user_id).then(validate).then(transform).then(save).run()
```

This works identically whether the functions involved are synchronous or asynchronous. Quent detects coroutines at runtime and transitions the chain to async execution automatically.

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

1. Wraps it in an `asyncio.Task` (with `eager_start=True` on Python 3.14+ for immediate execution)
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

```python
result = (
  Chain(fetch_data, id)
  .do(log_operation)       # log but discard return value
  .then(transform)         # receives fetch_data result
  .then(save)
  .run()
)
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

When Quent detects a coroutine during chain evaluation, it wraps it in an `asyncio.Task`. On Python 3.14+, this uses `eager_start=True` so sync-completing coroutines execute immediately without a round-trip through the event loop.

## Error Handling

### except_ and finally_

Each chain supports one `except_` handler and one `finally_` handler.

```python
result = (
  Chain(risky_operation)
  .then(process_result)
  .except_(handle_error)       # called on exception
  .finally_(cleanup, ...)      # always called (... means no args)
  .run()
)
```

The `except_` handler receives the caught exception. If the exception matches, it is swallowed and the handler's return value becomes the chain result:

```python
# Catch any exception
chain.except_(handle_error)

# Filter by exception type
chain.except_(handle_value_error, exceptions=ValueError)

# Filter by multiple exception types
chain.except_(handle_error, exceptions=(ValueError, TypeError))
```

The `finally_` handler always runs, even if an exception occurred. It receives the root value:

```python
chain.finally_(cleanup)         # cleanup(root_value)
chain.finally_(cleanup, ...)    # cleanup() -- no arguments
```

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

## Flow Control

### Loops

```python
# Iterate over items and process each one; results collected into a list
Chain(get_items).map(process_item).then(summarize).run()

# Iterate as a side effect (result discarded, original items collected)
Chain(get_items).foreach(log_item).then(continue_processing).run()

# Filter an iterable, keeping elements where fn returns truthy
Chain(get_items).filter(is_valid).then(process).run()

# Run multiple functions concurrently on the current value
Chain(fetch_data).gather(analyze, summarize, archive).run()

# Produce an iterator from chain results
for item in Chain(get_items).iterate(transform):
  print(item)

# Async iteration
async for item in Chain(get_items_async).iterate(transform):
  print(item)

# iterate_do discards fn's return values
for item in Chain(get_items).iterate_do(log_item):
  print(item)
```

### Flow Statements

`Chain.return_()` and `Chain.break_()` are class methods that control chain execution flow:

```python
# Return early from a chain
Chain(get_data).then(lambda v: Chain.return_(v) if v else None).then(transform).run()

# Break out of a map loop
Chain(get_items).map(lambda item: Chain.break_() if item is None else process(item)).run()
```

### Context Managers

```python
# Execute fn within a context manager; result is passed forward
Chain(acquire_lock).with_(perform_operation).run()

# Execute as a side effect within a context manager
Chain(open_file, path).with_do(write_header).then(continue_processing).run()
```

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
  print('No value was produced')
```

## Chain Reuse and Immutability

### freeze and FrozenChain

Freeze a chain into an immutable, callable snapshot. A `FrozenChain` cannot be modified but can be called repeatedly and is safe for concurrent use:

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

## How Quent Compares

| Feature | Quent | pipe | toolz | tenacity | unasync | asyncer |
|---------|-------|------|-------|----------|---------|---------|
| Sync/async transparency | Yes | No | No | No | Code generation | Runtime wrappers |
| Fluent chaining API | Yes | Partial (`\|>` only) | Partial | No | No | No |
| Error handling (except/finally) | Yes | No | No | No | No | No |
| Zero dependencies | Yes | Yes | No | No | No | No |
| Reusable/frozen chains | Yes | No | No | No | No | No |

Quent is the only Python library that combines fluent chaining and transparent sync/async handling in a single package. Where other tools address one of these concerns -- pipe for composition, unasync for dual sync/async support -- Quent unifies them into a single chain interface with minimal overhead.

## Performance

Quent is pure Python with minimal overhead over direct function calls:

- **No external dependencies** -- Zero runtime cost from third-party packages
- **Eager Task Creation** -- On Python 3.14+, `asyncio.create_task(eager_start=True)` allows sync-completing coroutines to execute immediately without event loop round-trips
- **Lightweight linked list** -- Chain links are allocated as simple slot-based objects; frozen chains avoid repeated chain construction

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `Chain` | Core chain class; each operation receives the previous result |
| `FrozenChain` | Immutable callable chain snapshot created by `.freeze()` |
| `Null` | Sentinel value representing "no value" (distinct from `None`) |
| `QuentException` | Base exception class for Quent errors |

### Core Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `Chain()` | `Chain(v=Null, /, *args, **kwargs)` | Initialize chain with an optional root value |
| `.then()` | `.then(v, /, *args, **kwargs) -> Chain` | Add operation; result becomes current value |
| `.do()` | `.do(fn, /, *args, **kwargs) -> Chain` | Side effect; result is discarded |
| `.run()` | `.run(v=Null, /, *args, **kwargs) -> Any` | Execute the chain |
| `.__call__()` | `(v=Null, /, *args, **kwargs) -> Any` | Alias for `.run()` |

### Loops and Iteration

| Method | Signature | Description |
|--------|-----------|-------------|
| `.map()` | `.map(fn, /) -> Chain` | Apply `fn` to each item; results collected into list |
| `.foreach()` | `.foreach(fn, /) -> Chain` | Apply `fn` to each item as side effect; original items collected |
| `.filter()` | `.filter(fn, /) -> Chain` | Filter iterable; keep elements where `fn` returns truthy |
| `.gather()` | `.gather(*fns) -> Chain` | Run multiple functions concurrently on current value |
| `.iterate()` | `.iterate(fn=None) -> _Generator` | Sync/async iterator over chain output |
| `.iterate_do()` | `.iterate_do(fn=None) -> _Generator` | Iterator discarding fn's return values |

### Context Managers

| Method | Signature | Description |
|--------|-----------|-------------|
| `.with_()` | `.with_(fn, /, *args, **kwargs) -> Chain` | Execute `fn` inside current value as context manager |
| `.with_do()` | `.with_do(fn, /, *args, **kwargs) -> Chain` | Side-effect `fn` inside current value as context manager |

### Error Handling

| Method | Signature | Description |
|--------|-----------|-------------|
| `.except_()` | `.except_(fn, /, *args, exceptions=None, **kwargs) -> Chain` | Register exception handler (one per chain) |
| `.finally_()` | `.finally_(fn, /, *args, **kwargs) -> Chain` | Register cleanup handler (one per chain; always runs) |

### Chain Reuse

| Method | Signature | Description |
|--------|-----------|-------------|
| `.freeze()` | `.freeze() -> FrozenChain` | Create immutable callable snapshot |
| `.decorator()` | `.decorator() -> Callable` | Use chain as a function decorator |

### Flow Control (Class Methods)

| Method | Signature | Description |
|--------|-----------|-------------|
| `Chain.return_()` | `Chain.return_(v=Null, /, *args, **kwargs) -> NoReturn` | Exit chain early with a value |
| `Chain.break_()` | `Chain.break_(v=Null, /, *args, **kwargs) -> NoReturn` | Break out of `map` or `filter` loop |

## Requirements

- **Python** >= 3.10
- **Runtime dependencies**: none
- **Build dependencies**: setuptools, wheel

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, project structure, and contribution guidelines.

## License

MIT License -- see [LICENSE](LICENSE) for details.
