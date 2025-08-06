# Quent

[![PyPI version](https://img.shields.io/pypi/v/quent)](https://pypi.org/project/quent/)
[![Python version](https://img.shields.io/pypi/pyversions/quent)](https://pypi.org/project/quent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Quent** is a Cython-compiled chain interface library for Python that transparently handles both synchronous and asynchronous operations. You write your pipeline code once using a fluent API -- `Chain(fetch).then(validate).then(save).run()` -- and Quent automatically detects coroutines at runtime, transitioning to async execution only when needed. It includes built-in retry, timeout, context propagation, and enhanced stack traces, all composable within the chain. No code generation, no runtime bridges, no dual implementations. One codebase, both worlds.

## Installation

```bash
pip install quent
```

Requires Python 3.14 or later. No runtime dependencies.

## Quick Start

```python
from quent import Chain, run

# Fluent API -- build readable pipelines by chaining operations
result = Chain(fetch_data, user_id).then(validate).then(transform).then(save).run()

# Pipe syntax -- expressive alternative
result = Chain(fetch_data, user_id) | validate | transform | save | run()
```

Both forms work identically whether the functions involved are synchronous or asynchronous. Quent detects coroutines at runtime and transitions the chain to async execution automatically.

### Sync and Async with One Chain

```python
from quent import Chain

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

## Key Features

- **Fluent Chain Interface** -- Build readable pipelines by chaining operations without intermediate variables or nested calls
- **Transparent Async Handling** -- Use the exact same API for synchronous and asynchronous code; Quent automatically detects coroutines and handles them
- **Pipe Operator Syntax** -- Write expressive pipelines with `Chain(f1) | f2 | run()`
- **Built-in Resilience** -- Retry failed operations, enforce timeouts on async chains, and use `safe_run` for thread-safe execution
- **High Performance** -- Cython-compiled core with C-level coroutine detection and eager task creation via `asyncio.create_task(eager_start=True)`
- **Exceptional Stack Traces** -- Chain visualization on exceptions using `add_note()`, showing the exact chain state and the operation that failed
- **Context Propagation** -- Pass metadata across chain execution boundaries using contextvars with `.with_context()` and `Chain.get_context()`
- **Zero Dependencies** -- No runtime dependencies

## Next Steps

- [Getting Started](getting-started.md) -- Installation and your first chain
- [Chains & Cascades](guide/chains.md) -- Learn about the different chain types
- [Async Handling](guide/async.md) -- How transparent async works
- [Resilience](guide/resilience.md) -- Retry, timeout, and safe_run
- [Error Handling](guide/error-handling.md) -- Exception handling and stack traces
- [API Reference](reference.md) -- Full API documentation
