---
title: "Quent vs unasync — Sync/Async Approaches Compared"
description: "Compare quent and unasync for solving Python's sync/async code duplication. Runtime bridging vs code generation — when to use each approach."
tags:
  - comparison
  - unasync
  - sync-async
search:
  boost: 4
---

# Quent vs unasync

## The Shared Problem

Both quent and unasync exist because of the same pain point: **maintaining both
sync and async versions of the same code**. If you have a function that validates,
transforms, and stores data, and you need it to work for both sync and async
callers, you end up writing the same logic twice. Two functions. Two test suites.
They drift apart.

quent and unasync solve this problem, but they take fundamentally different
approaches. Neither is universally better -- they make different tradeoffs that
suit different situations.

## How unasync Works

[unasync](https://github.com/python-trio/unasync) is a **code generation** tool.
You write your code once as async, and unasync generates the sync version
automatically at build time.

### The approach

1. You write the async version of your code in a designated source directory
   (by default, `_async/`).
2. unasync processes the source files and generates sync copies in a destination
   directory (by default, `_sync/`).
3. The transformation happens at build time, typically via a setuptools plugin.
4. The generated sync code replaces async/await keywords and other async-specific
   tokens with their synchronous equivalents.

### Setup

Integration is straightforward -- you add a `cmdclass` to your `setup.py`:

```python
from setuptools import setup
import unasync

setup(
  # ...
  cmdclass={"build_py": unasync.cmdclass_build_py()},
)
```

You can customize the transformation with `Rule` objects for custom source/destination
directories and additional token replacements:

```python
setup(
  cmdclass={
    "build_py": unasync.cmdclass_build_py(
      rules=[
        unasync.Rule(
          "/mylib/_async/",
          "/mylib/_sync/",
          additional_replacements={"AsyncClient": "Client"},
        ),
      ]
    )
  },
)
```

### What gets transformed

unasync performs token-level replacements. The default transformations include:

- `async def` becomes `def`
- `await ` is removed
- `async for` becomes `for`
- `async with` becomes `with`
- `__aenter__` becomes `__enter__`
- `__aexit__` becomes `__exit__`
- `__aiter__` becomes `__iter__`
- `__anext__` becomes `__next__`

You can add custom replacements for your own async/sync type pairs (e.g.,
`AsyncSession` to `Session`).

unasync is a proven, stable tool used by established projects including the
official Elasticsearch Python client and httpcore. Dual-licensed MIT/Apache 2.0.
[Repository](https://github.com/python-trio/unasync).

## How quent Works

quent is a **runtime bridging** tool. You write a pipeline definition with
callables, and quent detects at runtime whether each step returns an awaitable.

### The approach

1. You define a `Q` pipeline -- a sequence of callables.
2. When you call `.run()`, quent executes each step synchronously.
3. After each step, it checks the return value for awaitability.
4. If the result is awaitable (a coroutine, Task, Future), execution transitions
   to async mode for the remaining steps.
5. `.run()` returns either a plain value or a coroutine -- the caller decides
   whether to `await`.

### Setup

No build step. No configuration. Install and use:

```python
from quent import Q

pipeline = (
  Q()
  .then(fetch_data)
  .then(validate)
  .then(transform)
  .then(save)
)

result = pipeline.run(data)          # sync callables -> returns value
result = await pipeline.run(data)    # async callables -> returns coroutine
```

The same pipeline works with any mix of sync and async callables without changes.

## Head-to-Head Comparison

| Aspect | unasync | quent |
|--------|---------|-------|
| **Approach** | Code generation (build time) | Runtime bridging |
| **You write** | Async Python code | Q definitions with callables |
| **You get** | Generated sync copy of your code | Same pipeline works for both sync and async |
| **Runtime overhead** | Zero -- generated code is plain Python | Minimal -- one awaitable check per step |
| **Build step required** | Yes (setuptools plugin) | No |
| **Dependencies** | Build-time dependency | Zero runtime dependencies |
| **Code style** | Standard Python (async version) | Fluent pipeline API |
| **Output** | Two separate code files (async + sync) | One pipeline definition |
| **Mixed sync/async in one pipeline** | No -- each file is either async or sync | Yes -- any step can be either |
| **Debugging** | Two code paths to debug | One code path + enhanced tracebacks |
| **Error handling** | Standard try/except in each version | `except_()`, `finally_()` built in |
| **Scope** | Sync/async code duplication only | Sync/async bridging + pipeline features |

## Side-by-Side Example

Consider a data processing function that needs to work with both sync and async
database clients.

### With unasync

You write the async version:

```python
# mylib/_async/processor.py

async def process_record(db, record):
  validated = await db.validate(record)
  enriched = await db.enrich(validated)
  normalized = normalize(enriched)  # sync helper
  await db.store(normalized)
  return normalized
```

unasync generates the sync version at build time:

```python
# mylib/_sync/processor.py  (generated -- do not edit)

def process_record(db, record):
  validated = db.validate(record)
  enriched = db.enrich(validated)
  normalized = normalize(enriched)
  db.store(normalized)
  return normalized
```

You now have two functions in two files. Users import from the appropriate module:

```python
from mylib._async.processor import process_record  # async callers
from mylib._sync.processor import process_record   # sync callers
```

### With quent

You write one pipeline definition:

```python
# mylib/processor.py

from quent import Q

def process_record(db, record):
  return (
    Q(record)
    .then(db.validate)
    .then(db.enrich)
    .then(normalize)
    .do(db.store)
    .run()
  )
```

One function. One file. The caller decides:

```python
from mylib.processor import process_record

# Sync caller
result = process_record(sync_db, record)

# Async caller
result = await process_record(async_db, record)
```

### What the example shows

- **unasync** gives you standard Python in both versions. No new API to learn.
  The generated sync code is readable and debuggable like any normal Python.
- **quent** gives you a single definition. No generated files, no build step,
  no separate import paths. But you do learn the quent API.

## When to Choose unasync

unasync is the better choice when:

- **You have a large existing async codebase** and want to offer a sync variant
  without rewriting. unasync fits into your existing code structure -- you keep
  writing normal Python.

- **You want zero runtime overhead.** The generated sync code is plain Python
  with no framework involved at runtime.

- **Your team prefers standard Python syntax.** unasync does not introduce new
  APIs or patterns.

- **Your sync and async versions are structurally identical** -- same function
  signatures, same control flow, with the only difference being `async`/`await`
  keywords. This is unasync's sweet spot.

- **You need to support Python versions older than 3.10.** unasync has broader
  Python version compatibility.

## When to Choose quent

quent is the better choice when:

- **You need mixed sync/async in the same pipeline.** A pipeline can have some steps
  that are sync and others that are async. unasync generates code that is entirely
  sync or entirely async -- it cannot mix.

- **You want runtime flexibility.** The same pipeline adapts to whatever callables
  it receives. You can pass a sync database client or an async one to the same
  function.

- **You want built-in pipeline features.** `except_()` for error handling,
  `gather()` for concurrent execution,
  `with_()` for context managers, `if_()`/`else_()` for conditionals --
  all working transparently across the sync/async boundary.

- **You want a single import path.** One function, one module. No `_async` vs
  `_sync` directories, no conditional imports.

- **You are building a library where callers provide the callables.** If your
  library accepts user-provided functions that might be sync or async, quent
  handles both without requiring two API surfaces.

## Can You Use Both?

Yes. They are not mutually exclusive. You could use unasync for the bulk of your
codebase (keeping standard Python everywhere) and use quent for specific
integration points where runtime flexibility or mixed sync/async is needed.

## Summary

| Question | unasync | quent |
|----------|---------|-------|
| Do I write standard Python? | Yes | Q API |
| Is there a build step? | Yes | No |
| Can I mix sync and async? | No | Yes |
| Is there runtime overhead? | No | Minimal |
| Do I get two code files? | Yes | No |
| Pipeline features? | No (use standard Python) | Yes (error handling, gather, etc.) |
| Who provides the callables? | You (in the async source) | Anyone (runtime flexibility) |

Both tools solve the sync/async duplication problem. unasync does it through code
generation with zero runtime cost. quent does it through runtime detection with
zero build cost. Choose based on which tradeoffs matter more for your project.

## Further Reading

- [Why Quent](../why-quent.md) -- the problem quent solves and when to use it
- [Getting Started](../getting-started.md) -- install quent and build your first pipeline
- [Quent vs Other Libraries](vs-others.md) -- how quent compares to returns, toolz, pipe, and Expression
