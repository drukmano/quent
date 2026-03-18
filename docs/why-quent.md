---
title: "Why Quent -- Solving the Sync/Async Tax"
description: "Why quent exists: eliminate the sync/async code duplication tax in Python. Learn when quent is the right choice, when it isn't, and how it compares to alternatives."
tags:
  - motivation
  - sync-async
  - design
  - comparison
search:
  boost: 6
---

# Why Quent

## The Sync/Async Tax

Python's `async`/`await` syntax is clean and powerful -- until you need to support
both sync and async callers with the same logic. That is when you start paying the
**sync/async tax**: a real, compounding cost measured in duplicated code, diverging
implementations, and doubled test surfaces.

Here is how it typically unfolds.

### It Starts Simple

You write a data-validation pipeline. Everything is synchronous, everything works:

```python
def validate_and_store(record):
  checked = check_schema(record)
  enriched = enrich(checked)
  normalized = normalize(enriched)
  store(normalized)
  return normalized
```

Four steps, one function, nothing to maintain.

### Then Requirements Change

A new data source requires an async HTTP call for enrichment. The database driver
switches to an async client. Suddenly you need an async variant:

```python
async def validate_and_store_async(record):
  checked = check_schema(record)          # still sync
  enriched = await enrich(checked)        # now async
  normalized = normalize(enriched)        # still sync
  await store(normalized)                 # now async
  return normalized
```

Two functions. Same logic. Same steps. The only differences are `async def`, two
`await` keywords, and the `_async` suffix.

### The Tax Compounds

You keep **both** versions because some callers are sync and some are async. Over
time, they drift:

- A colleague adds input sanitization to the sync version but forgets the async one.
- A bug fix in error handling lands in the async path but not the sync path.
- Tests double -- every scenario needs a sync and an async variant.
- Code reviews now check "did you update both versions?" and the answer is often no.

This is not hypothetical. It is a pattern that repeats in every Python codebase
that straddles the sync/async boundary: web frameworks that support both WSGI and
ASGI, ORMs with sync and async engines, HTTP clients with dual interfaces, data
pipelines that run locally (sync) and in production (async).

The cost scales linearly with the number of pipelines. Ten pipelines means twenty
functions, twenty test suites, and twenty opportunities for the two versions to
diverge.

---

## How Quent Solves It

The key insight: **you should not have to choose sync or async at definition time.**
The pipeline logic is identical in both cases. Only the callables differ -- and Python
already tells you whether a return value is awaitable.

quent uses this to bridge the gap automatically.

### One Definition, Both Worlds

The same validation pipeline, written once:

```python
from quent import Q

validate_and_store = (
  Q()
  .then(check_schema)
  .then(enrich)
  .then(normalize)
  .then(store)
)
```

That is the entire implementation. No `async def`, no `await`, no duplication.

### The Caller Decides

```python
# Sync caller -- all callables are sync, returns a plain value
result = validate_and_store.run(record)

# Async caller -- enrich() and store() return awaitables
result = await validate_and_store.run(record)
```

quent starts executing synchronously. After each step, it inspects the return value
using a fast custom awaitable check (~10x faster than `inspect.isawaitable()`).
The moment it encounters an awaitable, execution seamlessly transitions to async
mode for the remainder of the pipeline. The caller gets back either a plain value or
a coroutine -- and the caller already knows which one it expects.

No annotations. No wrappers. No ceremony. The pipeline definition does not mention
sync or async at all.

### Real-World Example: A Library That Supports Both Callers

```python
from quent import Q

def process_data(data_source):
  """Works with both sync and async data sources."""
  return (
    Q(data_source.fetch)
    .then(validate)
    .then(transform)
    .then(data_source.save)
    .run()
  )

# Sync caller
result = process_data(sync_database)

# Async caller
result = await process_data(async_database)
```

The function `process_data` does not know or care whether `data_source` is sync or
async. It defines the pipeline once. quent handles the rest.

---

## Design Philosophy

quent is built on three principles.

### Minimum Viable Abstraction

Every feature must justify itself by solving a real sync/async bridging problem or
eliminating genuine code duplication. Features that merely wrap standard library
functionality do not belong.

quent provides pipeline primitives: `.then()`, `.do()`, `.foreach()`,
`.gather()`, `.with_()`, `.if_()`, `.except_()`, `.finally_()`. That is the entire
surface area. Each one exists because it solves a concrete problem in multi-step
pipelines that need to work across the sync/async boundary.

### Transparent

quent detects and handles async automatically. You do not annotate your pipeline as
sync or async. You do not use different APIs for the two cases. You do not think
about it.

The mechanism is straightforward: after evaluating each step, quent runs a fast
custom awaitable check on the result. If the result is a coroutine, Task, or
Future, execution transitions to async mode. If not, it stays sync. This happens
at runtime, per-step, with no user intervention.

### Unopinionated

quent imposes no patterns, no required base classes, no framework lock-in. It works
with any callable -- functions, methods, lambdas, classes with `__call__`,
coroutine functions, anything Python considers callable.

You can adopt quent in one function of one module. You can remove it just as easily.
There is no buy-in beyond the pipeline itself.

---

## What quent Is NOT

quent solves one problem well. It is important to understand what it is *not*:

**Not a collections library.** quent has `.foreach()` and `.foreach_do()` as
pipeline utilities, but it is not a replacement for `itertools`, `more-itertools`,
or list comprehensions. Those tools are better for pure collection processing.

**Not a framework.** quent has no opinions about your application structure. It does
not provide routing, dependency injection, middleware stacks, or lifecycle management.
It is a building block, not an architecture.

**Not a functional programming toolkit.** If you want monadic composition,
railway-oriented programming, or algebraic effects, use
[returns](https://github.com/dry-python/returns),
[toolz](https://github.com/pytoolz/toolz), or
[Expression](https://github.com/dbrattli/Expression). quent provides pipeline
composition, not a full FP type system.

**Not a task queue or job scheduler.** quent pipelines execute inline. They do not
distribute work across processes, schedule retries, or persist state. For that,
use Celery, Dramatiq, or similar tools.

---

## When to Use quent

!!! tip "Good fit"

    quent is the right tool when you are paying the sync/async tax -- when the same
    logic needs to work in both sync and async contexts.

**Libraries and frameworks that serve both sync and async callers.** If you maintain
a library with a sync API and an async API that share the same underlying logic,
quent lets you write that logic once.

**Incremental sync-to-async migration.** You have a sync codebase and you are
migrating to async one module at a time. quent lets existing sync code coexist with
new async code in the same pipelines, without rewriting everything at once.

**Multi-step processing pipelines.** ETL jobs, request handlers, validation pipelines,
data transformation pipelines -- anywhere you have a sequence of operations that
flows a value from step to step.

**Code that wraps external services with mixed sync/async clients.** Database
drivers, HTTP clients, message queues, and cache layers increasingly offer both sync
and async interfaces. quent lets you write wrapper code once.

**Decorator factories.** Wrap functions in reusable processing pipelines that work
regardless of whether the decorated function is sync or async.

---

## When NOT to Use quent

!!! warning "Not a good fit"

    quent is not a general-purpose tool. It solves one problem well. If you are not
    facing that problem, plain Python is the better choice.

**Pure sync code with no async plans.** If every callable in your pipeline is
synchronous and will stay that way, a chain of function calls or intermediate
variables is simpler and more readable. quent adds no value here.

**Pure async code.** If everything is already async and you have no sync callers,
just use `async`/`await` directly. quent's bridging capability is irrelevant when
there is nothing to bridge.

**Simple one-step operations.** Wrapping a single function call in a `Q` pipeline is
overhead with no benefit. quent is useful for multi-step pipelines, not individual
function calls.

**Trivial pipelines.** If your pipeline is three lines of straight-line code with
no branching, no error handling, and no reuse, plain Python is clearer:

```python
# Just write this directly -- no pipeline needed
data = fetch(source)
result = transform(data)
save(result)
```

**Do not add quent for the sake of it.** Add it when you are solving the sync/async
duplication problem. If you are not duplicating code across sync and async
boundaries, you do not need it.

---

## What Makes quent Different

### Zero Dependencies, Pure Python

quent has no runtime dependencies. It is pure Python, compatible with any
environment that runs Python 3.10+. No C extensions, no compiled components, no
transitive dependency tree to audit.

### PEP 561 Typed

Inline type annotations throughout the codebase. Your editor and type checker
understand quent's API. `py.typed` marker included for PEP 561 compliance.

### Enhanced Tracebacks

When an exception occurs inside a pipeline, quent injects a visualization into the
traceback showing the full pipeline and marking exactly which step failed:

```
Traceback (most recent call last):
  File "example.py", line 28, in <module>
    .run()
     ^^^^^
  File "<quent>", line 1, in
    Q(fetch_data, 42)
    .then(validate) <----
    .then(transform)
    .then(save)
  File "example.py", line 11, in validate
    raise ValueError("Value too large")
ValueError: Value too large
```

The `<----` marker points to the failing step. Internal quent frames are
automatically cleaned from the traceback. Disable with `QUENT_NO_TRACEBACK=1`
if you prefer raw tracebacks.

### Works with Any Callable

Functions, methods, lambdas, classes, coroutine functions, objects with `__call__`
-- if Python considers it callable, quent accepts it. No adapters, no wrappers, no
protocol implementations required.

---

## Next Steps

- **[Getting Started](getting-started.md)** -- install quent and build your first pipeline
- **[Pipelines](guide/pipelines.md)** -- comprehensive guide to all pipeline operations
- **[Async Handling](guide/async.md)** -- deep dive into the sync/async bridging mechanism
- **[Error Handling](guide/error-handling.md)** -- exception handlers, cleanup, and enhanced tracebacks
