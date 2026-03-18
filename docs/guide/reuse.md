---
title: "Reuse and Patterns -- Decorators, Nesting, Cloning"
description: "Reuse quent chains with clone, nested chains, and decorators. Common patterns for validation pipelines, ETL, request processing, and middleware."
tags:
  - patterns
  - decorator
  - clone
  - nested chains
  - reuse
search:
  boost: 4
---

# Reuse and Patterns

## Pipeline Reuse

A `Q` instance is a mutable object. Every call to `.then()`, `.do()`, `.foreach()`, or any other pipeline method appends a node to the internal linked list. This means extending a pipeline after creation modifies the same instance for all code that holds a reference.

Running a pipeline multiple times with different inputs is always safe:

```python
from quent import Q

pipeline = Q().then(validate).then(transform).then(save)

# Safe -- .run() does not modify the pipeline's structure
result_a = pipeline.run(input_a)
result_b = pipeline.run(input_b)
```

But appending steps after the pipeline is shared creates a problem:

```python
base = Q().then(validate).then(normalize)

# PROBLEM: this modifies `base` itself
base.then(serialize_json)

# Now `base` has three steps, not two.
# Any other code using `base` sees the extra step.
```

This is where `clone()` comes in.

---

## clone() -- Independent Copies

`q.clone()` creates a new, independent Q pipeline by copying the linked list structure. The new pipeline has its own nodes, so appending to one does not affect the other.

```python
from quent import Q

base = Q().then(validate).then(normalize)

for_api = base.clone().then(serialize_json)
for_db = base.clone().then(serialize_sql)
for_csv = base.clone().then(serialize_csv)
```

Each variant is independent. Extending `for_api` does not affect `for_db`, `for_csv`, or `base`.

### What clone Copies vs Shares

`clone()` copies the pipeline's **structure**, not its **contents**. This is a shallow structural copy:

**Copied (independent):**

- The linked list of nodes -- each clone has its own nodes
- Nested chains within steps are **recursively cloned** via their own `clone()` method, preventing cross-clone state sharing
- Conditional operations (`if_`/`else_`) are deep-copied (they carry mutable state)
- Error handlers (`except_`/`finally_`) have their nodes cloned. If the handler callable is a `Q` instance, it is recursively cloned
- Keyword argument dictionaries are shallow-copied (dicts are mutable)

**Shared by reference:**

- All callables (functions, lambdas, bound methods) -- except Q instances, which are always recursively cloned
- Values and argument objects (individual args tuple elements, kwargs dict values)
- Exception type tuples for `except_()`
- Positional argument tuples (tuples are immutable)

This means stateful callables are shared between original and clone:

```python
from quent import Q

class Counter:
  def __init__(self):
    self.count = 0
  def __call__(self, v):
    self.count += 1
    return v

counter = Counter()
base = Q().do(counter)
clone = base.clone()

base.run(1)   # counter.count = 1
clone.run(2)  # counter.count = 2 (same Counter instance)
```

If you need completely isolated state, create new callables for each clone.

### State Reset

Clones always behave as top-level pipelines, regardless of whether the original was being used as a nested pipeline. When a clone is subsequently used as a step in another pipeline, it adopts nested behavior at that point.

---

## Fork-and-Extend Pattern

The most common use of `clone()` is building a base pipeline and forking it into specialized variants:

```python
from quent import Q

base = (
  Q()
  .then(validate)
  .then(normalize)
  .except_(handle_error)
)

# Each fork gets its own pipeline with shared base steps
for_api = base.clone().then(serialize_json).then(send_to_api)
for_db = base.clone().then(serialize_sql).then(insert_to_db)
for_csv = base.clone().then(serialize_csv).then(write_to_file)

for_api.run(record)
for_db.run(record)
for_csv.run(record)
```

The `except_()` and `finally_()` configurations are cloned along with the pipeline steps.

---

## as_decorator() -- Wrap Pipeline as Function Decorator

`q.as_decorator()` returns a decorator that wraps a function with the pipeline. The decorated function's return value becomes the pipeline's input.

```python
from quent import Q

@Q().then(lambda x: x.strip()).then(str.upper).as_decorator()
def get_name():
  return '  alice  '

get_name()  # 'ALICE'
```

### How decorator Works

1. `as_decorator()` **clones** the pipeline internally. Modifying the original pipeline after calling `as_decorator()` does not affect the decorated function.
2. When the decorated function is called, its arguments are forwarded to the original function.
3. The function's return value is used as the run value for the cloned pipeline.
4. The pipeline's steps process the value, and the final result is returned.

```python
from quent import Q

q = Q().then(lambda x: x.upper()).then(lambda x: f"[{x}]")

@q.as_decorator()
def greet(name, greeting="Hello"):
  return f"{greeting}, {name}"

result = greet("World")
# greet("World") returns "Hello, World"
# pipeline processes: "Hello, World" -> "HELLO, WORLD" -> "[HELLO, WORLD]"
# result = "[HELLO, WORLD]"
```

### Async Transparency

Async transparency works with decorators too. If any step in the pipeline returns an awaitable, the decorated function returns a coroutine:

```python
@pipeline.as_decorator()
def handle(request):
  return parse(request)

# If pipeline steps are sync:
result = handle(request)

# If any step is async:
result = await handle(request)
```

### Internal Clone

`as_decorator()` clones the pipeline, so modifying the original after calling `as_decorator()` is safe:

```python
q = Q().then(step_a)

@q.as_decorator()
def my_func(x):
  return x

# Safe -- decorator() cloned internally
q.then(step_b)  # does not affect my_func
```

The decorated function preserves its original signature via `functools.wraps`.

---

## Nested Pipelines for Composition

A Q pipeline can be passed to `.then()` or `.do()` of another chain. The inner chain executes as a single step:

```python
from quent import Q

validate = Q().then(check_schema).then(check_permissions)
transform = Q().then(normalize).then(enrich)

pipeline = (
  Q(fetch_data)
  .then(validate)     # runs entire validate pipeline as one step
  .then(transform)    # runs entire transform pipeline as one step
  .then(save)
  .run()
)
```

### How Nesting Works

1. The current pipeline value is passed as input to the inner pipeline.
2. The inner pipeline executes all its steps.
3. For `.then()`, the inner pipeline's result replaces the current value. For `.do()`, the result is discarded.

### Control Flow Propagation

Control flow signals propagate through nested pipelines:

```python
from quent import Q

validate = (
  Q()
  .if_(lambda x: not x.get('valid')).then(lambda _: Q.return_({'error': 'invalid'}))
  .then(check_permissions)
)

pipeline = (
  Q()
  .then(validate)    # return_() propagates to outer pipeline
  .then(transform)   # skipped if validate returned early
  .then(save)
)

result = pipeline.run({'valid': False})
# result = {'error': 'invalid'}
```

### Independent Error Handling

Each nested pipeline's `except_()` and `finally_()` handlers apply only to that pipeline's execution. Unhandled exceptions propagate to the outer pipeline:

```python
from quent import Q

inner = (
  Q()
  .then(risky_operation)
  .except_(lambda ei: 'inner fallback')
)

outer = (
  Q()
  .then(inner)   # inner handles its own errors
  .then(process)
  .except_(lambda ei: 'outer fallback')
)
```

---

## iterate() / iterate\_do() -- Reusable Generators

`.iterate()` and `.iterate_do()` return a `QuentIterator` that supports reuse through calling:

```python
from quent import Q

gen = Q(fetch_page).iterate(transform_record)

# First run
for record in gen(page=1):
  process(record)

# Second run with different args
for record in gen(page=2):
  process(record)
```

Each call creates a fresh iterator with the specified run arguments. The original configuration (fn, side-effect mode) is preserved.

### Sync and Async Iteration

The same `QuentIterator` supports both protocols:

```python
# Sync
for item in gen(page=1):
  ...

# Async
async for item in gen(page=1):
  ...
```

---

## Building a Pipeline Library

### Reusable Sub-Pipeline Library

Build a library of nested chains:

```python
from quent import Q

# Library of reusable sub-pipelines
strip_whitespace = Q().foreach(str.strip)
remove_empty = Q().then(lambda xs: [x for x in xs if x])
lowercase = Q().foreach(str.lower)
sort_unique = Q().then(set).then(sorted)

# Compose for specific use cases
clean_tags = (
  Q()
  .then(strip_whitespace)
  .then(remove_empty)
  .then(lowercase)
  .then(sort_unique)
)

tags = clean_tags.run(["  Python ", "", "ASYNC ", "python", " Q"])
# tags = ['async', 'q', 'python']
```

---

## Real-World Composition Patterns

### Validation Pipeline

```python
from quent import Q

validate = (
  Q()
  .then(check_not_empty)
  .then(check_schema)
  .then(check_permissions)
  .except_(lambda ei: ValidationError(str(ei.exc)))
)

validate.run(user_input)
validate.run(api_payload)
validate.run(form_data)
```

### ETL Pipeline

```python
from quent import Q

etl = (
  Q()
  .then(extract)
  .then(transform)
  .then(load)
  .except_(handle_etl_error)
  .finally_(cleanup_resources)
)

etl.run(database_source)
etl.run(api_source)
etl.run(file_source)
```

### Middleware Pipeline

```python
from quent import Q

middleware = Q().then(authenticate).then(authorize).then(rate_limit)

@middleware.clone().then(handle_api).as_decorator()
def api_endpoint(request):
  return request

@middleware.clone().then(render_page).as_decorator()
def web_endpoint(request):
  return request

@middleware.clone().then(stream_events).as_decorator()
def sse_endpoint(request):
  return request
```

Each endpoint gets an independent copy of the middleware pipeline with a specific handler appended.

### Sync/Async Service Wrapper

```python
from quent import Q

class UserService:
  def __init__(self, db, cache):
    self.db = db
    self.cache = cache

  def get_user(self, user_id):
    return (
      Q(self.cache.get, user_id)
      .if_(lambda cached: cached is None).then(self.db.fetch_user, user_id)
      .do(lambda user: self.cache.set(user_id, user))
      .run()
    )

# Works with sync backends
service = UserService(sync_db, sync_cache)
user = service.get_user(123)

# Works with async backends -- same code
service = UserService(async_db, async_cache)
user = await service.get_user(123)
```

### Conditional Branching with Nested Pipelines

```python
from quent import Q

premium_flow = Q().then(premium_validate).then(premium_process)
standard_flow = Q().then(standard_validate).then(standard_process)

pipeline = (
  Q()
  .if_(is_premium).then(premium_flow)
  .else_(standard_flow)
  .then(finalize)
)

result = pipeline.run(user_request)
```

---

## Summary

| Mechanism | Purpose | When to Use |
|-----------|---------|-------------|
| `.run(value)` | Run the same pipeline with different inputs | Always safe; does not modify the pipeline |
| `.clone()` | Create an independent copy | When you need to extend a shared pipeline differently |
| Nested chains | Compose pipelines from reusable sub-pipelines | Modular, testable pipeline components |
| `.as_decorator()` | Wrap a chain as a function decorator | Process a function's return value through a pipeline |
| `.iterate()` | Create reusable iterators | Lazy, streaming consumption of pipeline output |

---

## Next Steps

- **[Pipelines](chains.md)** -- pipeline building, context managers, conditionals, and control flow
- **[Error Handling](error-handling.md)** -- exception handling and cleanup
- **[Async Handling](async.md)** -- sync/async bridging and concurrency
