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

## Chain Reuse

A `Chain` is a mutable object. Every call to `.then()`, `.do()`, `.foreach()`, or any other pipeline method appends a node to the internal linked list. This means extending a chain after creation modifies the same instance for all code that holds a reference.

Running a chain multiple times with different inputs is always safe:

```python
from quent import Chain

pipeline = Chain().then(validate).then(transform).then(save)

# Safe -- .run() does not modify the chain's structure
result_a = pipeline.run(input_a)
result_b = pipeline.run(input_b)
```

But appending steps after the chain is shared creates a problem:

```python
base = Chain().then(validate).then(normalize)

# PROBLEM: this modifies `base` itself
base.then(serialize_json)

# Now `base` has three steps, not two.
# Any other code using `base` sees the extra step.
```

This is where `clone()` comes in.

---

## clone() -- Independent Copies

`chain.clone()` creates a new, independent Chain by copying the linked list structure. The new chain has its own nodes, so appending to one does not affect the other.

```python
from quent import Chain

base = Chain().then(validate).then(normalize)

for_api = base.clone().then(serialize_json)
for_db = base.clone().then(serialize_sql)
for_csv = base.clone().then(serialize_csv)
```

Each variant is independent. Extending `for_api` does not affect `for_db`, `for_csv`, or `base`.

### What clone Copies vs Shares

`clone()` copies the chain's **structure**, not its **contents**. This is a shallow structural copy:

**Copied (independent):**

- The linked list of nodes -- each clone has its own nodes
- Nested chains within steps are **recursively cloned** via their own `clone()` method, preventing cross-clone state sharing
- Conditional operations (`if_`/`else_`) are deep-copied (they carry mutable state)
- Error handlers (`except_`/`finally_`) have their nodes cloned. If the handler callable is a `Chain`, it is recursively cloned
- Keyword argument dictionaries are shallow-copied (dicts are mutable)

**Shared by reference:**

- All callables (functions, lambdas, bound methods) -- except Chain instances, which are always recursively cloned
- Values and argument objects (individual args tuple elements, kwargs dict values)
- Exception type tuples for `except_()`
- Positional argument tuples (tuples are immutable)

This means stateful callables are shared between original and clone:

```python
from quent import Chain

class Counter:
  def __init__(self):
    self.count = 0
  def __call__(self, v):
    self.count += 1
    return v

counter = Counter()
base = Chain().do(counter)
clone = base.clone()

base.run(1)   # counter.count = 1
clone.run(2)  # counter.count = 2 (same Counter instance)
```

If you need completely isolated state, create new callables for each clone.

### State Reset

Clones always behave as top-level chains, regardless of whether the original was being used as a nested chain. When a clone is subsequently used as a step in another chain, it adopts nested behavior at that point.

---

## Fork-and-Extend Pattern

The most common use of `clone()` is building a base pipeline and forking it into specialized variants:

```python
from quent import Chain

base = (
  Chain()
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

## decorator() -- Wrap Chain as Function Decorator

`chain.decorator()` returns a decorator that wraps a function with the chain's pipeline. The decorated function's return value becomes the chain's input.

```python
from quent import Chain

@Chain().then(lambda x: x.strip()).then(str.upper).decorator()
def get_name():
  return '  alice  '

get_name()  # 'ALICE'
```

### How decorator Works

1. `decorator()` **clones** the chain internally. Modifying the original chain after calling `decorator()` does not affect the decorated function.
2. When the decorated function is called, its arguments are forwarded to the original function.
3. The function's return value is used as the run value for the cloned chain.
4. The chain's steps process the value, and the final result is returned.

```python
from quent import Chain

chain = Chain().then(lambda x: x.upper()).then(lambda x: f"[{x}]")

@chain.decorator()
def greet(name, greeting="Hello"):
  return f"{greeting}, {name}"

result = greet("World")
# greet("World") returns "Hello, World"
# chain processes: "Hello, World" -> "HELLO, WORLD" -> "[HELLO, WORLD]"
# result = "[HELLO, WORLD]"
```

### Async Transparency

Async transparency works with decorators too. If any step in the chain returns an awaitable, the decorated function returns a coroutine:

```python
@pipeline.decorator()
def handle(request):
  return parse(request)

# If pipeline steps are sync:
result = handle(request)

# If any step is async:
result = await handle(request)
```

### Internal Clone

`decorator()` clones the chain, so modifying the original after calling `decorator()` is safe:

```python
chain = Chain().then(step_a)

@chain.decorator()
def my_func(x):
  return x

# Safe -- decorator() cloned internally
chain.then(step_b)  # does not affect my_func
```

The decorated function preserves its original signature via `functools.wraps`.

---

## Nested Chains for Composition

A Chain can be passed to `.then()` or `.do()` of another chain. The inner chain executes as a single step:

```python
from quent import Chain

validate = Chain().then(check_schema).then(check_permissions)
transform = Chain().then(normalize).then(enrich)

pipeline = (
  Chain(fetch_data)
  .then(validate)     # runs entire validate chain as one step
  .then(transform)    # runs entire transform chain as one step
  .then(save)
  .run()
)
```

### How Nesting Works

1. The current pipeline value is passed as input to the inner chain.
2. The inner chain executes all its steps.
3. For `.then()`, the inner chain's result replaces the current value. For `.do()`, the result is discarded.

### Control Flow Propagation

Control flow signals propagate through nested chains:

```python
from quent import Chain

validate = (
  Chain()
  .if_(lambda x: not x.get('valid')).then(lambda _: Chain.return_({'error': 'invalid'}))
  .then(check_permissions)
)

pipeline = (
  Chain()
  .then(validate)    # return_() propagates to outer chain
  .then(transform)   # skipped if validate returned early
  .then(save)
)

result = pipeline.run({'valid': False})
# result = {'error': 'invalid'}
```

### Independent Error Handling

Each nested chain's `except_()` and `finally_()` handlers apply only to that chain's execution. Unhandled exceptions propagate to the outer chain:

```python
from quent import Chain

inner = (
  Chain()
  .then(risky_operation)
  .except_(lambda ei: 'inner fallback')
)

outer = (
  Chain()
  .then(inner)   # inner handles its own errors
  .then(process)
  .except_(lambda ei: 'outer fallback')
)
```

---

## iterate() / iterate\_do() -- Reusable Generators

`.iterate()` and `.iterate_do()` return a `ChainIterator` that supports reuse through calling:

```python
from quent import Chain

gen = Chain(fetch_page).iterate(transform_record)

# First run
for record in gen(page=1):
  process(record)

# Second run with different args
for record in gen(page=2):
  process(record)
```

Each call creates a fresh iterator with the specified run arguments. The original configuration (fn, side-effect mode) is preserved.

### Sync and Async Iteration

The same `ChainIterator` supports both protocols:

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
from quent import Chain

# Library of reusable sub-pipelines
strip_whitespace = Chain().foreach(str.strip)
remove_empty = Chain().then(lambda xs: [x for x in xs if x])
lowercase = Chain().foreach(str.lower)
sort_unique = Chain().then(set).then(sorted)

# Compose for specific use cases
clean_tags = (
  Chain()
  .then(strip_whitespace)
  .then(remove_empty)
  .then(lowercase)
  .then(sort_unique)
)

tags = clean_tags.run(["  Python ", "", "ASYNC ", "python", " Chain"])
# tags = ['async', 'chain', 'python']
```

---

## Real-World Composition Patterns

### Validation Pipeline

```python
from quent import Chain

validate = (
  Chain()
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
from quent import Chain

etl = (
  Chain()
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

### Middleware Chain

```python
from quent import Chain

middleware = Chain().then(authenticate).then(authorize).then(rate_limit)

@middleware.clone().then(handle_api).decorator()
def api_endpoint(request):
  return request

@middleware.clone().then(render_page).decorator()
def web_endpoint(request):
  return request

@middleware.clone().then(stream_events).decorator()
def sse_endpoint(request):
  return request
```

Each endpoint gets an independent copy of the middleware chain with a specific handler appended.

### Sync/Async Service Wrapper

```python
from quent import Chain

class UserService:
  def __init__(self, db, cache):
    self.db = db
    self.cache = cache

  def get_user(self, user_id):
    return (
      Chain(self.cache.get, user_id)
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

### Conditional Branching with Nested Chains

```python
from quent import Chain

premium_flow = Chain().then(premium_validate).then(premium_process)
standard_flow = Chain().then(standard_validate).then(standard_process)

pipeline = (
  Chain()
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
| `.run(value)` | Run the same chain with different inputs | Always safe; does not modify the chain |
| `.clone()` | Create an independent copy | When you need to extend a shared chain differently |
| Nested chains | Compose pipelines from reusable sub-pipelines | Modular, testable pipeline components |
| `.decorator()` | Wrap a chain as a function decorator | Process a function's return value through a pipeline |
| `.iterate()` | Create reusable iterators | Lazy, streaming consumption of chain output |

---

## Next Steps

- **[Chains](chains.md)** -- pipeline building, context managers, conditionals, and control flow
- **[Error Handling](error-handling.md)** -- exception handling and cleanup
- **[Async Handling](async.md)** -- sync/async bridging and concurrency
