# Quent vs pipe

Both Quent and [pipe](https://pypi.org/project/pipe/) provide function composition and chaining in Python. They differ in scope, async support, and philosophy.

## What pipe Does

pipe is a lightweight library that provides an infix pipe operator (`|`) for function composition. It focuses on iterable processing with a functional programming style.

### pipe Example

```python
from pipe import select, where, sort

result = (
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  | where(lambda x: x % 2 == 0)
  | select(lambda x: x * 10)
  | sort(reverse=True)
  | list
)
# [100, 80, 60, 40, 20]
```

pipe also supports custom pipe functions via the `@Pipe` decorator:

```python
from pipe import Pipe

@Pipe
def double(iterable):
  for item in iterable:
    yield item * 2

result = [1, 2, 3] | double | list
# [2, 4, 6]
```

## What Quent Does Differently

Quent provides a general-purpose chain interface (not limited to iterables) with transparent async handling, resilience features, and error handling.

### Quent Example

```python
from quent import Chain, run

# General pipeline -- not limited to iterables
result = Chain(fetch_user, user_id) | validate | transform | save | run()

# With resilience and error handling
result = (
  Chain(fetch_user, user_id)
  .retry(3, delay=0.5)
  .timeout(10.0)
  .then(validate)
  .then(transform)
  .then(save)
  .except_(handle_error)
  .run()
)
```

## Side-by-Side: Function Composition

### pipe Approach

```python
from pipe import Pipe

@Pipe
def validate(data):
  if not data:
    raise ValueError("Empty")
  return data

@Pipe
def normalize(data):
  return {k.lower(): v for k, v in data.items()}

@Pipe
def save(data):
  db.insert(data)
  return data

result = fetch_data(id) | validate | normalize | save
```

### Quent Approach

```python
from quent import Chain, run

result = Chain(fetch_data, id) | validate | normalize | save | run()
```

Both approaches look similar for simple composition. The differences emerge when you need async support, error handling, or resilience.

## Comparison

| Aspect | pipe | Quent |
|--------|------|-------|
| **Primary focus** | Iterable processing, functional composition | General pipeline with resilience |
| **Pipe syntax** | `value \| fn` (via `@Pipe` decorator) | `Chain(v) \| fn \| run()` |
| **Async support** | No | Yes (transparent sync/async) |
| **Retry** | No | Yes (`.retry()`) |
| **Timeout** | No | Yes (`.timeout()`) |
| **Error handling** | No (use try/except) | Yes (`.except_()`, `.finally_()`) |
| **Context propagation** | No | Yes (`.with_context()`) |
| **Side effects** | No dedicated support | Yes (`.do()`) |
| **Chain reuse** | No | Yes (`.clone()`, `.freeze()`) |
| **Iterable operations** | Core strength (select, where, sort, etc.) | Via `.foreach()`, `.iterate()`, `.filter()`, `.reduce()` |
| **Performance** | Pure Python | Cython-compiled |
| **Dependencies** | Zero | Zero |
| **Python version** | 3.7+ | 3.14+ |

## Where pipe Excels

pipe is particularly strong for iterable processing with its built-in operations:

```python
from pipe import select, where, groupby, dedup, sort

result = (
  users
  | where(lambda u: u.active)
  | select(lambda u: u.name)
  | dedup
  | sort
  | list
)
```

This reads naturally and is concise for data filtering and transformation. Quent does not provide equivalent built-in iterable operations (though you can achieve similar results with `.foreach()`, `.filter()`, and `.reduce()`).

## Where Quent Excels

Quent is stronger when you need features beyond simple composition:

```python
from quent import Chain

# Async-transparent pipeline with resilience
result = (
  Chain(fetch_users)            # sync or async -- Quent handles both
  .retry(3, delay=1.0)         # retry on failure
  .timeout(10.0)               # enforce time limit
  .then(validate)
  .then(transform)
  .then(save)
  .except_(handle_error)       # structured error handling
  .finally_(cleanup, ...)      # cleanup that always runs
  .with_context(trace_id=tid)  # context propagation
  .run()
)
```

None of these features (async handling, retry, timeout, except_, finally_, context) are available in pipe.

## When to Use Which

**Use pipe when:**

- You primarily work with iterable processing (filter, map, sort)
- You want lightweight functional composition for synchronous code
- You need to support older Python versions (3.7+)
- You do not need async support, retry, timeout, or error handling in the pipeline

**Use Quent when:**

- You need transparent sync/async handling
- You want retry, timeout, and error handling composed in the pipeline
- You are building service layers, API clients, or data processing pipelines
- You need chain reuse (clone, freeze, decorator)
- Performance matters (Cython-compiled)

**Both can coexist.** pipe is great for iterable transformations. Quent is great for general-purpose pipelines with resilience. You could use pipe for data processing within a Quent chain operation.

## Further Reading

- [Chains & Cascades](../guide/chains.md) -- Quent's chain types and pipe operator
- [Async Handling](../guide/async.md) -- How transparent async works
- [Resilience](../guide/resilience.md) -- Retry, timeout, and safe_run
- [Quent vs unasync](vs-unasync.md) -- Comparison with the code generation approach
- [Quent vs tenacity](vs-tenacity.md) -- Comparison with the retry library
