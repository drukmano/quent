---
title: "Quent vs Returns, Toolz, Pipe, Expression"
description: "How quent compares to returns, toolz, pipe, and Expression. Different tools for different problems — find the right one for your use case."
tags:
  - comparison
  - returns
  - toolz
  - pipe
  - functional programming
search:
  boost: 5
---

# Quent vs Returns, Toolz, Pipe, Expression

!!! note "Version information"
    Comparisons reflect the libraries' documented capabilities. Each library is actively maintained and evolving.

## Different Tools for Different Problems

quent is **not** a functional programming library. It is a sync/async bridge that
happens to use a pipeline syntax. The libraries on this page -- returns, toolz,
pipe, and Expression -- are functional programming tools. They solve different
problems, but because they all involve "piping values through functions," they
are often compared.

This page explains what each library actually does, where it overlaps with quent,
and when you should use one over the other.

## quent vs returns

### What returns is

[returns](https://github.com/dry-python/returns) (dry-python/returns) brings
typed monadic containers to Python. It implements railway-oriented programming
with containers like `Result`, `Maybe`, `IO`, and `Future`.

### What returns provides

returns gives you **type-safe containers** that enforce error handling at the
type level:

```python
from returns.result import Result, Success, Failure

def divide(a: float, b: float) -> Result[float, str]:
  if b == 0:
    return Failure("division by zero")
  return Success(a / b)

# The caller must handle both cases -- the type system enforces it
result = divide(10, 0)
# result is Failure("division by zero")
```

Composition uses `flow()` with pointfree combinators like `bind()`:

```python
from returns.pipeline import flow
from returns.pointfree import bind

result = flow(
  user_id,
  _make_request,
  bind(_parse_json),
  bind(_validate),
)
```

returns also provides `Future` and `FutureResult` containers for async code,
allowing you to compose async operations while maintaining the Result pattern.

### The key difference

**returns is about type-safe error handling.** It forces you to handle errors
through the type system using containers like `Result[Success, Failure]` and
`Maybe[Some, Nothing]`. You wrap values in containers and compose functions that
operate on those containers.

**quent is about sync/async unification.** It lets you write a pipeline once and
run it with sync or async callables. Values flow through the pipeline unwrapped --
plain Python objects, not monadic containers.

```python
# returns: values live inside containers
from returns.result import Success
result = Success(42).bind(lambda x: Success(x * 2))
# result is Success(84) -- still wrapped

# quent: values flow through unwrapped
from quent import Q
result = Q(42).then(lambda x: x * 2).run()
# result is 84 -- a plain int
```

### When to use each

**Use returns when:**

- You want the type system to enforce error handling. `Result` makes it
  impossible to forget to handle the error case.
- You are building domain logic with complex error flows and want compile-time
  (mypy) guarantees about error handling.
- You want monadic composition -- `bind`, `foreach`, `alt`, `rescue`, and
  the rest of the pointfree vocabulary.
- You prefer railway-oriented programming as an architectural pattern.

**Use quent when:**

- You need sync/async bridging. returns' `Future` container handles async, but
  it does not transparently bridge sync and async the way quent does. With quent,
  the same pipeline works with sync callables, async callables, or any mix.
- You want pipeline features like error handling, context manager handling,
  concurrent execution via `gather()`, and enhanced tracebacks.
- You prefer working with plain Python values rather than wrapping everything
  in container types.
- You want zero dependencies.

### They solve different problems

returns and quent are not competitors in the usual sense. returns is a
functional programming framework for type-safe composition. quent is a pipeline
engine for sync/async bridging. If you need both -- monadic error handling AND
sync/async bridging -- you could use returns containers inside quent pipelines, or
use quent pipelines inside returns flows. They are not mutually exclusive.

## quent vs toolz

### What toolz is

[toolz](https://github.com/pytoolz/toolz) is a collection of functional
programming utilities for Python: `pipe`, `curry`, `compose`, `memoize`,
`groupby`, and many more.

### What toolz provides

toolz gives you **functional programming primitives** for composing functions
and working with iterables:

```python
from toolz import pipe, curry, compose

# pipe: pass a value through a sequence of functions
result = pipe(data, validate, transform, serialize)

# curry: partial application
@curry
def add(x, y):
  return x + y

add5 = add(5)  # returns a function that adds 5

# compose: create a new function from a chain of functions
process = compose(serialize, transform, validate)
result = process(data)
```

toolz also provides iterable utilities (`groupby`, `unique`, `interpose`,
`partition`) and dictionary utilities (`assoc`, `merge`, `update_in`).

### The key difference

**toolz is a functional programming utility belt.** It gives you building blocks
for function composition, currying, memoization, and iterable processing.
Everything is synchronous.

**quent is a pipeline execution engine with async bridging.** It runs a sequence
of steps and handles sync/async transitions automatically.

The surface-level similarity is `pipe` vs quent:

```python
# toolz
from toolz import pipe
result = pipe(data, validate, transform, save)

# quent
from quent import Q
result = Q(data).then(validate).then(transform).then(save).run()
```

These look similar for the simple case, but they diverge quickly:

- toolz `pipe` is synchronous only. If `validate` returns a coroutine, `pipe`
  passes the coroutine object to `transform` -- it does not await it.
- quent detects the awaitable and transitions to async mode automatically.

### When to use each

**Use toolz when:**

- You want pure functional programming utilities -- currying, memoization,
  function composition, and iterable processing.
- Everything is synchronous.
- You want a mature, widely-used FP toolkit with a large API surface.
- You need utilities beyond pipelines -- `curry`, `memoize`, `groupby`,
  `merge`, `assoc`, etc.

**Use quent when:**

- You need async support. toolz is sync-only.
- You want pipeline features: error handling with `except_()`, concurrent
  execution with `gather()`, context managers with `with_()`.
- You need the pipeline to handle both sync and async callables in the same
  definition.

### Complementary, not competing

toolz and quent have almost no overlap. toolz provides FP utilities (curry,
memoize, compose); quent provides sync/async pipeline execution. You can use
toolz's `curry` to prepare functions and then pass them to quent's `.then()`.
They work well together.

## quent vs pipe

### What pipe is

[Pipe](https://github.com/JulienPalard/Pipe) provides infix syntax for piping
values through functions using Python's `|` operator.

### What pipe provides

pipe gives you **syntactic sugar** for chaining iterable operations with the `|`
operator:

```python
from pipe import select, where, take

result = (
  range(100)
  | where(lambda x: x % 2 == 0)
  | select(lambda x: x ** 2)
  | take(5)
  | list
)
# result = [0, 4, 16, 36, 64]
```

All pipes operate on iterables and return iterables, using lazy evaluation
via generators throughout. You can also create custom pipes with the `@Pipe`
decorator.

### The key difference

**pipe is syntax sugar for iterable processing.** It provides a nicer syntax for
chaining operations on iterables, similar to Unix pipes. It is small, focused,
and elegant.

**quent is a pipeline execution engine.** It manages value flow, error handling,
context managers, conditionals, and sync/async bridging.

For pure iterable processing, pipe's syntax is arguably cleaner. But pipe does
not handle async callables, error handling, non-iterable pipelines, context
managers, or conditional logic.

### When to use each

**Use pipe when:**

- You want minimal, elegant syntax for iterable processing.
- Everything is synchronous and iterable-based.
- You want lazy evaluation via generators.
- You do not need error handling or any pipeline infrastructure.

**Use quent when:**

- You need async support.
- You are building general-purpose pipelines, not just iterable processing.
- You need error handling, context managers, or conditional logic.
- You need non-iterable value transformation.

## quent vs Expression

### What Expression is

[Expression](https://github.com/dbrattli/Expression) brings F#-inspired
functional programming patterns to Python: `pipe`, `Option`, `Result`,
computational expressions, and immutable collections.

### What Expression provides

Expression gives you **F#-style patterns** in Python:

```python
from expression import pipe, Ok, Error
from expression.collections import seq

# Pipe: F#-style function composition
result = pipe(
  data,
  seq.foreach(transform),
  seq.filter(validate),
  seq.fold(accumulate, initial),
)

# Option: Maybe monad
from expression import Some, Nothing

value = Some(42).foreach(lambda x: x * 2)
# value is Some(84)

# Result: Railway-oriented programming
def safe_divide(a, b):
  if b == 0:
    return Error("division by zero")
  return Ok(a / b)
```

Expression also provides computational expressions via decorators
(`@effect.option`, `@effect.result`) for cleaner monadic composition,
tagged unions, and immutable collection types (`Seq`, `Block`, `Map`).

### The key difference

**Expression is an F#-flavored functional programming framework.** It provides
F#-style patterns -- `Option`, `Result`, `pipe`, computational expressions,
tagged unions, and immutable collections.

**quent is a sync/async pipeline bridge.** It provides a pipeline-based API for
running callables with transparent async detection.

Expression does have some async support through `AsyncResult` and `AsyncSeq`,
but its focus is bringing F# patterns to Python, not bridging the sync/async
divide.

### When to use each

**Use Expression when:**

- You want F#-style functional programming in Python.
- You are familiar with F# patterns and want to use them in Python projects.
- You want computational expressions, tagged unions, or immutable collections.
- You want Option/Result types with a different flavor than returns.

**Use quent when:**

- You need sync/async bridging. Expression's async support is pattern-specific,
  not a transparent bridge.
- You want pipeline features like error handling, concurrent execution.
- You prefer working with plain Python values.
- You want zero dependencies.

## Comparison Matrix

| Feature | quent | returns | toolz | pipe | Expression |
|---------|-------|---------|-------|------|------------|
| **Primary purpose** | Sync/async bridge | Monadic FP | FP utilities | Syntax sugar | F#-style FP |
| **Sync/async bridge** | Transparent | Partial (Future) | No | No | Partial (AsyncResult) |
| **Pipeline syntax** | `.then()` pipeline | `flow()` + `bind()` | `pipe()` / `compose()` | `\|` operator | `pipe()` / fluent |
| **Error handling** | `except_()`, `finally_()` | Result/Maybe types | No | No | Result/Option types |
| **Runtime dependencies** | 0 | Several | 0 | 0 | Minimal |
| **PEP 561 typed** | Yes | Yes (+ mypy plugin) | No | No | Yes |
| **Context managers** | `with_()`, `with_do()` | No | No | No | No |
| **Concurrent execution** | `gather()` | No | No | No | No |
| **Enhanced tracebacks** | Yes | No | No | No | No |
| **Currying/memoization** | No | No | Yes | No | No |
| **Monadic containers** | No | Yes | No | No | Yes |
| **Iterable utilities** | `foreach`, `foreach_do` | No | Yes (extensive) | Yes (via `\|`) | Yes (Seq, Block) |
| **Immutable collections** | No | No | No | No | Yes |
| **Python requirement** | 3.10+ | 3.10+ | 3.9+ | 3.8+ | 3.10+ |
| **Values** | Plain Python objects | Wrapped in containers | Plain Python objects | Iterables | Wrapped in containers |

## Choosing the Right Tool

quent occupies a specific niche: **sync/async pipeline bridging**. Every feature
exists because removing it would force users to write separate sync and async
code paths. It is not trying to be a functional programming framework, and it is
not competing with returns, toolz, pipe, or Expression on their home turf.

- **Sync/async bridging** -- use quent. No other library on this page provides
  transparent bridging where the same definition works with sync callables,
  async callables, or any mix of both.
- **Type-safe error handling** -- use returns or Expression.
- **FP utilities** (curry, memoize, compose) -- use toolz.
- **Iterable syntax sugar** -- use pipe for simple cases.
- **Pipeline infrastructure** (error handling + context managers + concurrent
  execution + tracebacks, all across the sync/async boundary) --
  use quent.

These tools are largely complementary. You can use toolz's `curry` to prepare
functions for quent pipelines. You can use returns' `Result` type inside quent
pipelines. There is no reason to choose only one if your project benefits from
multiple.

## Further Reading

- [Why Quent](../why-quent.md) -- the problem quent solves and when to use it
- [Getting Started](../getting-started.md) -- install quent and build your first pipeline
- [Quent vs unasync](vs-unasync.md) -- how quent compares to the code-generation approach
