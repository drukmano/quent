<div align="center">
  <h1>quent</h1>
  <p><strong>Write it once. Run it sync or async.</strong></p>
  <br>
  <p>
    <a href="https://pypi.org/project/quent/"><img src="https://img.shields.io/pypi/v/quent?style=flat-square" alt="PyPI"></a>
    &nbsp;
    <a href="https://pypi.org/project/quent/"><img src="https://img.shields.io/pypi/pyversions/quent?style=flat-square" alt="Python"></a>
    &nbsp;
    <a href="https://github.com/drukmano/quent/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/quent?style=flat-square" alt="License"></a>
    &nbsp;
    <a href="https://github.com/drukmano/quent/actions/workflows/ci.yml"><img src="https://github.com/drukmano/quent/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    &nbsp;
    <a href="https://app.codecov.io/gh/drukmano/quent"><img src="https://img.shields.io/codecov/c/github/drukmano/quent?style=flat-square" alt="Coverage"></a>
    &nbsp;
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
    &nbsp;
    <a href="https://pepy.tech/project/quent"><img src="https://static.pepy.tech/badge/quent/month" alt="Downloads"></a>
    &nbsp;
    <a href="https://quent.readthedocs.io"><img src="https://readthedocs.org/projects/quent/badge/?version=latest" alt="Docs"></a>
  </p>
</div>

---

<p align="center">
  A transparent sync/async bridge for Python.<br>
  Define a pipeline once — quent handles the rest.
</p>

---

- **One definition, two worlds** &mdash; a single chain works for both sync and async callers. Zero code duplication.
- **Zero ceremony** &mdash; no decorators, no base classes, no type wrappers. Just chain your functions.
- **Drop-in migration** &mdash; unify existing sync and async implementations into one pipeline. Stop maintaining two versions.
- **Pure Python** &mdash; zero runtime dependencies on Python 3.11+. Fully typed (PEP 561).
- **Focused** &mdash; every feature exists because removing it would force separate sync and async code paths.

---

## The Problem

Any codebase that supports both sync and async callers ends up maintaining two versions of the same logic:

```python
# Without quent -- the same pipeline, written twice

def process_sync(data):
  validated = validate_sync(data)
  transformed = transform_sync(validated)
  return save_sync(transformed)

async def process_async(data):
  validated = await validate_async(data)
  transformed = await transform_async(validated)
  return await save_async(transformed)
```

Every function, every pipeline, every utility &mdash; duplicated. When a bug is fixed in one version, the other falls out of sync. When a new step is added, it must be added in both places.

---

## The Solution

```python
# With quent -- write it once

pipeline = Chain().then(validate).then(transform).then(save)

result = pipeline.run(data)          # sync if all steps are sync
result = await pipeline.run(data)    # async if any step is async
```

One definition. The chain starts executing synchronously. The moment any step returns an awaitable, execution seamlessly transitions to async and stays there. The caller decides whether to `await`.

---

## Installation

```bash
pip install quent
```

**Requires Python 3.10+.** Supports 3.10 through 3.14, including free-threaded builds. Zero runtime dependencies on Python 3.11+ (`typing_extensions` on 3.10).

---

## Quick Start

```python
from quent import Chain

# Basic pipeline
result = Chain(5).then(lambda x: x * 2).then(lambda x: x + 1).run()
print(result)  # 11

# Side effects -- do() runs the function but passes the value through
result = Chain(42).then(lambda x: x * 2).do(print).then(str).run()  # prints: 84
print(result)  # '84'

# Works with any callable
result = Chain('  hello  ').then(str.strip).then(str.upper).run()
print(result)  # HELLO
```

The same chain works whether your functions are sync, async, or a mix:

```python
pipeline = Chain().then(fetch_data).then(validate).then(normalize)

# Sync context
result = pipeline.run(id)

# Async context -- same chain, no changes
result = await pipeline.run(id)
```

---

## Features

Build pipelines fluently. Every builder method returns `self` for chaining.

```python
from quent import Chain

result = (
  Chain(fetch_user)               # root callable
  .then(validate)                  # transform
  .do(log)                         # side-effect
  .foreach(normalize_field)        # per-element
  .gather(enrich, score)           # concurrent
  .then(merge)                     # combine
  .if_(has_premium).then(upgrade)  # conditional
  .except_(handle_error)           # error handling
  .finally_(cleanup)               # cleanup
  .run(user_id)                    # execute
)
```

<details>
<summary><strong>Collection Operations</strong> &mdash; foreach, foreach_do</summary>

<br>

```python
# foreach -- transform each element, collect results
Chain([1, 2, 3]).foreach(lambda x: x ** 2).run()  # [1, 4, 9]

# foreach_do -- side-effect per element, keep originals
Chain([1, 2, 3]).foreach_do(print).run()  # prints 1, 2, 3; returns [1, 2, 3]

# filter via list comprehension
Chain([1, 2, 3, 4, 5]).then(lambda xs: [x for x in xs if x % 2 == 0]).run()  # [2, 4]
```

</details>

<details>
<summary><strong>Concurrent Execution</strong> &mdash; gather, concurrency parameter</summary>

<br>

Run multiple functions on the same value concurrently:

```python
Chain('hello').gather(str.upper, len).run()  # ('HELLO', 5)
```

Limit concurrency on collection operations with the `concurrency` parameter. Uses `ThreadPoolExecutor` for sync callables and `asyncio.Semaphore` + `TaskGroup` for async:

```python
# Process up to 10 items at a time
Chain(urls).foreach(fetch, concurrency=10).run()

# Limit concurrent gather branches
Chain(data).gather(analyze, compress, upload, concurrency=5).run()
```

Pass a custom executor for sync concurrent operations:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
  Chain(urls).foreach(fetch, concurrency=4, executor=pool).run()
```

</details>

<details>
<summary><strong>Conditionals</strong> &mdash; if_ / else_</summary>

<br>

```python
Chain(5).if_(lambda x: x > 0).then(lambda x: x * 2).run()  # 10
Chain(-5).if_(lambda x: x > 0).then(str).else_(abs).run()   # 5

# When predicate is omitted, uses truthiness of the current value
Chain('hello').if_().then(str.upper).run()                     # 'HELLO'
Chain('').if_().then(str.upper).else_(lambda _: 'empty').run() # 'empty'

# Literal predicate -- truthiness used directly
Chain(value).if_(is_admin).then(grant_access).run()

# Side-effect conditional branch
Chain(user).if_(is_premium).do(log_premium_access).then(next_step).run()
```

</details>

<details>
<summary><strong>Context Managers</strong> &mdash; with_ / with_do</summary>

<br>

Transparently handles both sync and async context managers:

```python
Chain(open('data.txt')).with_(lambda f: f.read()).run()

# Side-effect variant (result discarded, original value passes through)
Chain(open('log.txt', 'w')).with_do(lambda f: f.write('done')).run()
```

</details>

<details>
<summary><strong>Error Handling</strong> &mdash; except_ / finally_</summary>

<br>

One exception handler and one finally handler per chain:

```python
from quent import Chain, ChainExcInfo

Chain(0).then(lambda x: 1 / x).except_(lambda ei: -1).run()  # -1

Chain(url)
  .then(fetch)
  .then(parse)
  .except_(handle_error, exceptions=ConnectionError)
  .finally_(cleanup)
  .run()
```

`except_` catches `Exception` by default. The handler receives a `ChainExcInfo(exc, root_value)` as its current value. Use `reraise=True` to re-raise after handling (handler runs for side-effects only). `finally_` always runs and receives the chain's root value.

</details>

<details>
<summary><strong>Control Flow</strong> &mdash; return_ / break_</summary>

<br>

```python
# Early return -- skips all remaining steps
Chain(5) \
  .then(lambda x: Chain.return_(x * 10) if x > 0 else x) \
  .then(str) \
  .run()  # 50 (str step is skipped)

# Break from iteration -- break value is appended to partial results
Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_(x) if x == 3 else x * 2).run()
# [2, 4, 3]
```

</details>

<details>
<summary><strong>Composition</strong> &mdash; clone, decorator</summary>

<br>

**clone** &mdash; fork-and-extend without modifying the original:

```python
base = Chain().then(validate).then(normalize)
for_api = base.clone().then(to_json)    # base is untouched
for_db  = base.clone().then(to_record)  # independent copy
```

**decorator** &mdash; wrap a chain as a function decorator:

```python
@Chain().then(lambda x: x.strip()).then(str.upper).decorator()
def get_name():
  return '  alice  '

get_name()  # 'ALICE'
```

</details>

<details>
<summary><strong>Iteration</strong> &mdash; iterate / iterate_do</summary>

<br>

Dual sync/async generators over chain output:

```python
for item in Chain(range(5)).iterate(lambda x: x ** 2):
  print(item)  # 0, 1, 4, 9, 16

async for item in Chain(async_source).iterate(transform):
  print(item)  # works with async sources too
```

</details>

---

### Calling Conventions

How arguments flow through the pipeline is determined by two rules, checked in priority order:

| Condition | Behavior |
|:----------|:---------|
| Explicit args/kwargs provided | Call `fn(*args, **kwargs)` -- current value NOT passed |
| No args (default) | Call `fn(current_value)`, `fn()` if no value, or return value as-is if non-callable |

```python
Chain(5).then(str).run()                    # str(5) -- current value passed
Chain(5).then(print, 'hello').run()         # print('hello') -- explicit args used
```

---

### Enhanced Tracebacks

When an exception occurs inside a chain, quent injects a visualization directly into the traceback showing exactly which step failed:

```
Traceback (most recent call last):
  ...
  File "<quent>", line 0, in Chain(fetch_data).then(validate).then(transform) <----.do(log)
  ...
ZeroDivisionError: division by zero
```

The `<----` marker points to the step that raised. Internal quent frames are cleaned from the traceback. On Python 3.11+, a concise exception note is also attached.

Opt out by setting `QUENT_NO_TRACEBACK=1` before importing quent.

---

## API Reference

### Constructor

```python
Chain(v=Null, /, *args, **kwargs)
```

<br>

### Pipeline Building

All methods return `self` for fluent chaining.

| Method | Description |
|:-------|:------------|
| `.then(v, /, *args, **kwargs)` | Append step; result replaces current value |
| `.do(fn, /, *args, **kwargs)` | Side-effect step; fn must be callable, result discarded |
| `.foreach(fn, /, *, concurrency=None, executor=None)` | Transform each element, collect results |
| `.foreach_do(fn, /, *, concurrency=None, executor=None)` | Side-effect per element, keep originals |
| `.gather(*fns, concurrency=-1, executor=None)` | Run multiple fns on current value, collect results as tuple |
| `.with_(fn, /, *args, **kwargs)` | Enter current value as context manager, call fn |
| `.with_do(fn, /, *args, **kwargs)` | Same as with_, but fn result discarded |
| `.if_(predicate=None, /, *args, **kwargs)` | Begin conditional; must be followed by `.then()` or `.do()` |
| `.if_(...).then(fn, /, *args, **kwargs)` | Conditional transform -- runs fn if predicate is truthy, result replaces current value |
| `.if_(...).do(fn, /, *args, **kwargs)` | Conditional side-effect -- runs fn if predicate is truthy, result discarded |
| `.else_(v, /, *args, **kwargs)` | Else branch (must follow `.then()` or `.do()`) |
| `.else_do(fn, /, *args, **kwargs)` | Side-effect else branch (result discarded) |
| `.except_(fn, /, *args, exceptions=None, reraise=False, **kwargs)` | Exception handler (one per chain) |
| `.finally_(fn, /, *args, **kwargs)` | Cleanup handler (one per chain) |
| `.name(label)` | Assign a label for traceback identification |

### Execution

| Method | Description |
|:-------|:------------|
| `.run(v=Null, /, *args, **kwargs)` | Execute the chain; returns value or coroutine |
| `chain(...)` | Alias for `.run()` |

### Reuse and Iteration

| Method | Description |
|:-------|:------------|
| `.decorator()` | Wrap chain as a function decorator |
| `.iterate(fn=None)` | Dual sync/async generator over output |
| `.iterate_do(fn=None)` | Like iterate, fn results discarded |
| `.clone()` | Deep copy for fork-and-extend |

### Control Flow

<sub>Class methods</sub>

| Method | Description |
|:-------|:------------|
| `Chain.return_(v=Null, /, *args, **kwargs)` | Signal early return from chain |
| `Chain.break_(v=Null, /, *args, **kwargs)` | Signal break from iteration; value is appended to partial results |

### Exports and Instrumentation

| Name | Description |
|:-----|:------------|
| `Chain` | Main pipeline class |
| `ChainExcInfo` | NamedTuple `(exc, root_value)` passed to except handlers |
| `ChainIterator` | Type alias for `.iterate()` / `.iterate_do()` return values |
| `Null` | Sentinel for "no value provided" (distinct from `None`) |
| `QuentException` | Exception type for quent-specific errors |
| `__version__` | Package version string |
| `Chain.on_step` | Optional callback `(chain, step_name, input_value, result, elapsed_ns)` for instrumentation |

> **Note:** Chain objects cannot be pickled (security measure -- see [Troubleshooting](https://quent.readthedocs.io/en/latest/troubleshooting/#8-typeerror-when-pickling-a-chain)). Define chains at module level and reference by name instead of serializing.

---

## Examples

See the [examples/](examples/) directory for complete, runnable recipes covering ETL pipelines, API gateways, fan-out/fan-in patterns, retry with backoff, and testing chains.

---

## Documentation

Full documentation &mdash; including guides, advanced usage, recipes, and framework integration examples &mdash; is available at **[quent.readthedocs.io](https://quent.readthedocs.io)**.

---

## Contributing

See the [contributing guide](https://github.com/drukmano/quent/blob/master/.github/CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

```bash
git clone https://github.com/drukmano/quent.git
cd quent
uv sync --group dev       # or: pip install -e . && pip install coverage ruff mypy
bash scripts/run_tests.sh
```

---

<p align="center">
  <a href="https://quent.readthedocs.io"><strong>Docs</strong></a>
  &nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/drukmano/quent"><strong>GitHub</strong></a>
  &nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://pypi.org/project/quent/"><strong>PyPI</strong></a>
  &nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://quent.readthedocs.io/en/latest/getting-started/"><strong>Getting Started</strong></a>
  &nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://github.com/drukmano/quent/releases"><strong>Changelog</strong></a>
</p>

<p align="center">
  <sub>MIT &mdash; Copyright (c) 2023&ndash;2026 Ohad Drukman</sub>
</p>
