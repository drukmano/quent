<div align="center">
  <img src="https://raw.githubusercontent.com/drukmano/quent/master/docs/assets/logo.png" alt="quent" width="200">
  <h1>quent</h1>
  <p><strong>Write it once. Run it sync or async.</strong></p>
  <p>
    <a href="https://pypi.org/project/quent/"><img src="https://img.shields.io/pypi/v/quent?style=flat-square" alt="PyPI"></a>
    &nbsp;
    <a href="https://pypi.org/project/quent/"><img src="https://img.shields.io/pypi/pyversions/quent?style=flat-square" alt="Python"></a>
    &nbsp;
    <a href="https://github.com/drukmano/quent/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/quent?style=flat-square" alt="License"></a>
    &nbsp;
    <a href="https://github.com/drukmano/quent/actions/workflows/ci.yml"><img src="https://github.com/drukmano/quent/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    &nbsp;
    <a href="https://codecov.io/gh/drukmano/quent"><img src="https://codecov.io/gh/drukmano/quent/branch/master/graph/badge.svg" alt="Coverage"></a>
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

- **One definition, two worlds** &mdash; a single pipeline works for both sync and async callers. Zero code duplication.
- **Zero ceremony** &mdash; no decorators, no base classes, no type wrappers. Just pipeline your functions.
- **Drop-in migration** &mdash; unify existing sync and async implementations into one pipeline. Stop maintaining two versions.
- **Pure Python** &mdash; zero runtime dependencies. Fully typed (PEP 561).
- **Works with asyncio, trio, and curio** &mdash; async pipelines run transparently under any of these event loops. Event loop detection uses `sys.modules` lookups (~50ns), adding zero overhead when those libraries are not loaded. Dual-protocol objects (context managers and iterables supporting both sync and async protocols) automatically prefer the async protocol under any running event loop.
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

pipeline = Q().then(validate).then(transform).then(save)

result = pipeline.run(data)          # sync if all steps are sync
result = await pipeline.run(data)    # async if any step is async
```

One definition. The pipeline starts executing synchronously. The moment any step returns an awaitable, execution seamlessly transitions to async and stays there. The caller decides whether to `await`.

---

## Installation

```bash
pip install quent
```

**Requires Python 3.10+.** Supports 3.10 through 3.14, including free-threaded builds. Zero runtime dependencies on Python 3.11+ (`typing_extensions` on 3.10).

---

## Quick Start

```python
from quent import Q

# Basic pipeline
result = Q(5).then(lambda x: x * 2).then(lambda x: x + 1).run()
print(result)  # 11

# Side effects -- do() runs the function but passes the value through
result = Q(42).then(lambda x: x * 2).do(print).then(str).run()  # prints: 84
print(result)  # '84'

# Works with any callable
result = Q('  hello  ').then(str.strip).then(str.upper).run()
print(result)  # HELLO
```

The same pipeline works whether your functions are sync, async, or a mix:

```python
pipeline = Q().then(fetch_data).then(validate).then(normalize)

# Sync context
result = pipeline.run(id)

# Async context -- same pipeline, no changes
result = await pipeline.run(id)
```

---

## Features

Build pipelines fluently. Every builder method returns `self` for chaining.

```python
from quent import Q

result = (
  Q(fetch_user, user_id)             # fetch user by id
  .then(validate)                    # transform
  .do(log)                           # side-effect
  .foreach(normalize_field)          # per-element
  .gather(enrich, score)             # concurrent
  .then(merge)                       # combine
  .if_(has_premium).then(upgrade)    # conditional
  .except_(handle_error)             # error handling
  .finally_(cleanup)                 # cleanup
  .run()                             # execute
)
```

<details>
<summary><strong>Collection Operations</strong> &mdash; foreach, foreach_do</summary>

<br>

```python
# foreach -- transform each element, collect results
Q([1, 2, 3]).foreach(lambda x: x ** 2).run()  # [1, 4, 9]

# foreach_do -- side-effect per element, keep originals
Q([1, 2, 3]).foreach_do(print).run()  # prints 1, 2, 3; returns [1, 2, 3]

# filter via list comprehension
Q([1, 2, 3, 4, 5]).then(lambda xs: [x for x in xs if x % 2 == 0]).run()  # [2, 4]
```

</details>

<details>
<summary><strong>Concurrent Execution</strong> &mdash; gather, concurrency parameter</summary>

<br>

Run multiple functions on the same value concurrently:

```python
Q('hello').gather(str.upper, len).run()  # ('HELLO', 5)
```

Limit concurrency on collection operations with the `concurrency` parameter. Uses `ThreadPoolExecutor` for sync callables and `asyncio.Semaphore` + `TaskGroup` for async:

```python
# Process up to 10 items at a time
Q(urls).foreach(fetch, concurrency=10).run()

# Limit concurrent gather branches
Q(data).gather(analyze, compress, upload, concurrency=5).run()
```

Pass a custom executor for sync concurrent operations:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
  Q(urls).foreach(fetch, concurrency=4, executor=pool).run()
```

</details>

<details>
<summary><strong>Conditionals</strong> &mdash; if_ / else_</summary>

<br>

```python
Q(5).if_(lambda x: x > 0).then(lambda x: x * 2).run()  # 10
Q(-5).if_(lambda x: x > 0).then(str).else_(abs).run()   # 5

# When predicate is omitted, uses truthiness of the current value
Q('hello').if_().then(str.upper).run()                     # 'HELLO'
Q('').if_().then(str.upper).else_(lambda _: 'empty').run() # 'empty'

# Literal predicate -- truthiness used directly
Q(value).if_(is_admin).then(grant_access).run()

# Side-effect conditional branch
Q(user).if_(is_premium).do(log_premium_access).then(next_step).run()
```

</details>

<details>
<summary><strong>Context Managers</strong> &mdash; with_ / with_do</summary>

<br>

Transparently handles both sync and async context managers:

```python
Q(open('data.txt')).with_(lambda f: f.read()).run()

# Side-effect variant (result discarded, original value passes through)
Q(open('log.txt', 'w')).with_do(lambda f: f.write('done')).run()
```

</details>

<details>
<summary><strong>Error Handling</strong> &mdash; except_ / finally_</summary>

<br>

One exception handler and one finally handler per pipeline:

```python
from quent import Q, QuentExcInfo

Q(0).then(lambda x: 1 / x).except_(lambda ei: -1).run()  # -1

Q(url)
  .then(fetch)
  .then(parse)
  .except_(handle_error, exceptions=ConnectionError)
  .finally_(cleanup)
  .run()
```

`except_` catches `Exception` by default. The handler receives a `QuentExcInfo(exc, root_value)` as its current value. Use `reraise=True` to re-raise after handling (handler runs for side-effects only). `finally_` always runs and receives the pipeline's root value.

</details>

<details>
<summary><strong>Control Flow</strong> &mdash; return_ / break_</summary>

<br>

```python
# Early return -- skips all remaining steps
Q(5) \
  .then(lambda x: Q.return_(x * 10) if x > 0 else x) \
  .then(str) \
  .run()  # 50 (str step is skipped)

# Break from iteration -- break value is appended to partial results
Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_(x) if x == 3 else x * 2).run()
# [2, 4, 3]
```

</details>

<details>
<summary><strong>Composition</strong> &mdash; clone, as_decorator</summary>

<br>

**clone** &mdash; fork-and-extend without modifying the original:

```python
base = Q().then(validate).then(normalize)
for_api = base.clone().then(to_json)    # base is untouched
for_db  = base.clone().then(to_record)  # independent copy
```

**as_decorator** &mdash; wrap a pipeline as a function decorator:

```python
@Q().then(lambda x: x.strip()).then(str.upper).as_decorator()
def get_name():
  return '  alice  '

get_name()  # 'ALICE'
```

</details>

<details>
<summary><strong>Iteration</strong> &mdash; iterate / iterate_do</summary>

<br>

Dual sync/async generators over pipeline output:

```python
for item in Q(range(5)).iterate(lambda x: x ** 2):
  print(item)  # 0, 1, 4, 9, 16

async for item in Q(async_source).iterate(transform):
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
Q(5).then(str).run()                    # str(5) -- current value passed
Q(5).then(print, 'hello').run()         # print('hello') -- explicit args used
```

---

### Enhanced Tracebacks

When an exception occurs inside a pipeline, quent injects a visualization directly into the traceback showing exactly which step failed:

```
Traceback (most recent call last):
  ...
  File "<quent>", line 1, in
    Q(fetch_data)
    .then(validate)
    .then(transform) <----
    .do(log)
  ...
ZeroDivisionError: division by zero
```

The `<----` marker points to the step that raised. Internal quent frames are cleaned from the traceback. On Python 3.11+, a concise exception note is also attached.

Opt out by setting `QUENT_NO_TRACEBACK=1` before importing quent.

---

## API Reference

### Constructor

```python
Q(v=<no value>, /, *args, **kwargs)
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
| `.except_(fn, /, *args, exceptions=None, reraise=False, **kwargs)` | Exception handler (one per pipeline) |
| `.finally_(fn, /, *args, **kwargs)` | Cleanup handler (one per pipeline) |
| `.name(label)` | Assign a label for traceback identification |
| `.set(key)` / `.set(key, value)` | Store a value in the execution context (current value unchanged) |
| `.get(key)` / `.get(key, default)` | Retrieve a value from context; replaces current value |

### Execution

| Method | Description |
|:-------|:------------|
| `.run(v=Null, /, *args, **kwargs)` | Execute the pipeline; returns value or coroutine |
| `q(...)` | Alias for `.run()` |

### Reuse and Iteration

| Method | Description |
|:-------|:------------|
| `.as_decorator()` | Wrap pipeline as a function decorator |
| `.iterate(fn=None)` | Dual sync/async generator over output |
| `.iterate_do(fn=None)` | Like iterate, fn results discarded |
| `.flat_iterate(fn=None, *, flush=None)` | Flatmap iterator; flattens one level or maps fn to sub-iterables |
| `.flat_iterate_do(fn=None, *, flush=None)` | Like flat_iterate, fn results discarded; original elements yielded |
| `.clone()` | Deep copy for fork-and-extend |
| `Q.from_steps(*steps)` | Construct a pipeline from a sequence of `.then()` steps |

### Control Flow

<sub>Class methods</sub>

| Method | Description |
|:-------|:------------|
| `Q.return_(v=Null, /, *args, **kwargs)` | Signal early return from pipeline |
| `Q.break_(v=Null, /, *args, **kwargs)` | Signal break from iteration; value is appended to partial results |

### Context API (Class-Level)

| Method | Description |
|:-------|:------------|
| `Q.set(key, value)` | Store a value in the execution context immediately (not a pipeline step) |
| `Q.get(key)` / `Q.get(key, default)` | Retrieve a value from the execution context immediately |

### Exports and Instrumentation

| Name | Description |
|:-----|:------------|
| `Q` | Main pipeline class |
| `QuentExcInfo` | NamedTuple `(exc, root_value)` passed to except handlers |
| `QuentIterator` | Type alias for `.iterate()` / `.iterate_do()` return values |
| `QuentException` | Exception type for quent-specific errors |
| `__version__` | Package version string |
| `Q.on_step` | Optional callback `(q, step_name, input_value, result, elapsed_ns)` for instrumentation |

> **Note:** `copy.copy()` and `copy.deepcopy()` are blocked on Q objects (`TypeError`). Use `.clone()` to produce a correct independent copy. Pickling is not blocked — most pipeline contents (lambdas, closures) will naturally fail to pickle, but quent does not enforce this.

---

## Examples

See the [examples/](examples/) directory for complete, runnable recipes covering ETL pipelines, API gateways, fan-out/fan-in patterns, retry with backoff, and testing pipelines.

---

## Testing

quent's correctness rests on a single guarantee: any pipeline step can be swapped between sync and async without changing the result. The test suite is purpose-built to prove this exhaustively.

### Scale

- **1,342+ test methods** across 24 test modules and 286+ test classes
- **21 CI matrix combinations** &mdash; 3 OSes (Ubuntu, macOS, Windows) &times; 5 Python versions (3.10&ndash;3.14), plus free-threaded builds (3.13t, 3.14t)
- **Security scanning** &mdash; `pip-audit` for dependency vulnerabilities, `bandit` SAST for source code

### Exhaustive Bridge Testing

The core testing infrastructure proves the sync/async bridge contract across a 7-axis combinatorial space:

1. **Operation type** &mdash; 96 "bricks" covering every pipeline operation &times; every calling convention
2. **Pipeline length** &mdash; pipelines of 1 to N steps
3. **Operation order** &mdash; every permutation of operations (with repetition)
4. **Sync/async per position** &mdash; each step independently sync or async (2<sup>N</sup> combinations per pipeline)
5. **Error injection** &mdash; exceptions, base exceptions, and control flow signals at each position
6. **Concurrency** &mdash; sequential and concurrent variants
7. **Handler configuration** &mdash; 18 error handler combinations (except/finally/both, sync/async, consuming/reraising/failing)

For each configuration, all 2<sup>N</sup> sync/async permutations run and must produce identical results. No expected values are precomputed &mdash; the invariant is that **all permutations agree with each other**. Correctness is independently verified by composing pure-Python oracle functions.

### Additional Testing Strategies

- **Transition matrix** &mdash; all 17,576 triplets of 26 atomic operations verify that every method adjacency produces correct results in all sync/async variants
- **Property-based testing** &mdash; Hypothesis generates random inputs for 179 property and fuzz tests, including CWE-117 repr sanitization with adversarial ANSI escape sequences
- **Thread safety** &mdash; 30&ndash;50 concurrent threads with barrier synchronization verify safe concurrent execution of fully constructed pipelines
- **Oracle validation** &mdash; each of the 96 bricks has an independent oracle function; oracles are verified against quent before being used in bridge assertions
- **Warning validation** &mdash; all warnings emitted during exhaustive runs are captured and validated against expected patterns

### Running Tests

```bash
# Full suite (format + lint + type check + tests)
./run_tests.sh

# Tests only (parallel -- wall-clock time = slowest module)
python scripts/run_tests_parallel.py

# Single module
python -m unittest tests.bridge_tests
```

---

## Documentation

Full documentation &mdash; including guides, advanced usage, recipes, and framework integration examples &mdash; is available at **[quent.readthedocs.io](https://quent.readthedocs.io)**.

---

## Contributing

See the [contributing guide](https://github.com/drukmano/quent/blob/master/.github/CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

```bash
git clone https://github.com/drukmano/quent.git
cd quent
uv sync --group dev       # or: pip install -e . && pip install ruff mypy
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
