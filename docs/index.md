---
title: "quent — Transparent Sync/Async Bridge for Python"
description: "Write pipeline code once — runs sync or async automatically. Pure Python, zero dependencies."
tags:
  - home
  - overview
search:
  boost: 10
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<img src="assets/logo.png" alt="quent" class="hero-logo">

# quent

**Write pipeline code once — runs sync or async automatically.**

A transparent sync/async bridge for Python. Define a pipeline once — quent detects awaitables at runtime and handles the sync/async transition automatically. No wrappers. No decorators. No ceremony.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[API Reference](reference.md){ .md-button }

</div>

---

## The Problem

When a codebase gains async support, every pipeline gets written twice.

=== "Without quent"

    ```python
    # Sync version
    def process(id):
        data = fetch_data(id)
        data = validate(data)
        return save(data)

    # Async version — identical logic, maintained separately
    async def process_async(id):
        data = await fetch_data_async(id)
        data = validate(data)
        return await save_async(data)
    ```

    Two functions. Same steps. They drift apart. A bug fix in one gets forgotten in the other. Every new step doubles the maintenance surface.

=== "With quent"

    ```python
    from quent import Q

    pipeline = (
        Q()
        .then(fetch_data)      # sync or async — doesn't matter
        .then(validate)
        .then(save)
    )

    result = pipeline.run(id)           # sync context
    result = await pipeline.run(id)     # async context
    ```

    One definition. Both worlds. quent inspects each return value at runtime — if it is awaitable, execution transitions to async and continues from there.

---

<div class="how-it-works" markdown>

## How It Works

<div class="steps" markdown>

<div class="step" markdown>
<div class="step-number">1</div>

**Build your pipeline**

Compose your functions — sync, async, or mixed. Every builder method returns `self` for fluent composition.

</div>

<div class="step" markdown>
<div class="step-number">2</div>

**Call `.run()`**

Execution starts synchronously. After each step, quent checks if the result is awaitable.

</div>

<div class="step" markdown>
<div class="step-number">3</div>

**Automatic bridging**

The moment an awaitable appears, execution seamlessly transitions to async. The caller decides whether to `await`.

</div>

</div>

</div>

---

## Quick Taste

=== "Pipeline"

    ```python
    from quent import Q

    result = (
        Q(5)
        .then(lambda x: x * 2)
        .then(str)
        .run()
    )
    # result: "10"
    ```

    `.then()` replaces the current value with each step's result. `.do()` runs a side-effect and passes the value through unchanged.

=== "Collections"

    ```python
    result = (
        Q([1, 2, 3, 4, 5])
        .then(lambda xs: [x for x in xs if x % 2 == 0])
        .foreach(lambda x: x ** 2)
        .run()
    )
    # result: [4, 16]
    ```

    `.foreach()` transforms each element. Use `.then()` with a list comprehension to filter. Both work on sync and async iterables.

=== "Error Handling"

    ```python
    result = (
        Q(url)
        .then(fetch)
        .then(parse)
        .except_(handle_error, exceptions=ConnectionError)
        .finally_(cleanup)
        .run()
    )
    # handler replaces the result on failure; cleanup always runs
    ```

    `.except_()` catches errors and optionally replaces the pipeline result. `.finally_()` runs unconditionally. Both support sync and async callables.

=== "Concurrency"

    ```python
    results = (
        Q(data)
        .gather(validate, enrich, score)
        .run()
    )
    # results: (validate_result, enrich_result, score_result)
    ```

    `.gather()` runs multiple functions on the current value. If any returns an awaitable, all are gathered concurrently via `asyncio.gather`.

---

## Installation

```bash
pip install quent
```

!!! info ""
    Requires **Python 3.10+**. Zero runtime dependencies. Fully typed ([PEP 561](https://peps.python.org/pep-0561/)).

---

## Features

<div class="grid cards" markdown>

-   :material-sync: **Transparent Sync/Async**

    ---

    Detects awaitables at runtime and transitions automatically. One pipeline definition works in both sync and async contexts.

    [Sync/Async Bridging](guide/async.md)

-   :material-transit-connection-variant: **Pipeline Building**

    ---

    Sequential steps with `.then()` and `.do()`. Full method set includes `foreach`, `foreach_do`, `gather`, `with_`, and `if_`/`else_`.

    [Pipelines & Methods](guide/chains.md)

-   :material-shield-check: **Error Handling**

    ---

    `.except_()` catches named exception types and optionally replaces the result. `.finally_()` runs cleanup unconditionally.

    [Error Handling](guide/error-handling.md)

-   :material-application-brackets: **Context Managers**

    ---

    `.with_()` enters the current value as a context manager and calls your function with the context value. Works with sync and async context managers.

    [Pipelines & Methods](guide/chains.md)

-   :material-source-branch: **Conditional Logic**

    ---

    `.if_()` applies a step only when a predicate is truthy. `.else_()` registers the fallback branch. Both predicates and branches can be sync or async.

    [Pipelines & Methods](guide/chains.md)

-   :material-debug-step-over: **Control Flow**

    ---

    `Q.return_()` exits early with an optional value. `Q.break_()` stops iteration inside `.foreach()` or `.foreach_do()`.

    [Pipelines & Methods](guide/chains.md)

-   :material-text-search: **Enhanced Tracebacks**

    ---

    When an exception occurs, quent injects a pipeline visualization into the traceback with a `<----` marker on the failing step.

    [Getting Started](getting-started.md)

-   :material-repeat: **Iteration**

    ---

    `.iterate()` wraps pipeline output as a lazy generator. The same object supports both `for` and `async for` loops.

    [Pipelines & Methods](guide/chains.md)

-   :material-package-variant-closed-check: **Zero Dependencies**

    ---

    Pure Python. No runtime dependencies. PEP 561 typed with inline annotations. Tested on Python 3.10 through 3.14.

    [API Reference](reference.md)

</div>

---

## Next Steps

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install quent and build your first pipeline in under five minutes.

    [Getting Started](getting-started.md){ .md-button }

-   **Guide**

    ---

    Learn chains, async handling, error handling, and reuse patterns in depth.

    [Read the Guide](guide/chains.md){ .md-button }

-   **API Reference**

    ---

    Complete reference for `Q`, `Null`, `QuentException`, and every pipeline method.

    [API Reference](reference.md){ .md-button }

</div>
