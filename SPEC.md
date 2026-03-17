# quent — Behavioral Specification

**Version:** 5.2.0 | **Date:** 2026-03-17

> This document defines quent's behavioral contracts. It describes what quent does in terms
> of observable behavior — not how it is implemented. The source code is the implementation;
> this spec is the contract. When they disagree, investigate.
>
> Internal details are documented only where they affect the behavioral contract (e.g., the Null sentinel's
> effect on calling conventions). Every statement is phrased in terms of observable behavior whenever possible.

## Table of Contents

- [1. Identity](#1-identity)
- [2. The Bridge Contract](#2-the-bridge-contract)
- [3. The Pipeline Model](#3-the-pipeline-model)
- [4. Calling Conventions](#4-calling-conventions)
- [5. Operations](#5-operations)
- [6. Error Handling](#6-error-handling)
- [7. Control Flow](#7-control-flow)
- [8. Execution](#8-execution)
- [9. Iteration](#9-iteration)
  - [9.5 flat_iterate()](#95-flat_itefn--none---on_exhaust--none)
  - [9.6 flat_iterate_do()](#96-flat_iterate_dofn--none---on_exhaust--none)
  - [9.7 Deferred with_ in Iteration](#97-deferred-with_-in-iteration)
- [10. Reuse](#10-reuse)
- [11. Concurrency](#11-concurrency)
- [12. Null Sentinel](#12-null-sentinel)
- [13. Traceback Enhancement](#13-traceback-enhancement)
- [14. Instrumentation (`on_step`)](#14-instrumentation-on_step)
- [15. Context API](#15-context-api)
- [16. Design Decisions & Rationale](#16-design-decisions--rationale)
- [17. Known Asymmetries](#17-known-asymmetries)
- [18. Patterns](#18-patterns)
- [19. Public API](#19-public-api)

---

## 1. Identity

**quent** is a fluent pipeline builder that transparently bridges synchronous and asynchronous Python execution. Pure Python, zero dependencies.

### The Fundamental Promise

Write pipeline code once — it runs sync or async automatically. Any sync callable at any position in a pipeline can be swapped for its async equivalent (or vice versa), and the pipeline produces the same result. The user never selects a mode, never wraps coroutines, never writes conditional `await` logic. The bridge is invisible.

### Target Audience

- Developers building libraries or frameworks that must support both sync and async callers without duplicating pipeline logic.
- Application developers composing multi-step data processing, I/O, or request-handling pipelines where some steps may be sync and others async.
- Anyone who wants to express "do A, then B, then C" as a single reusable object, regardless of whether A/B/C are sync functions, async functions, or a mix.

### Use Cases

- HTTP request pipelines: build once, run from sync Flask or async FastAPI.
- Data processing chains: validate, transform, persist — some steps may hit async databases.
- Resource management: open/use/close patterns with context managers that may be sync or async.
- Decorator factories: wrap functions in reusable processing chains.

---

## 2. The Bridge Contract

### Core Invariant: Sync/Async Transparency

The bridge contract is the central guarantee of quent:

> **Given a pipeline of N steps, replacing any step's callable with a functionally equivalent callable of the opposite sync/async kind produces the same observable result.**

"Functionally equivalent" means: given the same input, the sync callable returns value `V` and the async callable returns a coroutine that resolves to the same value `V`.

This guarantee holds for all pipeline operations — steps, side-effects, iteration, gathering, context managers, conditionals, error handlers, and cleanup handlers.

### How Execution Works: The Two-Tier Model

Execution always begins synchronously. The engine walks the pipeline's linked list of steps, evaluating each one in order. After each step, the engine inspects the result:

- **If the result is not awaitable:** The engine records it as the new current value and advances to the next step. This is the fast path — pure sync execution with no async overhead.

- **If the result is awaitable:** The engine immediately transitions to an async continuation. The async continuation receives the pending awaitable and all accumulated state (current value, root value, position in the linked list). It awaits the result, then continues walking the remaining steps in async mode, awaiting any further awaitables inline.

This means:

1. A pipeline where every step is synchronous executes entirely synchronously. No event loop is created, no coroutines are allocated, no async machinery is touched.

2. A pipeline where any step returns an awaitable transparently transitions to async at that point. The caller receives a coroutine from `.run()` and must `await` it.

3. The transition point can be anywhere — first step, middle step, last step. Steps before the transition run synchronously; steps from the transition onward run in async mode.

4. Once the engine transitions to async, it stays async for the remainder of that execution. There is no "transition back to sync."

### What "Transparent" Means

Transparency means the user performs **zero ceremony** to handle the sync/async boundary:

- No explicit mode selection (no `async_mode=True`, no separate `AsyncChain` class).
- No manual wrapping of coroutines or futures.
- No conditional `if isawaitable(result): result = await result` in user code.
- No separate API surface for sync vs. async usage.

The same `Chain` object, built with the same API, handles both. The only observable difference is whether `.run()` returns a plain value (all-sync pipeline) or a coroutine (pipeline that encountered an awaitable). The caller decides whether to `await` based on their context.

### The Bridge Guarantee

The bridge guarantee can be stated precisely:

> For any pipeline `P` and any step `i` in `P`:
> - Let `P_sync` be `P` with step `i` using sync callable `f` where `f(x) == v`.
> - Let `P_async` be `P` with step `i` using async callable `g` where `await g(x) == v`.
> - Then: the final result of `P_sync` equals the final result of `P_async`.

This holds because the engine's step evaluation and value threading are identical in both tiers. The only difference is whether an intermediate result is used directly or awaited first — and the engine handles that distinction internally.

### Rationale

**Why start synchronous?** Most Python code is synchronous. Starting sync means that purely synchronous pipelines pay zero async overhead — no event loop creation, no coroutine allocation, no `__await__` protocol. This is the common case and it should be fast.

**Why not two separate classes?** A single `Chain` class that handles both modes eliminates API duplication, prevents the "colored function" problem from infecting library interfaces, and allows a single pipeline definition to be reused across sync and async contexts.

**Why transition on first awaitable rather than inspecting callables upfront?** Inspecting callables at build time is unreliable (a regular function might return a coroutine; a nested chain's sync/async nature depends on its own steps). Runtime detection at evaluation time is the only correct approach.

---

## 3. The Pipeline Model

### Chains as Sequential Pipelines

A chain is a sequential pipeline with a **current value** threaded through it. Each step in the pipeline receives the current value, does something with it, and (unless it is a side-effect step) its result becomes the new current value for the next step.

The pipeline is modeled internally as a singly-linked list. Steps are appended to the tail in O(1) time. Execution walks the list from head to tail, evaluating each step in order.

### Root Value

#### Constructor Signature

`Chain(v=<no value>, /, *args, **kwargs)`

- **`v` is callable:** When the chain runs, `v` is called with `(*args, **kwargs)`. If no `args`/`kwargs` are provided, `v` is called with no arguments. The return value becomes the root value.
- **`v` is not callable:** `v` is used as-is as the root value. `args`/`kwargs` must not be provided (enforced at build time; raises `TypeError`).
- **`Chain()`** — creates a chain with no root value. The first step evaluates with no current value (standard calling conventions apply — a callable is called with no arguments).
- **`Chain(None)`** — creates a chain with root value `None`.
- **Kwargs require a root value:** `args` and `kwargs` are only valid when a positional root value `v` is present. `Chain(key=val)` with no positional root raises `TypeError` at build time. This is because `args`/`kwargs` are arguments *for* the root callable — without a root callable, they have no target.

#### Providing the Root Value

The root value seeds the pipeline. There are two ways to provide it:

- **At build time:** `Chain(v)` sets a root value that is evaluated when the chain runs. If `v` is callable, it is called (with optional args/kwargs); if not, it is used as-is.

- **At run time:** `chain.run(v)` injects a value that replaces the build-time root. When both exist, the run-time value wins and the build-time root is ignored entirely. `Chain(A).then(B).run(C)` is equivalent to `Chain(C).then(B).run()`.

The root value, once evaluated, also becomes the **current value for the `finally_()` handler**: the `finally_()` handler receives the root value (not the current pipeline value at the point of completion or failure) as its current value. This is by design — the root value represents "what this chain was invoked with," which is the most useful context for cleanup. The `except_()` handler receives `ChainExcInfo(exc, root_value)` as its current value.

### Steps: `.then()`

Each `.then(v)` step appends to the pipeline. When the step is evaluated:

- If `v` is callable, it is invoked per the calling conventions (Section 4). Its return value becomes the new current value.
- If `v` is not callable, it replaces the current value directly. `Chain(1).then(2).then(3).run()` produces `3`.

Steps are the primary pipeline-building primitive. All other operations (`.foreach()`, `.gather()`, `.if_()`, `.with_()`) are conceptually specialized steps that follow the same linked-list execution model.

### Side-Effect Steps: `.do()`

A `.do(fn)` step receives the current value but its return value is **discarded**. The current value passes through unchanged. This is for operations that should observe or act on the value without transforming it.

```python
Chain(5).do(print).then(lambda x: x * 2).run()
# prints: 5
# returns: 10 (print's None return is discarded; 5 passes through to the lambda)
```

`.do()` requires its argument to be callable. This is enforced at build time to prevent silent no-ops — a non-callable value used as a side-effect would do nothing, which is always a bug.

**Rationale for `.do()` requiring callability:** A non-callable `.do(42)` would evaluate `42` as a literal, discard it, and pass the current value through — indistinguishable from not having the step at all. Requiring callability catches this mistake at build time rather than silently doing nothing at run time.

### Execution Model: Append-Only Linked List

The pipeline is an append-only singly-linked list:

- **Building** appends nodes to the tail. The chain maintains a tail pointer for O(1) insertion. Building is not thread-safe — chains must be fully constructed before being shared across threads.

- **Execution** walks the list from head to tail, never mutating the list structure. A fully constructed chain is safe to execute concurrently from multiple threads (including under free-threaded Python / PEP 703), because execution uses only function-local state.

- **The linked list is never modified after construction.** This separation between build-time mutation and run-time immutability is the foundation of the thread-safety model.

### Value Flow Summary

```
Chain(root)          root is evaluated → current_value = result
  .then(f)           f(current_value) → current_value = result
  .do(g)             g(current_value) → result discarded, current_value unchanged
  .then(h)           h(current_value) → current_value = result
  .run()             returns current_value (or None if no value was ever produced)
```

When the pipeline completes with no value ever having been produced (e.g., `Chain().do(print).run()`), the result is `None`. The internal "no value" sentinel is never exposed to users.

---

## 4. Calling Conventions

The calling conventions define **exactly how a step's callable is invoked**. They are the most important contract in quent — every pipeline step, every operation, and every handler invocation goes through these rules.

There are **2 rules**, applied in strict priority order. The first matching rule wins. These same 2 rules apply universally to ALL contexts: standard steps (`then`, `do`, `with_`, etc.), `except_()` handlers, `finally_()` handlers, and `if_()` predicates. The only difference per context is what "current value" means.

### Rule 1: Explicit Args/Kwargs

**Trigger:** Positional arguments or keyword arguments were provided at registration time (e.g., `.then(fn, arg1, key=val)`).

**Behavior:** The callable is invoked with **only the explicit arguments**. The current pipeline value is **not passed**.

```python
Chain(5).then(format_number, 'USD', decimals=2).run()
# calls: format_number('USD', decimals=2)  — the 5 is NOT passed
```

**Constraints:**

- The step must be callable. Providing arguments to a non-callable raises `TypeError` at build time.

**Rationale:** When the user provides explicit arguments, they are declaring "call this function with exactly these arguments." Silently prepending the current value would be surprising and would require the user to account for an extra first parameter in every function signature.

**Design note:** This means there is no built-in way to pass both the current value AND explicit arguments in a single `.then()` call. This is intentional — that scenario is handled by using a lambda/closure that captures the current value.

### Rule 2: Default Passthrough

**Trigger:** None of the above rules matched.

**Behavior depends on callability:**

- **Callable, current value exists:** `fn(current_value)` — the callable receives the current value as its sole argument.
- **Callable, no current value:** `fn()` — the callable is called with no arguments.
- **Not callable:** The value itself is returned as-is, becoming the new current value.

```python
Chain(5).then(str).run()          # str(5) → '5'
Chain().then(get_timestamp).run() # get_timestamp() — no current value
Chain(5).then(42).run()           # 42 — non-callable replaces current value
```

**Rationale:** The default is the most intuitive behavior: a function receives the thing flowing through the pipeline. Non-callables act as constant injections — useful for resetting or replacing the pipeline value.

### Summary Table

| Priority | Rule | Trigger | Invocation |
|---|---|---|---|
| 1 | Explicit args | Args/kwargs provided | `fn(*args, **kwargs)` |
| 2 | Default | None of the above | `fn(cv)`, `fn()`, or `v` as-is |

### Nested Chains

A `Chain` is callable — it supports `__call__`, which is an alias for `run()`. When a `Chain` instance is used as a step value (e.g., `.then(inner_chain)`), it follows the same 2-rule calling convention as any other callable:

- **Rule 1 (Explicit Args):** `.then(inner_chain, arg1, key=val)` — the chain is called with explicit args, and the current pipeline value is NOT passed.
- **Rule 2 (Default):** `.then(inner_chain)` with no explicit args — the chain is called with the current pipeline value as its argument (i.e., `inner_chain(current_value)`), which maps to `inner_chain.run(current_value)`.

The additional behaviors that apply when a Chain is used as a step in another chain are:

**Signal propagation:** Control flow signals (`return_()`, `break_()`) propagate from the nested chain to the outer chain — they are not trapped at the nested chain boundary. When a chain is executed directly via `.run()`, escaped control flow signals are caught and wrapped in `QuentException`.

**Composition:** Nested chains enable composition. By running the inner chain with the outer chain's current value, pipelines can be decomposed into reusable sub-pipelines. Signal propagation across nesting boundaries ensures that `Chain.return_()` in a nested chain exits the outermost chain, matching the intuition of "return from this pipeline."

**Invocation forms:**

| Registration | Invocation |
|---|---|
| `.then(inner_chain)` | `inner_chain` runs with `current_value` as its input |
| `.then(inner_chain, arg1, arg2)` | `inner_chain` runs with `arg1` as input and `(arg2,)` as extra args — `current_value` is NOT passed |
| `.then(inner_chain, key=val)` | `inner_chain` runs with kwargs only, no run value — `current_value` is NOT passed |
| `.then(inner_chain, arg1, key=val)` | `inner_chain` runs with `arg1` as input, kwargs forwarded — `current_value` is NOT passed |

**Args/kwargs replacement:** When a nested chain is invoked with explicit args or kwargs (Rule 1), the caller's args/kwargs replace the inner chain's build-time root args/kwargs entirely — there is no merging. The inner chain's root callable is preserved, but its build-time arguments are overridden. This is consistent with the run-time replacement semantics: `Chain(A, key=1).run(B)` replaces the entire root; similarly, `.then(inner_chain, key=val)` replaces the inner chain's root arguments.

### The Except Handler Calling Convention

The `except_()` handler follows the **standard 2-rule calling convention**. The "current value" is `ChainExcInfo(exc, root_value)` — a NamedTuple containing the caught exception and the chain's evaluated root value.

| Priority | Rule | Invocation |
|---|---|---|
| 1 | Explicit args | `handler(*args, **kwargs)` — `ChainExcInfo` is NOT passed |
| 2 | Default | `handler(ChainExcInfo(exc, root_value))` |

**Non-callable constraint:** `except_()` enforces callability at registration — a non-callable value raises `TypeError`. This is stricter than standard steps (which allow non-callable literals) because a non-callable error handler is always a mistake.

### The Finally Handler Calling Convention

The `finally_()` handler follows the **standard 2-rule calling convention**. The "current value" is the chain's root value (normalized to `None` if absent).

```python
Chain(resource).then(process).finally_(cleanup).run()
# On completion (success or failure): cleanup(resource)
```

The finally handler's return value is always discarded. If it raises an exception, that exception propagates (replacing any active exception, with the original preserved in `__context__`).

**Rationale for standard conventions:** The finally handler is conceptually "do this cleanup with the root value" — the same pattern as any pipeline step. The standard rules apply naturally: explicit args suppress the root value, consistent with every other context.

---

## 5. Operations

**Build-time enforcement principle:** quent enforces constraints at build time whenever possible. Only constraints that cannot be evaluated without runtime state (e.g., the current pipeline value) are deferred to evaluation time. Build-time enforcement catches errors early, before the pipeline is ever executed.

Every operation is a builder method on the Chain. Each method returns `self` (for fluent chaining) and appends one step to the pipeline's internal structure. Operations execute during `run()` — the builder methods only record intent.

All operations participate in the transparent sync/async bridge: if any operation returns an awaitable, the pipeline seamlessly transitions to async execution.

### 5.1 `then(v, /, *args, **kwargs)`

**Contract:** Append a pipeline step whose result replaces the current value.

**Arguments:**
- `v` — callable, literal value, or nested Chain.
- `*args, **kwargs` — forwarded to `v` when it is invoked.

**Behavior:**
- When `v` is callable with no explicit args and a current value exists: `v(current_value)`.
- When `v` is callable with explicit args: `v(*args, **kwargs)`. The current value is NOT implicitly passed.
- When `v` is not callable: `v` itself becomes the new current value (args/kwargs must not be provided; enforced at build time).
- When `v` is a nested Chain: the nested chain is executed with the current value as its run value. Control flow signals propagate through to the outer chain.
- The result of evaluating `v` replaces the current pipeline value for subsequent steps.

**Calling convention:** Follows the standard calling convention (see Section 4).

**Error behavior:** Any exception raised during evaluation propagates through the pipeline's error handling (except/finally handlers if registered).

### 5.2 `do(fn, /, *args, **kwargs)`

**Contract:** Append a side-effect step. `fn` is called, but its result is discarded — the current pipeline value passes through unchanged.

**Arguments:**
- `fn` — must be callable. This is enforced at build time.
- `*args, **kwargs` — forwarded to `fn` when invoked.

**Behavior:**
- `fn` is invoked following the standard calling convention.
- The return value of `fn` is discarded. The pipeline value before this step passes through to the next step.
- If `fn` returns an awaitable, it is still awaited (to complete the side-effect), but its resolved value is discarded.

**Rationale for requiring callable:** A literal value used as a side-effect would silently do nothing — this is almost certainly a bug. Enforcing callability catches this at build time rather than silently succeeding.

**Error behavior:**
- `TypeError` is raised at build time if `fn` is not callable.
- Runtime exceptions from `fn` propagate normally through the pipeline's error handling.

### 5.3 `foreach(fn, /, *, concurrency=None, executor=None)`

**Contract:** Apply `fn` to each element of the current iterable value, collecting the results into a list that replaces the current value.

**Arguments:**
- `fn` — callable applied to each element. Must be callable (enforced at build time).
- `concurrency` — controls parallelism. When `None` (default), elements are processed sequentially. When `-1`, all elements run concurrently with no limit (unbounded; effective concurrency equals `len(items)` at runtime). When a positive integer, elements are processed concurrently up to that limit.
- `executor` — optional `concurrent.futures.Executor` instance. When provided, used for sync concurrent execution instead of creating a new `ThreadPoolExecutor`. Quent does NOT shut it down — lifecycle management is the caller's responsibility. Only has effect when `concurrency` is also set; ignored on the sequential path. Non-`Executor` values raise `TypeError`.

**Behavior:**
- The current pipeline value must be iterable.
- Each element is passed to `fn(element)`, and the return value is collected.
- The result is a `list` of all return values, in the same order as the input elements.
- When `fn` is `None` or not provided — this is not supported; `fn` is required and must be callable.

**Sequential execution (concurrency=None):**
- Elements are processed one at a time in iteration order.
- If `fn` returns an awaitable for any element, the pipeline transitions to async and awaits it. Subsequent elements continue in async mode.
- Supports both sync iterables (`__iter__`) and async iterables (`__aiter__`). When both protocols are present, the async protocol is preferred if an async event loop is running (asyncio, trio, or curio); otherwise, the sync protocol is used.

**Concurrent execution (concurrency=-1 or concurrency=N):**
- The entire input iterable is eagerly materialized into a list before processing begins. Do not use with infinite or very large iterables.
- The dual-protocol preference applies when materializing the iterable: when both `__iter__` and `__aiter__` are present and an async event loop is running (asyncio, trio, or curio), the async protocol is used; otherwise, the sync protocol is used.
- Sync path: uses a thread pool. The first element is probed to determine sync vs async. If the first call returns a non-awaitable, all remaining elements are dispatched to worker threads.
- Async path: uses semaphore-limited async tasks. If the first call returns an awaitable, all elements are dispatched as async tasks with concurrency bounded by the semaphore.
- Mixed sync/async is not supported within a single concurrent operation — if the first element determines sync execution but a later element's callable returns an awaitable, a `TypeError` is raised.
- Results preserve input order regardless of completion order.
- When `concurrency=-1` (unbounded), the effective concurrency is resolved to `len(items)` at runtime.

**Error behavior (sequential):**
- Exceptions propagate immediately, stopping iteration at the failing element.
- `StopIteration` raised by a user callback propagates out — quent does not intercept or reinterpret it. It is treated as a regular exception that propagates through error handling; it does not silently terminate the iteration. Note: in async callbacks, Python's PEP 479 behavior applies — `StopIteration` raised inside a coroutine is automatically wrapped as `RuntimeError` by the Python runtime.

**Error behavior (concurrent):**
- When a single concurrent worker fails, that exception propagates directly (not wrapped in an `ExceptionGroup`).
- When multiple concurrent workers fail, all regular exceptions (`Exception` subclasses) are wrapped in an `ExceptionGroup`. `BaseException` subclasses (e.g., `KeyboardInterrupt`, `SystemExit`) are not wrapped — the one from the earliest input index takes priority over regular exceptions.
- Control flow signals take priority: a `return_()` signal takes absolute priority over all other exceptions. A `break_()` signal takes priority over regular exceptions but not over `return_()`.

**`break_()` behavior:**
- In sequential mode: `break_()` stops iteration immediately. The results collected so far are returned. If `break_()` carries a value, that value is appended to the partial results.
- In concurrent mode: `break_()` causes results to be truncated to elements before the break index. If `break_()` carries a value, that value is appended to the truncated results.

### 5.4 `foreach_do(fn, /, *, concurrency=None, executor=None)`

**Contract:** Apply `fn` to each element of the current iterable for side-effects. The original elements (not `fn`'s return values) are collected into a list.

**Arguments:** Same as `foreach()`, including `concurrency` and `executor`. The `concurrency` parameter accepts `None` (sequential), `-1` (unbounded concurrent), or a positive integer (bounded concurrent). The `executor` parameter behaves identically to `foreach()` — only has effect when `concurrency` is also set.

**Behavior:**
- Identical to `foreach()` in execution mechanics, but the collection strategy differs: `fn`'s return values are discarded, and the original input elements are collected into the result list.
- The result is a `list` of the original elements, in input order.

**Concurrent, error, and break behavior:** Same as `foreach()`.

### 5.5 `gather(*fns, concurrency=-1, executor=None)`

**Contract:** Run multiple functions on the current pipeline value concurrently. Results are returned as a tuple in the same positional order as `fns`.

**Arguments:**
- `*fns` — one or more callables. Each must be callable (enforced at build time). Each receives the current pipeline value as its argument.
- `concurrency` — controls parallelism. `-1` (default) means unbounded — all functions run concurrently with no limit (effective concurrency equals `len(fns)` at runtime). A positive integer limits the number of simultaneous executions. Unlike `foreach()`/`foreach_do()`, `gather()` does not accept `concurrency=None` — gather is always concurrent.
- `executor` — optional `concurrent.futures.Executor` instance. When provided, used for sync concurrent execution instead of creating a new `ThreadPoolExecutor`. Quent does NOT shut it down — lifecycle management is the caller's responsibility. Because `gather()` is always concurrent, `executor` always applies to the sync path. Non-`Executor` values raise `TypeError`.

**Behavior:**
- Gather is ALWAYS concurrent — there is no sequential fallback.
- Sync path: uses a thread pool. The first function is probed — if it returns a non-awaitable, remaining functions are dispatched to worker threads.
- Async path: uses semaphore-limited async tasks. If the first function returns an awaitable, all functions are dispatched as async tasks.
- Mixed sync/async is not supported — if the first function determines sync execution but a later function returns an awaitable, a `TypeError` is raised.
- The result is a `tuple` with one element per function, in the same order as the `fns` arguments.
- At least one function must be provided. Passing zero functions raises `QuentException`.
- When one function is provided: still returns a single-element tuple `(result,)`.

**Rationale for always-concurrent gather:** Gather's purpose is to fan out computation. A sequential gather would be semantically identical to chaining multiple `then()` calls, offering no benefit. By always executing concurrently, the sync and async paths have symmetric behavior, and users get the expected parallelism without ceremony.

**Error behavior:**
- When a single function fails: that exception propagates directly (not wrapped in an `ExceptionGroup`).
- When multiple functions fail: all regular exceptions (`Exception` subclasses) are wrapped in an `ExceptionGroup`. The message indicates the number of exceptions, e.g., `"gather() encountered 3 exceptions"`.
- `BaseException` subclasses (e.g., `KeyboardInterrupt`, `SystemExit`) are not wrapped — the one from the earliest position in `fns` takes priority over regular exceptions.
- `Chain.return_()` signals take absolute priority and propagate immediately. If a return signal is encountered alongside regular exceptions, the regular exceptions are discarded (with a WARNING-level log message). If a `break_()` signal is raised during `gather()`, it is caught and wrapped in a `QuentException` with a message indicating that break signals are not allowed in gather operations. Break signals are scoped to iteration operations (`foreach`, `foreach_do`).

### 5.6 `with_(fn, /, *args, **kwargs)`

**Contract:** Enter the current pipeline value as a context manager, invoke `fn` following the standard calling convention with the context value (the result of `__enter__` or `__aenter__`) as the current value, and replace the current pipeline value with `fn`'s return value. The context manager is properly exited regardless of whether `fn` succeeds or fails.

**Arguments:**
- `fn` — callable to invoke in the context. Receives the context value per the standard calling convention. Must be callable (enforced at build time).
- `*args, **kwargs` — forwarded to `fn` following the standard calling convention.

**Behavior:**
- The current pipeline value must support the context manager protocol (`__enter__`/`__exit__` or `__aenter__`/`__aexit__`). If it supports neither, a `TypeError` is raised.
- For sync context managers: `__enter__()` is called, `fn` is invoked with the resulting context value, and `__exit__()` is called on the success or failure path.
- For async context managers: `__aenter__()` is awaited, `fn` is invoked, and `__aexit__()` is awaited.
- For dual-protocol context managers (supporting both sync and async): when an async event loop is running (asyncio, trio, or curio), the async protocol is preferred. Otherwise, the sync protocol is used.
- `fn`'s return value replaces the current pipeline value.
- If `fn` returns an awaitable, it is awaited. The pipeline transitions to async execution if it hasn't already.

**Exception suppression:** If `fn` raises and `__exit__` returns a truthy value (suppressing the exception), the pipeline continues. The current value becomes `None` in this case (since the body's result is unavailable). If `__exit__` returns falsy, the exception propagates.

**Exit handler failure:** If `__exit__` itself raises an exception, that exception propagates and **replaces** the original body exception, matching Python's native `with` statement behavior. The original body exception is preserved as `__context__` on the new exception.

**Control flow signals:** If `fn` raises a control flow signal (`return_()` or `break_()`), `__exit__` is called with no exception info (clean exit), and the signal propagates to the outer pipeline.

**Bare form — `with_()`:** When `fn` is omitted, the context value (the result of `__enter__`/`__aenter__`) replaces the pipeline value directly. This form is only valid before an iteration terminal (`iterate()`, `iterate_do()`, `flat_iterate()`, `flat_iterate_do()`) — the context manager entry is deferred to iteration time (see §9.7). A bare `with_()` not followed by an iterate variant raises `TypeError` at run time. Positional or keyword arguments without `fn` raise `TypeError` at build time.

### 5.7 `with_do(fn, /, *args, **kwargs)`

**Contract:** Like `with_()`, but `fn`'s result is discarded. The original pipeline value (the context manager object itself, not the `__enter__` result) passes through unchanged.

**Arguments:** Same as `with_()`.

**Behavior:**
- Identical to `with_()` in context manager handling.
- `fn`'s return value is discarded. The pipeline value before this step (the context manager object) passes through to the next step.
- If an exception is suppressed by `__exit__`, the original pipeline value passes through (not `None`).

### 5.8 `if_(predicate=None, /, *args, **kwargs)`

**Contract:** Set a pending conditional flag on the chain. The next `.then()` or `.do()` call after `if_()` becomes the truthy branch. Returns `self` for fluent chaining.

**Arguments:**
- `predicate` — positional-only. Controls what is tested for truthiness:
  - `None` (default): the truthiness of the current pipeline value itself is used.
  - callable: invoked following the standard 2-rule calling convention (with the current pipeline value as the "current value"). The return value is tested for truthiness.
  - nested `Chain`: the chain is executed with the current pipeline value as input, following the standard default rule (Rule 2). The result is tested for truthiness. `return_()` inside the predicate chain propagates to the outer chain (early exit is valid from a predicate). `break_()` inside a predicate chain raises `QuentException` with the message `'break_() cannot be used inside an if_() predicate.'` — predicates are not iteration contexts, so `break_()` is nonsensical there.
  - any other literal value: its truthiness is used directly — the current pipeline value is not examined.
- `*args, **kwargs` — forwarded to the predicate when it is invoked, following the standard calling convention (Rule 1: explicit args suppress the current value).

**Build-time constraints:**
- Calling `if_()` while another `if_()` is already pending (its truthy branch has not yet been registered via `.then()`/`.do()`) raises `QuentException`.
- Calling `.run()` while an `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.
- Calling `.else_()` while `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.
- Calling `.decorator()` while an `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.
- Calling `.iterate()` while an `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.
- Calling `.iterate_do()` while an `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.
- Calling `.set()` while an `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.
- Calling `.get()` while an `if_()` is still pending (no `.then()`/`.do()` followed it) raises `QuentException`.

**Behavior:**
- The next `.then(v, *args, **kwargs)` or `.do(fn, *args, **kwargs)` call after `if_()` is absorbed as the truthy branch rather than appended as a normal step.
- At run time, the predicate is evaluated first following the standard 2-rule calling convention. If it returns an awaitable, it is awaited.
- If the predicate result is truthy, the absorbed truthy branch is evaluated following the standard calling convention. For `.then()` the result replaces the current pipeline value; for `.do()` the result is discarded and the current value passes through.
- If the predicate result is falsy and no `else_()` is registered, the current value passes through unchanged.
- If the predicate result is falsy and an `else_()` is registered, the else branch is evaluated instead.

When `predicate` is `None` and the chain has no current value (internal Null sentinel), the predicate evaluates to falsy. The Null sentinel is always falsy for predicate purposes.

**Usage examples:**

```python
# Test current value truthiness, transform if truthy
Chain(data).if_().then(process)

# Test with a predicate callable
Chain(data).if_(is_valid).then(process)

# Predicate with explicit args — current value NOT passed
Chain(data).if_(check_flag, 'feature_x').then(process)

# Literal predicate — use its own truthiness directly
Chain(data).if_(True).then(always_run)

# With an else branch
Chain(data).if_(is_valid).then(process).else_(handle_invalid)

# Truthy branch as a side-effect
Chain(data).if_(needs_logging).do(log_item)
```

### 5.9 `else_(v, /, *args, **kwargs)`

**Contract:** Register an alternative branch for the immediately preceding `if_()` step. Evaluated when `if_()`'s predicate was falsy.

**Arguments:**
- `v` — callable or literal value for the else branch.
- `*args, **kwargs` — forwarded to `v` when it is invoked. If args/kwargs are provided, `v` must be callable.

**Constraints:**
- Must follow immediately after the truthy branch (`.then()` or `.do()`) that was registered for the preceding `if_()`. No other operations may appear in between. If the chain has no steps, or the last completed `if_()` step does not have a pending else slot, a `QuentException` is raised.
- Calling `else_()` while `if_()` is still pending (no `.then()`/`.do()` has been registered yet) raises `QuentException`.
- Only one `else_()` per `if_()`. A second `else_()` on the same `if_()` raises `QuentException`.

**Behavior:**
- When the preceding `if_()`'s predicate is falsy, `v` is evaluated following the standard calling convention. The result replaces the current pipeline value.
- `v` can be a callable, a literal value, or a nested Chain, following the same rules as `then()`.

#### 5.9.1 `else_do(fn, /, *args, **kwargs)`

**Contract:** Register a side-effect alternative branch for the immediately preceding `if_()` step. Like `else_()` but `fn`'s return value is discarded — the current pipeline value passes through unchanged when the else branch is taken.

**Arguments:**
- `fn` — callable to invoke for its side-effects. Must be callable (non-callable raises `TypeError`).
- `*args, **kwargs` — forwarded to `fn` when it is invoked.

**Constraints:**
- Same positioning constraints as `else_()`: must follow immediately after the truthy branch (`.then()` or `.do()`), with no other operations in between.
- Calling `else_do()` while `if_()` is still pending raises `QuentException`.
- Calling `else_do()` on an empty chain or after a non-`if_()` step raises `QuentException`.
- Only one else branch per `if_()` — `else_do()` and `else_()` count the same; a second call raises `QuentException`.

**Behavior:**
- When the preceding `if_()`'s predicate is falsy, `fn` is called for its side-effects. Its return value is always discarded. The current pipeline value passes through unchanged.
- If `fn` returns an awaitable (async function), the awaitable is awaited and then discarded — the original current value is still returned.

**Example:**
```python
Chain(-5).if_(lambda x: x > 0).then(str).else_do(print).run()
# prints: -5
# returns: -5  (print's return value discarded, original value passes through)
```

### 5.10 `name(label)`

**Contract:** Assign a user-provided label for traceback identification. Returns `self` for fluent chaining.

**Parameter:** `label` — a `str`. Positional-only.

**Behavior:**
- When a name is set, chain visualizations render as `Chain[label](root)` instead of `Chain(root)`.
- The name appears in:
  - Traceback `<quent>` frames (chain visualization).
  - Exception notes (Python 3.11+): `quent: exception at .then(validate) in Chain[auth_pipeline](fetch_data)`.
  - `repr(chain)`: `Chain[auth_pipeline](fetch_data).then(validate)...`.
- The name has no effect on execution semantics — purely for debuggability.
- `clone()` copies the name. `decorator()` preserves the name on the cloned chain.
- Zero runtime cost beyond storing the string.

**Example:**
```python
Chain(fetch).name('auth_pipeline').then(validate).run()
```

---

## 6. Error Handling

### 6.1 Design Rationale

Chains support exactly one exception handler and one cleanup handler each. This constraint keeps the execution model simple and predictable: the user always knows which handler runs, in what order, and what happens to the exception. For per-step error handling, compose nested chains — each nested chain gets its own except/finally pair.

Both the except handler and the finally handler follow the standard 2-rule calling convention. The only difference is what "current value" means in each context: for `except_()`, the current value is `ChainExcInfo(exc, root_value)`; for `finally_()`, it is the root value (normalized to `None` if absent).

### 6.2 Exception Handler — `except_(fn, /, *args, exceptions=None, reraise=False, **kwargs)`

Registers an exception handler for the chain. At most one `except_()` per chain; a second call raises `QuentException`.

#### 6.2.1 Handler Registration

- `fn` must be callable; `TypeError` is raised otherwise.
- `exceptions` specifies which exception types to catch. Accepts a single exception type, an iterable of types, or `None` (default: `Exception`).
  - An empty iterable raises `QuentException`.
  - Non-`BaseException` subclasses raise `TypeError`.
  - String values raise `TypeError` (common mistake: passing `"ValueError"` instead of `ValueError`).
- If any specified exception type is a `BaseException` subtype that is not an `Exception` subtype (e.g., `KeyboardInterrupt`, `SystemExit`), a `RuntimeWarning` is emitted advising the user to consider `Exception` instead — catching system signals can suppress critical shutdown behavior.
- `reraise` controls whether the original exception is re-raised after the handler runs (see below).

#### 6.2.2 Except Handler Calling Convention

The except handler follows the standard 2-rule calling convention. The "current value" for the except handler is `ChainExcInfo(exc, root_value)` — a NamedTuple containing the caught exception and the chain's evaluated root value.

**Dispatch rules** (in priority order):

| Registration form | Handler invocation |
|---|---|
| `except_(handler)` | `handler(ChainExcInfo(exc, root_value))` |
| `except_(handler, arg1, arg2)` | `handler(arg1, arg2)` |
| `except_(handler, key=val)` | `handler(key=val)` — `ChainExcInfo` is NOT passed |
| `except_(nested_chain)` | `nested_chain` runs with `ChainExcInfo(exc, root_value)` as input |
| `except_(nested_chain, arg1)` | `nested_chain` runs with `arg1` as input |

Key behaviors:

- **Default (no explicit args):** `handler(ChainExcInfo(exc, root_value))`. The `ChainExcInfo` is the current value, so it is passed as the sole argument.
- **With explicit args:** `handler(*args, **kwargs)`. The `ChainExcInfo` is **not** passed — explicit arguments fully replace it, consistent with the standard convention.
- **Nested chain:** The nested chain runs with `ChainExcInfo(exc, root_value)` as its input value (standard default rule, Rule 2). With explicit args, the first arg becomes the input and remaining args/kwargs flow through.

**Execution mode for nested chain handlers:** The nested chain executes via the standard internal path, following the same calling convention as any other callable. This means:

- Control flow signals (`return_()`, `break_()`) raised inside the handler chain are caught by the except handler's own signal guard and wrapped in `QuentException`, enforcing the §6.2.6 restriction.
- The handler chain's own `except_()` and `finally_()` handlers apply independently.

#### 6.2.3 Exception Consumption vs Re-raise

- **`reraise=False` (default):** The handler's return value becomes the chain's result. The exception is consumed — it does not propagate. The chain is considered to have succeeded (the finally handler, if present, sees a success context).
- **`reraise=True`:** The handler runs for side-effects only (e.g., logging, alerting). After the handler completes, the original exception is re-raised. The handler's return value is ignored.

#### 6.2.4 Handler Failure with `reraise=True`

When the handler itself raises an exception while `reraise=True`:

- If the handler raises an `Exception` subclass: the handler's exception is **discarded**. A `RuntimeWarning` is emitted, a note is attached to the original exception (Python 3.11+), `__suppress_context__` is set to `True` on the original exception (to prevent the handler's exception from appearing in the implicit exception chain), and the original exception is re-raised. This ensures the caller always sees the original failure, even if the error-reporting handler is broken.
- If the handler raises a `BaseException` subclass (e.g., `KeyboardInterrupt`, `SystemExit`): the handler's exception propagates naturally — system signals are never suppressed.

#### 6.2.5 Handler Failure with `reraise=False`

When the handler itself raises while `reraise=False`, the handler's exception propagates. The original chain exception is set as the `__cause__` of the handler exception (via `raise handler_exc from original_exc`).

#### 6.2.6 Control Flow in Except Handlers

Using `Chain.return_()` or `Chain.break_()` inside an except handler raises `QuentException`. Control flow signals are not allowed in error handlers — they must be used in the main pipeline.

### 6.3 Cleanup Handler — `finally_(fn, /, *args, **kwargs)`

Registers a cleanup handler. At most one `finally_()` per chain; a second call raises `QuentException`.

#### 6.3.1 Handler Registration

- `fn` must be callable; `TypeError` is raised otherwise.
- Arguments follow the standard calling conventions (same as `then()`).

#### 6.3.2 Execution Semantics

- **Always runs** — on both success and failure paths, matching Python's `try/finally` semantics.
- **Receives the root value**, not the current pipeline value. The root value is normalized to `None` if the chain was created with no root value (i.e., `Chain()`).
- **Return value is always discarded** — the finally handler cannot alter the chain's result.
- **Follows the standard 2-rule calling convention** with the root value as the "current value."

#### 6.3.3 Finally Handler Failure

- If the finally handler raises while an exception is already active: the finally handler's exception **replaces** the original exception (matching Python's `try/finally` behavior). The original exception is preserved as `__context__` on the finally exception. A note is attached (Python 3.11+) describing the replaced exception.
- If the finally handler raises on the success path: the finally handler's exception propagates as the chain's error.
- If both `except_()` and `finally_()` handlers raise: the finally handler's exception propagates (suppressing the except handler's exception). The except handler's exception is preserved as `__context__` on the finally exception. This matches Python's `try/except/finally` semantics where `finally`'s exception always wins.

#### 6.3.4 Control Flow in Finally Handlers

Using `Chain.return_()` or `Chain.break_()` inside a finally handler raises `QuentException`. Control flow signals are not allowed in cleanup handlers.

#### 6.3.5 Async Finally in Sync Chains

When a sync chain's finally handler returns a coroutine, the engine performs an **async transition**: `run()` returns a coroutine instead of a plain value. When the caller awaits this coroutine, the finally handler's coroutine is awaited first, and then the chain's result is returned (success path) or the active exception is re-raised (failure path). The chain result flows through the async wrapper — nothing is discarded.

For `iterate()` / `iterate_do()`, the behavior differs: when a sync chain's finally handler returns a coroutine during sync iteration (`for` loop), the coroutine is closed and a `RuntimeWarning` is issued. The handler is **not** awaited. Use `async for` to ensure async finally handlers are properly awaited.

### 6.4 Execution Order

The full error handling flow for a chain execution:

1. Pipeline steps execute sequentially.
2. If a step raises an exception matching `exceptions`:
   a. The except handler runs (if registered).
   b. If `reraise=False`: handler's return value becomes the result; finally handler (if any) runs in success context.
   c. If `reraise=True`: original exception is re-raised; finally handler (if any) runs in failure context.
3. If a step raises an exception not matching `exceptions`: the exception propagates; finally handler (if any) runs in failure context.
4. On success: finally handler (if any) runs in success context.
5. The finally handler always runs last, regardless of what happened above.

### 6.5 ExceptionGroup

When concurrent operations (`gather()`, or `foreach()`/`foreach_do()` with `concurrency`) encounter multiple failures, the exceptions are wrapped in an `ExceptionGroup`. On Python 3.11+, the builtin `ExceptionGroup` is used. On Python 3.10, a polyfill is provided.

The polyfill implements:

- `ExceptionGroup(message, exceptions)` — constructor. Requires a non-empty list of `Exception` instances.
- `.exceptions` — tuple of contained exceptions.
- `.subgroup(condition)` — filter to matching exceptions.
- `.split(condition)` — split into `(matching, rest)` groups.
- `.derive(excs)` — create a new group with the same message but different exceptions, preserving traceback and cause/context chains.

When only a single function/worker fails, the exception is raised directly (not wrapped in an `ExceptionGroup`).

For concurrent iteration operations, the message follows the same pattern: `"foreach() encountered N exceptions"` or `"foreach_do() encountered N exceptions"`.

---

## 7. Control Flow

### 7.1 Design Rationale

Chains are linear pipelines with no built-in branching beyond `if_()`. For early exit and iteration control, quent provides two class-method signals: `Chain.return_()` and `Chain.break_()`. These are implemented as internal exceptions that propagate through the call stack — the user never sees these exceptions; they are caught by the chain execution engine or iteration operations.

The signals carry optional values with lazy evaluation: the signal's value (if callable) is only evaluated when the signal is caught, avoiding unnecessary work if the signal propagates through multiple nested chains.

### 7.2 Early Return — `Chain.return_(v=<no value>, /, *args, **kwargs)`

Class method. Signals early termination of chain execution. Must be called as `return Chain.return_(value)` to ensure the signal propagates through the call stack.

#### 7.2.1 Value Semantics

- **With no value:** `Chain.return_()` — the chain returns `None`.
- **With a non-callable value:** `Chain.return_(42)` — the chain returns the value as-is.
- **With a callable:** `Chain.return_(fn, *args, **kwargs)` — the callable is invoked when the signal is caught, and its return value becomes the chain's result. Dispatch: explicit args → `fn(*args, **kwargs)`, callable with no args → `fn()`, non-callable → value as-is.

#### 7.2.2 Nested Chain Propagation

When `return_()` is used inside a nested chain, the signal propagates up to the **outermost** chain. The nested chain does not catch the signal — it re-raises it so the parent chain can handle it. This enables patterns like:

```python
result = (
  Chain(data)
  .then(Chain().then(lambda x: Chain.return_('early') if x > 10 else x))
  .then(further_processing)
  .run()
)
# If data > 10: result = 'early' (further_processing is skipped)
```

#### 7.2.3 Restrictions

- **In except/finally handlers:** Raises `QuentException`. Control flow signals cannot be used inside error or cleanup handlers.
- **Signal escape:** If a return signal somehow escapes the chain's `run()` boundary (should not happen in normal usage), it is caught and wrapped in a `QuentException`.

### 7.3 Iteration Break — `Chain.break_(v=<no value>, /, *args, **kwargs)`

Class method. Signals early termination of a `foreach()` or `foreach_do()` iteration. Must be called as `return Chain.break_(value)`.

#### 7.3.1 Value Semantics

- **With no value:** `Chain.break_()` — the iteration returns results collected so far.
- **With a value:** `Chain.break_(value)` — the break value is **appended** to the results collected so far. The iteration returns the partial results list with the break value as the final element. This is uniform with `iterate()` behavior, where the break value is yielded as one additional item.
- **Callable values** follow the same conventions as `return_()`: called when the signal is caught.

Example:

```python
result = Chain([1, 2, 3, 4, 5]).foreach(
  lambda x: Chain.break_(x * 10) if x == 3 else x * 2
).run()
# result = [2, 4, 30]
# x=1 -> 2, x=2 -> 4, x=3 -> break with value 30 (appended to [2, 4])

result = Chain([1, 2, 3, 4, 5]).foreach(
  lambda x: Chain.break_() if x == 3 else x * 2
).run()
# result = [2, 4]
# x=1 -> 2, x=2 -> 4, x=3 -> break with no value (partial results preserved)
```

#### 7.3.2 Outside Iteration

Using `break_()` outside of a `foreach()`/`foreach_do()` operation raises `QuentException` with the message: `Chain.break_() cannot be used outside of a foreach/foreach_do iteration.`

#### 7.3.3 In Except/Finally Handlers

Raises `QuentException`. Control flow signals are not allowed in error or cleanup handlers.

#### 7.3.4 Concurrent Iteration Break

When `break_()` is used inside a concurrent iteration (with `concurrency` parameter):

- Multiple concurrent workers may raise `break_()` simultaneously.
- The break from the **earliest index** by original input order wins — not whichever worker happened to finish first in wall-clock time.
- Results from indices before the winning break index are collected; results from later indices are discarded.
- If both a regular exception and a break signal occur, the break signal takes priority regardless of index order.

#### 7.3.5 Priority in Concurrent Iteration

When multiple concurrent workers raise different signals, the priority is:

1. **Return signals** — `Chain.return_()` always takes priority (immediate propagation).
2. **Break signals** — take priority over regular exceptions. When multiple break signals occur, the one from the earliest index wins.
3. **Regular exceptions** — a single exception is propagated directly. Multiple exceptions are collected into an `ExceptionGroup`.

---

## 8. Execution

### 8.1 `run(v=<no value>, /, *args, **kwargs)`

**Contract:** Execute the pipeline and return the final value.

**Arguments:**
- `v` — optional value injected into the pipeline as the run value. Overrides the root value if both are present.
- `*args, **kwargs` — positional and keyword arguments for `v` when `v` is callable.

If `v` is not callable and `args` or `kwargs` are provided, a `TypeError` is raised immediately, matching the constructor's build-time enforcement.

If `v` is not provided (defaulting to no value) and `args` or `kwargs` are provided, a `TypeError` is raised — keyword arguments require a root value as the first positional argument, matching the constructor's rule.

**Behavior:**
- Execution walks the pipeline's steps in order, threading the current value through each step.
- The return value is either:
  - A plain value, if all steps were synchronous.
  - A coroutine, if any step returned an awaitable (the caller must `await` it).
- When an unawaited coroutine is returned and the caller does not await it, `finally_()` handlers will NOT execute and resources may leak. Python will emit a "coroutine was never awaited" warning.

**Run value vs root value:**
- `Chain(root)` sets a root value. `chain.run(v)` provides a run value.
- When both exist, the run value **replaces** the root — the build-time root is ignored entirely. `Chain(A).then(B).run(C)` is equivalent to `Chain(C).then(B).run()`.
- When only a root exists: the root is evaluated first, its result becomes the initial current value.
- When only a run value exists: `v` is evaluated first, its result becomes the initial current value.
- When neither exists: the pipeline starts with no current value. The first step receives no argument if it is callable.

**Root value capture:** The result of evaluating the root value (or run value, which replaces it) is captured as the "root value" for the `finally_()` handler. The `finally_()` handler receives the root value as its current value — not the current pipeline value at the point of completion or failure. The `except_()` handler receives `ChainExcInfo(exc, root_value)` as its current value.

**Root callable failure:** If the root value callable raises an exception during evaluation, the same error handling flow applies as for any other step failure:

- The `except_()` handler is invoked (if registered and the exception matches). It receives `ChainExcInfo(exc, root_value)` as its current value (per the standard calling convention).
- The `finally_()` handler runs (if registered). It receives the root value normalized to `None` (since no root value was successfully produced).
- This follows the same execution order described in Section 6.4.

**Error behavior:**
- If a control flow signal (`return_()` or `break_()`) escapes the pipeline, it is caught and wrapped in a `QuentException`. This indicates a bug in user code (e.g., `break_()` used outside an iteration context).
- If `if_()` was called but no subsequent `.then()` or `.do()` consumed it before `run()`, a `QuentException` is raised.

### 8.2 `__call__(v=<no value>, /, *args, **kwargs)`

**Contract:** Alias for `run()`.

**Behavior:** `chain(v)` is equivalent to `chain.run(v)`. Enables chains to be used as callables in any context that expects a function.

### 8.3 `__bool__`

`Chain.__bool__` always returns `True`. This prevents chains from being treated as falsy in boolean contexts — an empty chain (no root, no steps) is still truthy. Without this override, Python's default `__bool__` could return `False` for chains with certain internal states, leading to surprising behavior when chains are used in `if` checks or `or`/`and` expressions.

### 8.4 Sync/Async Execution Model

The execution engine implements a two-tier model:

1. **Sync tier:** Execution always starts synchronously. Each step is evaluated, and the result is checked for awaitability.
2. **Async transition:** On the first awaitable result, the engine immediately hands off to the async tier. The async tier receives all accumulated state (current value, root value, position in the pipeline) and continues execution, awaiting all subsequent awaitables inline.

This means:
- A fully synchronous pipeline has zero async overhead — no event loop is touched.
- The transition is seamless and one-way: once the pipeline goes async, it stays async. There is no "back to sync" transition.
- The caller sees a coroutine return from `run()` only if an async transition occurred. Otherwise, they get a plain value.
- Async transitions can occur not only during normal step execution, but also in `except_()` and `finally_()` handlers. If a sync chain's error or cleanup handler returns an awaitable, the engine transitions to async to complete the handler.

---

## 9. Iteration

### 9.1 `iterate(fn=None)`

**Contract:** Return a dual sync/async iterator over the chain's output. The chain is executed, and each element of its iterable result is yielded.

**Arguments:**
- `fn` — optional callable to transform each element before yielding. When `None`, elements are yielded as-is.

**Return value:** Returns a `ChainIterator` object — a callable dual-protocol iterator supporting both `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops).

**Behavior:**
- The chain is executed when iteration begins (not when `iterate()` is called).
- The chain's result must be iterable. Each element is passed through `fn` (if provided), and the transformed result is yielded.
- Supports both sync and async iteration:
  - `for item in chain.iterate(fn):` — synchronous iteration. If the chain or `fn` returns an awaitable, a `TypeError` is raised directing the user to use `async for`.
  - `async for item in chain.iterate(fn):` — asynchronous iteration. Awaitables from the chain and `fn` are automatically awaited.
- If the chain's result is a sync iterable but `async for` is used, the iterator automatically wraps it as an async iterable.

**Error behavior:**
- Exceptions from `fn` propagate to the caller at the iteration point.
- If `fn` returns an awaitable during sync iteration (`for`), a `TypeError` is raised with a message directing the user to use `async for`.

**Error handling:** The chain's `except_()` handler applies to the chain execution that produces the iterable (the `run()` phase). If the chain's execution raises an exception, the except handler is invoked as part of normal chain error handling before the exception reaches the iteration layer. Exceptions raised by the iteration callback `fn` during iteration are NOT covered by `except_()` — they propagate directly to the caller at the iteration point.

`finally_()` is **deferred**: rather than running immediately after the chain's `run()` phase, it runs in the generator's `finally:` block — after iteration ends. This ensures that resources acquired during the chain's run phase remain alive throughout the entire iteration. The deferred finally runs on all exit paths: normal exhaustion, generator `.close()`, `break`, `return`, `fn` errors during iteration, and chain errors during the run phase. For `run()`, `finally_()` behavior is unchanged — it runs immediately after chain execution.

### 9.2 `iterate_do(fn=None)`

**Contract:** Like `iterate()`, but `fn` runs as a side-effect. The original elements are yielded, not `fn`'s return values.

**Arguments:** Same as `iterate()`.

**Behavior:**
- `fn` is invoked for each element (for side-effects), but `fn`'s return value is discarded.
- The original element is yielded.
- All other behavior (sync/async support, error handling) matches `iterate()`.

### 9.3 Iterator Reuse via Calling

The returned iterator object is callable with signature `it(v=<no value>, /, *args, **kwargs)`, matching `run()`. Calling it creates a new iterator with those arguments as the run-time parameters for the chain's execution:

```python
it = chain.iterate(fn)
for item in it:          # runs chain with no arguments
  ...
for item in it(value):   # runs chain with `value` as the run value
  ...
```

Each call to the iterator returns a fresh iterator instance. The original iterator's configuration (fn, ignore_result) is preserved; only the run arguments change.

**Rationale:** This enables reusable iteration patterns. A single `iterate()` call defines the iteration shape, and repeated calls with different arguments execute the chain with different inputs.

### 9.4 Control Flow in Iteration

- **`return_(v)`**: During iteration, `return_()` yields the return value (if one is provided) and then stops iteration. The return value is evaluated following the standard calling convention — if it is callable, it is called; if it is a literal, it is yielded as-is. This differs from normal chain execution, where `return_()` replaces the chain's result entirely. In iteration, previously yielded values have already been emitted to the caller and cannot be "replaced." Therefore, the return value is yielded as a final item before stopping iteration, rather than replacing all prior output.
- **`break_(v)`**: Stops iteration. If a value is provided, it is yielded before stopping. If no value is provided, iteration stops immediately with no additional value yielded.

### 9.5 `flat_iterate(fn=None, *, on_exhaust=None)`

**Contract:** Return a dual sync/async flatmap iterator over the chain's output. Each element of the chain's iterable result is either iterated directly (when `fn` is `None`, flattening one level of nesting) or transformed by `fn` into a sub-iterable whose items are individually yielded.

**Arguments:**
- `fn` — optional callable that receives each element and returns an iterable. Each item from the returned iterable is yielded individually. When `None`, each source element is iterated directly (flattening one level).
- `on_exhaust` — optional zero-argument callable invoked once after the source iterable is fully consumed. Must return an iterable; each item is yielded into the stream. Intended for emitting buffered or remaining items after the source ends (e.g., flushing a codec buffer).

**Return value:** Returns a `ChainIterator` object (same as `iterate()`).

**Behavior:**
- Flattens one level of nesting: source `[[1, 2], [3]]` yields `1, 2, 3`.
- When `fn` is provided: each source element is passed to `fn`, and each sub-item from `fn`'s returned iterable is yielded.
- When `fn` is `None`: each source element is iterated directly (it must be iterable).
- After the source is exhausted, if `on_exhaust` is provided, `on_exhaust()` is called and each item from its return value is yielded into the stream.
- All iteration behavior — sync/async support, error handling, deferred `finally_()`, control flow (§9.4), iterator reuse (§9.3) — matches `iterate()` (§9.1).

**Error behavior:**
- Exceptions from `fn` propagate to the caller at the iteration point, as with `iterate()`.
- If `on_exhaust()` raises, the exception propagates at the iteration point.
- If `fn` or `on_exhaust` returns an awaitable during sync iteration (`for`), a `TypeError` is raised directing the user to use `async for`.

### 9.6 `flat_iterate_do(fn=None, *, on_exhaust=None)`

**Contract:** Like `flat_iterate()`, but `fn` runs as a side-effect — its returned iterable is fully consumed (driving side-effects) but not yielded. The original source elements are yielded instead.

**Arguments:** Same as `flat_iterate()`.

**Behavior:**
- `fn` is invoked for each element. Its returned iterable is fully consumed (executing side-effects), but the sub-items are discarded.
- The original source element is yielded for each iteration step.
- `on_exhaust` output is yielded normally (not discarded) — the "do" discard semantic applies only to `fn`'s results.
- When `fn` is `None`: behaves identically to `flat_iterate()` with no `fn` (flattens one level).
- All other behavior matches `flat_iterate()`.

### 9.7 Deferred `with_` in Iteration

**Contract:** When `with_(fn)`, `with_do(fn)`, or bare `with_()` is the last pipeline step before an iteration terminal (`iterate()`, `iterate_do()`, `flat_iterate()`, `flat_iterate_do()`), context manager entry is **deferred** to iteration time. The context manager remains open for the entire duration of iteration and is exited when iteration ends.

**Motivation:** Without deferral, `with_()` would enter the CM during the chain's `run()` phase and exit it before iteration begins — the resource would be closed before any items are consumed. Deferral keeps the CM open throughout iteration, matching the natural lifetime of `with` blocks in Python.

**Detection:** The iterate method inspects the chain's last link. If it holds a `_WithOp`, the inner link and `ignore_result` flag are extracted. The `_WithOp` is executed in the generator rather than during the chain's run phase.

**Lifecycle:**
1. The chain runs normally, producing a value that must be a context manager.
2. At iteration start, the CM is entered via `__enter__()` (or `__aenter__()`).
3. If `with_(fn)` was used: `fn` is invoked with the context value per the standard calling convention. The result becomes the iterable for iteration.
4. If `with_do(fn)` was used: `fn` runs as a side-effect (result discarded); the CM object itself becomes the iterable (it must be iterable).
5. If bare `with_()` was used: the context value (the `__enter__` result) becomes the iterable directly.
6. Iteration proceeds with the CM open.
7. The CM is exited in the generator's `finally:` block, guaranteeing cleanup on all exit paths.

**CM exit semantics:**
- **Normal completion / source exhausted:** `__exit__(None, None, None)`.
- **`break`, `return_()`, `break_()` (control flow):** `__exit__(None, None, None)` — control flow signals are not errors.
- **Generator `.close()` / `GeneratorExit`:** `__exit__(None, None, None)`.
- **Exception during iteration:** `__exit__(*sys.exc_info())` — the CM receives the exception. If `__exit__` returns truthy, the exception is suppressed and the generator stops cleanly. If falsy, the exception propagates.
- **`__exit__` itself raises:** The new exception replaces the original (if any), matching Python's native `with` statement behavior.

**Ordering with deferred `finally_()`:** When both a deferred `with_` and a deferred `finally_()` are active, the CM exits first, then the deferred `finally_()` runs. This is enforced by nesting: `try { cm.__exit__ } finally { deferred_finally }`. The deferred finally runs even if `__exit__` raises.

**Protocol selection:** For dual-protocol CMs (supporting both `__enter__`/`__exit__` and `__aenter__`/`__aexit__`), the async protocol is preferred when an async event loop is running (asyncio, trio, or curio), matching §5.6. The exit protocol always matches the entry protocol.

**Sync/async rules:**
- Sync iteration (`for`): only `__enter__`/`__exit__` is used. If the CM only supports async protocol, or if the inner `fn` returns an awaitable, a `TypeError` is raised directing the user to use `async for`.
- Async iteration (`async for`): both protocols are supported, with async preferred for dual-protocol CMs.

---

## 10. Reuse

### 10.1 `clone()`

**Contract:** Create an independent copy of the chain. The clone can be extended without affecting the original.

**Return value:** A new Chain of the same type (subclass-safe via `type(self).__new__`).

**What is copied:**
- The pipeline structure (all step nodes) is deep-copied — the clone has its own independent linked list.
- Nested chains within steps are recursively cloned via their own `clone()` method. This prevents cross-clone state sharing (e.g., nested/top-level execution state, concurrent execution of shared inner chains).
- Conditional operations (`if_`/`else_`) are deep-copied because they carry mutable state (the else branch reference).
- Error handlers (`except_`/`finally_`) have their step nodes cloned. If the handler callable is a `Chain` instance, it is recursively cloned via `clone()` — the same treatment as nested chains in pipeline steps. Non-chain handler callables are shared by reference.
- Keyword argument dictionaries are shallow-copied (since dicts are mutable). Positional argument tuples are shared by reference (tuples are immutable).
- The chain's name label (from `.name()`) is copied by value.

**What is shared by reference:**
- All callables (functions, lambdas, bound methods) across all steps and handlers — except Chain instances, which are always recursively cloned.
- Values and argument objects (individual args tuple elements, kwargs dict values).
- Exception type tuples for `except_()`.

**Rationale:** Deep-copying the linked list structure but sharing callables by reference strikes the right balance: clones are structurally independent (extending one doesn't affect the other), but the overhead is minimal since the expensive objects (callables, data) are shared. Recursive cloning of nested chains is necessary because nested chains carry mutable state (`is_nested`) that would cause bugs if shared.

**State reset:** Clones always behave as top-level chains by default, regardless of whether the original was being used as a nested chain. When a clone is subsequently used as a step in another chain, it adopts nested behavior at that point.

### 10.2 `decorator()`

**Contract:** Wrap the chain as a function decorator. The decorated function's return value becomes the chain's input.

**Return value:** A decorator function that can be applied to other functions via `@`.

**Behavior:**
- The chain is cloned internally when `decorator()` is called. This prevents the decorator from sharing mutable state with the original chain.
- When the decorated function is called, its arguments are forwarded to the original function, and the function's return value is used as the run value for the cloned chain.
- The chain's steps are then applied to the function's return value.
- The chain always executes as a top-level chain (independent of the original chain's execution context), ensuring thread-safe concurrent calls to the decorated function.

**Example:**
```python
@Chain().then(lambda x: x.strip()).then(str.upper).decorator()
def get_name():
  return '  alice  '

get_name()  # 'ALICE'
```

**Rationale for cloning:** Without cloning, the decorator and the original chain would share the same linked list. Mutations to either (e.g., adding steps) would affect both. Cloning ensures the decorator is a stable snapshot of the chain at the time `decorator()` was called.

**Error behavior:**
- If `if_()` was called but no subsequent `.then()` or `.do()` consumed it before `decorator()`, a `QuentException` is raised.
- Control flow signals that escape the decorated chain are caught and wrapped in `QuentException`, consistent with `run()` semantics.
- The decorated function preserves its original signature via `functools.wraps`.

### 10.3 `from_steps()`

**Contract:** Construct a chain from a sequence of steps, each appended via `.then()`.

**Arguments:**
- `*steps` — variadic positional arguments. Each becomes a `.then()` step.
- If a single argument is passed and it is a `list` or `tuple`, it is unpacked as the step sequence. This allows both `Chain.from_steps(a, b, c)` and `Chain.from_steps([a, b, c])`.

**Return value:** A new `Chain` instance with no root value.

**Behavior:**
- Creates an empty chain (no root value).
- Each step is appended via the same mechanism as `.then()` — standard calling conventions apply.
- Steps can be callables, literal values, or nested Chains — anything `.then()` accepts.
- Returns the constructed chain.
- `Chain.from_steps()` with no arguments returns an empty chain (equivalent to `Chain()`).

**Equivalence:** `Chain.from_steps(a, b, c)` is equivalent to `Chain().then(a).then(b).then(c)`.

**Use case:** Dynamic pipeline construction from plugin registries, configuration-driven workflows, or any scenario where the step sequence is determined at runtime.

### 10.4 Subclassing

`Chain` supports basic subclassing:

- `clone()` and `decorator()` are subclass-safe — they create new instances via `type(self).__new__`, so a subclass of `Chain` will produce clones/decorators of the same subclass type.
- `on_step` subclass overrides are respected — the engine reads `on_step` via `type(chain).on_step`, so a subclass can define its own instrumentation callback without affecting other `Chain` subclasses.
- No other subclassing guarantees are made. Internal implementation details (linked list structure, execution engine behavior) are not part of the subclassing contract.

---

## 11. Concurrency

### 11.1 Design Rationale

Concurrency in quent serves iteration (`foreach`, `foreach_do`) and gather operations. The `concurrency` parameter controls how many items/functions execute simultaneously. The design priorities are:

- **Deterministic cleanup:** A new executor/task group is created per invocation and shut down immediately after. No shared executor state leaks between calls.
- **Sync/async transparency:** The first item/function is probed to detect sync vs async execution. The rest follow the same path.
- **Context propagation:** `contextvars` propagate to thread workers, preserving user context.

### 11.2 The `concurrency` Parameter

Available on `foreach()`, `foreach_do()`, and `gather()`.

- **`None`:** Sequential execution. Only valid for iteration operations (`foreach`, `foreach_do`). Elements are processed one at a time.
- **`-1` (unbounded):** All items/functions run concurrently with no limit. The effective concurrency is resolved to `len(items)` or `len(fns)` at runtime. This is the default for `gather()`.
- **Positive integer:** Limits the number of simultaneous executions to this value.

#### 11.2.1 Bounds

- **Minimum:** `1` for bounded concurrency, or `-1` for unbounded. Values of `0` or less than `-1` (i.e., `-2`, `-3`, etc.) raise `ValueError` with the message `"{method}() concurrency must be -1 (unbounded) or a positive integer, got {value}"`, where `{method}` is the calling method name (e.g., `foreach`, `gather`).
- **Type:** Must be an integer (`-1` or positive). Booleans and non-integer types raise `TypeError`.

#### 11.2.2 The `executor` Parameter

Available on `foreach()`, `foreach_do()`, and `gather()`.

- **`None` (default):** A new `ThreadPoolExecutor` is created per invocation and shut down immediately after. This is the standard behavior — deterministic cleanup, no shared state.
- **`Executor` instance:** The provided executor is used for sync concurrent execution. Quent does NOT shut it down after the operation completes — lifecycle management is the caller's responsibility.

**Scope:** Only affects the sync concurrent path (thread pool). The async path (semaphore + TaskGroup/`asyncio.gather`) is not affected.

**For `foreach()`/`foreach_do()`:** The `executor` parameter only has effect when `concurrency` is also set. On the sequential path (no `concurrency`), the `executor` argument is ignored.

**For `gather()`:** Always concurrent, so `executor` always applies to the sync path.

**Type constraint:** Must be a `concurrent.futures.Executor` instance. Non-`Executor` values raise `TypeError`.

**Context propagation:** Worker submissions still use `copy_context().run()` regardless of whether the executor was created by quent or provided by the caller.

### 11.3 Sync Concurrent Execution

When the first item/function returns a non-awaitable result, the sync concurrent path is used.

**Mechanism:** When no user-provided executor is given (i.e., `executor=None`), a new `ThreadPoolExecutor` is created with `max_workers` set to `min(concurrency, number_of_remaining_items)` and shut down with `wait=True` immediately after all futures complete. When an executor is provided via the `executor` parameter, it is used directly and not shut down after the operation — the caller manages its lifecycle. All remaining items/functions (after the first, which was already probed) are submitted to the pool.

**Key behaviors:**

- **One executor per invocation:** Guarantees deterministic thread cleanup. No thread pool is shared across invocations.
- **First item is probed synchronously:** The first item is evaluated in the calling thread to detect sync vs async. Only subsequent items go to the thread pool.
- **Awaitable detection:** If a thread worker returns an awaitable, a `TypeError` is raised — once the sync path is chosen (based on the first item), all workers must be sync. The awaitable is closed to avoid resource warnings.
- **Memory ordering:** `concurrent.futures.wait()` establishes happens-before edges via the `Future`'s internal `threading.Condition`, ensuring worker writes are visible to the calling thread. This is safe under both GIL and free-threaded (PEP 703) Python.

### 11.4 Async Concurrent Execution

When the first item/function returns an awaitable, the async concurrent path is used.

**Mechanism:** An `asyncio.Semaphore` limits the number of concurrent tasks. Tasks are created and managed differently depending on the Python version:

- **Python 3.11+:** `asyncio.TaskGroup` is used. Task failures are collected via the `ExceptionGroup` that `TaskGroup` raises.
- **Python 3.10:** `asyncio.gather()` is used as a fallback. On failure, pending tasks are cancelled and awaited before exceptions are triaged.

**Key behaviors:**

- **Semaphore limiting:** Each worker acquires the semaphore before executing, ensuring at most `concurrency` tasks run simultaneously.
- **Async iterables:** When the input is an async iterable (has `__aiter__` but not `__iter__`), it is fully materialized into a list before concurrent processing begins. This is required because all items must be available upfront for dispatch. For large or unbounded async iterables, use the sequential (non-concurrent) variant.

### 11.5 Sync/Async Detection

The sync vs async execution path is determined by probing the first item or function:

1. The first item/function is called.
2. If the result is awaitable → async path (Semaphore + TaskGroup/gather).
3. If the result is not awaitable → sync path (ThreadPoolExecutor).
4. If a later worker returns an awaitable on the sync path → `TypeError` is raised.

This probe-based detection means the callable must be **consistently** sync or async across all items. Mixed sync/async callables within a single concurrent operation are not supported.

### 11.6 Async Transition for Sync Chain Handlers

See Section 6.3.5 for the `finally_()` async transition behavior. The same mechanism applies to `except_()` handlers:

**`except_()` with `reraise=True`:** When an except handler with `reraise=True` returns a coroutine in a sync chain, the behavior is **async transition**. `run()` returns a coroutine (the pipeline transitions to async). The caller awaits it, the handler completes, and then the original exception is re-raised. This ensures the handler's side-effects (e.g., async logging) are reliably completed before the exception propagates.

**`except_()` with `reraise=False`:** When the handler returns a coroutine and `reraise=False`, this is a normal async transition — the coroutine becomes the chain's result, and `run()` returns it for the caller to `await`.

### 11.7 Context Variable Propagation

`contextvars` are propagated to `ThreadPoolExecutor` workers via `copy_context().run()`. This ensures:

- **User-defined context variables** are visible to worker threads with the values they had when the task was submitted.

Async concurrent tasks inherit context naturally through asyncio's task creation mechanism.

---

## 12. Null Sentinel

### 12.1 Design Rationale

Python's `None` is a legitimate pipeline value. A chain created with `Chain(None)` should be able to pass `None` through its pipeline without ambiguity. The `Null` sentinel exists to distinguish "no value was provided" from "the value is `None`":

- `Chain()` — creates a chain with **no** root value (internal representation: `Null`).
- `Chain(None)` — creates a chain with root value `None`.

This distinction affects calling conventions: when the current value is `Null`, a callable in the pipeline is called with zero arguments. When the current value is `None` (or any other non-Null value), it is passed as the first argument.

### 12.2 External Boundary — Null is Never Exposed

Null is an internal implementation concept. It is **never** exposed to user code:

- **`run()` returns `None`:** When all pipeline steps use `.do()` (ignore result) and there is no root value, `run()` returns `None`, not Null.
- **Finally handler receives `None`:** The root value passed to the finally handler is normalized — `None` is passed when the chain has no root value.

This normalization happens at chain boundaries. Within the pipeline execution, Null flows internally to distinguish "no value" from `None`, but no user-visible callback ever sees Null.

### 12.3 Effect on Calling Conventions

The Null sentinel has a direct effect on how pipeline steps are called:

- **Current value is Null, callable has no explicit args:** The callable is called with **zero** arguments — `fn()`.
- **Current value is not Null (including `None`), callable has no explicit args:** The current value is passed as the first argument — `fn(current_value)`.
- **Explicit args provided:** The current value is not passed regardless of Null/non-Null status — `fn(*args, **kwargs)`.

This means `Chain().then(fn).run()` calls `fn()`, while `Chain(None).then(fn).run()` calls `fn(None)`.

### 12.4 Internal Safeguards

The following properties are internal implementation details that prevent misuse within the engine. Users never interact with `Null` directly — these safeguards exist as defense-in-depth.

- `Null` is a singleton — exactly one instance exists for the lifetime of the process.
- Duplicate instantiation of the singleton's type is blocked with a `TypeError`.
- Copying (`copy.copy`, `copy.deepcopy`) returns the same instance.
- `repr(Null)` returns `'<Null>'`.

---

## 13. Traceback Enhancement

### 13.1 Purpose

When an exception propagates out of a chain, quent injects a visualization of the chain structure into the traceback. The visualization shows every step in the chain and marks the step that raised the exception with a `<----` arrow. This transforms a generic Python traceback — which would show only quent-internal frames — into a diagnostic view of the user's pipeline.

### 13.2 Chain Visualization Injection

When an exception reaches the outermost chain's error handling boundary (the `except` block in the execution engine), a synthetic `<quent>` frame is constructed and grafted onto the exception's traceback. The synthetic frame's "function name" field contains the chain visualization string.

**Visualization format:**

```
Chain(fetch_data)
    .then(validate)
    .do(log) <----
    .foreach(transform)
    .except_(handle_error)
    .finally_(cleanup)
```

The `<----` marker points to the step that raised the exception. This appears in Python's traceback output as if it were a function name in a file called `<quent>`.

**Nested chains** are rendered with indentation:

```
Chain(fetch)
    .then(
        Chain(parse)
            .then(validate) <----
    )
    .do(log)
```

**Observable behavior:**

- When an exception propagates from a chain, the traceback includes a `<quent>` frame containing the chain visualization.
- The `<----` marker appears on exactly one step — the step that raised the exception.
- If the chain has `except_()` or `finally_()` handlers, they appear in the visualization.
- If the chain contains `if_()` with `else_()`, both branches appear.
- Nested chains are recursively rendered with increasing indentation (4 spaces per level).

### 13.3 Error Marker (`<----`)

The `<----` marker identifies the failing step in the chain visualization. It is placed on the step whose evaluation raised the exception, determined by exception metadata recorded at the point of failure.

**First-write-wins semantics:** When an exception propagates through nested chains, only the innermost (first) failing step is recorded. The marker always points to the deepest origin of the error, not an intermediate step that re-raised it.

**Rationale:** First-write-wins preserves the innermost failure context. When a nested chain raises and the exception propagates through outer chains, the user needs to see where the error *originated*, not where it was re-raised. Recording only the first write ensures the marker points to the root cause.

### 13.4 Frame Cleaning

Quent strips its own internal frames from exception tracebacks, leaving only:

1. **User code frames** — frames from files outside the quent package directory.
2. **Synthetic `<quent>` frames** — the injected visualization frame.

**Observable behavior:**

- When viewing a traceback from a chain exception, no frames from quent's internal modules appear.
- Only the user's code and the synthetic `<quent>` visualization frame are visible.
- The frame cleaning applies to the exception itself and to all chained exceptions (`__cause__` and `__context__` chains).

### 13.5 Chained Exception Cleaning

Frame cleaning applies recursively to chained exceptions:

- `__cause__` — exceptions linked via `raise ... from ...`.
- `__context__` — implicitly chained exceptions (exception raised while handling another).
- `ExceptionGroup` sub-exceptions (Python 3.11+) — each sub-exception in the group has its frames cleaned.

A depth limit (1000 exceptions) prevents unbounded traversal in pathological exception chains. A seen-set prevents infinite loops from circular exception references.

### 13.6 Exception Notes (Python 3.11+)

On Python 3.11 and later, quent attaches a concise one-line note to exceptions via `exc.add_note()`. The note identifies the failing step and the chain it belongs to:

```
quent: exception at .then(validate) in Chain(fetch_data)
```

**Observable behavior:**

- The note is attached only once per exception (idempotent — if a `quent:` note already exists, no duplicate is added).
- Notes survive traceback reformatting and stripping, providing a fallback when the full visualization is lost.
- If note generation fails, the failure is silently logged and the exception propagates unmodified.
- When the chain has a name (via `.name()`), it appears in the note: `quent: exception at .then(validate) in Chain[auth_pipeline](fetch_data)`.

### 13.7 Environment Variables

Two environment variables control traceback behavior:

#### `QUENT_NO_TRACEBACK=1`

Disables all traceback modifications. When set (to `1`, `true`, or `yes`, case-insensitive) before quent is imported:

- No chain visualizations are injected into tracebacks.
- No internal frames are cleaned.
- No `sys.excepthook` or `TracebackException.__init__` patches are installed.
- Exceptions propagate with their original, unmodified tracebacks.

**Rationale:** Some environments (debuggers, custom exception handlers, CI systems) need unmodified tracebacks. This variable provides a clean opt-out.

#### `QUENT_TRACEBACK_VALUES=0`

Suppresses argument values in chain visualizations while preserving step names and chain structure. When set (to `0`, `false`, or `no`, case-insensitive):

- Step names still appear (e.g., `.then(fetch)`, `.foreach(transform)`).
- Argument values and `repr()` output of pipeline values are replaced with type-name placeholders (e.g., `<str>` instead of `'secret_api_key'`).
- Debug log output respects the same suppression.

**Rationale:** Production environments may process sensitive data (API keys, user credentials, financial data). Tracebacks are often logged or sent to error tracking services. Suppressing values prevents sensitive pipeline data from leaking into logs and third-party systems.

### 13.8 Global Patches

Quent installs two global patches at import time (unless `QUENT_NO_TRACEBACK=1` is set):

#### `sys.excepthook` Replacement

The default `sys.excepthook` is replaced with a wrapper that cleans quent-internal frames from any exception that carries quent metadata before passing it to the original hook. This ensures the top-level exception display (interactive interpreter, script crashes) shows cleaned tracebacks.

#### `traceback.TracebackException.__init__` Patch

The `TracebackException.__init__` method is patched so that all rendering paths — `logging`, `traceback.format_exception()`, `traceback.print_exception()`, etc. — receive cleaned tracebacks. This covers code that formats exceptions without going through `sys.excepthook`.

**Patch safety:**

- The original hook and `__init__` are captured once at first import time. Subsequent `importlib.reload()` calls do not re-capture (which would create infinite recursion).
- Idempotency guards prevent stacking patches on reload.
- The `TracebackException.__init__` signature is verified at import time; a warning is emitted if the signature does not match the expected positional parameters.

### 13.9 Repr Sanitization (CWE-117)

All `repr()` output included in chain visualizations is sanitized as a defense-in-depth measure against log injection (CWE-117):

- **ANSI escape sequences** are stripped (CSI, OSC, and simple ESC sequences). This prevents malicious `__repr__` implementations from injecting terminal-manipulating sequences.
- **Unicode control characters** are stripped (C0/C1 controls except tab, newline, carriage return; zero-width characters; bidirectional overrides; byte order marks). This prevents invisible characters from confusing log parsers and terminal emulators.
- **Repr length** is truncated to 200 characters to prevent excessively large objects from bloating tracebacks.

**Rationale:** Chain visualizations appear in tracebacks and log output. A malicious or buggy `__repr__` implementation could inject ANSI escape sequences to manipulate terminal display, overwrite log lines, or confuse log aggregation systems. Sanitization ensures that visualization output is safe for all display contexts.

### 13.10 Visualization Limits

To prevent pathological chains from producing unbounded output:

- **Nesting depth limit:** Nested chain rendering is truncated at depth 50 with a `Chain(...<truncated at depth 50>...)` message.
- **Links per level:** At most 100 links are rendered per chain level. Additional links are summarized as `... and N more steps`.
- **Total visualization length:** Capped at 10,000 characters. Excess is truncated with `... <truncated>`.
- **Total recursive calls:** Capped at 500 per chain instance to prevent runaway rendering of deeply nested structures. Each nested chain gets its own budget.

### 13.11 Graceful Degradation

Visualization is best-effort. If visualization construction fails for any reason:

- A `RuntimeWarning` is emitted describing the failure.
- The failure is logged at DEBUG level.
- The exception's traceback is cleaned of internal frames (fallback behavior) — the exception still propagates with cleaned frames, just without the chain visualization.
- The underlying exception is never suppressed or altered by a visualization failure.

### 13.12 `__repr__` via Visualization

`repr(chain)` uses the same visualization format as traceback injection:

- **Format:** identical to the traceback chain visualization (multiline, indented), but without the `<----` error marker.
- **Name support:** when `.name(label)` has been called, the chain renders as `Chain[label](root)`.
- **`QUENT_TRACEBACK_VALUES=0`:** respected — argument values are replaced with type-name placeholders.
- **Visualization limits:** the same depth limit (50), links-per-level limit (100), total-length limit (10,000 characters), and total-calls limit (500) apply.
- **Example** (unnamed chain):
  ```
  Chain(fetch_data)
      .then(validate)
      .do(log)
      .foreach(transform)
  ```
- **Example** (named chain):
  ```
  Chain[auth_pipeline](fetch_data)
      .then(validate)
  ```

---

## 14. Instrumentation (`on_step`)

### 14.1 Class-Level Callback

`Chain.on_step` is a class-level attribute that, when set, is called after each step completes during chain execution.

```python
Chain.on_step = my_callback  # Enable
Chain.on_step = None          # Disable (default)
```

**Signature:**

```python
def on_step(chain: Chain, step_name: str, input_value: Any, result: Any, elapsed_ns: int) -> None
```

- `chain` — the `Chain` instance being executed.
- `step_name` — the name of the step that just completed. For the root value, this is `'root'`. For pipeline steps, this is the method name that registered the step: `'then'`, `'do'`, `'foreach'`, `'foreach_do'`, `'gather'`, `'with_'`, `'with_do'`, `'if_'`, `'except_'`, or `'finally_'`. The `'if_'` step name covers the entire conditional operation — `on_step` fires with `step_name='if_'` regardless of which branch was taken (truthy or falsy). The `'else_'` and `'else_do'` names appear in chain visualization but are not reported as separate `on_step` events; they are part of the `if_` operation.
- `input_value` — the current pipeline value that was passed to the step, normalized to `None` if absent (the internal `Null` sentinel is never exposed). For the root step, this is the run value (or `None` if no run value was provided). For `except_` steps, this is the `ChainExcInfo` that was passed to the handler. For `finally_` steps, this is the root value (or `None`). For all other steps, it is the current pipeline value before the step executed.
- `result` — the value produced by the step.
- `elapsed_ns` — wall-clock nanoseconds elapsed for this step, measured via `time.perf_counter_ns()`.

### 14.2 Zero Overhead When Disabled

When `on_step` is `None` (the default), no timing is performed and no callback dispatch occurs. The engine skips all instrumentation overhead entirely. This is not a no-op callback — the code path is short-circuited so there is genuinely zero instrumentation cost in the default case.

**Rationale:** Chain execution is a hot path. Any unconditional overhead — even checking a flag — accumulates across millions of step evaluations. The engine reads `on_step` once at the start of execution and uses a single boolean to gate all timing and callback logic.

### 14.3 Error Handling

If the `on_step` callback raises an exception:

- The exception is logged at WARNING level via the `'quent'` logger.
- A `RuntimeWarning` is emitted.
- Chain execution continues uninterrupted — the callback failure does not affect the pipeline's result or error handling.

**Rationale:** Instrumentation must never break the instrumented code. A logging callback that crashes should not cause a production pipeline to fail. The warning ensures the developer is alerted to fix the callback, without disrupting the application.

### 14.4 Thread Safety

`on_step` is a class-level attribute (not per-instance). It must be set before any concurrent chain execution begins.

**Observable behavior:**

- Mutating `on_step` while chains are executing concurrently is not safe and constitutes a data race under free-threaded Python (PEP 703).
- Subclass overrides are respected: the engine reads `on_step` via `type(chain).on_step`, so a subclass can define its own `on_step` without affecting other Chain subclasses.
- The callback itself must be thread-safe if chains execute concurrently — multiple threads may invoke the callback simultaneously.

**Rationale:** A class-level attribute avoids per-instance storage overhead. Since instrumentation is typically global (e.g., metrics collection, debug logging), class-level granularity matches the common use case. Users who need per-instance dispatch can use the `chain` argument within the callback to differentiate.

### 14.5 Debug Logging

The execution engine emits debug-level log messages via the `'quent'` logger at key points:

- **Chain start:** `[exec:<id>] chain <repr>: run started`
- **Step completion:** `[exec:<id>] chain <repr>: <step_name> -> <result_repr>`
- **Async transition:** `[exec:<id>] chain <repr>: async transition at <step_name>`
- **Chain completion:** `[exec:<id>] chain <repr>: completed -> <result_repr>`
- **Step failure:** `[exec:<id>] chain <repr>: failed at <step_name>: <exc_repr>`
- **Async continuation started:** `[exec:<id>] chain <repr>: async continuation started`

The `<id>` is a zero-padded 6-digit hexadecimal execution counter (e.g., `[exec:00002a]`), unique per `run()` invocation. It correlates log lines from the same chain execution, including across async transitions.

Debug logging is gated by `_log.isEnabledFor(DEBUG)` — when the logger is not at DEBUG level, no `repr()` calls or string formatting occurs.

When `QUENT_TRACEBACK_VALUES=0` is set, `repr()` output in debug logs is replaced with type-name placeholders (e.g., `<str>`) to prevent sensitive data leakage.

---

## 15. Context API

Pipeline steps are positional — each step receives only the current value from the immediately preceding step. When non-adjacent steps need to share data (e.g., an early step produces a value that a later step needs, but intermediate transformations change the current value), the alternatives are threading values through tuples or capturing them in closures. The context API provides a cleaner mechanism: named storage scoped to the execution context, accessible from any step without altering the pipeline's value flow.

### 15.1 `set(key)` / `set(key, value)` — Instance Method (Pipeline Step)

**Signatures:**
- `chain.set(key: str) -> Self`
- `chain.set(key: str, value: Any) -> Self`

Appends a pipeline step that stores a value under `key` in the execution context. The current value is **not changed** — the step's result is discarded, like `.do()`. The chain is returned for fluent chaining.

- **One-arg form** `chain.set(key)` — stores the current pipeline value under `key`.
- **Two-arg form** `chain.set(key, value)` — stores the explicit `value` under `key`. The current value is still unchanged.

```python
result = (
  Chain(fetch_user)
  .set('user')                        # store CV (the user) in context
  .set('source', 'api')              # store explicit value 'api' under 'source'
  .then(validate_permissions)         # transform continues with original user
  .get('user')                        # retrieve original user
  .then(format_response)
  .run(user_id)
)
```

**Calling convention:** The internal callable that performs the store follows Rule 2 (default calling convention). For the one-arg form, when a current value exists, it receives the current value as its sole argument; when no current value exists, it is called with no arguments. For the two-arg form, the explicit value is captured in a closure and the current value is ignored. The `Null` sentinel is normalized to `None` before storage — context values never contain `Null`.

**`if_()` constraint:** `.set()` does not consume a pending `if_()`. Calling `.set()` while `if_()` is pending raises `QuentException`, consistent with §5.8 (only `.then()`/`.do()` consume `if_()`). For conditional storage, use `.if_(pred).do(lambda cv: Chain.set('key', cv))`.

**Implementation detail:** Both forms use `ignore_result=True` internally, which is the same mechanism `.do()` uses to discard return values.

### 15.2 `Chain.set(key, value)` — Class-Level Call

**Signature:** `Chain.set(key: str, value: Any) -> None`

Stores an explicit `value` under `key` in the execution context immediately. This is **not** a pipeline step — it takes effect at the call site, not during `run()`. Returns `None`.

```python
Chain.set('config', load_config())    # pre-populate context
result = (
  Chain(fetch_data)
  .then(lambda data: process(data, Chain.get('config')))
  .run()
)
```

This form is useful for pre-populating context before running a pipeline, or for setting values from outside any pipeline.

### 15.3 `get(key)` / `Chain.get(key)` — Dual Dispatch

Like `set`, `get` is a descriptor (`_GetDescriptor`) that dispatches differently based on instance vs. class access.

**Instance access — pipeline step:**

**Signatures:**
- `chain.get(key: str) -> Self`
- `chain.get(key: str, default: Any) -> Self`

Appends a pipeline step that retrieves the value stored under `key` from the execution context. The retrieved value **replaces** the current value (like `.then()`). Returns the chain for fluent chaining.

- If `key` is found: the stored value becomes the new current value.
- If `key` is not found and no `default` was provided: raises `KeyError` at execution time.
- If `key` is not found and `default` was provided: the default becomes the new current value.

```python
result = (
  Chain(fetch_user)
  .set('user')                        # store user in context
  .then(transform)                    # current value changes
  .get('user')                        # retrieve original user → becomes CV
  .then(format_response)
  .run(user_id)
)
```

**`if_()` constraint:** `.get()` does not consume a pending `if_()`. Calling `.get()` while `if_()` is pending raises `QuentException`, consistent with §5.8 (only `.then()`/`.do()` consume `if_()`). For conditional retrieval, use `.if_(pred).then(Chain.get, 'key')`.

**Class access — immediate retrieval:**

**Signature:** `Chain.get(key: str, default: Any = <missing>) -> Any`

Retrieves a value from the execution context by `key` immediately. This is **not** a pipeline step — it takes effect at the call site, not during `run()`.

- If `key` is found: returns the stored value.
- If `key` is not found and no `default` was provided: raises `KeyError`.
- If `key` is not found and `default` was provided: returns `default` (like `dict.get()`).

```python
Chain.set('config', load_config())
result = (
  Chain(fetch_data)
  .then(lambda data: process(data, Chain.get('config')))  # immediate retrieval inside lambda
  .run()
)
```

### 15.4 Storage and Scoping

**Storage mechanism:** A single module-level `ContextVar[dict[str, Any]]` holds the context dictionary. Each `set()` operation creates a **new** dict (spread of the existing dict plus the new key) rather than mutating the existing dict in place. This copy-on-write strategy is essential for concurrent isolation.

**Main execution path (non-concurrent):** Context persists in the caller's thread. Values set during one pipeline execution are visible to subsequent executions in the same thread context. This is a direct consequence of using `contextvars` — the `ContextVar` retains its value until explicitly changed or until the context is replaced.

**Concurrent workers (`foreach`/`gather` with `concurrency`):** Workers inherit a snapshot of the context via `copy_context().run()` (see §11.7). Each worker starts with a copy of the parent's context at the time the worker was dispatched. Values set by a worker do **not** propagate back to the parent or to sibling workers. This is guaranteed by the combination of `copy_context()` (which creates a shallow copy of the context) and copy-on-write dict semantics (which ensure that a worker's `set()` creates a new dict visible only within that worker's context copy).

**Async concurrent tasks:** Async tasks inherit context naturally through Python's `asyncio` task creation mechanism, which copies the current context into the new task. The same isolation guarantees apply — a task's `set()` does not affect the parent or siblings.

```python
Chain.set('config', load_config())    # pre-populate context
Chain([url1, url2, url3])             \
  .foreach(fetch_data, concurrency=3) \  # workers see 'config'
  .run()
```

### 15.5 Dual Dispatch via Descriptor

Both `Chain.set` and `Chain.get` are implemented as Python descriptors (`_SetDescriptor` and `_GetDescriptor` respectively). The descriptor protocol enables dual dispatch based on how the attribute is accessed:

- **Instance access** (`chain.set` / `chain.get`): The descriptor's `__get__` receives the instance and returns a function that appends a pipeline step to that chain. This is why `chain.set('key')` and `chain.get('key')` are builder methods that return the chain.
- **Class access** (`Chain.set` / `Chain.get`): The descriptor's `__get__` receives `None` as the instance and returns a function that operates on context immediately. `Chain.set('key', value)` stores a value at the call site; `Chain.get('key')` retrieves a value at the call site.

This mechanism allows a single attribute name to serve both the fluent builder pattern (instance) and the direct utility pattern (class) without ambiguity.

---

## 16. Design Decisions & Rationale

### 16.1 Single `except_`/`finally_` Per Chain

Each chain permits at most one `except_()` handler and one `finally_()` handler. Attempting to register a second raises `QuentException`.

**Rationale:**

- **Simplicity:** A single error boundary per chain makes the execution model easy to reason about. There is exactly one place where exceptions are caught and one place where cleanup runs.
- **Predictability:** Multiple exception handlers would introduce ambiguity about ordering, precedence, and which handler "wins." A single handler eliminates these questions.
- **Composition via nesting:** Per-step error handling is achieved by composing chains: wrap the step that needs its own error handling in a nested chain with its own `except_()`. This reuses the same mechanism rather than introducing a second one.

### 16.2 Unified Calling Convention

All contexts — standard steps, `except_()` handlers, `finally_()` handlers, and `if_()` predicates — use the same 2-rule calling convention. The only difference per context is what "current value" means:

| Context | Current value |
|---------|--------------|
| Standard steps (`then`, `do`, etc.) | Pipeline's current value (from previous step output) |
| `except_()` handler | `ChainExcInfo(exc, root_value)` |
| `finally_()` handler | The root value (normalized to `None` if absent) |
| `if_()` predicate | Pipeline's current value |

**Rationale:** A single unified convention eliminates the cognitive overhead of learning separate dispatch rules for different contexts. The exception handler receives `ChainExcInfo(exc, root_value)` as its current value — the same way any pipeline step receives its input. This simplification reduces the API surface, makes behavior more predictable, and eliminates special cases (kwargs-only distinction, tuple-packing for nested chains, etc.).

### 16.3 Gather Is Always Concurrent

`gather()` always executes its functions concurrently, even when called from a synchronous context. In sync mode, it uses a `ThreadPoolExecutor`; in async mode, it uses `asyncio.TaskGroup` (Python 3.11+) or `asyncio.gather` (Python 3.10).

**Rationale:** This eliminates bridge asymmetry. If sync gather ran sequentially while async gather ran concurrently, the two modes would produce different observable behavior (different execution order, different error semantics). By making both modes concurrent, the sync/async bridge is transparent: switching between `chain.run()` and `await chain.run()` does not change gather's behavior. Both modes also produce `ExceptionGroup` on multiple failures, maintaining consistent error handling regardless of execution mode.

### 16.4 The `Null` Sentinel

`Null` is a singleton sentinel distinct from `None`. It represents "no value was provided."

**Rationale:** `None` is a perfectly valid pipeline value. A user may legitimately write `Chain(None)` to create a chain whose root value is `None`. Without a distinct sentinel, the engine could not distinguish "the user provided `None` as a value" from "no value was provided." `Null` resolves this ambiguity:

- `Chain()` — no root value; the root value is `Null` (internal).
- `Chain(None)` — root value is `None`.

`Null` is never exposed to user code during normal execution. When the pipeline's current value is `Null` at the end of execution, it is normalized to `None` before being returned to the caller.

### 16.5 Control Flow Signals Are `BaseException` Subclasses

`return_()` and `break_()` raise internal signals that inherit from `BaseException`, not `Exception`.

**Rationale:** Control flow signals must not be caught by user `except Exception` clauses. A user's `except_()` handler catches exception types specified by the user (defaulting to `Exception`). If control flow signals were `Exception` subclasses, a chain's own `except_()` handler would intercept them, preventing `return_()` from exiting the chain and `break_()` from terminating iteration. Inheriting from `BaseException` ensures they bypass `except Exception` and propagate through the chain machinery to their intended handler. When a control flow signal is used inside an `except_()` or `finally_()` handler (where it cannot be meaningfully handled), it is caught and wrapped in a `QuentException` with a descriptive error message.

### 16.6 Pickling and Copying

Chains do not block pickling. In practice, most chain contents (lambdas, closures, bound methods, nested chains) will naturally fail to pickle, so explicit prevention was redundant hand-holding inconsistent with §16.12 ("Unopinionated by Design"). Users are responsible for their own serialization security.

Shallow and deep copying (`copy.copy`, `copy.deepcopy`) of a Chain are blocked with `TypeError`. A shallow or deep copy would produce a broken object with shared linked-list structure, leading to subtle corruption. `clone()` is the correct way to copy a chain — it performs a proper structural copy of the linked list.

**Rationale:** Pickle prevention was removed because it guarded against a general Python hazard (`pickle.loads()` on untrusted data) at the library level — the same rationale would demand blocking `eval`, `exec`, and every other code execution vector. quent exposes primitives; users are responsible for how they deploy them. Copy blocking remains because it prevents a correctness bug: `copy.copy(chain)` silently produces a broken chain, while `clone()` produces a correct one.

### 16.7 Three-Tier Execution for Iteration

Iteration operations (foreach, foreach_do) use a three-tier execution pattern:

1. **Sync fast path:** Pure synchronous execution. Iterates items and calls `fn(item)` synchronously. Uses `next()` in a `while True` loop rather than a `for` loop (a `for` loop would silently consume StopIteration from `fn()`, and would not allow detecting an awaitable result mid-iteration).

2. **Mid-operation async transition:** When the sync fast path discovers that `fn(item)` returned a coroutine partway through iteration, it hands off the *live iterator* and all partial results accumulated so far to an async continuation. The async continuation picks up exactly where the sync path left off — no items are re-processed.

3. **Full async path:** When the input is an async iterable from the start (has `__aiter__` but not `__iter__`), the entire operation runs asynchronously.

**Rationale:** This pattern ensures no work is repeated during the sync-to-async transition. If the first 50 items were processed synchronously and the 51st returns a coroutine, the async continuation starts at item 51 with the 50 results already collected. The alternative — restarting from the beginning in async mode — would be wasteful and could produce side effects from re-evaluating pure functions.

### 16.8 First-Write-Wins for Exception Metadata

Exception metadata (the failing step, runtime arguments) is recorded using first-write-wins semantics: only the innermost failing step is stored on the exception.

**Rationale:** When an exception propagates through nested chains, each chain's error handling boundary sees the exception. First-write-wins preserves the innermost (original) failure context — the step where the error actually originated. If later chains could overwrite the metadata, the traceback visualization would point to an intermediate step rather than the root cause, making debugging harder.

### 16.9 `ThreadPoolExecutor` Per Invocation

Sync concurrent operations (gather, concurrent foreach/foreach_do) create a new `ThreadPoolExecutor` for each invocation and shut it down immediately after the operation completes.

**Rationale:**

- **Deterministic cleanup:** The executor's threads are joined before the operation returns. No background threads linger after the chain completes.
- **No shared state:** A shared executor would introduce thread pool exhaustion risks, lifecycle management complexity, and subtle bugs from interactions between unrelated chain executions.
- **Simplicity:** The `with ThreadPoolExecutor(...) as executor:` pattern handles creation, submission, waiting, and shutdown in a single scope.

**Escape hatch:** When executor creation overhead is a concern for hot paths, users can provide their own `Executor` instance via the `executor` parameter on `foreach()`, `foreach_do()`, and `gather()`. In this case, quent uses the provided executor and does not shut it down — the user manages its lifecycle. See §11.2.2.

### 16.10 Dual-Protocol Objects Prefer Async

When a pipeline value supports both sync and async protocols — context managers (`__enter__`/`__exit__` vs `__aenter__`/`__aexit__`) or iterables (`__iter__` vs `__aiter__`) — and an async event loop is currently running, the async protocol is preferred. When no async event loop is running, the sync protocol is used.

Loop detection covers asyncio, trio, and curio without importing them — detection uses `sys.modules` lookups (~50ns dict get when a library is not loaded), so there is zero overhead when those libraries are absent.

**Rationale:** Objects like `aiohttp.ClientSession` implement both protocols but their sync protocol is a compatibility stub — the real resource management happens in the async protocol. When running inside an async event loop (which the async execution path implies), using the async protocol ensures correct behavior. This heuristic applies uniformly to both context managers and iterables, producing correct behavior without requiring the user to specify which protocol to use.

### 16.11 `if_()` Design: Pending Flag and Predicate Calling Convention

**Two-call API (`if_().then()`):** `if_()` sets a pending flag and returns `self`. The immediately following `.then()` or `.do()` is absorbed as the truthy branch rather than appended as a normal step. This design makes the truthy branch syntactically explicit and avoids keyword arguments (`then=`, `args=`) on `if_()` itself — which were easy to misread and did not compose naturally with the rest of the fluent API. Build-time validation (pending flag checks in `run()`, `else_()`, and `if_()` itself) catches usage mistakes early.

**Predicate calling convention:** `if_()` invokes predicates using the standard 2-rule calling convention. This means predicates support explicit args/kwargs (Rule 1) and the default passthrough (Rule 2). Nested Chain predicates follow the standard default rule (Rule 2) since a Chain is callable. The predicate's "current value" is the pipeline's current value.

**Literal predicate values:** When `predicate` is not callable and not `None`, its truthiness is used directly without calling it. This allows patterns like `.if_(feature_flag_enabled).then(extra_step)` where the flag is a plain boolean evaluated at build time.

**Rationale:**

- **Consistency:** Predicates use the same 2-rule calling convention as all other contexts. This eliminates special-case knowledge — users learn one set of rules that applies everywhere.
- **Nested Chain predicates:** When a predicate is a nested Chain, it follows the standard default rule (Rule 2) — the chain is called with the current pipeline value as its argument. `return_()` propagates to the outer chain (early exit from a predicate is valid). `break_()` raises `QuentException` — predicates are not iteration contexts, so `break_()` is nonsensical there.

### 16.12 Unopinionated by Design

quent is a pipeline builder, not a framework. It exposes the full power of the underlying primitives — threads, iterators, concurrency, context managers — without artificial caps, warnings, or safety nets.

Specifically, quent does **not**:

- Cap thread pool sizes or concurrency levels. `concurrency=-1` means unbounded — the user decides the limit.
- Warn on unbounded materialization (e.g., collecting an infinite iterator via `foreach`). The user controls what they iterate.
- Impose timeouts on steps, operations, or pipelines. Timeout policy is the caller's responsibility.
- Rate-limit concurrent operations. The user's executor or semaphore handles this.
- Guard against large `gather()` fan-outs. If the user submits 10,000 tasks, quent submits 10,000 tasks.

**Rationale:** quent builds pipelines; it does not manage resources, enforce limits, or second-guess the user. Adding guardrails would mean choosing defaults that are wrong for some use case — and would force every user to learn how to disable them. Instead, quent provides the escape hatches (custom executors via the `executor` parameter, bounded concurrency via the `concurrency` parameter) and trusts the user to configure them appropriately for their context.

---

## 17. Known Asymmetries

The following sections document known behavioral asymmetries in quent. Sections 17.1–17.2 cover sync/async asymmetries caused by fundamental language constraints. Section 17.3 covers an operational asymmetry between different pipeline operations.

### 17.1 Sync `iterate()` Raises `TypeError` on Coroutine Return

When using sync iteration (`for item in chain.iterate()`), if the chain itself returns a coroutine (because it contains async steps), a `TypeError` is raised:

> Cannot use sync iteration on an async chain; use 'async for' instead

Similarly, if the iteration callback `fn` returns a coroutine during sync iteration, a `TypeError` is raised:

> iterate() callback returned a coroutine. Use "async for" with `__aiter__` instead of "for" with `__iter__`.

**Why:** A synchronous generator cannot `await` a coroutine. There is no language mechanism to bridge this gap within a `__iter__`/`__next__` protocol. The user must switch to `async for` with `__aiter__`.

### 17.2 Concurrent Sync Workers Detecting Awaitable Results

When a concurrent operation (gather, concurrent foreach/foreach_do) runs in sync mode (determined by probing the first function/item), subsequent workers execute in `ThreadPoolExecutor` threads. If a later worker's function returns an awaitable (coroutine, Task, Future):

- The awaitable is closed (if it has a `.close()` method) to prevent resource leaks.
- A `TypeError` is raised after all workers complete, explaining that the first function was sync so `ThreadPoolExecutor` was used, and the callable must be consistently sync or async.

**Why:** The sync/async mode is determined once by probing the first function's result. All subsequent workers run in the same mode. If a later worker returns a coroutine from a thread pool thread, there is no event loop in that thread to await it. The error message directs the user to make their callables consistently sync or async.

### 17.3 `return_(value)` Semantics Differ Between Chain Execution and `iterate()`

- **In normal chain execution:** `return_(value)` replaces the chain's entire result. The value becomes what `run()` returns.
- **In `iterate()`:** `return_(value)` yields the value as one final item before stopping iteration. Previously yielded items are preserved — they have already been emitted to the caller.

**Why:** Same streaming constraint as `break_()` — the iterator cannot retract items already consumed by the caller. The return value is added as a final yield rather than replacing prior output.

---

## 18. Patterns

This section documents common pipeline compositions. Each pattern is a concise recipe — a representative code sketch with a one-sentence explanation of when to use it. All patterns follow the contracts defined in the preceding sections.

### 18.1 Fan-Out and Combine

Use `gather` to compute multiple independent results from a single value, then combine the tuple in the next step.

```python
Chain(url)                            \
  .then(fetch)                        \
  .gather(extract_title, extract_body, extract_meta) \
  .then(lambda t: {'title': t[0], 'body': t[1], 'meta': t[2]}) \
  .run()
```

### 18.2 Concurrent Transform and Aggregate

Use `foreach` with `concurrency` to process a collection in parallel, then aggregate the results.

```python
Chain(image_paths)           \
  .foreach(resize, concurrency=8) \
  .then(lambda results: sum(r.size for r in results)) \
  .run()
```

### 18.3 Conditional Branching

Use `if_`/`else_` with nested chains to select between multi-step branches based on the current value.

```python
Chain(request)                                      \
  .if_(is_authenticated)                            \
    .then(Chain().then(load_profile).then(render))  \
  .else_(Chain().then(redirect_to_login))           \
  .run()
```

### 18.4 Per-Step Error Handling

Use nested chains with their own `except_()` to recover from failures at individual steps without aborting the entire pipeline.

```python
safe_fetch = Chain().then(fetch).except_(lambda exc: default_response)

Chain(urls)              \
  .foreach(safe_fetch)   \
  .then(merge_responses)  \
  .run()
```

### 18.5 Observation Points

Use `do()` to insert logging or debugging taps that observe the current value without altering the pipeline flow.

```python
Chain(raw_data)           \
  .then(validate)          \
  .do(lambda v: log.info('validated: %s', v)) \
  .then(transform)         \
  .do(lambda v: log.info('transformed: %s', v)) \
  .run()
```

### 18.6 Function Decoration

Use `decorator()` to wrap an existing function so its return value flows through a fixed processing pipeline.

```python
@Chain().then(json.loads).then(normalize).then(validate).decorator()
def read_config(path):
  with open(path) as f:
    return f.read()

read_config('settings.json')  # returns validated, normalized dict
```

### 18.7 Context Manager Integration

Use `with_()` to acquire a resource, process it, and guarantee cleanup — all within the pipeline.

```python
Chain(db_url)            \
  .then(connect)          \
  .with_(lambda conn: conn.execute(query)) \
  .then(process_rows)     \
  .run()
```

Use `with_do()` when the resource itself (not the body's return value) should continue through the pipeline.

```python
Chain(db_url)             \
  .then(connect)           \
  .with_do(lambda conn: conn.execute('PRAGMA optimize')) \
  .then(lambda conn: conn.execute(query)) \
  .run()
```

### 18.8 Transparent Sync/Async Bridging

Build one pipeline, run it from sync or async code. Swapping any sync step for its async equivalent produces the same result — the bridge contract in action.

```python
# Define the pipeline once — works identically whether steps are sync or async.
pipeline = (
  Chain()
  .then(fetch_user)          # sync: requests.get(...)  OR  async: aiohttp.get(...)
  .then(parse_profile)       # pure sync — works in either mode
  .do(log_access)            # sync: print(...)  OR  async: await async_logger.info(...)
  .then(enrich_with_prefs)   # async: await db.query(...)
  .run
)

# Sync caller — all-sync steps return a plain value:
result = pipeline(user_id)

# Async caller — if any step is async, run() returns a coroutine:
result = await pipeline(user_id)
```

---

## 19. Public API

All public symbols are exported from the `quent` package via `__all__`. These are the only names users should import.

| Symbol | Description |
|--------|-------------|
| `Chain` | Fluent pipeline builder — the core primitive for constructing and executing sequential pipelines with transparent sync/async bridging. |
| `ChainExcInfo` | NamedTuple passed to `except_()` handlers as the current value. Fields: `exc` (the caught exception) and `root_value` (the chain's evaluated root value, normalized to `None` if absent). |
| `ChainIterator` | Dual sync/async iterator returned by `iterate()` and `iterate_do()`. Supports both `__iter__` (for `for` loops) and `__aiter__` (for `async for` loops). Callable to create new iterators with different run arguments. |
| `QuentException` | Base exception type for quent-specific runtime errors (escaped control flow signals, duplicate handler registration, invalid break context). |
| `__version__` | Package version string (PEP 440). Resolved from installed package metadata; falls back to `'0.0.0-dev'` when not installed. |
