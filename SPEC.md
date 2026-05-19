# quent — Behavioral Specification

**Version:** 7.0.0 | **Date:** 2026-05-19

> Source of truth. Implementation and tests derive from these contracts.

## Table of Contents

- [1. Intent](#1-intent)
- [2. The Bridge Contract](#2-the-bridge-contract)
- [3. Pipeline Model](#3-pipeline-model)
- [4. Calling Conventions](#4-calling-conventions)
- [5. Operations](#5-operations)
- [6. Error Handling](#6-error-handling)
- [7. Control Flow](#7-control-flow)
- [8. Execution](#8-execution)
- [9. Iteration](#9-iteration)
- [10. Reuse](#10-reuse)
- [11. Concurrency](#11-concurrency)
- [12. Null Sentinel](#12-null-sentinel)
- [13. Traceback Enhancement](#13-traceback-enhancement)
- [14. Instrumentation](#14-instrumentation)
- [15. Context API](#15-context-api)
- [16. Design Decisions](#16-design-decisions)
- [17. Known Asymmetries](#17-known-asymmetries)
- [18. Patterns](#18-patterns)
- [19. Public API](#19-public-api)

---

## 1. Intent

**quent** is a sync/async-transparent pipeline builder for Python 3.10+. Pure Python; zero runtime deps on 3.11+ (3.10 needs `typing_extensions` for stdlib backports).

**Problem.** Supporting both sync and async callers normally means writing the same pipeline logic twice — `def process_sync(...)` and `async def process_async(...)` — with duplicated structure, divergent bugs, and no shared mental model. quent eliminates that duplication.

*Build once; run sync or async.* Any callable at any position is interchangeable with its async equivalent — observable result is identical. Caller selects no mode, wraps no coroutines, writes no conditional `await`. `run()` returns a plain value if all-sync, a coroutine if any step transitioned. The **bridge contract** (§2) is the load-bearing invariant; every behavior upholds it or is documented as an exception (§17).

---

## 2. The Bridge Contract

### 2.1 Invariant

> For any pipeline `P` and step `i`, replacing step `i`'s callable with a functionally equivalent callable of the opposite sync/async kind produces the same observable result.

**Functionally equivalent:** for the same input, sync returns `V`; async returns a coroutine resolving to `V`. Holds for every operation, handler, predicate, branch, body. Exceptions: §17.

**Observable result** = `run()`'s return value; the type, value, and chain (`__cause__`/`__context__`) of any propagating exception; and the **sequence** of pipeline steps executed at the top level (linked-list order, sequential). Within a concurrent op (`gather`, `foreach`/`foreach_do` with concurrency), the relative ordering of sibling callables is **undefined** — they run concurrently and may interleave or execute in any order. Across the sync/async transition, Python's single-threaded semantics serialize side-effects of sequential steps. OS-level effects (file buffer flushing, network frame send timing, wall-clock between-step intervals) are not part of the bridge contract.

### 2.2 Two-Tier Engine

Begins sync, walking the list head-to-tail. After each step:

| Result | Action |
|---|---|
| Non-awaitable | Record as new CV, advance. Sync fast path — no event loop, no coroutine. |
| Awaitable | Transition: continuation receives pending awaitable + accumulated state (CV, root, position); runs rest async, awaiting inline. |

Consequences: (1) all-sync pipelines pay zero async overhead; (2) transition point is anywhere; once async, stays async; (3) `except_()`/`finally_()` returning awaitables also trigger transition (§6.2, §11.6).

Build-time mode inspection is unreliable (sync fn can return coroutine; nested-pipeline mode depends on contents; dual-protocol depends on ambient loop). Runtime detection only.

### 2.3 Transparency

One API surface — no `async_mode=`, no `AsyncQ`, no coroutine/future wrapping, no conditional `await`. Caller decides whether to `await` based on `run()`'s return type.

### 2.4 Awaitability

"Awaitable" ≡ `inspect.isawaitable(x)` — coroutine objects, `__await__` implementations, generator-based coroutines (`@asyncio.coroutine`). Checked after every step result.

`concurrent.futures.Future` is **not** awaitable (no `__await__`). `asyncio.Future` is awaitable but resolves only under an asyncio loop — quent does not adapt cross-runtime awaitables (a trio/curio task that returns an `asyncio.Future` will block when awaited under that loop, per Python's normal semantics).

### 2.5 Awaitable Detection (performance contract)

Awaitable detection runs after every step. To keep all-sync pipelines paying zero async overhead, the engine uses a tiered check rather than calling `inspect.isawaitable()` directly:

1. `type(result) is CoroutineType` — the common case (`async def fn(...)` returns), constant-time identity check.
2. Short-circuit reject if `result is None` or `type(result)` is in a frozenset of common sync types (`int`, `str`, `float`, `bool`, `list`, `dict`, `tuple`, `set`, `bytes`). The check is **exact-type identity** (`type(x) in frozenset`), not `isinstance` — subclasses such as `numpy.int64`, `pathlib.Path`, `IntEnum` members fall through to step 3.
3. Otherwise, a custom `_isawaitable(result)` that uses `isinstance(value, CoroutineType)` (handles subclasses), checks generator-based coroutines via the `CO_ITERABLE_COROUTINE` code flag, and falls back to `__await__` attribute probing.

This is roughly an order of magnitude faster than `inspect.isawaitable()` on common sync return types. The optimization is part of the bridge contract's "zero async overhead" claim (§2.2) — implementations that fall back to the naive `inspect.isawaitable()` call may functionally conform but break the performance promise.

---

## 3. Pipeline Model

### 3.1 Shape and Storage

Sequential computation threading a **current value (CV)**. Each step takes CV; result becomes new CV (side-effect steps preserve CV).

Append-only singly-linked list. Append O(1); walks head-to-tail; never mutated post-construction.

**Thread safety:** building not thread-safe. A constructed pipeline executes safely from multiple threads (incl. PEP 703 free-threaded) — execution uses only function-local state.

**Recursion / depth limits:** quent imposes no limit on pipeline length or nesting depth. Pipeline length is bounded only by available memory. Nested pipelines invoke the execution engine recursively at run time and are subject to Python's standard recursion limit (`sys.getrecursionlimit()`, default ~1000). Visualization rendering has its own independent limits (§13.8) that exist purely to bound traceback rendering — they do not constrain execution.

### 3.2 Root Value

`Q(v=<no value>, /, *args, **kwargs)`:

| Form | Behavior |
|---|---|
| `Q()` | No root. First step invoked per §4 with no value. |
| `Q(v)`, `v` callable | At `run()`: `v(*args, **kwargs)` (or `v()` if absent). Return → root. |
| `Q(v)`, `v` non-callable | `v` is root as-is. `args`/`kwargs` absent or build `TypeError`. |
| `Q(None)` | Root is `None`. |
| `Q(key=val)` (no positional) | Build `TypeError` — kwargs require root callable. |

**Root dispatch:** The root is the first link in the pipeline; it is evaluated through the standard calling-convention machinery of §4 with starting `CV = Null`. The forms above are the four ways the root link is *constructed* — once constructed, no special root semantics apply. Rule 1 fires when build-time `args`/`kwargs` are present, Rule 2 otherwise. The root appears in `on_step` events with `step_name='root'` (§14.1).

**Run-time root:** `q.run(v, ...)` replaces build-time root entirely. `Q(A).then(B).run(C)` ≡ `Q(C).then(B).run()`.

Root has two roles: initial CV; captured **root value** for handler dispatch — `finally_()` receives it; `except_()` receives `QuentExcInfo(exc, root_value)`. **Root callable failure:** standard error flow (§6.3); `except_` sees `QuentExcInfo(exc, None)`; `finally_` sees `None`.

### 3.3 Value Flow

```
Q(root)            root evaluated → CV = result
  .then(f)           CV = f(CV)
  .do(g)             g(CV); CV unchanged
  .then(h)           CV = h(CV)
  .run()             returns CV (None if none produced)
```

Internal "no value" sentinel `Null` (§12) is normalized to `None` at every user-visible boundary.

---

## 4. Calling Conventions

Canonical reference. Other sections refer by rule number.

### 4.1 The Two Rules

Strict priority, first match wins. Apply universally — steps, predicates, handlers, branches, bodies. Sole exception: `drive_gen()`'s step fn (§5.11).

| Priority | Rule | Trigger | Invocation |
|---|---|---|---|
| 1 | Explicit args | Args or kwargs provided | `fn(*args, **kwargs)` — CV NOT passed |
| 2 | Default | Neither | `fn(CV)`, `fn()`, or `v` as-is (below) |

**Rule 1 constraint:** `fn` callable required; non-callable + args/kwargs → build `TypeError`. Rationale: auto-prepending CV would force every fn to handle an extra leading parameter; use a lambda to combine.

**Rule 2 dispatch** — Rule 2 collapses three operationally distinct sub-cases under one "default" header. The table below enumerates them; the umbrella name "Rule 2" refers to the union.

| `fn` callable? | CV present? | Invocation | Sub-case |
|---|---|---|---|
| Yes | Yes | `fn(CV)` | 2a |
| Yes | No (Null) | `fn()` | 2b |
| No | — | `fn` itself becomes new CV (literal replacement) | 2c |

```python
Q(5).then(format_number, 'USD', decimals=2)  # Rule 1: format_number('USD', decimals=2); 5 NOT passed
Q(5).then(str)                                # Rule 2 callable + CV: str(5) → '5'
Q().then(get_timestamp)                       # Rule 2 callable, no CV: get_timestamp()
Q(5).then(42)                                 # Rule 2 non-callable: CV = 42
```

### 4.2 Nested Pipelines

`Q.__call__` ≡ `run()` at the public-API level. **Internally**, when a `Q` is registered as a step (e.g. `.then(inner)`), the engine invokes it via a nested dispatch path — not via `run()` — so that `Q.break_()` raised inside the nested `Q` can propagate outward through the nested boundary to reach an enclosing iteration scope. A nested `Q` dispatches per §4.1. Caller-provided args/kwargs **replace** the inner's build-time root args/kwargs entirely (no merging); inner root callable is preserved.

| Registration | Invocation |
|---|---|
| `.then(inner)` | nested dispatch with `CV` |
| `.then(inner, arg, key=val)` | nested dispatch with `arg, key=val` — CV NOT passed |

**Signal semantics in nested `Q`s** (each `Q` is a function-like boundary in the Python analogy):

| Signal | Behavior at the nested `Q` boundary |
|---|---|
| `Q.return_()` | **Absorbed.** The nested `Q`'s execution ends with the given value; that value flows to the outer pipeline as the nested step's result; the outer chain continues. (Like Python's `return` exits the current function.) |
| `Q.break_()` | **Propagates through.** The nested `Q` does not catch it; it continues outward toward the nearest enclosing iteration scope. (Like Python's labeled break — a function boundary is not a loop boundary.) |

**Lambda wrapping breaks `Q.break_()` propagation:** `.then(lambda cv: inner(cv))` invokes `inner.run(cv)` directly. `inner.run()` is the **outermost** `run()` from `inner`'s perspective, so a `Q.break_()` that escapes `inner` (without being caught by an iteration scope inside `inner`) is wrapped as `QuentException` at the lambda's call — it never reaches the outer pipeline's iteration scope. To preserve `Q.break_()` propagation across nesting, register the inner `Q` directly via `.then(inner)`. (`Q.return_()` is unaffected: each `run()` absorbs its own `Q.return_()` either way.)

### 4.3 CV By Context

| Context | "CV" means |
|---|---|
| `then`, `do`, `with_`, `with_do`, `foreach`, `foreach_do`, `gather`, `else_`, `else_do` | Pipeline CV (output of previous step) |
| `if_()` predicate | Pipeline CV |
| `while_()` predicate, body | Current loop value |
| `except_()` handler | `QuentExcInfo(exc, root_value)` |
| `finally_()` handler | Root value (normalized `None` if absent / root raised) |
| `drive_gen()` step fn | Yielded value — direct call, ignores §4.1 (§5.11) |

**Handler callability:** `except_()` and `finally_()` require callable — `TypeError` at registration (stricter than `then()`; non-callable handler is always a mistake). **Finally return:** always discarded.

---

## 5. Operations

Builder methods on `Q`, return `self`. Builders only record intent; execution happens at `run()`. Build- vs run-time enforcement: §16.3.

### 5.1 `then(v, /, *args, **kwargs)`

Append step whose result replaces CV. Per §4. Non-callable `v` replaces CV directly; args/kwargs absent or build `TypeError`. Nested `Q`: §4.2.

### 5.2 `do(fn, /, *args, **kwargs)`

Side-effect step. Per §4; return **discarded**; CV unchanged. Awaitable returns awaited before discard. `fn` callable (build `TypeError`) — non-callable would silently no-op.

### 5.3 `foreach(fn=None, /, *, concurrency=None, executor=None)`

Apply `fn` to each element of CV iterable; collect into list. `fn=None` is identity. Non-callable → build `TypeError`.

**Sequential** (`concurrency=None`): in iteration order; awaitable from `fn` → async transition. Sync (`__iter__`) and async (`__aiter__`) iterables both supported; dual-protocol prefers async under running loop (§16.2).

**Concurrent** (`concurrency=-1` unbounded or positive int): input is **eagerly materialized** before dispatch — sync iterables via `list(iterable)`, async iterables via full `async for` drain into a list. Not for infinite/very-large iterables. Sync vs async path probed on first element (§11.5). Mixed → `TypeError`. Results preserve **input order**. `-1` resolves to `len(items)`. Params: §11.

**Errors:**
- Sequential: propagates immediately. `StopIteration` from callback propagates as regular exception; PEP 479 wraps it as `RuntimeError` in async (§17.5).
- Concurrent: single → propagates directly. Multiple `Exception`s → `ExceptionGroup`. `BaseException` never wraps; earliest-input-index wins. Signal priority (§7.3): `return_` > `break_` > regular.

**`break_()`:** sequential — stops at break index; partial results + carried value appended. Concurrent — results truncated to elements before earliest-index break; carried value appended.

### 5.4 `foreach_do(fn, /, *, concurrency=None, executor=None)`

Same as `foreach()` except: `fn`'s returns discarded; **original input elements** collected in input order. Error/break behavior matches `foreach()`.

> `Q.break_(value)` appends `value` directly to the results list — even in `foreach_do` mode where the list otherwise contains original elements. The break value is not coerced or filtered; results may be heterogeneous (e.g., `Q([1,2,3]).foreach_do(lambda x: Q.break_('STOP') if x==2 else None).run()` yields `[1, 'STOP']`).

### 5.5 `gather(*fns, concurrency=-1, executor=None)`

Run multiple fns on CV concurrently. Each `fn` receives CV. Result is **tuple** in positional order.

- **Always concurrent** — `concurrency=None` rejected (sequential gather ≡ chained `then()`; always-concurrent eliminates bridge asymmetry). Tuple (not list) signals fixed structure: `len(result) == len(fns)`.
- Single-fn → `(result,)`. Zero fns → `QuentException`.
- Sync/async probed on first fn (§11.5). Mixed → `TypeError`.
- Each fn callable (build `TypeError`).

**Errors:** single → propagates directly. Multiple → `ExceptionGroup("gather() encountered N exceptions")`. `BaseException` never wraps (earliest-position wins). `Q.return_()` raised inside a `gather()` worker **returns from that worker** — the value becomes that gather position's tuple element (per §7.4; the worker is treated as its own scope, sibling workers continue). When a `_Return` signal is observed at the gather-level triage (e.g. a worker re-raised one from a deeper nested `Q.run()` whose signal had escaped its iteration scope), it wins with absolute priority over co-occurring exceptions; regular exceptions are discarded with a `RuntimeWarning`. `Q.break_()` inside any gather worker → `QuentException` (gather is concurrent fan-out, not iteration scope; see §7.4).

### 5.6 `with_(fn, /, *args, **kwargs)`

Enter CV as context manager; invoke `fn` per §4 with `__enter__`/`__aenter__` result; replace CV with `fn`'s return. CM exits on success or failure.

- `fn` callable required (build `TypeError`).
- CV must support `__enter__`/`__exit__` or `__aenter__`/`__aexit__` else `TypeError`. Dual-protocol prefers async under running loop (§16.2).
- Awaitable from `fn` → async transition.

> **Rule 1 drops the context value:** when `args`/`kwargs` are provided, Rule 1 fires and `fn` does NOT receive the `__enter__`/`__aenter__` result — it is silently dropped. To combine the context value with extra arguments, use a lambda: `.with_(lambda ctx: fn(ctx, extra))`. Same caveat applies to `with_do()`.

**Exception suppression:** `fn` raises + `__exit__` returns truthy → pipeline continues with CV `None` (body result unavailable). Falsy → propagates.

**Exit handler failure:** `__exit__` raising replaces body exception via `raise exit_exc from exc`. This sets `exit_exc.__cause__ = body_exc` (explicit) and, as a Python consequence of `raise from`, also sets `exit_exc.__suppress_context__ = True`. Default traceback formatters render the `__cause__` chain ("The above exception was the direct cause of..."). The body exception is also reachable as `__context__` on `exit_exc` (Python's implicit chaining set the moment exit_exc was raised inside the `except`), but suppressed for default display; access programmatically via `exit_exc.__context__`. Clearer than Python's native `with` where the equivalent is `__context__`-only without `__cause__`.

**Control flow signals:** `Q.return_()`/`Q.break_()` inside `fn` → `__exit__` called with no exception info (clean exit); signal propagates.

**Sync CM after async transition:** sync CM entered + `fn` returns awaitable → `__exit__()` called synchronously from async tier; blocking I/O blocks the loop. Prefer dual-protocol or async-only CMs.

### 5.7 `with_do(fn, /, *args, **kwargs)`

Like `with_()` but `fn`'s return discarded; **original CV** (CM object, not `__enter__` result) passes through. CM suppression returns original CV (not `None`).

**Iteration context:** when last step before an iteration terminal (§9.7), CM object passes through as iterable — it must support iteration.

### 5.8 `if_(predicate=None, /, *args, **kwargs)`

Set pending conditional flag. Next `.then()`/`.do()` becomes truthy branch.

**Predicate (positional-only):**
- `None` — CV truthiness.
- callable — per §4; return truthiness tested.
- nested `Q` — nested-Q dispatch (§4.2) with `CV`; result tested. `Q.return_()` inside the predicate's nested `Q` returns from that `Q` with the given value; the value is used as the predicate result (truthy/falsy test). `Q.break_()` inside the predicate **propagates outward** through `if_` toward the nearest enclosing iteration scope (per §7.2/§4.2). If `if_` has no enclosing iteration, the break escapes and is wrapped as `QuentException` at the outermost `run()`.
- other non-callable literal — its own truthiness; CV NOT examined. Args/kwargs with such a literal → build `TypeError` (Rule 1 requires callable).
- `*args, **kwargs` forwarded to predicate per §4.

**Constraints:** `if_()` while another pending → `QuentException`. Non-`.then()`/`.do()` while pending → `QuentException`. Only `.then()`/`.do()` consume the flag.

**Run-time:** predicate evaluated per §4; awaitable awaited. Truthy → branch per §4 (`.then()` replaces CV, `.do()` discards). Falsy + no `else_` → CV passes through. Falsy + `else_` → else branch. `predicate=None` + Null CV → falsy.

### 5.9 `else_(v, /, *args, **kwargs)` and `else_do(fn, /, *args, **kwargs)`

Alternative for preceding `if_()`. Evaluated when predicate was falsy.

- `else_(v, ...)` — like `then()`.
- `else_do(fn, ...)` — like `do()` (callable required).

**Constraints:** must immediately follow truthy branch (no intervening ops) or `QuentException`. `else_*` while `if_()` pending → `QuentException`. One per `if_()` (both forms count the same; second → `QuentException`).

### 5.10 `while_(predicate=None, /, *args, **kwargs)`

Set pending loop flag. Next `.then()`/`.do()` becomes body. **Pre-tested loop:** predicate evaluated before each iteration; if truthy, body runs and its result (or the unchanged loop value, for `.do()`) becomes the next iteration's loop value; if falsy, loop exits with the current loop value. The first iteration's predicate is evaluated against the pre-`while_` CV. Awaitable result from predicate or body triggers async transition; the remaining iterations run async.

**Predicate forms:**
- `while_()` or `while_(None)` (no predicate; `None` is the default and is treated identically): predicate is loop value's own truthiness. `Null` loop value → falsy → body never runs. `Q().while_().do(fn).run()` exits immediately.
- `while_(callable)`: invoked per §4. Rule 1 if `args`/`kwargs` provided; Rule 2 otherwise. Return value tested for truthiness.
- `while_(nested_q)`: nested-Q dispatch (§4.2). `Q.return_()` inside the predicate's nested `Q` returns from that `Q` with the value; the value is used as the predicate result. `Q.break_()` inside the predicate propagates outward through the nested boundary; `while_` catches it as a normal loop break (the `while_` is iteration scope; same semantics as a callable predicate raising `break_`).
- `while_(literal)`: non-callable, non-`None` literal — its own truthiness used (constant). Args/kwargs with such a literal → build `TypeError`.

**Constraints:** `while_` while `if_`/`while_` pending → `QuentException` (combine via nested pipeline). `if_` while `while_` pending → `QuentException`. Non-`.then()`/`.do()` while pending → `QuentException`. `else_*` after `while_().then()`/`.do()` → `QuentException`.

**Body modes:**
- `.then(fn)` — result feeds back as loop value next iteration; on termination, final loop value becomes new pipeline CV.
- `.do(fn)` — side-effect; loop value unchanged; on termination, CV passes through unchanged.

> With `.do()` + predicate testing loop value (incl. default `None`), loop value never changes → infinite loop unless `break_()`.

**`break_()`:** `Q.break_()` stops the loop; result is the **current loop value**. `Q.break_(value)` stops; `value` **replaces** the loop value and becomes the loop's result — **not** appended (this differs from `foreach`/`iterate*` semantics in §7.2, which append/yield). Valid from body or predicate, including a nested-Q predicate (per §7.2 break propagates outward through `Q` boundaries until iteration catches; `while_` catches). Signal priority within a single iteration: if the predicate raises a signal, the body for that iteration never runs.

**`return_()`:** per §7.1 — returns from the **current `Q`** (the `Q` that contains the `while_`), not just the loop. The loop is one step of that `Q`; `Q.return_()` ends the entire `Q`. If the `Q` is nested in an outer pipeline, only the current `Q` returns and the outer chain continues with the returned value. (If you want to exit just the loop, use `break_()`.)

**No iteration limit.** **Errors** propagate through pipeline error handling (§6). **Immediately-falsy predicate:** body never runs; pre-`while_` CV passes through. **Cloning:** deep-cloned (predicate and body links carry state). **Traceback:** `.while_(predicate_name)`.

### 5.11 `drive_gen(fn, /)`

Drive sync or async generator bidirectionally via send protocol. `fn` processes each yielded value; result sent back. Generator stops → last `fn` return becomes pipeline CV.

- `fn` callable (build `TypeError`).
- At execution: CV must be sync/async generator OR callable producing one (else `TypeError`).
- `fn` **always called as `fn(yielded_value)`** — ignores §4; no args/kwargs dispatch. Nested `Q` works because `Q.__call__(yielded) == Q.run(yielded)`.

**Loop:** (1) CV callable → invoke to get generator; (2) first yield: sync `next(gen)` / async `await gen.__anext__()`; (3) `result = fn(yielded)`, await if awaitable; (4) send: sync `gen.send(result)` / async `await gen.asend(result)`; Stop → exit, pipeline CV = `result`; (5) **cleanup always:** sync `gen.close()` / async `await gen.aclose()` — on normal termination, exceptions, signals.

**Return:** last `fn` return → CV. No yields (immediate Stop) → CV `None`; `fn` never called. Generator's own `StopIteration.value` does NOT replace CV.

**Mode:**

| Generator | `fn` returns | Mode |
|---|---|---|
| Sync | Plain | Fully sync |
| Sync | Awaitable | Mid-transition — sync `gen.send()`; `fn` results awaited |
| Async | Plain | Fully async |
| Async | Awaitable | Fully async |

Mid-transition (sync gen + async `fn`) is the primary motivating use case — e.g. httpx's auth flow.

> **Mid-transition blocking:** in mid-transition mode the sync `next(gen)`/`gen.send()` calls run **on the event loop's thread**, not on a worker. A blocking generator step (long-running CPU work, blocking I/O, blocking `time.sleep`) blocks the loop until it yields. This is identical to running any synchronous code inside an `async def`. Use sync generators that are themselves quick between yields, or move to a fully-async generator.

**Errors:** `fn` or send exception (not Stop) propagates; generator closed in cleanup. NOT injected (no `gen.throw()`). Generator ignoring `GeneratorExit` on close → Python `RuntimeError`; propagates from cleanup.

**Control flow signals (per §7.4 carve-out):**
- `Q.return_(value)` raised inside `fn` **returns from `fn`** — `value` becomes the pipeline's CV (drive_gen's normal "last `fn` return → CV" rule applies). Generator is closed in cleanup. The signal is **not** propagated to the outer pipeline.
- `Q.break_()` raised inside `fn` propagates outward through drive_gen toward the nearest enclosing iteration scope (per §7.2). Generator is closed first via `close()`/`aclose()` in the cleanup `finally:`, then the signal propagates.

**Cloning:** reference-copied (no mutable state). **Traceback:** `.drive_gen(fn_name)`.

### 5.12 `name(label, /)`

Assign string label for traceback identification. No execution effect.

Visualizations: `Q[label](root)`. Exception notes (3.11+) include name. `repr(q)` includes. `clone()`/`as_decorator()` preserve.

---

## 6. Error Handling

At most one `except_()` and one `finally_()` per pipeline. Second → build `QuentException`. For per-step handling, compose nested pipelines (each gets its own pair). Both follow §4; CV per §4.3.

### 6.1 `except_(fn, /, *args, exceptions=None, reraise=False, **kwargs)`

**Registration:**
- `fn` callable (`TypeError`).
- `exceptions` — type, iterable of types, or `None` (default `Exception`):
  - Empty iterable → `QuentException`.
  - Non-`BaseException` subclass → `TypeError`.
  - String value → `TypeError` (common mistake: `"ValueError"` vs `ValueError`).
  - `BaseException` subtype not `Exception` (e.g. `KeyboardInterrupt`, `SystemExit`) → `RuntimeWarning` (catching system signals can suppress critical shutdown).

**Filter enforcement:** The `exceptions` filter is engine-side, not handler-side. If a raised exception is not an instance of `exceptions`, the handler is **bypassed entirely** — it never sees out-of-filter exceptions, and the exception propagates as if no `except_()` were registered. `finally_()` still runs in failure-context.

**Consumption vs re-raise:**

| `reraise` | Effect |
|---|---|
| `False` (default) | Handler's return → pipeline result; exception consumed; finally runs in success context. |
| `True` | Handler runs for side-effects only; after, original is re-raised; handler's return ignored. |

**Handler failure with `reraise=True`:**
- `Exception` subclass → handler exception **discarded**; `RuntimeWarning` emitted; note attached (3.11+); `__context__`/`__suppress_context__` on original **restored** to pre-handler values (prevents handler from permanently mutating chain); original re-raised.
- `BaseException` subclass → propagates naturally; system signals never suppressed.

> **Restoration mechanism:** `original.__context__` and `original.__suppress_context__` are snapshotted **before** the handler is invoked. After a discardable `Exception` handler-failure, the snapshots are written back to `original` before re-raise. This protects against the handler intentionally mutating the original's chain (e.g., `original.__context__ = elsewhere`); Python's automatic `handler_exc.__context__ = original` chaining never touches `original`'s own attributes.

**Handler failure with `reraise=False`:** handler exception propagates; original set as `__cause__` (`raise handler_exc from original_exc`).

**Control flow:** `Q.return_()`/`Q.break_()` inside except → `QuentException`.

### 6.2 `finally_(fn, /, *args, **kwargs)`

- `fn` callable (`TypeError`). CV per §4.3: root value (normalized `None`).
- **Always runs** — success path, exception path, **and** control-flow signal (`Q.return_()`/`Q.break_()`) propagation. `finally_()` executes **after** the lazy callable value of `Q.return_(fn, ...)` is evaluated (see §7.1).
- **Return always discarded** — cannot alter result.

**Failure:**
- Finally raises while exception active → finally's exception **replaces** original (Python `try/finally`); original preserved as `__context__`; note attached.
- Finally raises on success → finally's exception propagates as pipeline error.
- Both `except_` and `finally_` raise → finally wins; except's exception preserved as `__context__`.
- Finally raises while a control-flow signal (`Q.return_()`/`Q.break_()`) is propagating → finally's exception wins (Python `try/finally` semantics); the signal is preserved as `__context__` on finally's exception; finally's exception propagates as a regular pipeline error (NOT as a signal). The lazy callable of `Q.return_(fn)` / `Q.break_(fn)` has already been evaluated before finally runs (§7.1, §7.2), so its result is lost.

**Control flow:** `Q.return_()`/`Q.break_()` inside finally → `QuentException`.

**Async finally in sync pipeline:** coroutine return triggers **async transition** — `run()` returns coroutine; when awaited, finally awaited first, then result returned (success) or active exception re-raised (failure). Nothing discarded. Also applies when `Q.return_()` is the active "result" — the sync pipeline transitions async, the awaited coroutine evaluates the (lazy) return value, awaits the async finally, and resolves to the return value. For `iterate()`/`iterate_do()` during sync `for`: coroutine-returning finally → `TypeError` (sync generators cannot await; use `async for`). See §11.6 for the same async-transition mechanism applied to `except_(reraise=True)`.

### 6.3 Execution Order

1. Steps execute sequentially.
2. Step raises matching `exceptions`:
   - `reraise=False` → except runs; return → result; finally success-context.
   - `reraise=True` → except runs; original re-raised; finally failure-context.
3. Step raises non-matching → propagates; finally failure-context.
4. Success → finally success-context.
5. Finally **always last**.

### 6.4 ExceptionGroup

Concurrent ops wrap multiple failures. 3.11+: builtin. 3.10: polyfill — `ExceptionGroup(message, exceptions)` (non-empty list of `Exception`), `.exceptions`, `.subgroup(condition)`, `.split(condition)` → `(matching, rest)`, `.derive(excs)` — constructs a new group from `excs` (a non-empty sequence of `Exception` instances) and copies the original group's `__traceback__`, `__cause__`, `__context__`, and `__notes__` (if present) onto the new group; `subgroup`/`split` use `derive` internally so chain attributes and notes survive filtering. `__suppress_context__` is not copied (matches builtin `ExceptionGroup.derive` behavior). Single failure → not wrapped. Iteration messages: `"foreach()/foreach_do() encountered N exceptions"`.

---

## 7. Control Flow

`Q.return_()` and `Q.break_()` are classmethods raising internal `BaseException` subclasses — bypass user `except Exception`, including pipeline's own `except_()` (default `Exception`); see §16.4. User does not catch them directly.

quent's control-flow model mirrors Python's:

- **`Q.return_()` ≈ Python `return`** — exits the **current `Q`** (the innermost `Q` whose pipeline contains the call). Like Python's `return` exits the current function, `Q.return_()` exits the current `Q` with a value. If you're inside a nested `Q`, only that nested `Q` returns — its value flows to the outer pipeline.
- **`Q.break_()` ≈ Python labeled-`break`-to-nearest-loop** — propagates outward through `Q` boundaries until it reaches the **nearest enclosing iteration scope**, which catches it and exits with the break value. If no enclosing iteration scope exists when the signal escapes the outermost `run()`, it's wrapped as `QuentException`.
- **`Q.exit_()` ≈ Python `sys.exit()`** — propagates through **all** nesting levels and all carve-outs (except_/finally_/gather/drive_gen); absorbed only at the outermost `run()`. Use to terminate the entire pipeline from arbitrary depth.

The model is intentionally simple — there are no quent-specific control-flow rules to learn beyond the Python analogy and a small set of intentional traps (§7.4).

Values are **lazy** — callable values invoked only when caught.

Idiomatic: `return Q.return_(value)` / `return Q.break_(value)` — satisfies type checkers, avoids unreachable-code warnings (the mechanism does not require `return`).

### 7.1 `Q.return_(v=<no value>, /, *args, **kwargs)`

Signal early return from the **current `Q`**.

The current `Q` is the innermost `Q` whose pipeline contains the call. When `Q.return_()` is caught at the current `Q`'s boundary, that `Q`'s execution ends with the given value. If the current `Q` is a nested step in an outer `Q`, the outer pipeline continues with that value as the nested step's result. If the current `Q` is the outermost `run()`, that value becomes `run()`'s return.

| Form | Result of the current Q |
|---|---|
| `Q.return_()` | `None` |
| `Q.return_(42)` | `42` (non-callable as-is) |
| `Q.return_(fn)` | `fn()` lazily |
| `Q.return_(fn, *args, **kwargs)` | `fn(*args, **kwargs)` lazily |

**Lazy evaluation:**
- `fn(*args, **kwargs)` runs at the catch frame for the `_Return` signal — at the boundary of the current `Q`, **before** that `Q`'s registered `finally_()` runs (the finally sees the lazy fn's result, or its exception, as the active state).
- If `fn` raises, the raised exception's `__context__` is the `_Return` signal (Python's automatic `except` chaining). The `_Return` instance's internal `value`/`args`/`kwargs` attributes are private quent state — users must not inspect them; they are cleared eagerly during unwind. `finally_()` still runs.
- If `fn` returns an awaitable in a sync pipeline, the current `Q`'s execution returns a coroutine resolving to `fn`'s value (async transition; §11.6).
- If `fn` raises a control-flow signal (`Q.return_`/`Q.break_`), it is wrapped in `QuentException` — signals inside lazy values are misuse.

**Nested `Q`:** each `Q` boundary absorbs its own `Q.return_()`. See §4.2 for the propagation table.

**Restrictions:** see §7.4.

### 7.2 `Q.break_(v=<no value>, /, *args, **kwargs)`

Signal early termination of the **nearest enclosing iteration scope**.

`Q.break_()` propagates outward through `Q` boundaries (in contrast to `Q.return_()`, which is absorbed at each `Q` boundary). It continues outward until it is caught by the nearest enclosing iteration scope. Iteration scopes are: `foreach`, `foreach_do`, `iterate`, `iterate_do`, `flat_iterate`, `flat_iterate_do`, `while_`.

| Form | Effect at the catching iteration |
|---|---|
| `Q.break_()` | Stops iteration. `foreach`/`foreach_do`: returns results collected so far. `iterate*`: generator completes without further yields. `while_`: result is the current loop value. |
| `Q.break_(value)` | As above, **plus**: `foreach`/`foreach_do` — `value` appended to results; `iterate*` — `value` yielded as one final item before stopping; `while_` — `value` **replaces** the loop value (not append/yield). |
| `Q.break_(fn, ...)` | `fn(...)` lazy; resulting value handled per the row above. Awaitable result → async transition (§11.6); awaited before append/yield/replace. |

**Where `Q.break_()` propagates freely (not trapped):**
- Through `if_()`/`else_*` predicates and branches (if_ has no iteration scope; signal propagates to outer iteration).
- Through `with_`/`with_do` (CM exits cleanly first via `__exit__(None, None, None)`, then signal propagates).
- Through `drive_gen` (generator is closed first via `close()`/`aclose()`, then signal propagates).
- Through nested `Q` boundaries (a `Q` is not itself an iteration scope; `Q.break_()` is not absorbed by `Q.run()` unless that `Q.run()` is the outermost — see §7.4).

**Restrictions:** see §7.4.

**Wrapping form when escaping outermost `run()`:** the `QuentException` raised at the outermost boundary sets `__suppress_context__ = True` and leaves `__cause__ = None`. The internal `_Break` signal is intentionally hidden from default traceback rendering — it is an implementation type, not user data.

### 7.3 Priority in Concurrent Iteration

1. **`Q.return_()`** — absolute, immediate propagation. Wins over `BaseException`, `Q.break_()`, and regular exceptions regardless of input index.
2. **`Q.break_()`** — over regular exceptions; multiple → earliest **input index** wins. (Note: `Q.break_()` inside `gather()` → `QuentException` — gather is not an iteration scope; see §7.4.)
3. **`BaseException`** (e.g. `KeyboardInterrupt`, `SystemExit`) — never wraps in `ExceptionGroup`; earliest input-index over regular exceptions; co-occurring regular exceptions discarded.
4. **Regular exceptions** — single propagates; multiple → `ExceptionGroup` (§6.4) with op-specific message.

**Discard logging:** when `Q.return_()` wins, co-occurring regular exceptions are logged via `RuntimeWarning` on the `'quent'` logger as discarded; co-occurring `BaseException` is silently dropped (no warning). Other discard paths (e.g., `BaseException` winning over regulars) do not warn.

### 7.4 Restrictions and Carve-Outs

The following are the **only** places where the §7.1 / §7.2 propagation model is overridden. **`Q.exit_()` (§7.5) bypasses every entry in this table** and propagates regardless.

| Scope | `Q.return_()` | `Q.break_()` | `Q.exit_()` | Rationale |
|---|---|---|---|---|
| Inside `except_` handler | `QuentException` | `QuentException` | Propagates | (`exit_`) terminates pipeline regardless; (`return_`/`break_`) would skip handler invariants. |
| Inside `finally_` handler | `QuentException` | `QuentException` | Propagates | (`exit_`) terminates pipeline regardless; (`return_`/`break_`) would skip finally's "always runs" guarantee. |
| Inside `gather()` worker callable | **Returns from worker** — value becomes that gather position's tuple element. | `QuentException` (gather is concurrent fan-out, not iteration). | Propagates (sibling tasks cancelled per asyncio/threadpool semantics). | Workers are independent for `return_`; `exit_` is unconditional. |
| Inside `drive_gen`'s `fn` | **Returns from `fn`** — value becomes pipeline CV (drive_gen's normal "last fn return → CV" rule). | Propagates outward to the nearest iteration scope (drive_gen does NOT trap; generator closed first). | Propagates (generator closed first via `gen.close()`/`aclose()`). | drive_gen's `fn` is treated as a step-result producer. |
| Escaping outermost `run()` with no enclosing iteration scope (for `Q.break_()`) | n/a (each `Q` absorbs) | `QuentException` | n/a (absorbed at outermost `run()`) | No iteration target to break. |

Anywhere else — including `if_`/`else_*` predicates and branches, `with_`/`with_do` bodies, `then`/`do` callables, `while_` predicate and body, iteration callbacks, nested `Q` steps — signals follow §7.1 / §7.2 / §7.5 without trap.

### 7.5 `Q.exit_(v=<no value>, /, *args, **kwargs)`

Signal hard exit from the **entire** pipeline regardless of nesting depth.

Like Python's `sys.exit()`: propagates through every `Q` boundary, every signal carve-out (§7.4), every level of nesting. Absorbed only at the outermost `run()`, which produces the value as the entire pipeline's result.

| Form | Pipeline result |
|---|---|
| `Q.exit_()` | `None` |
| `Q.exit_(42)` | `42` (non-callable as-is) |
| `Q.exit_(fn)` | `fn()` lazily, evaluated at the outermost `run()`'s catch frame |
| `Q.exit_(fn, *args, **kwargs)` | `fn(*args, **kwargs)` lazily |

**Cleanup respects Python `try/finally` semantics:** as `_Exit` propagates, every `finally_()` runs, every CM's `__exit__` runs, every `drive_gen` generator closes, every concurrent task cancels and awaits. Resources release. Only after all cleanup unwinds does `_Exit` reach the outermost `run()`.

**Lazy evaluation:** same model as §7.1. `fn` runs at the outermost `run()` catch frame, **before** that pipeline's outermost `finally_()` (if any) would observe a successful completion — though by the time `fn` runs, all `finally_()` handlers at every nesting level have already executed (they ran during propagation). If `fn` raises a control-flow signal, it is wrapped in `QuentException` (signals inside lazy values are misuse).

**Use when:**
- Deep nested logic needs to terminate the entire pipeline (e.g., a fatal condition discovered five levels down).
- An invariant violation should abort everything immediately.

**Don't use when:**
- Returning from the current `Q` suffices — use `Q.return_()`.
- Exiting the nearest iteration suffices — use `Q.break_()`.

### 7.3 Priority in Concurrent Iteration

1. **`Q.return_()`** — absolute, immediate propagation. Wins over `BaseException`, `Q.break_()`, and regular exceptions regardless of input index.
2. **`Q.break_()`** — over regular exceptions; multiple → earliest **input index** wins. (Note: `Q.break_()` inside `gather()` → `QuentException` — gather is not an iteration scope.)
3. **`BaseException`** (e.g. `KeyboardInterrupt`, `SystemExit`) — never wraps in `ExceptionGroup`; earliest input-index over regular exceptions; co-occurring regular exceptions discarded.
4. **Regular exceptions** — single propagates; multiple → `ExceptionGroup` (§6.4) with op-specific message.

**Discard logging:** when `Q.return_()` wins, co-occurring regular exceptions are logged via `RuntimeWarning` on the `'quent'` logger as discarded; co-occurring `BaseException` is silently dropped (no warning). Other discard paths (e.g., `BaseException` winning over regulars) do not warn.

---

## 8. Execution

### 8.1 `run(v=<no value>, /, *args, **kwargs)`

Execute pipeline; return final value.

- `v` — optional run value; replaces build-time root (§3.2).
- `*args, **kwargs` — for `v` when callable.

**Errors:**
- `v` non-callable + args/kwargs → `TypeError`.
- `v` absent + args/kwargs → `TypeError`.
- Pending `if_`/`while_` → `QuentException`.
- Control flow signal escaping outermost → `QuentException` (user bug).

**Return:** plain value if all-sync; coroutine if any step or handler returned awaitable. Caller must `await`.

> Unawaited coroutine → `finally_()` will NOT execute, resources may leak, Python emits "coroutine was never awaited" warning.

**Root capture:** evaluated root (or run value) captured for handler dispatch. Root callable failure → standard error flow (§6.3).

### 8.2 `__call__`

Alias for `run()`. Enables pipelines as callables wherever a function is expected.

### 8.3 `__bool__`

Always `True`. Empty pipeline still truthy. Prevents surprises in `if`/`or`/`and`.

---

## 9. Iteration

### 9.1 `iterate(fn=None)`

Returns `QuentIterator` supporting `__iter__` (`for`) and `__aiter__` (`async for`).

- `fn` — optional callable transforming each element. `None` → as-is.
- Pipeline executes when **iteration begins**, not at `iterate()` call.
- Pipeline result must be iterable.

**Sync (`for`):** requires non-awaitable pipeline result + `fn` returns. Awaitable from either → `TypeError` directing to `async for`.

**Async (`async for`):** awaits awaitables. Sync iterable consumed via `async for` is auto-wrapped.

**Pipeline-level error handling:**
- `except_()` covers the **run phase** producing the iterable — exception during run is caught before reaching iteration layer.
- Exceptions from callback `fn` are NOT covered — propagate directly at the iteration point.

**Deferred `finally_()`:** runs in generator's `finally:` — **after** iteration ends, not after run phase. Resources from run remain alive throughout iteration. Runs on all exits: normal exhaustion, `.close()`, `break`, `return`, `fn` errors, pipeline run errors. For `run()` (no iteration terminal), not deferred.

**Empty pipeline:** `Q(iterable).iterate()` yields elements directly.

### 9.2 `iterate_do(fn=None)`

Like `iterate()` but `fn` is side-effect; return discarded; **original element** yielded.

### 9.3 Iterator Reuse via Calling

Returned `QuentIterator` is callable with `run()`'s signature. Each call returns a **fresh iterator** with those args as run params. Original configuration (`fn`, ignore_result, buffer size) preserved.

```python
it = q.iterate(fn)
for item in it: ...        # run with no args
for item in it(value): ... # run with `value` as run value
```

### 9.4 Control Flow in Iteration

- **`Q.return_(v)`:** yields `v` as one final item (if provided), then stops. Value per §7.1. **Differs from `run()`** (§17.3): previously-yielded items cannot be retracted.
- **`Q.break_(v)`:** stops; value yielded before stop if provided (§7.2).

### 9.5 `flat_iterate(fn=None, *, flush=None)`

Flatmap iterator. Each source element either iterates directly (`fn=None`, flattening one level) or is transformed by `fn` into a sub-iterable whose items are individually yielded.

- `fn` — callable returning iterable per source element. Items yielded individually.
- `flush` — optional zero-arg callable invoked once after source exhaustion. Returns iterable; items yielded into stream. Use: flushing buffered/remaining items.

All other behavior matches `iterate()`. **Errors:** `fn`/`flush` exceptions propagate at iteration point. Awaitable from `fn`/`flush` during sync iter → `TypeError`.

### 9.6 `flat_iterate_do(fn=None, *, flush=None)`

Like `flat_iterate()` but `fn`'s iterable is fully consumed (driving side-effects) and NOT yielded. Original source element is yielded. `flush` output yielded normally — "do" discard applies only to `fn`'s results. `fn=None` ≡ `flat_iterate(fn=None)`.

### 9.7 Deferred `with_` in Iteration

When `with_(fn)` or `with_do(fn)` is the **last** step before an iteration terminal, CM entry is **deferred** to iteration time. Without deferral the CM would exit before iteration begins — resource closed before items consumed.

**Lifecycle:** (1) pipeline runs; produces CM; (2) iteration start: `__enter__()`/`__aenter__()`; (3) `with_(fn)`: `fn(context_value)` per §4; result → iterable; (4) `with_do(fn)`: `fn` side-effect (discarded); CM object → iterable (must be iterable); (5) iteration with CM open; (6) `__exit__()` in generator's `finally:`.

**Exit args:**

| Exit cause | `__exit__` args |
|---|---|
| Normal completion / source exhausted | `(None, None, None)` |
| `break`, `Q.return_()`, `Q.break_()` | `(None, None, None)` — signals not errors |
| Generator `.close()` / `GeneratorExit` | `(None, None, None)` |
| Exception during iteration | `(*sys.exc_info())` |

Truthy from `__exit__` on exception → suppressed; generator stops cleanly. Falsy → propagates. `__exit__` raising → new exception replaces original.

**Ordering with deferred `finally_()`:** CM exits **first**, then deferred finally. Enforced by nesting: `try { cm.__exit__() } finally { deferred_finally() }`. Finally runs even if `__exit__` raised.

**Protocol:** dual-protocol prefers async under running loop (§16.2). Exit protocol matches entry. Sync iter + async-only CM/fn → `TypeError`.

### 9.8 `buffer(n)`

Backpressure-aware bounded buffer between producer (pipeline iterable) and consumer (iteration). Producer runs ahead up to `n` items.

- `n` positive integer (excluding `bool`). `0`, `-1`, non-integer → `ValueError`/`TypeError`.
- **Pipeline-level modifier**, not a step — no link added. Requires iteration terminal; `run()` with buffer set → `QuentException`.
- `buffer()` while `if_`/`while_` pending → `QuentException`.

**Behavior:** FIFO; full blocks producer (backpressure); empty blocks consumer.
- Sync: producer in background daemon thread + `queue.Queue(maxsize=n)`.
- Async: producer as `asyncio.Task` + `asyncio.Queue(maxsize=n)`.
- Wraps iterable **after** run phase and deferred `with_` entry, but **before** the `fn` callback.

**Errors:** producer exception propagates to consumer at next `get()`. Early consumer exit (`break`, `GeneratorExit`): sync → `threading.Event` set, producer checks during `put()`; async → producer task cancelled. Cleanup guaranteed via `finally:` on both ends.

**Interactions:** `clone()` preserves buffer size. Iterator reuse preserves. With deferred `with_`, buffer wraps iterable after CM entry and inner `fn`'s iterable.

---

## 10. Reuse

### 10.1 `clone()`

Independent copy. Subclass-safe via `type(self).__new__`.

**Deep-copied:** pipeline structure (linked list); nested `Q` within steps (recursively via own `clone()` — prevents cross-clone state sharing); `if_`/`else_` ops (mutable else-branch ref); `while_` ops (predicate + body links); `except_`/`finally_` step nodes (handler `Q` recursively cloned); kwargs dicts (shallow); pipeline name.

**Shared by reference:** non-`Q` callables (functions, lambdas, bound methods); positional args tuples (immutable); kwargs values; exception type tuples.

**State reset:** clones behave as top-level by default, regardless of original's nested usage. When used as a step, adopts nested behavior at that point.

### 10.2 `as_decorator()`

Wrap pipeline as function decorator. Decorated function's return → pipeline input.

- Pipeline cloned internally on `as_decorator()` — decorator does not share mutable state with original.
- Decorated wrapper forwards args to original fn; fn's return is run value for the cloned pipeline.
- Pipeline always executes as top-level — thread-safe concurrent calls.
- Wrapper preserves signature via `functools.wraps`.

**Async decorated:** `async def` decorated fn returns coroutine when called. Pipeline treats coroutine as root; standard awaitable detection triggers async transition. Wrapper is always sync (returns plain value or coroutine); caller `await`s on transition.

**Errors:** pending `if_`/`while_` → `QuentException`. Escaped signals → `QuentException`.

### 10.3 `from_steps(*steps)` (classmethod)

Construct pipeline by appending each step via `.then()`.

- Single `list`/`tuple` unpacked: `Q.from_steps([a, b, c])` ≡ `Q.from_steps(a, b, c)`. Literal tuple step: wrap (`Q.from_steps(Q().then((1,2,3)))`).
- Returns new `Q` with no root.
- Each step via `.then()` (standard §4).
- Zero args → empty pipeline ≡ `Q()`. `Q.from_steps().run()` → `None`.

`Q.from_steps(a, b, c)` ≡ `Q().then(a).then(b).then(c)`.

### 10.4 Pickling and Copying

- `copy.copy(q)`/`copy.deepcopy(q)` → `TypeError`. Would silently produce broken pipeline (shared linked-list). Use `clone()`. Enforced by `_UncopyableMixin`.
- Pickling NOT blocked. Most contents (lambdas, closures, methods, nested pipelines) naturally fail to pickle; explicit prevention was redundant hand-holding inconsistent with §16.1. Users responsible for serialization security.

### 10.5 Subclassing

- `clone()` and `as_decorator()` subclass-safe (via `type(self).__new__`).
- `on_step` subclass overrides respected (engine reads via `type(q).on_step`).
- No other subclassing guarantees. Internal implementation details not part of the contract.

### 10.6 Re-entrancy

A step calling `q.run()` on the same instance works. Execution uses only function-local state (§3.1); re-entrant calls each get their own context. No deadlock or shared-state corruption.

---

## 11. Concurrency

Serves iteration (`foreach`, `foreach_do`) and `gather`. Priorities: deterministic cleanup (new executor/task group per invocation, shut down immediately); sync/async transparency (probe-based detection); context propagation (`contextvars` to thread workers).

### 11.1 `concurrency` Parameter

On `foreach()`, `foreach_do()`, `gather()`.

| Value | Meaning |
|---|---|
| `None` | Sequential. Only `foreach`/`foreach_do`; `gather` rejects. |
| `-1` | Unbounded; resolves to `len(items)`/`len(fns)` at runtime. Default for `gather`. |
| positive int | Bounded — limit on simultaneous executions. |

**Bounds:** integer (`-1` or positive). `bool`/non-integer → `TypeError`. `0` or below `-1` → `ValueError("{method}() concurrency must be -1 (unbounded) or a positive integer, got {value}")`. Named pipeline appends `' (in pipeline \'{name}\')'`. `TypeError` for non-integer follows same suffix.

### 11.2 `executor` Parameter

| Value | Behavior |
|---|---|
| `None` (default) | New `ThreadPoolExecutor` per invocation; shut down (`wait=True`) after. |
| `Executor` instance | Used for sync concurrent; quent does NOT shut down — caller's responsibility. |
| anything else | `TypeError`. |

**Scope:** sync concurrent path only. Async path (semaphore + TaskGroup/`asyncio.gather`) unaffected.

**By op:** `foreach`/`foreach_do` use `executor` only when `concurrency` is set (sequential ignores). `gather` always uses on sync path.

**Context propagation:** worker submissions use `copy_context().run()` regardless of executor source.

### 11.3 Sync Concurrent Execution

Selected when first item/fn returns non-awaitable.

- `ThreadPoolExecutor` with `max_workers = min(concurrency, remaining_items)`. Shut down `wait=True` (unless user-provided).
- **First item probed synchronously** in calling thread. Only subsequent items go to pool.
- **Awaitable from thread worker** → `TypeError` (once sync chosen, all workers must be sync). Awaitable closed to avoid resource warnings.
- **Memory ordering:** `concurrent.futures.wait()` establishes happens-before between the worker's `Future.set_result(value)` and the parent's `future.result()` read, via the `Future`'s internal `threading.Condition`. This guarantees `value` (the result of `fn(item)`) is visible to the parent after `wait()` returns. quent does **not** guarantee visibility of worker-side mutations to **shared external state** that are not the returned value — such state must be synchronized by the user. Holds under GIL and PEP 703 free-threaded.

### 11.4 Async Concurrent Execution

Selected when first item/fn returns awaitable.

- `asyncio.Semaphore` limits concurrency.
- **3.11+:** `asyncio.TaskGroup`. The TaskGroup's own `ExceptionGroup` is **not** surfaced — quent catches it, extracts the sub-exceptions, and re-triages per §5.3/§5.5/§7.3. The user-visible exception is either a single propagating exception, a quent-specific `ExceptionGroup` with op-name in the message (`"gather() encountered N exceptions"` or `"foreach()/foreach_do() encountered N exceptions"`), or a propagating control-flow signal.
- **3.10:** `asyncio.gather()` fallback; on failure, pending tasks cancelled and awaited before triage; same user-visible wrapping rules as 3.11+.
- **Async iterables:** inputs with `__aiter__` but not `__iter__` fully materialized before dispatch. For large/unbounded → use sequential.

### 11.5 Sync/Async Detection

Probe first item/fn: call **exactly once**; awaitable → async path; non-awaitable → sync path; later worker returning awaitable on sync path → `TypeError`. The probe's result is reused as the first batch result — fn0 is **not** invoked a second time. Side effects of the probe execute regardless of the subsequent path (sync or async).

Callables must be **consistently** sync or async across all items/fns in one op. Mixed → `TypeError`.

### 11.6 Async Transition for Sync Pipeline Handlers

§6.2 covers `finally_`. Same mechanism for `except_`:

- **`reraise=True` + coroutine return in sync pipeline** → async transition. `run()` returns coroutine; caller awaits; handler completes; original re-raised. Ensures handler side-effects (e.g. async logging) complete before propagation.
- **`reraise=False` + coroutine return** → normal async transition. Coroutine → result; `run()` returns for `await`.
- **`Q.return_(fn)` lazy callable returning awaitable in sync pipeline** → same transition; outermost `run()` returns coroutine that resolves to `fn`'s value (§7.1 lazy evaluation).

**Mechanism.** The sync engine walks the linked list inline; when a handler (or lazy return value) yields an awaitable, the sync engine constructs and returns a small `async def` continuation coroutine that: (a) awaits the pending awaitable, (b) applies the appropriate post-handler semantics (re-raise / return / chain into `finally_`), (c) resolves to the final pipeline result. `run()` returns the continuation; the user sees a coroutine. The synchronous vs asynchronous distinction at the call site is preserved — caller `await`s only if `run()` returned a coroutine.

### 11.7 Context Variable Propagation

`contextvars` propagate to `ThreadPoolExecutor` workers via `copy_context().run()`. Workers see context vars with values at submit time. Async tasks inherit naturally through asyncio.

> **Isolation guarantee.** `copy_context().run(fn)` gives `fn` its own snapshot of all `ContextVar`s — any `ContextVar.set` inside is invisible to siblings and to the parent. This is asyncio/contextvars builtin behavior. The user-facing `Q.set`/`Q.get` API (§15) layers on a copy-on-write dict pattern (each `Q.set` creates a new dict, never mutates), which is defense-in-depth and also enables single-threaded re-entrancy. Either mechanism alone suffices for worker isolation.

> **Implementation note — loop detection.** §16.2's "is an event loop running?" check uses CPython's private `asyncio._get_running_loop()` (returns the running loop or `None`, no exception raised) for speed. This is implementation-internal and may need adaptation on alternate runtimes. The behavior contract: under an active asyncio/trio/curio loop, async protocol is preferred for dual-protocol objects.

### 11.8 Why Per-Invocation Executor

Deterministic cleanup (threads joined before op returns); no shared state (no cross-pipeline interaction risks); escape hatch via `executor` param for callers managing their own pool.

---

## 12. Null Sentinel

### 12.1 Why

`None` is a legitimate pipeline value (`Q(None)` passes `None` through). `Null` distinguishes "no value provided" from "value is `None`":

| Construction | Internal root | User-visible |
|---|---|---|
| `Q()` | `Null` | `None` |
| `Q(None)` | `None` | `None` |

Affects Rule 2: callable receives zero args when CV is `Null`, one otherwise (even for `None`).

### 12.2 Never Exposed

`Null` is internal — never exposed:
- `run()` returns `None` for no value.
- Finally handler receives `None` for no root.
- `on_step` `input_value` is `None` for absent.
- Context storage normalizes `Null` to `None` before storing.

Normalization at every user-visible boundary.

### 12.3 Effect on Rule 2

| CV | Callable invocation |
|---|---|
| `Null` (no value) | `fn()` |
| any other (incl. `None`) | `fn(CV)` |

`Q().then(fn).run()` → `fn()`. `Q(None).then(fn).run()` → `fn(None)`. Rule 1 ignores CV regardless.

### 12.4 Internal Safeguards

- Singleton — one instance per process; duplicate instantiation of singleton's type → `TypeError`.
- `copy.copy(Null)`/`copy.deepcopy(Null)` → same instance.
- `repr(Null)` → `'<Null>'`.

---

## 13. Traceback Enhancement

### 13.1 Visualization

On exception propagation out of a pipeline, quent grafts a synthetic `<quent>` frame onto the traceback. The frame's "function name" field holds a visualization of pipeline structure; a `<----` arrow marks the failing step.

```
Q(fetch_data)
    .then(validate)
    .do(log) <----
    .foreach(transform)
    .except_(handle_error)
    .finally_(cleanup)
```

Nested pipelines render with 4-space-per-level indentation. `<----` appears on exactly one step. `except_`, `finally_`, `if_` with `else_` (both branches) all appear.

> **Implementation note.** The `<quent>` frame is realized as a synthetic Python frame: at import time quent compiles a constant `raise __exc__` code object; on each pipeline exception, the code object's `co_name`/`co_qualname` is replaced (via `code.replace`) with the visualization string and the result is `exec()`'d inside a controlled namespace where `__exc__` is bound to the exception. The replaced `co_name` becomes the function-name field rendered by Python's `traceback` formatter — that is the visualization. The pre-compiled code object is the **only** thing `exec()`'d (security invariant: user data flows only into traceback metadata, never into executed code). Implementations filtering tracebacks by frame name should expect `co_name` to contain the visualization string and `co_filename` to be `'<quent>'`.

### 13.2 Error Marker — First-Write-Wins

Across nested pipelines, only the **innermost** failing step is recorded. Marker points to deepest origin, not intermediate re-raise.

### 13.3 Frame Cleaning

Quent strips own internal frames. Remaining: (1) user code (outside quent package); (2) synthetic `<quent>` frames.

Applies to exception itself and all chained (`__cause__`, `__context__`, `ExceptionGroup` sub-exceptions on 3.11+). Depth limit 1000 + seen-set prevent unbounded traversal and infinite loops.

### 13.4 Exception Notes (3.11+)

One-line note via `exc.add_note()`: `quent: exception at .then(validate) in Q(fetch_data)`. Attached once (idempotent — no duplicate if `quent:` note exists). Notes survive traceback reformatting. Note-generation failure silently logged; exception propagates unmodified. Named: `... in Q[label](fetch_data)`.

### 13.5 Environment Variables

| Var | Effect |
|---|---|
| `QUENT_NO_TRACEBACK=1` (also `true`, `yes`, case-insensitive) | Disables **all** traceback modifications — no visualizations, no frame cleaning, no patches. Must be set before import. For debuggers, custom handlers, CI. |
| `QUENT_TRACEBACK_VALUES=0` (also `false`, `no`) | Suppresses argument values; preserves names + structure. `repr()` replaced with type-name placeholders (e.g. `<str>`). Applies to visualizations and debug log. For production handling sensitive data. |

### 13.6 Global Patches

Unless `QUENT_NO_TRACEBACK=1`, two patches at import:
- **`sys.excepthook`** — replacement cleans quent-internal frames from exceptions carrying quent metadata before delegating.
- **`traceback.TracebackException.__init__`** — covers rendering paths (`logging`, `format_exception`, `print_exception`) that bypass `sys.excepthook`.

Originals captured once at first import (no re-capture on `importlib.reload()` — would recurse). Idempotency guards prevent stacking. `__init__` signature verified at import; mismatch → warning.

### 13.7 Repr Sanitization (CWE-117)

All `repr()` in visualizations is sanitized as defense-in-depth:
- **ANSI escape sequences** stripped (CSI, OSC, simple ESC) — prevent malicious `__repr__` injecting terminal sequences.
- **Unicode control characters** stripped (C0/C1 except tab/newline/CR; zero-width; bidi overrides; BOMs) — prevent invisible chars confusing log parsers.
- **Length truncated** to 200 chars.

### 13.8 Visualization Limits

| Limit | Value | Truncation marker |
|---|---|---|
| Nesting depth | 50 | `Q(...<truncated at depth 50>...)` |
| Links per level | 100 | `... and N more steps` |
| Total length | 10,000 chars | `... <truncated>` |
| Total recursive calls per pipeline | 500 | (rendering stops) |

Each nested pipeline gets its own budget.

### 13.9 Graceful Degradation

Visualization is best-effort. On failure: `RuntimeWarning` emitted; failure logged at DEBUG; traceback still cleaned of internal frames (fallback); underlying exception never suppressed or altered.

### 13.10 `__repr__` via Visualization

`repr(q)` uses same format without `<----`. Respects `QUENT_TRACEBACK_VALUES=0`. Same limits. Named: `Q[label](root)`. **Format is not stable across versions** — for debugging and logs only; use `name()` labels (§5.12) for stable identification in tests.

---

## 14. Instrumentation

### 14.1 `on_step` Callback

Class-level attribute. When set, called after each step completes or fails.

```python
Q.on_step = my_callback   # enable
Q.on_step = None          # disable (default)
```

```python
def on_step(q: Q, step_name: str, input_value: Any, result: Any,
            elapsed_ns: int, exception: BaseException | None) -> None
```

| Arg | Meaning |
|---|---|
| `q` | Instance being executed. |
| `step_name` | `'root'` for root, else registering method: `'then'`, `'do'`, `'foreach'`, `'foreach_do'`, `'gather'`, `'with_'`, `'with_do'`, `'if_'`, `'while_'`, `'drive_gen'`, `'except_'`, `'finally_'`. `'if_'` covers entire conditional regardless of branch. `'else_'`/`'else_do'` appear in visualizations but NOT reported as separate events — part of `if_`. Instance `q.set(...)` reports as `'do'` (it discards its return like `.do()`); instance `q.get(...)` reports as `'then'` (it replaces CV like `.then()`); both share the plain-callable name path. Iteration **terminals** (`iterate`, `iterate_do`, `flat_iterate`, `flat_iterate_do`) and the `buffer(n)` modifier do not themselves fire `on_step` — they construct an iterator whose `__iter__`/`__aiter__` runs the pipeline; the pipeline's own steps inside fire normally. |
| `input_value` | CV passed to step, normalized `None` if absent. `'root'`: run value (or `None`). `'except_'`: `QuentExcInfo`. `'finally_'`: root value (or `None`). |
| `result` | Value produced. `None` on failure. |
| `elapsed_ns` | Wall-clock ns via `time.perf_counter_ns()`. |
| `exception` | Exception raised or `None`. Fires **before** `except_` runs — raw exception visible. |

**Control flow:** `on_step` does NOT fire for steps raising `Q.return_()`/`Q.break_()` — intentional control flow.

### 14.2 Zero Overhead When Disabled

`on_step is None` → no timing, no dispatch; engine reads once at start, gates timing/callback on single boolean. Genuinely zero cost, not a no-op callback.

### 14.3 Callback Errors

`on_step` raising: WARNING via `'quent'` logger; `RuntimeWarning`; pipeline continues. Instrumentation must never break instrumented code.

### 14.4 Thread Safety

Class-level (not per-instance). Set before any concurrent execution. Mutating during concurrent execution is a data race (PEP 703). Subclass overrides respected (engine reads via `type(q).on_step`). Callback itself must be thread-safe under concurrent pipelines.

### 14.5 Debug Logging

DEBUG-level via `'quent'` logger:

| Event | Format |
|---|---|
| Pipeline start | `[exec:<id>] pipeline <repr>: run started` |
| Step completion | `[exec:<id>] pipeline <repr>: <step_name> -> <result_repr>` |
| Async transition | `[exec:<id>] pipeline <repr>: async transition at <step_name>` |
| Pipeline completion | `[exec:<id>] pipeline <repr>: completed -> <result_repr>` |
| Step failure | `[exec:<id>] pipeline <repr>: failed at <step_name>: <exc_repr>` |
| Async continuation | `[exec:<id>] pipeline <repr>: async continuation started` |

`<id>` is zero-padded 6-digit hex execution counter (e.g. `[exec:00002a]`), unique per `run()`. Correlates lines across async transitions.

Gated by `_log.isEnabledFor(DEBUG)` — no `repr()`/formatting when not DEBUG. `QUENT_TRACEBACK_VALUES=0` replaces `repr()` with type placeholders.

### 14.6 `debug(v=<no value>, /, *args, **kwargs)`

Signature matches `run()`. Executes with step-level instrumentation; returns `DebugResult` capturing the trace. **Original pipeline unmodified** — `debug()` clones internally.

**Mechanism:** lazy `Q` subclass with own `on_step`; clone converted to that subclass — independent instrumentation without touching global `Q.on_step`. Each step appends `StepRecord` via subclass callback.

**Return:** `DebugResult` for sync; coroutine resolving to `DebugResult` for async.

**`DebugResult` fields:**

| Field | Meaning |
|---|---|
| `value: T` | Final result. |
| `steps: list[StepRecord]` | Ordered list. |
| `elapsed_ns: int` | Total wall-clock ns. |
| `succeeded: bool` (prop) | True iff all steps succeeded. |
| `failed: bool` (prop) | True iff any step raised. |
| `print_trace(file=None)` | Prints formatted trace table to `file` (default `sys.stderr`). Columns: index, step name, input, result, elapsed, status (OK/FAIL). Values truncated at 60 chars. |

**`StepRecord` (frozen dataclass):** `step_name`, `input_value`, `result`, `elapsed_ns`, `exception`. Property `ok` iff `exception is None`.

**Exception behavior:** `debug()` does NOT suppress errors; exception propagates normally; `DebugResult` not returned on failure.

**Build constraint:** pending `if_`/`while_` → `QuentException`.

`DebugResult`, `StepRecord` exported in `__all__`.

---

## 15. Context API

Pipeline steps are positional — each receives only the CV from the previous step. The context API provides named storage scoped to the execution context: accessible from any step without altering value flow.

`Q.set`/`Q.get` are descriptors dual-dispatching on access form:
- **Instance access** (`q.set`, `q.get`): builder methods, append step, return pipeline.
- **Class access** (`Q.set`, `Q.get`): immediate operations at call site, not steps.

### 15.1 `set` — Store Under Key

**Instance step (`q.set(key) -> Self`, `q.set(key, value) -> Self`):** appends step that stores under `key`. CV **not changed** (like `.do()`).
- `q.set(key)` — stores CV under `key`.
- `q.set(key, value)` — stores explicit `value`; CV ignored (captured in closure).
- `Null` normalized to `None` before storage.

**Class immediate (`Q.set(key, value) -> None`):** stores at call site. Not a step. For pre-populating before `run()` or use inside lambdas.

```python
Q.set('config', load_config())
Q(fetch_data).then(lambda data: process(data, Q.get('config'))).run()
```

**`if_`/`while_` constraint:** `.set()` does NOT consume a pending flag. Calling while pending → `QuentException`. For conditional storage: `.if_(pred).do(lambda cv: Q.set('key', cv))`.

### 15.2 `get` — Retrieve By Key

**Instance step (`q.get(key) -> Self`, `q.get(key, default) -> Self`):** appends step retrieving under `key`. Retrieved value **replaces** CV (like `.then()`).
- Missing, no default → `KeyError` at execution.
- Missing, default → default becomes new CV.

**Class immediate (`Q.get(key, default=<missing>) -> Any`):** retrieves at call site. Missing, no default → `KeyError`. Missing, default → returns default (like `dict.get()`).

**`if_`/`while_` constraint:** same as `.set()` → `QuentException`. For conditional retrieval: `.if_(pred).then(Q.get, 'key')`.

### 15.3 Storage and Scoping

**Mechanism:** single module-level `ContextVar[dict[str, Any]]`. Each `set()` creates a **new** dict (spread of existing + new key) rather than mutating. Copy-on-write essential for concurrent isolation.

**Non-concurrent:** context persists in caller's thread. Values from one execution visible to subsequent in same thread (direct consequence of `contextvars`).

**Concurrent workers (`foreach`/`gather` with concurrency):** inherit snapshot via `copy_context().run()` (§11.7). Each worker starts with copy at dispatch time. Worker `set()` does NOT propagate back to parent or siblings — guaranteed by `copy_context()` (shallow copy) + copy-on-write dict semantics.

**Async concurrent tasks:** inherit naturally through asyncio. Same isolation guarantees.

---

## 16. Design Decisions

### 16.1 Unopinionated

quent is a pipeline builder, not a framework. Exposes primitives (threads, iterators, concurrency, CMs) without artificial caps, warnings, or safety nets:

- No cap on thread pool sizes or concurrency (`-1` is unbounded — user decides).
- No warn on unbounded materialization (user controls iteration).
- No imposed timeouts (caller's responsibility).
- No rate-limiting of concurrent ops (user's executor/semaphore).
- No guard on large `gather()` fan-outs (10,000 tasks = 10,000 dispatched).

Guardrails would mean wrong defaults for some use case — and forcing every user to learn how to disable them. Escape hatches (`executor`, `concurrency`) trust the user.

### 16.2 Dual-Protocol Objects Prefer Async

When a CV implements both sync and async protocols (CMs `__enter__`/`__exit__` + `__aenter__`/`__aexit__`; iterables `__iter__` + `__aiter__`) and an async loop is running → async preferred. Otherwise sync.

Loop detection covers asyncio, trio, curio without importing them — `sys.modules` lookups (~50ns when absent), zero overhead when those libraries are not loaded.

Objects like `aiohttp.ClientSession` implement both, but their sync protocol is a compatibility stub — real resource management is async. Preferring async under running loop ensures correct behavior. Applied in §5.3, §5.6, §5.7, §9.7.

### 16.3 Build- vs Run-Time Enforcement

| Constraint | When | Error |
|---|---|---|
| Non-callable for `do`, `foreach.fn`, `gather.fns`, `with_`, `with_do`, callable `if_`/`while_` predicates, `else_do`, `except_`, `finally_`, `drive_gen.fn` | Build | `TypeError` |
| Args/kwargs with non-callable (Rule 1) | Build | `TypeError` |
| Duplicate `except_()`/`finally_()` | Build | `QuentException` |
| Empty `exceptions` iterable | Build | `QuentException` |
| Non-`BaseException` subclass in `exceptions` | Build | `TypeError` |
| String value in `exceptions` | Build | `TypeError` |
| `else_*` after non-`if_` step or while `if_` pending | Build | `QuentException` |
| Two consecutive `if_`/`while_` without body | Build | `QuentException` |
| Non-`.then()`/`.do()` while `if_`/`while_` pending | Build | `QuentException` |
| `concurrency` out of range or wrong type | Build | `ValueError`/`TypeError` |
| `executor` not an `Executor` | Build | `TypeError` |
| `buffer(n)` with invalid `n` | Build | `ValueError`/`TypeError` |
| Pending `if_`/`while_` at `run()`/`as_decorator()`/`debug()` | Run | `QuentException` |
| CV not iterable (iteration ops) | Run | `TypeError` |
| CV not a CM (`with_`/`with_do`) | Run | `TypeError` |
| CV not generator/factory (`drive_gen`) | Run | `TypeError` |
| `Q.return_()` raised inside `except_`/`finally_` handler | Run | `QuentException` |
| `Q.break_()` raised inside `except_`/`finally_` handler | Run | `QuentException` |
| `Q.break_()` raised inside `gather()` worker | Run | `QuentException` |
| `Q.break_()` escaping outermost `run()` with no enclosing iteration scope | Run | `QuentException` |
| Control-flow signal (`Q.return_`/`Q.break_`) raised by a lazy callable inside another signal's value | Run | `QuentException` |

### 16.4 Signals as `BaseException`

`Q.return_()`/`Q.break_()` inherit from `BaseException`, not `Exception`. Must bypass user `except Exception`, including pipeline's own `except_()` (default `Exception`). If they were `Exception` subclasses, `except_()` would intercept them — preventing `return_()` from exiting and `break_()` from terminating iteration. Misuse inside handlers is caught and wrapped in `QuentException`.

### 16.5 First-Write-Wins for Exception Metadata

Failing step + runtime args use first-write-wins: only **innermost** failing step stored. Marker points to where error originated — not intermediate re-raise (§13.2).

### 16.6 Three-Tier Iteration

`foreach`/`foreach_do` use three tiers:
1. **Sync fast path** — `next()` in `while True` (a `for` loop would silently consume `StopIteration` from `fn()` and would not allow mid-iteration awaitable detection).
2. **Mid-operation async transition** — sync discovers `fn(item)` returned coroutine; hands **live iterator** + partial results to async continuation. Continuation picks up at that item — no item reprocessed.
3. **Full async** — input async from start (`__aiter__` and not `__iter__`).

No work repeated during sync-to-async transition. Item 51's coroutine doesn't restart from item 1; 50 prior results preserved. Avoids wasted execution; prevents side effects from re-evaluating pure functions.

### 16.7 Single Handler Per Pipeline

One `except_()` and one `finally_()` per pipeline. Multiple would introduce ambiguity about ordering, precedence, "winning". Per-step handling is achieved by nested pipelines — each gets its own pair. Reuses existing mechanism rather than introducing a second.

### 16.8 Pickle-Unblocked, Copy-Blocked

`copy.copy`/`copy.deepcopy` blocked — would silently produce broken pipeline (shared linked-list → subtle corruption). Use `clone()` (§10.1). Correctness, not philosophy.

Pickling NOT blocked. Most contents naturally fail to pickle; explicit prevention was redundant hand-holding inconsistent with §16.1.

---

## 17. Known Asymmetries

The **only** places where §2 does not hold without qualification.

### 17.1 Sync Iteration Raises `TypeError` on Awaitable

Sync iteration (`for item in q.iterate()`, `for item in q.iterate_do()`, `for item in q.flat_iterate()`, `for item in q.flat_iterate_do()`) cannot await. Every entry point where an awaitable could appear during sync iteration raises `TypeError`:

| Source of awaitable | Where it fails |
|---|---|
| Pipeline `run()` phase returns a coroutine | `iterate*` setup, before first yield |
| Callback `fn(item)` returns a coroutine | `iterate*` per-element |
| `flat_iterate.fn`/`flush` returns a coroutine | `flat_iterate*` per-element / at flush time |
| Deferred `with_(fn)` whose `fn` returns a coroutine, or async-only CM | At iteration start (deferred CM entry) |
| `finally_()` returns a coroutine in a sync `for` over `iterate*` | When the generator's `finally:` runs |

Sample messages: `"Cannot use sync iteration on an async pipeline; use 'async for' instead"`; `"iterate() callback returned a coroutine. Use 'async for' with __aiter__ instead of 'for' with __iter__"`. Sync generators cannot `await` — no language mechanism bridges this within `__iter__`/`__next__`. Switch to `async for`.

### 17.2 Concurrent Sync Workers Cannot Return Awaitables

In a concurrent op selected as sync (first probe non-awaitable), later worker returning awaitable → `TypeError`. Awaitable closed (if `.close()` exists) to prevent leaks. No event loop in worker threads.

### 17.3 `return_(value)` / `exit_(value)` During Deferred Iteration

When `Q.return_(value)` or `Q.exit_(value)` is raised by a step running during iteration of an iteration terminal (`iterate`/`iterate_do`/`flat_iterate`/`flat_iterate_do`), the iterator yields `value` as one final item and stops. Prior items are preserved (already emitted, cannot be retracted).

This is a quent-specific extension. The Python analogy — `return X` inside a generator function — discards `X` (it goes into `StopIteration.value`, but `for x in gen():` doesn't see it). quent's iterators surface the value as a final yield instead, because the pipeline's step is producing data for the consumer; discarding the final return would be surprising for a data pipeline.

`Q.exit_()` behaves the same as `Q.return_()` here because there is no outermost `run()` boundary during deferred iteration — the iterator is being driven externally. Treating exit_ as "yield-and-stop" is the most useful interpretation: it terminates iteration cleanly with a final value, mirroring `Q.return_()`.

In **non-iteration** contexts, `Q.return_()` follows §7.1 cleanly — returns from the current `Q` (the one whose pipeline contains the call); the value becomes that `Q`'s `run()` return (or, if nested, that nested step's result). `Q.exit_()` follows §7.5 — propagates to outermost `run()`.

### 17.4 Concurrent Operations Require Uniform Sync/Async

In `foreach`/`foreach_do` with concurrency, or `gather()`, all callables must be uniformly sync or async. First probe decides; opposite kind later → `TypeError`. Bridge does NOT hold for individual callable replacement here — replace all uniformly. Sync uses `ThreadPoolExecutor` (no loop in worker threads); async uses `asyncio.Task`. Mutually exclusive execution models.

### 17.5 `StopIteration` → `RuntimeError` (PEP 479) — Path-Dependent

PEP 479 applies to **generator frames**: a `StopIteration` raised inside a running generator becomes `RuntimeError`. Whether a callback's `StopIteration` is converted depends on which iteration path quent uses:

| Iteration path | quent internal | Callback `StopIteration` |
|---|---|---|
| `foreach` / `foreach_do` (sync) | `while next(it)` loop — not a generator frame | Propagates as-is |
| `iterate` / `iterate_do` / `flat_iterate*` (sync) | Generator frame (`def _sync_generator`) | Wrapped as `RuntimeError` (PEP 479) |
| Any async path with `async def` callback raising `StopIteration` | Python converts at call site | `RuntimeError` regardless |

Same logical error, different observable exception type depending on path. Language constraint, not a quent choice — but the spec calls out the path so users can predict.

### 17.6 Dual-Protocol Behavior Depends on Runtime State

A value implementing both protocols takes a different code path depending on whether an async loop is running (§16.2). Same pipeline + same value may behave differently under different ambient state. Trade-off: objects like `aiohttp.ClientSession` need this preference to behave correctly.

### 17.7 `drive_gen()` `fn` Ignores Standard Calling Conventions

§4.1 introduces the calling conventions as "universal — steps, predicates, handlers, branches, bodies. Sole exception: `drive_gen()`'s step fn (§5.11)." The exception is normative: `drive_gen`'s `fn` is **always** called as `fn(yielded_value)`, ignoring Rule 1 even when `args`/`kwargs` are provided (none can be provided — `drive_gen(fn, /)` has no varargs in its signature). This is a calling-convention asymmetry (not a bridge-contract asymmetry — sync/async equivalence still holds for `drive_gen`).

---

## 18. Patterns

```python
# Fan-out and combine
Q(url).then(fetch).gather(extract_title, extract_body, extract_meta) \
      .then(lambda t: {'title': t[0], 'body': t[1], 'meta': t[2]}).run()

# Concurrent transform and aggregate
Q(image_paths).foreach(resize, concurrency=8) \
              .then(lambda rs: sum(r.size for r in rs)).run()

# Conditional branching via nested pipelines
Q(request).if_(is_authenticated).then(Q().then(load_profile).then(render)) \
                                .else_(Q().then(redirect_to_login)).run()

# Per-step error handling via nested pipeline
safe_fetch = Q().then(fetch).except_(lambda exc: default_response)
Q(urls).foreach(safe_fetch).then(merge_responses).run()

# Function decoration
@Q().then(json.loads).then(normalize).then(validate).as_decorator()
def read_config(path):
  with open(path) as f: return f.read()

# Transparent sync/async bridging — one pipeline, both callers
pipeline = Q().then(fetch_user).then(parse_profile).do(log_access).then(enrich).run
result = pipeline(user_id)         # sync — plain value if all-sync
result = await pipeline(user_id)   # async — coroutine if any step transitioned

# Looping
Q(100).while_(lambda x: x > 1).then(lambda x: x // 2).run()  # → 1

# Generator driving (httpx auth flow: sync yield + async send)
await Q(auth_flow(req)).drive_gen(async_send).run()
```

---

## 19. Public API

Exported from `quent` via `__all__`.

| Symbol | Description |
|---|---|
| `Q` | Pipeline builder — the core primitive. |
| `QuentExcInfo` | NamedTuple passed to `except_()` handlers as CV. Fields: `exc`, `root_value` (normalized `None` if absent). |
| `QuentIterator` | Dual sync/async iterator from `iterate()`, `iterate_do()`, `flat_iterate()`, `flat_iterate_do()`. Supports `__iter__` and `__aiter__`. Callable to create new iterators with different run args. |
| `QuentException` | Base exception for quent-specific runtime errors — escaped signals, duplicate handler registration, invalid break context, etc. |
| `DebugResult` | Returned by `Q.debug()`. Fields: `value`, `steps` (`list[StepRecord]`), `elapsed_ns`. Properties: `succeeded`, `failed`. Method: `print_trace(file)`. |
| `StepRecord` | Frozen dataclass for one step in a `debug()` trace. Fields: `step_name`, `input_value`, `result`, `elapsed_ns`, `exception`. Property: `ok` iff `exception is None`. |
| `__version__` | PEP 440 version string. Resolved from installed package metadata; falls back to `'0.0.0-dev'`. |
