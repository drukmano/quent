# quent — Behavioral Specification

**Version:** 6.1.2 | **Date:** 2026-05-19

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

*Build once; run sync or async.* Any callable at any position is interchangeable with its async equivalent — observable result is identical. Caller selects no mode, wraps no coroutines, writes no conditional `await`. `run()` returns a plain value if all-sync, a coroutine if any step transitioned. The **bridge contract** (§2) is the load-bearing invariant; every behavior upholds it or is documented as an exception (§17).

---

## 2. The Bridge Contract

### 2.1 Invariant

> For any pipeline `P` and step `i`, replacing step `i`'s callable with a functionally equivalent callable of the opposite sync/async kind produces the same observable result.

**Functionally equivalent:** for the same input, sync returns `V`; async returns a coroutine resolving to `V`. Holds for every operation, handler, predicate, branch, body. Exceptions: §17.

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

---

## 3. Pipeline Model

### 3.1 Shape and Storage

Sequential computation threading a **current value (CV)**. Each step takes CV; result becomes new CV (side-effect steps preserve CV).

Append-only singly-linked list. Append O(1); walks head-to-tail; never mutated post-construction.

**Thread safety:** building not thread-safe. A constructed pipeline executes safely from multiple threads (incl. PEP 703 free-threaded) — execution uses only function-local state.

### 3.2 Root Value

`Q(v=<no value>, /, *args, **kwargs)`:

| Form | Behavior |
|---|---|
| `Q()` | No root. First step invoked per §4 with no value. |
| `Q(v)`, `v` callable | At `run()`: `v(*args, **kwargs)` (or `v()` if absent). Return → root. |
| `Q(v)`, `v` non-callable | `v` is root as-is. `args`/`kwargs` absent or build `TypeError`. |
| `Q(None)` | Root is `None`. |
| `Q(key=val)` (no positional) | Build `TypeError` — kwargs require root callable. |

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

**Rule 2 dispatch:**

| `fn` callable? | CV present? | Invocation |
|---|---|---|
| Yes | Yes | `fn(CV)` |
| Yes | No (Null) | `fn()` |
| No | — | `fn` itself becomes new CV (literal replacement) |

```python
Q(5).then(format_number, 'USD', decimals=2)  # Rule 1: format_number('USD', decimals=2); 5 NOT passed
Q(5).then(str)                                # Rule 2 callable + CV: str(5) → '5'
Q().then(get_timestamp)                       # Rule 2 callable, no CV: get_timestamp()
Q(5).then(42)                                 # Rule 2 non-callable: CV = 42
```

### 4.2 Nested Pipelines

`Q.__call__` ≡ `run()`. A nested `Q` dispatches per §4.1. Caller-provided args/kwargs **replace** the inner's build-time root args/kwargs entirely (no merging); inner root callable is preserved.

| Registration | Invocation |
|---|---|
| `.then(inner)` | `inner.run(CV)` |
| `.then(inner, arg, key=val)` | `inner.run(arg, key=val)` — CV NOT passed |

**Signal propagation:** `Q.return_()`/`Q.break_()` inside nested → propagates to **outermost** pipeline. Escaping outermost `.run()` → wrapped in `QuentException`.

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

**Concurrent** (`concurrency=-1` unbounded or positive int): input is **eagerly materialized** — not for infinite/very-large iterables. Sync vs async path probed on first element (§11.5). Mixed → `TypeError`. Results preserve **input order**. `-1` resolves to `len(items)`. Params: §11.

**Errors:**
- Sequential: propagates immediately. `StopIteration` from callback propagates as regular exception; PEP 479 wraps it as `RuntimeError` in async (§17.5).
- Concurrent: single → propagates directly. Multiple `Exception`s → `ExceptionGroup`. `BaseException` never wraps; earliest-input-index wins. Signal priority (§7.3): `return_` > `break_` > regular.

**`break_()`:** sequential — stops at break index; partial results + carried value appended. Concurrent — results truncated to elements before earliest-index break; carried value appended.

### 5.4 `foreach_do(fn, /, *, concurrency=None, executor=None)`

Same as `foreach()` except: `fn`'s returns discarded; **original input elements** collected in input order. Error/break behavior matches `foreach()`.

### 5.5 `gather(*fns, concurrency=-1, executor=None)`

Run multiple fns on CV concurrently. Each `fn` receives CV. Result is **tuple** in positional order.

- **Always concurrent** — `concurrency=None` rejected (sequential gather ≡ chained `then()`; always-concurrent eliminates bridge asymmetry). Tuple (not list) signals fixed structure: `len(result) == len(fns)`.
- Single-fn → `(result,)`. Zero fns → `QuentException`.
- Sync/async probed on first fn (§11.5). Mixed → `TypeError`.
- Each fn callable (build `TypeError`).

**Errors:** single → propagates directly. Multiple → `ExceptionGroup("gather() encountered N exceptions")`. `BaseException` never wraps (earliest-position wins). `Q.return_()` absolute priority — co-occurring regular exceptions discarded with WARNING. `Q.break_()` inside gather → `QuentException` (gather is not an iteration scope).

### 5.6 `with_(fn, /, *args, **kwargs)`

Enter CV as context manager; invoke `fn` per §4 with `__enter__`/`__aenter__` result; replace CV with `fn`'s return. CM exits on success or failure.

- `fn` callable required (build `TypeError`).
- CV must support `__enter__`/`__exit__` or `__aenter__`/`__aexit__` else `TypeError`. Dual-protocol prefers async under running loop (§16.2).
- Awaitable from `fn` → async transition.

**Exception suppression:** `fn` raises + `__exit__` returns truthy → pipeline continues with CV `None` (body result unavailable). Falsy → propagates.

**Exit handler failure:** `__exit__` raising replaces body exception via `raise exit_exc from exc` — `__cause__` set, `__context__` preserved (original reachable on both). Clearer than Python's native `with` (implicit-`__context__`-only).

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
- nested `Q` — `inner.run(CV)` (Rule 2); result tested. `Q.return_()` inside propagates (valid early exit). `Q.break_()` inside → `QuentException` (predicate is not iteration scope).
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

Set pending loop flag. Next `.then()`/`.do()` becomes body; body repeats while predicate truthy.

**Predicate:** same forms as `if_()`. `Null` always falsy. Args/kwargs with non-callable non-`None` literal → build `TypeError`.

**Constraints:** `while_` while `if_`/`while_` pending → `QuentException` (combine via nested pipeline). `if_` while `while_` pending → `QuentException`. Non-`.then()`/`.do()` while pending → `QuentException`. `else_*` after `while_().then()`/`.do()` → `QuentException`.

**Body modes:**
- `.then(fn)` — result feeds back as loop value next iteration; on termination, final loop value becomes new pipeline CV.
- `.do(fn)` — side-effect; loop value unchanged; on termination, CV passes through unchanged.

> With `.do()` + predicate testing loop value (incl. default `None`), loop value never changes → infinite loop unless `break_()`.

**`break_()`:** `Q.break_()` stops; result is current loop value. `Q.break_(value)` stops; break value becomes result. Valid from body or predicate. Distinct from `foreach` break (§5.3): `while_` preserves loop/break value; `foreach` preserves partial results (§7.2).

**`return_()`:** propagates to enclosing pipeline — exits whole pipeline, not just loop.

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

**Errors:** `fn` or send exception (not Stop) propagates; generator closed in cleanup. NOT injected (no `gen.throw()`). Control flow signals propagate; generator closed. Generator ignoring `GeneratorExit` on close → Python `RuntimeError`; propagates from cleanup.

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

**Consumption vs re-raise:**

| `reraise` | Effect |
|---|---|
| `False` (default) | Handler's return → pipeline result; exception consumed; finally runs in success context. |
| `True` | Handler runs for side-effects only; after, original is re-raised; handler's return ignored. |

**Handler failure with `reraise=True`:**
- `Exception` subclass → handler exception **discarded**; `RuntimeWarning` emitted; note attached (3.11+); `__context__`/`__suppress_context__` on original **restored** to pre-handler values (prevents handler from permanently mutating chain); original re-raised.
- `BaseException` subclass → propagates naturally; system signals never suppressed.

**Handler failure with `reraise=False`:** handler exception propagates; original set as `__cause__` (`raise handler_exc from original_exc`).

**Control flow:** `Q.return_()`/`Q.break_()` inside except → `QuentException`.

### 6.2 `finally_(fn, /, *args, **kwargs)`

- `fn` callable (`TypeError`). CV per §4.3: root value (normalized `None`).
- **Always runs** — success and failure paths.
- **Return always discarded** — cannot alter result.

**Failure:**
- Finally raises while exception active → finally's exception **replaces** original (Python `try/finally`); original preserved as `__context__`; note attached.
- Finally raises on success → finally's exception propagates as pipeline error.
- Both `except_` and `finally_` raise → finally wins; except's exception preserved as `__context__`.

**Control flow:** `Q.return_()`/`Q.break_()` inside finally → `QuentException`.

**Async finally in sync pipeline:** coroutine return triggers **async transition** — `run()` returns coroutine; when awaited, finally awaited first, then result returned (success) or active exception re-raised (failure). Nothing discarded. For `iterate()`/`iterate_do()` during sync `for`: coroutine-returning finally → `TypeError` (sync generators cannot await; use `async for`).

### 6.3 Execution Order

1. Steps execute sequentially.
2. Step raises matching `exceptions`:
   - `reraise=False` → except runs; return → result; finally success-context.
   - `reraise=True` → except runs; original re-raised; finally failure-context.
3. Step raises non-matching → propagates; finally failure-context.
4. Success → finally success-context.
5. Finally **always last**.

### 6.4 ExceptionGroup

Concurrent ops wrap multiple failures. 3.11+: builtin. 3.10: polyfill — `ExceptionGroup(message, exceptions)` (non-empty list of `Exception`), `.exceptions`, `.subgroup(condition)`, `.split(condition)` → `(matching, rest)`, `.derive(excs)` (preserves traceback and cause/context chains). Single failure → not wrapped. Iteration messages: `"foreach()/foreach_do() encountered N exceptions"`.

---

## 7. Control Flow

`Q.return_()` and `Q.break_()` are classmethods raising internal `BaseException` subclasses — bypass user `except Exception`, including pipeline's own `except_()` (default `Exception`); see §16.4. User does not catch them directly.

Values are **lazy** — callable values invoked only when caught, avoiding work on propagation through nested pipelines.

Idiomatic: `return Q.return_(value)` / `return Q.break_(value)` — satisfies type checkers, avoids unreachable-code warnings (the mechanism does not require `return`).

### 7.1 `Q.return_(v=<no value>, /, *args, **kwargs)`

Signal early termination of pipeline.

| Form | Pipeline result |
|---|---|
| `Q.return_()` | `None` |
| `Q.return_(42)` | `42` (non-callable as-is) |
| `Q.return_(fn)` | `fn()` lazily |
| `Q.return_(fn, *args, **kwargs)` | `fn(*args, **kwargs)` lazily |

**Nested:** propagates to **outermost** (§4.2). **Restrictions:** inside `except_`/`finally_` → `QuentException`; escaping outermost `run()` → `QuentException`.

### 7.2 `Q.break_(v=<no value>, /, *args, **kwargs)`

Signal early termination of iteration.

| Form | Effect |
|---|---|
| `Q.break_()` | Returns results collected so far. |
| `Q.break_(value)` | Value **appended** to results. |
| `Q.break_(fn, ...)` | `fn(...)` lazy; result appended. |

Applies to `foreach`, `foreach_do`, `iterate`, `iterate_do`, `flat_iterate`, `flat_iterate_do`. `while_` semantics: §5.10.

**Restrictions:** outside an iteration/loop scope, inside `except_`/`finally_`, inside `if_()` predicate, inside `gather()` → `QuentException`.

### 7.3 Priority in Concurrent Iteration

1. **`Q.return_()`** — absolute, immediate propagation.
2. **`Q.break_()`** — over regular exceptions; multiple → earliest **input index** wins.
3. **Regular exceptions** — single propagates; multiple → `ExceptionGroup` (§6.4).

`BaseException` (e.g. `KeyboardInterrupt`) never wraps; earliest-input-index over regular exceptions.

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
- **Memory ordering:** `concurrent.futures.wait()` establishes happens-before via `Future`'s `threading.Condition`. Safe under GIL and PEP 703 free-threaded.

### 11.4 Async Concurrent Execution

Selected when first item/fn returns awaitable.

- `asyncio.Semaphore` limits concurrency.
- **3.11+:** `asyncio.TaskGroup`; failures via `TaskGroup`'s `ExceptionGroup`.
- **3.10:** `asyncio.gather()` fallback; on failure, pending tasks cancelled and awaited before triage.
- **Async iterables:** inputs with `__aiter__` but not `__iter__` fully materialized before dispatch. For large/unbounded → use sequential.

### 11.5 Sync/Async Detection

Probe first item/fn: call; awaitable → async path; non-awaitable → sync path; later worker returning awaitable on sync path → `TypeError`.

Callables must be **consistently** sync or async across all items/fns in one op. Mixed → `TypeError`.

### 11.6 Async Transition for Sync Pipeline Handlers

§6.2 covers `finally_`. Same mechanism for `except_`:

- **`reraise=True` + coroutine return in sync pipeline** → async transition. `run()` returns coroutine; caller awaits; handler completes; original re-raised. Ensures handler side-effects (e.g. async logging) complete before propagation.
- **`reraise=False` + coroutine return** → normal async transition. Coroutine → result; `run()` returns for `await`.

### 11.7 Context Variable Propagation

`contextvars` propagate to `ThreadPoolExecutor` workers via `copy_context().run()`. Workers see context vars with values at submit time. Async tasks inherit naturally through asyncio.

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

`repr(q)` uses same format without `<----`. Respects `QUENT_TRACEBACK_VALUES=0`. Same limits. Named: `Q[label](root)`.

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
| `step_name` | `'root'` for root, else registering method: `'then'`, `'do'`, `'foreach'`, `'foreach_do'`, `'gather'`, `'with_'`, `'with_do'`, `'if_'`, `'while_'`, `'drive_gen'`, `'except_'`, `'finally_'`. `'if_'` covers entire conditional regardless of branch. `'else_'`/`'else_do'` appear in visualizations but NOT reported as separate events — part of `if_`. |
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
| Control flow signal escaping outermost `run()` | Run | `QuentException` |
| Signal misused (in handler, in `if_` predicate, in `gather`, outside scope) | Run | `QuentException` |

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

### 17.1 Sync `iterate()` Raises `TypeError` on Awaitable

`for item in q.iterate()` with awaitable from pipeline or `fn`:

> Cannot use sync iteration on an async pipeline; use 'async for' instead

> iterate() callback returned a coroutine. Use "async for" with `__aiter__` instead of "for" with `__iter__`.

Sync generator cannot `await`. No language mechanism bridges this within `__iter__`/`__next__`. Switch to `async for`. A sync pipeline's coroutine-returning `finally_()` during sync iteration also raises.

### 17.2 Concurrent Sync Workers Cannot Return Awaitables

In a concurrent op selected as sync (first probe non-awaitable), later worker returning awaitable → `TypeError`. Awaitable closed (if `.close()` exists) to prevent leaks. No event loop in worker threads.

### 17.3 `return_(value)` Differs Between `run()` and Iteration

| Context | `Q.return_(value)` |
|---|---|
| `run()` | Replaces pipeline's entire result. |
| `iterate()` etc. | Yields `value` as one final item before stopping. Prior items preserved — already emitted, cannot be retracted. |

### 17.4 Concurrent Operations Require Uniform Sync/Async

In `foreach`/`foreach_do` with concurrency, or `gather()`, all callables must be uniformly sync or async. First probe decides; opposite kind later → `TypeError`. Bridge does NOT hold for individual callable replacement here — replace all uniformly. Sync uses `ThreadPoolExecutor` (no loop in worker threads); async uses `asyncio.Task`. Mutually exclusive execution models.

### 17.5 `StopIteration` → `RuntimeError` in Async (PEP 479)

Sync callback raising `StopIteration` propagates as-is. Async callback raising `StopIteration` wrapped as `RuntimeError` by Python (PEP 479) before quent sees it. Same logical error, different observable exception type. Language constraint, not a quent choice.

### 17.6 Dual-Protocol Behavior Depends on Runtime State

A value implementing both protocols takes a different code path depending on whether an async loop is running (§16.2). Same pipeline + same value may behave differently under different ambient state. Trade-off: objects like `aiohttp.ClientSession` need this preference to behave correctly.

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
