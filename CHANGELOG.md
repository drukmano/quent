# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [7.0.0] - 2026-05-19

Control-flow model redesigned to mirror Python semantics. **Breaking** for code that relied on `Q.return_()` propagating to the outermost `run()` from nested pipelines.

See `BREAKING-CHANGES-from-6.1.1.md` for the full migration guide.

### Added

- **`Q.exit_()`** — new control-flow signal. Like Python's `sys.exit()`: propagates through every `Q` boundary and every signal carve-out (`except_`/`finally_`/`gather`/`drive_gen`); absorbed only at the outermost `run()`. Standard `try/finally` semantics apply during propagation (finally_ handlers run, CM `__exit__` runs, generators close). Same lazy callable forms as `Q.return_()`: `Q.exit_()`, `Q.exit_(value)`, `Q.exit_(fn)`, `Q.exit_(fn, *args, **kwargs)`.
- **§7.5** spec section defining `Q.exit_()`.
- **§7.4** carve-out table listing the only places where signal propagation is overridden, with explicit rationale for each.
- **§3.1** recursion / depth limits clause (quent imposes none; subject to Python's `sys.getrecursionlimit()` for nested pipelines).
- **§2.5** awaitable-detection performance contract (tiered check, exact-type frozenset, `~10×` faster than `inspect.isawaitable()`).
- **§17.5** PEP-479 path-dependent table (foreach uses `while next()` — no wrap; iterate* uses generator frame — wraps).
- **§17.7** drive_gen calling-convention asymmetry (the sole exception to §4.1's universality).
- New regression tests for the bug fixes below (concurrent foreach + exit_, sync-pipeline + async-finally + return_, lazy-value signal misuse, exit_ in iterate*).
- TDD audit pass (`spec-audit/tdd-audit-*.md`) verifying tests assert spec-mandated behavior, not code-observed behavior.

### Changed — Breaking

- **`Q.return_()` semantic redesign.** Now returns from the **current `Q` only** (like Python's `return`), not the outermost. If used inside a nested `Q`, only that nested `Q` returns — its value flows to the outer pipeline as the nested step's result. Use `Q.exit_()` for the old "exit entire pipeline" behavior.
- **`Q.return_()` in `gather()` worker** returns from the worker — the value becomes that gather position's tuple element (was: pipeline exit).
- **`Q.return_()` in `drive_gen` `fn`** returns from `fn` — the value becomes the pipeline CV; subsequent steps run (was: pipeline exit).
- **`Q.return_()` inside a nested-`Q` `except_`/`finally_` handler** — the nested `Q` absorbs the signal locally; the handler returns the value normally (was: `QuentException`). Plain callable handlers raising `Q.return_()` still raise `QuentException` (the handler trap applies to direct invocation).
- **`Q.break_()` in `if_()` predicate** now propagates outward to the nearest enclosing iteration scope (was: `QuentException("break_() cannot be used inside an if_() predicate")`). At top level (no enclosing iteration), still wraps as `QuentException` but with the generic *"outside of a loop or iteration context"* message.
- **`Q.break_()` propagates through more boundaries** — `if_()`/`with_`/`with_do`/`drive_gen`/nested-`Q` registered as a step no longer trap. Still trapped only in `except_`/`finally_` handlers and `gather()` workers.

### Changed — Non-breaking

- **§7 Control Flow** rewritten end-to-end around the three-signal model (`Q.return_()`, `Q.break_()`, `Q.exit_()`).
- **§4.2 Nested Pipelines** spec rewritten with signal-semantics-by-type table; "lambda wrapping breaks signal propagation" note clarified to apply specifically to `Q.break_()` (return_ is unaffected; exit_ propagates regardless).
- **§5.5 `gather`** spec: `Q.return_()` worker carve-out, `Q.break_()` rejection, `Q.exit_()` propagation.
- **§5.10 `while_`** spec: pre-tested-loop semantics pinned; predicate forms enumerated with explicit nested-`Q` predicate signal semantics.
- **§5.11 `drive_gen`** spec: mid-transition blocking note, `Q.return_()`/`Q.exit_()` carve-outs.
- **§5.6 `with_`** spec: Rule 1 drops the context value (footgun documented); `__exit__` failure chain semantics with `__cause__` / `__context__` / `__suppress_context__` made explicit.
- **§5.4 `foreach_do`** spec: heterogeneous break-value result documented (previously silent).
- **§6.1 `except_`** spec: filter-enforcement clarification; restoration mechanism (snapshot before handler) explicit.
- **§6.2 `finally_`** spec: now mandates signal preservation as `__context__` when finally raises during signal propagation; failure table extended.
- **§6.4 ExceptionGroup polyfill** spec: `.derive()` chain attribute list aligned with source (`__traceback__`/`__cause__`/`__context__`/`__notes__`; `__suppress_context__` not copied).
- **§7.2 `Q.break_()`** spec: `__suppress_context__ = True` and `__cause__ = None` pinned on the `QuentException`-wrapped form.
- **§7.3 priority** spec: explicit `Q.return_() > Q.break_() > BaseException > regular` ordering; discard-logging asymmetry (return_ logs regulars; BaseException-over-regulars does not).
- **§11.3 PEP 703 happens-before** spec: visibility narrowed to the returned value (not shared external state).
- **§11.4 TaskGroup unwrapping** spec: quent re-triages TaskGroup's `ExceptionGroup`; user-visible exception is quent-specific.
- **§11.5 probe-once** spec: pinned (fn0 invoked exactly once; result reused).
- **§11.6 async-transition mechanism** spec: documented.
- **§11.7 isolation guarantee** spec: two-mechanism layering (`copy_context().run` + copy-on-write dict).
- **§13.1 traceback implementation note** spec: code-object replacement hack documented.
- **§13.10 `repr(q)` stability** spec: pinned as non-stable (debugging only).
- **§14.1 `on_step` `step_name` enumeration** spec corrected: `q.set` reports as `'do'`, `q.get` reports as `'then'`; iteration terminals and `buffer()` do not fire `on_step`.
- **§16.3 build-vs-run-time enforcement** table revised: signal-misuse rows split per signal type.
- **§17.1 sync-iteration on awaitable** expanded to a 5-row table covering pipeline result, callback fn, `flat_iterate.fn`/`flush`, deferred `with_`, and async-finally during sync iteration.
- **§17.3** extended to cover `Q.exit_()` during deferred iteration (yield-as-final-item semantic, same as `Q.return_()`).

### Fixed

- **`except_(reraise=True)` async pipeline + sync handler** — when a sync handler with `reraise=True` raised an `Exception`, the handler's exception propagated instead of the original. This was a bridge-contract violation. Original is now re-raised with `RuntimeWarning`, note attached, and `__context__`/`__suppress_context__` properly restored.
- **`except_(reraise=True)` async pipeline + async handler** — Python's automatic `except` chaining was re-overwriting `exc.__context__` to the handler's exception immediately after `_except_handler_failed` restored it. Restoration now happens outside the active except block.
- **`finally_` raising during signal propagation** — `__context__` now correctly preserves the in-flight `Q.return_()`/`Q.break_()` signal (was being lost). Honors Python `try/finally` semantics.
- **`_triage_iter_exceptions` missing discard log** — concurrent `foreach`/`foreach_do` was silently dropping co-occurring regular exceptions when `Q.return_()` won. Now logs `RuntimeWarning` on the `'quent'` logger per §7.3, matching gather behavior.
- **`_triage_gather_exceptions` incorrect warning** — was logging a warning when `BaseException` won over regulars; spec §7.3 says "other discard paths do not warn". Warning removed.
- **`_triage_gather_exceptions` `_Break`/`_Return` priority** — was raising `QuentException` on first `_Break` encountered, preventing `_Return` at a later position from winning. Now scans the full exception list before applying priority.
- **`Q.exit_()` in concurrent `foreach`/`foreach_do`** — was wrapped as `QuentException("Unknown control flow signal: _Exit")`. Now propagates correctly per §7.5.
- **Sync pipeline + async `finally_` + absorbed `Q.return_()`** — `_run_sync_finally_dispatch` was treating `_active_exc != None` as "re-raise after finally", incorrectly re-raising the absorbed `_Return`. Dispatch now distinguishes "pipeline_result set ⇒ chain-only" from "no pipeline_result ⇒ re-raise".
- **Lazy callable raising a signal** — `_handle_return_exc`/`_handle_break_exc`/`_handle_exit_exc` now catch `_ControlFlowSignal` from the lazy callable and wrap as `QuentException` per §7.1/§7.2/§7.5 ("signals inside lazy values are misuse").
- **`Q.exit_()` in `iterate*`/`flat_iterate*`** terminals — raw `_Exit` was leaking from `__iter__`/`__aiter__`. Now yields the value as one final item and stops, mirroring `Q.return_()` per §17.3.
- **Async drive_gen `_Return` raised on awaited fn result** — the `await last_result` was outside the `try/except _Return` block, so async `fn`s raising `Q.return_()` leaked the raw signal. The await is now inside the try/except.
- **Async `Q.exit_()` outermost absorption** — `Q.run()`'s sync `except _Exit` only caught synchronously-raised `_Exit`; async transitions returned a coroutine whose `_Exit` escaped. Outermost `run()` now wraps the coroutine in `_await_outermost` for await-time absorption.

## [6.1.1] - 2026-03-22

### Added

### Changed

### Fixed

## [6.1.0] - 2026-03-19

### Changed

- Spec and documentation updates.

## [6.0.0] - 2026-03-18

### Changed

- **Identity makeover: Chain → Q** — the core class is now `Q` (formerly `Chain`). `ChainExcInfo` → `QuentExcInfo`, `ChainIterator` → `QuentIterator`. The `.decorator()` method is now `.as_decorator()`. Internal module `_chain.py` renamed to `_q.py`. All documentation, examples, and tests updated. `'chain'`/`'chaining'` kept in pyproject.toml keywords for SEO.

## [5.3.0] - 2026-03-17

### Added

- **Trio and Curio event loop detection** -- `_has_running_loop` now detects any running async event loop (asyncio, trio, curio) without importing them. Uses `sys.modules` to check if the library is already loaded, then probes its loop API. Zero overhead when a library is not loaded (~50ns dict lookup).

### Fixed

- **Dual-protocol detection under non-asyncio runtimes** -- `with_()` context manager protocol selection now uses the standard `_should_use_async_protocol` path for trio and curio, removing the previous `hasattr`-based workaround in the async generator.
- **Missing `predicate_true` import** in benchmark scripts.

## [5.2.0] - 2026-03-17

### Added

- **`flat_iterate()` / `flat_iterate_do()`** -- new flatmap iteration terminals with optional `flush` callback. `flat_iterate` flattens each element's sub-iterable one level; `flat_iterate_do` runs `fn` as a side-effect, yielding original items. Full sync/async support matching `iterate()` behavior.
- **Bare `with_()`** -- `with_()` now accepts an optional `fn`; the bare form (no `fn`) uses the context value directly as the pipeline value. Raises `TypeError` if bare form is used outside iteration.
- **Deferred `with_` in iteration** -- `iterate`/`iterate_do`/`flat_iterate`/`flat_iterate_do` detect the last `_WithOp` link and defer context manager entry to iteration time. The CM exits in the generator's `finally` block, with CM exit ordering before deferred `finally_()`. Supports exception info forwarding and suppression semantics.

## [5.1.0] - 2026-03-17

### Added

- **Context API** -- `_context.py` with `ContextVar`-backed `_ctx_set`/`_ctx_get` and copy-on-write dict semantics. Dual instance/class dispatch on `Q` via `_SetDescriptor`/`_GetDescriptor`.
- **Deferred `finally_()` in iteration** -- `iterate()`/`iterate_do()` defer the pipeline's `finally_()` handler to the generator's `finally:` block, ensuring cleanup runs after iteration ends (not before it begins).
- **`from_steps()` classmethod** -- dynamic pipeline construction from a sequence of steps.
- **Cross-platform CI matrix** -- 3 OS x 5 Python versions + free-threaded builds, bandit SAST scanning, release build provenance attestation.

### Changed

- **Concurrency refactoring** -- extracted `_make_dispatch()` and `_create_tasks_py310()` into `_concurrency.py`. Replaced `Null` with `_UNPROCESSED` sentinel in concurrent result arrays. Improved `BaseException` triage to select earliest-index exception.
- **Engine hardening** -- thread-safe execution counter with `Lock` (PEP 703 compatibility). `kwargs`-only dispatch now replaces (not merges) root link build-time args. Added debug logging for control flow signals.
- **Renamed `_UnpicklableMixin` to `_UncopyableMixin`** -- `Null` pickling now allowed.

### Fixed

- **`on_step=None` lookup bug** -- added `_UNSET_ON_STEP` sentinel to fix incorrect `on_step` callback detection.
- **Async `__exit__` during control flow signals** -- properly await async `__exit__` in `_with_ops.py` when control flow signals are raised.
- **Traceback injection hardening** -- guarded against `KeyboardInterrupt`/`SystemExit` during traceback enhancement.
- **Documentation fixes** -- corrected traceback visualization examples and `Q(callable).run(value)` examples.

## [5.0.0] - 2026-03-16

### Added

- **Sync/async bridge contract** -- write pipeline code once, run it sync or async automatically. Execution starts synchronously; on the first awaitable result, the engine transitions to async and stays there. Fully sync pipelines have zero async overhead.
- **`Q` class** -- fluent pipeline builder with a singly-linked list of steps. Build-time append (O(1)), run-time immutability. Thread-safe execution on fully constructed pipelines, including free-threaded Python (PEP 703).
- **Core pipeline steps** -- `.then()` for value-transforming steps, `.do()` for side-effect steps (result discarded, current value passes through), `.root()` via `Q(v, *args, **kwargs)` constructor with run-time override via `.run(v)`.
- **Calling conventions** -- two-rule dispatch applied uniformly across all contexts: (1) explicit args/kwargs suppress current value, (2) default passthrough calls `fn(current_value)`, `fn()` if absent, or returns literal as-is.
- **Iteration operations** -- `foreach(fn, concurrency=, executor=)` transforms each element; `foreach_do(fn, concurrency=, executor=)` runs side-effects per element, keeping originals. Three-tier execution: sync fast path, mid-operation async transition (hands off the live iterator with partial results), and full async path.
- **Concurrency** -- `gather(*fns, concurrency=-1, executor=)` fans out multiple functions on the same value, always concurrent. `ThreadPoolExecutor` on sync path, `asyncio.TaskGroup` (3.11+) or `asyncio.gather` (3.10) on async path. Bounded concurrency via positive integer, unbounded via `-1`. Optional user-provided `Executor` for sync operations.
- **Context manager integration** -- `with_(fn)` enters current value as a context manager, calls `fn` with the context value, replaces pipeline value; `with_do(fn)` discards result. Supports sync CMs, async CMs, and dual-protocol objects (async preferred when event loop is running).
- **Conditional steps** -- `if_(predicate).then(fn)` / `if_(predicate).do(fn)` with optional `else_(v)` / `else_do(fn)`. Predicate follows the standard calling convention; omitting predicate tests truthiness of current value. Literal predicates supported.
- **Error handling** -- single `except_(fn, exceptions=, reraise=)` and `finally_(fn)` per pipeline. `except_` handler receives `QuentExcInfo(exc, root_value)` as current value. `finally_` always runs, receives root value, return value discarded. Handler failures follow Python's `try/except/finally` semantics.
- **Control flow signals** -- `Q.return_(v)` for early exit (propagates through nested pipelines to outermost), `Q.break_(v)` for iteration termination (break value appended to partial results). Both are `BaseException` subclasses to bypass `except Exception`.
- **Iterator output** -- `iterate(fn=)` / `iterate_do(fn=)` return `QuentIterator`, a dual sync/async iterator. Callable for reuse with different run arguments. `return_()` and `break_()` supported during iteration.
- **Pipeline composition** -- `clone()` deep-copies the pipeline structure (recursive for nested pipelines) while sharing callables by reference. `as_decorator()` wraps a pipeline as a function decorator (cloned internally). `from_steps(*steps)` for dynamic pipeline construction.
- **Instrumentation** -- `Q.on_step` class-level callback with signature `(q, step_name, input_value, result, elapsed_ns)`. Zero overhead when disabled. Subclass-safe via `type(q).on_step` lookup.
- **Traceback enhancement** -- synthetic `<quent>` frame injection with pipeline visualization and `<----` error marker on the failing step. Internal frame cleaning. Recursive cleaning of `__cause__`, `__context__`, and `ExceptionGroup` sub-exceptions. Exception notes on Python 3.11+. `repr()` sanitization (ANSI stripping, control character removal, length truncation).
- **Named pipelines** -- `.name(label)` for traceback identification; renders as `Q[label](root)` in visualizations and exception notes.
- **Debug logging** -- step-level logging via the `quent` logger with per-execution hex IDs for correlation. Gated by `isEnabledFor(DEBUG)` for zero cost when disabled.
- **Environment controls** -- `QUENT_NO_TRACEBACK=1` disables all traceback modifications; `QUENT_TRACEBACK_VALUES=0` suppresses argument values in visualizations and debug logs.
- **`ExceptionGroup` support** -- native on Python 3.11+, polyfill on 3.10 (with `.subgroup()`, `.split()`, `.derive()`). Used by `gather()` and concurrent `foreach`/`foreach_do` when multiple workers fail.
- **Nested pipeline support** -- pipelines used as steps follow standard calling conventions; control flow signals propagate across nesting boundaries.
- **Security** -- `copy.copy()`/`copy.deepcopy()` blocked on pipelines and internal objects (use `clone()` instead). Repr sanitization guards against log injection (CWE-117).
- **Type safety** -- full `mypy --strict` compliance, `py.typed` marker (PEP 561), typed public API (`Q`, `QuentExcInfo`, `QuentIterator`, `QuentException`, `__version__`).
- **Python 3.10 through 3.14** support, including free-threaded builds. Zero runtime dependencies on Python 3.11+ (`typing_extensions` required only on 3.10).
- **Build-time validation** -- non-callable values with args raise `TypeError`, duplicate `except_`/`finally_` raise `QuentException`, pending `if_()` without `.then()`/`.do()` caught at `run()`/`as_decorator()`/`iterate()`.

[7.0.0]: https://github.com/drukmano/quent/releases/tag/v7.0.0
[6.1.1]: https://github.com/drukmano/quent/releases/tag/v6.1.1
[6.1.0]: https://github.com/drukmano/quent/releases/tag/v6.1.0
[6.0.0]: https://github.com/drukmano/quent/releases/tag/v6.0.0
[5.3.0]: https://github.com/drukmano/quent/releases/tag/v5.3.0
[5.2.0]: https://github.com/drukmano/quent/releases/tag/v5.2.0
[5.1.0]: https://github.com/drukmano/quent/releases/tag/v5.1.0
[5.0.0]: https://github.com/drukmano/quent/releases/tag/v5.0.0
