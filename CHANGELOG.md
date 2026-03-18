# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[6.0.0]: https://github.com/drukmano/quent/releases/tag/v6.0.0
[5.3.0]: https://github.com/drukmano/quent/releases/tag/v5.3.0
[5.2.0]: https://github.com/drukmano/quent/releases/tag/v5.2.0
[5.1.0]: https://github.com/drukmano/quent/releases/tag/v5.1.0
[5.0.0]: https://github.com/drukmano/quent/releases/tag/v5.0.0
