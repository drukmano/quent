# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [5.0.0] - 2026-03-16

### Added

- **Sync/async bridge contract** -- write pipeline code once, run it sync or async automatically. Execution starts synchronously; on the first awaitable result, the engine transitions to async and stays there. Fully sync pipelines have zero async overhead.
- **`Chain` class** -- fluent pipeline builder with a singly-linked list of steps. Build-time append (O(1)), run-time immutability. Thread-safe execution on fully constructed chains, including free-threaded Python (PEP 703).
- **Core pipeline steps** -- `.then()` for value-transforming steps, `.do()` for side-effect steps (result discarded, current value passes through), `.root()` via `Chain(v, *args, **kwargs)` constructor with run-time override via `.run(v)`.
- **Calling conventions** -- two-rule dispatch applied uniformly across all contexts: (1) explicit args/kwargs suppress current value, (2) default passthrough calls `fn(current_value)`, `fn()` if absent, or returns literal as-is.
- **Iteration operations** -- `foreach(fn, concurrency=, executor=)` transforms each element; `foreach_do(fn, concurrency=, executor=)` runs side-effects per element, keeping originals. Three-tier execution: sync fast path, mid-operation async transition (hands off the live iterator with partial results), and full async path.
- **Concurrency** -- `gather(*fns, concurrency=-1, executor=)` fans out multiple functions on the same value, always concurrent. `ThreadPoolExecutor` on sync path, `asyncio.TaskGroup` (3.11+) or `asyncio.gather` (3.10) on async path. Bounded concurrency via positive integer, unbounded via `-1`. Optional user-provided `Executor` for sync operations.
- **Context manager integration** -- `with_(fn)` enters current value as a context manager, calls `fn` with the context value, replaces pipeline value; `with_do(fn)` discards result. Supports sync CMs, async CMs, and dual-protocol objects (async preferred when event loop is running).
- **Conditional steps** -- `if_(predicate).then(fn)` / `if_(predicate).do(fn)` with optional `else_(v)` / `else_do(fn)`. Predicate follows the standard calling convention; omitting predicate tests truthiness of current value. Literal predicates supported.
- **Error handling** -- single `except_(fn, exceptions=, reraise=)` and `finally_(fn)` per chain. `except_` handler receives `ChainExcInfo(exc, root_value)` as current value. `finally_` always runs, receives root value, return value discarded. Handler failures follow Python's `try/except/finally` semantics.
- **Control flow signals** -- `Chain.return_(v)` for early exit (propagates through nested chains to outermost), `Chain.break_(v)` for iteration termination (break value appended to partial results). Both are `BaseException` subclasses to bypass `except Exception`.
- **Iterator output** -- `iterate(fn=)` / `iterate_do(fn=)` return `ChainIterator`, a dual sync/async iterator. Callable for reuse with different run arguments. `return_()` and `break_()` supported during iteration.
- **Chain composition** -- `clone()` deep-copies the pipeline structure (recursive for nested chains) while sharing callables by reference. `decorator()` wraps a chain as a function decorator (cloned internally). `from_steps(*steps)` for dynamic pipeline construction.
- **Instrumentation** -- `Chain.on_step` class-level callback with signature `(chain, step_name, input_value, result, elapsed_ns)`. Zero overhead when disabled. Subclass-safe via `type(chain).on_step` lookup.
- **Traceback enhancement** -- synthetic `<quent>` frame injection with chain visualization and `<----` error marker on the failing step. Internal frame cleaning. Recursive cleaning of `__cause__`, `__context__`, and `ExceptionGroup` sub-exceptions. Exception notes on Python 3.11+. `repr()` sanitization (ANSI stripping, control character removal, length truncation).
- **Named chains** -- `.name(label)` for traceback identification; renders as `Chain[label](root)` in visualizations and exception notes.
- **Debug logging** -- step-level logging via the `quent` logger with per-execution hex IDs for correlation. Gated by `isEnabledFor(DEBUG)` for zero cost when disabled.
- **Environment controls** -- `QUENT_NO_TRACEBACK=1` disables all traceback modifications; `QUENT_TRACEBACK_VALUES=0` suppresses argument values in visualizations and debug logs.
- **`ExceptionGroup` support** -- native on Python 3.11+, polyfill on 3.10 (with `.subgroup()`, `.split()`, `.derive()`). Used by `gather()` and concurrent `foreach`/`foreach_do` when multiple workers fail.
- **Nested chain support** -- chains used as steps follow standard calling conventions; control flow signals propagate across nesting boundaries.
- **Security** -- chains and internal objects are unpicklable (`TypeError` from `__reduce__`/`__reduce_ex__`) to prevent arbitrary code execution via deserialization (CWE-502). Repr sanitization guards against log injection (CWE-117).
- **Type safety** -- full `mypy --strict` compliance, `py.typed` marker (PEP 561), typed public API (`Chain`, `ChainExcInfo`, `ChainIterator`, `QuentException`, `__version__`).
- **Python 3.10 through 3.14** support, including free-threaded builds. Zero runtime dependencies on Python 3.11+ (`typing_extensions` required only on 3.10).
- **Build-time validation** -- non-callable values with args raise `TypeError`, duplicate `except_`/`finally_` raise `QuentException`, pending `if_()` without `.then()`/`.do()` caught at `run()`/`decorator()`/`iterate()`.

[5.0.0]: https://github.com/drukmano/quent/releases/tag/v5.0.0
