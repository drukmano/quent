# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

quent is a **transparent sync/async bridge**. It is not a collections library, not a functional programming toolkit, not an opinionated framework. It provides the minimum set of pipeline primitives that let developers:

- **Write code once** — a single chain definition works for both sync and async callables. Zero ceremony, zero code duplication.
- **Migrate existing codebases** — unify separate sync and async implementations of the same logic into one. Stop maintaining two versions of every function.
- **Stay out of the way** — quent is unopinionated. It bridges the sync/async divide and gets out of the way. No imposed patterns, paradigms, or abstractions beyond the pipeline itself.

**What quent is NOT:** a collections/iteration library (use itertools, more-itertools), a functional programming framework (use returns, toolz, Expression), or an opinionated architecture (use Effect, returns).

**Design principle:** Every feature must justify itself by solving a real sync/async bridging problem or eliminating genuine code duplication. Features that merely wrap stdlib functionality don't belong.

## Commands

**Full QA pipeline** (format, lint, typecheck, test with coverage):
```bash
bash scripts/run_tests.sh
```

**Run a single test file:**
```bash
python -m unittest tests.core_tests
```

**Run a single test method:**
```bash
python -m unittest tests.core_tests.CoreTests.test_null_sentinel
```

**Format and lint:**
```bash
ruff format quent/
ruff check --fix quent/
```

**Type check:**
```bash
mypy quent/
```

**Byte-compile (syntax check):**
```bash
bash scripts/compile.sh
```

**Build distribution:**
```bash
bash scripts/build.sh
```

## Code Style

- **2-space indentation** (enforced by ruff)
- Line length: 120
- Single quotes
- Ruff rules: E, W, F, I, UP, B, SIM, RUF
- Test files use `*_tests.py` naming, unittest framework (not pytest), `IsolatedAsyncioTestCase` for async

## Architecture

Quent is a fluent chain/pipeline library that transparently bridges sync and async execution. Zero runtime dependencies. Five source files in `quent/`.

### Core Abstractions

**Chain** (`_chain.py`): The main user-facing class. A linked list of `Link` objects forming a sequential pipeline. Methods like `.then()`, `.do()`, `.except_()`, `.finally_()` return `self` for fluent chaining. Execution starts via `.run()` or `__call__()`. `.freeze()` creates an immutable `_FrozenChain` snapshot for reuse.

**Link** (`_core.py`): An atomic operation node. Holds a callable/value (`v`), args/kwargs, and a `next_link` pointer. `_evaluate_value(link, current_value)` dispatches based on whether the link holds a chain, a callable, or a plain value. The `...` (Ellipsis) convention signals "call with no arguments" instead of passing the current value.

**Null** (`_core.py`): Singleton sentinel distinct from `None` to represent "no value provided."

### Sync/Async Bridging

The chain starts executing synchronously in `_run()`. When any link returns an awaitable (detected via `isawaitable()`), execution immediately delegates to `_run_async()` which awaits the coroutine and continues the remaining links in async mode. The caller decides whether to `await` the result. This means a single chain definition works for both sync and async callables without the user writing any async code.

### Operations (`_ops.py`)

Factory functions (`_make_foreach`, `_make_filter`, `_make_gather`, `_make_with`) that return callables fitting the link evaluation protocol. Each implements a three-tier sync/async handoff pattern: sync fast path, mid-operation async transition, and full async path.

### Traceback Enhancement (`_traceback.py`)

Injects chain visualizations into exception tracebacks by replacing the `co_name` of a synthetic code object. The chain structure (with a `<----` marker on the failing link) appears directly in Python's standard traceback output. Patches both `sys.excepthook` and `TracebackException.__init__`.

### Control Flow

`_Return` and `_Break` are internal exception subclasses used for non-local control flow (`.return_()` exits the chain early, `.break_()` exits map/while loops). They propagate through nested chains.

### Key Design Decisions

- Links form a singly-linked list with O(1) append via `current_link` tail pointer
- `_task_registry` holds strong references to fire-and-forget async tasks to prevent GC
- Frozen chains lack the `_is_chain` attribute, so they're treated as regular callables when nested (not as sub-chains)
- `_FrozenChain` is safe for concurrent use; unfrozen `Chain` is not
