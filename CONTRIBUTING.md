# Contributing to Quent

Thank you for your interest in contributing to Quent! This guide covers everything you need to get started.

## Prerequisites

- **Python 3.10+**
- **setuptools** and **build** (for packaging)
- **ruff** (formatting and linting)
- **mypy** (type checking)

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/drukmano/quent.git
cd quent
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install development dependencies:

```bash
pip install build coverage twine setuptools ruff mypy
```

4. Install the package in editable mode:

```bash
pip install -e .
```

5. Verify installation:

```bash
python -c 'import quent; print(quent.Chain)'
```

You should now be able to import `quent` and run the test suite.

## Project Structure

```
quent/
  __init__.py              # Public API exports (Chain, Null, QuentException)
  _chain.py                # Chain and _FrozenChain classes (execution engine)
  _core.py                 # Link node, evaluation dispatch, control flow signals
  _ops.py                  # foreach, filter, gather, with_ operation factories
  _traceback.py            # Traceback rewriting and chain visualization
scripts/
  compile.sh               # Byte-compile (syntax check)
  run_tests.sh             # Full QA pipeline (format, lint, typecheck, test)
  build.sh                 # Build distribution packages
  distribute.sh            # Upload to PyPI
tests/
  core_tests.py            # Core functionality tests
  traceback_tests.py       # Traceback and exception tests
  helpers.py               # Test utilities
```

## Development Workflow

The development loop is:

1. **Edit** the `.py` files in `quent/`
2. **Run** `bash scripts/run_tests.sh` (runs ruff format, ruff check, mypy, tests with coverage)
3. Or run individual steps:

```bash
ruff format quent/
ruff check --fix quent/
mypy quent/
python -m unittest tests.core_tests
```

### Running Tests

```bash
bash scripts/run_tests.sh            # Full QA pipeline
python -m unittest tests.core_tests  # Single test file
```

The test suite uses `unittest.IsolatedAsyncioTestCase` for async test support.

## Code Style

- **Indentation:** 2 spaces
- **Line length:** 120
- **Quotes:** Single quotes
- **Ruff rules:** E, W, F, I, UP, B, SIM, RUF
- Follow existing patterns in the codebase
- Keep changes minimal and focused

## Build and Distribution

To build distributable packages:

```bash
bash scripts/build.sh
```

This cleans previous build artifacts and runs `python3 -m build`, which uses the `pyproject.toml` configuration with the setuptools backend.

## Key Architecture Notes

- **Link-based chain evaluation:** Chains are composed of `Link` objects forming a singly-linked list. The `_evaluate_value` function in `_core.py` is the central dispatch for processing values through the chain. It dispatches based on whether the link holds a chain, uses `...` (Ellipsis) for no-arg calls, has explicit args/kwargs, is callable, or is a plain value.
- **Transparent async handling:** The chain starts executing synchronously in `_run()`. When any link returns an awaitable (detected via `inspect.isawaitable()`), execution immediately delegates to `_run_async()` which awaits the coroutine and continues the remaining links in async mode. There is no upfront sync/async decision.
- **Chain vs FrozenChain:** `Chain` (in `_chain.py`) is the main mutable pipeline. `_FrozenChain` wraps a Chain snapshot for safe reuse. `_FrozenChain` lacks the `_is_chain` attribute, so when nested inside another chain it is treated as a regular callable rather than a sub-chain.
- **Fire-and-forget tasks:** `_task_registry` (in `_core.py`) holds strong references to async tasks created via `_ensure_future()`, preventing garbage collection before completion.
- **Traceback injection:** `_traceback.py` injects chain visualizations into exception tracebacks by replacing the `co_name` of a synthetic code object. The chain structure (with a `<----` marker on the failing link) appears directly in Python's standard traceback output. It patches both `sys.excepthook` and `TracebackException.__init__`.

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes following the workflow above
3. Ensure all tests pass
4. Submit a Pull Request with a clear description of the changes

## Code Conventions

This section documents the internal conventions used throughout the codebase. Understanding these is essential for working on the core implementation.

### Link Slots

Each `Link` object (defined in `_core.py`) carries the following attributes:

| Attribute | Description |
|-----------|-------------|
| `v` | The executable value (callable, literal, or Chain) |
| `next_link` | Pointer to the next Link in the chain |
| `ignore_result` | If True, result is discarded after evaluation (used by `.do()`) |
| `args` | Positional call arguments |
| `kwargs` | Keyword call arguments |
| `is_chain` | True if `v` has the `_is_chain` attribute (duck-typed Chain detection) |
| `original_value` | Original value before wrapping (preserved for traceback display) |

### Sentinel Values

Two sentinel values are used to distinguish "no value provided" from `None` or other falsy values:

**`Null`** (instance of `_Null`)

`Null` is the primary "no value" sentinel. It is a singleton created at module level in `_core.py`. Where it appears:
- `Chain.__init__` defaults its root value parameter to `Null` (meaning "no root value")
- `_run` initializes `current_value` to `Null`
- At the end of `_run`, if `current_value is Null` the chain returns `None`
- `_Return.value` and `_Break.value` default to `Null` to mean "no explicit return/break value"

`Null` is exported publicly from `quent` for cases where user code needs to check for it.

**`...` (Ellipsis)**

The built-in `Ellipsis` (`...`) is used as a "no arguments" marker when the user explicitly wants to call a function with zero arguments, overriding the default behavior of passing the current value. For example, `chain.then(fn, ...)` calls `fn()` instead of `fn(current_value)`.

### Async Detection

The library uses `inspect.isawaitable()` to detect async results at runtime. The sync-to-async transition works as follows:

1. `Chain._run` executes links synchronously in a loop
2. After each `_evaluate_value` call, the result is checked with `isawaitable(result)`
3. If an awaitable is detected, `_run` immediately returns `self._run_async(...)` -- the awaitable result becomes the argument to the async continuation
4. `_run_async` awaits the result, then continues the remaining links asynchronously (awaiting any further awaitables inline)

This design means the chain starts synchronous and only becomes async at the exact point where an awaitable value is first encountered.

### Internal Control Flow Signals

Control flow within chains uses exception-based signals (defined in `_core.py`):

| Type | Purpose |
|------|---------|
| `_Return` | Raised by `Chain.return_()` to exit the entire chain with a value |
| `_Break` | Raised by `Chain.break_()` to exit a `foreach` loop |

Both carry a `value` field that defaults to `Null` to mean "no explicit value." These signals propagate through nested chains.

**Important:** Users must write `return Chain.break_()` / `return Chain.return_()` (not bare calls) so the signal propagates through return values.
