# Contributing to Quent

Thank you for your interest in contributing to Quent! This guide covers everything you need to get started.

## Prerequisites

- **Python 3.14+**
- **Cython 3.0+** (build-time dependency)
- **A C compiler** (gcc on Linux, clang on macOS, MSVC on Windows)
- **setuptools** and **build** (for packaging)

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
pip install -r dev_requirements.txt
```

4. Compile the Cython extensions:

```bash
bash scripts/compile.sh
```

You should now be able to import `quent` and run the test suite.

## Project Structure

```
quent/
  __init__.py              # Public API exports
  _internal.py             # Internal utilities
  quent.pyx                # Hub: imports, sentinels, coroutine setup, include directives
  quent.pxd                # Forward declarations for all cdef types
  quent.pyi                # Python type stubs for IDE support
  _link.pxi               # Link node, evaluation dispatch, clone utilities
  _operators.pxi           # Comparison, type-check, sleep, negation operators
  _control_flow.pxi        # Signals, conditionals, loops, context managers, generators
  _iteration.pxi           # foreach, filter, reduce, gather collection operations
  _helpers.pxi             # Async utilities, exception handling, chain stringification
  _chain.pxi               # Core Chain class (construction, execution, API methods)
  _variants.pxi            # Cascade, ChainAttr, CascadeAttr, FrozenChain, run
scripts/
  compile.sh               # Compile Cython -> C -> shared object
  tests_compile.sh         # Compile with profiling/line tracing for coverage
  run_tests.sh             # Run test suite with coverage
  build.sh                 # Build distribution packages
tests/
  main_tests.py            # Primary test suite
  except_tests.py          # Exception handling tests
  exception_clean_tests.py # Stack trace cleaning tests
  utils.py                 # Test utilities
  main.py                  # Manual test runner
```

### Cython File Types

| Extension | Purpose |
|-----------|---------|
| `.pyx` | Cython source code (compiled to C, then to a shared object) |
| `.pxd` | Cython declaration files (C-level type declarations, like C header files) |
| `.pxi` | Cython include files (`include`d by the `.pyx` hub; contain the actual implementation) |
| `.pyi` | Python type stub files (for IDE autocompletion and type checkers) |

When you modify a `.pyx` or `.pxi` file, the corresponding `.pxd` file may also need updating if you change function signatures or cdef declarations.

## Development Workflow

The core development loop for Cython code is:

1. **Edit** the `.pyx` or `.pxi` file(s)
2. **Compile** the extensions
3. **Test** your changes
4. Repeat

### Compiling

After modifying any `.pyx` or `.pxi` file, you must recompile:

```bash
bash scripts/compile.sh
```

This script:
- Cleans old `.c`, `.so`/`.cpython*`, and `.html` artifacts
- Runs `cython_setup.py` to transpile `.pyx` to `.c` and compile to shared objects
- Applies optimization flags (`-O3`) and Cython compiler directives (bounds checking disabled, etc.)

### Running Tests

Compile with test instrumentation, then run the suite:

```bash
# Compile with profiling/line tracing enabled
bash scripts/tests_compile.sh

# Run tests with coverage
bash scripts/run_tests.sh
```

To run a specific test file:

```bash
python -m unittest tests.main_tests
```

The test suite uses `unittest.IsolatedAsyncioTestCase` for async test support.

**Important:** Always compile with `tests_compile.sh` before running tests with coverage. The test compilation enables the `CYTHON_TRACE_NOGIL` macro, which is required for accurate coverage reporting on Cython code.

## Code Style

- **Indentation:** 2 spaces (both Python and Cython files)
- Follow existing patterns in the codebase
- Keep changes minimal and focused

## Build and Distribution

To build distributable packages:

```bash
bash scripts/build.sh
```

This cleans previous build artifacts and runs `python3 -m build`, which uses the `pyproject.toml` configuration (setuptools backend with Cython build requirement).

## Key Architecture Notes

- **Link-based chain evaluation:** Chains are internally composed of `Link` objects representing individual operations. The `evaluate_value` function in `_link.pxi` is the central dispatch for processing values through the chain.
- **Transparent async handling:** The helpers module (`_helpers.pxi`) provides async utilities for coroutine and future detection at runtime. When async values are encountered, they are automatically wrapped in `asyncio.Task` instances.
- **Chain vs Cascade:** `Chain` (in `_chain.pxi`) passes the result of each operation to the next. `Cascade` (in `_variants.pxi`) passes the root value to every operation.
- **Compiler directives:** Production builds disable `boundscheck` and `wraparound` for performance. Be careful when modifying code that indexes into sequences.
- **Hub architecture:** `quent.pyx` is a thin hub file that defines shared imports, sentinels, and coroutine detection, then `include`s the `.pxi` implementation files in dependency order. All implementation code lives in the `.pxi` files.

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes following the workflow above
3. Ensure all tests pass
4. Submit a Pull Request with a clear description of the changes

## Code Conventions

This section documents the internal conventions used throughout the Cython codebase. Understanding these is essential for working on the core implementation.

### Variable Naming

The codebase uses short abbreviated variable names for values flowing through chains. These appear most prominently in `Chain._run` and `Chain._run_async` (in `_chain.pxi`) and `evaluate_value` (in `_link.pxi`):

| Abbreviation | Meaning | Description |
|--------------|---------|-------------|
| `current_value` | current value | The value currently being passed through the chain. Updated after each link evaluates. |
| `root_value` | root value | The first value produced by the chain (the root link's result). In a `Cascade`, every link receives this. |
| `previous_value` | previous value | Saved copy of `current_value` before evaluating a link with `ignore_result=True`, so `current_value` can be restored afterward. |
| `v` | value | General-purpose value. Used as a link's stored callable/literal (`link.v`) and as a local in `evaluate_value`. |
| `fallback` | fallback value | Used in `handle_break_exc` (in `_control_flow.pxi`) as the fallback value when a `_Break` carries no explicit value. |
| `original_value` | original value | The original user-supplied value stored on a `Link` for display purposes (e.g. in `stringify_chain`). Preserved even when `link.v` is replaced by an internal wrapper. |
| `el` | element | An element during iteration in `foreach`, `filter`, `reduce`, etc. (in `_iteration.pxi`). |
| `fn` | function | A callable parameter, typically the user-provided function in operations like `foreach`, `with_`, and `iterate`. |
| `obj` | object | Parameter in `get_obj_name` (in `_helpers.pxi`) representing the object whose name is being resolved. |
| `output` | output string | Local variable in `stringify_chain` and `format_link` (in `_helpers.pxi`) accumulating the human-readable chain representation. |
| `outer_link` | outer link | Local variable in `format_link` (in `_helpers.pxi`) holding the original link before descending into nested structures. |

### Sentinel Values

Two sentinel values are used to distinguish "no value provided" from `None` or other falsy values:

**`Null`** (instance of `_Null`)

`Null` is the primary "no value" sentinel. It is a singleton created at module level in `quent.pyx`:

```cython
cdef class _Null:
  def __repr__(self):
    return '<Null>'

cdef _Null Null = _Null()
```

Where it appears:
- `Chain.__init__` defaults its root value parameter to `Null` (meaning "no root value")
- `_run` initializes `current_value`, `root_value`, and `previous_value` to `Null`
- At the end of `_run`, if `current_value is Null` the chain returns `None` (converting the internal sentinel to a Python-friendly value)
- `_Return.value` and `_Break.value` default to `Null` to mean "no explicit return/break value"

`Null` is exposed publicly as `Chain.null()` and as `PyNull` for cases where user code needs to check for it.

**`...` (Ellipsis)**

The built-in `Ellipsis` (`...`) is used as a "no arguments" marker when the user explicitly wants to call a function with zero arguments, overriding the default behavior of passing the current value. When a `Link` is created with `args=(Ellipsis,)`, it sets `eval_code = EVAL_CALL_WITHOUT_ARGS`:

```cython
if args[0] is ...:
  self.eval_code = EVAL_CALL_WITHOUT_ARGS
```

This means "call the function with no arguments at all" rather than "call it with the current value." For example, `chain.then(fn, ...)` calls `fn()` instead of `fn(current_value)`.

### Link Evaluation Codes (`eval_code`)

Each `Link` has an `eval_code` integer that determines how `evaluate_value` dispatches the call. The code is set once during `Link.__init__` based on the arguments provided:

| Code | Value | Meaning | Set when |
|------|-------|---------|----------|
| `EVAL_CALL_WITH_EXPLICIT_ARGS` | 1001 | Call with explicit positional and keyword args: `v(*args, **kwargs)` | User provided args (not Ellipsis) or kwargs |
| `EVAL_CALL_WITHOUT_ARGS` | 1002 | Call with no arguments: `v()` | User passed `...` as first arg, or `is_fattr=True`, or chain-with-kwargs-only |
| `EVAL_CALL_WITH_CURRENT_VALUE` | 1003 | Call with current value: `v(current_value)` or `v()` if `current_value is Null` | Value is callable and no explicit args/kwargs given |
| `EVAL_RETURN_AS_IS` | 1004 | Return the value as-is without calling: `return v` | Value is not callable and `allow_literal=True` |
| `EVAL_GET_ATTRIBUTE` | 1005 | Attribute access: `getattr(current_value, v)` | `is_attr=True` and not `is_fattr` |

The `evaluate_value` function (in `_link.pxi`) uses these codes to choose the invocation strategy. It also has a fast path: when `eval_code == EVAL_CALL_WITH_CURRENT_VALUE` and the link is not a chain or attribute, it directly calls `link.v(current_value)` without further branching.

### Chain vs Cascade (`is_with_root`)

The `is_with_root` flag on a `Link` controls which value the link receives as input:

- **`is_with_root = False`** (default in `Chain`): the link receives `current_value` (the result of the previous link)
- **`is_with_root = True`** (forced for all links in `Cascade`): the link receives `root_value`

This is the sole behavioral difference between `Chain` and `Cascade`:

```cython
# In Chain._then:
if self.is_cascade:
  link.is_with_root = True
```

In `Chain`, individual links can opt into root-value behavior via the `.root()` and `.root_do()` methods, which set `is_with_root=True` on that specific link.

At the end of chain execution, `Cascade` replaces the final `current_value` with `root_value`:

```cython
if self.is_cascade:
  current_value = root_value
```

### Async Detection and Sync-to-Async Transition

The library uses a custom `iscoro` function (defined in `quent.pxd` as an inline C function) to detect coroutines:

```cython
cdef inline bint iscoro(object obj) noexcept:
  return type(obj) is _PyCoroType or type(obj) is _CyCoroType
```

This checks for both standard Python coroutines (`types.CoroutineType`) and Cython-generated coroutines (whose type is captured at import time by creating and inspecting a dummy coroutine). This is faster than `inspect.isawaitable` because it performs exact type identity checks rather than `isinstance` lookups.

The sync-to-async transition works as follows:

1. `Chain._run` executes links synchronously in a `while` loop
2. After each `evaluate_value` call, the result is checked with `iscoro(current_value)`
3. If a coroutine is detected, `_run` immediately returns `self._run_async(link, current_value, root_value, previous_value, ...)` -- the coroutine result becomes the argument to the async continuation
4. `_run_async` awaits the coroutine, then continues the remaining links asynchronously (awaiting any further coroutines inline)
5. If `_autorun` is set, the returned coroutine is wrapped in `ensure_future()` to schedule it immediately

This design means the chain starts synchronous and only becomes async at the exact point where an awaitable value is first encountered -- there is no upfront async/sync decision.

### Link Flags Reference

Each `Link` carries several boolean flags that control its behavior during evaluation:

| Flag | Description |
|------|-------------|
| `is_with_root` | Receive root value instead of current value (see Chain vs Cascade above) |
| `ignore_result` | Evaluate the link but discard its result; restore `current_value` to its previous value (`previous_value`) |
| `is_attr` | The link's `v` is an attribute name string; use `getattr(current_value, v)` instead of calling `v` |
| `is_fattr` | The link is a callable attribute (method call); combines `is_attr` with `EVAL_CALL_WITHOUT_ARGS` or `EVAL_CALL_WITH_EXPLICIT_ARGS` |
| `is_chain` | The link's `v` is a nested `Chain` instance; dispatch to its `_run` method |
| `is_exception_handler` | This link is an `except_` handler; skip during normal iteration, used only in `_handle_exception` |
| `reraise` | On an exception handler link: if `True`, re-raise the exception after the handler runs |

### Internal Control Flow Signals

Control flow within chains uses exception-based signals (defined in `_control_flow.pxi`). Both are subclasses of `_InternalQuentException`, which itself inherits from `Exception`:

| Type | Purpose |
|------|---------|
| `_Return` | Raised by `Chain.return_()` to exit the entire chain with a value |
| `_Break` | Raised by `Chain.break_()` to exit a `while_true` or `foreach` loop |

Both `_Return` and `_Break` carry `value`, `args_`, and `kwargs_` fields so the return/break value can be lazily evaluated through `_eval_signal_value` (in `_operators.pxi`) if needed. The `value` field defaults to `Null` to mean "no explicit value."

These signals are caught by the chain's execution loop and resolved via `handle_return_exc` (in `_helpers.pxi`) and `handle_break_exc` (in `_control_flow.pxi`).

**Important:** Users must write `return Chain.break_()` / `return Chain.return_()` (not bare calls) so the signal propagates through return values.

### Other Constants

| Name | Value | Purpose |
|------|-------|---------|
| `EMPTY_TUPLE` | `()` | Shared empty tuple to avoid repeated allocations |
| `EMPTY_DICT` | `{}` | Shared empty dict to avoid repeated allocations |
| `__QUENT_INTERNAL__` | `object()` | Singleton sentinel in `_internal.py`, set on globals of internal modules to identify quent frames during traceback cleaning |
