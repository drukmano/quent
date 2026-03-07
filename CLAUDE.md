# CLAUDE.md

## Project Identity

quent is a **transparent sync/async bridge**. It is not a collections library, not a functional programming toolkit, not an opinionated framework. It provides the minimum set of pipeline primitives that let developers:

- **Write code once** — a single chain definition works for both sync and async callables. Zero ceremony, zero code duplication.
- **Migrate existing codebases** — unify separate sync and async implementations of the same logic into one. Stop maintaining two versions of every function.
- **Stay out of the way** — quent is unopinionated. It bridges the sync/async divide and gets out of the way. No imposed patterns, paradigms, or abstractions beyond the pipeline itself.

**What quent is NOT:** a collections/iteration library (use itertools, more-itertools), a functional programming framework (use returns, toolz, Expression), or an opinionated architecture (use Effect, returns).

**Design principle:** Every feature must justify itself by solving a real sync/async bridging problem or eliminating genuine code duplication. Features that merely wrap stdlib functionality don't belong.

## Package Metadata

- **Package name:** `quent`
- **Python requirement:** `>=3.10`
- **Zero runtime dependencies** (pure Python)
- **PEP 561 typed package** — `py.typed` marker + `.pyi` stubs for every module
- **Build system:** setuptools >= 70
- **License:** MIT

## Public API

Six exports in `__all__` (defined in `__init__.py`):

| Export | Source | Description |
|--------|--------|-------------|
| `Chain` | `_chain.py` | Main user-facing pipeline class |
| `Null` | `_core.py` | Singleton sentinel for "no value provided" |
| `QuentException` | `_core.py` | Public exception type |
| `__version__` | package metadata | Dynamic version string |
| `enable_traceback_patching()` | `_traceback.py` | Enable traceback enhancement |
| `disable_traceback_patching()` | `_traceback.py` | Disable traceback enhancement |

**Side effect on import:** Importing `quent` triggers `_traceback` module import, which auto-installs exception hook patching.

## Source Modules

Four source files in `quent/`:

| Module | Responsibility |
|--------|---------------|
| `__init__.py` | Public API exports; triggers traceback hook installation on import |
| `_chain.py` | Chain and _FrozenChain classes; execution engine; all user-facing pipeline methods |
| `_core.py` | Link node class, Null sentinel, _ControlFlowSignal exceptions (_Return, _Break), evaluation dispatch (_evaluate_value), fire-and-forget task registry (_ensure_future, _task_registry) |
| `_ops.py` | Operation factory functions (_make_foreach, _make_filter, _make_gather, _make_with, _make_if) and _Generator class for iteration |
| `_traceback.py` | Exception traceback enhancement; chain visualization injection; frame cleaning; sys.excepthook and TracebackException patching |

## Architecture: The Chain

### Chain Class

The main user-facing class. A Chain is a sequential pipeline of Link nodes forming a singly-linked list.

**Slots (10 attributes):**

| Attribute | Purpose |
|-----------|---------|
| `root_link` | Optional first Link holding the chain's root value/callable |
| `first_link` | First non-root link (first `.then()`/`.do()` step) |
| `current_link` | Tail pointer for O(1) append |
| `is_nested` | Whether this chain is embedded inside another chain's Link |
| `on_except_link` | Single exception handler Link (at most one per chain) |
| `on_except_exceptions` | Tuple of exception types the handler catches |
| `on_finally_link` | Single finally handler Link (at most one per chain) |
| `_retry_max_attempts` | Total retry attempts (None if retry not configured) |
| `_retry_on` | Tuple of exception types that trigger retry |
| `_retry_backoff` | Backoff strategy: None (no delay), float (flat delay), or callable(attempt_index) -> float |

**Class attribute:** `_is_chain = True` — Duck-typing marker used by Link to detect Chain instances.

### Chain Methods

#### Pipeline building (all return `self` for fluent chaining)

- **`.then(v, /, *args, **kwargs)`** — Append a step. `v` is evaluated; its result **replaces** the current pipeline value.
- **`.do(fn, /, *args, **kwargs)`** — Append a side-effect step. `fn` must be callable (enforced with TypeError). Its result is **discarded**; the current value passes through unchanged.
- **`.map(fn, /)`** — Apply `fn` to each element of the current iterable. Results are collected into a list that replaces the current value. `fn` must be callable.
- **`.foreach(fn, /)`** — Apply `fn` as a side-effect to each element. The **original** elements are kept (fn's return values discarded).
- **`.filter(fn, /)`** — Keep elements where `fn(element)` returns truthy. Returns a filtered list.
- **`.gather(*fns)`** — Run multiple functions concurrently on the current value. Returns a list of results in the same order as `fns`. If any fn returns an awaitable, all awaitables are gathered via `asyncio.gather`.
- **`.with_(fn, /, *args, **kwargs)`** — Enter current value as context manager, call `fn` with the context value, fn's result **replaces** the current value.
- **`.with_do(fn, /, *args, **kwargs)`** — Same as `.with_()` but fn's result is discarded (side-effect).
- **`.if_(predicate=None, *, then, args=None, kwargs=None)`** — Conditionally apply `then` if predicate is truthy. When `predicate` is `None`, the truthiness of the current pipeline value is used. `then` accepts a callable or a tuple: `(fn, args)`, `(fn, kwargs)`, or `(fn, args, kwargs)`. Separate `args` and `kwargs` keyword params apply when `then` is just a callable. Both predicate and `then` can be sync or async.
- **`.else_(fn, /, *args, **kwargs)`** — Must be called immediately after `.if_()`. Registers the else branch. If the preceding if\_'s predicate was falsy (or the current value was falsy when no predicate was provided), `fn` is evaluated instead. Raises QuentException if not preceded by if\_().
- **`.except_(fn, /, *args, exceptions=None, **kwargs)`** — Register exception handler. Only one per chain. `exceptions` can be a single type or iterable of exception types (defaults to `Exception`). Handler receives the exception object. Its return value becomes the chain's result.
- **`.finally_(fn, /, *args, **kwargs)`** — Register cleanup handler. Only one per chain. Always runs (success or failure). Receives the chain's root value. Its return value is discarded. If it raises, that exception propagates.
- **`.retry(max_attempts=3, on=(Exception,), backoff=None)`** — Configure chain-level retry. Only one per chain. `max_attempts` is total attempts (3 = 1 initial + 2 retries, must be >= 1). `on` filters which exception types trigger retry. `backoff` can be None, a float, or a callable(attempt_index) -> float.

#### Execution

- **`.run(v=Null, /, *args, **kwargs)`** — Execute the chain. Optional `v` is injected as the initial value (overriding root_link's value if both present). Returns the final pipeline value (or a coroutine if async transition occurred).
- **`.__call__`** — Alias for `.run()`.

#### Reuse

- **`.freeze()`** — Create an immutable `_FrozenChain` snapshot. The frozen chain is safe for concurrent use. Importantly, _FrozenChain lacks `_is_chain` attribute, so when nested inside another chain it's treated as a regular callable (not a sub-chain). This means `return_()` and `break_()` inside a frozen sub-chain do NOT propagate to the outer chain.
- **`.decorator()`** — Wrap chain as a function decorator. The decorated function's return value becomes the chain's input. Warning: captures chain by reference — use `.freeze()` first if the chain might be modified later.

#### Iteration

- **`.iterate(fn=None)`** — Return a `_Generator` object that yields each element of the chain's output. Optional `fn` transforms each element. Supports both `for` loops (sync) and `async for` loops (async).
- **`.iterate_do(fn=None)`** — Like `.iterate()` but fn's return values are discarded.

#### Control flow (class methods)

- **`Chain.return_(v=Null, /, *args, **kwargs)`** — Signal early return from the chain. Raises `_Return` internally. Must be used as `return Chain.return_(value)` so the signal propagates.
- **`Chain.break_(v=Null, /, *args, **kwargs)`** — Signal break from iteration (map/foreach/filter). Raises `_Break` internally. Only valid inside iteration operations.

#### Dunder methods

- `__bool__()` — Always returns True
- `__repr__()` — Shows chain structure with method names, e.g., `Chain(fetch).then(validate).do(log)`

### _FrozenChain Class

Immutable snapshot of a Chain.

- **Slots:** `_chain` (reference to underlying Chain)
- **Methods:** `run()`, `__call__` (alias for run), `__bool__` (always True), `__repr__` (shows `Frozen(Chain(...))`)
- **Key property:** No `_is_chain` attribute — treated as regular callable when nested

## Architecture: The Link

### Link Class (in `_core.py`)

An atomic operation node in the chain pipeline.

**Slots (7 attributes):**

| Attribute | Purpose |
|-----------|---------|
| `v` | The callable, value, or nested Chain to evaluate |
| `next_link` | Pointer to the next Link (singly-linked list) |
| `ignore_result` | If True, result is discarded (used by `.do()`, `.foreach()`, `.with_do()`) |
| `args` | Positional arguments tuple (or None) |
| `kwargs` | Keyword arguments dict (or None) |
| `original_value` | Stores the original user-provided value for traceback display (used when `v` is wrapped by an operation factory) |
| `is_chain` | Whether `v` is a Chain (detected via duck-typing: `hasattr(v, '_is_chain')`) |

**Chain detection:** Uses `getattr(v, '_is_chain', False)` to detect Chain instances without circular imports. If detected, sets `v.is_nested = True`.

## Architecture: The Null Sentinel

`Null` is a singleton instance of `_Null`, distinct from `None`. It represents "no value was provided."

**Design:**
- Singleton via `__new__` (only one instance ever exists)
- Immutable: `__copy__` and `__deepcopy__` return self
- Picklable: `__reduce__` returns `'Null'`
- Has `__slots__ = ()` (no instance dict)
- `repr()` returns `'<Null>'`
- Thread-safe singleton (Python's GIL protects `__new__`)

**Usage in the codebase:**
- `Chain.__init__` uses Null as default for "no root value"
- `_run()` initializes `current_value = Null` to detect "value not yet set"
- `_evaluate_value()` checks `current_value is Null` to decide calling convention
- `_ControlFlowSignal.value` defaults to Null for "no return/break value"

**Critical distinction:** `Null` means "no value was provided." `None` is a valid pipeline value that flows through the chain normally. `Chain(None)` creates a chain with root value `None`, while `Chain()` creates a chain with no root value.

## Calling Conventions

This is one of the most important concepts in quent. When a Link's callable is invoked, the calling convention depends on what arguments are present:

1. **First arg is `...` (Ellipsis):** Call `v()` with zero arguments. The Ellipsis is a signal meaning "call with no args" — it prevents the default behavior of passing the current value.
2. **Explicit args/kwargs provided:** Call `v(*args, **kwargs)`. The current value is NOT implicitly passed.
3. **No args, current_value is not Null, and v is callable:** Call `v(current_value)`. This is the default — the current pipeline value flows through as the first argument.
4. **No args, current_value is Null (or v is not callable):** Call `v()` if callable, or return `v` as-is if not callable.

For nested chains (when `link.is_chain` is True): `v._run()` is called directly (bypassing `run()`'s guard) to allow `_Return`/`_Break` to propagate. The Ellipsis convention still applies: if args[0] is `...`, run with no value; otherwise pass current_value.

**The `_evaluate_value(link, current_value)` function in `_core.py` is the central dispatch that implements these conventions.**

## Sync/Async Bridging

This is quent's defining feature. A single chain definition works for both sync and async callables with zero user ceremony.

### How it works

1. Chain execution always starts **synchronously** in `_run()`.
2. After each Link evaluation, the result is checked with `inspect.isawaitable()`.
3. If the result IS awaitable (a coroutine, Task, Future, etc.): execution **immediately delegates** to `_run_async()`, passing the awaitable and current chain state.
4. `_run_async()` is an `async def` that awaits the coroutine, then continues evaluating remaining Links in async mode (awaiting any further awaitables).
5. `_run()` returns either a plain value (if everything was sync) or a coroutine (if async transition occurred). The **caller decides** whether to `await`.

### Three-tier pattern in operations

Each operation factory (`_make_foreach`, `_make_filter`, `_make_with`, `_make_if`) implements three tiers:

- **Sync fast path:** Everything executes synchronously. Most common case.
- **Mid-operation async transition:** Sync iteration/evaluation is underway, but discovers an awaitable partway through. Delegates to an async function that picks up where sync left off.
- **Full async path:** Input is an async iterable or async context manager. Entire operation runs async from the start.

### Fire-and-forget pattern

When a sync chain's except/finally handler returns a coroutine but the chain itself is synchronous, the coroutine is scheduled as a fire-and-forget task via `_ensure_future()`. A warning is emitted. If no event loop is running, a QuentException is raised.

## Task Registry (`_core.py`)

Prevents garbage collection of fire-and-forget async tasks.

- `_task_registry: set[asyncio.Task]` — Holds strong references to in-flight tasks
- `_task_registry_lock: threading.Lock` — Thread-safe access
- `_ensure_future(coro)` — Creates task via `asyncio.create_task()`, adds to registry, registers done callback for auto-removal
- On Python 3.14+, uses `asyncio.create_task(eager_start=True)` for optimized task startup

## Control Flow Signals

`_Return` and `_Break` are internal exception subclasses of `_ControlFlowSignal` (which extends `Exception`). They implement non-local control flow within chains.

### _ControlFlowSignal base

- **Slots:** `value`, `args_`, `kwargs_`
- Stores the optional return/break value and its arguments
- Never displayed to users (no standard Exception args)

### _Return

Raised by `Chain.return_()`. Exits the chain early with an optional value. Propagates through nested chains (unless frozen).

### _Break

Raised by `Chain.break_()`. Exits map/foreach/filter iteration with an optional value. Only valid inside iteration operations. Using `break_()` outside iteration raises QuentException.

### Invariants

- The public `run()` method catches any escaped `_ControlFlowSignal` and raises `QuentException` — these signals must never leak to user code.
- Control flow signals are NEVER caught by `except_()` handlers.
- Using `return_()` or `break_()` inside an `except_()` or `finally_()` handler raises QuentException.
- Control flow signals are NEVER retried (retry catches `_ControlFlowSignal` separately and re-raises immediately).

## Retry Mechanism

Retry operates at the **entire chain level**, not per-link.

### Behavior

1. On retryable exception (matching `retry_on` types), the entire chain restarts from scratch — all state (root_value, current_value) is reset.
2. `except_()` and `finally_()` handlers fire only AFTER all retry attempts are exhausted.
3. `_ControlFlowSignal` exceptions are never retried — they propagate immediately.
4. Backoff: `_get_retry_delay(attempt)` returns 0.0 (if None), a flat float, or calls `backoff(attempt_index)`. Sync path uses `time.sleep()`, async path uses `asyncio.sleep()`.
5. When sync `_run()` discovers an awaitable mid-retry, it delegates to `_run_async()` with the current attempt index, which picks up the retry loop from that point.

**Idempotency assumption:** Retry restarts the entire chain, so operations should be idempotent or fault-tolerant. No automatic rollback.

## Operations (`_ops.py`)

### _make_foreach(link, ignore_result)

Creates the callable for `.map()` and `.foreach()`.
- `ignore_result=False` — map semantics (collect transformed results)
- `ignore_result=True` — foreach semantics (collect original items, discard fn results)
- Detects `__aiter__` for async iterables
- Uses manual `while`/`next()` loop (not `for`) to enable mid-iteration async handoff
- Handles `_Break` to stop iteration early and return partial results
- Attaches metadata: `_quent_op = 'map'`, `_ignore_result`

### _make_filter(link)

Creates the callable for `.filter()`.
- Keeps elements where `fn(element)` returns truthy
- Three-tier sync/async pattern
- Attaches metadata: `_quent_op = 'filter'`

### _make_gather(fns)

Creates the callable for `.gather()`.
- Calls each fn with current_value, collects results
- If any result is awaitable: uses `asyncio.gather()` to await all concurrently
- Attaches exception metadata (`__quent_gather_index__`, `__quent_gather_fn__`) for traceback display
- Attaches metadata: `_quent_op = 'gather'`, `_fns`

### _make_with(link, ignore_result)

Creates the callable for `.with_()` and `.with_do()`.
- Detects `__aenter__` for async context managers
- Properly handles `__exit__` return value (True suppresses exception)
- Handles awaitable `__exit__` return values
- Attaches metadata: `_quent_op = 'with'`, `_ignore_result`

### _make_if(predicate_link, fn_link)

Creates the callable for `.if_()`.
- `predicate_link` can be `None` — when None, the truthiness of the current pipeline value is used as the predicate (no function call)
- When `predicate_link` is provided, evaluates predicate with current_value (or no args if Null)
- Truthy — evaluate fn_link
- Falsy + `_else_link` set — evaluate else branch
- Falsy + no else — pass current value through
- Attaches metadata: `_quent_op = 'if'`, `_else_link` (initially None, set by `else_()`), `_predicate_link`

### _Generator Class

Wraps chain output as a dual sync/async iterable.
- `__iter__()` returns a sync generator via `_sync_generator()`
- `__aiter__()` returns an async generator via `_async_generator()`
- `__call__(v, *args, **kwargs)` returns a new _Generator with updated run args (makes it reusable with different inputs)
- Handles `_Break` for early termination
- Raises TypeError if fn returns coroutine in sync iteration mode

## Traceback Enhancement (`_traceback.py`)

Quent injects chain visualizations into exception tracebacks so developers can see exactly which step in a chain failed.

### Mechanism

1. When an exception occurs during chain execution, `_modify_traceback()` is called.
2. It stamps the exception with `__quent_source_link__` (the Link that caused the error).
3. It builds a string visualization of the chain via `_stringify_chain()`, marking the failing link with `<----`.
4. A synthetic `<quent>` frame is injected into the traceback using a code object compilation hack (`compile('raise __exc__', '<quent>', 'exec')`) where the chain visualization is set as `co_name`/`co_qualname`.
5. Internal quent frames are cleaned from the traceback (frames from files within the quent package directory).
6. Chained exceptions (`__cause__`, `__context__`, ExceptionGroup) are also cleaned.

### Hook patching

- `sys.excepthook` is replaced with `_quent_excepthook` that cleans internal frames for exceptions with `__quent__` flag.
- `traceback.TracebackException.__init__` is patched similarly for logging/formatting contexts.
- Auto-enabled on package import.
- Can be toggled with `enable_traceback_patching()` / `disable_traceback_patching()`.

### Chain visualization format

```
Chain(fetch_data).then(validate).then(transform) <----.do(log)
```

The `<----` marker points to the Link that raised the exception. Nested chains are shown indented.

### Operation metadata on callables

Operations attach attributes (`_quent_op`, `_fns`, `_else_link`, `_predicate_link`, `_ignore_result`) to their callable wrappers. The traceback formatter reads these to reconstruct human-readable method names.

## Key Design Decisions and Invariants

1. **Singly-linked list with O(1) append:** Links form a linked list. `current_link` is a tail pointer for constant-time append. No array resizing.

2. **Duck typing for chain detection:** `_is_chain = True` class attribute on Chain. Checked via `getattr()` to avoid import cycles. _FrozenChain intentionally lacks this attribute.

3. **Single except/finally/retry per chain:** Enforced at registration time. Simplifies execution model. Raises QuentException on duplicates.

4. **except_ catches Exception by default, not BaseException:** Follows Python convention. BaseException (KeyboardInterrupt, SystemExit) is not caught unless explicitly specified.

5. **finally_ receives root_value, not current_value:** The finally handler always gets the chain's original input, not the intermediate pipeline value. Its return value is always discarded.

6. **except_ handler return value replaces chain result:** If the handler returns successfully, that value becomes the chain's output.

7. **do() enforces callable:** Unlike `then()` which accepts any value, `do()` raises TypeError for non-callables. This prevents bugs where a literal value is accidentally used as a side-effect.

8. **Frozen chains are opaque boundaries:** When a frozen chain is nested inside another chain, it's treated as a regular callable. `_Return` and `_Break` from within a frozen chain are caught by the frozen chain's own `run()` guard and converted to QuentException — they do NOT propagate to the outer chain.

9. **Operation metadata via function attributes:** Rather than creating wrapper classes, operation factories attach metadata (`_quent_op`, `_ignore_result`, `_else_link`, etc.) directly to the callable. This keeps the link evaluation path simple.

10. **Traceback injection via code object hack:** The chain visualization is injected as `co_name` of a synthetic code object, making it appear in standard Python traceback output without modifying the traceback module itself.

11. **Fire-and-forget with strong references:** `_task_registry` prevents GC of scheduled tasks. Auto-cleanup via done callbacks.

12. **Monad law compliance:** Chain satisfies left identity, right identity, and associativity for both sync and async callables.

13. **Slots on every class:** No instance `__dict__` anywhere in the codebase.

14. **`from __future__ import annotations`** in every module for PEP 604 union types.

## Concurrency Safety

| Component | Thread-safe? | Notes |
|-----------|-------------|-------|
| Chain (unfrozen) | No | Mutable state (current_link, link pointers) |
| _FrozenChain | Yes | Immutable snapshot; underlying Chain is only read during execution |
| Null | Yes | Python GIL protects singleton `__new__` |
| _task_registry | Yes | Protected by `threading.Lock` |
| Link | No (writes) / Yes (reads) | Not designed for concurrent modification, but safe for concurrent reads (used in frozen chains) |

## Code Style and Conventions

- **2-space indentation** (enforced by ruff)
- **120-character line length**
- **Single quotes** for strings
- **Ruff rules:** E, W, F, I, UP, B, SIM, RUF (with ignores: E741, B905, SIM108, UP007)
- **Mypy:** Python 3.10 target, strict mode (warn_return_any, check_untyped_defs, disallow_incomplete_defs)
- **Type ignore comments** used for legitimate type violations

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

## Testing Conventions

- **Framework:** unittest (not pytest). `IsolatedAsyncioTestCase` for async tests.
- **File naming:** `*_tests.py`
- **Test discovery:** `python -m unittest discover` from project root
- **Coverage floor:** 80% (configured in pyproject.toml)
- **Helpers:** `tests/helpers.py` contains shared fixtures — sync/async callable pairs, context manager fixtures (sync, async, suppressing, raising variants), async iterables, custom exceptions, stateful callables, tracking utilities
- **Patterns used:** Dense value subtests, combinatorial matrix tests, property invariant tests, regression tests for specific bugs, exhaustive operation coverage, memory/reference tests, concurrent safety tests
