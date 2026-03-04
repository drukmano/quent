# Performance Optimizations in Quent

Quent is a Cython-compiled chain interface library. Every design decision in the implementation prioritizes minimizing overhead in the hot paths — link construction, value evaluation, and async detection. This document catalogs all 41 deliberate performance optimizations across the codebase, organized by category.

---

## Section 1: Cython Class Decorators

### #1 — `@cython.final` on all cdef classes

**What it does:** Marks every `cdef` class as non-subclassable — specifically `_Null`, `_ExecCtx`, `Link`, `_With`, `_Generator`, `_Foreach`, `_Filter`, `_Gather`, `_ChainCallWrapper`, `_DescriptorWrapper`, `Chain`, and `_FrozenChain`.

**Why it is faster:** When Cython knows a class cannot be subclassed, it can emit direct C function calls instead of vtable-dispatched virtual calls. Virtual dispatch requires dereferencing a pointer to a function table and then jumping through it; direct calls are resolved at compile time.

**Where it is located:** `quent.pyx`, `quent.pxd`, and all `.pxi` include files — on every `cdef class` declaration.

**Without this:** Cython must use slower indirect method dispatch to allow for potential subclasses, even though no subclasses will ever exist in practice.

---

### #2 — `@cython.freelist(N)` on frequently allocated classes

**What it does:** Pre-allocates a pool of previously freed objects so that `__new__` can return an existing allocation immediately without calling the system allocator. Freelist sizes: `Link(64)`, `Chain(32)`, `_ExecCtx(16)`, `_DescriptorWrapper(8)`, `_Foreach(8)`, `_Filter(8)`, `_FrozenChain(4)`, `_With(4)`, `_Gather(4)`, `_Generator(4)`.

**Why it is faster:** `PyObject_Malloc` and `PyObject_Free` are non-trivial operations involving heap management. For classes that are constructed and destroyed frequently (especially `Link` during chain construction), avoiding the allocator entirely on the hot path has measurable impact.

**Where it is located:** Decorator annotations on each `cdef class` declaration in `quent.pyx` and `quent.pxd`.

**Without this:** Every allocation goes through `PyObject_Malloc` and `PyObject_Free`, adding allocator overhead to each chain link construction and each `Chain` or `_ExecCtx` instantiation.

---

### #3 — `@cython.no_gc` on `_Null`

**What it does:** Disables cyclic garbage collector tracking for the `_Null` sentinel object.

**Why it is faster:** `_Null` holds no references to other Python objects and therefore cannot form reference cycles. GC tracking is pure overhead — the object is registered on creation, visited during each GC collection pass, and deregistered on deletion.

**Where it is located:** `quent.pyx` line 16.

**Without this:** The GC tracks the object, wasting CPU time on every collection cycle by checking an object that provably cannot be part of a cycle.

---

## Section 2: Inline Functions and Type Checks

### #4 — `cdef inline iscoro()` — inline C-level coroutine type check

**What it does:** Performs two pointer comparisons — `type(obj) is _PyCoroType` and `type(obj) is _CyCoroType` — inlined directly at every call site rather than dispatched as a function call. The `noexcept` qualifier eliminates Cython's automatic exception-state checking overhead after each call.

**Why it is faster:** This check sits in every evaluation loop iteration. An inline check is expanded directly into the caller's machine code, avoiding call setup, stack frame allocation, and return overhead. The `noexcept` annotation removes the generated `if (PyErr_Occurred())` check that Cython would otherwise emit after every function call.

**Where it is located:** `quent.pxd` lines 25-26. Called in every chain evaluation hot loop.

**Without this:** Using `asyncio.iscoroutine()` or `inspect.iscoroutine()` would involve Python function calls with additional overhead including dict lookup, frame creation, and multiple internal checks.

---

### #5 — Hot/warm/cold field ordering on `Link`

**What it does:** Fields on the `Link` struct are ordered by access frequency. Hot fields (`v`, `next_link`, `eval_code`, `is_chain`, `ignore_result`) are declared first; warm fields (`args`, `kwargs`) come next; cold fields (`original_value`, `temp_args`) are last.

**Why it is faster:** CPU caches operate on cache lines (typically 64 bytes). Placing the most frequently accessed fields at the start of the struct maximizes the probability that all hot fields reside in the same cache line, reducing cache misses during evaluation loop iteration.

**Where it is located:** `quent.pxd` lines 37-47.

**Without this:** Arbitrary field ordering scatters frequently accessed data across cache lines, increasing cache miss rates in the evaluation hot loop.

---

## Section 3: Pre-allocated Constants

### #6 — Pre-allocated `EMPTY_TUPLE` and `EMPTY_DICT`

**What it does:** Allocates `cdef tuple EMPTY_TUPLE = ()` and `cdef dict EMPTY_DICT = {}` once at module initialization and reuses these objects throughout the library.

**Why it is faster:** While CPython interns the empty tuple, it does not intern empty dicts. Every `{}` literal in a function body creates a new dict object. Using a single pre-allocated instance eliminates repeated allocation and reference counting for the common case where no args or kwargs are provided.

**Where it is located:** `quent.pyx` lines 26-27. Used throughout: `Link` construction, `_determine_eval_code`, `foreach()`, `filter_()`, `gather_()`, `_eval_signal_value`, `_DEFAULT_RUN_ARGS`.

**Without this:** Each use of `()` and `{}` literals potentially allocates new objects (especially for dicts), adding allocation and deallocation overhead throughout the hot path.

---

## Section 4: Cython Compiler Directives

### #7 — `binding=False`

**What it does:** Generates lighter-weight `CFunction` wrappers for Cython methods instead of full Python function objects.

**Why it is faster:** Full Python function objects carry significant overhead: a `__code__` object, `__globals__` dict reference, closure cells, default argument storage, and introspection metadata. `CFunction` wrappers are leaner and dispatch faster.

**Where it is located:** `cython_setup.py` line 15.

**Without this:** `binding=True` creates full Python function objects with introspection support, adding memory and dispatch overhead for every method call.

---

### #8 — `always_allow_keywords=False`

**What it does:** Allows Cython to use `METH_NOARGS` and `METH_O` CPython calling conventions for functions with zero or one positional argument respectively.

**Why it is faster:** `METH_NOARGS` and `METH_O` are the fastest calling conventions in CPython — they bypass keyword argument parsing entirely. Functions accepting keywords must go through `PyCFunction` with `METH_VARARGS | METH_KEYWORDS`, which involves parsing a `kwargs` dict even when none is provided.

**Where it is located:** `cython_setup.py` line 21.

**Without this:** All functions accept keyword arguments, requiring kwargs dict parsing machinery on every call regardless of whether keywords are actually used.

---

### #9 — `boundscheck=False`, `wraparound=False`

**What it does:** Disables array and buffer bounds checking (`boundscheck=False`) and negative-index wraparound checking (`wraparound=False`) for all array and memoryview accesses.

**Why it is faster:** Without these directives, every array access generates an `if (index < 0 || index >= length)` check and, for wraparound, an additional adjustment for negative indices. Eliminating these removes conditional branches from tight loops.

**Where it is located:** `cython_setup.py` lines 16-17.

**Without this:** Every array access incurs bounds and wraparound checks, adding conditional branches and potential exception paths to all indexed operations.

---

### #10 — `initializedcheck=False`

**What it does:** Disables runtime checks that verify memoryview and buffer variables have been assigned before use.

**Why it is faster:** These checks generate `if (view.memview == NULL)` guards before every memoryview access. Removing them eliminates unnecessary branches in code where initialization is guaranteed by construction.

**Where it is located:** `cython_setup.py` line 23.

**Without this:** Runtime null-pointer checks are emitted before every memoryview operation, adding branch overhead even when the check is provably redundant.

---

### #11 — `cdivision=True`

**What it does:** Uses C-style integer division semantics rather than Python's division semantics.

**Why it is faster:** Python integer division must check for zero divisors and raise `ZeroDivisionError`. This requires a conditional branch (`if (b == 0) PyErr_SetString(...)`) on every division operation. C-style division simply performs the hardware division instruction.

**Where it is located:** `cython_setup.py` line 24.

**Without this:** Every integer division includes a zero-check and potential exception path, adding overhead even when the divisor is known to be nonzero.

---

### #12 — `infer_types=True`

**What it does:** Enables automatic C type inference for local variables that are not explicitly typed. Cython deduces the C type from assignments and uses native C variables instead of Python objects where possible.

**Why it is faster:** Python object variables require reference counting on every assignment and access. C variables do not. For loop counters, boolean flags, and intermediate integer results, using inferred C types eliminates reference counting overhead entirely.

**Where it is located:** `cython_setup.py` line 22.

**Without this:** All local variables default to Python objects, requiring `Py_INCREF`/`Py_DECREF` pairs on every assignment and access.

---

### #13 — `closure_freelist_size=16`

**What it does:** Pre-allocates a free list of 16 closure objects, reducing allocation overhead when closures and lambdas are created repeatedly.

**Why it is faster:** Closure objects are allocated and freed frequently when the library creates callbacks internally. A freelist allows reuse of previously freed closure memory without returning to the system allocator.

**Where it is located:** `cython_setup.py` line 6.

**Without this:** The default freelist size of 8 is used, resulting in more frequent allocator calls for workloads that create many short-lived closures.

---

### #14 — `-O3` compiler flag

**What it does:** Passes GCC/Clang's maximum optimization level to the C compiler when building the Cython-generated C files.

**Why it is faster:** `-O3` enables all `-O2` optimizations plus aggressive inlining (including inlining of functions not marked `inline`), loop vectorization using SIMD instructions, loop unrolling, and additional instruction scheduling transformations.

**Where it is located:** `setup.py` line 19.

**Without this:** Using `-O2` or `-O0` leaves significant performance on the table — particularly for loop-heavy code like the chain evaluation path where the compiler cannot vectorize or unroll without `-O3`.

---

### #15 — LTO support (`QUENT_LTO=1`)

**What it does:** Enables Link-Time Optimization when the `QUENT_LTO=1` environment variable is set. LTO defers final code generation until link time, allowing the linker to inline and optimize across translation unit boundaries.

**Why it is faster:** With LTO, the linker has visibility into all compilation units simultaneously. It can inline small functions across `.c` file boundaries, eliminate dead code that is only apparent at link time, and perform global constant propagation. Without LTO, each `.c` file is optimized in isolation.

**Where it is located:** `setup.py` lines 21-23. Activated via `QUENT_LTO=1` env var or `bash scripts/compile.sh bench`.

**Without this:** Each `.c` file is optimized independently; cross-file inlining and global dead code elimination are impossible.

---

### #16 — PGO support (`QUENT_PGO=generate/use`)

**What it does:** Enables Profile-Guided Optimization via a two-pass compilation process. Pass 1 (`QUENT_PGO=generate`) instruments the binary to collect branch and call frequency data. The test suite is run against the instrumented binary to generate profile data. Pass 2 (`QUENT_PGO=use`) recompiles using the profile data to guide optimization decisions.

**Why it is faster:** The compiler can make better decisions about branch prediction hints, function inlining thresholds, basic block ordering (hot code paths get cache-friendly linear layout), and register allocation when it knows the actual runtime behavior of the code rather than relying on static heuristics.

**Where it is located:** `setup.py` lines 24-29; `scripts/compile.sh` lines 74-93.

**Without this:** The compiler must guess branch probabilities and hot paths using static heuristics, which are often suboptimal for real workloads.

---

## Section 5: Architectural Patterns

### #17 — Sync-first / async-on-demand strategy

**What it does:** The main evaluation loop runs synchronously by default. Only when `iscoro(result)` detects that a step returned a coroutine does the loop transition to the async path via `_run_async`. Purely synchronous chains never touch the async machinery.

**Why it is faster:** Coroutine creation in Python is not free — it allocates a frame, initializes generator state, and creates a coroutine object. Running the entire evaluation loop as a coroutine even for sync chains would impose this overhead on every chain execution. The sync-first approach pays zero async overhead for sync chains.

**Where it is located:** `_chain_core.pxi` lines 91-107; `_frozen_chain.pxi` lines 183-227.

**Without this:** Always using `async def` / `await` for the evaluation loop would impose coroutine creation and scheduling overhead on every chain execution, even those that are entirely synchronous.

---

### #18 — `_ExecCtx` lazy allocation

**What it does:** The `_ExecCtx` (execution context) object is only allocated when actually needed — specifically during async transitions or when entering exception handling paths. Furthermore, `_ExecCtx.__new__(_ExecCtx)` is used directly to bypass `__init__` overhead.

**Why it is faster:** For synchronous chains with no exceptions, `_ExecCtx` is never allocated. The most common case (fast, exception-free, sync execution) incurs zero allocation overhead for execution context.

**Where it is located:** `_chain_core.pxi` lines 39, 60, 76, 96, 127, 164, 172; `_frozen_chain.pxi` at analogous locations.

**Without this:** Allocating `_ExecCtx` on every `_run` call would add allocation, initialization, and deallocation overhead to every single chain execution.

---

### #19 — Function aliases (cached `cdef object` references)

**What it does:** Module-level `cdef object` variables cache references to frequently called functions at module initialization time, avoiding Python name lookup (global dict hash table lookup) at every call site.

**Why it is faster:** Python global variable lookup is a hash table operation on the module's `__dict__`. Caching the function reference in a C-level `cdef object` variable makes the lookup a single pointer dereference.

**Where it is located:** `_foreach.pxi` lines 100-101 (`_async_foreach_fn`, `_foreach_async_fn`); `_filter.pxi` lines 81-82 (`_filter_async_fn`, `_async_filter_fn`); `_with.pxi` lines 84-85 (`_async_with_fn`, `_with_async_fn`); `_generator.pxi` lines 87-88 (`_sync_generator_fn`, `_async_generator_fn`); `_gather.pxi` lines 41-42 (`_gather_async_fn`, `_asyncio_gather_fn`); `quent.pyx` line 55 (`_await_run` direct call).

**Without this:** Every call to these functions goes through a module-level global dict lookup — a hash computation, comparison, and pointer chase — on every invocation.

---

### #20 — `EvalCode` enum — C-level integer dispatch

**What it does:** When a `Link` is constructed, the evaluation mode (plain value, callable with no args, callable with args, chain, etc.) is pre-computed once and stored as a C `int` enum value in `link.eval_code`. The evaluation loop switches on this integer rather than recomputing it.

**Why it is faster:** Pre-computing the evaluation mode at construction time means the decision is made once, not on every evaluation. At evaluation time, `switch(eval_code)` is a single integer comparison — a near-zero-cost operation — rather than re-running `PyCallable_Check`, checking for args and kwargs, and branching accordingly.

**Where it is located:** `quent.pxd` lines 28-32 (enum definition); `_link.pxi` lines 123-162 (`evaluate_value` switch); `_frozen_chain.pxi` (analogous switch).

**Without this:** Each link evaluation would need to re-check `PyCallable_Check`, examine args and kwargs, and branch — effectively re-doing construction-time analysis on every call.

---

### #21 — `PyCallable_Check` — C-level callable detection

**What it does:** Uses the direct C API function `PyCallable_Check` instead of the Python builtin `callable()` to test whether an object can be called.

**Why it is faster:** `PyCallable_Check` is a direct C function call that checks the `tp_call` slot in the type object — a single pointer dereference and null check. The Python `callable()` builtin goes through `__Pyx_PyObject_FastCall`, which involves function lookup and call overhead.

**Where it is located:** Imported via `from cpython.object cimport PyCallable_Check` in `quent.pxd` line 1. Used in `_determine_eval_code` (`_link.pxi` line 5) and `_eval_signal_value` (`quent.pyx` line 86).

**Without this:** Using Python's `callable()` would go through `__Pyx_PyObject_FastCall` with associated overhead at every callable detection site.

---

### #22 — `ignore_finally` flag

**What it does:** A boolean flag on the execution context that prevents the `finally_` handler from being executed twice during the sync-to-async transition. When the sync path encounters a coroutine mid-chain and hands off to the async path, the async path must not re-execute cleanup handlers that were already set up.

**Why it is faster:** This is a correctness optimization that also avoids complex state tracking logic. A single boolean flag check is cheaper than any alternative mechanism for tracking handler execution state across the sync/async boundary.

**Where it is located:** `_chain_core.pxi` lines 42, 59, 75, 94, 157; `_frozen_chain.pxi` lines 140, 154, 170, 196, 212, 270.

**Without this:** Either complex state tracking logic to detect double-execution, or redundant handler execution that would change observable behavior.

---

### #23 — Separate exception handler storage

**What it does:** The `on_except_link` and `on_finally_link` handlers are stored as direct fields on `Chain`, separate from the main linked list of `Link` objects.

**Why it is faster:** Keeping exception handlers out of the main linked list eliminates a per-link type or flag check on every iteration of the evaluation loop. The main loop iterates only over normal operation links; exception handlers are only consulted when an exception actually occurs.

**Where it is located:** `quent.pxd` line 174 (field declaration); `_chain_core.pxi` lines 135, 157 (access during exception handling).

**Without this:** Storing exception handlers in the linked list would require a type or flag check on every single link during the main evaluation loop, adding overhead to every normal-case iteration.

---

## Section 6: C-Level Call Replacements and Low-Level Optimizations

### #24 — `-march=native`

**What it does:** Instructs the C compiler to generate code for the exact CPU architecture of the build machine, enabling use of all available instruction set extensions (AVX2, AVX-512, etc.).

**Why it is faster:** Generic builds must use the lowest-common-denominator instruction set. With `-march=native`, the compiler can use wider SIMD vectors, newer branch prediction hints, and CPU-specific optimizations.

**Where it is located:** `setup.py` line 31 (gated behind `QUENT_NATIVE=1` environment variable).

**Without this:** The compiler emits generic x86-64 code that runs on any compatible processor but cannot exploit newer instruction set extensions available on the build machine.

---

### #25 — `-funroll-loops`

**What it does:** Instructs the C compiler to unroll loops — replicating loop body statements multiple times and reducing the loop iteration count — when it determines this is profitable.

**Why it is faster:** Loop unrolling reduces loop overhead (counter update, condition check, branch) and exposes more instruction-level parallelism for the CPU's out-of-order execution engine.

**Where it is located:** `setup.py` line 31 (gated behind `QUENT_NATIVE=1`).

**Without this:** All loops execute one iteration at a time with full loop overhead on every iteration.

---

### #26 — `-fomit-frame-pointer`

**What it does:** Instructs the compiler not to store the frame pointer in the `rbp` register, freeing it for use as a general-purpose register.

**Why it is faster:** On x86-64, registers are a scarce resource. Freeing `rbp` gives the register allocator one additional general-purpose register to work with, reducing register spills to the stack.

**Where it is located:** `setup.py` line 31 (gated behind `QUENT_NATIVE=1`).

**Without this:** `rbp` is dedicated to frame pointer use, one fewer register is available for general allocation, and the compiler must spill more variables to the stack.

---

### #27 — `PyIter_Next` — C-slot iteration

**What it does:** Uses the C-level `PyIter_Next` function (which calls `tp_iternext` directly on the type object) instead of going through Python's `__next__()` protocol. On exhaustion, `PyIter_Next` returns `NULL` rather than raising `StopIteration`.

**Why it is faster:** `tp_iternext` is a direct C slot call — a pointer dereference and function call with no Python argument marshalling. Calling `__next__()` through the Python protocol involves attribute lookup and Python function call overhead. Additionally, avoiding `StopIteration` as the exhaustion signal saves the cost of constructing and raising a Python exception.

**Where it is located:** `quent.pxd` line 8 (import). Used in `_foreach.pxi` lines 18-20 and `_filter.pxi` lines 17-19.

**Without this:** Each iteration step involves Python `__next__()` call overhead, and loop termination would require catching `StopIteration` — an expensive exception-path operation.

---

### #28 — Cached `_MethodType`

**What it does:** Caches `types.MethodType` in a module-level `cdef object _MethodType` variable at module initialization.

**Why it is faster:** Accessing `types.MethodType` without caching would require a module attribute lookup on the `types` module — a dict hash table lookup — every time the type is needed. The cached reference is a direct pointer dereference.

**Where it is located:** `quent.pyx` line 28 (initialization); `_chain_wrappers.pxi` line 26 (use).

**Without this:** Each use of `types.MethodType` would require a hash table lookup on the `types` module's `__dict__`.

---

### #29 — Cached `_registry_discard`

**What it does:** Caches the `task_registry.discard` bound method in a module-level `cdef object _registry_discard` variable at module initialization.

**Why it is faster:** Attribute lookup on `task_registry` followed by method binding would occur on every task cleanup. The cached bound method is a direct call through a stored pointer.

**Where it is located:** `quent.pyx` line 139 (initialization); `quent.pyx` line 153 (use).

**Without this:** Every task completion would require attribute lookup on `task_registry` to find `discard`, plus bound method creation overhead.

---

### #30 — Pre-built eager-start task creation

**What it does:** A conditional `_create_task_fn` is selected once at initialization time based on configuration, avoiding per-call kwargs dict construction during task creation.

**Why it is faster:** If task creation always passed `**kwargs` conditionally, every call would involve dict creation or inspection. Pre-selecting the right function variant at initialization time means the hot-path call has no conditional logic and no kwargs overhead.

**Where it is located:** `quent.pyx` lines 141-148.

**Without this:** Each task creation call would need to construct a kwargs dict or use a conditional branch to decide whether to pass additional keyword arguments.

---

### #31 — `EMPTY_DICT` / `EMPTY_TUPLE` in specialized Link construction

**What it does:** The `foreach()`, `filter_()`, and `gather_()` constructors use the pre-allocated `EMPTY_TUPLE` and `EMPTY_DICT` singletons when no extra args or kwargs are provided.

**Why it is faster:** See optimization #6. This ensures that the allocation savings apply specifically to the frequently-used high-level API methods, not just the base `Link` constructor path.

**Where it is located:** `_foreach.pxi` line 41; `_filter.pxi` line 36; `_gather.pxi` line 23.

**Without this:** Each `foreach()`, `filter_()`, or `gather_()` call would allocate new tuple and dict objects for the default empty case.

---

### #32 — Pointer cast instead of `id()`

**What it does:** Uses `<uintptr_t><void*>link` to obtain a unique integer key for dict entries rather than Python's `id()` builtin.

**Why it is faster:** `id()` is a Python function call with full Python call overhead. A Cython pointer cast is a zero-cost C-level operation — no Python object creation, no function call, no reference counting.

**Where it is located:** Used as dict keys in `_foreach.pxi`, `_filter.pxi`, `_with.pxi`.

**Without this:** Using `id(link)` would involve a Python function call on every dict access, adding overhead to every operation that indexes into internal state dicts.

---

### #33 — `_ExecCtx` async state packing

**What it does:** Packs the state needed for `_run_async` into the 3 fields of `_ExecCtx` rather than passing 8 separate parameters.

**Why it is faster:** Python function calls with many arguments require building argument tuples or processing argument lists. Packing state into a struct-like object and passing a single reference reduces call overhead and keeps related state co-located in memory.

**Where it is located:** `quent.pxd` lines 70-76 (`_ExecCtx` field definitions); `_chain_core.pxi` line 185.

**Without this:** Passing 8 parameters to `_run_async` would require more argument-handling overhead at the call site and at the function entry point.

---

### #34 — `_create_link` cdef factory function

**What it does:** Provides a `cdef` factory function that constructs `Link` objects directly, bypassing Python's `__init__` argument parsing machinery.

**Why it is faster:** Python `__init__` must handle arbitrary positional and keyword arguments through `*args`/`**kwargs` parsing. A `cdef` factory function receives typed C-level parameters directly, with zero argument marshalling overhead.

**Where it is located:** `_link.pxi` lines 98-120.

**Without this:** Every `Link` construction would go through Python's `__init__` argument parsing, adding overhead to every chain operation addition.

---

### #35 — Inline root evaluation via `_eval_signal_value`

**What it does:** The root value evaluation (the first value in a chain) is handled inline without allocating a `Link` object for it. `_eval_signal_value` evaluates callable roots and signal values directly on the sync path.

**Why it is faster:** Avoids one `Link` allocation, initialization, and deallocation for the root value of every chain execution. Since every chain execution processes the root value, this saves at least one allocation per `run()` call.

**Where it is located:** `_chain_core.pxi` lines 49-71; `_frozen_chain.pxi` lines 148-165.

**Without this:** The root value would be wrapped in a `Link` like all other chain steps, incurring unnecessary allocation overhead on every chain execution.

---

### #36 — Pre-split `link.args[1:]`

**What it does:** When a `Link` is constructed with a method call that passes `self` as the first argument, the `args[1:]` slice is pre-computed at construction time and stored, rather than recomputed on every evaluation.

**Why it is faster:** Tuple slicing creates a new tuple object. If a link is evaluated many times (e.g., in a `_FrozenChain` used repeatedly), computing the slice once at construction time versus on every evaluation is a significant saving.

**Where it is located:** `_link.pxi` lines 50-53 (construction-time computation); `_link.pxi` lines 117-119 (storage); `_link.pxi` line 141 (use at evaluation time).

**Without this:** Every evaluation of such a link would perform a tuple slice operation, creating a new tuple object each time.

---

### #37 — `_FrozenChain` tuple iteration

**What it does:** `_FrozenChain` stores its links in a contiguous Python tuple rather than as a singly-linked list (`Link.next_link` pointer chain).

**Why it is faster:** Tuple storage is cache-friendly — elements are contiguous in memory. Linked list traversal requires pointer chasing through potentially non-contiguous heap allocations, causing cache misses. Tuple indexing also enables direct element access by index, which is useful for the `_all_simple` fast path.

**Where it is located:** `_frozen_chain.pxi` lines 20-34 (construction); `_frozen_chain.pxi` lines 183-223 (iteration).

**Without this:** A linked list requires following `next_link` pointers that may be scattered across the heap, causing cache misses on every step during iteration.

---

### #38 — `_all_simple` fast path

**What it does:** A boolean flag set at `_FrozenChain` construction time indicating that all links are "simple" — no async, no except/finally handlers, no special control flow. When `_all_simple` is true, the evaluation loop skips all special-case checks.

**Why it is faster:** For the common case of a simple synchronous chain with no exceptions and no async operations, every iteration of the evaluation loop can skip coroutine checks, exception handler checks, and control flow checks entirely. This collapses the loop to its minimal form.

**Where it is located:** `_frozen_chain.pxi` lines 24-28 (flag computation during construction); `_frozen_chain.pxi` lines 185-206 (fast path evaluation).

**Without this:** Every link evaluation would check for async, exceptions, and special control flow even when the chain is known to have none of these features.

---

### #39 — `PySet_Add` for task registry

**What it does:** Uses the C API `PySet_Add` function for inserting tasks into the internal task registry set rather than calling the Python `set.add` method.

**Why it is faster:** `PySet_Add` is a direct C function call that bypasses Python method lookup. Calling `set.add` via Python would require attribute lookup on the set object, method binding, and a Python-level function call.

**Where it is located:** `quent.pxd` line 3 (import); `quent.pyx` line 152 (use).

**Without this:** Each task registration would go through Python method lookup and call overhead for `set.add`.

---

### #40 — Initialized loop variables

**What it does:** Loop variables `el = None` and `result = None` are explicitly initialized before loop entry.

**Why it is faster:** Cython generates `RaiseUnboundLocalError` checks for variables that might be used before assignment. Explicit initialization to `None` before the loop tells Cython (and the compiler) that the variables are always initialized, eliminating the generated check.

**Where it is located:** `_foreach.pxi` line 13; `_filter.pxi` line 12.

**Without this:** Without explicit initialization, Cython may generate `if (el == NULL) raise UnboundLocalError` checks inside the loop body, adding a branch on every iteration.

---

### #41 — `nonecheck=False`, `overflowcheck=False`

**What it does:** Explicitly disables two safety checks: `nonecheck=False` disables automatic `None`-dereference checking on typed variables; `overflowcheck=False` disables integer overflow detection on C integer arithmetic.

**Why it is faster:** With `nonecheck=True`, Cython generates `if (obj == NULL) raise TypeError("'NoneType' ...")` before every method or attribute access on typed variables. With `overflowcheck=True`, every integer arithmetic operation includes overflow detection logic. Disabling both eliminates these checks from the generated C code.

**Where it is located:** `cython_setup.py` lines 26-27.

**Without this:** Every typed variable access would check for `None`, and every integer operation would check for overflow, adding conditional branches throughout the generated C code.
