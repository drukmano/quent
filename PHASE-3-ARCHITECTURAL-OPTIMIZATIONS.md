# Phase 3: Hot-Path Architectural Optimizations

These are larger changes to the evaluation loop and data structures that reduce per-link overhead.

## 3.1 Eliminate `_make_temp_link` for frozen chain root override

**File:** `_chain_core.pxi` (lines 49-53)

**Current:** Every `frozen.run(value)` allocates a `Link` via `_make_temp_link` (12 field assignments + `_determine_eval_code`), then sets `temp_root_link.next_link = self.first_link`.

**Proposed:** Inline the root evaluation directly in `_run` when `is_root_value_override and not has_root_value`:

```cython
if is_root_value_override:
  if has_root_value:
    raise QuentException(...)
  # Inline root evaluation instead of creating temp link
  if not args and not kwargs:
    if PyCallable_Check(v):
      result = v()
    else:
      result = v
  elif args and args[0] is ...:
    result = v()
  elif args or kwargs:
    # ... handle explicit args
  else:
    result = v

  if iscoro(result):
    # Need to handle async transition — create temp link only for async path
    temp_root_link = _make_temp_link(v, args, kwargs)
    temp_root_link.next_link = self.first_link
    ignore_finally = True
    return self._run_async(...)

  root_value = result
  current_value = result
  link = self.first_link
  has_root_value = True
```

This eliminates the `Link` allocation for the common case (sync root evaluation). Only the async fallback path allocates the temp link (needed for traceback context).

**Alternative simpler approach:** Create a dedicated `cdef object _eval_root_override(...)` that inlines the 4 eval code paths without creating a Link. This is cleaner and reusable.

## 3.2 Fix `link.args[1:]` tuple slice in nested chain evaluation

**File:** `_link.pxi` (line 122)

**Current:**
```cython
return (<Chain>link.v)._run(link.args[0], link.args[1:], link.kwargs, True)
```
Creates a new tuple every call (score 20).

**Proposed:** Pre-compute the split at Link construction time. In `_determine_eval_code` (or `Link.__init__`), when `eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS` and `link.is_chain`:
```cython
# Store pre-split values:
link.temp_args = link.args[1:] if len(link.args) > 1 else None
# The first arg is already link.args[0]
```

Then in `evaluate_value`:
```cython
return (<Chain>link.v)._run(link.args[0], link.temp_args, link.kwargs, True)
```

This moves the tuple slice from runtime (every evaluation) to construction time (once). The `temp_args` field already exists on Link but is used for diagnostics — we may need a dedicated field, or repurpose it since construction-time pre-split is compatible.

**Alternative:** Add a new field `Link.first_arg` (object) and rename `link.args` to store only the remaining args. This avoids reusing `temp_args` and is cleaner.

## 3.3 Pack `_run_async` parameters into `_ExecCtx`

**File:** `_chain_core.pxi` (lines 58, 69, 143)

**Current:** `_run_async` takes 7 parameters (plus `self`) via Python dispatch (score 97 at line 143). The `bint has_root_value` is boxed to Python bool and unboxed on the other side.

```cython
async def _run_async(self, Link temp_root_link, dict link_results, Link link, object awaitable, object current_value, object root_value, bint has_root_value):
```

**Proposed:** Extend `_ExecCtx` to carry all state needed for async transition:

```cython
@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  cdef Link temp_root_link
  cdef dict link_results
  cdef dict link_temp_args
  # NEW fields for async transition:
  cdef Link async_link
  cdef object current_value
  cdef object root_value
  cdef bint has_root_value
```

Then `_run_async` becomes:
```cython
async def _run_async(self, _ExecCtx ctx, object awaitable):
```

Only 3 parameters instead of 8. The `_ExecCtx` is created lazily only when async transition occurs, and its freelist makes allocation cheap.

This eliminates:
- 5 fewer Python arguments to parse
- No `bint` → `PyBool` boxing
- Simpler generated C wrapper

## 3.4 Replace `_await_run_fn` indirect call with direct invocation

**File:** `_chain_core.pxi` (lines 105, 136), `_link.pxi` (line 9)

**Current:** `_await_run_fn` is stored as `cdef object = _await_run` and called via generic `__Pyx_PyObject_FastCall` with `PyMethod_Check` overhead (score 20-34).

**Proposed:** Call `_await_run(result, chain, link, ctx)` directly instead of through the alias. Since `_await_run` is defined in the same compilation unit (via `.pxi` include), Cython can generate a direct coroutine creation call.

If Cython still goes through Python dispatch for `async def` functions, consider inlining the logic at call sites:
```cython
# Instead of:
ensure_future(_await_run_fn(result, self, exc_link, ctx))
# Use:
ensure_future(_await_run(result, self, exc_link, ctx))
```

## 3.5 Use `cdef` factory function for Link construction (bypass `__init__` wrapper)

**File:** `_link.pxi` (lines 40-60)

**Current:** Internal callers use `Link(v, args, kwargs)` which goes through the full Python `def __init__` argument parser (score 27).

**Proposed:** Create a `cdef` factory function similar to `_make_temp_link`:
```cython
cdef inline Link _create_link(object v, tuple args, dict kwargs, bint ignore_result=False, object original_value=None):
  cdef Link link = Link.__new__(Link)
  link.v = v
  link.is_chain = type(v) is Chain
  # ... set all fields ...
  _determine_eval_code(link, v, args, kwargs)
  return link
```

Replace all internal `Link(v, args, kwargs)` calls with `_create_link(v, args, kwargs)`. Keep `Link.__init__` for external API compatibility.

This eliminates the Python argument parsing overhead for every `.then()`, `.do()`, `.except_()`, etc. call.
