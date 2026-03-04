# Phase 2: C-Level Call Replacements (Python Overhead Elimination)

**Files:** All `.pxi` files and `quent.pxd`

This phase systematically replaces Python-level operations identified in the HTML annotation reports with direct C API equivalents. Each change is verified against the annotation scores.

## 2.1 Iteration protocol: Replace `__iter__()`/`__next__()` with C-slot calls

**Files:** `_iteration.pxi` (lines 14, 17, 28, 102, 105, 112)

**Current (per-element overhead):**
```cython
current_value = current_value.__iter__()
while True:
  el = current_value.__next__()
  ...
  except StopIteration:
    break
```
Score: `__iter__()` = 5, `__next__()` = 5, `except StopIteration` = 6 — total 16 per iteration.

**Proposed:**
```cython
from cpython.ref cimport PyObject
from cpython.exc cimport PyErr_Occurred

cdef extern from "Python.h":
  PyObject* PyIter_Next(object) except NULL

cdef object it = iter(current_value)
while True:
  el = <object>PyIter_Next(it)
  if el is None:
    if PyErr_Occurred():
      raise
    break
  ...
```

This eliminates:
- `__Pyx_PyObject_FastCallMethod(__iter__)` → replaced by `PyObject_GetIter` via `iter()`
- `__Pyx_PyObject_FastCallMethod(__next__)` → replaced by `tp_iternext` C-slot via `PyIter_Next`
- `except StopIteration` exception matching → replaced by NULL check (no exception raised)

**IMPORTANT CAVEAT:** The current pattern exists because mid-iteration, if `iscoro(result)` is detected, the code needs to hand off the iterator state to an async continuation. The `for el in` syntax doesn't give us access to the iterator object. Using manual `PyIter_Next` preserves this handoff capability while being as fast as `for el in`.

Apply to: `_Foreach.__call__` (lines 14, 17, 28) and `_Filter.__call__` (lines 102, 105, 112) sync loops, and their async transition functions.

## 2.2 Initialize loop variables to avoid `RaiseUnboundLocalError`

**Files:** `_iteration.pxi` (line 13 for `_Foreach.__call__`, line 101 for `_Filter.__call__` — `el` declared but uninitialized)

**Current:** `cdef object el, result` is declared without initialization before the loop, causing Cython to emit `__Pyx_RaiseUnboundLocalError("el")` checks in except blocks (score 11).

**Fix:** Change the declarations to `cdef object el = None, result = None` before the loop.

## 2.3 Cache `types.MethodType` as module-level cdef variable

**File:** `_variants.pxi` (line 14)

**Current:**
```cython
return types.MethodType(self, obj)
```
Generates `__Pyx_PyObject_GetAttrStr(types, 'MethodType')` on every descriptor access (score 20).

**Fix:** Add at module level (in `quent.pyx` or `_variants.pxi`):
```cython
cdef object _MethodType = types.MethodType
```
Then: `return _MethodType(self, obj)`

## 2.4 Cache `task_registry.discard` as module-level cdef variable

**File:** `_async_utils.pxi` (line 18)

**Current:**
```cython
task.add_done_callback(task_registry.discard)
```
Creates a new bound method object on every `ensure_future` call (score 8).

**Fix:**
```cython
cdef object _registry_discard = task_registry.discard
# Then:
task.add_done_callback(_registry_discard)
```

## 2.5 Pre-build `eager_start` kwargs dict for `_create_task`

**File:** `_async_utils.pxi` (line 14)

**Current:**
```cython
task = _create_task(coro, eager_start=True)
```
Builds a new kwargs dict on every call (`__Pyx_MakeVectorcallBuilderKwds`, score 21).

**Fix:** Cache two function references at module init:
```cython
if _HAS_EAGER_START:
  cdef object _create_task_fn = lambda coro: _create_task(coro, eager_start=True)
else:
  cdef object _create_task_fn = _create_task
```
Then `ensure_future` just calls `_create_task_fn(coro)` — no kwargs construction per call.

## 2.6 Replace `{}` literals with `EMPTY_DICT` in Link construction

**Files:** `_iteration.pxi` (line 36: `foreach()`, line 120: `filter_()`, line 182: `gather_()`)

**Current:**
```cython
Link(fn, (), {})  # allocates a new empty dict every time
```
Score: `__Pyx_PyDict_NewPresized(0)` = 6 per construction.

**Fix:**
```cython
Link(fn, EMPTY_TUPLE, EMPTY_DICT)
```
`EMPTY_TUPLE` and `EMPTY_DICT` are already defined as module-level `cdef` constants.

## 2.7 Replace `id(link)` with C pointer cast in async error paths

**Files:** `_iteration.pxi` (line 61 in `_foreach_to_async`, line 86 in `_foreach_full_async`, line 138 in `_filter_to_async`, line 156 in `_filter_full_async`); `_control_flow.pxi` (line 109 in `_with_full_async`)

**Current:**
```cython
exc.__quent_link_temp_args__[id(link)] = (el,)
```
Calls Python's `id()` builtin via `__Pyx_PyObject_FastCall`.

**Fix:**
```cython
exc.__quent_link_temp_args__[<Py_intptr_t><void*>link] = (el,)
```
Direct pointer-to-int cast at C level, no Python function call. The `Py_intptr_t` is automatically converted to a Python int for dict key usage. Since these are error paths, the dict key type change is invisible to users.

## 2.8 Use `PyException_GetTraceback` instead of `exc.__traceback__`

**File:** `_control_flow.pxi` (line 58: `current_value.__exit__(type(exc), exc, exc.__traceback__)` in `_With.__call__`; line 80: same pattern in `_with_to_async`)

**Current:** `exc.__traceback__` uses `__Pyx_PyObject_GetAttrStr` (attribute lookup).

**Fix:**
```cython
cdef extern from "Python.h":
  object PyException_GetTraceback(object)
```

## 2.9 Pre-build `_Generator` default run_args tuple

**File:** `_control_flow.pxi` (line 179)

**Current:**
```cython
self._run_args = (Null, (), {}, False)
```
Allocates a new dict `{}` and 4-tuple every `__init__`. Score 19.

**Fix:**
```cython
cdef tuple _DEFAULT_RUN_ARGS = (Null, EMPTY_TUPLE, EMPTY_DICT, False)
# Then:
self._run_args = _DEFAULT_RUN_ARGS
```

## 2.10 Use `PySet_Add` / pre-cached discard for task_registry

**File:** `_async_utils.pxi` (line 17)

**Current:**
```cython
task_registry.add(task)
```
Cython does NOT optimize `set.add()` to `PySet_Add` — it goes through method dispatch.

**Fix:**
```cython
from cpython.set cimport PySet_Add
# Then:
PySet_Add(task_registry, task)
```

## 2.11 Cache `_await_run` alias for link-level await dispatch

**File:** `_link.pxi` (line 9: `cdef object _await_run_fn = _await_run`)

The async def `_await_run` is defined at lines 1–7 and aliased to a `cdef object` at line 9. All call sites that dispatch through `_await_run_fn` avoid a Python name lookup, but the alias itself is set at module init. Verify that no call site references `_await_run` directly by name (which would bypass the cdef cache and force a global dict lookup).

## 2.12 Review all annotation HTML lines and replace remaining Python overhead

After the above changes, recompile in test mode and re-examine all HTML reports. Systematically address any remaining high-score lines that are on hot paths. This is an iterative process — each round may reveal new opportunities as Cython re-generates code.
