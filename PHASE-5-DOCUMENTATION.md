# Phase 5: Documentation

## 5.1 Create `PERFORMANCE.md` in project root

Document ALL performance optimizations, both existing and new:

**Existing optimizations to document:**
1. `@cython.final` on all cdef classes — prevents subclassing, enables direct C method dispatch
2. `@cython.freelist(N)` on frequently allocated classes — avoids malloc/free
3. `@cython.no_gc` on `_Null` — no GC tracking for sentinel
4. `cdef inline iscoro()` — inline C-level coroutine type check (two pointer comparisons)
5. Hot/cold field ordering on `Link` — hot fields first for cache locality
6. Pre-allocated `EMPTY_TUPLE` and `EMPTY_DICT` — avoid repeated empty container allocation
7. `binding=False` — lighter-weight CFunction wrappers
8. `always_allow_keywords=False` — METH_NOARGS/METH_O calling conventions
9. `boundscheck=False`, `wraparound=False` — no array bounds checking
10. `initializedcheck=False` — no memoryview init checks
11. `cdivision=True` — C-style integer division
12. `infer_types=True` — automatic C type inference
13. `closure_freelist_size=16` — larger freelist for closures
14. `-O3` compiler flag — maximum C optimization
15. LTO support (`QUENT_LTO=1`) — cross-function inlining
16. PGO support (`QUENT_PGO=generate/use`) — profile-guided optimization
17. Sync-first/async-on-demand strategy — no coroutine overhead for sync chains
18. `_ExecCtx` lazy allocation — only on exception paths
19. Function aliases (e.g., `cdef object _foreach_fn = _foreach_full_async`) — avoid module-level name lookups
20. `EvalCode` enum — C-level integer comparison for dispatch
21. `PyCallable_Check` — C-level callable detection
22. `ignore_finally` flag — prevents double-execution during async transition
23. Separate exception handler storage (`on_except_link`, `on_finally_link` on Chain) — no per-link branch in main evaluation loop

**New optimizations to document:**
24. `-march=native` — CPU-specific instruction selection
25. `-funroll-loops` — loop unrolling
26. `-fomit-frame-pointer` — register savings
27. `PyIter_Next` iteration — C-slot iteration avoiding StopIteration exceptions
28. Cached `_MethodType` — avoid attribute lookup on `types` module
29. Cached `_registry_discard` — avoid bound method allocation per task
30. Pre-built eager_start kwargs — avoid per-call dict construction
31. `EMPTY_DICT` in Link construction — avoid empty dict allocation
32. Pointer cast instead of `id()` — C-level pointer-to-int
33. `_ExecCtx` async state packing — reduced Python argument parsing
34. `_create_link` cdef factory — bypass `__init__` Python wrapper
35. [NEW: `_FrozenChain`] Inline root evaluation for frozen chains — eliminate temp Link allocation
36. Pre-split `link.args[1:]` at construction — avoid per-evaluation tuple slice
37. [NEW: `_FrozenChain`] C array (tuple) for frozen chain links — sequential memory access
38. [NEW: `_FrozenChain`] `_all_simple` fast path — skip evaluate_value dispatch entirely
39. `PySet_Add` for task_registry — C-level set operation
40. Initialized loop variables — avoid RaiseUnboundLocalError checks
41. `nonecheck=False`, `overflowcheck=False` — explicit safety opt-outs

## 5.2 Add inline comments to all optimization sites

Every optimization should have a comment in the source code explaining:
- What it does
- Why it's faster than the alternative
- What the alternative would be
- Any safety considerations

Example:
```cython
# PERF: Use PyIter_Next (C-slot tp_iternext) instead of .__next__() method dispatch.
# Avoids: Python method lookup + StopIteration exception creation/matching per element.
# See PERFORMANCE.md #26.
```
