# Phase 4: Data Structure Optimizations (C Array for Links)

## 4.1 Replace linked list with C array for frozen chain evaluation

**Files:** `_chain_core.pxi`, `_variants.pxi`, `quent.pxd`

**Status:** NEW functionality — `_FrozenChain` does not currently exist anywhere in the codebase and must be created from scratch. There is no `freeze()` method on `Chain`. Both must be added.

**Dependencies:** Phase 3.1 (inline root eval). Phase 3.3 is NOT a dependency — exception handlers are already stored separately on `Chain` as `on_except_link` and `on_finally_link` fields (not mixed into the linked list), so the frozen link array does not need to skip or filter them.

**Current Chain field layout (from `quent.pxd`):**
```cython
cdef class Chain:
  cdef:
    Link root_link, first_link, on_finally_link, on_except_link
    Link current_link
    object on_except_exceptions
    bint is_nested
```

**Current evaluation loop (from `_chain_core.pxi` lines 65-72):** Links form a singly-linked list. Evaluation follows `link = link.next_link` pointers, which may scatter across heap memory. There is already no per-link exception handler check — that concern is already resolved by the existing architecture.

```cython
while link is not None:
    result = evaluate_value(link, current_value)
    if iscoro(result):
        ignore_finally = True
        return self._run_async(temp_root_link, link_results, link, result, current_value, root_value, has_root_value)
    if not link.ignore_result:
        current_value = result
    link = link.next_link
```

**Proposed:** Create a new `_FrozenChain` class and a `freeze()` method on `Chain`. When `freeze()` is called, "compile" the linked list into a contiguous tuple:

Step 1 — Declare `_FrozenChain` in `quent.pxd`:
```cython
@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  cdef Chain _chain
  cdef tuple _links  # pre-built tuple of Links in order
  cdef int _n_links  # count for fast iteration
```

Step 2 — Implement `_FrozenChain` in `_variants.pxi`:
```cython
@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  def __init__(self, Chain chain):
    self._chain = chain
```

Step 3 — Add `freeze()` to `Chain` in `quent.pyx` (or `_chain_core.pxi`):
```cython
def freeze(self):
  fc = _FrozenChain(self)
  # Pre-build links array from the regular linked list.
  # Exception handlers (on_except_link, on_finally_link) are stored
  # directly on Chain — they are not in the linked list and require
  # no filtering here.
  links = []
  link = self.first_link
  while link is not None:
    links.append(link)
    link = link.next_link
  fc._links = tuple(links)
  fc._n_links = len(links)
  return fc
```

Step 4 — Add a dedicated `cdef _frozen_run` that iterates the tuple instead of the linked list:
```cython
cdef object _frozen_run(_FrozenChain fc, object v, tuple args, dict kwargs):
  # Inline root evaluation (from Phase 3.1)
  # Then iterate the pre-built array:
  cdef int i
  cdef Link link
  for i in range(fc._n_links):
    link = <Link>fc._links[i]  # C-level tuple access via PyTuple_GET_ITEM
    result = evaluate_value(link, current_value)
    if iscoro(result):
      # Transition to async...
      pass
    if not link.ignore_result:
      current_value = result
  return current_value
```

Benefits:
- CPU prefetcher can predict sequential access pattern
- No pointer-chasing through `next_link`
- Exception handlers already excluded from the linked list — no filtering needed at freeze time, and `on_except_link` / `on_finally_link` are accessed directly from `fc._chain` when needed
- Tuple provides contiguous `PyObject*` array access via `PyTuple_GET_ITEM`
- Combined with Phase 3.1 (no temp link), this eliminates most per-link overhead for frozen chains

## 4.2 Pre-compute flat evaluation plan for frozen chains (extreme optimization)

**Files:** `_variants.pxi`, `quent.pxd`

**Status:** Builds on 4.1. `_FrozenChain` must be created first (see 4.1 above).

Take 4.1 further: at freeze time, pre-compute an evaluation plan that separates common-case links from special cases.

Extend the `_FrozenChain` declaration in `quent.pxd`:
```cython
@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  cdef Chain _chain
  cdef tuple _links          # all links in order
  cdef int _n_links
  cdef bint _all_simple      # True if all links are EVAL_CALL_WITH_CURRENT_VALUE and not is_chain
  cdef bint _has_finally     # True if chain has a finally handler (on_finally_link is not None)
  cdef bint _has_except      # True if chain has an exception handler (on_except_link is not None)
```

Extend `freeze()` to compute the flags:
```cython
def freeze(self):
  fc = _FrozenChain(self)
  links = []
  link = self.first_link
  all_simple = True
  while link is not None:
    links.append(link)
    if link.eval_code != EVAL_CALL_WITH_CURRENT_VALUE or link.is_chain:
      all_simple = False
    link = link.next_link
  fc._links = tuple(links)
  fc._n_links = len(links)
  fc._all_simple = all_simple
  fc._has_finally = self.on_finally_link is not None
  fc._has_except = self.on_except_link is not None
  return fc
```

When `_all_simple` is True, use an ultra-fast path:
```cython
if fc._all_simple:
  for i in range(fc._n_links):
    link = <Link>fc._links[i]
    if current_value is Null:
      current_value = link.v()
    else:
      current_value = link.v(current_value)
    if iscoro(current_value):
      # async transition...
      pass
  return current_value
```

This skips `evaluate_value` entirely for the common case — no eval_code dispatch, no is_chain check, no ignore_result check.
