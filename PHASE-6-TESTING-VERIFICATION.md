# Phase 6: Testing & Verification

## 6.1 Compile and run full test suite after each phase

```bash
source .venv@3.tmp/bin/activate
bash scripts/tests_compile.sh
bash scripts/run_tests.sh
```

All existing tests must continue to pass. The test suite is comprehensive (70+ test files).

## 6.2 Re-generate HTML annotation reports after each phase

```bash
bash scripts/tests_compile.sh  # generates HTML annotations
```

Compare before/after scores for modified lines to verify Python overhead was reduced.

## 6.3 Run benchmarks after each phase

```bash
bash scripts/compile.sh --force  # production build (no linetrace)
python scripts/benchmark.py
```

Track improvement per phase. Expected cumulative improvements:
- Phase 1 (build flags): 2-5% on all benchmarks
- Phase 2 (C-level replacements): 5-15% on iteration, 3-8% on chain eval
- Phase 3 (architectural): 15-25% on frozen chains
- Phase 4 (C array): 10-20% additional on frozen chains

## 6.4 Write new tests for optimized paths

Every optimization that changes behavior or adds a new code path needs tests:
- Frozen chain with pre-built link array
- Inline root evaluation (sync and async)
- `_create_link` factory producing identical Links to `Link.__init__`
- `_all_simple` fast path for frozen chains
- Edge cases: empty chains, single-link chains, chains with only exception handlers

---

## Implementation Order (Phase-by-Phase)

| Step | Phase | Description | Risk | Dependencies |
|------|-------|-------------|------|--------------|
| 1 | 1.1-1.2 | Build system flags & directives | Low | None |
| 2 | 2.1-2.2 | PyIter_Next + init loop vars | Medium | None |
| 3 | 2.3-2.10 | Cache module vars, EMPTY_DICT, id() | Low | None |
| 4 | 3.5 | `_create_link` factory | Medium | None |
| 5 | 3.2 | Pre-split args[1:] | Low | Step 4 |
| 6 | 3.1 | Inline root evaluation | Medium | None |
| 7 | 3.3 | Pack _run_async params | Medium | None |
| 8 | 3.4 | Direct _await_run call | Low | None |
| 9 | 4.1 | C array for frozen links | High | Step 6 |
| 10 | 4.2 | _all_simple fast path | Medium | Step 9 |
| 11 | 5.1-5.2 | Documentation | Low | All above |
| 12 | 6.1-6.4 | Testing & verification | Low | All above |
| 13 | 2.11 | Re-analyze HTML reports, address remaining | Low | All above |

Steps 1-3 can be parallelized. Steps 4-8 can be partially parallelized. Steps 9-10 depend on Step 6.

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `quent/quent.pyx` | New cimports, cached module-level variables |
| `quent/quent.pxd` | New fields on `_ExecCtx`, `Chain`; new `_FrozenChain` type (created from scratch); new cdef declarations |
| `quent/_link.pxi` | `_create_link` factory, pre-split args, PyIter_Next in evaluate_value |
| `quent/_chain_core.pxi` | Inline root eval, pack async params, direct _await_run |
| `quent/_iteration.pxi` | PyIter_Next iteration, init loop vars, EMPTY_DICT, id() replacement |
| `quent/_control_flow.pxi` | EMPTY_DICT, pre-built run_args, PyException_GetTraceback |
| `quent/_async_utils.pxi` | Cache discard, pre-build kwargs, PySet_Add |
| `quent/_variants.pxi` | Cache MethodType, new `_FrozenChain` class with C array, _all_simple fast path |
| `quent/_operators.pxi` | Minor: EMPTY_DICT usage |
| `setup.py` | New compiler flags with env var gating |
| `cython_setup.py` | Explicit directive documentation |
| `scripts/compile.sh` | Benchmark compilation mode |
| `PERFORMANCE.md` | New file — comprehensive optimization documentation |
| `tests/` | New test files for optimized paths |
