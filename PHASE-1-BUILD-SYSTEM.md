# Phase 1: Build System & Compiler Optimizations

**Files:** `setup.py`, `cython_setup.py`, `scripts/compile.sh`

## 1.1 Add aggressive C compiler flags for dev/benchmark builds

In `setup.py`, add to `compile_args`:
- `-march=native` — CPU-specific tuning (AVX2, NEON, etc.)
- `-funroll-loops` — unroll loops with known trip counts
- `-fomit-frame-pointer` — free up a register (rbp on x86_64)

Gate these behind an env var (e.g., `QUENT_NATIVE=1`) so wheel builds remain portable.

## 1.2 Add explicit Cython directives for documentation

In `cython_setup.py`, explicitly add directives that are already defaults but should be documented:
- `nonecheck: False`
- `overflowcheck: False`
- `optimize.use_switch: True`
- `optimize.unpack_method_calls: True`

## 1.3 Add benchmark compilation mode to `scripts/compile.sh`

`scripts/compile.sh` already contains a full PGO 2-pass workflow (invoked via `bash scripts/compile.sh pgo`): pass 1 compiles with `-fprofile-generate`, runs the test suite to collect profile data, then pass 2 recompiles with `-fprofile-use`. No changes to PGO support are needed.

What is missing is a dedicated **benchmark** mode that combines the native-tuning flags with LTO in a single command. Add a `bench` argument to the script that sets `QUENT_NATIVE=1` and `QUENT_LTO=1` automatically before invoking `cython_setup.py`, so that:

```bash
bash scripts/compile.sh bench
```

produces a maximally optimised local build without requiring the caller to set env vars manually. Document the three compile modes in a comment block at the top of the script:

| Mode | Command | Effect |
|------|---------|--------|
| Standard | `bash scripts/compile.sh` | Portable, incremental, no extra flags |
| Benchmark | `bash scripts/compile.sh bench` | Sets `QUENT_NATIVE=1 QUENT_LTO=1`, full recompile |
| PGO | `bash scripts/compile.sh pgo` | 2-pass profile-guided optimisation (already implemented) |
