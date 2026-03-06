# TEST_EXP.md -- Exhaustive Test Suite Expansion Plan for quent

## Table of Contents

1. [Status Quo](#1-status-quo)
2. [Testing Axes and Variant Matrix](#2-testing-axes-and-variant-matrix)
3. [Method-to-Axis Applicability Map](#3-method-to-axis-applicability-map)
4. [New Helpers Required in helpers.py](#4-new-helpers-required-in-helperspy)
5. [New Test Files -- Complete Specification](#5-new-test-files----complete-specification)
6. [Known Bug Regression Tests](#6-known-bug-regression-tests)
7. [Property-Based Invariants](#7-property-based-invariants)
8. [Implementation Sequencing](#8-implementation-sequencing)

---

## 1. Status Quo

### Current Coverage
- **1,047 tests** across **25 test files**, all passing
- **Line coverage: 99%** (944 statements, 7 missed)
- Missed lines:
  - `_chain.py:297` -- `_run_async` finally: async finally handler raises `_ControlFlowSignal` after awaiting
  - `_chain.py:305` -- `_run_async` finally: `finally_exc.__context__` is None and `_active_exc` is not None
  - `_core.py:112` -- Python <3.14 branch for `_create_task_fn` (version-conditional; uncoverable on 3.14)
  - `_ops.py:33` -- `_to_async` in `_make_with`: `_ControlFlowSignal` in async body + sync CM with awaitable `__exit__`
  - `_ops.py:84` -- `_await_exit_success` with `ignore_result=True` path
  - `_traceback.py:110` -- Python <3.11 branch for `_RAISE_CODE.replace` without `co_qualname` (version-conditional)
  - `_traceback.py:212` -- `_resolve_nested_chain`: nested chain link with explicit args that have a non-Null first element

### Existing Test Files (25)
| File | Focus | Tests |
|------|-------|-------|
| `core_tests.py` | `_Null`, `QuentException`, `Link`, `_resolve_value`, `_evaluate_value` | ~30 |
| `chain_construction_tests.py` | Constructor, fluency, link ordering | ~25 |
| `chain_run_tests.py` | `run()`, value propagation, falsy preservation, reuse | ~30 |
| `chain_async_tests.py` | Sync/async bridging, async detection, run values | ~20 |
| `calling_convention_tests.py` | All 6 calling conventions across methods | ~25 |
| `gather_tests.py` | `gather()` sync, async, exceptions, edge cases | ~15 |
| `finally_tests.py` | `finally_()` registration, execution, async, fire-and-forget | ~25 |
| `except_tests.py` | `except_()` registration, execution, async, fire-and-forget | ~35 |
| `iterate_tests.py` | `iterate()`/`iterate_do()`, `_Generator`, break/return | ~30 |
| `decorator_tests.py` | `decorator()` sync, async, control flow, reuse, except/finally | ~15 |
| `freeze_tests.py` | `freeze()`, `_FrozenChain`, concurrent, not-nested | ~12 |
| `repr_tests.py` | `__repr__` for Chain, FrozenChain, Generator, Null | ~25 |
| `nested_chain_tests.py` | Nested detection, value propagation, return/break, async | ~18 |
| `callable_matrix_tests.py` | All callable types in then/do/except_/finally_/root | ~35 |
| `monad_law_tests.py` | Left identity, right identity, associativity | ~10 |
| `interaction_tests.py` | Pairwise operation interactions | ~20 |
| `adversarial_tests.py` | Reentrant, sentinel abuse, threading, edge cases | ~35 |
| `foreach_tests.py` | `foreach()`/`foreach_do()` sync, async, break, exceptions | ~30 |
| `filter_tests.py` | `filter()` sync, async, edge cases | ~20 |
| `with_tests.py` | `with_()`/`with_do()` sync, async, dual protocol, contextlib | ~25 |
| `control_flow_tests.py` | `return_()`, `break_()`, signal escape, interaction with except/finally | ~25 |
| `async_transition_tests.py` | 3-tier sync/async matrix for foreach, filter, with_ | ~150 |
| `coverage_gap_tests.py` | Targeted uncovered-line tests | ~25 |
| `gap_audit_tests.py` | Critical/moderate/low priority gaps | ~30 |
| `traceback_format_tests.py` | Traceback visualization, frame cleaning, stringification | ~100+ |

### Identified Gaps in Existing Coverage

Despite 99% line coverage, the following **semantic** and **combinatorial** gaps exist:

1. **Remaining uncovered lines** (7 lines, 5 semantically distinct scenarios)
2. **No `pickle`/`copy`/`deepcopy` tests** for `_Null` singleton robustness
3. **No `__hash__`/`__eq__` tests** for `_Null` -- `_Null` has no `__eq__`/`__hash__`, defaults to identity-based
4. **No GC/weakref tests** for `_task_registry` lifecycle under memory pressure
5. **No test for `_ControlFlowSignal.__init__` skipping `super().__init__()`** -- verify `args` tuple is empty
6. **No test that `_evaluate_value` with nested chain + kwargs but empty args drops kwargs** (known bug 1)
7. **No test that `set_initial_values` leaks `do()` result** when `ignore_result=True` on the initial link (known bug 2)
8. **No test for `_run_async` missing `__quent_source_link__` stamp** (known bug 3)
9. **No test for `except_()` accepting non-BaseException types** (known bug 4)
10. **No test for `_modify_traceback` using `sys.exc_info()[1]` instead of `exc`** (known bug 5)
11. **No test for `_ensure_future` not closing coroutine on `RuntimeError`** (known bug 6)
12. **No test for `__quent_link_temp_args__` leaking across runs** (known bug 7)
13. **No `self.subTest()` parameterized matrices** for dense combinatorial coverage
14. **No tests for `iterate()` with async iterable input** (chain returns coroutine, `__aiter__` must be used)
15. **No test for `iterate()` exception in fn with traceback modification** (extra_links parameter)
16. **No test for `gather()` with all-async fns where `_to_async` collects indices** 
17. **No test for `_with_op` where sync CM `__exit__` returns awaitable on exception path** with `ignore_result=True`
18. **No test for `_run_async` finally block where async finally handler raises after body exception** -- context chaining
19. **No test for PEP 479 behavior** -- `StopIteration` raised inside a generator-based iterate step
20. **No test for `DualCM` in `with_do`** context
21. **No tests for `_Generator.__call__` with `*args` and `**kwargs`**
22. **No test for `Chain.__repr__` with except_ and finally_ links** (they are separate from main list)
23. **No test for traceback with `extra_links` parameter** (used by iterate)
24. **No test for `_resolve_nested_chain` with kwargs-only args** (no positional, only keyword)
25. **No test for `_clean_chained_exceptions` with circular exception references**
26. ANY MANY MANY MANY MORE GAPS

---

## 2. Testing Axes and Variant Matrix

### Axis A: Callable Type (11 variants)
| ID | Variant | Example |
|----|---------|---------|
| A1 | Regular function | `def fn(x): return x + 1` |
| A2 | Async function | `async def fn(x): return x + 1` |
| A3 | Lambda | `lambda x: x + 1` |
| A4 | Built-in function | `str`, `int`, `len`, `abs` |
| A5 | `functools.partial` | `partial(operator.add, 10)` |
| A6 | Class constructor | `Adder(x)` |
| A7 | Callable object (`__call__`) | `CallableObj()(x)` |
| A8 | Async callable object | `AsyncCallableObj()(x)` |
| A9 | Bound method | `BoundMethodHolder().method` |
| A10 | Nested Chain | `Chain().then(lambda x: x + 1)` |
| A11 | Frozen Chain | `Chain().then(lambda x: x + 1).freeze()` |

### Axis B: Calling Convention (6 variants)
| ID | Variant | Link state | Behavior |
|----|---------|------------|----------|
| B1 | No args (default) | `args=None, kwargs=None` | `fn(current_value)` or `fn()` if Null |
| B2 | Positional args | `args=(1, 2)` | `fn(1, 2)` |
| B3 | Kwargs only | `kwargs={'k': 'v'}` | `fn(k='v')` |
| B4 | Args + kwargs | `args=(1,), kwargs={'k': 2}` | `fn(1, k=2)` |
| B5 | Ellipsis | `args=(...,)` | `fn()` (no args, ignores current_value) |
| B6 | Ellipsis + trailing | `args=(..., 2, 3)` | `fn()` (trailing silently ignored) |

### Axis C: Sync/Async Execution Tier (3 variants)
| ID | Variant | Description |
|----|---------|-------------|
| C1 | Pure sync | All operations return non-awaitable values |
| C2 | Sync-to-async transition | Starts sync, fn returns awaitable mid-operation |
| C3 | Full async | Input is async iterable/CM, entire operation is async |

### Axis D: Current Value State (5 variants)
| ID | Variant | Value |
|----|---------|-------|
| D1 | Null (no value) | `Null` sentinel |
| D2 | None | Python `None` |
| D3 | Falsy non-None | `0`, `False`, `''`, `[]`, `{}`, `set()`, `b''`, `()`, `0.0`, `0j` |
| D4 | Truthy value | `42`, `'hello'`, `[1,2,3]` |
| D5 | Callable value | `lambda: 42`, `int` |

### Axis E: Error Condition (8 variants)
| ID | Variant | Description |
|----|---------|-------------|
| E1 | No error | Normal execution |
| E2 | `ValueError` (standard Exception subclass) | Most common test error |
| E3 | `TypeError` | Type mismatch |
| E4 | `KeyboardInterrupt` (BaseException, not Exception) | Non-catchable by default except_ |
| E5 | `StopIteration` | PEP 479 behavior, special in generators |
| E6 | `StopAsyncIteration` | Async iteration termination |
| E7 | `RuntimeError` | Generic runtime error |
| E8 | Custom exception class | User-defined hierarchy |

### Axis F: Control Flow Signal (4 variants)
| ID | Variant | Description |
|----|---------|-------------|
| F1 | No signal | Normal flow |
| F2 | `Chain.return_(v)` | Early exit with value |
| F3 | `Chain.return_()` | Early exit, no value (returns None) |
| F4 | `Chain.break_(v)` | Loop exit with value |
| F5 | `Chain.break_()` | Loop exit, no value (returns accumulated list) |

### Axis G: Context Manager Protocol (8 variants)
| ID | Variant | Description |
|----|---------|-------------|
| G1 | `SyncCM` | Standard sync CM |
| G2 | `AsyncCM` | Standard async CM |
| G3 | `DualCM` | Both `__enter__`/`__aenter__` |
| G4 | `SyncCMSuppresses` | `__exit__` returns `True` |
| G5 | `AsyncCMSuppresses` | `__aexit__` returns `True` |
| G6 | `SyncCMRaisesOnEnter` | `__enter__` raises |
| G7 | `SyncCMRaisesOnExit` | `__exit__` raises |
| G8 | `SyncCMWithAwaitableExit` | `__exit__` returns coroutine (False) |
| G9 | `SyncCMSuppressesAwaitable` | `__exit__` returns coroutine (True) |
| G10 | `AsyncCMRaisesOnExit` | `__aexit__` raises |
| G11 | `contextlib.contextmanager` | Generator-based CM |
| G12 | `contextlib.asynccontextmanager` | Async generator-based CM |
| G13 | `contextlib.nullcontext` | No-op CM |

### Axis H: Iterable Type (8 variants)
| ID | Variant | Description |
|----|---------|-------------|
| H1 | `list` | Standard list |
| H2 | `tuple` | Immutable sequence |
| H3 | `range` | Lazy sequence |
| H4 | `generator` (via `iter()`) | One-shot iterator |
| H5 | `set` | Unordered (single-element to avoid order issues) |
| H6 | `str` | Character iteration |
| H7 | `dict` | Key iteration |
| H8 | `AsyncRange` | Async iterable |
| H9 | `AsyncEmpty` | Empty async iterable |
| H10 | `AsyncRangeRaises` | Async iterable that raises mid-iteration |
| H11 | Consumed generator | Already-exhausted iterator |
| H12 | Partially consumed generator | Mid-stream iterator |

---

## 3. Method-to-Axis Applicability Map

| Method | A | B | C | D | E | F | G | H |
|--------|---|---|---|---|---|---|---|---|
| `Chain(v)` | A1-A11 | B1-B6 | -- | D1-D5 | -- | -- | -- | -- |
| `.then(v)` | A1-A11 | B1-B6 | C1-C2 | D1-D5 | E1-E8 | F1-F3 | -- | -- |
| `.do(fn)` | A1-A9 | B1-B6 | C1-C2 | D1-D5 | E1-E8 | F1-F3 | -- | -- |
| `.foreach(fn)` | A1-A3,A7-A8 | -- | C1-C3 | -- | E1-E3,E5,E7 | F2-F5 | -- | H1-H12 |
| `.foreach_do(fn)` | A1-A3,A7-A8 | -- | C1-C3 | -- | E1-E3,E5,E7 | F2-F5 | -- | H1-H12 |
| `.filter(fn)` | A1-A3,A7-A8 | -- | C1-C3 | -- | E1-E3,E7 | F2-F3 | -- | H1-H12 |
| `.gather(*fns)` | A1-A3,A7-A8 | -- | C1-C2 | D2-D5 | E1-E3,E7 | F2-F3 | -- | -- |
| `.with_(fn)` | A1-A3,A7-A8,A10 | B1-B5 | C1-C3 | -- | E1-E3,E7 | F2-F3 | G1-G13 | -- |
| `.with_do(fn)` | A1-A3,A7-A8,A10 | B1-B5 | C1-C3 | -- | E1-E3,E7 | F2-F3 | G1-G13 | -- |
| `.except_(fn)` | A1-A3,A5,A7-A8 | B1-B5 | C1-C2 | -- | E2-E4,E7 | -- | -- | -- |
| `.finally_(fn)` | A1-A3,A5,A7-A8 | B1-B5 | C1-C2 | D1-D5 | E1-E3,E7 | -- | -- | -- |
| `.iterate(fn)` | A1-A3,A7-A8 | -- | C1-C3 | -- | E1-E3,E7 | F2-F5 | -- | H1-H12 |
| `.iterate_do(fn)` | A1-A3,A7-A8 | -- | C1-C3 | -- | E1-E3,E7 | F2-F5 | -- | H1-H12 |
| `.decorator()` | -- | -- | C1-C2 | -- | E1-E2,E7 | F2-F3 | -- | -- |
| `.freeze()` | -- | -- | C1-C2 | -- | E1-E2,E7 | -- | -- | -- |
| `.return_(v)` | -- | B1-B5 | -- | -- | -- | -- | -- | -- |
| `.break_(v)` | -- | B1-B5 | -- | -- | -- | -- | -- | -- |
| `.run(v)` | A1-A11 | B1-B6 | C1-C2 | D1-D5 | E1-E8 | -- | -- | -- |

---

## 4. New Helpers Required in helpers.py

### New Callable Fixtures
```python
class StatefulCallable:
  """Tracks call count and arguments for verification."""
  def __init__(self):
    self.calls = []
  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return len(self.calls)

class AsyncStatefulCallable:
  """Async version of StatefulCallable."""
  def __init__(self):
    self.calls = []
  async def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return len(self.calls)
```

### New Context Manager Fixtures
```python
class SyncCMRaisesOnExitFrom:
  """__exit__ raises with `from exc` chaining."""
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_val is not None:
      raise RuntimeError('exit error') from exc_val
    return False

class AsyncCMRaisesOnEnter:
  """Async CM whose __aenter__ raises."""
  async def __aenter__(self):
    raise RuntimeError('async enter error')
  async def __aexit__(self, *args):
    return False

class SyncCMExitReturnsAwaitableOnException:
  """Sync CM whose __exit__ returns an awaitable only on exception."""
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      async def _exit():
        return False
      return _exit()
    return False

class TrackingCM:
  """CM that records enter/exit calls and arguments."""
  def __init__(self):
    self.entered = False
    self.exited = False
    self.exit_args = None
    self.enter_result = 'tracked_ctx'
  def __enter__(self):
    self.entered = True
    return self.enter_result
  def __exit__(self, *args):
    self.exited = True
    self.exit_args = args
    return False

class AsyncTrackingCM:
  """Async CM that records enter/exit calls."""
  def __init__(self):
    self.entered = False
    self.exited = False
    self.exit_args = None
    self.enter_result = 'async_tracked_ctx'
  async def __aenter__(self):
    self.entered = True
    return self.enter_result
  async def __aexit__(self, *args):
    self.exited = True
    self.exit_args = args
    return False

class SyncCMEnterReturnsNone:
  """CM whose __enter__ returns None."""
  def __enter__(self):
    return None
  def __exit__(self, *args):
    return False

class SyncCMEnterReturnsSelf:
  """CM whose __enter__ returns self."""
  def __enter__(self):
    return self
  def __exit__(self, *args):
    return False
```

### New Iterable Fixtures
```python
class RaisingIterable:
  """Sync iterable that raises at a specific index."""
  def __init__(self, n, raise_at):
    self.n = n
    self.raise_at = raise_at
  def __iter__(self):
    for i in range(self.n):
      if i == self.raise_at:
        raise RuntimeError('iteration error')
      yield i

class StopIterationIterable:
  """Iterable whose elements raise StopIteration when called."""
  def __init__(self, items, stop_at):
    self.items = items
    self.stop_at = stop_at
  def __iter__(self):
    return iter(self.items)

class InfiniteIterable:
  """Iterable that yields forever (for break testing)."""
  def __init__(self, start=0):
    self.start = start
  def __iter__(self):
    i = self.start
    while True:
      yield i
      i += 1

class AsyncInfiniteIterable:
  """Async iterable that yields forever."""
  def __init__(self, start=0):
    self.start = start
  def __aiter__(self):
    self._i = self.start
    return self
  async def __anext__(self):
    val = self._i
    self._i += 1
    return val
```

### New Exception Fixtures
```python
class CustomException(Exception):
  """Custom exception for testing exception type filtering."""
  pass

class CustomBaseException(BaseException):
  """Custom BaseException for testing BaseException handling."""
  pass

class NestedCustomException(CustomException):
  """Subclass of CustomException for inheritance testing."""
  pass
```

### New Utility Functions
```python
def make_tracker():
  """Return a callable that records calls and a list to inspect."""
  calls = []
  def fn(*args, **kwargs):
    calls.append((args, kwargs))
    return 'tracked'
  fn.calls = calls
  return fn

async def async_make_tracker():
  """Async version of make_tracker."""
  calls = []
  async def fn(*args, **kwargs):
    calls.append((args, kwargs))
    return 'tracked'
  fn.calls = calls
  return fn
```

---

## 5. New Test Files -- Complete Specification

### File 1: `tests/null_sentinel_tests.py`

**Purpose:** Exhaustive testing of `_Null` singleton semantics, identity, robustness under serialization/copy, and interactions with Python object protocols.

**Class: `TestNullSingleton`** (unittest.TestCase)

Methods:
- `test_identity_across_imports` -- Import `Null` from `quent`, `quent._core`, verify `is` identity
- `test_hash_is_stable` -- `hash(Null)` returns consistent value across calls
- `test_hash_differs_from_none` -- `hash(Null) != hash(None)`
- `test_bool_is_truthy` -- `bool(Null)` is True (default object truthiness)
- `test_repr_is_angle_bracket` -- `repr(Null) == '<Null>'`
- `test_str_fallback` -- `str(Null)` returns something (falls back to repr)
- `test_not_equal_to_any_falsy` -- `self.subTest()` over `[None, False, 0, '', [], {}, set(), 0.0, b'', (), frozenset(), 0j]`
- `test_not_equal_to_any_truthy` -- `self.subTest()` over `[1, True, 'x', [1], {1:2}]`
- `test_type_is_null_class` -- `type(Null).__name__ == '_Null'`
- `test_slots_enforced` -- `Null.__dict__` raises `AttributeError` (slots prevent arbitrary attributes)
- `test_no_eq_override` -- `_Null.__eq__` is `object.__eq__` (identity-based)
- `test_pickle_roundtrip_fails_gracefully` -- `pickle.dumps(Null)` / `pickle.loads` should either produce the same singleton or raise (verify no crash)
- `test_copy_returns_same_object` -- `copy.copy(Null) is Null`
- `test_deepcopy_returns_same_object` -- `copy.deepcopy(Null) is Null`
- `test_null_in_set` -- `{Null}` has length 1, `Null in {Null}` is True
- `test_null_as_dict_key` -- `{Null: 'val'}[Null] == 'val'`
- `test_isinstance_check` -- `isinstance(Null, _Null)` is True
- `test_cannot_instantiate_second` -- `_Null()` creates a different object (document that it is NOT enforced as singleton at class level)

**Class: `TestControlFlowSignalInternals`** (unittest.TestCase)

Methods:
- `test_return_init_skips_super` -- `_Return(42, (), {}).args` (the Exception.args tuple) is empty because `__init__` doesn't call `super().__init__()`
- `test_break_init_skips_super` -- Same for `_Break`
- `test_return_holds_value` -- `_Return(42, (1,), {'k': 2}).value == 42` and `.args_ == (1,)` and `.kwargs_ == {'k': 2}`
- `test_break_holds_value` -- Same for `_Break`
- `test_control_flow_signal_is_exception` -- `isinstance(_Return(Null, (), {}), Exception)` is True
- `test_control_flow_signal_hierarchy` -- `issubclass(_Return, _ControlFlowSignal)` and `issubclass(_Break, _ControlFlowSignal)`
- `test_return_is_catchable_by_except_exception` -- User code doing `except Exception` catches `_Return`
- `test_return_with_null_value` -- `_Return(Null, (), {}).value is Null`

---

### File 2: `tests/evaluate_value_exhaustive_tests.py`

**Purpose:** Dense combinatorial coverage of `_evaluate_value` and `_resolve_value` with `self.subTest()` parameterization across all callable types and calling conventions.

**Class: `TestEvaluateValueCallableMatrix`** (unittest.TestCase)

For each callable type in Axis A (A1-A9, excluding A10-A11 which are chain-specific), test against Axis B calling conventions:

Methods (using `self.subTest(callable_type=..., convention=...)`):
- `test_callable_type_x_convention_no_args` -- For each type, `Link(fn)` with `_evaluate_value(link, current_value)` 
- `test_callable_type_x_convention_pos_args` -- `Link(fn, (arg1, arg2))`
- `test_callable_type_x_convention_kwargs` -- `Link(fn, None, {'k': 'v'})`
- `test_callable_type_x_convention_args_kwargs` -- `Link(fn, (arg1,), {'k': 'v'})`
- `test_callable_type_x_convention_ellipsis` -- `Link(fn, (...,))`
- `test_callable_type_x_convention_ellipsis_trailing` -- `Link(fn, (..., 'extra'))`

**Class: `TestEvaluateValueChainNested`** (unittest.TestCase)

Methods:
- `test_nested_chain_no_args_passes_current_value` -- `link.is_chain=True`, no args: `v._run(current_value, None, None)`
- `test_nested_chain_ellipsis_passes_null` -- `link.is_chain=True`, args=(...,): `v._run(Null, None, None)`
- `test_nested_chain_with_args_first_is_value` -- `link.is_chain=True`, args=(5, 6): `v._run(5, (6,), kwargs or {})`
- `test_nested_chain_with_args_and_kwargs` -- `link.is_chain=True`, args=(5,), kwargs={'k': 1}: `v._run(5, (), {'k': 1})`
- `test_nested_chain_with_args_empty_and_kwargs_BUG1` -- Known bug: args=(), kwargs={'k': 1} falls through to `v._run(current_value, None, None)` -- kwargs dropped. Test must verify current (buggy) behavior, then after fix, verify correct behavior.
- `test_nested_chain_with_kwargs_only_BUG1` -- Same: args=None, kwargs={'k': 1} falls through.

**Class: `TestEvaluateValueNonCallable`** (unittest.TestCase)

Methods:
- `test_literal_int_returned` -- `Link(42)` returns `42`
- `test_literal_none_returned` -- `Link(None)` returns `None`
- `test_literal_false_returned` -- `Link(False)` returns `False`
- `test_literal_empty_string_returned` -- `Link('')` returns `''`
- `test_literal_list_returned` -- `Link([1, 2])` returns `[1, 2]`
- `test_literal_null_returned` -- `Link(Null)` returns `Null` (the sentinel as a value)
- `test_non_callable_with_args_still_called` -- `Link(42, (1,))`: `42(1)` raises TypeError (verifying args path taken)

**Class: `TestResolveValueDense`** (unittest.TestCase)

Methods (using `self.subTest()`):
- `test_callable_no_args_no_kwargs` -- `_resolve_value(fn, None, None)` calls `fn()` when callable
- `test_callable_with_args` -- `_resolve_value(fn, (1, 2), None)` calls `fn(1, 2)`
- `test_callable_with_kwargs` -- `_resolve_value(fn, None, {'k': 1})` calls `fn(k=1)`
- `test_callable_with_both` -- `_resolve_value(fn, (1,), {'k': 2})` calls `fn(1, k=2)`
- `test_callable_with_ellipsis` -- `_resolve_value(fn, (...,), None)` calls `fn()`
- `test_callable_with_ellipsis_and_kwargs` -- `_resolve_value(fn, (...,), {'k': 1})` calls `fn()` (kwargs ignored)
- `test_non_callable_no_args` -- `_resolve_value(42, None, None)` returns `42`
- `test_non_callable_with_args` -- `_resolve_value(42, (1,), None)` calls `42(1)` raises TypeError
- `test_empty_args_tuple_treated_as_no_args` -- `_resolve_value(fn, (), None)` calls `fn()` (empty tuple is falsy)
- `test_empty_kwargs_dict_treated_as_no_kwargs` -- `_resolve_value(fn, None, {})` calls `fn()` (empty dict is falsy)

---

### File 3: `tests/run_initial_value_tests.py`

**Purpose:** Exhaustive testing of the `set_initial_values` flag in `_run()` and `_run_async()`, including the known bug where `do()` result leaks.

**Class: `TestSetInitialValuesSync`** (unittest.TestCase)

Methods:
- `test_root_link_sets_root_value` -- `Chain(42).finally_(tracker).run()`, tracker receives 42
- `test_run_value_sets_root_value` -- `Chain().then(...).finally_(tracker).run(42)`, tracker receives 42
- `test_callable_root_evaluated_as_root_value` -- `Chain(lambda: 5).finally_(tracker).run()`, tracker receives 5
- `test_do_as_first_link_does_not_leak_BUG2` -- Known bug 2: `Chain(5).do(lambda x: 999).then(lambda x: x).run()` should return 5, not 999. Verify current behavior.
- `test_first_link_ignore_result_preserves_initial` -- After the bug fix, `do()` on root should not change current_value
- `test_multiple_do_links_preserve_value` -- `Chain(5).do(fn1).do(fn2).do(fn3).run()` returns 5
- `test_run_value_callable_result_is_root` -- `Chain().then(lambda x: x + 1).finally_(tracker).run(lambda: 10)`, root_value = 10
- `test_no_root_no_run_value_first_then_is_root` -- `Chain().then(lambda: 42).then(lambda x: x + 1).finally_(tracker).run()`, root_value = 42
- `test_empty_chain_root_value_is_null` -- `Chain().finally_(handler).run()`, handler receives Null -> called with no args

**Class: `TestSetInitialValuesAsync`** (unittest.IsolatedAsyncioTestCase)

Methods (mirror sync tests in async):
- `test_async_root_sets_root_value` -- `Chain(async_fn, 1).finally_(tracker).run()`, tracker receives 2
- `test_async_do_first_link_BUG2` -- Known bug: verify async path has same issue
- `test_async_transition_preserves_root_value` -- Root evaluated sync, transitions to async mid-chain, finally_ still receives original root
- `test_async_run_value_callable` -- `Chain().then(async_fn).finally_(tracker).run(lambda: 10)`, root_value = 10

---

### File 4: `tests/source_link_stamp_tests.py`

**Purpose:** Tests for `__quent_source_link__` first-write-wins stamping behavior, including known bug 3 where `_run_async` is missing the stamp.

**Class: `TestSourceLinkStampSync`** (unittest.TestCase)

Methods:
- `test_source_link_set_on_exception` -- Raise in chain step, verify `exc.__quent_source_link__` is set
- `test_source_link_first_write_wins` -- Inner chain raises, outer chain catches: source_link points to inner link
- `test_source_link_not_overwritten` -- Already-stamped exception re-entering chain: stamp preserved
- `test_source_link_set_for_root_link_exception` -- Exception in root link callable
- `test_source_link_set_for_nth_link_exception` -- Exception in 3rd link, verify points to 3rd link

**Class: `TestSourceLinkStampAsync`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_async_source_link_set_BUG3` -- Known bug 3: `_run_async` is missing `__quent_source_link__` stamp. Verify current (buggy) behavior, then after fix, verify stamp is set.
- `test_async_source_link_first_write_wins` -- Inner async chain raises, outer catches: stamp points to inner
- `test_async_transition_source_link_preserved` -- Sync root, async step raises: verify stamp set by sync path persists

---

### File 5: `tests/except_validation_tests.py`

**Purpose:** Tests for exception type validation in `except_()`, including known bug 4 where non-BaseException types are accepted.

**Class: `TestExceptTypeValidation`** (unittest.TestCase)

Methods:
- `test_single_valid_exception_type` -- `except_(handler, exceptions=ValueError)` works
- `test_multiple_valid_exception_types` -- `except_(handler, exceptions=[ValueError, TypeError])` works
- `test_tuple_of_valid_types` -- `except_(handler, exceptions=(ValueError, TypeError))` works
- `test_generator_of_valid_types` -- `except_(handler, exceptions=(t for t in [ValueError]))` works
- `test_set_of_valid_types` -- `except_(handler, exceptions={ValueError, TypeError})` works
- `test_none_uses_default` -- `except_(handler, exceptions=None)` uses `(Exception,)`
- `test_string_raises_type_error` -- `except_(handler, exceptions='ValueError')` raises TypeError
- `test_empty_list_raises_quent_exception` -- `except_(handler, exceptions=[])` raises QuentException
- `test_empty_tuple_raises_quent_exception` -- `except_(handler, exceptions=())` raises QuentException
- `test_int_as_type_BUG4` -- Known bug 4: `except_(handler, exceptions=42)` should raise, currently wraps to `(42,)` which fails at runtime isinstance
- `test_float_as_type_BUG4` -- Same with `3.14`
- `test_non_exception_class_as_type_BUG4` -- `except_(handler, exceptions=str)` -- str is not BaseException subclass. Currently accepted, fails at isinstance. After fix, should reject at registration.
- `test_base_exception_subclass_accepted` -- `except_(handler, exceptions=KeyboardInterrupt)` should work
- `test_custom_exception_hierarchy` -- `except_(handler, exceptions=CustomException)` catches `NestedCustomException`
- `test_exception_type_in_iterable_with_non_exception_BUG4` -- `except_(handler, exceptions=[ValueError, 42])` -- mixed valid/invalid

---

### File 6: `tests/traceback_modification_tests.py`

**Purpose:** Tests for `_modify_traceback` behavior, including known bug 5 where `sys.exc_info()[1]` is used instead of `exc`.

**Class: `TestModifyTracebackBug5`** (unittest.TestCase)

Methods:
- `test_exc_info_vs_exc_mismatch_BUG5` -- Known bug 5: when except_ handler re-raises a different exception, `sys.exc_info()[1]` may not be `exc`. Verify current behavior.
- `test_traceback_modified_on_sync_path` -- Verify `__quent__` flag is set
- `test_traceback_modified_on_async_path` -- Verify `__quent__` flag set in async
- `test_traceback_cleaned_for_chained_exceptions` -- `exc.__cause__` and `exc.__context__` both cleaned
- `test_traceback_cleaned_for_circular_context` -- Exception with circular `__context__` (via `seen` set)

**Class: `TestModifyTracebackVisualization`** (unittest.TestCase)

Methods:
- `test_chain_visualization_contains_chain_header` -- `Chain(` prefix in visualization
- `test_chain_visualization_arrow_on_source_link` -- `<----` marker on correct link
- `test_chain_visualization_nested_chain_indentation` -- Nested chain indented
- `test_chain_visualization_except_and_finally_shown` -- except_ and finally_ links in visualization
- `test_chain_visualization_with_extra_links` -- `extra_links` parameter used by iterate

---

### File 7: `tests/ensure_future_tests.py`

**Purpose:** Tests for `_ensure_future`, `_task_registry`, and fire-and-forget lifecycle, including known bug 6.

**Class: `TestEnsureFuture`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_task_added_to_registry` -- After `_ensure_future(coro)`, task in `_task_registry`
- `test_task_removed_on_completion` -- After task completes, no longer in registry
- `test_task_removed_on_exception` -- Task that raises is still removed from registry
- `test_multiple_tasks_tracked` -- Multiple concurrent tasks all tracked
- `test_no_event_loop_raises_runtime_error_BUG6` -- Known bug 6: when no event loop, `_ensure_future` should close the coroutine. Verify current behavior (coroutine leaks), then after fix, verify close() called.
- `test_registry_does_not_grow_unbounded` -- Create 100 tasks, all complete, registry returns to initial size
- `test_done_callback_is_discard` -- Verify `task.add_done_callback(_task_registry.discard)` works

---

### File 8: `tests/temp_args_cleanup_tests.py`

**Purpose:** Tests for `__quent_link_temp_args__` lifecycle, including known bug 7 where the attribute is never cleaned.

**Class: `TestTempArgsLifecycle`** (unittest.TestCase)

Methods:
- `test_temp_args_set_on_foreach_exception` -- `__quent_link_temp_args__` set with item/index
- `test_temp_args_set_on_filter_exception` -- Same for filter
- `test_temp_args_set_on_with_exception` -- Set with ctx value
- `test_temp_args_set_on_chain_exception_with_current_value` -- Set with current_value
- `test_temp_args_keyed_by_link_id` -- Different links get different entries keyed by `id(link)`
- `test_temp_args_not_cleaned_after_handling_BUG7` -- Known bug 7: after except_ handles the error, `__quent_link_temp_args__` still attached to exception
- `test_temp_args_accumulate_through_nesting_BUG7` -- Nested chains each add their own temp args, all accumulate
- `test_temp_args_first_write_does_not_overwrite` -- Multiple links setting args on same exception: both preserved (different keys)

---

### File 9: `tests/with_exhaustive_tests.py`

**Purpose:** Complete coverage of all `_make_with` paths: `_with_op`, `_to_async`, `_full_async`, `_await_exit_suppress`, `_await_exit_success`.

**Class: `TestWithOpSyncPaths`** (unittest.TestCase)

Methods (using `self.subTest()` over CM types):
- `test_sync_cm_sync_body_success` -- Normal path, exit called
- `test_sync_cm_sync_body_exception_not_suppressed` -- Exit returns False, exception propagates
- `test_sync_cm_sync_body_exception_suppressed` -- Exit returns True, returns None (with_) or outer_value (with_do)
- `test_sync_cm_sync_body_control_flow_signal` -- `_Return` in body: exit called with (None, None, None), signal re-raised
- `test_sync_cm_exit_raises_on_success` -- Exit raises RuntimeError on normal exit
- `test_sync_cm_exit_raises_on_body_exception` -- Both body and exit raise: exit_exc from body_exc
- `test_sync_cm_enter_raises` -- Enter raises, exit never called
- `test_with_do_preserves_outer_value` -- `with_do` returns original CM, not fn result

**Class: `TestWithToAsyncPaths`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_sync_cm_async_body_success` -- SyncCM + async body: _to_async path
- `test_sync_cm_async_body_exception_not_suppressed` -- Exit returns False
- `test_sync_cm_async_body_exception_suppressed` -- Exit returns True
- `test_sync_cm_async_body_control_flow_signal` -- _Return in async body
- `test_sync_cm_async_body_exit_raises` -- Exit raises during exception handling
- `test_sync_cm_awaitable_exit_false_on_exception` -- Exit returns awaitable(False) on exception: `_await_exit_suppress` path, not suppressed
- `test_sync_cm_awaitable_exit_true_on_exception` -- Exit returns awaitable(True): suppressed
- `test_sync_cm_awaitable_exit_on_success` -- Exit returns awaitable on success: `_await_exit_success` path
- `test_sync_cm_awaitable_exit_success_ignore_result_UNCOVERED` -- Line 84: `ignore_result=True` in `_await_exit_success` -- currently uncovered
- `test_with_do_to_async_preserves_outer_value` -- with_do + SyncCM + async body
- `test_with_do_to_async_exception_suppressed` -- with_do + SyncCMSuppressesAwaitable + async body raises

**Class: `TestWithFullAsyncPaths`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_async_cm_sync_body_success` -- AsyncCM + sync body
- `test_async_cm_async_body_success` -- AsyncCM + async body
- `test_async_cm_body_exception` -- AsyncCM + body raises: `__aexit__` called with exc info
- `test_async_cm_body_exception_suppressed` -- AsyncCMSuppresses + body raises
- `test_async_cm_body_control_flow_signal` -- _Return in body: __aexit__ called, signal re-raised after
- `test_async_cm_body_returns_null` -- Body returns Null: result is None (with_) or outer_value (with_do)
- `test_async_cm_aexit_raises` -- AsyncCMRaisesOnExit: error propagates
- `test_async_cm_aenter_raises` -- AsyncCMRaisesOnEnter: enter raises, exit not called
- `test_with_do_full_async_preserves_outer_value` -- with_do + AsyncCM + body
- `test_dual_protocol_cm_prefers_aenter` -- DualCM: `__aenter__` used over `__enter__`
- `test_dual_protocol_cm_with_do` -- DualCM with with_do

---

### File 10: `tests/foreach_exhaustive_tests.py`

**Purpose:** Complete coverage of all `_make_foreach` paths with dense parameterization.

**Class: `TestForeachSyncTier`** (unittest.TestCase)

Methods (using `self.subTest()` over iterable types):
- `test_foreach_each_iterable_type` -- `self.subTest(type=...)` over H1-H7: list, tuple, range, generator, set, str, dict
- `test_foreach_do_each_iterable_type` -- Same for foreach_do
- `test_foreach_empty_each_type` -- Empty versions of each type
- `test_foreach_single_element` -- Single element for each type
- `test_foreach_fn_receives_elements_in_order` -- Tracker records order
- `test_foreach_fn_result_collected` -- foreach: fn results in output list
- `test_foreach_do_fn_result_discarded` -- foreach_do: original items in output list
- `test_foreach_break_on_each_position` -- `self.subTest(break_at=...)` for positions 0, mid, last
- `test_foreach_break_with_value` -- Break with callable value, literal value, Null
- `test_foreach_return_in_fn` -- _Return propagates out of foreach
- `test_foreach_stop_iteration_in_fn` -- StopIteration ends iteration early (PEP 479 in generators)
- `test_foreach_exception_in_fn_sets_temp_args` -- Verify item and index in temp args
- `test_foreach_exception_in_iterable_itself` -- RaisingIterable: error during iteration

**Class: `TestForeachToAsyncTier`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_sync_iterable_async_fn_first_item` -- fn returns coroutine on first item: enters _to_async
- `test_sync_iterable_async_fn_mid_item` -- fn returns coroutine on 3rd of 5 items
- `test_sync_iterable_async_fn_last_item` -- fn returns coroutine on last item
- `test_to_async_break_with_sync_value` -- Break in _to_async path with sync value
- `test_to_async_break_with_async_value` -- Break with callable that returns coroutine
- `test_to_async_return_signal` -- _Return in _to_async path
- `test_to_async_exception_sets_temp_args` -- Exception in _to_async with item/index
- `test_to_async_stop_iteration` -- StopIteration in _to_async (caught by except StopIteration)
- `test_to_async_foreach_do_preserves_items` -- foreach_do in _to_async: original items collected
- `test_to_async_mixed_sync_async_results` -- Some items sync, some async, all collected

**Class: `TestForeachFullAsyncTier`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_async_iterable_sync_fn` -- AsyncRange + sync fn
- `test_async_iterable_async_fn` -- AsyncRange + async fn
- `test_async_iterable_empty` -- AsyncEmpty
- `test_async_iterable_break` -- Break in full_async path
- `test_async_iterable_break_with_async_value` -- Break with async value in full_async
- `test_async_iterable_return` -- _Return in full_async
- `test_async_iterable_exception_sets_temp_args` -- Exception with item/index
- `test_async_iterable_foreach_do` -- foreach_do with async iterable
- `test_async_iterable_raises_mid_iteration` -- AsyncRangeRaises

---

### File 11: `tests/filter_exhaustive_tests.py`

**Purpose:** Complete coverage of all `_make_filter` paths.

**Class: `TestFilterSyncTier`** (unittest.TestCase)

Methods:
- `test_filter_each_iterable_type` -- `self.subTest()` over H1-H7
- `test_filter_empty_each_type` -- Empty iterables
- `test_filter_all_pass` -- Predicate always True
- `test_filter_none_pass` -- Predicate always False
- `test_filter_mixed_truthiness` -- `[0, 1, None, '', 'a', [], [1]]` with `bool`
- `test_filter_preserves_order` -- Order maintained
- `test_filter_exception_sets_temp_args` -- item and index in temp args
- `test_filter_control_flow_signal_propagates` -- _Return in filter fn raises up to chain
- `test_filter_stop_iteration_in_predicate` -- StopIteration in predicate

**Class: `TestFilterToAsyncTier`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_sync_iterable_async_predicate` -- Predicate returns coroutine
- `test_to_async_exception_sets_temp_args` -- Exception in _to_async
- `test_to_async_control_flow_signal` -- _Return in predicate during _to_async
- `test_to_async_mixed_sync_async_predicate` -- Some items sync, some async

**Class: `TestFilterFullAsyncTier`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_async_iterable_sync_predicate` -- AsyncRange + sync predicate
- `test_async_iterable_async_predicate` -- AsyncRange + async predicate
- `test_async_iterable_empty` -- AsyncEmpty
- `test_async_iterable_exception_sets_temp_args`
- `test_async_iterable_control_flow_signal`

---

### File 12: `tests/gather_exhaustive_tests.py`

**Purpose:** Complete coverage of `_make_gather`, including coroutine cleanup on exception and `_to_async` index tracking.

**Class: `TestGatherSyncTier`** (unittest.TestCase)

Methods:
- `test_zero_fns` -- Empty gather returns `[]`
- `test_single_fn` -- Single fn
- `test_multiple_fns_order_preserved` -- 5 fns, verify order
- `test_fn_receives_current_value` -- All fns get same current_value
- `test_fn_raises_first` -- First fn raises, subsequent not called
- `test_fn_raises_middle` -- Middle fn raises, prior results discarded
- `test_fn_raises_last` -- Last fn raises
- `test_fn_returns_none` -- None is valid result
- `test_fn_returns_callable` -- Callable returned as value, not invoked
- `test_many_fns_20` -- 20 fns

**Class: `TestGatherAsyncTier`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_all_async_fns` -- All return coroutines
- `test_mixed_sync_async` -- Some sync, some async
- `test_single_async_fn` -- Single async fn triggers _to_async
- `test_preserves_order_with_async` -- Order maintained despite concurrent resolution
- `test_async_fn_raises` -- asyncio.gather propagates first exception
- `test_coroutine_cleanup_on_setup_exception` -- First fn creates coroutine, second fn raises during setup: coroutine closed
- `test_coroutine_cleanup_no_resource_warning` -- Verify no RuntimeWarning about unawaited coroutines
- `test_to_async_index_tracking` -- `_to_async` correctly maps indices to resolved values
- `test_many_async_fns` -- 20 async fns

---

### File 13: `tests/iterate_exhaustive_tests.py`

**Purpose:** Complete coverage of `_sync_generator`, `_async_generator`, `_aiter_wrap`, and `_Generator`.

**Class: `TestGeneratorObject`** (unittest.TestCase)

Methods:
- `test_call_creates_new_generator` -- `g(42)` returns new `_Generator`, not same object
- `test_call_preserves_fn_and_ignore_result` -- New generator has same fn, ignore_result
- `test_call_sets_run_args` -- `g(42, 1, k=2)._run_args == (42, (1,), {'k': 2})`
- `test_call_does_not_mutate_original` -- Original `_run_args` unchanged
- `test_default_run_args` -- `(Null, (), {})`
- `test_repr` -- `'<Quent._Generator>'`
- `test_has_iter_and_aiter` -- `__iter__` and `__aiter__` both present
- `test_reusable_sync_iteration` -- Multiple `list(g)` calls produce same result
- `test_reusable_async_iteration` -- Multiple async iterations produce same result

**Class: `TestSyncGenerator`** (unittest.TestCase)

Methods:
- `test_no_fn_yields_items` -- `iterate()` without fn
- `test_fn_transforms_items` -- `iterate(fn)` applies fn
- `test_iterate_do_discards_fn_result` -- `iterate_do(fn)` yields original items
- `test_iterate_do_no_fn` -- `iterate_do()` yields items unchanged
- `test_break_stops_iteration` -- `_Break` in fn stops generator
- `test_return_raises_quent_exception` -- `_Return` in fn raises QuentException
- `test_exception_in_fn_sets_temp_args` -- Exception sets item/index in temp args
- `test_exception_in_fn_modifies_traceback` -- `_modify_traceback` called with extra_links
- `test_exception_in_fn_without_link` -- When link is None, no temp args set
- `test_pep479_stop_iteration_in_fn` -- StopIteration in fn: in sync generator, PEP 479 converts to RuntimeError inside the generator
- `test_empty_iterable` -- No items yielded
- `test_string_iterable_yields_chars` -- Character-by-character

**Class: `TestAsyncGenerator`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_sync_chain_output_wrapped` -- Chain returns sync iterable, `_aiter_wrap` wraps it
- `test_async_chain_output_used_directly` -- Chain returns async iterable, used directly
- `test_chain_returns_coroutine_then_iterable` -- Chain run returns coroutine that resolves to iterable
- `test_fn_transforms_items_async` -- Async fn applied to items
- `test_fn_awaitable_result` -- Fn returns coroutine, awaited
- `test_iterate_do_async_discards` -- iterate_do with async fn
- `test_break_stops_async_iteration` -- _Break stops generator
- `test_return_raises_quent_exception_async` -- _Return raises QuentException
- `test_exception_in_fn_sets_temp_args_async` -- Exception sets temp args
- `test_exception_in_async_fn_modifies_traceback` -- Traceback modified
- `test_empty_async_iterable` -- No items

---

### File 14: `tests/run_async_coverage_tests.py`

**Purpose:** Tests targeting the remaining uncovered lines in `_run_async`, including finally handler paths.

**Class: `TestRunAsyncFinallyControlFlow`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_async_finally_handler_raises_control_flow_UNCOVERED` -- Line 297: async finally handler (its awaitable) raises `_ControlFlowSignal` after being awaited. Need: chain enters async path, then async finally handler's coroutine raises `_Return` when awaited.
- `test_async_finally_handler_raises_base_exception` -- Line 298-302: async finally handler raises BaseException when awaited. Verify temp args set and traceback modified.
- `test_async_finally_context_chaining_UNCOVERED` -- Line 305: `finally_exc.__context__` is None but `_active_exc` is not None. Need: body raises, except handler handles it (so `_active_exc` is set), then async finally handler raises a brand-new exception with no context.
- `test_async_finally_context_already_set` -- `finally_exc.__context__` is already set (not None), so line 305 is skipped.

**Class: `TestRunAsyncIgnoreResult`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_do_in_async_path_preserves_value` -- `.do(async_fn)` in async path: ignore_result=True
- `test_multiple_do_in_async_path` -- Multiple .do() in async path
- `test_async_initial_value_with_do` -- First link is .do() in async path: set_initial_values interaction

---

### File 15: `tests/uncovered_lines_tests.py`

**Purpose:** Direct tests for the 7 remaining uncovered lines.

**Class: `TestUncoveredLines`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_ops_line_33_to_async_control_flow_awaitable_exit` -- `_ops.py:33`: Sync CM + async body raises `_ControlFlowSignal` + `__exit__` returns awaitable. Need: SyncCMWithAwaitableExit + async body that raises `_Return`.
- `test_ops_line_84_await_exit_success_ignore_result` -- `_ops.py:84`: `_await_exit_success` with `ignore_result=True`. Need: `with_do` + SyncCMWithAwaitableExit + sync body that succeeds. The `__exit__` returns awaitable, triggering `_await_exit_success`, and `ignore_result=True`.
- `test_chain_line_297_async_finally_control_flow` -- `_chain.py:297`: Async finally handler coroutine raises `_ControlFlowSignal` when awaited. Need: chain in async path, async finally handler that raises `_Return`.
- `test_chain_line_305_finally_context_none_active_exc` -- `_chain.py:305`: Finally exception has no context but `_active_exc` is set. Need: body raises, except handler raises (setting `_active_exc`), async finally handler raises NEW exception (no implicit context).
- `test_traceback_line_212_resolve_nested_chain_with_args` -- `_traceback.py:212`: `_resolve_nested_chain` where link has args with non-Null first element. Need: Chain visualization for a nested chain that was called with explicit args. Trigger by creating a chain error in a nested chain called with args.
- `test_traceback_line_110_no_qualname` -- `_traceback.py:110`: Python <3.11 branch. Cannot cover on 3.11+. Mark as version-conditional skip.
- `test_core_line_112_python_314_eager_start` -- `_core.py:112`: Python <3.14 branch. Cannot cover on 3.14. Mark as version-conditional skip.

---

### File 16: `tests/pep479_stopiteration_tests.py`

**Purpose:** Tests for StopIteration semantics inside chain operations, verifying PEP 479 compliance and interaction with the `while True / next()` pattern.

**Class: `TestStopIterationInForeach`** (unittest.TestCase)

Methods:
- `test_stop_iteration_in_fn_ends_early` -- fn raises StopIteration: caught by `except StopIteration` in `_foreach_op`'s `while True` loop, returns accumulated list
- `test_stop_iteration_on_first_item` -- Returns `[]`
- `test_stop_iteration_on_last_item` -- Returns all but last
- `test_stop_iteration_with_message` -- StopIteration with value attribute: value ignored, list returned
- `test_runtime_error_from_generator_fn` -- If fn is a generator function and raises StopIteration, Python 3.7+ converts it to RuntimeError per PEP 479. But fn is not a generator here -- it's called as a regular function. StopIteration is caught directly.

**Class: `TestStopIterationInFilter`** (unittest.TestCase)

Methods:
- `test_stop_iteration_in_predicate` -- Predicate raises StopIteration: caught by `except StopIteration` in `_filter_op`'s `while True` loop
- `test_stop_iteration_on_first_item_filter` -- Returns `[]`

**Class: `TestStopIterationInIterate`** (unittest.TestCase)

Methods:
- `test_stop_iteration_in_iterate_fn_sync` -- In `_sync_generator`, fn raises StopIteration: per PEP 479, this becomes RuntimeError inside the generator. The generator terminates with error.
- `test_stop_iteration_in_iterate_fn_async` -- In `_async_generator`, same behavior.

---

### File 17: `tests/memory_reference_tests.py`

**Purpose:** Tests for memory management: task registry GC, chain reference cycles, weakref behavior.

**Class: `TestTaskRegistryLifecycle`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_registry_starts_empty_or_known` -- Record initial size
- `test_task_added_and_removed` -- Single task lifecycle
- `test_concurrent_tasks_all_tracked` -- 10 concurrent tasks
- `test_exception_task_still_removed` -- Task that raises is removed
- `test_registry_size_after_bulk_operations` -- Create 100 tasks, verify all removed
- `test_cancelled_task_removed` -- Cancel a task: done callback fires, removed

**Class: `TestChainReferenceSemantics`** (unittest.TestCase)

Methods:
- `test_chain_does_not_hold_run_value` -- After `run(42)`, chain does not retain 42
- `test_frozen_chain_wraps_same_object` -- `_FrozenChain._chain is chain`
- `test_nested_chain_is_nested_flag_persistent` -- Once is_nested set, stays set
- `test_link_holds_strong_reference_to_value` -- Link.v is same object as input
- `test_link_next_link_chain` -- Linked list traversal from first_link

---

### File 18: `tests/concurrent_safety_tests.py`

**Purpose:** Tests for concurrent access patterns: frozen chain from multiple async tasks, unfrozen chain reentrance, thread safety.

**Class: `TestFrozenChainConcurrency`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_concurrent_runs_same_frozen` -- 100 concurrent `frozen.run(i)` via asyncio.gather
- `test_concurrent_runs_with_delay` -- Concurrent with `asyncio.sleep` to stress interleaving
- `test_concurrent_runs_different_values` -- Verify each gets correct independent result
- `test_concurrent_runs_with_except` -- Frozen chain with except_, concurrent
- `test_concurrent_runs_with_finally` -- Frozen chain with finally_, concurrent

**Class: `TestUnfrozenChainReentrance`** (unittest.TestCase)

Methods:
- `test_recursive_same_chain` -- Chain calls itself recursively via step lambda
- `test_recursive_with_accumulation` -- Recursive chain accumulates result
- `test_chain_run_inside_chain_step` -- New chain created/run inside step
- `test_deep_recursion_50_levels` -- 50 levels of recursion

**Class: `TestThreadSafety`** (unittest.TestCase)

Methods:
- `test_frozen_chain_from_multiple_threads` -- 10 threads, each runs frozen.run(i)
- `test_chain_created_one_thread_run_another` -- Create in main, run in worker
- `test_frozen_chain_10_threads_100_runs_each` -- Stress test

---

### File 19: `tests/decorator_exhaustive_tests.py`

**Purpose:** Complete decorator coverage including edge cases.

**Class: `TestDecoratorCallingConventions`** (unittest.TestCase)

Methods:
- `test_decorated_fn_receives_args_kwargs` -- fn(a, b, c=0) called with positional and keyword args
- `test_decorated_fn_return_value_is_root` -- fn's return is the run value
- `test_decorated_fn_no_args` -- fn() with no args
- `test_decorated_fn_varargs` -- fn(*args, **kwargs)
- `test_preserves_name_and_doc` -- __name__, __doc__, __wrapped__
- `test_decorator_reusable_on_multiple_fns` -- Same decorator applied to 3 functions
- `test_decorator_with_empty_chain` -- `Chain().decorator()` -- fn result passes through

**Class: `TestDecoratorWithAllOperations`** (unittest.TestCase)

Methods:
- `test_decorator_with_then` -- Basic then step
- `test_decorator_with_do` -- Side-effect step
- `test_decorator_with_foreach` -- Foreach on fn result
- `test_decorator_with_filter` -- Filter on fn result
- `test_decorator_with_gather` -- Gather on fn result
- `test_decorator_with_with` -- Context manager on fn result
- `test_decorator_with_except` -- Exception handling
- `test_decorator_with_finally` -- Cleanup handler
- `test_decorator_with_nested_chain` -- Nested chain step
- `test_decorator_with_freeze` -- Decorated function uses frozen inner

**Class: `TestDecoratorAsync`** (unittest.IsolatedAsyncioTestCase)

Methods:
- `test_async_fn_with_sync_chain` -- Async fn, sync chain steps
- `test_sync_fn_with_async_chain` -- Sync fn, async chain steps
- `test_async_fn_with_async_chain` -- Both async
- `test_decorated_async_preserves_name` -- Name preservation for async

---

### File 20: `tests/freeze_exhaustive_tests.py`

**Purpose:** Complete freeze coverage including mutation-after-freeze and concurrent use.

**Class: `TestFreezeSemantics`** (unittest.TestCase)

Methods:
- `test_frozen_has_no_is_chain` -- `_FrozenChain` lacks `_is_chain`
- `test_frozen_as_then_step_not_nested` -- Link(frozen).is_chain is False
- `test_frozen_callable_alias` -- `frozen() == frozen.run()`
- `test_frozen_bool_true` -- `bool(frozen)` is True
- `test_frozen_repr_wraps_chain` -- `repr(frozen)` contains `Frozen(Chain(...))`
- `test_frozen_wraps_same_object` -- `frozen._chain is chain`
- `test_mutation_after_freeze_visible` -- Add step after freeze: frozen sees it (documented as UB)
- `test_except_after_freeze_visible` -- Add except_ after freeze
- `test_finally_after_freeze_visible` -- Add finally_ after freeze

**Class: `TestFreezeWithAllOperations`** (unittest.TestCase)

Methods (using `self.subTest()`):
- `test_frozen_with_then` -- Frozen chain with .then()
- `test_frozen_with_do` -- Frozen chain with .do()
- `test_frozen_with_foreach` -- Frozen chain with .foreach()
- `test_frozen_with_filter` -- Frozen chain with .filter()
- `test_frozen_with_gather` -- Frozen chain with .gather()
- `test_frozen_with_with` -- Frozen chain with .with_()
- `test_frozen_with_except` -- Frozen chain with .except_()
- `test_frozen_with_finally` -- Frozen chain with .finally_()
- `test_frozen_with_iterate` -- Frozen chain with .iterate()
- `test_frozen_with_nested_chain` -- Frozen chain with nested chain step

---

### File 21: `tests/repr_exhaustive_tests.py`

**Purpose:** Complete repr/stringification coverage including edge cases.

**Class: `TestChainReprExhaustive`** (unittest.TestCase)

Methods:
- `test_repr_with_every_operation_type` -- Chain with then, do, foreach, filter, gather, with_, with_do -- all in repr
- `test_repr_except_and_finally_NOT_in_main_repr` -- except_ and finally_ are separate, not in __repr__
- `test_repr_long_chain_100_steps` -- 100 steps, repr does not crash
- `test_repr_nested_chain_in_then` -- Nested chain shows Chain(...) in repr

**Class: `TestGetObjNameEdgeCases`** (unittest.TestCase)

Methods:
- `test_obj_with_name_raising_non_attribute_error` -- ObjWithBadName: falls to repr()
- `test_obj_with_bad_name_and_bad_repr` -- ObjWithBadNameAndRepr: falls to type().__name__
- `test_partial_shows_wrapped_name` -- `partial(add)` format
- `test_chain_shows_type_name` -- Chain object shows 'Chain'
- `test_none_shows_repr` -- `_get_obj_name(None)` -> `'None'`
- `test_builtin_function` -- `_get_obj_name(len)` -> `'len'`

---

### File 22: `tests/traceback_resolve_nested_tests.py`

**Purpose:** Coverage for `_resolve_nested_chain` and `_stringify_chain` with nested chains and extra_links.

**Class: `TestResolveNestedChain`** (unittest.TestCase)

Methods:
- `test_nested_chain_no_args` -- No args/kwargs: no nested_root_link created
- `test_nested_chain_with_positional_args_UNCOVERED` -- Line 212: args=(5, 6), first element is 5 (not Null): creates nested_root_link
- `test_nested_chain_with_ellipsis_args` -- args=(...,): first element is Ellipsis, not Null -- creates nested_root_link? No, Ellipsis is not Null. Verify.
- `test_nested_chain_kwargs_only_no_positional` -- kwargs={'k': 1}, no positional args: `_temp_args` is empty, `_temp_kwargs` is truthy. `_temp_v` would be from `_temp_args[0]` but `_temp_args` is empty... Actually `_temp_args or ()` is `()`, so `_temp_args` is `()`. Then `if _temp_args or _temp_kwargs` is True (because _temp_kwargs). Then `_temp_v = _temp_args[0] if _temp_args else Null` -> `Null`. Then `if _temp_v is not Null` is False. So line 212 is NOT reached. This is a gap.
- `test_nested_chain_found_flag_propagation` -- ctx.found propagated from nested to parent

**Class: `TestStringifyChainExtraLinks`** (unittest.TestCase)

Methods:
- `test_extra_links_displayed` -- extra_links parameter adds links to visualization
- `test_extra_links_arrow_marking` -- Arrow marks correct extra link if it is source
- `test_except_and_finally_in_visualization` -- on_except_link and on_finally_link shown in visualization
- `test_gather_fns_shown` -- Gather shows function names in visualization

---

### File 23: `tests/exception_context_chain_tests.py`

**Purpose:** Comprehensive exception chaining tests: __cause__, __context__, __suppress_context__ across all error-handling paths.

**Class: `TestExceptionChainingSync`** (unittest.TestCase)

Methods:
- `test_except_handler_raises_with_from` -- `raise X from exc`: `__cause__` set
- `test_except_handler_raises_without_from` -- Framework does `raise exc_ from exc`: `__cause__` set anyway
- `test_except_handler_raises_suppress_context_set` -- `__suppress_context__` is True
- `test_finally_raises_after_body_exception` -- Finally exc has body exc as `__context__`
- `test_finally_raises_after_except_exception` -- Finally exc has except exc as `__context__`, except has body as `__cause__`
- `test_finally_raises_after_success` -- Finally exc has no `__context__` or `__cause__`
- `test_with_exit_raises_from_body_exception` -- `raise exit_exc from body_exc`
- `test_no_circular_exception_references` -- Walk __context__ chain, verify no cycles

**Class: `TestExceptionChainingAsync`** (unittest.IsolatedAsyncioTestCase)

Methods (mirror sync tests):
- `test_async_except_handler_raises_with_from`
- `test_async_finally_raises_after_body_exception`
- `test_async_finally_raises_after_except_exception`
- `test_async_finally_context_chaining_when_active_exc_exists`

---

### File 24: `tests/property_invariant_tests.py`

**Purpose:** Property-based invariants verified with `self.subTest()` over many inputs.

**Class: `TestDoNeverChangesValue`** (unittest.TestCase)

Invariant: `do()` NEVER changes current_value regardless of what fn returns.

Methods:
- `test_do_preserves_value_dense` -- `self.subTest(fn_returns=...)` over `[None, 0, False, '', [], {}, 42, 'hello', lambda: 99, Null, Exception('x')]`
  For each: `Chain(input_val).do(lambda x: fn_returns).run() == input_val`
- `test_do_preserves_value_with_exception_in_fn` -- fn raises: do still preserves value (exception propagates, but value wasn't changed before the raise)
- `test_multiple_do_preserves` -- `Chain(5).do(f1).do(f2).do(f3).run() == 5`

**Class: `TestThenAlwaysReplaces`** (unittest.TestCase)

Invariant: `then()` ALWAYS replaces current_value with fn's result.

Methods:
- `test_then_replaces_dense` -- `self.subTest()` over many fn return values
- `test_then_replaces_with_none` -- `Chain(5).then(lambda x: None).run()` is None
- `test_then_replaces_with_false` -- `Chain(5).then(lambda x: False).run()` is False
- `test_then_replaces_with_callable` -- Result is a callable, not invoked

**Class: `TestForeachDoPreservesItems`** (unittest.TestCase)

Invariant: `foreach_do()` ALWAYS returns the original items, regardless of fn result.

Methods:
- `test_foreach_do_preserves_dense` -- `self.subTest()` over various fn return values
- `test_foreach_do_preserves_even_when_fn_returns_none` -- Original items preserved

**Class: `TestFinallyDoesNotAffectResult`** (unittest.TestCase)

Invariant: `finally_()` handler result NEVER affects the chain's return value.

Methods:
- `test_finally_result_ignored_dense` -- `self.subTest()` over handler return values
- `test_finally_result_ignored_sync` -- `Chain(42).finally_(lambda rv: 999).run() == 42`
- `test_finally_result_ignored_async` -- Same in async path

**Class: `TestMonadLawsDense`** (unittest.TestCase)

Invariant: Monad laws hold for all value types.

Methods:
- `test_left_identity_dense` -- `self.subTest()` over `[0, None, False, '', [], {}, 42, 'hello']`
- `test_right_identity_dense` -- Same values
- `test_associativity_dense` -- Multiple f/g combinations

**Class: `TestChainBoolAlwaysTrue`** (unittest.TestCase)

Invariant: `bool(Chain(...))` is ALWAYS True, regardless of root value.

Methods:
- `test_bool_dense` -- `self.subTest()` over `[None, False, 0, '', [], Chain(), Chain(0), Chain(False)]`

---

### File 25: `tests/negative_tests.py`

**Purpose:** Negative tests for invalid usage, wrong types, double registration, unsupported operations.

**Class: `TestDoubleRegistration`** (unittest.TestCase)

Methods:
- `test_double_except_raises` -- Second except_() call raises QuentException
- `test_double_finally_raises` -- Second finally_() call raises QuentException
- `test_except_message_mentions_one` -- Error message says "one"
- `test_finally_message_mentions_one` -- Error message says "one"

**Class: `TestInvalidTypes`** (unittest.TestCase)

Methods:
- `test_foreach_on_non_iterable_types` -- `self.subTest()` over `[42, 3.14, True, None, object()]`
- `test_filter_on_non_iterable_types` -- Same
- `test_with_on_non_cm` -- `self.subTest()` over `[42, 'hello', None, [1,2,3]]`
- `test_gather_with_non_callable` -- `gather(42)` raises TypeError at runtime
- `test_iterate_on_non_iterable` -- TypeError when iterating

**Class: `TestCallableWithCallNone`** (unittest.TestCase)

Methods:
- `test_object_with_call_none` -- `callable()` returns True, but calling raises TypeError
- `test_object_with_call_not_implemented` -- `__call__` raises NotImplementedError

**Class: `TestControlFlowInHandlers`** (unittest.TestCase)

Methods:
- `test_return_in_except_raises_quent_exception` -- QuentException with "control flow"
- `test_break_in_except_raises_quent_exception` -- Same
- `test_return_in_finally_raises_quent_exception` -- Same
- `test_break_in_finally_raises_quent_exception` -- Same
- `test_break_outside_foreach_raises_quent_exception` -- "cannot be used in this context"
- `test_break_in_filter_propagates_to_chain` -- QuentException

**Class: `TestBreakInNonForeachContext`** (unittest.TestCase)

Methods:
- `test_break_in_then_raises` -- QuentException
- `test_break_in_do_raises` -- QuentException
- `test_break_in_gather_fn_propagates` -- _Break propagates out of gather -> chain catches -> QuentException
- `test_break_in_with_body` -- _ControlFlowSignal path in _make_with: __exit__ called, signal re-raised -> chain catches _Break -> QuentException

---

## 6. Known Bug Regression Tests

Each known bug must have at least two tests:
1. A test that documents the **current (buggy) behavior** (to verify the bug exists before the fix)
2. A test that verifies the **expected (fixed) behavior** (to serve as regression test after the fix)

### Bug 1: `_evaluate_value` drops kwargs for nested chains when args empty
- **Location:** `_core.py:190-191`
- **File:** `tests/evaluate_value_exhaustive_tests.py::TestEvaluateValueChainNested`
- **Tests:** `test_nested_chain_with_args_empty_and_kwargs_BUG1`, `test_nested_chain_with_kwargs_only_BUG1`

### Bug 2: `set_initial_values` ignores `ignore_result`, leaking `do()` results
- **Location:** `_chain.py:158-159` (interaction with `set_initial_values` flag)
- **File:** `tests/run_initial_value_tests.py::TestSetInitialValuesSync`
- **Tests:** `test_do_as_first_link_does_not_leak_BUG2`, `test_first_link_ignore_result_preserves_initial`

### Bug 3: `_run_async` missing `__quent_source_link__` stamp
- **Location:** `_chain.py:272-280` (async except handler -- no stamp before line 274)
- **File:** `tests/source_link_stamp_tests.py::TestSourceLinkStampAsync`
- **Tests:** `test_async_source_link_set_BUG3`

### Bug 4: `except_()` doesn't validate exception types
- **Location:** `_chain.py:359-360` (else branch wraps non-types to tuple)
- **File:** `tests/except_validation_tests.py::TestExceptTypeValidation`
- **Tests:** `test_int_as_type_BUG4`, `test_float_as_type_BUG4`, `test_non_exception_class_as_type_BUG4`

### Bug 5: `_modify_traceback` uses `sys.exc_info()[1]` instead of `exc`
- **Location:** `_traceback.py:105`
- **File:** `tests/traceback_modification_tests.py::TestModifyTracebackBug5`
- **Tests:** `test_exc_info_vs_exc_mismatch_BUG5`

### Bug 6: `_ensure_future` doesn't close coroutine on `RuntimeError`
- **Location:** `_core.py:121-125`
- **File:** `tests/ensure_future_tests.py::TestEnsureFuture`
- **Tests:** `test_no_event_loop_raises_runtime_error_BUG6`

### Bug 7: `__quent_link_temp_args__` never cleaned
- **Location:** `_core.py:101-105`
- **File:** `tests/temp_args_cleanup_tests.py::TestTempArgsLifecycle`
- **Tests:** `test_temp_args_not_cleaned_after_handling_BUG7`, `test_temp_args_accumulate_through_nesting_BUG7`

### Bug 8: `_Null` not robust under pickle/copy
- **Location:** `_core.py:12-21`
- **File:** `tests/null_sentinel_tests.py::TestNullSingleton`
- **Tests:** `test_pickle_roundtrip_fails_gracefully`, `test_copy_returns_same_object`, `test_deepcopy_returns_same_object`

---

## 7. Property-Based Invariants

These are structural invariants that must hold universally. Each is tested with dense parameterization via `self.subTest()`.

| # | Invariant | Test Location |
|---|-----------|---------------|
| P1 | `do()` NEVER changes current_value | `property_invariant_tests.py::TestDoNeverChangesValue` |
| P2 | `then()` ALWAYS replaces current_value | `property_invariant_tests.py::TestThenAlwaysReplaces` |
| P3 | `foreach_do()` ALWAYS returns original items | `property_invariant_tests.py::TestForeachDoPreservesItems` |
| P4 | `finally_()` result NEVER affects chain result | `property_invariant_tests.py::TestFinallyDoesNotAffectResult` |
| P5 | `bool(Chain(...))` is ALWAYS True | `property_invariant_tests.py::TestChainBoolAlwaysTrue` |
| P6 | Monad left identity holds for all values | `property_invariant_tests.py::TestMonadLawsDense` |
| P7 | Monad right identity holds for all values | `property_invariant_tests.py::TestMonadLawsDense` |
| P8 | Monad associativity holds | `property_invariant_tests.py::TestMonadLawsDense` |
| P9 | `with_do()` ALWAYS returns the original CM value | `with_exhaustive_tests.py` (verified across all CM types) |
| P10 | `except_()` handler is called IFF an exception matching the filter is raised | `except_validation_tests.py` |
| P11 | `finally_()` handler is ALWAYS called (success or error) | `finally_tests.py` (existing, extended) |
| P12 | `_Return` propagates through unlimited nesting depth | `nested_chain_tests.py` (existing) |
| P13 | `_Break` is isolated to its foreach scope | `gap_audit_tests.py::ModerateTests::test_nested_foreach_break_isolation` (existing) |
| P14 | Frozen chain is safe for concurrent use | `concurrent_safety_tests.py::TestFrozenChainConcurrency` |
| P15 | `run(Null)` is equivalent to `run()` | `adversarial_tests.py::TestNullAsExplicitValue` (existing) |

---

## 8. Implementation Sequencing

### Phase 1: Helpers (1 file, no tests)
1. Add all new fixtures to `tests/helpers.py`

### Phase 2: Core and Sentinel (2 files)
2. `tests/null_sentinel_tests.py` -- No dependencies
3. `tests/evaluate_value_exhaustive_tests.py` -- Depends on helpers

### Phase 3: Chain Execution Engine (3 files)
4. `tests/run_initial_value_tests.py` -- Tests `_run()` internals
5. `tests/source_link_stamp_tests.py` -- Tests `__quent_source_link__`
6. `tests/run_async_coverage_tests.py` -- Tests `_run_async()` finally paths

### Phase 4: Operations (4 files)
7. `tests/foreach_exhaustive_tests.py` -- 3-tier foreach
8. `tests/filter_exhaustive_tests.py` -- 3-tier filter
9. `tests/gather_exhaustive_tests.py` -- Gather paths
10. `tests/with_exhaustive_tests.py` -- 3-tier with_

### Phase 5: Iteration and Generator (2 files)
11. `tests/iterate_exhaustive_tests.py` -- _Generator, sync/async generators
12. `tests/pep479_stopiteration_tests.py` -- StopIteration semantics

### Phase 6: Exception Handling (3 files)
13. `tests/except_validation_tests.py` -- Type validation
14. `tests/exception_context_chain_tests.py` -- Exception chaining
15. `tests/traceback_modification_tests.py` -- Traceback modification

### Phase 7: Infrastructure (3 files)
16. `tests/ensure_future_tests.py` -- Task registry
17. `tests/temp_args_cleanup_tests.py` -- Temp args lifecycle
18. `tests/memory_reference_tests.py` -- Memory management

### Phase 8: High-Level Features (3 files)
19. `tests/decorator_exhaustive_tests.py` -- Decorator
20. `tests/freeze_exhaustive_tests.py` -- Freeze
21. `tests/concurrent_safety_tests.py` -- Concurrency

### Phase 9: Visualization and Repr (2 files)
22. `tests/repr_exhaustive_tests.py` -- Repr edge cases
23. `tests/traceback_resolve_nested_tests.py` -- Nested chain stringification

### Phase 10: Invariants, Negatives, Uncovered (3 files)
24. `tests/property_invariant_tests.py` -- Property-based invariants
25. `tests/negative_tests.py` -- Negative/invalid usage
26. `tests/uncovered_lines_tests.py` -- Targeted uncovered lines

### Expected Test Count After Expansion

| Category | Existing | New | Total |
|----------|----------|-----|-------|
| Existing files (25) | 1,047 | 0 | 1,047 |
| null_sentinel_tests.py | 0 | ~25 | 25 |
| evaluate_value_exhaustive_tests.py | 0 | ~65 | 65 |
| run_initial_value_tests.py | 0 | ~15 | 15 |
| source_link_stamp_tests.py | 0 | ~10 | 10 |
| run_async_coverage_tests.py | 0 | ~10 | 10 |
| foreach_exhaustive_tests.py | 0 | ~40 | 40 |
| filter_exhaustive_tests.py | 0 | ~25 | 25 |
| gather_exhaustive_tests.py | 0 | ~20 | 20 |
| with_exhaustive_tests.py | 0 | ~35 | 35 |
| iterate_exhaustive_tests.py | 0 | ~30 | 30 |
| pep479_stopiteration_tests.py | 0 | ~10 | 10 |
| except_validation_tests.py | 0 | ~18 | 18 |
| exception_context_chain_tests.py | 0 | ~12 | 12 |
| traceback_modification_tests.py | 0 | ~12 | 12 |
| ensure_future_tests.py | 0 | ~8 | 8 |
| temp_args_cleanup_tests.py | 0 | ~10 | 10 |
| memory_reference_tests.py | 0 | ~12 | 12 |
| decorator_exhaustive_tests.py | 0 | ~20 | 20 |
| freeze_exhaustive_tests.py | 0 | ~20 | 20 |
| concurrent_safety_tests.py | 0 | ~15 | 15 |
| repr_exhaustive_tests.py | 0 | ~12 | 12 |
| traceback_resolve_nested_tests.py | 0 | ~10 | 10 |
| property_invariant_tests.py | 0 | ~25 | 25 |
| negative_tests.py | 0 | ~20 | 20 |
| uncovered_lines_tests.py | 0 | ~7 | 7 |
| **TOTAL** | **1,047** | **~506** | **~1,553** |

### Target Coverage After Implementation
- **Line coverage:** 100% (excluding version-conditional branches for Python <3.11 and <3.14)
- **Semantic coverage:** Every code path in every function covered by at least one test
- **Bug regression:** All 8 known bugs have explicit regression tests
- **Combinatorial coverage:** All applicable axis combinations tested via `self.subTest()`

---

### Critical Files for Implementation
- `/Users/user/Documents/quent/tests/helpers.py` - Must add ~15 new helper classes/functions before any new test file
- `/Users/user/Documents/quent/quent/_core.py` - Contains `_evaluate_value`, `_resolve_value`, `_Null`, `Link`, `_ensure_future` -- most-tested module
- `/Users/user/Documents/quent/quent/_chain.py` - Contains `_run`, `_run_async`, `except_()`, `finally_()`, `decorator()` -- execution engine with 2 uncovered lines
- `/Users/user/Documents/quent/quent/_ops.py` - Contains `_make_foreach`, `_make_filter`, `_make_gather`, `_make_with` -- 3-tier operations with 2 uncovered lines
- `/Users/user/Documents/quent/quent/_traceback.py` - Contains `_modify_traceback`, `_resolve_nested_chain`, `_stringify_chain` -- traceback system with 2 uncovered lines
