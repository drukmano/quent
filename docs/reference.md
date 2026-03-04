# API Reference

Complete API reference for all Quent classes and methods. All classes are importable from the `quent` package.

```python
from quent import Chain, Cascade, ChainAttr, CascadeAttr, FrozenChain, run, Null, QuentException
```

## Type Aliases

```python
type ResultOrAwaitable[T] = T | Awaitable[T]

type ChainLink[IN, OUT] = Callable[[IN], OUT]
type AnyLink[OUT, IN] = OUT | Callable[[IN], OUT]
type FuncT = Callable[..., Any]
```

`ResultOrAwaitable` is the return type of `.run()` and `.__call__()` -- it is either the result value directly or an `Awaitable` that resolves to it, depending on whether async operations were encountered.

---

## Chain

The core chain class. Each operation receives the result of the previous one.

### Constructor

```python
Chain(v=None, *args, **kwargs)
```

Initialize a chain with an optional root value. If `v` is callable, it is called with `*args, **kwargs` during execution and its return value becomes the root value. If `v` is not callable, it is used directly.

### Core Operations

#### .then()

```python
.then(v, *args, **kwargs) -> Self
```

Add an operation to the chain. If `v` is callable, it is called with the current value (plus `*args, **kwargs`) and its result becomes the new current value. If `v` is not callable, it replaces the current value directly.

#### .do()

```python
.do(fn, *args, **kwargs) -> Self
```

Add a side-effect operation. `fn` is called with the current value but its return value is **discarded**. The current value remains unchanged.

#### .root()

```python
.root() -> Self
.root(fn, *args, **kwargs) -> Self
```

Without arguments: reset the current value to the root value.

With arguments: call `fn` with the **root value** (not the current value). The result becomes the new current value.

#### .root_do()

```python
.root_do(fn, *args, **kwargs) -> Self
```

Call `fn` with the **root value** as a side effect. The return value is discarded.

### Execution

#### .run()

```python
.run(v=None, *args, **kwargs) -> ResultOrAwaitable
```

Execute the chain and return the result. If `v` is provided, it overrides the root value.

Returns either the result directly (sync) or an `Awaitable` (if any async operation was encountered).

#### .__call__()

```python
chain(v=None, *args, **kwargs) -> ResultOrAwaitable
```

Alias for `.run()`.

#### .__or__()

```python
chain | other -> Self | ResultOrAwaitable
```

Pipe operator. If `other` is a `run` instance, executes the chain. Otherwise, appends `other` as a `.then()` operation.

### Attribute Access

#### .attr()

```python
.attr(name) -> Self
```

Get attribute `name` from the current value.

#### .attr_fn()

```python
.attr_fn(name, *args, **kwargs) -> Self
```

Call method `name` on the current value with the given arguments.

### Conditionals

#### .if_()

```python
.if_(v, *args, **kwargs) -> Self
```

Execute `v` if the current value is truthy. If `v` is callable, it is called with the current value.

#### .else_()

```python
.else_(v, *args, **kwargs) -> Self
```

Execute `v` if the current value is falsy.

#### .if_not()

```python
.if_not(v, *args, **kwargs) -> Self
```

Execute `v` if the current value is falsy. Alias for `.else_()`.

#### .condition()

```python
.condition(fn, *args, **kwargs) -> Self
```

Execute the **next** link in the chain only if `fn(current_value)` returns a truthy value.

#### .not_()

```python
.not_() -> Self
```

Negate the current value (`not current_value`).

#### .eq()

```python
.eq(value) -> Self
```

Replace current value with `current_value == value`.

#### .neq()

```python
.neq(value) -> Self
```

Replace current value with `current_value != value`.

#### .is_()

```python
.is_(value) -> Self
```

Replace current value with `current_value is value`.

#### .is_not()

```python
.is_not(value) -> Self
```

Replace current value with `current_value is not value`.

#### .in_()

```python
.in_(value) -> Self
```

Replace current value with `current_value in value`.

#### .not_in()

```python
.not_in(value) -> Self
```

Replace current value with `current_value not in value`.

#### .isinstance_()

```python
.isinstance_(*types) -> Self
```

Replace current value with `isinstance(current_value, types)`.

#### .or_()

```python
.or_(value) -> Self
```

Replace current value with `current_value or value`.

#### .if_raise()

```python
.if_raise(exc) -> Self
```

Raise `exc` if the current value is truthy.

#### .else_raise()

```python
.else_raise(exc) -> Self
```

Raise `exc` if the current value is falsy.

#### .if_not_raise()

```python
.if_not_raise(exc) -> Self
```

Raise `exc` if the current value is falsy. Alias for `.else_raise()`.

### Loops and Iteration

#### .foreach()

```python
.foreach(fn) -> Self
```

Apply `fn` to each item in the current value (which must be iterable). The result of iterating is passed forward.

#### .foreach_do()

```python
.foreach_do(fn) -> Self
```

Apply `fn` to each item as a side effect. The current value remains unchanged.

#### .iterate()

```python
.iterate(fn=None) -> Iterator | AsyncIterator
```

Yield chain results as an iterator. If `fn` is provided, it is applied to each item before yielding. Returns an async iterator if async operations are encountered.

#### .iterate_do()

```python
.iterate_do(fn=None) -> Iterator | AsyncIterator
```

Like `.iterate()` but as a side effect -- the original items are yielded, not the results of `fn`.

#### .while_true()

```python
.while_true(fn, *args, max_iterations=0, **kwargs) -> Self
```

Loop while `fn` returns a truthy value. If `max_iterations > 0`, the loop stops after that many iterations.

#### .filter()

```python
.filter(fn, *args, **kwargs) -> Self
```

Filter the current value (iterable) using `fn` as a predicate.

#### .reduce()

```python
.reduce(fn, initial=..., *args, **kwargs) -> Self
```

Reduce the current value (iterable) using `fn`. If `initial` is provided, it is used as the starting value.

#### .gather()

```python
.gather(*fns) -> Self
```

Execute multiple functions concurrently on the current value and collect results.

### Context Managers

#### .with_()

```python
.with_(fn, *args, **kwargs) -> Self
```

Use the current value as a context manager and execute `fn` inside it. The result of `fn` is passed forward.

#### .with_do()

```python
.with_do(fn, *args, **kwargs) -> Self
```

Use the current value as a context manager and execute `fn` inside it as a side effect. The result is discarded.

### Error Handling

#### .except_()

```python
.except_(fn, *args, exceptions=None, reraise=True, **kwargs) -> Self
```

Register an exception handler.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | -- | Handler function, receives the exception |
| `exceptions` | `tuple \| type \| None` | `None` | Exception types to catch (None = all) |
| `reraise` | `bool` | `True` | Re-raise after handling? |

#### .suppress()

```python
.suppress(*exceptions) -> Self
```

Suppress specific exception types. The chain returns `None` if a suppressed exception occurs.

#### .finally_()

```python
.finally_(fn, *args, **kwargs) -> Self
```

Register a cleanup handler that always runs, regardless of success or failure.

#### .on_success()

```python
.on_success(fn, *args, **kwargs) -> Self
```

Register a handler that runs only when the chain completes without raising an exception.

### Resilience

#### .retry()

```python
.retry(count, *, delay=0.0, exceptions=None) -> Self
```

Retry the chain on failure. Raises `ExceptionGroup` when all attempts are exhausted.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `count` | `int` | -- | Number of retry attempts |
| `delay` | `float` | `0.0` | Delay between attempts (seconds) |
| `exceptions` | `tuple \| type \| None` | `None` | Exception types to retry on (None = all) |

#### .timeout()

```python
.timeout(delay) -> Self
```

Enforce a time limit on async chain execution (in seconds).

#### .safe_run()

```python
.safe_run(v=None, *args, **kwargs) -> ResultOrAwaitable
```

Execute the chain via automatic clone, making it safe for concurrent use.

### Configuration

#### .config()

```python
.config(*, autorun=None, debug=None, timeout=None, retry=None) -> Self
```

Set multiple configuration options at once. The `retry` parameter accepts either an `int` or a `dict` with keys matching `.retry()` parameters.

#### .autorun()

```python
.autorun(autorun=True) -> Self
```

When enabled, async chain results are automatically scheduled via `asyncio.create_task`.

#### .with_context()

```python
.with_context(**context) -> Self
```

Attach context metadata (key-value pairs) to the chain, accessible via `Chain.get_context()`.

### Chain Reuse

#### .clone()

```python
.clone() -> Self
```

Create a deep copy of the chain for independent reuse.

#### .freeze()

```python
.freeze() -> FrozenChain
```

Create an immutable, callable `FrozenChain` snapshot.

#### .decorator()

```python
.decorator() -> Callable[[FuncT], FuncT]
```

Return a decorator that uses this chain. The decorated function becomes the root value.

### Flow Control (Class Methods)

#### Chain.return_()

```python
@classmethod
Chain.return_(v=None, *args, **kwargs) -> None
```

Exit the chain early, returning `v` as the result. If `v` is callable, it is called with `*args, **kwargs`.

#### Chain.break_()

```python
@classmethod
Chain.break_(v=None, *args, **kwargs) -> None
```

Break out of a `.foreach()` or `.while_true()` loop. If `v` is provided, it becomes the loop's result.

#### Chain.null()

```python
@classmethod
Chain.null() -> Null
```

Return the `Null` sentinel.

#### Chain.get_context()

```python
@staticmethod
Chain.get_context() -> dict
```

Retrieve the current chain context (set via `.with_context()`).

#### Chain.compose()

```python
@staticmethod
Chain.compose(*chains) -> Chain
```

Combine multiple chains into a single chain.

### Miscellaneous

#### .sleep()

```python
.sleep(delay) -> Self
```

Insert an async delay (in seconds) into the chain.

#### .raise_()

```python
.raise_(exc) -> Self
```

Unconditionally raise `exc` at this point in the chain.

#### .pipe()

```python
.pipe(other) -> Self
```

Pipe the current value into `other`.

#### .set_async()

```python
.set_async(enabled=True) -> Self
```

Force the chain into async mode.

#### .__bool__()

```python
bool(chain) -> bool
```

Returns `True` if the chain has any operations defined.

#### .__repr__()

```python
repr(chain) -> str
```

String representation of the chain.

---

## Cascade

```python
class Cascade(Chain)
```

Chain variant where every operation receives the **root value** instead of the previous result. The final result is always the root value.

Inherits all methods from `Chain`. The only difference is in how values are passed between operations.

### Constructor

```python
Cascade(v=None, *args, **kwargs)
```

Same as `Chain.__init__()`.

---

## ChainAttr

```python
class ChainAttr(Chain)
```

Chain with dynamic attribute access via `__getattr__`. Accessing an attribute on the chain appends an attribute-access operation. Calling the accessed attribute appends a method-call operation.

```python
result = ChainAttr("hello world").upper().split().run()
# ['HELLO', 'WORLD']
```

### .__getattr__()

```python
chain_attr.some_attribute -> Self
```

Appends an operation that accesses `some_attribute` on the current value.

### .__call__()

```python
chain_attr(*args, **kwargs) -> Self | ResultOrAwaitable
```

If the last operation was an attribute access, converts it to a method call with the given arguments. Otherwise, acts as `.run()`.

---

## CascadeAttr

```python
class CascadeAttr(ChainAttr)
```

Combines `Cascade` value-passing semantics with `ChainAttr`'s dynamic attribute access.

```python
result = CascadeAttr(list()).append(1).append(2).extend([3, 4]).run()
# [1, 2, 3, 4]
```

### Constructor

```python
CascadeAttr(v=None, *args, **kwargs)
```

---

## FrozenChain

```python
class FrozenChain
```

Immutable callable chain snapshot, created by `Chain.freeze()`. Cannot be modified but can be called repeatedly.

### .run()

```python
.run(v=None, *args, **kwargs) -> ResultOrAwaitable
```

Execute the frozen chain with an optional root value.

### .__call__()

```python
frozen(v=None, *args, **kwargs) -> ResultOrAwaitable
```

Alias for `.run()`.

### .decorator()

```python
.decorator() -> Callable[[FuncT], FuncT]
```

Return a decorator that uses this frozen chain.

---

## run

```python
class run
```

Helper class for pipe operator termination. When piped into a chain, triggers execution.

### Constructor

```python
run(v=None, *args, **kwargs)
```

Optional arguments override the root value when the chain executes.

### Usage

```python
from quent import Chain, run

result = Chain(fetch_data, id) | validate | transform | save | run()
result = Chain().then(validate).then(transform) | run(fetch_data, id)
```

---

## Null

Sentinel value representing "no value", distinct from `None`. Useful when `None` is a valid value in your domain.

```python
from quent import Null

result = chain.run()
if result is Null:
  print("No value was produced")
```

Also accessible via `Chain.null()`.

---

## QuentException

```python
class QuentException(Exception)
```

Base exception class for Quent-specific errors.

---

