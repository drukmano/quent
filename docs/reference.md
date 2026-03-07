# API Reference

Complete API reference for all Quent classes and methods. All classes are importable from the `quent` package.

```python
from quent import Chain, Null, QuentException
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
Chain(v=Null, /, *args, **kwargs)
```

Initialize a chain with an optional root value. If `v` is callable, it is called with `*args, **kwargs` during execution and its return value becomes the root value. If `v` is not callable, it is used directly. If no root value is provided, the first link in the chain becomes the root.

### Core Operations

#### .then()

```python
.then(v, /, *args, **kwargs) -> Chain
```

Add an operation to the chain. If `v` is callable, it is called with the current value (plus `*args, **kwargs`) and its result becomes the new current value. If `v` is not callable, it replaces the current value directly.

#### .do()

```python
.do(fn, /, *args, **kwargs) -> Chain
```

Add a side-effect operation. `fn` is called with the current value but its return value is **discarded**. The current value remains unchanged.

### Execution

#### .run()

```python
.run(v=Null, /, *args, **kwargs) -> ResultOrAwaitable
```

Execute the chain and return the result. If `v` is provided, it overrides the root value.

Returns either the result directly (sync) or an `Awaitable` (if any async operation was encountered).

#### .__call__()

```python
chain(v=Null, /, *args, **kwargs) -> ResultOrAwaitable
```

Alias for `.run()`.

### Iteration

#### .map()

```python
.map(fn, /) -> Chain
```

Apply `fn` to each item in the current value (which must be iterable). The results of calling `fn` on each element are collected and passed forward.

#### .foreach()

```python
.foreach(fn, /) -> Chain
```

Apply `fn` to each item as a side effect. The original items (not `fn`'s return values) are collected and passed forward.

#### .filter()

```python
.filter(fn, /) -> Chain
```

Filter the current value (iterable) using `fn` as a predicate. Elements where `fn` returns a truthy value are kept.

#### .gather()

```python
.gather(*fns) -> Chain
```

Execute multiple functions concurrently on the current value and collect results. Each function receives the current value as its argument. If any function is async, all are run concurrently via `asyncio.gather`.

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

### Context Managers

#### .with_()

```python
.with_(fn, /, *args, **kwargs) -> Chain
```

Use the current value as a context manager and execute `fn` inside it. The result of `fn` becomes the new current value.

#### .with_do()

```python
.with_do(fn, /, *args, **kwargs) -> Chain
```

Use the current value as a context manager and execute `fn` inside it as a side effect. The result is discarded; the current value remains unchanged.

### Error Handling

#### .except_()

```python
.except_(fn, /, *args, exceptions=None, **kwargs) -> Chain
```

Register an exception handler. Only **one** handler is allowed per chain -- calling `.except_()` a second time raises `QuentException`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable` | -- | Handler function. Receives the exception as its first argument. |
| `*args` | `Any` | -- | Additional positional arguments passed to the handler |
| `exceptions` | `type \| Iterable[type] \| None` | `None` | Exception types to catch. `None` catches all `Exception` subclasses. |
| `**kwargs` | `Any` | -- | Additional keyword arguments passed to the handler |

When an exception occurs:

- If `exceptions` is set and the exception matches, the handler runs and its return value becomes the chain's result (the exception is swallowed).
- If `exceptions` is set but the exception does not match, the exception propagates normally -- the handler is not called.
- If `exceptions` is `None`, the handler catches all `Exception` subclasses.

#### .finally_()

```python
.finally_(fn, /, *args, **kwargs) -> Chain
```

Register a cleanup handler that **always runs**, regardless of whether the chain succeeded or raised an exception. Only **one** handler is allowed per chain -- calling `.finally_()` a second time raises `QuentException`.

The handler receives the **root value** (not the current value) as its argument. Use the ellipsis (`...`) as the first argument to call it with no arguments:

```python
# cleanup(root_value) -- called with the root value
Chain(data).then(process).finally_(cleanup)

# cleanup() -- called with no arguments
Chain(data).then(process).finally_(cleanup, ...)
```

### Chain Reuse

#### .freeze()

```python
.freeze() -> FrozenChain
```

Create an immutable, callable `FrozenChain` snapshot. The original chain must not be modified after freezing.

#### .decorator()

```python
.decorator() -> Callable
```

Return a decorator that uses this chain. The decorated function becomes the root value.

```python
@Chain().then(validate).then(save).decorator()
def process(data):
  return data
```

When `process(data)` is called, the chain executes with the decorated function's return value as the root.

### Flow Control (Class Methods)

#### Chain.return_()

```python
@classmethod
Chain.return_(v=Null, /, *args, **kwargs) -> NoReturn
```

Exit the chain early, returning `v` as the result. If `v` is callable, it is called with `*args, **kwargs`.

#### Chain.break_()

```python
@classmethod
Chain.break_(v=Null, /, *args, **kwargs) -> NoReturn
```

Break out of a `.map()` loop. If `v` is provided, it becomes the loop's result.

### Special Methods

#### .__bool__()

```python
bool(chain) -> True
```

Always returns `True`.

#### .__repr__()

```python
repr(chain) -> str
```

String representation of the chain showing its structure, e.g. `Chain(fetch_data).then(...).do(...)`.

---

## FrozenChain

```python
class FrozenChain
```

Immutable callable chain snapshot, created by `Chain.freeze()`. Cannot be modified but can be called repeatedly. Safe for concurrent use -- each call operates on a fresh execution state.

### .run()

```python
.run(v=Null, /, *args, **kwargs) -> ResultOrAwaitable
```

Execute the frozen chain with an optional root value.

### .__call__()

```python
frozen(v=Null, /, *args, **kwargs) -> ResultOrAwaitable
```

Alias for `.run()`.

### .__bool__()

```python
bool(frozen) -> True
```

Always returns `True`.

### .__repr__()

```python
repr(frozen) -> str
```

String representation, e.g. `Frozen(Chain(fetch_data).then(...))`.

---

## Null

Sentinel value representing "no value", distinct from `None`. Useful when `None` is a valid value in your domain.

```python
from quent import Null

result = chain.run()
if result is Null:
  print('No value was produced')
```

---

## QuentException

```python
class QuentException(Exception)
```

Base exception class for Quent-specific errors. Raised when chain constraints are violated (e.g., registering a second except handler, control flow signals escaping the chain).

---
