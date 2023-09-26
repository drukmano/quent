# Quent
#### Yet Another Chain Interface.


## Installation
```
pip install quent
```

## Table of Contents
- [Introduction](#introduction)
- [Real World Example](#real-world-example)
- [Details & Examples](#details--examples)
  - [Literal Values](#literal-values)
  - [Custom Arguments (`args`, `kwargs`, `Ellipsis`)](#custom-arguments)
  - [Flow Modifiers](#flow-modifiers)
  - [Chain Template / Reuse](#reusing-a-chain)
  - [Chain Nesting](#nesting-a-chain)
  - [Pipe Syntax](#pipe-syntax)
- [API](#api)
  - [Core](#core)
  - [`except`, `finally`](#callbacks)
  - [Conditionals](#conditionals)
- [Cascade](#cascade)
- [Direct Attribute Access](#direct-attribute-access)
- [Limitations](#limitations)

**Suggestions and contributions are more than welcome.**

## Introduction
Quent is an [enhanced](#core), [chain interface](https://en.wikipedia.org/wiki/Method_chaining) implementation for Python. As opposed to
other simple chain implementations, Quent seamlessly handles coroutines.

Quent is written in C (using Cython) to minimize it's overhead as much as possible.

As an example, take this function:
```python
async def handle_request(id):
  data = await fetch_data(id)
  data = validate_data(data)
  data = normalize_data(data)
  return await send_data(data)
```
It uses intermediate variables that only serve to make to code more readable, as opposed to:
```python
async def handle_request(id):
  return await send_data(normalize_data(validate_data(await fetch_data(id))))
```

With Quent, we can chain these operations:
```python
from quent import Chain

async def handle_request(id):
  return await Chain(fetch_data, id).then(validate_data).then(normalize_data).then(send_data).run()
```
Once there is at least one coroutine in the chain, `.run()` will return a coroutine.
This opens up a lot of new ways to write elegant and readable code.

Besides `Chain`, Quent provides the [Cascade](#cascade) class which implements the [fluent interface](https://en.wikipedia.org/wiki/Fluent_interface).

Quent aims to provide you with all the necessary tools that you may need to write better code.
Visit [API](#api) to see the full power of Quent.

## Real World Example
This snippet is taken from a thin Redis wrapper I wrote, which supports both the sync and async versions
of `redis` without having a separate implementation for the async version.
```python
def flush(self) -> Any | Coroutine:
  """ Execute the current pipeline and return the results, excluding
      the results of inline 'expire' commands.
  """
  pipe = self.r.pipeline(transaction=self.transaction)
  # this applies a bunch of Redis operations onto the `pipe` object.
  self.apply_operations(pipe)
  return (
    Chain(pipe.execute, raise_on_error=True)
    .then(self.remove_ignored_commands)
    .finally_(pipe.reset, ...)
    .run()
  )
```
Once the chain runs, it will execute the pipeline commands, remove the unwanted results, and return the rest
of them. Finally, it will reset the `pipe` object. Any function passed to `.finally_()` will **always** be invoked,
even if an exception has been raised during the execution of the chain. The purpose of the `...` here is explained
in the [Ellipsis](#ellipsis) section.

`pipe.execute` and `pipe.reset` are both performing a network request to Redis, and in the case of an `async` Redis
object - are coroutines and would have to be awaited.
Notice that I return without an explicit `await` - if the user of this wrapper has initialized the class with an
async `Redis` instance, they will know that they need to `await` it. This allows me to focus on the actual logic,
without caring about `sync` vs `async`.

Some would say that this pattern can cause unexpected behavior, since it isn't clear when it
will return a coroutine or not. I see it no differently than any undocumented code - with a proper
and clear documentation (be it an external documentation or just a simple docstring), there shouldn't be
any *unexpected* behavior.

## Details & Examples
### Literal Values
You don't have to pass a callable as a chain item - literal values works just as well.
```python
Chain(fetch_data, id).then(True).run()
```
will execute `fetch_data(id)`, and then return `True`.

### Custom Arguments
You may provide `args` or `kwargs` to a chain item - by doing so, Quent assumes that the item is a callable
and will evaluate it with the provided arguments, instead of evaluating it with the current chain value.
```python
Chain(fetch_data, id).then(fetch_data, another_id, password=password).run()
```
will execute `fetch_data(id)`, and then `fetch_data(another_id, password=password)`.
#### Ellipsis
The `Ellipsis` / `...` is a special case - if the first argument for *most* functions that register a chain item
or a callback is `...`,
the item will be evaluated without any arguments.
```python
Chain(fetch_data, id).then(do_something, ...).run()
```
will execute `fetch_data(id)`, and then `do_something()`.

### Flow Modifiers
While the default operation of a chain is to, well, chain operations (using `.then()`), there are cases where you may
want to break out of this flow. For this, `Chain` provides the functions `.root()` and `.ignore()`.
They both behave like `.then()`, but with a small difference:

- `.root()` evaluates the item using the root value, instead of the current chain value.
- `.ignore()` evaluates the item with the current chain value but will not propagate its result forwards.

There is also a `.root_ignore()` which is the combination of `.root()` and `.ignore()`.

### Reusing A Chain
You may reuse a chain as many times as you wish.
```python
chain = Chain(fetch_data, id).then(validate_data).then(normalize_data).then(send_data)
chain.run()
chain.run()
...
```

There are many cases where you may need to apply the same sequence of operations but with different inputs. Take our
previous example:
```python
Chain(fetch_data, id).then(validate_data).then(normalize_data).then(send_data).run()
```
Instead, we can create a template chain and reuse it, passing different values to `.run()`:
```python
handle_data = Chain().then(validate_data).then(normalize_data).then(send_data)

for id in list_of_ids:
  handle_data.run(fetch_data, id)
```
Re-using a `Chain` object will significantly reduce its overhead, as most of the performance hit is due
to the creation of a new `Chain` instance. Nonetheless, the performance hit is negligible and not worth to sacrifice
readability for. So unless it makes sense (or you need to really squeeze out performance),
it's better to create a new `Chain` instance.

### Nesting A Chain
You can nest a `Chain` object within another `Chain` object:
```python
Chain(fetch_data, id)
.then(Chain().then(validate_data).then(normalize_data))
.then(send_data)
.run()
```
A nested chain must always be a template chain.

A nested chain will be evaluated with the current chain value of the parent chain passed to its `.run()` method.

### Pipe Syntax
Pipe syntax is supported:
```python
from quent import Chain, run

(Chain(fetch_data) | process_data | normalize_data | send_data).run()
Chain(fetch_data) | process_data | normalize_data | send_data | run()
```
You can also use [Pipe](https://github.com/JulienPalard/Pipe) with Quent:
```python
from pipe import where

Chain(get_items).then(where(lambda item: item.is_valid()))
Chain(get_items) | where(lambda item: item.is_valid())
```

## API
#### Value Evaluation
Most of the methods in the following section receives `value`, `args`, and `kwargs`. Unless explicitly told otherwise,
the evaluation of `value` in all of those methods is roughly equivalent to:
```python
if args[0] is Ellipsis:
  return value()

elif args or kwargs:
  return value(*args, **kwargs)

elif callable(value):
  return value(current_chain_value)

else:
  return value
```
The `evaluate_value` function contains the full evaluation logic.

### Core
#### `__init__(value: Any = None, *args, **kwargs)`
Creates a new chain with `value` as the chain's root item. `value` can be anything - a literal value,
a function, a class, etc.
If `args` or `kwargs` are provided, `value` is assumed to be a callable and will be evaluated with those
arguments. Otherwise, a check is performed to determine whether `value` is a callable. If it is, it is
called without any arguments.

Not passing a value will create a template chain (see: [Reusing A Chain](#reusing-a-chain)). You can still normally use
it, but then you must call `.run()` with a value (see the next section).

A few examples:
```python
Chain(42)
Chain(fn, True)
Chain(cls, name='foo')
Chain(lambda v: v*10, 4.2)
```

#### `run(value: Any = None, *args, **kwargs) -> Any | Coroutine`
Evaluates the chain and returns the result, or a coroutine if there are any coroutines in the chain.

If the chain is a template chain (initialized without a value), you must call `.run()` with a value, which will act
as the root item of the chain.

Conversely, if `.run()` is called with a value and the chain is a non-template chain, then an exception will be raised.
The only case where you can both create a template chain and run it without a value is for the `Cascade` class,
which is documented below in [Cascade - Void Mode](#cascade---void-mode).

Similarly to the examples above,
```python
Chain().run(42)
Chain().run(fn, True)
Chain().run(cls, name='foo')
Chain().run(lambda v: v*10, 2)
```

#### `then(value: Any, *args, **kwargs) -> Chain`
Adds `value` to the chain as a chain item. `value` can be anything - a literal value, a function, a class, etc.

Sets the evaluation of `value` as the current chain value.

This is the main and default way of adding items to the chain.

(see: [Ellipsis](#ellipsis) if you need to invoke `value` without arguments)

```python
Chain(fn).then(False)
Chain(42).then(verify_result)
Chain('<uuid>').then(uuid.UUID)
```

#### `root(value: Any = None, *args, **kwargs) -> Chain`
Like `.then()`, but it first sets the root value as the current chain value. Then it evaluates `value`
by the default [evaluation procedure](#value-evaluation).

Calling `.root()` without a value simply sets the root value as the current chain value.

Read more in [Flow Modifiers](#flow-modifiers).

```python
Chain(42).then(lambda v: v/10).root(lambda v: v == 42)
```

#### `ignore(value: Any, *args, **kwargs) -> Chain`
Like `.then()`, but keeps the current chain value unchanged.
Read more in [Flow Modifiers](#flow-modifiers).

```python
Chain(fetch_data, id).ignore(print).then(validate_data)
```

#### `root_ignore(value: Any, *args, **kwargs) -> Chain`
The combination of `.root()` and `.ignore()`.

```python
Chain(fetch_data, id).then(validate_data).root_ignore(print).then(normalize_data)
```

#### `attr(name: str) -> Chain`
Like `.then()`, but evaluates to `getattr(current_chain_item, name)`.

```python
class A:
  @property
  def a1(self):
    # I return something important
    pass

Chain(A()).attr('a1')
ChainAttr(A()).a1
```

#### `attr_fn(name: str, *args, **kwargs) -> Chain`
Like `.attr()`, but evaluates to `getattr(current_chain_item, name)(*args, **kwargs)`.

```python
class A:
  def a1(self, foo=None):
    # I do something important
    pass

Chain(A()).attr_fn('a1', foo=1)
ChainAttr(A()).a1(2)
```

#### `foreach(fn: Callable) -> Chain`
Iterates over the current chain value and invokes `fn(element)` for each element. Similarly to `.ignore()`,
this function does not change the current chain value.

```python
Chain(list_of_ids)
.foreach(Chain().then(fetch_data).then(validate_data).then(normalize_data).then(send_data))
.run()
```
will iterate over `list_of_ids`, invoke the nested chain with each different `id`, and then return `list_of_ids`.

#### `with_(self, context: Any | Ellipsis = ..., value: Any | Callable = None, *args, **kwargs) -> Chain`
Evaluates `value` by the default [evaluation procedure](#value-evaluation) inside a context. Returns the
result.

If `context` is an `Ellipsis`, Quent uses the current chain value as the context. Otherwise,
`context` is used as-is.

If `value` is not provided, the function returns the object that is returned from the `__enter__`
or `__aenter__` methods of the context. A check is performed to determine whether the context
should be used in a `with ...` or a `async with ...` statement.

**Note: if a context implements both `__enter__` and `__aenter__` methods, it will always be used in a regular `with ...`
statement. If this is the case, and you need to explicitly use `async with ...`, use `Chain.async_with()`.**

#### `async_with(self, context: Any | Ellipsis = ..., value: Any | Callable = None, *args, **kwargs) -> Chain`
Just like `.with()`, but explicitly for async contexts.

Do not use this method unless you have to. Opt to use `.with_()` instead, as it can handle both sync and async
contexts.

#### `Chain.from_(*args) -> Chain`
Creates a `Chain` template, and registers `args` as chain items.

```python
Chain.from_(validate_data, normalize_data, send_data).run(fetch_data, id)
# is the same as doing
Chain().then(validate_data).then(normalize_data).then(send_data).run(fetch_data, id)
```

### Callbacks
#### `except_(fn: Callable | str, *args, **kwargs) -> Chain`
Register a callback that will be called if an exception is raised anytime during the chain's
evaluation. The callback is evaluated with the root value, or with `args` and `kwargs`.

If `fn` is a string, then it is assumed to be an attribute method of the root value.

```python
Chain(fetch_data).then(validate_data).except_(discard_data)
```

#### `finally_(fn: Callable | str, *args, **kwargs) -> Chain`

Register a callback that will **always** be called after the chain's evaluation. The callback is evaluated with
the root value, or with `args` and `kwargs`.

If `fn` is a string, then it is assumed to be an attribute method of the root value.

```python
Chain(get_id).then(aqcuire_lock).root(fetch_data).finally_(release_lock)
```

### Conditionals
#### `if_(fn: Callable | Ellipsis = ..., on_true: Any | Callable = None, *args, **kwargs) -> Chain`
Registers a function `fn` which will be called with the current chain value. If `on_true` is provided and
the result of `fn` is truthy, evaluates `on_true` and sets the result as the current chain value.
If `on_true` is not provided, sets the result of `fn` as the current chain value.

If `fn` is an `Ellipsis`, evaluates the truthiness of the current chain value (`bool(current_chain_value)`).

`on_true` may be anything and follows the default [evaluation procedure](#value-evaluation) as described above.

```python
Chain(get_random_number).if_(lambda num: num > 5, you_win, prize=1)
```

#### `else_(on_false: Any | Callable, *args, **kwargs) -> Chain`
If a previous conditional result is falsy, evaluates `on_false` and sets the result as the current chain value.

`on_false` may be anything and follows the default [evaluation procedure](#value-evaluation) as described above.

**Can only be called immediately following a conditional.**

```python
Chain(get_random_number).if_(lambda num: num > 5, you_win, prize=1).else_(you_lose, cost=10)
```

#### `not_() -> Chain`
- `not current_chain_value`

This method currently does not support the `on_true` argument since it looks confusing.
I might add it in the future.

```python
Chain(is_valid, 'something').not_()
```

#### `eq(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_chain_value == value`

```python
Chain(420).then(lambda v: v/10).eq(42)
Chain(420).then(lambda v: v/10).eq(40).else_(on_fail)
```

#### `neq(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_chain_value != value`

```python
Chain(420).then(lambda v: v/10).neq(40)
```

#### `is_(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_chain_value is value`

```python
Chain(object()).is_(1)
```

#### `is_not(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_chain_value is not value`

```python
Chain(object()).is_not(object())
```

#### `in_(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_chain_value in value`

```python
Chain('sub').in_('subway')
```

#### `not_in(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_chain_value not in value`

```python
Chain('bus').then(lambda s: s[::-1]).not_in('subway')
```

### Cascade
Although considered unpythonic, in some cases the [cascade design](https://en.wikipedia.org/wiki/Fluent_interface)
can be very helpful. The `Cascade` class is identical to `Chain`, except that during the chain's evaluation,
each chain item is evaluated using the root value as an argument
(or in other words, the current chain value is always the chain's root value).
The return value of `Cascade.run()` is always its root value.
```python
from quent import Cascade

fetched_data = (
  Cascade(fetch_data, id)
  .then(send_data_to_backup)
  .then(lambda data: send_data(data, to_id=1))
  .then(print)
  .run()
)
```
will execute `fetch_data(id)`, then `send_data_to_backup(data)`, then `send_data(data, to_id=1)`,
and then `print(data)`.

You can also use `Cascade` to make existing classes behave the same way:
```python
from quent import CascadeAttr


class Foo:
  def foo(self):
    ...
  async def bar(self):
    ...
  def baz(self):
    ...

async def get_foo():
  f = Foo()
  f.foo()
  await f.bar()
  f.baz()
  return f

def better_get_foo():
  return CascadeAttr(Foo()).foo().bar().baz().run()
```

`Cascade` works for any kind of object:
```python
from quent import CascadeAttr

CascadeAttr([]).append(1).append(2).append(3).run() == [1, 2, 3]
```

#### Cascade - Void Mode
In some cases it may be desired to run a bunch of independent operations. Using `Cascade`, one can
achieve this by simply not passing a root value to the constructor nor to `.run()`. All the chain items
will not receive any arguments (excluding explicitly provided `args` / `kwargs`).
```python
await (
  Cascade()
  .then(foo, False)
  .then(bar)
  .then(baz)
  .run()
)
```
will execute `foo(False)`, then `bar()`, then `baz()`.

A void `Cascade` will always return `None`.

### Direct Attribute Access
Both `Chain` and `Cascade` can support "direct" attribute access via the `ChainAttr` and `CascadeAttr` classes.
See the [Cascade](#cascade) section above to see an example of `CascadeAttr` usage. The same principle holds for
`ChainAttr`. Accessing attributes without using the `Attr` subclass is possible using `.attr()` and `.attr_fn()`.

The reason I decided to separate this functionality from the main classes is due to the fact
that it requires overriding `__getattr__`, which drastically increases the overhead of both creating an instance and
accessing any properties / methods. And since I don't think this kind of usage will be common, I decided
to keep this functionality opt-in.

## Limitations
### Asynchronous `except` and `finally` callbacks
If an except/finally callback is a coroutine function, and an exception is raised *before*
the first coroutine of the chain has been evaluated, or if there aren't any coroutines in the chain - the
callbacks will **not** be awaited.

This limitation is due to the fact that we cannot (nor want to) return the result of the callbacks so that
they will be awaited downstream. So in order to `await` the callbacks, the execution must be inside a coroutine, so that
we can `await` the callbacks.
And the only case where we evaluate the chain inside a coroutine is when we detect a coroutine during the chain's
evaluation.

This shouldn't be an issue in most use cases, but it is important to be aware of this limitation.

As an example, suppose that `fetch_data` is synchronous, and `report_usage` is asynchronous.
```python
Chain(fetch_data).then(raise_exception, ...).finally_(report_usage).run()
```
will execute `fetch_data()`, then `raise_exception()`, and then `report_usage(data)`. But `report_usage(data)` is a
coroutine, and `fetch_data` and `raise_exceptions` are not. This will cause `report_usage(data)` to be invoked **but
not awaited**.
