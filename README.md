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
  - [Safety Callbacks](#safety-callbacks)
  - [Comparisons](#comparisons)
  - [Iterators](#iterators)
  - [Contexts](#contexts)
- [API](#api)
  - [Core](#core)
  - [`except`, `finally`](#callbacks)
  - [Conditionals](#conditionals)
- [Cascade](#cascade)
- [Direct Attribute Access](#direct-attribute-access)
- [Important Notes](#important-notes)

**Suggestions and contributions are more than welcome.**

## Introduction

Quent is an [enhanced](#details--examples), [chain interface](https://en.wikipedia.org/wiki/Method_chaining) implementation for
Python, designed to handle coroutines transparently. The interface and usage of Quent remains exactly the same,
whether you feed it synchronous or asynchronous objects - it can handle almost any use case.

*Every documented API supports both regular functions and coroutines. It will work the exact same way as with a regular
function. Quent automatically awaits any coroutines, even a coroutine that the function passed to `.foreach()` may
return.*

Quent is written in C (using Cython) to minimize it's overhead as much as possible.

As a basic example, take this function:
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

def handle_request(id):
  return Chain(fetch_data, id).then(validate_data).then(normalize_data).then(send_data).run()
```

**Upon evaluation (calling `.run()`), if an awaitable object is detected, Quent wraps it in a Task and returns it.
The task is automatically scheduled for execution and the chain evaluation continues within the task.
As Task objects need not be `await`-ed in order to run, you may or may not `await` it, depending on your needs.**

Besides `Chain`, Quent provides the [Cascade](#cascade) class which implements the [fluent interface](https://en.wikipedia.org/wiki/Fluent_interface).

Quent aims to provide all the necessary tools to handle every use case.
See the full capabilities of Quent in the [API Section](#api).

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
will return a Task or not. I see it no differently than any undocumented code - with a proper
and clear documentation (be it an external documentation or just a simple docstring), there shouldn't be
any truly *unexpected* behavior (barring any unknown bugs).

## Details & Examples
### Literal Values
You don't have to pass a callable as a chain item - literal values works just as well.
```python
Chain(fetch_data, id).then(True).run()
```
will execute `fetch_data(id)`, and then return `True`.

### Custom Arguments
You may provide `args` or `kwargs` to a chain item - by doing so, Quent assumes that the item is a callable
and will evaluate it with the provided arguments, instead of evaluating it with the current value.
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

- `.root()` evaluates the item using the root value, instead of the current value.
- `.ignore()` evaluates the item with the current value but will not propagate its result forwards.

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
A nested chain must always be a template chain (i.e. initialized without arguments).

A nested chain will be evaluated with the current value of the parent chain passed to its `.run()` method.

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

### Safety Callbacks
The usage of `except` and `finally` is supported:
```python
Chain(open('<file-path>')).then(do_something_with_file).finally_(lambda f: f.close())
```
Read more about it in [Callbacks](#callbacks).

### Comparisons
Most basic comparison operations are supported:
```python
Chain(get_key).in_(list_of_keys)  # == get_key() in list_of_keys
```
See the full list of operations in [Conditionals](#conditionals).

### Iterators
You can easily iterate over the result of something:
```python
Chain(fetch_keys).foreach(do_something_with_key)
```
The full details of `.foreach()` are explained
[here](#foreach).

### Contexts
You can execute a function (or do quite anything you want) inside a context:
```python
Chain(get_lock, id).with_(fetch_data, id)
```
The full details of `.with_()` are explained
[here](#with).

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
  return value(current_value)

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

#### `run(value: Any = None, *args, **kwargs) -> Any | asyncio.Task`
Evaluates the chain and returns the result, or a Task if there are any coroutines in the chain.

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

Returns the [evaluation](#value-evaluation) of `value`.

This is the main and default way of adding items to the chain.

(see: [Ellipsis](#ellipsis) if you need to invoke `value` without arguments)

```python
Chain(fn).then(False)
Chain(42).then(verify_result)
Chain('<uuid>').then(uuid.UUID)
```

#### `root(value: Any = None, *args, **kwargs) -> Chain`
Like `.then()`, but it first sets the root value as the current value, and then it evaluates `value`
by the default [evaluation procedure](#value-evaluation).

Calling `.root()` without a value simply returns the root value.

Read more in [Flow Modifiers](#flow-modifiers).

```python
Chain(42).then(lambda v: v/10).root(lambda v: v == 42)
```

#### `ignore(value: Any, *args, **kwargs) -> Chain`
Like `.then()`, but keeps the current value unchanged.

In other words, this function does not affect the flow of the chain.

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
Like `.then()`, but evaluates to `getattr(current_value, name)`.

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
Like `.attr()`, but evaluates to `getattr(current_value, name)(*args, **kwargs)`.

```python
class A:
  def a1(self, foo=None):
    # I do something important
    pass

Chain(A()).attr_fn('a1', foo=1)
ChainAttr(A()).a1(2)
```

#### Foreach
#### `foreach(fn: Callable) -> Chain`
Iterates over the current value and invokes `fn(element)` for each element. Returns a list
that is the result of `fn(element)` for each `element`.

If the iterator implements `__aiter__`, `async for ...` will be used.

Example:
```python
Chain(list_of_ids).foreach(validate_id).run()
```
will iterate over `list_of_ids`, and return a list that is equivalent to `[validate_id(id) for id in list_of_ids]`.

#### `foreach_do(fn: Callable) -> Chain`
Like `.foreach()`, but returns nothing. In other words, this is the combination of
`.foreach()` and `.ignore()`.

Example:
```python
Chain(list_of_ids)
.foreach_do(Chain().then(fetch_data).then(validate_data).then(normalize_data).then(send_data))
.run()
```
will iterate over `list_of_ids`, invoke the nested chain with each different `id`, and then return `list_of_ids`.

#### With
#### `with_(self, value: Any | Callable = None, *args, **kwargs) -> Chain`
Executes `with current_value as ctx` and evaluates `value` inside the context block,
**with `ctx` as the current value**, and returns the result. If `value` is not provided, returns `ctx`.
This method follows the [default evaluation](#value-evaluation) procedure, so passing `args` or `kwargs`
is perfectly valid.

Depending on `value` (and `args`/`kwargs`), this is roughly equivalent to
```python
with current_value as ctx:
  return value(ctx)
```
If the context object implements `__aenter__`, `async with ...` will be used.

Example:
```python
Chain(get_lock, id).with_(fetch_data, id).run()
```
is roughly equivalent to:
```python
with get_lock(id) as lock:
  # `lock` is not used here since we passed a custom argument `id`.
  return fetch_data(id)
```

#### `with_do(self, value: Any | Callable, *args, **kwargs) -> Chain`
Like `.with_()`, but returns nothing. In other words, this is the combination of
`.with_()` and `.ignore()`.

#### Class Methods
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
evaluation. The callback is evaluated with the root value, or with `args` and `kwargs` if provided.

If `fn` is a string, then it is assumed to be an attribute method of the root value.

```python
Chain(fetch_data).then(validate_data).except_(discard_data)
```

#### `finally_(fn: Callable | str, *args, **kwargs) -> Chain`

Register a callback that will **always** be called after the chain's evaluation. The callback is evaluated with
the root value, or with `args` and `kwargs` if provided.

If `fn` is a string, then it is assumed to be an attribute method of the root value.

```python
Chain(get_id).then(aqcuire_lock).root(fetch_data).finally_(release_lock)
```

### Conditionals
#### `if_(on_true: Any | Callable, *args, **kwargs) -> Chain`
Evaluates the truthiness of the current value (`bool(current_value)`).
If `on_true` is provided and the result is `True`, evaluates `on_true` and returns the result.
If `on_true` is not provided, simply returns the truthiness result (`bool`).

`on_true` may be anything and follows the default [evaluation procedure](#value-evaluation) as described above.

```python
Chain(get_random_number).then(lambda n: n > 5).if_(you_win, prize=1)
```

#### `else_(on_false: Any | Callable, *args, **kwargs) -> Chain`
If a previous conditional result is falsy, evaluates `on_false` and returns the result.

`on_false` may be anything and follows the default [evaluation procedure](#value-evaluation) as described above.

**Can only be called immediately following a conditional.**

```python
Chain(get_random_number).then(lambda n: n > 5).if_(you_win, prize=1).else_(you_lose, cost=10)
```

#### `not_() -> Chain`
- `not current_value`

This method currently does not support the `on_true` argument since it looks confusing.
I might add it in the future.

```python
Chain(is_valid, 'something').not_()
```

#### `eq(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_value == value`

```python
Chain(420).then(lambda v: v/10).eq(42)
Chain(420).then(lambda v: v/10).eq(40).else_(on_fail)
```

#### `neq(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_value != value`

```python
Chain(420).then(lambda v: v/10).neq(40)
```

#### `is_(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_value is value`

```python
Chain(object()).is_(1)
```

#### `is_not(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_value is not value`

```python
Chain(object()).is_not(object())
```

#### `in_(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_value in value`

```python
Chain('sub').in_('subway')
```

#### `not_in(value: Any, on_true: Any | Callable = None, *args, **kwargs) -> Chain`
- `current_value not in value`

```python
Chain('bus').then(lambda s: s[::-1]).not_in('subway')
```

### Cascade
Although considered unpythonic, in some cases the [cascade design](https://en.wikipedia.org/wiki/Fluent_interface)
can be very helpful. The `Cascade` class is identical to `Chain`, except that during the chain's evaluation,
each chain item is evaluated using the root value as an argument
(or in other words, the current value is always the chain's root value).
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

## Important Notes
### Asynchronous `except` and `finally` callbacks
If an except/finally callback is a coroutine function, and an exception is raised *before*
the first coroutine of the chain has been evaluated, or if there aren't any coroutines in the chain - each
callback will be invoked inside a new Task, which you won't have access to.
This is due to the fact that we cannot return the new Task(s) from the `except`/`finally` clauses.

This shouldn't be an issue in most use cases, but important to be aware of.
A `RuntimeWarning` will be emitted in such a case.

As an example, suppose that `fetch_data` is synchronous, and `report_usage` is asynchronous.
```python
Chain(fetch_data).then(raise_exception, ...).finally_(report_usage).run()
```
will execute `fetch_data()`, then `raise_exception()`, and then `report_usage(data)`. But `report_usage(data)` is a
coroutine, and `fetch_data` and `raise_exceptions` are not. Then Quent will wrap `report_usage(data)` in a Task
and "forget" about it.

If you must, you can "force" an async chain by giving it a dummy coroutine:
```python
async def fn(v):
  return v

await Chain(fetch_data).then(fn).then(raise_exception, ...).finally_(report_usage).run()
```
This will ensure that `report_usage()` will be awaited properly.
