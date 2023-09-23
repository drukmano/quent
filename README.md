# Quent — Yet Another Chain Interface

Transparent `async` syntax - The unification of synchronous and asynchronous syntax.

TODO table of contents

## Introduction
Quent implements the chaining design to allow 
TODO have another, different, example
Quent allows you to unify code that uses synchronous and asynchronous objects.
As a simple example, consider this line of code:
```python
await send_data(await normalize_data(process_data(await fetch_data())))
```
Many would say that this code should be written like this:
```python
data = await fetch_data()
data = process_data(data)
data = await normalize_data(processed_data)
await send_data(data)
```
And they would not be wrong. But a lot of the time there isn't a need for these intermediate values,
except to make the code more readable compared to the one-liner above.

With Quent, one could achieve the same result using a clearer, chain syntax:
```python
await Chain(fetch_data).then(process_data).then(normalize_data).then(send_data).run()

await (
  Chain(fetch_data)
  .then(process_data)
  .then(normalize_data)
  .then(send_data)
).run()
```
Yes, you only need to await once. Quent will await the necessary functions.

Pipe syntax is also supported
```python
Chain(fetch_data) | process_data | normalize_data | send_data | run()
```
Quent is written in C (using Cython) to maximize execution speed and minimize memory footprint as much as possible, to make it truly "transparent" --- most of the time, there is little to none performance drop when using it.

## Getting Started
To install Quent,
```
pip install quent
```
Simple usage:
```python
from quent import Chain

def do_something(foo: Any) -> bool | Coroutine[bool]:
  if foo is Foo:
    return Chain(some_io_operation).then(True).run()
  elif foor is Bar:
    return Chain(another_io_operation).then(True).run()
  else:
    return Chain(close_connection).then(False).run()
```

## Usage
TODO detail everything, give use cases, showcase Recipes, explain naming convention (r suffix), etc.
explain that for maximum performance, avoid the 'r' suffix classes, etc.
explain issue with async and except/finally
```python
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
  return CascadeR(Foo()).foo().bar().baz().run()
```
You can use *Pipe* with Quent:
```python
from pipe import map
Chain(get_items).then(map(lambda item: item.is_valid()))
Chain(get_items) | map(lambda item: item.is_valid())
```

## Motivation
The motivation to create Quent came while I was working on a library with various utilities, wrappers
for popular libraries, and SDKs for some web services. I wanted developers to be able to use it regardless if they are in a synchronous or an asynchronous environment. To do that, I had to create an additional `async` version for each module and override the necessary functions, which left me with a lot of duplicate code - just for the ability to `await` some (IO) operations.
For example, SQLAlchemy has an asynchronous version for their engine and session objects. It felt wrong to do something like this, over and over again for multiple functions:
```python
class Query:
  ...
  def execute(self):
    return self.session.scalars(self.query)

  def all(self):
    return self.execute().all()
  ...

class QueryAsync(Query):
  ...
  async def all(self):
    return (await self.execute()).all()
  ...
```
This also forces developers to differentiate between the sync and async versions of each module.
With Quent, I can have a single class and all I have to do is:
```python
class Query:
  def all(self):
    return ChainR(self.execute).all().run()
```
You can also re-use the Chain to significantly improve performance,
```python
chain_all: Chain = Chain().then(lambda r: r.all())
def all(self):
  return chain_all.run(self.execute)
```
With this pattern, there is no performance drop. It's like you're not even using Quent.

And, for my Redis Pipeline wrapper:
```python
def flush(self):
  pipe = self.r.pipeline(transaction=self.transaction)
  self.apply_pipe_chain(pipe)
  return (
    Chain(pipe.execute, raise_on_error=True)
    .then(self.remove_ignored_commands)
    .finally_(pipe.reset, ...)
    .run()
  )
```
Or, if you really like one-liners, Quent can do almost anything:
```python
def flush(self):
  return (
    ChainR(self.r.pipeline(transaction=self.transaction))
    .then(self.apply_pipe_chain)
    .root().execute(raise_on_error=True)
    .then(self.remove_ignored_commands)
    .finally_('reset')
    .run()
  )
```
