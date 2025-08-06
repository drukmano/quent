# Chains & Cascades

Quent provides several chain types, each suited to different composition patterns. All chain types share the same method API and support transparent async handling.

## Chain

The `Chain` class is the core building block. Each operation receives the result of the previous one, forming a sequential pipeline.

```python
from quent import Chain

result = (
  Chain(fetch_user, user_id)
  .then(validate_permissions)
  .then(apply_transformations)
  .then(save_to_database)
  .run()
)
```

### Constructor

```python
Chain(v=None, *args, **kwargs)
```

The first argument `v` sets the root value of the chain. If `v` is callable, it is called with `*args` and `**kwargs` during execution, and its return value becomes the root value. If `v` is not callable, it is used directly as the root value.

```python
# Callable root -- fetch_user(42) is called at execution time
Chain(fetch_user, 42)

# Literal root -- the dict is the root value
Chain({"id": 42, "name": "Alice"})

# No root -- must provide a value via .run()
Chain().then(validate).then(transform).run(initial_data)
```

### Core Operations

| Method | Receives | Result used? | Description |
|--------|----------|-------------|-------------|
| `.then(v)` | current value | Yes | Add operation, pass result forward |
| `.do(fn)` | current value | No | Side effect, discard result |
| `.root()` | -- | Yes | Reset current value to root value |
| `.root(fn)` | root value | Yes | Transform root value, use as current |
| `.root_do(fn)` | root value | No | Side effect on root value, discard result |

```python
result = (
  Chain(fetch_data, id)
  .do(log_operation)                    # log but discard return value
  .then(transform)                      # receives fetch_data result
  .root(lambda original: log(original)) # receives the original fetch_data result
  .then(save)
  .run()
)
```

### Running a Chain

Use `.run()` to execute the chain. You can pass arguments to `.run()` which override the root value.

```python
# Root value set in constructor
Chain(fetch_data, id).then(validate).run()

# Root value set at run time
Chain().then(validate).then(transform).run(initial_data)

# __call__ is an alias for .run()
chain = Chain().then(validate).then(transform)
result = chain(initial_data)
```

## Cascade

The `Cascade` class implements the fluent interface pattern. Every operation receives the **root value**, and the final result is always the root value.

```python
from quent import Cascade

# All operations receive the original data
processed_data = (
  Cascade(fetch_data, id)
  .then(send_to_backup)     # backup receives data
  .then(send_to_analytics)  # analytics receives data
  .then(log_operation)      # logger receives data
  .run()  # returns the original data
)
```

Cascade is useful when you need to perform multiple operations on the same value -- fan-out patterns, builder-style initialization, or broadcasting to multiple consumers.

## ChainAttr

`ChainAttr` extends `Chain` with dynamic attribute access via `__getattr__`. This allows natural dot-notation to access attributes and call methods on the current chain value.

```python
from quent import ChainAttr

result = (
  ChainAttr("hello world")
  .upper()        # calls str.upper()
  .split()        # calls str.split()
  .run()
)
# Returns: ['HELLO', 'WORLD']
```

When you access an attribute on a `ChainAttr` instance, it appends an attribute access operation to the chain. If the accessed attribute is then called (i.e., you use parentheses), it becomes a method call on the current value.

```python
from quent import ChainAttr

# Access a nested attribute
result = ChainAttr(some_object).config.timeout.run()

# Call a method with arguments
result = ChainAttr(some_object).get_profile(detailed=True).run()
```

## CascadeAttr

`CascadeAttr` combines `Cascade` semantics with `ChainAttr`'s dynamic attribute access. Every operation receives the root value, and dynamic attribute access is available.

```python
from quent import CascadeAttr

result = CascadeAttr(list()).append(1).append(2).extend([3, 4]).run()
# Returns: [1, 2, 3, 4]
```

This is particularly useful for making any class fluent. Since `list.append()` returns `None`, a regular chain would lose the list after the first `.append()`. `CascadeAttr` always passes the root value (the list), so chained method calls work naturally.

## Attribute Access Methods

For `Chain` and `Cascade` (without the `Attr` variants), you can access attributes and call methods using explicit methods:

```python
from quent import Chain

# Get an attribute
Chain(fetch_user).attr("name").then(print).run()

# Call a method
Chain(fetch_user).attr_fn("get_profile", detailed=True).then(process).run()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `.attr(name)` | `.attr(name) -> Self` | Get attribute `name` from current value |
| `.attr_fn(name, *args, **kwargs)` | `.attr_fn(name, *args, **kwargs) -> Self` | Call method `name` on current value |

## Pipe Operator

The `|` operator provides an alternative syntax for building chains. Use the `run` class to terminate and execute.

```python
from quent import Chain, run

# These are equivalent
result = Chain(fetch_data, id).then(validate).then(transform).then(save).run()
result = Chain(fetch_data, id) | validate | transform | save | run()

# Pass arguments to run
result = Chain().then(validate).then(transform) | run(fetch_data, id)
```

The pipe operator appends each operand as a `.then()` operation. When a `run` instance is piped, the chain executes.

## Flow Control

### Conditionals

Quent provides conditional operations that execute based on the truthiness of the current value.

```python
from quent import Chain

# if_ / else_
result = (
  Chain(get_user)
  .then(lambda u: u.age)
  .then(lambda a: a >= 18)
  .if_(grant_access)
  .else_(deny_access)
  .run()
)

# Raise on condition
result = (
  Chain(get_config, key)
  .else_raise(ValueError("Config key not found"))
  .then(parse_config)
  .run()
)
```

| Method | Condition |
|--------|-----------|
| `.if_(v)` | Execute `v` if current value is truthy |
| `.else_(v)` | Execute `v` if current value is falsy |
| `.if_not(v)` | Execute `v` if current value is falsy |
| `.condition(fn)` | Execute next link only if `fn(current_value)` is truthy |
| `.not_()` | Negate current value |
| `.eq(value)` | `current_value == value` |
| `.neq(value)` | `current_value != value` |
| `.is_(value)` | `current_value is value` |
| `.is_not(value)` | `current_value is not value` |
| `.in_(value)` | `current_value in value` |
| `.not_in(value)` | `current_value not in value` |
| `.isinstance_(*types)` | `isinstance(current_value, types)` |
| `.or_(value)` | `current_value or value` |
| `.if_raise(exc)` | Raise `exc` if current value is truthy |
| `.else_raise(exc)` | Raise `exc` if current value is falsy |
| `.if_not_raise(exc)` | Raise `exc` if current value is falsy |

### Loops

```python
from quent import Chain

# Iterate over items and process each one; result is passed forward
Chain(get_items).foreach(process_item).then(summarize).run()

# Iterate as a side effect (result discarded)
Chain(get_items).foreach_do(log_item).then(continue_processing).run()

# Produce an iterator from chain results
for item in Chain(get_items).iterate(transform):
  print(item)

# Async iteration
async for item in Chain(get_items_async).iterate(transform):
  print(item)

# Loop while a function returns truthy
Chain(initial_state).while_true(step_function).run()
```

### Flow Statements

`Chain.return_()` and `Chain.break_()` are class methods that control chain execution flow:

```python
from quent import Chain

# Return early from a chain
Chain(get_data).then(lambda v: Chain.return_(v) if v else None).then(transform).run()

# Break out of a foreach or while_true loop
Chain(get_items).foreach(lambda item: Chain.break_() if item is None else process(item)).run()
```

### Context Managers

```python
from quent import Chain

# Execute fn within a context manager; result is passed forward
Chain(acquire_lock).with_(perform_operation).run()

# Execute as a side effect within a context manager
Chain(open_file, path).with_do(write_header).then(continue_processing).run()
```

### Sleep and Raise

```python
from quent import Chain

# Insert a delay into the chain
Chain(start_job).then(submit).sleep(2.0).then(check_status).run()

# Unconditionally raise an exception
Chain(get_value).then(validate).raise_(RuntimeError("unexpected state")).run()
```

## Special Values

### Ellipsis (`...`)

When passed as the first argument to a chain operation, the ellipsis signals that the function should be called with **zero arguments**, overriding the default behavior of passing the current value:

```python
from quent import Chain

# cleanup() is called with no arguments
Chain(data).then(process).finally_(cleanup, ...)

# compare with: cleanup(current_value)
Chain(data).then(process).finally_(cleanup)
```

### Null

The `Null` sentinel represents "no value" and is distinct from `None`. It is useful when `None` is a valid value in your domain:

```python
from quent import Chain, Null

# Check if a chain produced no value
result = chain.run()
if result is Null:
  print("No value was produced")

# Access via class method
sentinel = Chain.null()
```

## Further Reading

- [Async Handling](async.md) -- How transparent async works
- [Resilience](resilience.md) -- Retry, timeout, and safe_run
- [Error Handling](error-handling.md) -- Exception handling and stack traces
- [Reuse & Patterns](reuse.md) -- clone, freeze, and decorator
- [API Reference](../reference.md) -- Full method signatures
