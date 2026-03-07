# Chains

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

```python
result = (
  Chain(fetch_data, id)
  .do(log_operation)                    # log but discard return value
  .then(transform)                      # receives fetch_data result
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

## Flow Control

### Loops

```python
from quent import Chain

# Iterate over items and process each one; result is passed forward
Chain(get_items).map(process_item).then(summarize).run()

# Iterate as a side effect (result discarded)
Chain(get_items).foreach(log_item).then(continue_processing).run()

# Produce an iterator from chain results
for item in Chain(get_items).iterate(transform):
  print(item)

# Async iteration
async for item in Chain(get_items_async).iterate(transform):
  print(item)
```

### Flow Statements

`Chain.return_()` and `Chain.break_()` are class methods that control chain execution flow:

```python
from quent import Chain

# Return early from a chain
Chain(get_data).then(lambda v: Chain.return_(v) if v else None).then(transform).run()

# Break out of a map loop
Chain(get_items).map(lambda item: Chain.break_() if item is None else process(item)).run()
```

### Context Managers

```python
from quent import Chain

# Execute fn within a context manager; result is passed forward
Chain(acquire_lock).with_(perform_operation).run()

# Execute as a side effect within a context manager
Chain(open_file, path).with_do(write_header).then(continue_processing).run()
```

### Filter and Gather

```python
# Filter elements from an iterable
Chain(get_items).filter(lambda x: x > 0).then(process).run()

# Run multiple functions concurrently on the current value
Chain(fetch_data).gather(analyze, validate, summarize).then(combine_results).run()
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
```

## Further Reading

- [Async Handling](async.md) -- How transparent async works
- [Error Handling](error-handling.md) -- Exception handling and stack traces
- [Reuse & Patterns](reuse.md) -- freeze and decorator
- [API Reference](../reference.md) -- Full method signatures
