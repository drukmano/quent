# Reuse & Patterns

Quent provides several mechanisms for reusing chains across your codebase: cloning, freezing into immutable callables, and using chains as decorators.

## clone

Create a deep copy of a chain for independent reuse. The cloned chain is a separate object -- modifying one does not affect the other.

```python
from quent import Chain

base = Chain().then(validate).then(normalize)

chain_a = base.clone().then(save_to_db)
chain_b = base.clone().then(send_to_api)
```

This pattern is useful when you have a shared pipeline prefix and want to fork it into different variants.

### Parameters

```python
.clone() -> Self
```

Returns a new chain with the same operations and configuration as the original.

## freeze and FrozenChain

The `.freeze()` method converts a chain into an immutable, callable `FrozenChain`. A frozen chain cannot be modified (no `.then()`, `.except_()`, etc.) but can be called repeatedly.

```python
from quent import Chain

processor = Chain().then(validate).then(normalize).then(save).freeze()

# Use as a callable -- each call creates a fresh execution
for item in items:
  processor(item)
```

### FrozenChain API

```python
frozen.run(v=None, *args, **kwargs)   # execute the frozen chain
frozen(v=None, *args, **kwargs)        # alias for .run()
frozen.decorator()                     # use as a function decorator
```

A `FrozenChain` is inherently thread-safe because each call operates on a fresh execution state.

### When to Freeze

Freeze a chain when:

- You want to reuse it as a callable without risk of accidental modification
- You need thread-safe reuse without the overhead of `.safe_run()` cloning every time
- You want to store a chain in a module-level variable or pass it around as a function

## decorator

Use a chain as a function decorator. The decorated function becomes the root value of the chain.

```python
from quent import Chain

@Chain().then(validate).then(normalize).then(save).decorator()
def process(data):
  return data
```

When `process(data)` is called, the chain executes with `data` (the return value of the decorated function) as the root value, then passes it through `validate`, `normalize`, and `save`.

The `.decorator()` method is available on both `Chain` and `FrozenChain`:

```python
# From a chain
@Chain().then(validate).then(save).decorator()
def process(data):
  return data

# From a frozen chain
pipeline = Chain().then(validate).then(save).freeze()

@pipeline.decorator()
def process(data):
  return data
```

## Common Patterns

### Middleware Chain

Build a middleware stack where each layer wraps the next:

```python
from quent import Chain

auth_chain = Chain().then(authenticate).then(authorize)
logging_chain = Chain().do(log_request)
validation_chain = Chain().then(validate_input)

# Compose a full middleware pipeline
pipeline = (
  Chain()
  .then(auth_chain.run)
  .then(logging_chain.run)
  .then(validation_chain.run)
  .then(handle_request)
  .freeze()
)

# Use the composed pipeline
result = pipeline(request)
```

### Validation Pipeline

Create a reusable validation chain:

```python
from quent import Chain

email_validator = (
  Chain()
  .then(lambda s: s.strip())
  .then(lambda s: s.lower())
  .then(lambda s: s if "@" in s else None)
  .else_raise(ValueError("Invalid email"))
  .freeze()
)

# Reuse across your codebase
email = email_validator("Alice@Example.COM")  # "alice@example.com"
email = email_validator("not-an-email")        # raises ValueError
```

### Data Transformation Pipeline

Build a reusable data transformation pipeline that can be applied to collections:

```python
from quent import Chain

normalize = (
  Chain()
  .then(lambda d: {k.lower(): v for k, v in d.items()})
  .then(lambda d: {k: v.strip() if isinstance(v, str) else v for k, v in d.items()})
  .then(lambda d: {k: v for k, v in d.items() if v is not None})
  .freeze()
)

# Apply to a single record
clean = normalize({"Name": "Alice ", "Email": "alice@example.com", "Phone": None})
# {"name": "Alice", "email": "alice@example.com"}

# Apply to a collection
Chain(get_records).foreach(normalize).then(save_batch).run()
```

### Template Chain with safe_run

When you need a reusable chain that is safe for concurrent use:

```python
from quent import Chain

template = Chain().then(validate).then(transform).then(save)

# Thread-safe: each call clones before execution
import threading

for data in work_items:
  threading.Thread(target=template.safe_run, args=(data,)).start()
```

### compose

Combine multiple chains into a single chain using `Chain.compose()`:

```python
from quent import Chain

step1 = Chain().then(fetch).then(validate)
step2 = Chain().then(transform).then(enrich)
step3 = Chain().then(save).then(notify)

# Compose into a single pipeline
full_pipeline = Chain.compose(step1, step2, step3)
result = full_pipeline.run(initial_data)
```

## Further Reading

- [Chains & Cascades](chains.md) -- Core chain operations
- [Resilience](resilience.md) -- safe_run for thread-safe execution
- [API Reference](../reference.md) -- Full method signatures for clone, freeze, decorator
