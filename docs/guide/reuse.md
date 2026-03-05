# Reuse & Patterns

Quent provides mechanisms for reusing chains across your codebase: freezing into immutable callables, and using chains as decorators.

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
frozen.run(v=Null, *args, **kwargs)   # execute the frozen chain
frozen(v=Null, *args, **kwargs)        # alias for .run()
```

A `FrozenChain` is inherently thread-safe because each call operates on a fresh execution state.

### When to Freeze

Freeze a chain when:

- You want to reuse it as a callable without risk of accidental modification
- You need thread-safe reuse across concurrent calls
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

```python
# From a chain
@Chain().then(validate).then(save).decorator()
def process(data):
  return data

# From a frozen chain
pipeline = Chain().then(validate).then(save).freeze()

@pipeline.run
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
  .then(lambda s: s if '@' in s else (_ for _ in ()).throw(ValueError('Invalid email')))
  .freeze()
)

# Reuse across your codebase
email = email_validator('Alice@Example.COM')  # 'alice@example.com'
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
clean = normalize({'Name': 'Alice ', 'Email': 'alice@example.com', 'Phone': None})
# {'name': 'Alice', 'email': 'alice@example.com'}

# Apply to a collection
Chain(get_records).foreach(normalize).then(save_batch).run()
```

## Further Reading

- [Chains](chains.md) -- Core chain operations
- [API Reference](../reference.md) -- Full method signatures for freeze and decorator
