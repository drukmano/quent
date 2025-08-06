# Quent

A high-performance chain interface library for Python with transparent async/await handling.

[![PyPI version](https://badge.fury.io/py/quent.svg)](https://badge.fury.io/py/quent)

## Key Features

**🔗 Fluent Chain Interface** - Write elegant, readable code without intermediate variables or nested function calls

**⚡ Transparent Async Handling** - Use the exact same API for both synchronous and asynchronous code. Quent automatically detects and handles coroutines, futures, and async contexts

**🚀 High Performance** - Written in Cython for minimal overhead, with optimized C extensions

**🔄 Universal Compatibility** - Works seamlessly with any Python codebase, whether sync, async, or mixed

**🔍 Exceptional Stack Traces** - Clean, informative exception traces that show the exact chain state and operation where errors occurred

## Installation

```bash
pip install quent
```

## Why Quent?

### The Problem

Traditional Python code often suffers from poor readability when chaining operations:

```python
# Nested function calls - hard to read
result = send_data(normalize_data(validate_data(fetch_data(id))))

# Intermediate variables - verbose
data = fetch_data(id)
data = validate_data(data)
data = normalize_data(data)
result = send_data(data)

# Async makes it worse
data = await fetch_data(id)
data = validate_data(data)  # Is this async?
data = await normalize_data(data)  # What if it's sometimes async?
result = await send_data(data)
```

### The Solution

Quent provides a clean, chainable interface that works identically for sync and async code:

```python
from quent import Chain

# Clean, readable, works for both sync and async
result = Chain(fetch_data, id).then(validate_data).then(normalize_data).then(send_data).run()
```

## Core Concepts

### 1. Chain Interface

The `Chain` class enables method chaining for sequential operations:

```python
from quent import Chain

# Basic chaining
result = (
    Chain(fetch_user, user_id)
    .then(validate_permissions)
    .then(apply_transformations)
    .then(save_to_database)
    .run()
)

# With error handling
result = (
    Chain(fetch_resource)
    .then(process_resource)
    .except_(log_error)
    .finally_(cleanup_resources)
    .run()
)
```

### 2. Transparent Async Handling

**This is Quent's killer feature.** Write your code once, use it everywhere:

```python
def process_data(data_source):
    """Works with both sync and async data sources!"""
    return (
        Chain(data_source.fetch)
        .then(validate)
        .then(transform)
        .then(data_source.save)
        .run()
    )

# Use with synchronous code
result = process_data(sync_database)

# Use with asynchronous code - returns a Task that auto-executes
result = process_data(async_database)  # Can be awaited if needed
```

When Quent detects a coroutine, it automatically:
- Wraps it in an `asyncio.Task`
- Schedules it for execution
- Continues the chain evaluation within the task
- Returns the task (which can be awaited or not, as needed)

### 3. Cascade Pattern

The `Cascade` class implements the fluent interface pattern where each method returns the original object:

```python
from quent import Cascade, CascadeAttr

# All operations receive the original data
processed_data = (
    Cascade(fetch_data, id)
    .then(send_to_backup)      # backup service receives data
    .then(send_to_analytics)   # analytics receives data
    .then(log_operation)       # logger receives data
    .run()  # returns the original data
)

# Make any class fluent
result = CascadeAttr(list()).append(1).append(2).extend([3, 4]).run()
# Returns: [1, 2, 3, 4]
```

### 4. Exceptional Exception Handling

Quent provides incredibly clean and informative stack traces that show exactly where in your chain an error occurred, along with the intermediate values at each step:

```python
# When this code fails...
result = (
    Chain(fetch_data, 42)
    .then(validate)
    .then(transform)
    .then(save)
    .run()
)

# You get a beautiful, informative stack trace:
Traceback (most recent call last):
  File "example.py", line 28, in <module>
    .run()
     ^^^^^
  File "<quent>", line 1, in 
    Chain(fetch_data, 42) = {'id': 42, 'value': 100}
    .then(validate) <----
    .then(transform)
    .then(save)
  File "example.py", line 11, in validate
    raise ValueError("Value too large")
ValueError: Value too large
```

Notice how Quent:
- Shows the entire chain structure
- Displays intermediate values (e.g., `= {'id': 42, 'value': 100}`)
- Points to the exact operation that failed with `<----`
- Preserves the original exception and its location
- Works seamlessly with nested chains, showing the full hierarchy

This makes debugging chain operations incredibly straightforward - you can see the exact state of your data at each step and immediately identify where things went wrong.

## Advanced Features

### Flow Control

```python
# Conditional execution
Chain(get_user).then(lambda u: u.age).then(lambda a: a >= 18).if_(grant_access).else_(deny_access)

# Iteration
Chain(get_items).foreach(process_item)

# Context management
Chain(acquire_lock).with_(perform_operation)

# Ignore operations (side effects)
Chain(fetch_data).ignore(log_operation).then(process_data)

# Access root value
Chain(fetch_data, id).then(transform).root(lambda original: log(f"Processed {original}"))
```

### Error Handling

```python
result = (
    Chain(risky_operation)
    .then(process_result)
    .except_(handle_error)      # Called on exception
    .finally_(cleanup, ...)     # Always called (... means no args)
    .run()
)
```

### Template Chains (Reusable)

```python
# Create reusable chain templates
processor = Chain().then(validate).then(normalize).then(save)

# Use with different inputs
for item in items:
    processor.run(item)
```

### Nested Chains

```python
# Chains can be nested for complex workflows
result = (
    Chain(fetch_data)
    .then(Chain().then(validate).then(normalize))
    .then(send_data)
    .run()
)
```

## Real-World Example

Here's how Quent simplifies a Redis wrapper that supports both sync and async modes:

```python
def flush(self) -> Any | Coroutine:
    """Execute pipeline and return results.
    
    Returns a coroutine if using async Redis, otherwise returns the result directly.
    The caller knows which Redis client they're using and can await if needed.
    """
    pipe = self.r.pipeline(transaction=self.transaction)
    self.apply_operations(pipe)
    
    return (
        Chain(pipe.execute, raise_on_error=True)
        .then(self.remove_ignored_commands)
        .finally_(pipe.reset, ...)  # Always reset, even on error
        .run()
    )
```

This single implementation works for both `redis` and `redis.asyncio` without any code duplication or special handling.

## Performance

Quent is built for performance:
- **Cython Implementation**: Core logic compiled to C extensions
- **Minimal Overhead**: Optimized for chain reuse and evaluation
- **Zero-cost Async**: No performance penalty for async transparency

## API Reference

### Chain Methods

#### Core Operations
- `Chain(value, *args, **kwargs)` - Initialize chain with a value
- `.then(value, *args, **kwargs)` - Add operation to chain
- `.run(value, *args, **kwargs)` - Execute chain and return result

#### Flow Control
- `.root(value, *args, **kwargs)` - Access root value
- `.ignore(value, *args, **kwargs)` - Execute without affecting chain
- `.foreach(fn)` - Iterate over current value
- `.with_(value, *args, **kwargs)` - Execute in context

#### Conditionals
- `.if_(on_true, *args, **kwargs)` - Conditional execution
- `.else_(on_false, *args, **kwargs)` - Alternative execution
- `.eq(value, on_true, *args, **kwargs)` - Equality check
- `.in_(value, on_true, *args, **kwargs)` - Membership test

#### Error Handling
- `.except_(handler, *args, **kwargs)` - Exception handler
- `.finally_(cleanup, *args, **kwargs)` - Cleanup (always runs)

### Special Arguments

- **Ellipsis (`...`)**: When passed as first argument, calls function without arguments
  ```python
  Chain(data).then(process).finally_(cleanup, ...)  # cleanup()
  ```

## Installation Requirements

- Python 3.8+
- No runtime dependencies
- Cython (build-time only)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.
