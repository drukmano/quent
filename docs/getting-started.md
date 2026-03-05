# Getting Started

## Installation

```bash
pip install quent
```

Quent requires Python 3.10 or later and has no runtime dependencies.

## Your First Chain

A `Chain` sequences operations where each step receives the result of the previous one. Call `.run()` to execute.

```python
from quent import Chain

def fetch_user(user_id):
  return {"id": user_id, "name": "Alice", "email": "alice@example.com"}

def validate(user):
  if not user.get("email"):
    raise ValueError("Missing email")
  return user

def format_greeting(user):
  return f"Hello, {user['name']}!"

# Build and execute the chain
result = (
  Chain(fetch_user, 42)
  .then(validate)
  .then(format_greeting)
  .run()
)
print(result)  # "Hello, Alice!"
```

Here is what happens step by step:

1. `Chain(fetch_user, 42)` -- Sets `fetch_user` as the first operation, with `42` as its argument. This becomes the **root value** of the chain.
2. `.then(validate)` -- Adds `validate` as the next operation. It will receive the result of `fetch_user(42)`.
3. `.then(format_greeting)` -- Adds `format_greeting`. It will receive the result of `validate(...)`.
4. `.run()` -- Executes the entire chain and returns the final result.

## Your First Async Chain

The same chain works transparently with async functions. No changes to the chain structure are needed.

```python
from quent import Chain

async def fetch_user(user_id):
  # Imagine this calls an async database or HTTP client
  return {"id": user_id, "name": "Alice", "email": "alice@example.com"}

def validate(user):
  if not user.get("email"):
    raise ValueError("Missing email")
  return user

async def save_user(user):
  # Imagine this writes to an async database
  return user

# The exact same chain structure works with async functions
result = await (
  Chain(fetch_user, 42)
  .then(validate)
  .then(save_user)
  .run()
)
```

When Quent encounters a coroutine during chain evaluation, it:

1. Wraps it in an `asyncio.Task` with `eager_start=True` for immediate execution
2. Continues the remaining chain evaluation within the task
3. Returns the task, which the caller can `await`

The chain starts synchronous and only transitions to async at the exact point where an awaitable value is first encountered -- there is no upfront async/sync decision.

## Side Effects with `.do()`

Use `.do()` when you want to perform a side effect (like logging) without changing the current value in the chain.

```python
result = (
  Chain(fetch_data, id)
  .do(lambda data: print(f"Fetched: {data}"))  # prints but discards return
  .then(transform)                              # receives fetch_data result, not print's None
  .run()
)
```

## Next Steps

Now that you understand the basics, explore the detailed guides:

- [Chains](guide/chains.md) -- Chain operations in depth
- [Async Handling](guide/async.md) -- How transparent async works under the hood
- [Error Handling](guide/error-handling.md) -- Exception handling and enhanced stack traces
- [Reuse & Patterns](guide/reuse.md) -- freeze and decorator
- [API Reference](reference.md) -- Full method signatures and descriptions
