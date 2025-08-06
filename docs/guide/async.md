# Async Handling

Quent's defining feature is transparent sync/async handling. You write your pipeline code once, and it works correctly regardless of whether the functions involved are synchronous or asynchronous.

## The Problem

Python's type system treats sync and async functions as fundamentally different types. This creates what is known as the "function coloring" problem: once a function is async, every caller must also be async, and code written for one world cannot be used in the other without modification.

```python
# Synchronous version
def get_user_sync(db, user_id):
  data = db.fetch(user_id)
  data = validate(data)
  data = normalize(data)
  return save(data)

# Async version -- structurally identical, but must be written separately
async def get_user_async(db, user_id):
  data = await db.fetch(user_id)
  data = validate(data)
  data = await normalize(data)
  return await save(data)
```

Library authors face this constantly: maintain two nearly-identical implementations, use code generation tools like unasync, or force callers into one paradigm.

## How Quent Solves It

Quent detects coroutines at runtime and transitions the chain to async execution at the exact point where an awaitable is encountered. There is no upfront async/sync decision.

```python
from quent import Chain

def process_data(data_source):
  """Works with both sync and async data sources."""
  return (
    Chain(data_source.fetch)
    .then(validate)
    .then(transform)
    .then(data_source.save)
    .run()
  )

# Synchronous -- returns the result directly
result = process_data(sync_database)

# Asynchronous -- returns a Task that can be awaited
result = await process_data(async_database)
```

The same `process_data` function works for both cases. When `data_source.fetch` returns a regular value, the chain executes synchronously. When it returns a coroutine, Quent wraps it in a task and continues asynchronously.

## The Transition Mechanism

When Quent encounters a coroutine during chain evaluation, it:

1. **Wraps it in an `asyncio.Task`** with `eager_start=True` for immediate execution
2. **Continues the remaining chain** evaluation within the task
3. **Returns the task**, which the caller can `await` if needed

The `eager_start=True` flag is significant: sync-completing coroutines execute immediately without a round-trip through the event loop. This yields 2-5x faster async chains compared to standard task scheduling.

## Mixing Sync and Async

You can freely mix synchronous and asynchronous operations in the same chain. Quent handles the transitions automatically.

```python
from quent import Chain

def validate(data):
  """Pure sync validation."""
  if not data:
    raise ValueError("Empty data")
  return data

async def fetch_from_api(query):
  """Async HTTP call."""
  async with aiohttp.ClientSession() as session:
    async with session.get(f"https://api.example.com/{query}") as resp:
      return await resp.json()

def transform(data):
  """Pure sync transformation."""
  return {k: v.upper() for k, v in data.items()}

async def save_to_db(data):
  """Async database write."""
  await db.insert(data)
  return data

# All four operations in one chain -- Quent handles the sync/async boundaries
result = await (
  Chain(fetch_from_api, "users/1")  # async -- chain transitions here
  .then(validate)                    # sync -- runs inside the task
  .then(transform)                   # sync -- runs inside the task
  .then(save_to_db)                  # async -- awaited inside the task
  .run()
)
```

The chain starts by calling `fetch_from_api`, which returns a coroutine. At that point, Quent creates a task and continues all remaining operations (`validate`, `transform`, `save_to_db`) within that task. If `save_to_db` also returns a coroutine, it is awaited inside the same task context.

## Writing Dual-Mode Libraries

Quent's primary use case is enabling library authors to write a single implementation that serves both sync and async callers.

### Example: Redis Pipeline

A single implementation that works for both `redis` and `redis.asyncio`:

```python
from quent import Chain
from typing import Any, Coroutine

def flush(self) -> Any | Coroutine:
  """Execute pipeline and return results.

  Returns a coroutine if using async Redis, otherwise returns
  the result directly. The caller knows which client they're
  using and can await if needed.
  """
  pipe = self.r.pipeline(transaction=self.transaction)
  self.apply_operations(pipe)

  return (
    Chain(pipe.execute, raise_on_error=True)
    .then(self.remove_ignored_commands)
    .finally_(pipe.reset, ...)  # always reset, even on error
    .run()
  )
```

The caller uses this naturally:

```python
# Sync caller
results = pipeline.flush()

# Async caller
results = await pipeline.flush()
```

### Example: Generic Service Layer

```python
from quent import Chain

class UserService:
  def __init__(self, db):
    self.db = db  # can be sync or async database client

  def get_user(self, user_id):
    return (
      Chain(self.db.fetch, "users", user_id)
      .then(self._validate)
      .then(self._enrich)
      .run()
    )

  def _validate(self, user):
    if user is None:
      raise ValueError("User not found")
    return user

  def _enrich(self, user):
    user["full_name"] = f"{user['first']} {user['last']}"
    return user
```

This single `UserService` class works with any database client -- sync ORM, async ORM, or a mock -- without any code changes.

## Performance Implications

Quent is designed for minimal overhead:

- **C-level coroutine detection**: Exact type identity checks instead of `isinstance` calls, implemented in Cython
- **Eager task creation**: `asyncio.create_task(eager_start=True)` allows sync-completing coroutines to execute immediately without event loop round-trips
- **No unnecessary wrapping**: Sync chains execute with no async overhead at all -- they are pure C-level function calls

### Benchmark Results

Average of 10 iterations of 100,000 loops each:

| Scenario | Time (seconds) |
|----------|---------------|
| Direct function call | 1.19 |
| With Quent chain | 1.20 |
| With Quent frozen chain | 1.06 |

Quent adds negligible overhead to direct function calls. Frozen chains can actually be faster due to pre-built link structures that avoid repeated chain construction.

## The `autorun` Option

When enabled, async chain results are automatically scheduled via `asyncio.create_task`:

```python
from quent import Chain

Chain(async_operation).autorun().run()  # Task is scheduled immediately
```

This is useful in fire-and-forget scenarios where you want to schedule an async operation without explicitly awaiting the result.

## Further Reading

- [Getting Started](../getting-started.md) -- Basic chain usage
- [Chains & Cascades](chains.md) -- All chain types in depth
- [Resilience](resilience.md) -- Retry and timeout (both work across sync/async)
- [Comparisons: vs unasync](../comparisons/vs-unasync.md) -- Quent's runtime approach vs unasync's code generation
