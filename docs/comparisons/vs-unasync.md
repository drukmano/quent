# Quent vs unasync

Both Quent and [unasync](https://github.com/python-trio/unasync) solve the same fundamental problem: avoiding the need to maintain separate synchronous and asynchronous implementations of the same logic. They take fundamentally different approaches.

## What unasync Does

unasync is a code generation tool. You write the async version of your code, and unasync generates the synchronous version by mechanically transforming it:

- `async def` becomes `def`
- `await` is removed
- `async for` becomes `for`
- `async with` becomes `with`
- Import paths are rewritten (e.g., `asyncio` references are adjusted)

The transformation happens at build time, typically via a setup.py hook or a CLI command.

### unasync Example

You write the async version:

```python
# _async/client.py (source of truth)
import httpx

class AsyncClient:
  async def get_user(self, user_id: int):
    async with httpx.AsyncClient() as client:
      response = await client.get(f"/users/{user_id}")
      return response.json()

  async def save_user(self, user):
    async with httpx.AsyncClient() as client:
      response = await client.put(f"/users/{user['id']}", json=user)
      return response.json()
```

unasync generates the sync version:

```python
# _sync/client.py (auto-generated)
import httpx

class SyncClient:
  def get_user(self, user_id: int):
    with httpx.Client() as client:
      response = client.get(f"/users/{user_id}")
      return response.json()

  def save_user(self, user):
    with httpx.Client() as client:
      response = client.put(f"/users/{user['id']}", json=user)
      return response.json()
```

## What Quent Does

Quent takes a runtime approach. Instead of generating two versions of the code, you write a single chain that detects coroutines at execution time and transitions between sync and async automatically.

### Quent Example

```python
from quent import Chain

class Client:
  def __init__(self, http_client):
    # http_client can be sync (httpx.Client) or async (httpx.AsyncClient)
    self.http = http_client

  def get_user(self, user_id: int):
    return (
      Chain(self.http.get, f"/users/{user_id}")
      .then(lambda r: r.json())
      .run()
    )

  def save_user(self, user):
    return (
      Chain(self.http.put, f"/users/{user['id']}", json=user)
      .then(lambda r: r.json())
      .run()
    )
```

The same `Client` class works with both sync and async HTTP clients:

```python
# Synchronous
sync_client = Client(httpx.Client())
user = sync_client.get_user(42)

# Asynchronous
async_client = Client(httpx.AsyncClient())
user = await async_client.get_user(42)
```

## Side-by-Side Comparison

| Aspect | unasync | Quent |
|--------|---------|-------|
| **Approach** | Code generation (build-time) | Runtime coroutine detection |
| **Source files** | Write async, generate sync | Write once, works for both |
| **Build step required** | Yes | No |
| **Generated code to maintain** | Yes (sync version) | No |
| **Import path rewriting** | Yes (manual configuration) | Not needed |
| **Works with any API shape** | Yes (mechanical transformation) | Requires chain-compatible API |
| **Async-specific patterns** | Handles `async for`, `async with` | Handles via `.iterate()`, `.with_()` |
| **Built-in error handling** | No | Yes (except, finally, enhanced tracebacks) |
| **Performance overhead** | Zero (generated code is native) | Negligible (pure Python) |
| **Dependencies** | unasync (build-time only) | quent (runtime) |

## Pros and Cons

### unasync

**Pros:**

- Zero runtime overhead -- the generated sync code is identical to hand-written code
- Works with any code pattern, not just pipelines
- Mature and battle-tested (used by httpcore, urllib3, and other major libraries)
- No runtime dependency

**Cons:**

- Requires a build step and configuration
- Generated code must be committed or generated during CI
- Import path rewriting can be fragile and requires manual mapping
- The sync and async implementations must have exactly parallel structure
- Does not help with mixed sync/async in the same code path
- No additional features (error handling, etc.)

### Quent

**Pros:**

- No code generation or build step
- Single implementation -- no generated files to track
- Includes error handling and enhanced tracebacks
- Handles mixed sync/async in the same chain
- Pure Python with minimal overhead

**Cons:**

- Runtime dependency (quent must be installed)
- API must be structured as a chain/pipeline
- Not suitable for complex async patterns (`async for` loops with complex bodies, deeply nested `async with` blocks)
- Newer and less battle-tested than unasync
- Requires Python >= 3.10

## When to Use Which

**Use unasync when:**

- You are maintaining a foundational library (like an HTTP client) where zero runtime dependencies matter
- Your async and sync implementations have complex, non-pipeline structures
- You need `async for` and `async with` transformations across large codebases
- You want the generated sync code to be fully independent and debuggable on its own

**Use Quent when:**

- Your operations naturally form a pipeline or chain
- You want error handling composed in the pipeline
- You want a single runtime implementation without generated code
- You are building application-level code (API clients, service layers, data pipelines)
- You want the same function to handle both sync and async callers transparently

## Further Reading

- [Async Handling](../guide/async.md) -- How Quent's transparent async works in detail
