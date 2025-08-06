# Context Propagation

Quent supports passing metadata through chain execution using Python's `contextvars`. The context is isolated per chain run and works across async boundaries.

## Attaching Context

Use `.with_context()` to attach key-value metadata to a chain:

```python
from quent import Chain

result = (
  Chain(fetch_user, user_id)
  .with_context(request_id="abc-123", trace=True)
  .then(validate)
  .then(enrich)
  .run()
)
```

The context is available to any operation in the chain via `Chain.get_context()`.

## Reading Context

Use the static method `Chain.get_context()` inside any chain operation to retrieve the current context:

```python
from quent import Chain

def enrich(user):
  ctx = Chain.get_context()
  request_id = ctx["request_id"]
  # Use request_id for logging, tracing, etc.
  return user

result = (
  Chain(fetch_user, user_id)
  .with_context(request_id="abc-123", trace=True)
  .then(validate)
  .then(enrich)
  .run()
)
```

`Chain.get_context()` returns a `dict` containing all the key-value pairs passed to `.with_context()`.

## Use Cases

### Request-Scoped Metadata

Pass request identifiers through a processing pipeline for logging and tracing:

```python
from quent import Chain

def api_request(method, url, **kwargs):
  return (
    Chain(http_client.request, method, url, **kwargs)
    .with_context(method=method, url=url)
    .retry(3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
    .timeout(10.0)
    .then(validate_response)
    .then(parse_json)
    .except_(log_api_error, reraise=False)
    .run()
  )

def validate_response(response):
  ctx = Chain.get_context()
  if response.status >= 400:
    raise ValueError(f"{ctx['method']} {ctx['url']} returned {response.status}")
  return response

def log_api_error(exc):
  ctx = Chain.get_context()
  logger.error(f"API error for {ctx['method']} {ctx['url']}: {exc}")
```

The context flows through the entire chain, so both `validate_response` and `log_api_error` can access the request method and URL without those values being passed through intermediate operations.

### Logging Context

Attach structured logging metadata to every operation in a processing pipeline:

```python
from quent import Chain
import logging

logger = logging.getLogger(__name__)

def log_step(step_name):
  def _log(value):
    ctx = Chain.get_context()
    logger.info(
      f"[{ctx.get('trace_id', 'unknown')}] {step_name}: "
      f"processing {type(value).__name__}"
    )
    return value
  return _log

result = (
  Chain(fetch_data, id)
  .with_context(trace_id="tx-001", user="alice")
  .do(log_step("fetch"))
  .then(validate)
  .do(log_step("validate"))
  .then(transform)
  .do(log_step("transform"))
  .then(save)
  .run()
)
```

### Transaction Identifiers

Track operations through a multi-step transaction:

```python
from quent import Chain
import uuid

def process_order(order):
  tx_id = str(uuid.uuid4())
  return (
    Chain(validate_order, order)
    .with_context(transaction_id=tx_id, order_id=order["id"])
    .then(reserve_inventory)
    .then(charge_payment)
    .then(confirm_order)
    .except_(rollback_transaction, reraise=False)
    .run()
  )

def rollback_transaction(exc):
  ctx = Chain.get_context()
  logger.error(
    f"Transaction {ctx['transaction_id']} failed for order "
    f"{ctx['order_id']}: {exc}"
  )
  cancel_reservation(ctx["order_id"])
```

## Context Isolation

Each chain run has its own context. Concurrent chain executions do not interfere with each other because the context is stored in a `ContextVar`, which is isolated per task/thread.

```python
import asyncio
from quent import Chain

async def process(item, trace_id):
  return await (
    Chain(fetch, item)
    .with_context(trace_id=trace_id)
    .then(transform)
    .run()
  )

# Each task has its own context -- no interference
async def main():
  await asyncio.gather(
    process("a", "trace-1"),
    process("b", "trace-2"),
    process("c", "trace-3"),
  )
```

## Further Reading

- [Resilience](resilience.md) -- Context is useful with retry and error handling
- [Error Handling](error-handling.md) -- Access context in exception handlers
- [Chains & Cascades](chains.md) -- Core chain operations
- [API Reference](../reference.md) -- `with_context()` and `get_context()` signatures
