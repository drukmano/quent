# Quent vs tenacity

Both Quent and [tenacity](https://github.com/jd/tenacity) provide retry functionality for Python. They differ in scope and composition model.

## What tenacity Does

tenacity is a dedicated retry library that uses decorators to add retry behavior to functions. It is mature, widely used (~247M downloads/month), and provides extensive retry configuration including wait strategies, stop conditions, retry filtering, and callbacks.

### tenacity Example

```python
import tenacity

@tenacity.retry(
  stop=tenacity.stop_after_attempt(3),
  wait=tenacity.wait_fixed(1),
  retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
)
def call_api(url):
  response = requests.get(url)
  response.raise_for_status()
  return response.json()

result = call_api("https://api.example.com/data")
```

For async:

```python
@tenacity.retry(
  stop=tenacity.stop_after_attempt(3),
  wait=tenacity.wait_fixed(1),
  retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError)),
)
async def call_api_async(url):
  async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
      return await response.json()

result = await call_api_async("https://api.example.com/data")
```

## What Quent Offers

Quent is a pipeline library with retry as one of several built-in resilience features. Retry is composed inline within the chain, alongside timeout, error handling, and context propagation.

### Quent Example

```python
from quent import Chain

def call_api(url):
  return (
    Chain(http_client.get, url)
    .retry(3, delay=1.0, exceptions=(ConnectionError, TimeoutError))
    .timeout(10.0)
    .then(lambda r: r.json())
    .except_(log_error, reraise=False)
    .run()
  )

# Works for both sync and async HTTP clients
result = call_api("https://api.example.com/data")         # sync
result = await call_api("https://api.example.com/data")    # async
```

## Side-by-Side: Retry Use Case

### tenacity Approach

```python
import tenacity

@tenacity.retry(
  stop=tenacity.stop_after_attempt(3),
  wait=tenacity.wait_fixed(0.5),
  retry=tenacity.retry_if_exception_type(ConnectionError),
)
def fetch_user(user_id):
  response = requests.get(f"/users/{user_id}")
  response.raise_for_status()
  return response.json()

def process_user(user_id):
  try:
    user = fetch_user(user_id)
    validated = validate(user)
    return save(validated)
  except Exception as e:
    logger.error(f"Failed: {e}")
    return None
```

### Quent Approach

```python
from quent import Chain

def process_user(user_id):
  return (
    Chain(requests.get, f"/users/{user_id}")
    .retry(3, delay=0.5, exceptions=(ConnectionError,))
    .then(lambda r: r.json())
    .then(validate)
    .then(save)
    .except_(lambda e: logger.error(f"Failed: {e}"), reraise=False)
    .run()
  )
```

## Comparison

| Aspect | tenacity | Quent |
|--------|----------|-------|
| **Primary purpose** | Retry logic | Pipeline with retry built in |
| **Composition model** | Decorators | Fluent chain |
| **Retry configuration** | Extensive (exponential backoff, jitter, custom wait strategies, stop conditions, callbacks) | Basic (count, delay, exception filter) |
| **Timeout** | Not included (use asyncio.timeout separately) | Built in (`.timeout()`) |
| **Error handling** | Via callbacks or try/except around the call | Built in (`.except_()`, `.finally_()`) |
| **Sync/async** | Separate `@retry` for sync, auto-detects async | Transparent -- same chain works for both |
| **Context propagation** | Not included | Built in (`.with_context()`) |
| **Pipeline composition** | Not included | Core feature |
| **Dependencies** | tenacity | quent (zero deps) |
| **Maturity** | Very mature, widely used | Newer |
| **Python version** | 3.8+ | 3.14+ |

## tenacity's Advanced Retry Features

tenacity provides retry capabilities that Quent does not match in isolation:

- **Wait strategies**: exponential backoff, random jitter, combined strategies
- **Stop conditions**: after N attempts, after N seconds, custom predicates
- **Retry predicates**: retry on specific return values, not just exceptions
- **Before/after callbacks**: hooks before each attempt, after each attempt
- **Statistics**: access retry statistics (attempt count, elapsed time)
- **Reraise**: control which exception is re-raised (first, last, or custom)

If you need these advanced retry configurations, tenacity is the better choice for the retry component specifically.

## Using Them Together

tenacity and Quent are not mutually exclusive. You can use tenacity-decorated functions inside a Quent chain:

```python
import tenacity
from quent import Chain

@tenacity.retry(
  stop=tenacity.stop_after_attempt(5),
  wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
  retry=tenacity.retry_if_exception_type(ConnectionError),
)
def fetch_with_backoff(url):
  return requests.get(url).json()

# Use tenacity for advanced retry, Quent for the rest of the pipeline
result = (
  Chain(fetch_with_backoff, url)
  .timeout(30.0)
  .then(validate)
  .then(transform)
  .then(save)
  .except_(handle_error, reraise=False)
  .run()
)
```

## When to Use Which

**Use tenacity when:**

- You only need retry logic without a pipeline
- You need advanced retry features (exponential backoff with jitter, retry on return values, statistics)
- You need to support Python < 3.14
- You want a mature, battle-tested retry solution

**Use Quent when:**

- Retry is one part of a larger pipeline (fetch, validate, transform, save)
- You want retry, timeout, and error handling composed in one place
- You need transparent sync/async handling
- Basic retry configuration (count, delay, exception filter) is sufficient

**Use both when:**

- You need advanced tenacity retry features inside a Quent pipeline

## Further Reading

- [Resilience](../guide/resilience.md) -- Quent's retry, timeout, and safe_run in detail
- [Error Handling](../guide/error-handling.md) -- Quent's except_ and finally_
- [Quent vs unasync](vs-unasync.md) -- Comparison with the code generation approach
- [Quent vs pipe](vs-pipe.md) -- Comparison with the pipe library
