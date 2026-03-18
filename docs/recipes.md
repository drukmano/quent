---
title: "Recipes — Practical Pipeline Patterns"
description: "Real-world quent recipes: API clients with error handling, data processing pipelines, file handling, validation pipelines, concurrent enrichment, and more."
tags:
  - recipes
  - examples
  - patterns
  - real-world
search:
  boost: 6
---

# Recipes

Practical, copy-and-paste patterns for common tasks. Each recipe is a self-contained example that demonstrates how quent's pipeline primitives compose to solve real problems.

Every recipe works identically with sync and async callables -- that is the whole point. Where the sync/async duality matters, it is called out explicitly.

---

## API Client with Error Handling

Fetch data from an API, parse the response, and validate the result. If a
connection error occurs, the exception handler returns a structured
error response instead of raising.

```python
import logging
from quent import Q

log = logging.getLogger(__name__)

def fetch_and_parse(url):
  """Fetch, parse, and validate a JSON API response.

  Works with both sync (requests) and async (aiohttp) HTTP clients --
  the same pipeline handles either transparently.
  """
  return (
    Q(url)
    .then(http_client.get)
    .then(lambda r: r.json())
    .then(validate_response)
    .except_(
      lambda ei: {'error': str(ei.exc), 'url': ei.root_value},
      exceptions=ConnectionError,
    )
    .finally_(lambda url: log.info('Request to %s completed', url))
    .run()
  )

# Sync caller
result = fetch_and_parse('https://api.example.com/data')

# Async caller (if http_client.get is async)
result = await fetch_and_parse('https://api.example.com/data')
```

Key points:

- **`except_` receives a single `QuentExcInfo(exc, root_value)` NamedTuple.** Access the exception via `.exc` and the root value via `.root_value`.
- **`finally_` also receives the root value.** It logs regardless of success or failure. Its return value is always discarded.

---

## Data Processing Pipeline

An ETL-style pipeline: extract records from a source, log counts, filter invalid
entries, normalize the valid ones, and load them into a database. `.do()` steps
log progress without altering the data flowing through the pipeline.

```python
import logging
from quent import Q

log = logging.getLogger(__name__)

def is_valid(record):
  return record.get('status') != 'deleted' and 'id' in record

def normalize(record):
  return {
    'id': record['id'],
    'name': record.get('name', '').strip().title(),
    'email': record.get('email', '').lower(),
  }

pipeline = (
  Q()
  .then(extract_records)
  .do(lambda records: log.info('Extracted %d records', len(records)))
  .then(lambda records: [r for r in records if is_valid(r)])
  .do(lambda records: log.info('Filtered to %d valid records', len(records)))
  .foreach(normalize)
  .then(load_to_database)
  .run
)

# Reusable: call with different sources
pipeline('s3://bucket/data.json')
pipeline('postgres://db/table')
```

Key points:

- **`.do()` is for side-effects.** The logging steps observe the current value without modifying it.
- **`.then()` with a list comprehension** filters records before passing them to `.foreach()` for transformation.
- **`pipeline = ... .run`** (no parentheses) captures the bound method, making the pipeline callable like a regular function.

---

## File Processing with Context Manager

Read a file, process its content, and return the result. The context manager
ensures the file handle is properly closed even if processing raises.

```python
from quent import Q

def process_content(text):
  lines = text.strip().splitlines()
  return [line for line in lines if not line.startswith('#')]

result = (
  Q(input_path)
  .then(open)
  .with_(lambda f: f.read())
  .then(process_content)
  .run()
)
```

For a write-after-read pattern with two context managers:

```python
from quent import Q

def transform(text):
  return text.upper()

def read_and_write(input_path, output_path):
  return (
    Q(input_path)
    .then(open)
    .with_(lambda f: f.read())
    .then(transform)
    .do(lambda content: (
      Q(output_path)
      .then(lambda p: open(p, 'w'))
      .with_(lambda f: f.write(content))
      .run()
    ))
    .run()
  )
```

Key points:

- **`.with_(fn)` enters the current value as a context manager.** `fn` receives the context value and its result replaces the current pipeline value.
- **Context managers are properly closed** on both success and exception.
- **Works with async context managers too** -- if `open()` returns an async context manager (e.g., `aiofiles.open()`), quent handles it transparently.

---

## Validation Pipeline with Early Return

Multi-step validation where each step can short-circuit the pipeline by returning
early. If any check fails, the pipeline exits immediately with a structured error.

```python
from quent import Q

def check_required(data):
  if not data.get('name'):
    return Q.return_({'valid': False, 'error': 'name is required'})
  if not data.get('email'):
    return Q.return_({'valid': False, 'error': 'email is required'})
  return data

def check_format(data):
  if '@' not in data.get('email', ''):
    return Q.return_({'valid': False, 'error': 'invalid email format'})
  return data

def check_permissions(data):
  if data.get('role') not in ('admin', 'user'):
    return Q.return_({'valid': False, 'error': 'invalid role'})
  return data

def validate(data):
  return (
    Q(data)
    .then(check_required)
    .then(check_format)
    .then(check_permissions)
    .then(lambda d: {'valid': True, 'data': d})
    .except_(lambda ei: {'valid': False, 'error': str(ei.exc)})
    .run()
  )

result = validate({'name': 'Alice', 'email': 'alice@example.com', 'role': 'admin'})
# {'valid': True, 'data': {'name': 'Alice', ...}}

result = validate({'name': 'Bob'})
# {'valid': False, 'error': 'email is required'}
```

Key points:

- **`Q.return_(value)` exits the pipeline immediately.** Must be used as `return Q.return_(...)`.
- **`except_` is a safety net.** Catches unexpected exceptions. Control flow signals (`return_`, `break_`) bypass `except_`.
- **Validators are plain functions.** They receive data, check it, and either return it or call `Q.return_()`.

---

## Concurrent Enrichment with Gather

Enrich a user record by fetching data from multiple sources simultaneously.

```python
from quent import Q

def get_preferences(user):
  return db.query('SELECT * FROM prefs WHERE user_id = %s', user['id'])

def get_purchase_history(user):
  return api.get(f'/purchases/{user["id"]}')

def get_recommendations(user):
  return ml_service.recommend(user['id'])

enriched = (
  Q(user_id)
  .then(fetch_user)
  .gather(
    get_preferences,
    get_purchase_history,
    get_recommendations,
  )
  .then(lambda results: {
    'preferences': results[0],
    'purchases': results[1],
    'recommendations': results[2],
  })
  .run()
)
```

Key points:

- **`.gather()` calls each function with the current value.** Results are returned as a tuple in the same order.
- **Always concurrent.** Sync: `ThreadPoolExecutor`. Async: `TaskGroup`/`asyncio.gather`.
- **Access results by index** or destructure in a lambda.
- **Multiple failures produce an `ExceptionGroup`.** A single failure propagates directly.

---

## Conditional Processing

Route data through different processing paths based on a runtime condition.

```python
from quent import Q

result = (
  Q(request)
  .if_(lambda r: r.content_type == 'application/json').then(parse_json)
  .else_(parse_xml)
  .then(validate)
  .then(process)
  .run()
)
```

When the predicate is `None`, the truthiness of the current value is used:

```python
from quent import Q

def fetch_cached(key):
  return (
    Q(key)
    .then(cache.get)
    .if_().then(lambda v: v)          # truthy -> cache hit
    .else_(lambda _: db.fetch(key))  # falsy -> cache miss
    .run()
  )
```

Key points:

- **`.if_(predicate).then(fn)`** evaluates the predicate. If truthy, `fn` runs and its result replaces the current value.
- **`.else_()` must immediately follow `.if_()`** with no operations in between.
- **Both predicate and branch functions** can be sync or async.

---

## Iteration with Early Termination

Process items one at a time, stopping when a condition is met.

```python
from quent import Q

def process_item(item):
  result = transform(item)
  if result.get('found'):
    return Q.break_(result)  # stop, yield this as final item
  return result

# Sync
for result in Q(items).iterate(process_item):
  if result.get('found'):
    return result

# Async -- same pipeline, same iterate() call
async for result in Q(items).iterate(process_item):
  if result.get('found'):
    return result
```

Key points:

- **`.iterate(fn)` yields each element** of the pipeline's output. If `fn` is provided, it transforms each element.
- **`Q.break_(value)` stops iteration early.** The optional value is yielded as the final element.
- **Same definition, both iteration modes.** `__iter__` for sync, `__aiter__` for async.

---

## Retry with Exponential Backoff

A reusable retry wrapper using nested pipelines and `except_()`.

```python
import time
import random
from quent import Q

def retry(fn, *, max_attempts=3, base_delay=0.1):
  """Wrap a callable with exponential backoff retry."""
  def wrapper(*args, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
      try:
        return fn(*args, **kwargs)
      except Exception as exc:
        last_exc = exc
        if attempt < max_attempts - 1:
          delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
          time.sleep(delay)
    raise last_exc
  return wrapper

# Use in a pipeline
result = (
  Q('https://api.example.com/data')
  .then(retry(http_client.get, max_attempts=4, base_delay=0.05))
  .then(lambda r: r.json())
  .except_(lambda ei: {'error': str(ei.exc), 'url': ei.root_value})
  .run()
)
```

For async retry, replace `time.sleep` with `asyncio.sleep`:

```python
import asyncio
import random

def async_retry(fn, *, max_attempts=3, base_delay=0.1):
  """Wrap an async callable with exponential backoff retry."""
  async def wrapper(*args, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
      try:
        return await fn(*args, **kwargs)
      except Exception as exc:
        last_exc = exc
        if attempt < max_attempts - 1:
          delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
          await asyncio.sleep(delay)
    raise last_exc
  return wrapper
```

---

## Circuit Breaker Pattern

Prevent cascading failures by tracking error rates with a closure.

```python
import time
from quent import Q

def make_circuit_breaker(threshold=5, reset_timeout=30):
  """Create a circuit breaker that opens after `threshold` failures."""
  state = {'failures': 0, 'open_until': 0}

  def check_circuit(value):
    if time.monotonic() < state['open_until']:
      return Q.return_({'error': 'circuit open', 'retry_after': state['open_until'] - time.monotonic()})
    return value

  def on_success(result):
    state['failures'] = 0

  def on_failure(ei):
    state['failures'] += 1
    if state['failures'] >= threshold:
      state['open_until'] = time.monotonic() + reset_timeout
    raise ei.exc  # re-raise after recording

  return check_circuit, on_success, on_failure

check, on_ok, on_fail = make_circuit_breaker(threshold=3, reset_timeout=60)

result = (
  Q(request)
  .then(check)
  .then(call_service)
  .do(on_ok)
  .except_(on_fail)
  .run()
)
```

---

## Rate-Limited Concurrent Processing

Process items concurrently with a bounded concurrency limit.

```python
from quent import Q

# Process at most 4 items concurrently
results = (
  Q(urls)
  .foreach(fetch_and_parse, concurrency=4)
  .then(lambda results: [r for r in results if r['status'] == 'ok'])
  .run()
)

# Gather with a concurrency cap
result = (
  Q(user_id)
  .gather(
    get_profile,
    get_orders,
    get_preferences,
    concurrency=2,  # at most 2 running at once
  )
  .then(lambda r: merge_user_data(r[0], r[1], r[2]))
  .run()
)
```

Key points:

- **`concurrency=N` limits simultaneous executions.** Sync: `ThreadPoolExecutor(max_workers=N)`. Async: `asyncio.Semaphore(N)`.
- **Results preserve input order** regardless of completion order.
- **Mixed sync/async not supported** within a single concurrent operation -- the first item's return type determines the path.

---

## Caching Pipeline

Memoize expensive computations using a wrapper.

```python
from functools import lru_cache
from quent import Q

@lru_cache(maxsize=128)
def expensive_transform(data):
  # CPU-intensive computation
  return complex_algorithm(data)

result = (
  Q(raw_data)
  .then(normalize)
  .then(expensive_transform)
  .then(format_output)
  .run()
)
```

For a more sophisticated cache-through pattern:

```python
from quent import Q

def make_cache_steps(cache, key_fn):
  """Return (check, store) callables for cache-through logic."""
  def check_cache(value):
    key = key_fn(value)
    cached = cache.get(key)
    if cached is not None:
      return Q.return_(cached)
    return value

  def store_result(result):
    cache.set(key_fn(result), result)

  return check_cache, store_result

check_cache, store_result = make_cache_steps(redis_cache, lambda uid: f'user:{uid}')

result = (
  Q(user_id)
  .then(check_cache)
  .then(fetch_user)
  .then(enrich_profile)
  .do(store_result)
  .run()
)
```

---

## Streaming with iterate

Lazy processing of large datasets using `iterate()`.

```python
from quent import Q

def fetch_pages():
  """Generator that yields pages of data."""
  page = 1
  while True:
    data = fetch_page(page)
    if not data:
      break
    yield from data
    page += 1

# Process items lazily -- only one page at a time in memory
for item in Q(fetch_pages).iterate(transform):
  save(item)

# Async variant
async for item in Q(async_fetch_pages).iterate(async_transform):
  await save(item)
```

---

## Fan-Out/Fan-In

Split processing across multiple functions, then merge results.

```python
from quent import Q

def analyze_text(text):
  return (
    Q(text)
    .gather(
      count_words,
      detect_language,
      extract_entities,
      compute_sentiment,
    )
    .then(lambda r: {
      'word_count': r[0],
      'language': r[1],
      'entities': r[2],
      'sentiment': r[3],
    })
    .run()
  )
```

For nested fan-out where gathered results are further processed:

```python
from quent import Q

result = (
  Q(document)
  .gather(
    extract_metadata,
    extract_content,
  )
  .then(lambda r: (
    Q(r[1])
    .gather(
      summarize,
      classify,
      extract_keywords,
    )
    .then(lambda inner: {
      'metadata': r[0],
      'summary': inner[0],
      'category': inner[1],
      'keywords': inner[2],
    })
    .run()
  ))
  .run()
)
```

---

## Redis Batch Pipeline

A Redis pipeline wrapper that works with both sync and async clients.

```python
from quent import Q

class RedisBatch:
  def __init__(self, r, transaction=False):
    self.r = r
    self.transaction = transaction
    self.operations = []

  def add(self, op):
    self.operations.append(op)

  def flush(self):
    pipe = self.r.pipeline(transaction=self.transaction)
    for op in self.operations:
      op(pipe)
    return (
      Q(pipe.execute)
      .then(self.process_results)
      .finally_(lambda _: pipe.reset())
      .run()
    )

  def process_results(self, results):
    processed = []
    for op, result in zip(self.operations, results):
      processed.append({'op': op.__name__, 'result': result})
    self.operations.clear()
    return processed

# Works with sync redis
import redis
batch = RedisBatch(redis.Redis())
batch.add(lambda pipe: pipe.set('key1', 'value1'))
results = batch.flush()

# Works with async redis -- same class, no changes
import redis.asyncio as aioredis
batch = RedisBatch(aioredis.Redis())
results = await batch.flush()
```

Key points:

- **`finally_` guarantees cleanup** -- the pipeline is always reset.
- **One class, both clients.** `pipe.execute` is sync or async depending on the client.

---

## Timeout Pattern

Wrap async operations with a timeout.

```python
import asyncio
from quent import Q

async def with_timeout(fn, timeout_seconds):
  """Wrap an async callable with a timeout."""
  async def wrapper(*args, **kwargs):
    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_seconds)
  return wrapper

result = await (
  Q(url)
  .then(with_timeout(fetch, 10.0))
  .then(parse_response)
  .except_(
    lambda ei: {'error': 'timeout', 'url': ei.root_value},
    exceptions=asyncio.TimeoutError,
  )
  .run()
)
```
