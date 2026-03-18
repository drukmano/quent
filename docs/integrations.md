---
title: Framework Integrations
description: Real-world integration recipes for popular Python frameworks and libraries
search:
  boost: 1.5
tags:
  - integrations
  - fastapi
  - django
  - sqlalchemy
  - httpx
  - redis
  - pytest
  - celery
  - pydantic
  - boto3
---

# Framework Integrations

These recipes show how quent integrates with popular Python frameworks and libraries. Each example demonstrates the core value proposition: **write your pipeline logic once, and it works with both sync and async code**.

Every recipe follows the same structure: first, the duplicated sync + async code you would write without quent, then the single unified chain that replaces both.

---

## FastAPI -- Shared Logic Across Async Endpoints and Sync Workers

FastAPI route handlers are async by default, but many applications also run the same business logic in sync contexts -- CLI scripts, Celery workers, or test harnesses.

### Without quent

```python
# Duplicated logic: one sync, one async.

import requests
import aiohttp

def process_order_sync(order_id: str) -> dict:
  data = requests.get(f'https://api.internal/orders/{order_id}').json()
  if data['status'] == 'cancelled':
    raise ValueError('Order is cancelled')
  total = sum(item['price'] * item['qty'] for item in data['items'])
  data['total'] = total
  data['processed'] = True
  requests.post('https://api.internal/notifications', json={
    'order_id': order_id, 'total': total,
  })
  return data

async def process_order_async(order_id: str) -> dict:
  async with aiohttp.ClientSession() as session:
    async with session.get(f'https://api.internal/orders/{order_id}') as resp:
      data = await resp.json()
  if data['status'] == 'cancelled':
    raise ValueError('Order is cancelled')
  total = sum(item['price'] * item['qty'] for item in data['items'])
  data['total'] = total
  data['processed'] = True
  async with aiohttp.ClientSession() as session:
    await session.post('https://api.internal/notifications', json={
      'order_id': order_id, 'total': total,
    })
  return data
```

### With quent

```python
from quent import Chain
from fastapi import FastAPI

app = FastAPI()

def validate_order(data: dict) -> dict:
  if data['status'] == 'cancelled':
    raise ValueError('Order is cancelled')
  return data

def calculate_total(data: dict) -> dict:
  data['total'] = sum(item['price'] * item['qty'] for item in data['items'])
  data['processed'] = True
  return data

def process_order(order_id: str, *, fetch, notify) -> Chain:
  """Build the order processing pipeline.

  `fetch` and `notify` can be sync or async.
  """
  return (
    Chain(order_id)
    .then(fetch)
    .then(validate_order)
    .then(calculate_total)
    .do(lambda data: notify(order_id, data['total']))
    .run()
  )

@app.post('/orders/{order_id}/process')
async def process_order_endpoint(order_id: str):
  return await process_order(
    order_id,
    fetch=async_fetch_order,
    notify=async_send_notification,
  )

@celery_app.task
def process_order_task(order_id: str):
  return process_order(
    order_id,
    fetch=sync_fetch_order,
    notify=sync_send_notification,
  )
```

!!! tip "How it works"
    The `process_order` function builds a single Chain. When called with async callables (from the FastAPI endpoint), the chain returns a coroutine. When called with sync callables (from the Celery task), it returns a value directly. The validation and total calculation steps are plain sync functions that work in both paths.

!!! warning "Chains are not picklable in practice"
    Most chain contents (lambdas, closures, bound methods) naturally fail to pickle. Do not serialize a Chain for Celery task arguments. Build the chain inside the task function, as shown above.

### Using `Depends` for dependency injection

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncSession:
  async with async_session_factory() as session:
    yield session

@app.get('/users/{user_id}/summary')
async def user_summary(user_id: int, db: AsyncSession = Depends(get_db)):
  return await (
    Chain(user_id)
    .then(lambda uid: db.get(User, uid))
    .then(build_summary)
    .except_(lambda ei: {'error': str(ei.exc)}, exceptions=ValueError)
    .run()
  )
```

---

## Django -- Migrating from Sync to Async Views

Django supports both sync and async views side by side. During a migration, the same business logic often needs to live in both.

!!! note "Django async ORM"
    Django provides async variants: `aget()`, `acreate()`, `afirst()`, `acount()`, etc. Methods that return querysets without executing (like `filter()`, `exclude()`, `order_by()`) are safe from both contexts.

### With quent

```python
from django.http import JsonResponse
from quent import Chain

def format_article_response(article, related):
  return {
    'title': article.title,
    'body': article.body,
    'related': [{'id': r.pk, 'title': r.title} for r in related],
  }

def make_article_pipeline(get_article, save_article, list_related):
  """Build the article detail pipeline.

  `get_article`, `save_article`, and `list_related` can be sync or async.
  """
  def pipeline(article_id):
    return (
      Chain(article_id)
      .then(get_article)
      .do(save_article)
      .then(lambda article: (
        Chain(article)
        .then(list_related)
        .then(lambda related: format_article_response(article, related))
        .run()
      ))
      .except_(
        lambda exc: JsonResponse({'error': 'Not found'}, status=404),
        exceptions=Article.DoesNotExist,
      )
      .run()
    )
  return pipeline

# Build both views from the same pipeline
sync_pipeline = make_article_pipeline(
  get_article_sync, save_article_sync, list_related_sync,
)
async_pipeline = make_article_pipeline(
  get_article_async, save_article_async, list_related_async,
)

def article_detail(request, article_id):
  return JsonResponse(sync_pipeline(article_id))

async def article_detail_async(request, article_id):
  return JsonResponse(await async_pipeline(article_id))
```

!!! tip "How it works"
    `make_article_pipeline` builds a Chain agnostic to sync/async. The formatting and exception handling are written once. When Django finishes migrating to async, remove the sync callables and the sync view -- the pipeline definition does not change.

---

## SQLAlchemy -- One Repository for Sync and Async Sessions

SQLAlchemy 2.0 provides both `Session` (sync) and `AsyncSession` (async).

### Without quent

```python
# Two separate repository classes, identical logic, different await keywords.

class UserRepoSync:
  def get_by_id(self, user_id):
    return self.session.get(User, user_id)

class UserRepoAsync:
  async def get_by_id(self, user_id):
    return await self.session.get(User, user_id)
```

### With quent

```python
from sqlalchemy import select
from quent import Chain

class UserRepo:
  """Works with both sync Session and async AsyncSession."""

  def __init__(self, session):
    self.session = session

  def get_by_id(self, user_id: int):
    return Chain(self.session.get, User, user_id).run()

  def create(self, name: str, email: str):
    user = User(name=name, email=email)
    self.session.add(user)
    return (
      Chain(lambda: self.session.commit())
      .then(lambda: user)
      .run()
    )

  def list_active(self):
    stmt = select(User).where(User.is_active == True)
    return (
      Chain(self.session.scalars, stmt)
      .then(lambda result: list(result))
      .run()
    )
```

One class, both sessions:

```python
# Sync usage
with sync_session_factory() as session:
  repo = UserRepo(session)
  user = repo.get_by_id(42)

# Async usage
async with async_session_factory() as session:
  repo = UserRepo(session)
  user = await repo.get_by_id(42)
```

### Using `.with_()` for transaction scoping

```python
def transfer_funds(session, from_id: int, to_id: int, amount: float):
  return (
    Chain(session.begin)
    .with_(lambda txn: (
      Chain(session.get, Account, from_id)
      .then(lambda acc: _debit(acc, amount))
      .do(lambda _: session.flush())
      .then(lambda _: session.get(Account, to_id))
      .then(lambda acc: _credit(acc, amount))
      .run()
    ))
    .run()
  )
```

!!! warning "Session scope"
    Each `Session` / `AsyncSession` instance is not safe for concurrent use. Create one session per thread (sync) or per task (async).

---

## httpx -- One API Client for Sync and Async

httpx provides `httpx.Client` (sync) and `httpx.AsyncClient` (async) with nearly identical APIs.

### With quent

```python
import httpx
from quent import Chain

class ApiClient:
  BASE_URL = 'https://api.example.com/v2'

  def __init__(self, client_factory=httpx.AsyncClient):
    self.client_factory = client_factory

  def _request(self, method: str, path: str, **kwargs):
    return (
      Chain(self.client_factory)
      .with_(lambda client: getattr(client, method)(
        f'{self.BASE_URL}{path}', **kwargs
      ))
      .then(lambda resp: resp.raise_for_status() or resp)
      .then(lambda resp: resp.json())
      .except_(
        lambda exc: {'error': str(exc), 'status': getattr(exc, 'response', None) and exc.response.status_code},
        exceptions=(httpx.HTTPStatusError,),
      )
      .run()
    )

  def get_user(self, user_id: int):
    return self._request('get', f'/users/{user_id}')

  def create_user(self, data: dict):
    return self._request('post', '/users', json=data)
```

```python
# Sync
sync_client = ApiClient(client_factory=httpx.Client)
user = sync_client.get_user(42)

# Async
async_client = ApiClient(client_factory=httpx.AsyncClient)
user = await async_client.get_user(42)
```

!!! tip "How it works"
    `httpx.Client` is a sync context manager. `httpx.AsyncClient` is an async context manager. quent's `.with_()` detects which protocol the object implements and handles it transparently.

### Adding concurrency with `.gather()`

```python
def get_user_with_details(self, user_id: int):
  return (
    Chain(self.client_factory)
    .with_(lambda client: (
      Chain(user_id)
      .gather(
        lambda uid: client.get(f'{self.BASE_URL}/users/{uid}'),
        lambda uid: client.get(f'{self.BASE_URL}/users/{uid}/orders'),
        lambda uid: client.get(f'{self.BASE_URL}/users/{uid}/preferences'),
      )
      .foreach(lambda resp: resp.json())
      .then(lambda results: {
        'user': results[0],
        'orders': results[1],
        'preferences': results[2],
      })
      .run()
    ))
    .run()
  )
```

---

## Redis -- One Caching Layer for Sync and Async

`redis-py` provides `redis.Redis` (sync) and `redis.asyncio.Redis` (async) with the same method names.

### With quent

```python
import json
from quent import Chain

class Cache:
  """Cache-aside layer for both sync and async Redis clients."""

  def __init__(self, r, ttl: int = 300):
    self.r = r
    self.ttl = ttl

  def get_or_fetch(self, key: str, fetch_fn):
    return (
      Chain(key)
      .then(self.r.get)
      .if_().then(lambda cached: json.loads(cached))
      .else_(lambda _: (
        Chain(fetch_fn)
        .do(lambda value: self.r.setex(key, self.ttl, json.dumps(value)))
        .run()
      ))
      .run()
    )

  def invalidate(self, key: str):
    return Chain(self.r.delete, key).run()
```

```python
import redis
import redis.asyncio as aioredis

# Sync
cache = Cache(redis.Redis())
value = cache.get_or_fetch('user:42', lambda: db.get_user(42))

# Async
cache = Cache(aioredis.Redis())
value = await cache.get_or_fetch('user:42', lambda: db.aget_user(42))
```

### Cache-aside with error fallback

```python
def get_or_fetch_resilient(self, key: str, fetch_fn):
  return (
    Chain(key)
    .then(self.r.get)
    .if_().then(lambda cached: json.loads(cached))
    .else_(lambda _: (
      Chain(fetch_fn)
      .do(lambda value: self.r.setex(key, self.ttl, json.dumps(value)))
      .run()
    ))
    .except_(
      lambda exc: fetch_fn(),
      exceptions=(redis.ConnectionError, redis.TimeoutError),
    )
    .run()
  )
```

---

## Celery -- Shared Logic Between Tasks and Async Endpoints

### With quent

```python
from quent import Chain
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

def calculate_invoice(order: dict) -> dict:
  subtotal = sum(item['price'] * item['qty'] for item in order['items'])
  tax = subtotal * order.get('tax_rate', 0.1)
  return {**order, 'subtotal': subtotal, 'tax': tax, 'total': subtotal + tax}

def build_invoice_pipeline(order_id, *, fetch, store, notify):
  return (
    Chain(order_id)
    .then(fetch)
    .then(calculate_invoice)
    .do(store)
    .do(lambda invoice: notify(order_id))
    .run()
  )

@celery_app.task
def generate_invoice_task(order_id: str) -> dict:
  return build_invoice_pipeline(
    order_id,
    fetch=lambda oid: requests.get(f'https://api.internal/orders/{oid}').json(),
    store=lambda inv: requests.post('https://api.internal/invoices', json=inv),
    notify=lambda oid: requests.post('https://api.internal/notifications', json={
      'type': 'invoice_generated', 'order_id': oid,
    }),
  )

@app.post('/orders/{order_id}/invoice')
async def generate_invoice_endpoint(order_id: str) -> dict:
  return await build_invoice_pipeline(
    order_id,
    fetch=async_fetch_order,
    store=async_store_invoice,
    notify=async_send_notification,
  )
```

!!! warning "Chains are not picklable in practice"
    Most chain contents (lambdas, closures, bound methods) naturally fail to pickle. Do not serialize a Chain for Celery task arguments. Build the chain inside the task function, as shown above.

### Reusable pipeline factories with helper functions

```python
def build_invoice_pipeline(order_id, *, fetch, store, notify, log_fn, emit_fn):
  return (
    Chain(order_id)
    .then(fetch)
    .then(calculate_invoice)
    .do(store)
    .do(lambda result: log_fn({'action': 'invoice_generated', 'result_id': result.get('id')}))
    .do(lambda _: emit_fn('invoices.generated'))
    .do(lambda invoice: notify(order_id))
    .run()
  )
```

---

## Pydantic -- Validation in Pipelines

### With quent

```python
from pydantic import BaseModel, ValidationError
from quent import Chain

class UserProfile(BaseModel):
  id: int
  name: str
  email: str
  role: str = 'member'

def normalize_profile(profile: UserProfile) -> dict:
  return {
    'user_id': profile.id,
    'display_name': profile.name.strip().title(),
    'email': profile.email.lower(),
    'role': profile.role,
  }

def ingest_user(fetch_fn):
  return (
    Chain(fetch_fn)
    .then(UserProfile.model_validate)
    .then(normalize_profile)
    .except_(
      lambda ei: {'error': 'validation_failed', 'details': ei.exc.errors()},
      exceptions=ValidationError,
    )
    .run()
  )
```

```python
# Sync
result = ingest_user(lambda: json.load(open('user.json')))

# Async
result = await ingest_user(async_fetch_user_profile)
```

### Multi-model validation with `.gather()`

```python
from pydantic import BaseModel, ValidationError
from quent import Chain

class Address(BaseModel):
  street: str
  city: str
  country: str

class PaymentMethod(BaseModel):
  type: str
  last_four: str

class OrderRequest(BaseModel):
  product_id: int
  quantity: int

def validate_checkout(payload: dict):
  return (
    Chain(payload)
    .gather(
      lambda p: Address.model_validate(p['shipping_address']),
      lambda p: PaymentMethod.model_validate(p['payment']),
      lambda p: OrderRequest.model_validate(p['order']),
    )
    .then(lambda r: {
      'shipping': r[0].model_dump(),
      'payment': r[1].model_dump(),
      'order': r[2].model_dump(),
      'valid': True,
    })
    .except_(
      lambda ei: {'valid': False, 'error': str(ei.exc)},
      exceptions=ValidationError,
    )
    .run()
  )
```

---

## pytest -- Testing Chains with Sync and Async Callables

### With quent

```python
import unittest
from unittest.mock import AsyncMock, MagicMock
from quent import Chain

def build_pipeline(fetch, transform, save, item_id):
  return (
    Chain(item_id)
    .then(fetch)
    .then(transform)
    .do(save)
    .run()
  )

class TestPipeline(unittest.IsolatedAsyncioTestCase):

  def _make_mocks(self, *, async_: bool):
    Mock = AsyncMock if async_ else MagicMock
    fetch = Mock(return_value={'id': 1, 'raw': 'data'})
    transform = Mock(return_value={'id': 1, 'processed': True})
    save = Mock()
    return fetch, transform, save

  async def _run_pipeline(self, fetch, transform, save, item_id=1):
    result = build_pipeline(fetch, transform, save, item_id)
    if hasattr(result, '__await__'):
      result = await result
    return result

  async def test_sync_pipeline(self):
    fetch, transform, save = self._make_mocks(async_=False)
    result = await self._run_pipeline(fetch, transform, save)

    self.assertEqual(result, {'id': 1, 'processed': True})
    fetch.assert_called_once_with(1)
    transform.assert_called_once_with({'id': 1, 'raw': 'data'})
    save.assert_called_once_with({'id': 1, 'processed': True})

  async def test_async_pipeline(self):
    fetch, transform, save = self._make_mocks(async_=True)
    result = await self._run_pipeline(fetch, transform, save)

    self.assertEqual(result, {'id': 1, 'processed': True})
    fetch.assert_called_once_with(1)

  async def test_error_handler(self):
    fetch = MagicMock(side_effect=ValueError('bad'))
    result = (
      Chain(1)
      .then(fetch)
      .except_(lambda ei: {'error': str(ei.exc), 'input': ei.root_value})
      .run()
    )
    self.assertEqual(result, {'error': 'bad', 'input': 1})
```

!!! tip "How it works"
    `build_pipeline` defines the chain once. With `MagicMock` (sync), `.run()` returns a plain value. With `AsyncMock`, it returns a coroutine. The assertions are identical -- only the mock factory changes.

### Observing intermediate values

```python
async def test_intermediate_values(self):
  seen = []
  fetch = MagicMock(return_value={'id': 1, 'status': 'raw'})
  transform = MagicMock(return_value={'id': 1, 'status': 'processed'})

  result = (
    Chain(1)
    .then(fetch)
    .do(lambda v: seen.append(('after_fetch', v)))
    .then(transform)
    .do(lambda v: seen.append(('after_transform', v)))
    .run()
  )

  self.assertEqual(seen, [
    ('after_fetch', {'id': 1, 'status': 'raw'}),
    ('after_transform', {'id': 1, 'status': 'processed'}),
  ])
```

!!! note "pytest-asyncio users"
    Replace `IsolatedAsyncioTestCase` with `@pytest.mark.asyncio` on each async test function. The chain-testing patterns are identical.

---

## boto3/aiobotocore -- One AWS Service Layer

### With quent

```python
import json
from quent import Chain

class S3Service:
  """Unified S3 service for boto3 (sync) or aiobotocore (async) clients."""

  def __init__(self, client):
    self.client = client

  def get_json(self, bucket: str, key: str):
    return (
      Chain(self.client.get_object, Bucket=bucket, Key=key)
      .then(lambda resp: resp['Body'].read())
      .then(json.loads)
      .run()
    )

  def put_json(self, bucket: str, key: str, data: dict):
    return (
      Chain(self.client.put_object, Bucket=bucket, Key=key,
            Body=json.dumps(data), ContentType='application/json')
      .run()
    )

  def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str):
    return (
      Chain(self.client.copy_object,
            CopySource={'Bucket': src_bucket, 'Key': src_key},
            Bucket=dst_bucket, Key=dst_key)
      .run()
    )
```

```python
import boto3
import aiobotocore.session

# Sync (Lambda, CLI)
s3 = S3Service(boto3.client('s3'))
data = s3.get_json('my-bucket', 'config.json')

# Async (web server)
aio_session = aiobotocore.session.get_session()
async with aio_session.create_client('s3') as async_client:
  s3 = S3Service(async_client)
  data = await s3.get_json('my-bucket', 'config.json')
```

!!! tip "How it works"
    `boto3` methods return dicts synchronously. `aiobotocore` methods return coroutines. quent detects the awaitable return and transitions accordingly. Similarly, `resp['Body'].read()` is sync for boto3 and async for aiobotocore.

### Async client lifecycle with `.with_()`

```python
import aiobotocore.session
from quent import Chain

aio_session = aiobotocore.session.get_session()

def get_json_oneshot(bucket: str, key: str):
  return (
    Chain(aio_session.create_client, 's3')
    .with_(lambda client: (
      Chain(client.get_object, Bucket=bucket, Key=key)
      .then(lambda resp: resp['Body'].read())
      .then(json.loads)
      .run()
    ))
    .run()
  )

data = await get_json_oneshot('my-bucket', 'config.json')
```

---

## Flask -- Request Handling Pipelines

### With quent

```python
from flask import Flask, request, jsonify
from quent import Chain

app = Flask(__name__)

def validate_payload(data):
  if not data.get('name'):
    return Chain.return_(jsonify({'error': 'name required'}), 400)
  return data

def process_and_save(data):
  data['name'] = data['name'].strip().title()
  db.save(data)
  return data

@app.post('/users')
def create_user():
  return (
    Chain(request.get_json)
    .then(validate_payload)
    .then(process_and_save)
    .then(lambda data: jsonify(data))
    .except_(
      lambda ei: (jsonify({'error': str(ei.exc)}), 500),
    )
    .run()
  )
```

---

## click -- CLI Pipelines

### With quent

```python
import click
from quent import Chain

@click.command()
@click.argument('input_file')
@click.option('--output', '-o', default='output.json')
@click.option('--format', 'fmt', type=click.Choice(['json', 'csv']), default='json')
def process(input_file, output, fmt):
  """Process a data file through a transformation pipeline."""
  result = (
    Chain(input_file)
    .then(read_file)
    .then(lambda records: [r for r in records if is_valid(r)])
    .foreach(normalize)
    .then(format_output if fmt == 'json' else format_csv)
    .do(lambda data: write_file(output, data))
    .except_(lambda ei: click.echo(f'Error: {ei.exc}', err=True) or exit(1))
    .run()
  )
  click.echo(f'Processed {len(result)} records to {output}')
```
