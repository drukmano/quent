---
title: "Cookbook — Real-World Examples"
description: "Real-world quent examples grouped by domain: data processing, API orchestration, database operations, file processing, configuration, and testing."
tags:
  - cookbook
  - examples
  - runnable
  - recipes
search:
  boost: 5
---

# Cookbook

Real-world examples grouped by domain. Each example shows how quent solves a concrete problem, with code you can adapt to your own projects.

!!! tip "Looking for quick copy-paste snippets?"
    See [Recipes](recipes.md) for shorter, pattern-focused examples.

---

## Data Processing

### CSV to JSON Transform

Read a CSV file, filter rows, normalize fields, and output JSON.

```python
import csv
import json
from quent import Q

def read_csv(path):
  with open(path) as f:
    return list(csv.DictReader(f))

def normalize_record(record):
  return {
    'id': int(record['id']),
    'name': record['name'].strip().title(),
    'email': record['email'].strip().lower(),
    'active': record.get('status', '') == 'active',
  }

result = (
  Q('users.csv')
  .then(read_csv)
  .then(lambda records: [r for r in records if r.get('email')])  # drop rows without email
  .foreach(normalize_record)
  .then(lambda records: json.dumps(records, indent=2))
  .run()
)
```

### Aggregation Pipeline

Group, aggregate, and sort data in a single pipeline.

```python
from collections import Counter
from quent import Q

def group_by_category(items):
  groups = {}
  for item in items:
    groups.setdefault(item['category'], []).append(item)
  return groups

def aggregate(groups):
  return {
    cat: {'count': len(items), 'total': sum(i['price'] for i in items)}
    for cat, items in groups.items()
  }

summary = (
  Q(fetch_sales_data)
  .then(lambda items: [item for item in items if item['price'] > 0])
  .then(group_by_category)
  .then(aggregate)
  .then(lambda agg: sorted(agg.items(), key=lambda x: x[1]['total'], reverse=True))
  .run()
)
```

---

## API Orchestration

### Multiple Endpoint Calls with Error Recovery

Fetch data from several APIs, merge results, and handle partial failures.

```python
from quent import Q

def fetch_user_data(user_id):
  return (
    Q(user_id)
    .gather(
      lambda uid: api.get(f'/users/{uid}'),
      lambda uid: api.get(f'/users/{uid}/orders'),
      lambda uid: api.get(f'/users/{uid}/preferences'),
      concurrency=2,
    )
    .then(lambda r: {
      'user': r[0],
      'orders': r[1],
      'preferences': r[2],
    })
    .except_(
      lambda ei: {'error': str(ei.exc), 'user_id': ei.root_value},
    )
    .run()
  )

# Sync
data = fetch_user_data(42)

# Async (if api.get is async)
data = await fetch_user_data(42)
```

### Request Pipeline with Middleware

Build a reusable request pipeline with composable middleware.

```python
from quent import Q

def add_headers(req, headers):
  return {**req, 'headers': {**req.get('headers', {}), **headers}}

def add_auth(req, token):
  return {
    **req,
    'headers': {**req.get('headers', {}), 'Authorization': f'Bearer {token}'},
  }

result = (
  Q({'method': 'GET', 'url': '/api/users'})
  .then(add_headers, {'Accept': 'application/json'})
  .then(add_auth, auth_token)
  .do(lambda req: logger.info('Request: %s %s', req['method'], req['url']))
  .then(send_request)
  .then(lambda resp: resp.json())
  .run()
)
```

---

## Database Operations

### Connection Management with with\_()

Use `.with_()` to manage database connections as context managers.

```python
from quent import Q

def query_users(db):
  return db.execute('SELECT * FROM users WHERE active = 1').fetchall()

def format_users(rows):
  return [{'id': r[0], 'name': r[1], 'email': r[2]} for r in rows]

# sqlite3 (sync)
import sqlite3

users = (
  Q(lambda: sqlite3.connect('app.db'))
  .with_(lambda db: (
    Q(db)
    .then(query_users)
    .then(format_users)
    .run()
  ))
  .run()
)

# aiosqlite (async) -- same pipeline structure
import aiosqlite

users = await (
  Q(aiosqlite.connect, 'app.db')
  .with_(lambda db: (
    Q(db)
    .then(query_users)
    .then(format_users)
    .run()
  ))
  .run()
)
```

---

## File Processing

### Read, Transform, Write Pipeline

Process files in a pipeline with guaranteed cleanup.

```python
from quent import Q

def transform_lines(text):
  lines = text.strip().splitlines()
  return '\n'.join(
    line.upper()
    for line in lines
    if line.strip() and not line.startswith('#')
  )

def process_file(input_path, output_path):
  return (
    Q(input_path)
    .then(open)
    .with_(lambda f: f.read())
    .then(transform_lines)
    .do(lambda content: (
      Q(output_path)
      .then(lambda p: open(p, 'w'))
      .with_(lambda f: f.write(content))
      .run()
    ))
    .run()
  )
```

---

## Configuration Pipelines

### Load, Validate, Merge

Load configuration from multiple sources, validate, and merge into a final config.

```python
import json
import os
from quent import Q

def load_file(path):
  with open(path) as f:
    return json.load(f)

def load_env_overrides(config):
  overrides = {}
  for key in config:
    env_key = f'APP_{key.upper()}'
    if env_key in os.environ:
      overrides[key] = os.environ[env_key]
  return {**config, **overrides}

def validate_config(config):
  required = ['database_url', 'secret_key']
  missing = [k for k in required if k not in config]
  if missing:
    return Q.return_({'error': f'Missing keys: {missing}'})
  return config

config = (
  Q('config.json')
  .then(load_file)
  .then(load_env_overrides)
  .then(validate_config)
  .except_(lambda ei: {'error': str(ei.exc)})
  .run()
)
```

---

## Testing Patterns

### Unit Testing Pipelines with Mocks

Q pipelines work naturally with `unittest.mock` -- swap callables with mocks to test in isolation.

```python
import unittest
from unittest.mock import AsyncMock, MagicMock
from quent import Q

def build_pipeline(fetch, transform, save):
  return (
    Q()
    .then(fetch)
    .then(transform)
    .do(save)
  )

class TestPipeline(unittest.IsolatedAsyncioTestCase):

  async def test_sync_pipeline(self):
    fetch = MagicMock(return_value={'id': 1, 'raw': 'data'})
    transform = MagicMock(return_value={'id': 1, 'processed': True})
    save = MagicMock()

    pipeline = build_pipeline(fetch, transform, save)
    result = pipeline.run(42)

    self.assertEqual(result, {'id': 1, 'processed': True})
    fetch.assert_called_once_with(42)
    save.assert_called_once()

  async def test_async_pipeline(self):
    fetch = AsyncMock(return_value={'id': 1, 'raw': 'data'})
    transform = AsyncMock(return_value={'id': 1, 'processed': True})
    save = AsyncMock()

    pipeline = build_pipeline(fetch, transform, save)
    result = await pipeline.run(42)

    self.assertEqual(result, {'id': 1, 'processed': True})
    fetch.assert_called_once_with(42)
    save.assert_called_once()

  async def test_error_handling(self):
    fetch = MagicMock(side_effect=ValueError('bad input'))

    result = (
      Q(1)
      .then(fetch)
      .except_(lambda ei: {'error': str(ei.exc), 'input': ei.root_value})
      .run()
    )

    self.assertEqual(result, {'error': 'bad input', 'input': 1})
```

Key points:

- **Same pipeline, both mock types.** `MagicMock` for sync, `AsyncMock` for async -- the pipeline handles either transparently.
- **`.do()` observers let you inspect intermediate values** without altering the pipeline flow.
- **`except_()` handlers are testable** by making mocks raise.
