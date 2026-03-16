"""
ETL Pipeline Example -- quent sync/async bridge.

Demonstrates a complete Extract-Transform-Load pipeline using quent's
collection operations (.foreach, .foreach_do), pipeline composition
(.clone), and error handling (.except_, .finally_).

Patterns shown:
  - Chain as the pipeline backbone
  - Multiple simulated data sources (CSV-like, JSON-like, API-like)
  - .then() for transforms, .do() for logging side effects
  - .foreach() for per-record transformation
  - .then() with a list comprehension for validation/filtering
  - .except_() for error recovery
  - .finally_() for cleanup
  - .clone() for forking a base pipeline to different destinations
  - Both sync and async variants showing the bridge contract

Run with:
    python examples/etl_pipeline.py
"""

from __future__ import annotations

import asyncio
import datetime

from quent import Chain

# ---------------------------------------------------------------------------
# Simulated source data
# ---------------------------------------------------------------------------

USERS_CSV: list[dict] = [
  {'id': 1, 'name': '  Alice Smith ', 'email': 'Alice.Smith@Example.COM', 'age': 34, 'status': 'active'},
  {'id': 2, 'name': 'bob jones', 'email': 'bob@example.com', 'age': -5, 'status': 'active'},
  {'id': 3, 'name': '  Carol White  ', 'email': 'carol@EXAMPLE.COM', 'age': 27, 'status': 'deleted'},
  {'id': 4, 'name': 'dave', 'email': 'dave@example.com', 'age': 200, 'status': 'active'},
  {'id': 5, 'name': '  Eve Torres', 'email': 'eve@EXAMPLE.COM', 'age': 41, 'status': 'active'},
  {'id': 6, 'email': 'ghost@example.com', 'age': 30, 'status': 'active'},  # missing 'name'
  {'id': 7, 'name': 'Frank Hill', 'email': 'FRANK@Example.COM', 'age': 55, 'status': 'inactive'},
  {'id': 8, 'name': 'Grace Lee', 'age': 23, 'status': 'active'},  # missing 'email'
  {'id': 9, 'name': 'henry brown', 'email': 'henry@example.com', 'age': 119, 'status': 'active'},
  {'id': 10, 'name': '  Iris Kim  ', 'email': 'iris@example.com', 'age': 31, 'status': 'active'},
]

ORDERS_JSON: list[dict] = [
  {'order_id': 101, 'user_id': 1, 'amount': 49.99, 'currency': 'USD'},
  {'order_id': 102, 'user_id': 2, 'amount': -10.00, 'currency': 'USD'},  # invalid amount
  {'order_id': 103, 'user_id': 5, 'amount': 125.50, 'currency': 'EUR'},
  {'order_id': 104, 'user_id': 99, 'amount': 30.00, 'currency': 'USD'},  # unknown user
  {'order_id': 105, 'user_id': 10, 'amount': 75.25, 'currency': 'GBP'},
]

# Module-level "database" destinations used by the load step.
loaded_users: list[dict] = []
loaded_users_b: list[dict] = []
loaded_orders: list[dict] = []


# ---------------------------------------------------------------------------
# Extract functions
# ---------------------------------------------------------------------------

def extract_users(source: str) -> list[dict]:
  """Simulate reading user records from a source."""
  print(f'  [extract] reading users from {source!r} -> {len(USERS_CSV)} raw records')
  return list(USERS_CSV)


async def async_extract_users(source: str) -> list[dict]:
  """Async version of extract_users -- demonstrates bridge contract."""
  await asyncio.sleep(0)  # simulate async I/O
  print(f'  [extract-async] reading users from {source!r} -> {len(USERS_CSV)} raw records')
  return list(USERS_CSV)


def extract_orders(source: str) -> list[dict]:
  """Simulate reading order records from a source."""
  print(f'  [extract] reading orders from {source!r} -> {len(ORDERS_JSON)} raw records')
  return list(ORDERS_JSON)


# ---------------------------------------------------------------------------
# Validation / filtering
# ---------------------------------------------------------------------------

def is_valid_user(record: dict) -> bool:
  """Return True when the user record has required fields and is not deleted."""
  return (
    'id' in record
    and 'name' in record
    and 'email' in record
    and record.get('status') != 'deleted'
  )


def is_valid_order(order: dict) -> bool:
  """Return True when the order has a positive amount."""
  return order.get('amount', 0) > 0


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------

def normalize_user(record: dict) -> dict:
  """Return a cleaned copy of a user record."""
  raw_age = record.get('age', 0)
  return {
    **record,
    'name': record['name'].strip().title(),
    'email': record['email'].strip().lower(),
    'age': max(0, min(120, raw_age)),
    'processed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }


def enrich_user(record: dict) -> dict:
  """Add computed fields derived from existing data."""
  domain = record['email'].split('@')[-1] if '@' in record['email'] else ''
  return {**record, 'email_domain': domain}


async def async_normalize_user(record: dict) -> dict:
  """Async version -- bridge contract: same logic, different sync/async kind."""
  await asyncio.sleep(0)
  raw_age = record.get('age', 0)
  return {
    **record,
    'name': record['name'].strip().title(),
    'email': record['email'].strip().lower(),
    'age': max(0, min(120, raw_age)),
    'processed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }


def normalize_order(order: dict) -> dict:
  """Normalize currency field to uppercase."""
  return {**order, 'currency': order.get('currency', 'USD').upper()}


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_users(records: list[dict]) -> int:
  """Append records to the loaded_users store; return the count loaded."""
  loaded_users.extend(records)
  return len(records)


def load_users_b(records: list[dict]) -> int:
  """Alternate destination -- appends to loaded_users_b."""
  loaded_users_b.extend(records)
  return len(records)


def load_orders(records: list[dict]) -> int:
  """Append order records to the loaded_orders store."""
  loaded_orders.extend(records)
  return len(records)


# ---------------------------------------------------------------------------
# Logging / side-effect helpers (for .do())
# ---------------------------------------------------------------------------

def log_count(records: list[dict]) -> None:
  """Print the current record count (for use with .do())."""
  print(f'  [pipeline] {len(records)} record(s) in flight')


def log_loaded(count: int) -> None:
  """Print how many records were loaded."""
  print(f'  [load] {count} record(s) loaded successfully')


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def handle_etl_error(ei) -> int:
  """Log the error and return 0 as a safe fallback result.

  Registered with except_(handler) -- handler receives ChainExcInfo(exc, root_value)
  as the current value (standard 2-rule calling convention).
  """
  print(f'  [error] ETL failed: {type(ei.exc).__name__}: {ei.exc}')
  return 0


def print_summary() -> None:
  """Always-run cleanup that prints a completion notice.

  Registered with finally_(lambda _: print_summary()) -- the lambda
  accepts the root value and discards it, calling print_summary() with
  zero arguments.
  """
  print('  [finally] pipeline finished')


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':

  # -----------------------------------------------------------------------
  # (a) Basic ETL pipeline
  # -----------------------------------------------------------------------
  print('=' * 60)
  print('(a) Basic ETL pipeline')
  print('=' * 60)

  count = (
    Chain(extract_users, 'users_2024.csv')
    .do(log_count)                                                            # side-effect: log raw count, value unchanged
    .then(lambda records: [r for r in records if is_valid_user(r)])          # keep only valid records
    .do(log_count)                                                            # side-effect: log filtered count
    .foreach(normalize_user)                                                  # transform each record
    .foreach(enrich_user)                                                     # add computed fields
    .then(load_users)                                                         # load into destination; returns count
    .run()
  )
  print(f'  Loaded {count} record(s) into primary store.')
  print(f'  loaded_users now has {len(loaded_users)} entry/entries.\n')

  # -----------------------------------------------------------------------
  # (b) Reusable pipeline with .clone()
  # -----------------------------------------------------------------------
  print('=' * 60)
  print('(b) Reusable base pipeline cloned into two destinations')
  print('=' * 60)

  # Reset destinations for a clean demo.
  loaded_users.clear()
  loaded_users_b.clear()

  # Build a shared transform pipeline (no load step yet).
  base_transform = (
    Chain(extract_users, 'users_2024.csv')
    .then(lambda records: [r for r in records if is_valid_user(r)])
    .foreach(normalize_user)
    .foreach(enrich_user)
  )

  # Clone once per destination -- each clone is fully independent.
  pipeline_a = base_transform.clone().then(load_users)
  pipeline_b = base_transform.clone().then(load_users_b)

  count_a = pipeline_a.run()
  count_b = pipeline_b.run()

  print(f'  Database A received {count_a} record(s). store size = {len(loaded_users)}')
  print(f'  Database B received {count_b} record(s). store size = {len(loaded_users_b)}\n')

  # -----------------------------------------------------------------------
  # (c) Pipeline with error handling (.except_ and .finally_)
  # -----------------------------------------------------------------------
  print('=' * 60)
  print('(c) Error-handling pipeline (bad record injected)')
  print('=' * 60)

  # Inject a record that will crash normalize_user(): name is None, so
  # str.strip() raises AttributeError.
  bad_records = [
    {'id': 99, 'name': None, 'email': 'bad@example.com', 'age': 25, 'status': 'active'},
  ]

  def extract_bad_data(source: str) -> list[dict]:
    return bad_records

  result = (
    Chain(extract_bad_data, 'corrupt_feed.csv')
    .then(lambda records: [r for r in records if is_valid_user(r)])
    .foreach(normalize_user)          # will raise AttributeError on None name
    .foreach(enrich_user)
    .then(load_users)
    .except_(handle_etl_error)   # Ellipsis: handler(exc) only
    .finally_(lambda _: print_summary())     # Ellipsis: called with zero args
    .run()
  )
  print(f'  Pipeline result after error recovery: {result}\n')

  # -----------------------------------------------------------------------
  # (d) Async variant -- same pipeline, async callables
  # -----------------------------------------------------------------------
  print('=' * 60)
  print('(d) Async ETL pipeline (bridge contract demo)')
  print('=' * 60)

  loaded_users.clear()

  # Replace sync extract and normalize with async equivalents.
  # The pipeline structure is identical -- quent handles the bridging.
  async_count = (
    Chain(async_extract_users, 'users_2024.csv')
    .do(log_count)
    .then(lambda records: [r for r in records if is_valid_user(r)])
    .do(log_count)
    .foreach(async_normalize_user)    # async map function
    .foreach(enrich_user)             # sync -- mixed sync/async in one pipeline
    .then(load_users)
    .run()
  )
  async_count = asyncio.run(async_count)
  print(f'  Async pipeline loaded {async_count} record(s).')
  print(f'  loaded_users now has {len(loaded_users)} entry/entries.\n')

  # -----------------------------------------------------------------------
  # (e) Multi-source ETL: orders pipeline
  # -----------------------------------------------------------------------
  print('=' * 60)
  print('(e) Multi-source ETL: orders pipeline')
  print('=' * 60)

  loaded_orders.clear()

  order_count = (
    Chain(extract_orders, 'orders_q4.json')
    .do(log_count)
    .then(lambda orders: [o for o in orders if is_valid_order(o)])
    .do(log_count)
    .foreach(normalize_order)
    .then(load_orders)
    .except_(handle_etl_error)
    .finally_(lambda _: print_summary())
    .run()
  )
  print(f'  Loaded {order_count} order(s).')

  # Show a sample record to confirm the pipeline worked.
  if loaded_users:
    print('\n  Sample enriched user record:')
    sample = loaded_users[0]
    for key, value in sample.items():
      print(f'    {key}: {value!r}')

  if loaded_orders:
    print('\n  Sample normalized order record:')
    sample = loaded_orders[0]
    for key, value in sample.items():
      print(f'    {key}: {value!r}')
