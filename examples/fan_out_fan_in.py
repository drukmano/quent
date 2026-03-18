"""
Fan-out / fan-in pattern with quent.

Fan-out splits a single input across multiple independent processors that run
in parallel (or concurrently). Fan-in collects every branch result and merges
them back into a single value.

quent expresses this naturally:

  Q(user_id)
    .gather(get_profile, get_orders, get_prefs)   # fan-out -> tuple of 3 results
    .then(lambda r: merge(r[0], r[1], r[2]))       # fan-in -> single merged dict

The same pipeline definition works for both sync and async callables -- quent
detects awaitables automatically and switches to async execution transparently.

Patterns shown:
  - gather() for parallel execution
  - Multiple processing branches with different logic
  - Nested pipelines for branch-specific error handling
  - clone() for creating branch variations
  - concurrency= for controlling parallelism
  - Both CPU-bound (ThreadPoolExecutor) and IO-bound (async) examples

Run with:
    python examples/fan_out_fan_in.py
"""

from __future__ import annotations

import asyncio
import json
import time

from quent import Q

# ---------------------------------------------------------------------------
# Simulated sync services (each costs ~50 ms to simulate network latency)
# ---------------------------------------------------------------------------

def get_user_profile(user_id: int) -> dict:
  time.sleep(0.05)
  return {
    'id': user_id,
    'name': f'User {user_id}',
    'email': f'user{user_id}@example.com',
  }


def get_user_orders(user_id: int) -> list[dict]:
  time.sleep(0.05)
  return [
    {'order_id': user_id * 100 + i, 'item': f'item_{i}', 'amount': round(9.99 * i, 2)}
    for i in range(1, 5)
  ]


def get_user_preferences(user_id: int) -> dict:
  time.sleep(0.05)
  return {
    'theme': 'dark' if user_id % 2 == 0 else 'light',
    'notifications': True,
    'language': 'en',
  }


def compute_recommendations(user_id: int) -> list[str]:
  time.sleep(0.05)
  return [f'rec_{user_id}_{i}' for i in range(1, 4)]


# ---------------------------------------------------------------------------
# Async counterparts (same logic, asyncio.sleep instead of time.sleep)
# ---------------------------------------------------------------------------

async def async_get_user_profile(user_id: int) -> dict:
  await asyncio.sleep(0.05)
  return {'id': user_id, 'name': f'User {user_id}', 'email': f'user{user_id}@example.com'}


async def async_get_user_orders(user_id: int) -> list[dict]:
  await asyncio.sleep(0.05)
  return [
    {'order_id': user_id * 100 + i, 'item': f'item_{i}', 'amount': round(9.99 * i, 2)}
    for i in range(1, 5)
  ]


async def async_get_user_preferences(user_id: int) -> dict:
  await asyncio.sleep(0.05)
  return {'theme': 'dark' if user_id % 2 == 0 else 'light', 'notifications': True, 'language': 'en'}


async def async_compute_recommendations(user_id: int) -> list[str]:
  await asyncio.sleep(0.05)
  return [f'rec_{user_id}_{i}' for i in range(1, 4)]


# ---------------------------------------------------------------------------
# Merge / reduce helpers
# ---------------------------------------------------------------------------

def merge_user_data(profile: dict, orders: list[dict], prefs: dict) -> dict:
  """Combine three branch results into a single enriched user dict."""
  return {**profile, 'orders': orders, 'preferences': prefs}


def merge_with_stats(
  profile: dict,
  orders: list[dict],
  prefs: dict,
  recs: list[str],
) -> dict:
  """Fan-in with per-branch processing: filter low-value orders, count recs."""
  high_value_orders = [o for o in orders if o['amount'] >= 10.0]
  return {
    **profile,
    'high_value_orders': high_value_orders,
    'order_count': len(orders),
    'preferences': prefs,
    'recommendation_count': len(recs),
    'recommendations': recs,
  }


def enrich_order(order: dict) -> dict:
  """Add a computed field to a single order (used in nested fan-out demo)."""
  return {**order, 'discounted_amount': round(order['amount'] * 0.9, 2)}


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
  print(f'\n{"=" * 60}')
  print(f'  {title}')
  print('=' * 60)


def _show(label: str, value: object) -> None:
  print(f'\n{label}:')
  print(json.dumps(value, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  USER_ID = 42

  # ------------------------------------------------------------------
  # a) Basic fan-out / fan-in (sync)
  # ------------------------------------------------------------------
  _header('a) Basic fan-out / fan-in (sync)')

  t0 = time.perf_counter()
  result = (
    Q(USER_ID)
    # gather() fans out to 3 services concurrently via ThreadPoolExecutor.
    # Returns a tuple: (profile, orders, prefs).
    .gather(get_user_profile, get_user_orders, get_user_preferences)
    .then(lambda r: merge_user_data(r[0], r[1], r[2]))
    .run()
  )
  elapsed = time.perf_counter() - t0

  _show('Merged user data', result)
  print(f'\n(completed in {elapsed:.3f}s -- 3 services ran concurrently via ThreadPoolExecutor)')

  # ------------------------------------------------------------------
  # b) Concurrent fan-out with concurrency limit (async)
  # ------------------------------------------------------------------
  _header('b) Concurrent fan-out with concurrency=2 (async)')

  t0 = time.perf_counter()
  result = asyncio.run(
    Q(USER_ID)
    .gather(
      async_get_user_profile,
      async_get_user_orders,
      async_get_user_preferences,
      concurrency=2,  # at most 2 coroutines running simultaneously
    )
    .then(lambda r: merge_user_data(r[0], r[1], r[2]))
    .run()
  )
  elapsed = time.perf_counter() - t0

  _show('Merged user data (async, concurrency=2)', result)
  print(f'\n(completed in {elapsed:.3f}s -- 2 concurrent then 1, so ~0.10 s total)')

  # ------------------------------------------------------------------
  # c) Fan-out + per-branch processing (4 branches, async, unlimited)
  # ------------------------------------------------------------------
  _header('c) Fan-out + per-branch processing (4 async services, unlimited)')

  t0 = time.perf_counter()
  result = asyncio.run(
    Q(USER_ID)
    .gather(
      async_get_user_profile,
      async_get_user_orders,
      async_get_user_preferences,
      async_compute_recommendations,
    )
    .then(lambda r: merge_with_stats(r[0], r[1], r[2], r[3]))
    .run()
  )
  elapsed = time.perf_counter() - t0

  _show('Enriched user data with stats', result)
  print(f'\n(completed in {elapsed:.3f}s -- all 4 services ran concurrently, ~0.05 s total)')

  # ------------------------------------------------------------------
  # d) Nested fan-out: fetch orders, then enrich each order via .foreach()
  # ------------------------------------------------------------------
  _header('d) Nested fan-out: gather -> map each order')

  result = (
    Q(USER_ID)
    # Level 1: fan-out to 3 services.
    .gather(get_user_profile, get_user_orders, get_user_preferences)
    .then(lambda r: merge_user_data(r[0], r[1], r[2]))          # fan-in -> enriched dict
    # Level 2: fan-out over the orders list.
    .then(lambda d: d['orders'])    # extract orders
    .foreach(enrich_order)              # enrich each order independently
    .run()
  )

  _show('Enriched orders (each with 10% discount applied)', result)

  # ------------------------------------------------------------------
  # e) Clone: base pipeline + branch-specific extensions
  # ------------------------------------------------------------------
  _header('e) clone() for branch variations')

  # Build a base fan-out pipeline that gathers profile + orders.
  base_q = (
    Q()
    .gather(get_user_profile, get_user_orders)
  )

  # Clone A: merge into a summary dict.
  summary_q = (
    base_q.clone()
    .then(lambda r: {
      'name': r[0]['name'],
      'total_orders': len(r[1]),
      'total_amount': sum(o['amount'] for o in r[1]),
    })
  )

  # Clone B: extract only the order list.
  orders_only_q = (
    base_q.clone()
    .then(lambda tup: tup[1])  # second element of the gather tuple
  )

  summary = summary_q.run(USER_ID)
  orders = orders_only_q.run(USER_ID)

  _show('Summary (clone A)', summary)
  _show('Orders only (clone B)', orders)

  # ------------------------------------------------------------------
  # f) Error handling per branch via nested pipelines
  # ------------------------------------------------------------------
  _header('f) Per-branch error handling with nested pipelines')

  def flaky_service(user_id: int) -> dict:
    """A service that always fails -- demonstrates per-branch recovery."""
    raise ConnectionError(f'Service unavailable for user {user_id}')

  # Wrap the flaky service in a nested pipeline with its own except_().
  # The outer pipeline continues even if this branch fails.
  safe_flaky = (
    Q()
    .then(flaky_service)
    .except_(lambda ei: {'error': str(ei.exc)})
  )

  result = (
    Q(USER_ID)
    .gather(
      get_user_profile,
      lambda uid: safe_flaky.run(uid),  # nested pipeline handles its own errors
      get_user_preferences,
    )
    .then(lambda r: {
      'profile': r[0],
      'flaky': r[1],  # contains {'error': '...'} instead of crashing
      'preferences': r[2],
    })
    .run()
  )

  _show('Result with per-branch error recovery', result)
