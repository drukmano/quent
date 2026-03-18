"""
API Gateway Example -- quent sync/async bridge.

Demonstrates how to compose authentication, rate limiting, routing,
concurrent service calls, response transformation, and error handling
into quent pipelines.

Patterns shown:
  - gather() for concurrent requests to multiple services
  - concurrency= parameter to limit parallel requests
  - except_() for centralised error handling
  - Nested pipelines for per-service error handling
  - if_/else_ for conditional routing
  - do() for request/response logging
  - Q.return_() for early rejection (auth/rate-limit failures)
  - Simulated HTTP calls (sync and async variants)
  - Same pipeline works sync and async

Run with:
    python examples/api_gateway.py
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from quent import Q, QuentExcInfo

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class Request:
  method: str
  path: str
  headers: dict
  body: dict | None
  client_ip: str


@dataclass
class Response:
  status: int
  body: dict
  headers: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulated infrastructure
# ---------------------------------------------------------------------------

TOKEN_DB: dict[str, dict] = {
  'token-alice': {'user_id': 1, 'name': 'Alice', 'role': 'admin'},
  'token-bob': {'user_id': 2, 'name': 'Bob', 'role': 'reader'},
}

USERS_DB: dict[int, dict] = {
  1: {'id': 1, 'name': 'Alice', 'role': 'admin'},
  2: {'id': 2, 'name': 'Bob', 'role': 'reader'},
}

ORDERS_DB: dict[int, list[dict]] = {
  1: [{'order_id': 101, 'item': 'Widget', 'amount': 49.99}],
  2: [{'order_id': 201, 'item': 'Gadget', 'amount': 25.00}],
}

RATE_LIMITS: dict[str, int] = {}
RATE_LIMIT_MAX = 5


# ---------------------------------------------------------------------------
# Simulated service calls (sync)
# ---------------------------------------------------------------------------

def fetch_user_profile(user_id: int) -> dict:
  """Simulate a sync call to the user profile service."""
  time.sleep(0.02)
  user = USERS_DB.get(user_id)
  if user is None:
    raise LookupError(f'User {user_id} not found')
  return {**user, 'source': 'profile-service'}


def fetch_user_orders(user_id: int) -> list[dict]:
  """Simulate a sync call to the orders service."""
  time.sleep(0.02)
  return ORDERS_DB.get(user_id, [])


def fetch_user_prefs(user_id: int) -> dict:
  """Simulate a sync call to the preferences service."""
  time.sleep(0.02)
  return {'theme': 'dark' if user_id % 2 == 0 else 'light', 'language': 'en'}


# ---------------------------------------------------------------------------
# Simulated service calls (async)
# ---------------------------------------------------------------------------

async def async_fetch_user_profile(user_id: int) -> dict:
  await asyncio.sleep(0.02)
  user = USERS_DB.get(user_id)
  if user is None:
    raise LookupError(f'User {user_id} not found')
  return {**user, 'source': 'profile-service-async'}


async def async_fetch_user_orders(user_id: int) -> list[dict]:
  await asyncio.sleep(0.02)
  return ORDERS_DB.get(user_id, [])


async def async_fetch_user_prefs(user_id: int) -> dict:
  await asyncio.sleep(0.02)
  return {'theme': 'dark' if user_id % 2 == 0 else 'light', 'language': 'en'}


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

def log_request(request: Request) -> None:
  """Side-effect: log the incoming request."""
  print(f'  --> {request.method} {request.path}  ip={request.client_ip}  '
        f'auth={request.headers.get("Authorization", "<none>")}')


def log_response(response: Response) -> None:
  """Side-effect: log the outgoing response."""
  print(f'  <-- status={response.status}  body={response.body}')


def authenticate(request: Request) -> Request:
  """Validate the Authorization header. Early-return a 401 on failure."""
  token = request.headers.get('Authorization', '').removeprefix('Bearer ').strip()
  user = TOKEN_DB.get(token)
  if user is None:
    # Q.return_() signals early termination of the entire pipeline.
    # The return value becomes the pipeline's result -- downstream steps are skipped.
    return Q.return_(Response(
      status=401,
      body={'error': 'Unauthorized', 'detail': 'Invalid or missing token'},
    ))
  # Attach user info so downstream steps can read it.
  enriched_headers = {**request.headers, 'X-User': user}
  return Request(
    method=request.method,
    path=request.path,
    headers=enriched_headers,
    body=request.body,
    client_ip=request.client_ip,
  )


def rate_limit(request: Request) -> Request:
  """Increment per-IP counter. Early-return a 429 when the limit is exceeded."""
  count = RATE_LIMITS.get(request.client_ip, 0) + 1
  RATE_LIMITS[request.client_ip] = count
  if count > RATE_LIMIT_MAX:
    return Q.return_(Response(
      status=429,
      body={
        'error': 'Too Many Requests',
        'detail': f'Limit of {RATE_LIMIT_MAX} requests exceeded for {request.client_ip}',
      },
    ))
  return request


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_get_users(_request: Request) -> Response:
  return Response(status=200, body={'users': list(USERS_DB.values())})


_USER_PATH_RE = re.compile(r'^/users/(\d+)$')


def _handle_get_user(request: Request, user_id: int) -> Response:
  user = USERS_DB.get(user_id)
  if user is None:
    return Response(status=404, body={'error': 'Not Found', 'detail': f'User {user_id} does not exist'})
  return Response(status=200, body={'user': user})


def _handle_create_user(request: Request) -> Response:
  if not request.body or 'name' not in request.body:
    return Response(status=400, body={'error': 'Bad Request', 'detail': 'Field "name" is required'})
  new_id = max(USERS_DB) + 1
  new_user = {'id': new_id, 'name': request.body['name'], 'role': request.body.get('role', 'reader')}
  USERS_DB[new_id] = new_user
  return Response(status=201, body={'user': new_user})


def _handle_get_user_dashboard(request: Request, user_id: int) -> Response:
  """Aggregated dashboard: gather profile, orders, and prefs concurrently."""
  # gather() fans out to three services concurrently via ThreadPoolExecutor (sync)
  # or TaskGroup (async). Results are accessed by index from the returned tuple.
  result = (
    Q(user_id)
    .gather(fetch_user_profile, fetch_user_orders, fetch_user_prefs)
    .then(lambda r: {
      'profile': r[0],
      'orders': r[1],
      'preferences': r[2],
    })
    .run()
  )
  return Response(status=200, body={'dashboard': result})


_DASHBOARD_RE = re.compile(r'^/users/(\d+)/dashboard$')


def route(request: Request) -> Response:
  """Dispatch to the correct handler based on method + path."""
  if request.method == 'GET' and request.path == '/users':
    return _handle_get_users(request)

  m = _USER_PATH_RE.match(request.path)
  if request.method == 'GET' and m:
    return _handle_get_user(request, int(m.group(1)))

  dm = _DASHBOARD_RE.match(request.path)
  if request.method == 'GET' and dm:
    return _handle_get_user_dashboard(request, int(dm.group(1)))

  if request.method == 'POST' and request.path == '/users':
    return _handle_create_user(request)

  # Simulate an unexpected server error.
  if request.path == '/crash':
    raise RuntimeError('upstream database connection lost')

  return Response(
    status=404,
    body={'error': 'Not Found', 'detail': f'{request.method} {request.path} is not a known route'},
  )


# ---------------------------------------------------------------------------
# Conditional response transformation with if_/else_
# ---------------------------------------------------------------------------

def is_success(response: Response) -> bool:
  """Predicate: True when the response status is 2xx."""
  return 200 <= response.status < 300


def add_cors_headers(response: Response) -> Response:
  """Append CORS and content-type headers to successful responses."""
  return Response(
    status=response.status,
    body=response.body,
    headers={
      **response.headers,
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Content-Type': 'application/json',
    },
  )


def add_error_headers(response: Response) -> Response:
  """For error responses, add only content-type."""
  return Response(
    status=response.status,
    body=response.body,
    headers={**response.headers, 'Content-Type': 'application/json'},
  )


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

def error_handler(ei: QuentExcInfo) -> Response:
  """Centralized error handler.

  Registered with except_() default calling convention: handler(QuentExcInfo).
  ei.exc is the caught exception; ei.root_value is the Request object passed to .run().
  """
  print(f'  [error_handler] caught {type(ei.exc).__name__}: {ei.exc}')
  return Response(
    status=500,
    body={'error': 'Internal Server Error', 'detail': str(ei.exc)},
  )


# ---------------------------------------------------------------------------
# Gateway pipeline factory
# ---------------------------------------------------------------------------

def handle(request: Request) -> Response:
  """Process *request* through the full gateway pipeline."""
  return (
    Q(request)
    .do(log_request)                           # side-effect: log request
    .then(authenticate)                        # may Q.return_() a 401
    .then(rate_limit)                          # may Q.return_() a 429
    .then(route)                               # dispatch to route handler
    .if_(is_success).then(add_cors_headers)     # conditional: CORS on success
    .else_(add_error_headers)                  # else: minimal headers on error
    .do(log_response)                          # side-effect: log response
    .except_(error_handler)                    # catch unhandled exceptions
    .run()
  )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  def _section(title: str) -> None:
    print()
    print('=' * 60)
    print(title)
    print('=' * 60)

  # (a) Successful request -- full pipeline execution
  _section('(a) Successful request: GET /users')
  result = handle(Request(
    method='GET',
    path='/users',
    headers={'Authorization': 'Bearer token-alice'},
    body=None,
    client_ip='10.0.0.1',
  ))
  print(f'  Response: {result.status}  headers={list(result.headers.keys())}')

  # (b) Auth failure -- early return at authenticate step
  _section('(b) Auth failure: invalid token')
  result = handle(Request(
    method='GET',
    path='/users',
    headers={'Authorization': 'Bearer invalid-token'},
    body=None,
    client_ip='10.0.0.2',
  ))
  print(f'  Response: {result.status}')

  # (c) Rate limited -- send 6 requests from same IP (limit is 5)
  _section(f'(c) Rate limiting: {RATE_LIMIT_MAX + 1} requests from 10.0.0.3 (limit={RATE_LIMIT_MAX})')
  for i in range(1, RATE_LIMIT_MAX + 2):
    req = Request(
      method='GET',
      path='/users',
      headers={'Authorization': 'Bearer token-bob'},
      body=None,
      client_ip='10.0.0.3',
    )
    resp = handle(req)
    if resp.status == 429:
      print(f'  Request #{i}: RATE LIMITED ({resp.status})')
      break
    print(f'  Request #{i}: OK ({resp.status})')

  # (d) Dashboard -- gather() fans out to 3 services concurrently
  _section('(d) Dashboard: GET /users/1/dashboard (concurrent gather)')
  result = handle(Request(
    method='GET',
    path='/users/1/dashboard',
    headers={'Authorization': 'Bearer token-alice'},
    body=None,
    client_ip='10.0.0.6',
  ))
  print(f'  Response: {result.status}')
  if result.status == 200:
    dashboard = result.body['dashboard']
    print(f'    Profile: {dashboard["profile"]["name"]}')
    print(f'    Orders: {len(dashboard["orders"])} order(s)')
    print(f'    Prefs: {dashboard["preferences"]}')

  # (e) Server error -- route handler raises, caught by except_()
  _section('(e) Server error: GET /crash triggers RuntimeError, caught by except_()')
  result = handle(Request(
    method='GET',
    path='/crash',
    headers={'Authorization': 'Bearer token-alice'},
    body=None,
    client_ip='10.0.0.5',
  ))
  print(f'  Response: {result.status}  body={result.body}')

  # (f) Async variant -- same gateway, async service calls
  _section('(f) Async dashboard: same pipeline, async service calls')

  async def async_handle_dashboard(user_id: int) -> dict:
    """Gather async services and merge results."""
    return await (
      Q(user_id)
      .gather(
        async_fetch_user_profile,
        async_fetch_user_orders,
        async_fetch_user_prefs,
        concurrency=2,  # limit to 2 concurrent async tasks
      )
      .then(lambda r: {
        'profile': r[0],
        'orders': r[1],
        'preferences': r[2],
      })
      .run()
    )

  dashboard = asyncio.run(async_handle_dashboard(1))
  print('  Async dashboard for user 1:')
  print(f'    Profile: {dashboard["profile"]["name"]} (source: {dashboard["profile"]["source"]})')
  print(f'    Orders: {len(dashboard["orders"])} order(s)')
  print(f'    Prefs: {dashboard["preferences"]}')

  print()
  print('All demos complete.')
