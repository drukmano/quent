"""Mega real-world integration tests for quent.

Simulates real-world usage patterns: data pipelines, web/API patterns,
event processing, resource management, builder/config patterns,
functional programming patterns, decorator patterns, and edge case
combinations.
"""
import asyncio
import functools
import types
from contextlib import contextmanager
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class SyncCM:
  """Reusable sync context manager for testing."""
  def __init__(self, value='ctx', *, on_enter=None, on_exit=None, suppress=False):
    self.value = value
    self.on_enter = on_enter
    self.on_exit = on_exit
    self.suppress = suppress
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    if self.on_enter:
      self.on_enter()
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    if self.on_exit:
      self.on_exit()
    return self.suppress


class AsyncCM:
  """Reusable async context manager for testing."""
  def __init__(self, value='actx', *, on_enter=None, on_exit=None, suppress=False):
    self.value = value
    self.on_enter = on_enter
    self.on_exit = on_exit
    self.suppress = suppress
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    if self.on_enter:
      self.on_enter()
    return self.value

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    if self.on_exit:
      self.on_exit()
    return self.suppress


class AppendLog:
  """Collector for recording side-effects in order."""
  def __init__(self):
    self.entries = []

  def __call__(self, msg):
    self.entries.append(msg)

  def log(self, tag):
    def _log(v):
      self.entries.append((tag, v))
      return v
    return _log

  def log_do(self, tag):
    """Side-effect logger: appends but return value is ignored by `.do()`."""
    def _log(v):
      self.entries.append((tag, v))
    return _log


# ===================================================================
# A. Data Processing Pipelines
# ===================================================================

class DataProcessingPipelineTests(MyTestCase):

  # 1. ETL pipeline: extract -> transform -> load
  async def test_etl_pipeline_sync(self):
    """Sync ETL: extract raw rows, transform (uppercase names), load into store."""
    raw = [{'name': 'alice', 'age': 30}, {'name': 'bob', 'age': 25}]
    store = []

    def extract():
      return raw

    def transform(rows):
      return [{'name': r['name'].upper(), 'age': r['age']} for r in rows]

    def load(rows):
      store.extend(rows)
      return len(rows)

    result = Chain(extract).then(transform).then(load).run()
    await self.assertEqual(result, 2)
    super(MyTestCase, self).assertEqual(store, [
      {'name': 'ALICE', 'age': 30},
      {'name': 'BOB', 'age': 25},
    ])

  async def test_etl_pipeline_async(self):
    """Async ETL: same pipeline but with async extract."""
    raw = [{'id': 1, 'v': 'x'}, {'id': 2, 'v': 'y'}]
    store = []

    async def extract():
      return raw

    def transform(rows):
      return [{'id': r['id'], 'v': r['v'].upper()} for r in rows]

    def load(rows):
      store.extend(rows)
      return len(rows)

    result = await await_(Chain(extract).then(transform).then(load).run())
    super(MyTestCase, self).assertEqual(result, 2)
    super(MyTestCase, self).assertEqual(store[0]['v'], 'X')

  # 2. Data validation pipeline
  async def test_validation_pipeline(self):
    """Multi-stage validation: format, range, logic checks."""
    def validate_format(data):
      if not isinstance(data, dict):
        raise ValueError('not a dict')
      return data

    def validate_range(data):
      if data.get('age', 0) < 0 or data.get('age', 0) > 150:
        raise ValueError('age out of range')
      return data

    def validate_logic(data):
      if data.get('role') == 'admin' and data.get('age', 0) < 18:
        raise ValueError('admin must be 18+')
      return data

    valid_input = {'name': 'alice', 'age': 30, 'role': 'admin'}
    result = Chain(valid_input).then(validate_format).then(validate_range).then(validate_logic).run()
    await self.assertEqual(result, valid_input)

    # invalid input
    invalid_input = {'name': 'kid', 'age': 10, 'role': 'admin'}
    with self.assertRaises(ValueError):
      Chain(invalid_input).then(validate_format).then(validate_range).then(validate_logic).run()

  # 3. Data transformation with error recovery
  async def test_transform_with_error_recovery_sync(self):
    """On transform failure, use default value via except_(reraise=False)."""
    default = {'status': 'fallback'}

    def bad_transform(data):
      raise TestExc('transform failed')

    result = Chain({'raw': True}).then(bad_transform).except_(
      lambda: default, ..., reraise=False
    ).run()
    await self.assertEqual(result, default)

  async def test_transform_with_error_recovery_async(self):
    """Async transform failure recovers with default."""
    default = 'default_result'

    async def bad_async_transform(data):
      raise TestExc('async fail')

    result = await await_(
      Chain(aempty, 'input').then(bad_async_transform).except_(
        lambda: default, ..., reraise=False
      ).run()
    )
    super(MyTestCase, self).assertEqual(result, default)

  # 4. Batch processing
  async def test_batch_processing_foreach(self):
    """Process each item in a batch, then aggregate."""
    data = [1, 2, 3, 4, 5]

    def process_item(x):
      return x * 10

    def aggregate(results):
      return sum(results)

    result = Chain(data).foreach(process_item).then(aggregate).run()
    await self.assertEqual(result, 150)

  async def test_batch_processing_foreach_async(self):
    """Async batch processing with foreach."""
    data = [10, 20, 30]

    async def process_item(x):
      return x + 1

    def aggregate(results):
      return sum(results)

    result = await await_(
      Chain(aempty, data).foreach(process_item).then(aggregate).run()
    )
    super(MyTestCase, self).assertEqual(result, 63)

  # 5. Filtering pipeline
  async def test_filtering_pipeline(self):
    """Filter items, then transform each remaining item."""
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    result = Chain(items).filter(lambda x: x % 2 == 0).foreach(lambda x: x ** 2).run()
    await self.assertEqual(result, [4, 16, 36, 64, 100])

  async def test_filtering_pipeline_async_predicate(self):
    """Filter with async predicate function."""
    items = [1, 2, 3, 4, 5]

    async def is_even(x):
      return x % 2 == 0

    result = await await_(Chain(items).filter(is_even).run())
    super(MyTestCase, self).assertEqual(result, [2, 4])

  # 6. Multi-stage aggregation
  async def test_multi_stage_aggregation(self):
    """Parse raw strings, filter valid, sum values."""
    raw = ['10', 'abc', '20', '', '30', 'xyz']

    def parse(s):
      try:
        return int(s)
      except ValueError:
        return None

    result = Chain(raw).foreach(parse).filter(lambda x: x is not None).then(sum).run()
    await self.assertEqual(result, 60)

  # 7. Pipeline with logging side-effects
  async def test_pipeline_with_logging(self):
    """Side-effects via .do() don't alter the pipeline value."""
    log = AppendLog()

    result = Chain(5).do(log.log_do('input')).then(lambda v: v * 3).do(log.log_do('output')).run()
    await self.assertEqual(result, 15)
    super(MyTestCase, self).assertEqual(log.entries, [('input', 5), ('output', 15)])

  async def test_pipeline_with_logging_async(self):
    """Async pipeline with side-effect logging."""
    log = AppendLog()

    async def double(v):
      return v * 2

    result = await await_(
      Chain(aempty, 7).do(log.log_do('in')).then(double).do(log.log_do('out')).run()
    )
    super(MyTestCase, self).assertEqual(result, 14)
    super(MyTestCase, self).assertEqual(log.entries, [('in', 7), ('out', 14)])

  # 8. Conditional processing with early return
  async def test_conditional_early_return(self):
    """Chain.return_() exits chain early with a value."""
    def check(v):
      if v is None:
        Chain.return_('was_none')
      return v

    result = Chain(None).then(lambda v: None).then(check).run()
    await self.assertEqual(result, 'was_none')

    result2 = Chain(42).then(check).run()
    await self.assertEqual(result2, 42)

  async def test_conditional_early_return_nested_chain(self):
    """Nested chain with early return propagates to outer chain correctly."""
    inner = Chain().then(lambda v: Chain.return_(v * 100) if v > 5 else v)
    result = Chain(10).then(inner).run()
    await self.assertEqual(result, 1000)

    result2 = Chain(3).then(inner).run()
    await self.assertEqual(result2, 3)

  # 9. Pipeline with cleanup
  async def test_pipeline_with_cleanup(self):
    """finally_ ensures cleanup runs regardless of success."""
    cleaned = []

    def cleanup(v):
      cleaned.append('cleaned')

    result = Chain(42).then(lambda v: v + 8).finally_(cleanup).run()
    await self.assertEqual(result, 50)
    super(MyTestCase, self).assertEqual(cleaned, ['cleaned'])

  async def test_pipeline_with_cleanup_on_error(self):
    """finally_ runs even when an exception occurs."""
    cleaned = []

    def cleanup(v):
      cleaned.append('cleaned')

    def explode(v):
      raise TestExc('boom')

    with self.assertRaises(TestExc):
      Chain(42).then(explode).finally_(cleanup).run()
    # finally_ should have still run
    super(MyTestCase, self).assertEqual(cleaned, ['cleaned'])

  # 10. Parallel processing with gather
  async def test_gather_merge(self):
    """Gather extracts multiple aspects, then merge them."""
    data = {'name': 'alice', 'age': 30, 'city': 'nyc'}

    def extract_name(d):
      return d['name'].upper()

    def extract_age(d):
      return d['age'] * 2

    def extract_city(d):
      return d['city'].upper()

    def merge(parts):
      return {'name': parts[0], 'age': parts[1], 'city': parts[2]}

    result = Chain(data).gather(extract_name, extract_age, extract_city).then(merge).run()
    await self.assertEqual(result, {'name': 'ALICE', 'age': 60, 'city': 'NYC'})

  async def test_gather_async_fns(self):
    """Gather with async functions uses asyncio.gather internally."""
    async def fn_a(v):
      return v + 1

    async def fn_b(v):
      return v + 2

    async def fn_c(v):
      return v + 3

    result = await await_(Chain(10).gather(fn_a, fn_b, fn_c).then(sum).run())
    super(MyTestCase, self).assertEqual(result, 36)  # 11+12+13

  async def test_gather_mixed_sync_async(self):
    """Gather with a mix of sync and async functions."""
    def sync_fn(v):
      return v * 2

    async def async_fn(v):
      return v * 3

    result = await await_(Chain(5).gather(sync_fn, async_fn).then(sum).run())
    super(MyTestCase, self).assertEqual(result, 25)  # 10 + 15


# ===================================================================
# B. Web/API Service Patterns
# ===================================================================

class WebAPIServicePatternTests(MyTestCase):

  # 11. Request middleware chain
  async def test_request_middleware_chain(self):
    """Chain simulates request -> auth -> validate -> rate_limit -> handle."""
    log = AppendLog()

    def auth(req):
      log('auth')
      req['authed'] = True
      return req

    def validate(req):
      log('validate')
      if not req.get('authed'):
        raise TestExc('not authed')
      req['validated'] = True
      return req

    def rate_limit(req):
      log('rate_limit')
      req['rate_checked'] = True
      return req

    def handle(req):
      log('handle')
      return {'status': 200, 'body': 'ok', 'req': req}

    request = {'path': '/api/data'}
    result = Chain(request).then(auth).then(validate).then(rate_limit).then(handle).run()
    await self.assertEqual(result['status'], 200)
    super(MyTestCase, self).assertEqual(log.entries, ['auth', 'validate', 'rate_limit', 'handle'])

  # 12. Response pipeline
  async def test_response_pipeline(self):
    """Parse raw response -> validate schema -> transform."""
    import json

    raw_response = json.dumps({'data': [1, 2, 3], 'status': 'ok'})

    def parse_json(raw):
      return json.loads(raw)

    def validate_schema(parsed):
      if 'data' not in parsed:
        raise ValueError('missing data field')
      return parsed

    def transform(parsed):
      return [x * 10 for x in parsed['data']]

    result = Chain(raw_response).then(parse_json).then(validate_schema).then(transform).run()
    await self.assertEqual(result, [10, 20, 30])

  # 13. Retry with fallback
  async def test_retry_with_fallback(self):
    """On request failure, fallback handler provides default response."""
    def make_request(v=None):
      raise TestExc('connection refused')

    def fallback_handler():
      return {'status': 503, 'body': 'fallback'}

    result = Chain(make_request).except_(fallback_handler, ..., reraise=False).run()
    await self.assertEqual(result, {'status': 503, 'body': 'fallback'})

  async def test_retry_with_async_fallback(self):
    """Async fallback on async request failure."""
    async def make_request():
      raise TestExc('timeout')

    async def fallback_handler():
      return 'cached_response'

    result = await await_(
      Chain(make_request).except_(fallback_handler, ..., reraise=False).run()
    )
    super(MyTestCase, self).assertEqual(result, 'cached_response')

  # 14. Request with context manager
  async def test_request_with_sync_cm(self):
    """Chain(make_cm()).with_(process_in_context) for sync CM."""
    cm = SyncCM(value={'conn': 'active'})

    def process_in_context(ctx):
      return f"processed:{ctx['conn']}"

    result = Chain(cm).with_(process_in_context).run()
    await self.assertEqual(result, 'processed:active')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_request_with_async_cm(self):
    """Chain with async context manager."""
    cm = AsyncCM(value='async_conn')

    def process(ctx):
      return f'used:{ctx}'

    result = await await_(Chain(cm).with_(process).run())
    super(MyTestCase, self).assertEqual(result, 'used:async_conn')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  # 15. Same chain works for sync/async transparently
  async def test_transparent_sync_async(self):
    """Same chain definition works with both sync and async root values."""
    pipeline = Chain().then(lambda v: v * 2).then(lambda v: v + 1).freeze()

    # Sync execution
    sync_result = pipeline.run(5)
    await self.assertEqual(sync_result, 11)

    # Async execution (root is async)
    async_result = await await_(pipeline.run(aempty, 5))
    super(MyTestCase, self).assertEqual(async_result, 11)

  # 16. Database transaction pattern
  async def test_db_transaction_pattern(self):
    """Simulate DB transaction: get connection, use as CM, run query."""
    log = AppendLog()
    cm = SyncCM(value='db_cursor')

    def execute_query(cursor):
      log(f'query:{cursor}')
      return [{'id': 1}, {'id': 2}]

    result = Chain(cm).with_(execute_query).run()
    await self.assertEqual(result, [{'id': 1}, {'id': 2}])
    super(MyTestCase, self).assertEqual(log.entries, ['query:db_cursor'])
    super(MyTestCase, self).assertTrue(cm.exited)

  # 17. Webhook processing
  async def test_webhook_processing(self):
    """Parse webhook, validate signature, dispatch handlers."""
    def validate_signature(data):
      if data.get('sig') != 'valid':
        raise TestExc('invalid signature')
      return data

    def parse_event(data):
      return data.get('events', [])

    def process_event(e):
      return f"handled:{e}"

    webhook = {'sig': 'valid', 'events': ['push', 'pr']}
    result = Chain(webhook).then(validate_signature).then(parse_event).foreach(process_event).run()
    await self.assertEqual(result, ['handled:push', 'handled:pr'])

  async def test_webhook_invalid_signature_rejected(self):
    """Invalid webhook signature raises exception."""
    def validate_signature(data):
      if data.get('sig') != 'valid':
        raise TestExc('bad sig')
      return data

    webhook = {'sig': 'invalid', 'events': []}
    with self.assertRaises(TestExc):
      Chain(webhook).then(validate_signature).run()

  # 18. Circuit breaker pattern
  async def test_circuit_breaker_pattern(self):
    """Error counting with early return when circuit is open."""
    call_count = [0]
    max_failures = 2

    def check_circuit(v):
      if call_count[0] >= max_failures:
        Chain.return_({'status': 'circuit_open'})
      return v

    def make_call(v):
      call_count[0] += 1
      raise TestExc('service down')

    def on_error():
      return {'status': 'error', 'count': call_count[0]}

    # First call: circuit not open, call fails, error handler catches
    r1 = Chain('req').then(check_circuit).then(make_call).except_(
      on_error, ..., reraise=False
    ).run()
    await self.assertEqual(r1, {'status': 'error', 'count': 1})

    # Second call: circuit still not open (count=1 < 2)
    r2 = Chain('req').then(check_circuit).then(make_call).except_(
      on_error, ..., reraise=False
    ).run()
    await self.assertEqual(r2, {'status': 'error', 'count': 2})

    # Third call: circuit open, early return
    r3 = Chain('req').then(check_circuit).then(make_call).except_(
      on_error, ..., reraise=False
    ).run()
    await self.assertEqual(r3, {'status': 'circuit_open'})

  # 19. Async API calls with gather
  async def test_async_api_gather(self):
    """Multiple async API calls executed concurrently via gather."""
    async def api_users(v):
      return ['alice', 'bob']

    async def api_roles(v):
      return ['admin', 'user']

    async def api_perms(v):
      return ['read', 'write']

    def merge(results):
      return {'users': results[0], 'roles': results[1], 'perms': results[2]}

    result = await await_(
      Chain('auth_token').gather(api_users, api_roles, api_perms).then(merge).run()
    )
    super(MyTestCase, self).assertEqual(result, {
      'users': ['alice', 'bob'],
      'roles': ['admin', 'user'],
      'perms': ['read', 'write'],
    })

  # 20. Health check chain
  async def test_health_check_chain(self):
    """Check health of multiple services and aggregate."""
    services = ['db', 'cache', 'queue']

    def check_health(svc):
      return {'service': svc, 'healthy': True}

    def aggregate_status(results):
      all_healthy = all(r['healthy'] for r in results)
      return {'overall': 'healthy' if all_healthy else 'degraded', 'services': results}

    result = Chain(services).foreach(check_health).then(aggregate_status).run()
    await self.assertEqual(result['overall'], 'healthy')
    super(MyTestCase, self).assertEqual(len(result['services']), 3)

  async def test_health_check_chain_degraded(self):
    """One unhealthy service leads to degraded status."""
    services = ['db', 'cache', 'queue']

    def check_health(svc):
      return {'service': svc, 'healthy': svc != 'cache'}

    def aggregate_status(results):
      all_healthy = all(r['healthy'] for r in results)
      return {'overall': 'healthy' if all_healthy else 'degraded', 'services': results}

    result = Chain(services).foreach(check_health).then(aggregate_status).run()
    await self.assertEqual(result['overall'], 'degraded')


# ===================================================================
# C. Event Processing / Pub-Sub Patterns
# ===================================================================

class EventProcessingTests(MyTestCase):

  # 21. Event handler chain
  async def test_event_handler_chain(self):
    """Chain: parse -> validate -> dispatch, with exception logging."""
    errors = []

    def parse(raw):
      return {'type': raw.split(':')[0], 'data': raw.split(':')[1]}

    def validate(event):
      if not event.get('type'):
        raise TestExc('no type')
      return event

    def dispatch(event):
      return f"dispatched:{event['type']}"

    def log_error():
      errors.append('error_logged')

    result = Chain('click:button1').then(parse).then(validate).then(dispatch).except_(
      log_error, ...
    ).run()
    await self.assertEqual(result, 'dispatched:click')
    super(MyTestCase, self).assertEqual(errors, [])

  async def test_event_handler_chain_error(self):
    """Event handler chain logs error on parse failure."""
    errors = []

    def bad_parse(raw):
      raise TestExc('parse error')

    def log_error():
      errors.append('logged')
      return 'error_handled'

    result = Chain('bad_data').then(bad_parse).except_(
      log_error, ..., reraise=False
    ).run()
    await self.assertEqual(result, 'error_handled')
    super(MyTestCase, self).assertEqual(errors, ['logged'])

  # 22. Event enrichment pipeline
  async def test_event_enrichment(self):
    """Gather enriches an event from multiple sources, then merges."""
    event = {'user_id': 1, 'ip': '1.2.3.4', 'device_id': 'abc'}

    def enrich_user(evt):
      return {'user_name': 'alice'}

    def enrich_geo(evt):
      return {'country': 'US'}

    def enrich_device(evt):
      return {'device_type': 'mobile'}

    def merge_enrichments(parts):
      merged = {}
      for p in parts:
        merged.update(p)
      return merged

    result = Chain(event).gather(enrich_user, enrich_geo, enrich_device).then(merge_enrichments).run()
    await self.assertEqual(result, {'user_name': 'alice', 'country': 'US', 'device_type': 'mobile'})

  async def test_event_enrichment_async(self):
    """Async event enrichment with gather."""
    event = {'id': 99}

    async def enrich_a(evt):
      return {'a': evt['id'] * 2}

    async def enrich_b(evt):
      return {'b': evt['id'] * 3}

    def merge(parts):
      merged = {}
      for p in parts:
        merged.update(p)
      return merged

    result = await await_(Chain(event).gather(enrich_a, enrich_b).then(merge).run())
    super(MyTestCase, self).assertEqual(result, {'a': 198, 'b': 297})

  # 23. Event filtering
  async def test_event_filtering(self):
    """Filter irrelevant events, then process the rest."""
    events = [
      {'type': 'click', 'target': 'button'},
      {'type': 'hover', 'target': 'link'},
      {'type': 'click', 'target': 'submit'},
      {'type': 'scroll', 'target': 'page'},
    ]

    def is_click(e):
      return e['type'] == 'click'

    def process_event(e):
      return f"clicked:{e['target']}"

    result = Chain(events).filter(is_click).foreach(process_event).run()
    await self.assertEqual(result, ['clicked:button', 'clicked:submit'])

  # 24. Cascade for event broadcasting
  async def test_cascade_event_broadcasting(self):
    """Cascade sends same event to multiple listeners; returns original event."""
    log_a = []
    log_b = []
    log_c = []

    def notify_a(evt):
      log_a.append(evt['type'])

    def notify_b(evt):
      log_b.append(evt['type'])

    def log_event(evt):
      log_c.append(f"logged:{evt['type']}")

    event = {'type': 'user_created', 'user': 'alice'}
    result = Cascade(event).do(notify_a).do(notify_b).do(log_event).run()
    # Cascade always returns root value
    await self.assertEqual(result, event)
    super(MyTestCase, self).assertEqual(log_a, ['user_created'])
    super(MyTestCase, self).assertEqual(log_b, ['user_created'])
    super(MyTestCase, self).assertEqual(log_c, ['logged:user_created'])

  async def test_cascade_event_broadcasting_async(self):
    """Async cascade broadcasting."""
    log_a = []
    log_b = []

    async def notify_a(evt):
      log_a.append(evt['type'])

    def notify_b(evt):
      log_b.append(evt['type'])

    event = {'type': 'order_placed'}
    result = await await_(Cascade(event).do(notify_a).do(notify_b).run())
    super(MyTestCase, self).assertEqual(result, event)
    super(MyTestCase, self).assertEqual(log_a, ['order_placed'])
    super(MyTestCase, self).assertEqual(log_b, ['order_placed'])

  # 25. Event sourcing
  async def test_event_sourcing(self):
    """Validate commands and apply them to produce events."""
    commands = [
      {'cmd': 'create', 'data': 'item1'},
      {'cmd': 'update', 'data': 'item1_v2'},
      {'cmd': 'delete', 'data': 'item1'},
    ]

    events = []

    def validate_and_apply(cmd):
      if cmd['cmd'] not in ('create', 'update', 'delete'):
        raise TestExc('invalid command')
      event = {'event_type': f"{cmd['cmd']}d", 'payload': cmd['data']}
      events.append(event)
      return event

    def persist_events(evt_list):
      return {'persisted': len(evt_list)}

    result = Chain(commands).foreach(validate_and_apply).then(persist_events).run()
    await self.assertEqual(result, {'persisted': 3})
    super(MyTestCase, self).assertEqual(len(events), 3)
    super(MyTestCase, self).assertEqual(events[0]['event_type'], 'created')


# ===================================================================
# D. Resource Management Patterns
# ===================================================================

class ResourceManagementTests(MyTestCase):

  # 26. File handling with CM
  async def test_file_handling_sync_cm(self):
    """Chain(open_file()).with_(read_and_process) for sync CM."""
    cm = SyncCM(value='file_content_here')

    def read_and_process(content):
      return content.upper()

    result = Chain(cm).with_(read_and_process).run()
    await self.assertEqual(result, 'FILE_CONTENT_HERE')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  # 27. DB connection pool
  async def test_db_connection_pool(self):
    """Chain(get_connection()).with_(execute) simulates connection pool usage."""
    pool_state = {'borrowed': 0, 'returned': 0}

    class PoolCM:
      def __enter__(self_):
        pool_state['borrowed'] += 1
        return 'conn_obj'

      def __exit__(self_, *args):
        pool_state['returned'] += 1
        return False

    def execute(conn):
      return f'result_from_{conn}'

    result = Chain(PoolCM()).with_(execute).run()
    await self.assertEqual(result, 'result_from_conn_obj')
    super(MyTestCase, self).assertEqual(pool_state['borrowed'], 1)
    super(MyTestCase, self).assertEqual(pool_state['returned'], 1)

  # 28. Lock management
  async def test_lock_management(self):
    """Chain(get_lock()).with_(critical_section) ensures lock release."""
    lock_state = {'acquired': False, 'released': False}

    class LockCM:
      def __enter__(self_):
        lock_state['acquired'] = True
        return 'lock_token'

      def __exit__(self_, *args):
        lock_state['released'] = True
        return False

    result = Chain(LockCM()).with_(lambda token: f'did_work_with_{token}').run()
    await self.assertEqual(result, 'did_work_with_lock_token')
    super(MyTestCase, self).assertTrue(lock_state['acquired'])
    super(MyTestCase, self).assertTrue(lock_state['released'])

  # 29. Temp dir with cleanup
  async def test_temp_dir_with_cleanup(self):
    """CM for temp dir, process files, cleanup guaranteed."""
    cleanup_log = []
    cm = SyncCM(value='/tmp/test_dir_12345')

    def process_files(path):
      return f'processed:{path}'

    def cleanup_dir(v):
      cleanup_log.append('dir_cleaned')

    result = Chain(cm).with_(process_files).finally_(cleanup_dir).run()
    await self.assertEqual(result, 'processed:/tmp/test_dir_12345')
    super(MyTestCase, self).assertEqual(cleanup_log, ['dir_cleaned'])

  # 30. Multiple resource management (chained with_)
  async def test_multiple_resources(self):
    """Chain multiple with_ calls for nested resource management."""
    log = AppendLog()
    cm_a = SyncCM(value='resource_a', on_exit=lambda: log('exit_a'))
    cm_b = SyncCM(value='resource_b', on_exit=lambda: log('exit_b'))

    def use_a(ctx_a):
      return cm_b

    def use_b(ctx_b):
      return f'used:{ctx_b}'

    result = Chain(cm_a).with_(use_a).with_(use_b).run()
    await self.assertEqual(result, 'used:resource_b')
    super(MyTestCase, self).assertTrue(cm_a.exited)
    super(MyTestCase, self).assertTrue(cm_b.exited)

  async def test_multiple_resources_async(self):
    """Async context managers chained."""
    acm = AsyncCM(value='async_resource')

    async def use_resource(ctx):
      return f'async_used:{ctx}'

    result = await await_(Chain(acm).with_(use_resource).run())
    super(MyTestCase, self).assertEqual(result, 'async_used:async_resource')
    super(MyTestCase, self).assertTrue(acm.entered)
    super(MyTestCase, self).assertTrue(acm.exited)


# ===================================================================
# E. Builder / Configuration Patterns
# ===================================================================

class BuilderConfigTests(MyTestCase):

  # 31. Config builder
  async def test_config_builder(self):
    """Chain: default -> merge env -> merge file -> validate."""
    default_config = {'db': 'sqlite', 'port': 3000, 'debug': False}

    def merge_env(cfg):
      cfg = cfg.copy()
      cfg['port'] = 8080  # env override
      return cfg

    def merge_file(cfg):
      cfg = cfg.copy()
      cfg['debug'] = True  # file override
      return cfg

    def validate(cfg):
      if cfg['port'] < 1 or cfg['port'] > 65535:
        raise ValueError('bad port')
      return cfg

    result = Chain(default_config).then(merge_env).then(merge_file).then(validate).run()
    await self.assertEqual(result, {'db': 'sqlite', 'port': 8080, 'debug': True})

  # 32. Query builder
  async def test_query_builder(self):
    """Build a query step by step."""
    base = {'table': 'users', 'filters': [], 'sort': None, 'limit': None}

    def add_filter(q):
      q = q.copy()
      q['filters'] = q['filters'] + [('age', '>', 18)]
      return q

    def add_sort(q):
      q = q.copy()
      q['sort'] = 'name ASC'
      return q

    def add_limit(q):
      q = q.copy()
      q['limit'] = 100
      return q

    result = Chain(base).then(add_filter).then(add_sort).then(add_limit).run()
    await self.assertEqual(result['sort'], 'name ASC')
    super(MyTestCase, self).assertEqual(result['limit'], 100)
    super(MyTestCase, self).assertEqual(result['filters'], [('age', '>', 18)])

  # 33. HTML builder
  async def test_html_builder(self):
    """Build HTML from template step by step."""
    template = ''

    def add_header(html):
      return html + '<header>My Site</header>'

    def add_body(html):
      return html + '<main>Content</main>'

    def add_footer(html):
      return html + '<footer>Footer</footer>'

    result = Chain(template).then(add_header).then(add_body).then(add_footer).run()
    await self.assertEqual(result, '<header>My Site</header><main>Content</main><footer>Footer</footer>')

  # 34. Email builder
  async def test_email_builder(self):
    """Build email message step by step."""
    msg = {'to': 'user@example.com'}

    def set_subject(m):
      m = m.copy()
      m['subject'] = 'Hello'
      return m

    def set_body(m):
      m = m.copy()
      m['body'] = 'World'
      return m

    def add_attachment(m):
      m = m.copy()
      m['attachment'] = 'file.pdf'
      return m

    result = Chain(msg).then(set_subject).then(set_body).then(add_attachment).run()
    await self.assertEqual(result['subject'], 'Hello')
    super(MyTestCase, self).assertEqual(result['body'], 'World')
    super(MyTestCase, self).assertEqual(result['attachment'], 'file.pdf')

  # 35. CLI argument processing
  async def test_cli_args_processing(self):
    """Parse, validate, and normalize CLI arguments."""
    raw_args = '--verbose --output /tmp/out --count 5'

    def parse(raw):
      parts = raw.split()
      result = {}
      i = 0
      while i < len(parts):
        if parts[i] == '--verbose':
          result['verbose'] = True
          i += 1
        elif parts[i] == '--output':
          result['output'] = parts[i + 1]
          i += 2
        elif parts[i] == '--count':
          result['count'] = int(parts[i + 1])
          i += 2
        else:
          i += 1
      return result

    def validate(args):
      if 'output' not in args:
        raise ValueError('output required')
      return args

    def normalize(args):
      args = args.copy()
      args.setdefault('verbose', False)
      args.setdefault('count', 1)
      return args

    result = Chain(raw_args).then(parse).then(validate).then(normalize).run()
    await self.assertEqual(result, {'verbose': True, 'output': '/tmp/out', 'count': 5})


# ===================================================================
# F. Functional Programming Patterns
# ===================================================================

class FunctionalProgrammingTests(MyTestCase):

  # 36. Map-reduce
  async def test_map_reduce(self):
    """Map each item, then reduce with a single function."""
    data = [1, 2, 3, 4, 5]

    def mapper(x):
      return x ** 2

    def reducer(mapped):
      return sum(mapped)

    result = Chain(data).foreach(mapper).then(reducer).run()
    await self.assertEqual(result, 55)

  async def test_map_reduce_async(self):
    """Async map-reduce."""
    data = [10, 20, 30]

    async def mapper(x):
      return x // 10

    result = await await_(Chain(data).foreach(mapper).then(sum).run())
    super(MyTestCase, self).assertEqual(result, 6)

  # 37. Compose: Chain(x).then(f).then(g).then(h) == h(g(f(x)))
  async def test_compose(self):
    """Function composition via chaining."""
    def f(x):
      return x + 1

    def g(x):
      return x * 2

    def h(x):
      return x - 3

    result = Chain(10).then(f).then(g).then(h).run()
    expected = h(g(f(10)))  # h(g(11)) = h(22) = 19
    await self.assertEqual(result, expected)

  # 38. Pipeline (pipe syntax)
  async def test_pipe_syntax(self):
    """Chain(x) | f | g | h | run()."""
    def f(x):
      return x + 1

    def g(x):
      return x * 2

    def h(x):
      return x - 3

    result = Chain(10) | f | g | h | run()
    await self.assertEqual(result, 19)

  async def test_pipe_syntax_with_root_override(self):
    """Chain() | f | g | run(x)."""
    result = Chain() | (lambda v: v + 1) | (lambda v: v * 2) | run(10)
    await self.assertEqual(result, 22)

  # 39. Currying / partial application
  async def test_currying_partial(self):
    """Chain().then(fn) passes current value as first arg. Use explicit args for partial."""
    def add(a, b):
      return a + b

    # then(fn) with no extra args: fn receives current_value
    # So we use a lambda to bind the second argument
    result = Chain(3).then(lambda v: add(v, 5)).run()
    await self.assertEqual(result, 8)

  async def test_currying_with_explicit_args(self):
    """then(fn, arg1, arg2) calls fn(arg1, arg2) -- current value is NOT passed."""
    def compute(a, b, c, d=0):
      return a + b * c + d

    # Explicit args: compute(2, 3, 4, d=10) = 2 + 3*4 + 10 = 24
    result = Chain(None).then(compute, 2, 3, 4, d=10).run()
    await self.assertEqual(result, 24)

  async def test_currying_call_without_args(self):
    """then(fn, ...) calls fn() -- the ellipsis means 'call without args'."""
    def get_value():
      return 42

    result = Chain(None).then(get_value, ...).run()
    await self.assertEqual(result, 42)

  # 40. Higher-order chains: Chain as a decorator
  async def test_chain_as_decorator(self):
    """Chain().then(...).decorator() creates a function decorator."""
    @Chain().then(lambda v: v * 2).then(lambda v: v + 1).decorator()
    def compute(x):
      return x + 10

    result = compute(5)  # (5+10)*2 + 1 = 31
    await self.assertEqual(result, 31)

  # 41. Monadic error handling (Maybe monad via except_)
  async def test_monadic_error_handling(self):
    """except_ with reraise=False acts like a Maybe monad: errors return None."""
    def safe_div(v):
      return v / 0  # ZeroDivisionError

    result = Chain(10).then(safe_div).except_(
      lambda: None, ..., reraise=False
    ).run()
    await self.assertIsNone(result)

  async def test_monadic_chained_operations(self):
    """Chain of operations where first failure short-circuits via except_."""
    def step1(v):
      return v * 2

    def step2(v):
      if v > 10:
        raise TestExc('too big')
      return v

    def step3(v):
      return v + 100

    # With input 3: 3*2=6, 6<10 -> 6, 6+100=106
    result = Chain(3).then(step1).then(step2).then(step3).except_(
      lambda: 'failed', ..., reraise=False
    ).run()
    await self.assertEqual(result, 106)

    # With input 10: 10*2=20, 20>10 raises, caught -> 'failed'
    result2 = Chain(10).then(step1).then(step2).then(step3).except_(
      lambda: 'failed', ..., reraise=False
    ).run()
    await self.assertEqual(result2, 'failed')

  # 42. Lazy evaluation via freeze
  async def test_lazy_evaluation_freeze(self):
    """freeze() creates a reusable lazy evaluator."""
    frozen = Chain().then(lambda v: v ** 2).then(lambda v: v + 1).freeze()

    # Not evaluated until run
    r1 = frozen.run(3)
    await self.assertEqual(r1, 10)  # 3^2 + 1

    r2 = frozen.run(5)
    await self.assertEqual(r2, 26)  # 5^2 + 1

  # 43. Function composition with freeze
  async def test_compose_with_freeze(self):
    """Compose = Chain().then(f).then(g).freeze() -> reusable composed function."""
    compose = Chain().then(lambda v: v.strip()).then(lambda v: v.upper()).freeze()

    r1 = compose.run('  hello  ')
    await self.assertEqual(r1, 'HELLO')

    r2 = compose.run('  world  ')
    await self.assertEqual(r2, 'WORLD')

  async def test_compose_freeze_async(self):
    """Freeze with async operations."""
    async def double(v):
      return v * 2

    compose = Chain().then(double).then(lambda v: v + 1).freeze()

    r = await await_(compose.run(5))
    super(MyTestCase, self).assertEqual(r, 11)


# ===================================================================
# G. Complex Integration Scenarios
# ===================================================================

class ComplexIntegrationTests(MyTestCase):

  # 44. Nested chains for conditional branching
  async def test_nested_chains_conditional(self):
    """Outer chain uses inner chains as links for branching logic."""
    inner_positive = Chain().then(lambda v: f'positive:{v}')
    inner_negative = Chain().then(lambda v: f'negative:{v}')

    def branch(v):
      if v >= 0:
        return inner_positive.clone().run(v)
      return inner_negative.clone().run(v)

    result = Chain(5).then(branch).run()
    await self.assertEqual(result, 'positive:5')

    result2 = Chain(-3).then(branch).run()
    await self.assertEqual(result2, 'negative:-3')

  # 45. Chain reuse via clone
  async def test_chain_reuse_clone(self):
    """Define once, clone and customize for different inputs."""
    base = Chain().then(lambda v: v * 2)

    # Clone and extend
    c1 = base.clone().then(lambda v: v + 1)
    c2 = base.clone().then(lambda v: v - 1)

    r1 = c1.run(5)
    await self.assertEqual(r1, 11)  # 5*2+1

    r2 = c2.run(5)
    await self.assertEqual(r2, 9)  # 5*2-1

    # Original unchanged
    r_base = base.run(5)
    await self.assertEqual(r_base, 10)

  async def test_clone_independence_after_modification(self):
    """Clones are fully independent: modifying one doesn't affect the other."""
    original = Chain(1).then(lambda v: v + 1)
    clone1 = original.clone()
    clone2 = original.clone()

    clone1.then(lambda v: v * 100)
    clone2.then(lambda v: v * 200)

    await self.assertEqual(original.run(), 2)
    await self.assertEqual(clone1.run(), 200)
    await self.assertEqual(clone2.run(), 400)

  # 46. FrozenChain as shared processor
  async def test_frozen_chain_shared_processor(self):
    """FrozenChain processes many inputs without mutation."""
    processor = Chain().then(lambda v: v.lower()).then(lambda v: v.replace(' ', '_')).freeze()

    inputs = ['Hello World', 'FOO BAR', 'Test Case']
    results = [processor.run(i) for i in inputs]
    await self.assertEqual(results[0], 'hello_world')
    super(MyTestCase, self).assertEqual(results[1], 'foo_bar')
    super(MyTestCase, self).assertEqual(results[2], 'test_case')

  # 47. Cascade for fluent configuration
  async def test_cascade_fluent_config(self):
    """Cascade(config_obj).do(set_a).do(set_b).run() returns original config."""
    config = {'a': None, 'b': None, 'c': None}

    def set_a(cfg):
      cfg['a'] = 1

    def set_b(cfg):
      cfg['b'] = 2

    def set_c(cfg):
      cfg['c'] = 3

    result = Cascade(config).do(set_a).do(set_b).do(set_c).run()
    # Cascade returns root value (the same config dict, now mutated)
    await self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3})
    # Verify it's the same object
    super(MyTestCase, self).assertIs(result, config)

  # 48. Mixed sync/async links
  async def test_mixed_sync_async_links(self):
    """Chain with interleaved sync and async operations."""
    async def async_add(v):
      return v + 1

    def sync_mul(v):
      return v * 2

    async def async_sub(v):
      return v - 3

    result = await await_(
      Chain(5).then(async_add).then(sync_mul).then(async_sub).run()
    )
    # (5+1)*2-3 = 9
    super(MyTestCase, self).assertEqual(result, 9)

  # 49. Error recovery chain: try -> alt -> default
  async def test_error_recovery_cascade(self):
    """First operation fails, except handler returns fallback."""
    def primary():
      raise TestExc('primary failed')

    result = Chain(primary).except_(
      lambda: 'fallback', ..., reraise=False
    ).run()
    await self.assertEqual(result, 'fallback')

  async def test_error_recovery_specific_exception(self):
    """Exception handler targets specific exception type."""
    class DBError(TestExc):
      pass

    class NetworkError(TestExc):
      pass

    def operation():
      raise DBError('db down')

    # Handler only catches DBError
    result = Chain(operation).except_(
      lambda: 'db_fallback', ..., exceptions=DBError, reraise=False
    ).run()
    await self.assertEqual(result, 'db_fallback')

    # NetworkError is not caught
    def operation2():
      raise NetworkError('net down')

    with self.assertRaises(NetworkError):
      Chain(operation2).except_(
        lambda: 'db_fallback', ..., exceptions=DBError, reraise=False
      ).run()

  # 50. Multiple exception handlers
  async def test_multiple_exception_handlers(self):
    """Different handlers for different exception types."""
    class AuthError(Exception):
      pass

    class RateLimit(Exception):
      pass

    def fail_auth():
      raise AuthError('auth failed')

    result = Chain(fail_auth).except_(
      lambda: 'auth_recovery', ..., exceptions=AuthError, reraise=False
    ).except_(
      lambda: 'rate_recovery', ..., exceptions=RateLimit, reraise=False
    ).run()
    await self.assertEqual(result, 'auth_recovery')

    # Now test with RateLimit
    def fail_rate():
      raise RateLimit('too fast')

    result2 = Chain(fail_rate).except_(
      lambda: 'auth_recovery', ..., exceptions=AuthError, reraise=False
    ).except_(
      lambda: 'rate_recovery', ..., exceptions=RateLimit, reraise=False
    ).run()
    await self.assertEqual(result2, 'rate_recovery')

  # 51. Complex foreach with frozen chain
  async def test_foreach_with_frozen_chain(self):
    """foreach uses a frozen chain to validate and transform each item.
    Chains must be frozen before use in foreach (unfrozen chains trigger nested error)."""
    inner = Chain().then(lambda v: v * 10).freeze()

    data = [1, 2, 3]
    result = Chain(data).foreach(inner).run()
    await self.assertEqual(result, [10, 20, 30])

  async def test_foreach_frozen_chain_async(self):
    """Async foreach with frozen chain."""
    async def double(v):
      return v * 2

    inner = Chain().then(double).freeze()
    data = [5, 10, 15]
    result = await await_(Chain(data).foreach(inner).run())
    super(MyTestCase, self).assertEqual(result, [10, 20, 30])

  # 52. Generator / iterate for lazy streaming
  async def test_iterate_lazy_streaming(self):
    """iterate() returns a lazy generator over chain results."""
    data = [1, 2, 3, 4, 5]
    gen = Chain(data).iterate(lambda x: x * 10)

    results = list(gen)
    super(MyTestCase, self).assertEqual(results, [10, 20, 30, 40, 50])

  async def test_iterate_async_streaming(self):
    """Async iterate over chain results."""
    data = [10, 20, 30]

    async def transform(x):
      return x + 1

    gen = Chain(data).iterate(transform)
    results = []
    async for item in gen:
      results.append(item)
    super(MyTestCase, self).assertEqual(results, [11, 21, 31])

  async def test_iterate_no_transform(self):
    """Iterate without a transform function yields raw elements."""
    data = [1, 2, 3]
    gen = Chain(data).iterate()
    results = list(gen)
    super(MyTestCase, self).assertEqual(results, [1, 2, 3])

  # 53. Complete CRUD workflow
  async def test_crud_workflow(self):
    """Create -> Read -> Update -> Delete with error handling."""
    db = {}

    def create(item):
      db[item['id']] = item
      return item['id']

    def read(item_id):
      if item_id not in db:
        raise TestExc('not found')
      return db[item_id]

    def update(item):
      if item['id'] not in db:
        raise TestExc('not found for update')
      db[item['id']].update(item)
      return db[item['id']]

    def delete(item):
      item_id = item['id']
      if item_id in db:
        del db[item_id]
      return item_id

    # Create
    create_result = Chain({'id': 1, 'name': 'alice'}).then(create).run()
    await self.assertEqual(create_result, 1)

    # Read
    read_result = Chain(1).then(read).run()
    await self.assertEqual(read_result, {'id': 1, 'name': 'alice'})

    # Update
    update_result = Chain({'id': 1, 'name': 'alice_v2'}).then(update).run()
    await self.assertEqual(update_result['name'], 'alice_v2')

    # Delete
    delete_result = Chain({'id': 1}).then(delete).run()
    await self.assertEqual(delete_result, 1)
    super(MyTestCase, self).assertNotIn(1, db)

    # Read deleted -> error recovered
    read_deleted = Chain(1).then(read).except_(
      lambda: None, ..., reraise=False
    ).run()
    await self.assertIsNone(read_deleted)


# ===================================================================
# H. Decorator Pattern Tests
# ===================================================================

class DecoratorPatternTests(MyTestCase):

  # 54. Simple function decorator
  async def test_decorator_simple_function(self):
    """Chain.decorator() on a plain function."""
    @Chain().then(lambda v: v * 3).decorator()
    def triple(x):
      return x

    await self.assertEqual(triple(5), 15)

  # 55. Async function decorator
  async def test_decorator_async_function(self):
    """Chain.decorator() on an async function."""
    @Chain().then(lambda v: v + 100).decorator()
    async def async_fn(x):
      return x * 2

    result = await await_(async_fn(5))
    super(MyTestCase, self).assertEqual(result, 110)  # 5*2 + 100 = 110

  # 56. Decorated method as class method (descriptor protocol)
  async def test_decorator_descriptor_protocol(self):
    """Decorated function works as an instance method."""
    class MyService:
      def __init__(self, multiplier):
        self.multiplier = multiplier

      @Chain().then(lambda v: v).decorator()
      def compute(self, x):
        return x * self.multiplier

    svc = MyService(5)
    result = svc.compute(3)
    await self.assertEqual(result, 15)

  async def test_decorator_different_instances(self):
    """Descriptor binds to different instances correctly."""
    class Counter:
      def __init__(self, base):
        self.base = base

      @Chain().then(lambda v: v + 1).decorator()
      def next_val(self):
        return self.base

    a = Counter(10)
    b = Counter(20)
    await self.assertEqual(a.next_val(), 11)
    await self.assertEqual(b.next_val(), 21)

  # 57. Multiple stacked decorators
  async def test_stacked_decorators(self):
    """Multiple chain decorators stacked."""
    @Chain().then(lambda v: v * 2).decorator()
    @Chain().then(lambda v: v + 1).decorator()
    def base(x):
      return x

    # Inner: base(3) -> 3, +1 -> 4
    # Outer: 4, *2 -> 8
    await self.assertEqual(base(3), 8)

  async def test_triple_stacked_decorators(self):
    """Three chain decorators stacked."""
    @Chain().then(lambda v: v - 1).decorator()
    @Chain().then(lambda v: v * 2).decorator()
    @Chain().then(lambda v: v + 10).decorator()
    def base(x):
      return x

    # Inner: base(5) -> 5, +10 -> 15
    # Middle: 15, *2 -> 30
    # Outer: 30, -1 -> 29
    await self.assertEqual(base(5), 29)

  # 58. Decorated function preserves metadata
  async def test_decorator_preserves_name(self):
    """functools.update_wrapper preserves __name__."""
    @Chain().then(lambda v: v).decorator()
    def my_func(x):
      return x

    super(MyTestCase, self).assertEqual(my_func.__name__, 'my_func')

  async def test_decorator_preserves_doc(self):
    """functools.update_wrapper preserves __doc__."""
    @Chain().then(lambda v: v).decorator()
    def documented(x):
      """This is doc."""
      return x

    super(MyTestCase, self).assertEqual(documented.__doc__, 'This is doc.')

  async def test_decorator_preserves_module(self):
    """functools.update_wrapper preserves __module__."""
    @Chain().then(lambda v: v).decorator()
    def modular(x):
      return x

    super(MyTestCase, self).assertEqual(modular.__module__, __name__)


# ===================================================================
# I. Edge Case Combinations
# ===================================================================

class EdgeCaseCombinationTests(MyTestCase):

  # 59. Chain with ALL features combined
  async def test_chain_all_features(self):
    """Chain using then, do, except_, finally_, foreach, filter, gather."""
    log = AppendLog()
    cleanup_ran = []

    def on_cleanup(v):
      cleanup_ran.append(True)

    data = {'items': [1, 2, 3, 4, 5, 6]}

    result = Chain(data) \
      .do(log.log_do('start')) \
      .then(lambda d: d['items']) \
      .filter(lambda x: x > 2) \
      .foreach(lambda x: x * 10) \
      .then(sum) \
      .finally_(on_cleanup) \
      .run()

    await self.assertEqual(result, 180)  # (3+4+5+6)*10 = 180
    super(MyTestCase, self).assertEqual(log.entries, [('start', data)])
    super(MyTestCase, self).assertEqual(cleanup_ran, [True])

  async def test_chain_all_features_with_error(self):
    """Chain with all features, error occurs, except_ + finally_ both fire."""
    log = AppendLog()
    cleanup_ran = []

    def on_cleanup(v):
      cleanup_ran.append(True)

    def explode(v):
      raise TestExc('boom')

    result = Chain(42) \
      .do(log.log_do('before')) \
      .then(explode) \
      .except_(lambda: 'recovered', ..., reraise=False) \
      .finally_(on_cleanup) \
      .run()

    await self.assertEqual(result, 'recovered')
    super(MyTestCase, self).assertEqual(log.entries, [('before', 42)])
    super(MyTestCase, self).assertEqual(cleanup_ran, [True])

  # 60. Deeply nested chain with different features at each level
  async def test_deeply_nested_chain(self):
    """Three levels of nesting, each with its own features."""
    inner = Chain().then(lambda v: v * 2)
    mid = Chain().then(inner).then(lambda v: v + 1)
    outer = Chain(5).then(mid).then(lambda v: v * 3)

    # 5 -> inner: 10 -> mid: 10+1=11 -> outer: 11*3=33
    result = outer.run()
    await self.assertEqual(result, 33)

  async def test_deeply_nested_async(self):
    """Deeply nested chain with async at inner level."""
    async def async_double(v):
      return v * 2

    inner = Chain().then(async_double)
    mid = Chain().then(inner).then(lambda v: v + 5)
    outer = Chain(3).then(mid)

    # 3 -> async_double: 6 -> 6+5=11
    result = await await_(outer.run())
    super(MyTestCase, self).assertEqual(result, 11)

  # 61. Chain with empty / None operations
  async def test_chain_empty_operations(self):
    """Chain with operations that return None, filtered by bool."""
    result = Chain([None, 0, '', 'hello', 42]).filter(bool).run()
    await self.assertEqual(result, ['hello', 42])

  async def test_chain_void_then_none(self):
    """Chain with no root and a then that returns None."""
    result = Chain().then(lambda: None).run()
    await self.assertIsNone(result)

  # 62. Chain of chains: Chain(Chain(Chain(42)))
  async def test_chain_of_chains_triple_nesting(self):
    """Triple-nested Chain: inner chain is root of middle, middle is root of outer."""
    inner = Chain(42)
    mid = Chain(inner)
    outer = Chain(mid)

    result = outer.run()
    await self.assertEqual(result, 42)

  async def test_chain_of_chains_with_operations(self):
    """Nested chains each adding their own operation."""
    inner = Chain().then(lambda v: v + 1)
    mid = Chain().then(inner).then(lambda v: v * 2)
    outer = Chain(10).then(mid)

    # 10 -> inner: 11 -> mid: 22
    result = outer.run()
    await self.assertEqual(result, 22)

  # 63. Cascade inside Chain inside Cascade
  async def test_mixed_cascade_chain_nesting(self):
    """Cascade -> Chain -> Cascade: mixed nesting."""
    log = []

    # Inner cascade: receives value, runs side-effects, returns root value
    inner_cascade = Cascade().do(lambda v: log.append(f'inner:{v}'))

    # Middle chain: passes value through inner cascade, then transforms
    mid_chain = Chain().then(inner_cascade).then(lambda v: v * 3)

    # Outer cascade: runs mid_chain as side-effect on root
    outer_root = 5
    result = Cascade(outer_root).do(lambda v: log.append(f'outer:{v}')).then(mid_chain).run()

    # Cascade returns root value at the end, but mid_chain is a .then() link
    # which receives root_value (due to Cascade) and returns mid_chain result.
    # However since outer is a Cascade, the final result is root_value (5)
    await self.assertEqual(result, outer_root)
    super(MyTestCase, self).assertIn('outer:5', log)

  async def test_chain_inside_cascade(self):
    """Chain used as a link inside Cascade."""
    inner_chain = Chain().then(lambda v: v * 10)
    # In Cascade, inner_chain receives root_value, transforms it,
    # but Cascade discards the result and returns root_value.
    result = Cascade(7).then(inner_chain).run()
    # Cascade returns root value
    await self.assertEqual(result, 7)

  async def test_cascade_inside_chain(self):
    """Cascade used as a link inside Chain."""
    log = []
    inner_cascade = Cascade().do(lambda v: log.append(v))
    # In Chain, inner_cascade receives current_value as root,
    # runs do() on it, and returns root_value (same as input).
    result = Chain(42).then(inner_cascade).then(lambda v: v + 1).run()
    await self.assertEqual(result, 43)
    super(MyTestCase, self).assertEqual(log, [42])


# ===================================================================
# Additional integration tests
# ===================================================================

class AdditionalIntegrationTests(MyTestCase):

  async def test_foreach_with_index(self):
    """foreach with_index=True passes (index, element) to function."""
    data = ['a', 'b', 'c']

    result = Chain(data).foreach(lambda i, el: f'{i}:{el}', with_index=True).run()
    await self.assertEqual(result, ['0:a', '1:b', '2:c'])

  async def test_foreach_with_break(self):
    """foreach with Chain.break_() stops iteration early."""
    data = [1, 2, 3, 4, 5]

    def process(x):
      if x == 3:
        Chain.break_()
      return x * 10

    result = Chain(data).foreach(process).run()
    await self.assertEqual(result, [10, 20])

  async def test_foreach_break_with_value(self):
    """break_ with a value replaces the accumulated list."""
    data = [1, 2, 3, 4, 5]

    def process(x):
      if x == 3:
        Chain.break_('stopped_early')
      return x * 10

    result = Chain(data).foreach(process).run()
    await self.assertEqual(result, 'stopped_early')

  async def test_return_from_nested_chain(self):
    """Chain.return_() inside a nested chain exits the outer chain."""
    inner = Chain().then(lambda v: Chain.return_('early_exit') if v > 100 else v)
    result = Chain(200).then(inner).then(lambda v: 'should_not_reach').run()
    await self.assertEqual(result, 'early_exit')

    result2 = Chain(50).then(inner).then(lambda v: f'reached:{v}').run()
    await self.assertEqual(result2, 'reached:50')

  async def test_no_async_mode(self):
    """no_async() skips coroutine detection for pure sync usage."""
    c = Chain(10).then(lambda v: v * 2).no_async(True)
    result = c.run()
    await self.assertEqual(result, 20)

  async def test_config_autorun(self):
    """config(autorun=True) causes async chains to return Tasks."""
    c = Chain(aempty, 10).then(lambda v: v + 1).config(autorun=True)
    result = c.run()
    # autorun wraps the coroutine in a Task
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    awaited = await result
    super(MyTestCase, self).assertEqual(awaited, 11)

  async def test_pipe_or_with_multiple_operations(self):
    """Extended pipe chain with many operations."""
    result = (
      Chain(1)
      | (lambda v: v + 1)
      | (lambda v: v * 2)
      | (lambda v: v + 3)
      | (lambda v: v * 4)
      | run()
    )
    # 1+1=2, 2*2=4, 4+3=7, 7*4=28
    await self.assertEqual(result, 28)

  async def test_chain_bool_always_true(self):
    """Chain instances are always truthy."""
    super(MyTestCase, self).assertTrue(bool(Chain()))
    super(MyTestCase, self).assertTrue(bool(Chain(None)))
    super(MyTestCase, self).assertTrue(bool(Cascade()))

  async def test_chain_repr(self):
    """Chain has a string representation."""
    c = Chain(42).then(lambda v: v * 2)
    r = repr(c)
    super(MyTestCase, self).assertIsInstance(r, str)

  async def test_frozen_chain_call_vs_run(self):
    """FrozenChain.__call__() and .run() are equivalent."""
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    await self.assertEqual(frozen(), frozen.run())

  async def test_frozen_chain_with_override(self):
    """FrozenChain.run(override) passes override as root."""
    frozen = Chain().then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen.run(7), 21)
    await self.assertEqual(frozen(7), 21)

  async def test_chain_run_override_on_void_chain(self):
    """Chain without root can receive root at run time."""
    c = Chain().then(lambda v: v + 10)
    await self.assertEqual(c.run(5), 15)

  async def test_chain_root_override_raises(self):
    """Cannot override root value of a chain that already has one."""
    c = Chain(42)
    with self.assertRaises(QuentException):
      c.run(99)

  async def test_except_reraise_true(self):
    """except_ with reraise=True (default) still re-raises after handler."""
    ran = []

    def handler():
      ran.append('caught')

    def raise_test_exc():
      raise TestExc('x')

    with self.assertRaises(TestExc):
      Chain(raise_test_exc).except_(handler, ...).run()
    # Handler ran but exception was re-raised
    super(MyTestCase, self).assertEqual(ran, ['caught'])

  async def test_except_with_specific_exception_class(self):
    """except_ only catches specified exception classes."""
    class SpecificError(Exception):
      pass

    def handler():
      return 'caught_specific'

    def raise_specific():
      raise SpecificError('x')

    def raise_type_error():
      raise TypeError('x')

    # SpecificError is caught
    result = Chain(raise_specific).except_(
      handler, ..., exceptions=SpecificError, reraise=False
    ).run()
    await self.assertEqual(result, 'caught_specific')

    # TypeError is not caught by SpecificError handler
    with self.assertRaises(TypeError):
      Chain(raise_type_error).except_(
        handler, ..., exceptions=SpecificError, reraise=False
      ).run()

  async def test_finally_receives_root_value(self):
    """finally_ callback receives the root value."""
    received = []

    def on_finally(root):
      received.append(root)

    Chain(42).then(lambda v: v * 2).finally_(on_finally).run()
    super(MyTestCase, self).assertEqual(received, [42])

  async def test_finally_only_one_allowed(self):
    """Only one finally_ callback can be registered."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: v).finally_(lambda v: v)

  async def test_sleep_in_chain(self):
    """sleep() pauses execution without altering the value."""
    import time
    start = time.monotonic()
    result = await await_(Chain(42).sleep(0.05).then(lambda v: v + 1).run())
    elapsed = time.monotonic() - start
    super(MyTestCase, self).assertEqual(result, 43)
    super(MyTestCase, self).assertGreaterEqual(elapsed, 0.04)

  async def test_cascade_with_then(self):
    """Cascade.then() still receives root value, not previous result."""
    results = []

    def record_and_return(v):
      results.append(v)
      return v * 100  # This result is passed as root to next .then() in cascade? No, root is always root.

    root = 5
    final = Cascade(root).then(record_and_return).then(record_and_return).run()
    # Each .then() receives root (5), not the previous result (500)
    await self.assertEqual(final, root)
    super(MyTestCase, self).assertEqual(results, [5, 5])

  async def test_cascade_do_vs_then(self):
    """Cascade: .do() is side-effect, .then() receives root. Both ignore result for final."""
    do_log = []
    then_log = []

    result = Cascade(10) \
      .do(lambda v: do_log.append(v)) \
      .then(lambda v: then_log.append(v)) \
      .run()

    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(do_log, [10])
    super(MyTestCase, self).assertEqual(then_log, [10])

  async def test_gather_with_single_fn(self):
    """Gather with a single function returns a single-element list."""
    result = Chain(5).gather(lambda v: v * 2).run()
    await self.assertEqual(result, [10])

  async def test_filter_empty_result(self):
    """Filter that removes all elements returns empty list."""
    result = Chain([1, 2, 3]).filter(lambda x: x > 100).run()
    await self.assertEqual(result, [])

  async def test_foreach_empty_list(self):
    """foreach on empty list returns empty list."""
    result = Chain([]).foreach(lambda x: x * 10).run()
    await self.assertEqual(result, [])

  async def test_chain_with_literal_values(self):
    """Chain can hold literal (non-callable) values in then()."""
    result = Chain(None).then(42).run()
    await self.assertEqual(result, 42)

    result2 = Chain(None).then('hello').run()
    await self.assertEqual(result2, 'hello')

    result3 = Chain(None).then([1, 2, 3]).run()
    await self.assertEqual(result3, [1, 2, 3])

  async def test_chain_run_returns_none_for_void(self):
    """Chain with no root and no operations returns None."""
    result = Chain().run()
    await self.assertIsNone(result)

  async def test_with_fn_pattern_sync_async(self):
    """Tests using with_fn pattern to test both sync and async roots."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 42).then(lambda v: v + 1).run()
        await self.assertEqual(result, 43)

  async def test_complex_data_pipeline_end_to_end(self):
    """Full end-to-end pipeline: load -> validate -> transform -> filter -> aggregate -> report."""
    raw_data = [
      {'name': 'alice', 'score': 85},
      {'name': 'bob', 'score': -5},  # invalid
      {'name': 'charlie', 'score': 92},
      {'name': 'diana', 'score': 78},
      {'name': 'eve', 'score': 200},  # invalid
    ]

    def validate(records):
      return [r for r in records if 0 <= r['score'] <= 100]

    def transform(records):
      return [{'name': r['name'].upper(), 'grade': 'A' if r['score'] >= 90 else 'B' if r['score'] >= 80 else 'C'} for r in records]

    def aggregate(records):
      grades = {}
      for r in records:
        grades.setdefault(r['grade'], []).append(r['name'])
      return grades

    result = Chain(raw_data).then(validate).then(transform).then(aggregate).run()
    await self.assertEqual(result, {
      'A': ['CHARLIE'],
      'B': ['ALICE'],
      'C': ['DIANA'],
    })

  async def test_error_in_gather_propagates(self):
    """Error in one of the gather functions propagates."""
    def good_fn(v):
      return v * 2

    def bad_fn(v):
      raise TestExc('gather fail')

    with self.assertRaises(TestExc):
      Chain(5).gather(good_fn, bad_fn).run()

  async def test_except_handler_receives_root_value(self):
    """except_ handler can receive the root value."""
    def handler(root_val):
      return f'recovered_from_{root_val}'

    def raise_test_exc(v):
      raise TestExc('x')

    result = Chain(42).then(raise_test_exc).except_(
      handler, reraise=False
    ).run()
    await self.assertEqual(result, 'recovered_from_42')

  async def test_async_foreach_with_sync_predicate_in_filter(self):
    """Filter with sync predicate after async root."""
    data = [1, 2, 3, 4, 5]

    result = await await_(
      Chain(aempty, data).filter(lambda x: x % 2 == 0).foreach(lambda x: x * 10).run()
    )
    super(MyTestCase, self).assertEqual(result, [20, 40])

  async def test_iterate_with_root_override(self):
    """Generator with root value override at call time."""
    gen = Chain().iterate(lambda x: x * 2)
    results = list(gen([1, 2, 3]))
    super(MyTestCase, self).assertEqual(results, [2, 4, 6])

  async def test_clone_preserves_cascade_mode(self):
    """Cloning a Cascade produces a Cascade."""
    c = Cascade(10)
    c2 = c.clone()
    super(MyTestCase, self).assertIsInstance(c2, Cascade)

  async def test_complex_chain_with_gather_and_foreach(self):
    """Chain: gather multiple extractors, then foreach on combined results."""
    def extract_names(data):
      return [d['name'] for d in data]

    def extract_ages(data):
      return [d['age'] for d in data]

    def combine(parts):
      return list(zip(parts[0], parts[1]))

    data = [{'name': 'alice', 'age': 30}, {'name': 'bob', 'age': 25}]
    result = Chain(data).gather(extract_names, extract_ages).then(combine).foreach(
      lambda pair: f'{pair[0]}:{pair[1]}'
    ).run()
    await self.assertEqual(result, ['alice:30', 'bob:25'])
