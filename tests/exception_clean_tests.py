import traceback
from unittest import TestCase

from quent import Chain


def level_1(x):
  return level_2(x * 2)


def level_2(x):
  result = Chain(x).then(level_3).then(level_4)()
  return result


def level_3(x):
  return x + 100


def level_4(x):
  return level_5(x / 2)


def level_5(x):
  chain = (Chain(x)
           .then(lambda v: v * 3)
           .then(level_6)
           .then(level_7))
  return chain()


def level_6(x):
  return x ** 2


def level_7(x):
  return level_8(x)


def level_8(x):
  result = Chain(x).then(level_9)()
  return level_10(result)


def level_9(x):
  if x > 1000000:
    return level_11(x)
  return x


def level_10(x):
  return level_11(x * 1.5)


def level_11(x):
  chain = Chain(x).then(level_12).then(lambda v: level_13(v))
  return chain()


def level_12(x):
  return x / 100


def level_13(x):
  result = (Chain(x)
            .then(level_14)
            .then(level_15))()
  return result


def level_14(x):
  return x + 50000


def level_15(x):
  return x / 0


def _get_tb_entries(exc):
  return traceback.extract_tb(exc.__traceback__)


def _get_tb_filenames(exc):
  return [entry.filename for entry in _get_tb_entries(exc)]


def _get_tb_func_names(exc):
  return [entry.name for entry in _get_tb_entries(exc)]


class ExceptionCleanTests(TestCase):
  def test_deep_chain_raises_correct_exception(self):
    with self.assertRaises(ZeroDivisionError):
      level_1(10)

  def test_deep_chain_has_quent_visualization_frame(self):
    try:
      level_1(10)
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      filenames = _get_tb_filenames(exc)
      self.assertIn(
        '<quent>', filenames,
        'Traceback should contain a <quent> visualization frame'
      )

  def test_deep_chain_no_helper_frames(self):
    try:
      level_1(10)
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      filenames = _get_tb_filenames(exc)
      for fn in filenames:
        self.assertNotIn(
          'quent/helpers', fn,
          f'Quent helpers frame should be cleaned: {fn}'
        )
        self.assertNotIn(
          'quent/custom', fn,
          f'Quent custom frame should be cleaned: {fn}'
        )

  def test_deep_chain_preserves_user_frames(self):
    try:
      level_1(10)
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      func_names = _get_tb_func_names(exc)
      self.assertIn('level_15', func_names,
                    'User function level_15 (error site) should be in traceback')
      self.assertIn('level_1', func_names,
                    'User function level_1 (entry point) should be in traceback')

  def test_deep_chain_quent_frame_shows_chain_info(self):
    try:
      level_1(10)
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_tb_entries(exc)
      quent_entries = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_entries), 0,
                         'Should have at least one <quent> frame')
      # The <quent> frame name contains chain visualization info
      for entry in quent_entries:
        self.assertTrue(
          len(entry.name) > 0,
          '<quent> frame should have a non-empty name (chain visualization)'
        )

  def test_mixed_chain_regular_key_error(self):
    def process_data(value):
      return Chain(value).then(validate_range)()

    def validate_range(value):
      return transform_data(value)

    def transform_data(value):
      result = (Chain(value)
                .then(lambda x: x * 10)
                .then(Chain().then(apply_business_logic))
                .then(format_output))()
      return result

    def apply_business_logic(value):
      return access_database(value / 2)

    def access_database(value):
      data = fetch_record(value)
      return Chain(data).then(process_record)()

    def fetch_record(record_id):
      return {"id": record_id, "data": "test"}

    def process_record(record):
      return record["missing_field"]

    def format_output(value):
      return str(value)

    with self.assertRaises(KeyError):
      process_data(42)

  def test_mixed_chain_has_visualization_and_user_frames(self):
    def process_data(value):
      return Chain(value).then(validate_range)()

    def validate_range(value):
      return transform_data(value)

    def transform_data(value):
      result = (Chain(value)
                .then(lambda x: x * 10)
                .then(Chain().then(apply_business_logic))
                .then(format_output))()
      return result

    def apply_business_logic(value):
      return access_database(value / 2)

    def access_database(value):
      data = fetch_record(value)
      return Chain(data).then(process_record)()

    def fetch_record(record_id):
      return {"id": record_id, "data": "test"}

    def process_record(record):
      return record["missing_field"]

    def format_output(value):
      return str(value)

    try:
      process_data(42)
      self.fail('Expected KeyError')
    except KeyError as exc:
      filenames = _get_tb_filenames(exc)
      func_names = _get_tb_func_names(exc)
      self.assertIn('<quent>', filenames,
                    'Traceback should contain <quent> visualization frame')
      self.assertIn('process_record', func_names,
                    'User function process_record should be in traceback')

  def test_lambda_exception_raises_correct_type(self):
    def start_processing(x):
      return Chain(x).then(step_one)()

    def step_one(x):
      result = (Chain(x)
                .then(lambda v: v * 2)
                .then(step_two)
                .then(lambda v: step_three(v + 100)))()
      return result

    def step_two(x):
      return x / 5

    def step_three(x):
      chain = (Chain(x)
               .then(lambda v: v ** 0.5)
               .then(lambda v: process_value(v))
               .then(lambda v: v["key"]))()
      return chain

    def process_value(x):
      return x * 1000

    with self.assertRaises(TypeError):
      start_processing(50)

  def test_custom_exception_in_chain(self):
    class DataValidationError(Exception):
      pass

    def process_request(request_id):
      return Chain(request_id).then(load_request)()

    def load_request(req_id):
      data = {"id": req_id, "amount": 1000}
      return Chain(data).then(validate_request).then(authorize_request)()

    def validate_request(data):
      if data["amount"] > 500:
        return Chain(data).then(check_special_rules)()
      return data

    def check_special_rules(data):
      result = apply_rule_engine(data)
      return Chain(result).then(verify_compliance)()

    def apply_rule_engine(data):
      data["risk_score"] = 85
      return data

    def verify_compliance(data):
      if data["risk_score"] > 80:
        raise DataValidationError(
          f"High risk transaction: score={data['risk_score']}"
        )
      return data

    def authorize_request(data):
      return Chain(data).then(final_check)()

    def final_check(data):
      raise DataValidationError("Authorization system offline")

    try:
      process_request(12345)
      self.fail('Expected DataValidationError')
    except DataValidationError as exc:
      filenames = _get_tb_filenames(exc)
      self.assertIn('<quent>', filenames,
                    'Traceback should contain a <quent> visualization frame')
      self.assertIn('High risk transaction', str(exc))
      # Verify no helper/custom frames leaked through
      for fn in filenames:
        self.assertNotIn(
          'quent/helpers', fn,
          f'Quent helpers frame should be cleaned: {fn}'
        )

  def test_custom_exception_preserves_message(self):
    class AppError(Exception):
      pass

    def do_work(x):
      return Chain(x).then(fail)()

    def fail(x):
      raise AppError("something went wrong")

    try:
      do_work(1)
      self.fail('Expected AppError')
    except AppError as exc:
      self.assertEqual(str(exc), "something went wrong")
      filenames = _get_tb_filenames(exc)
      func_names = _get_tb_func_names(exc)
      self.assertIn('<quent>', filenames)
      self.assertIn('fail', func_names)
