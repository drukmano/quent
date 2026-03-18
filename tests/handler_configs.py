"""Handler configurations, error injection types, and handler oracle functions for bridge testing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quent import Chain
from tests.fixtures import (
  CustomBaseError,
  Result,
  async_bad_cleanup,
  async_raise,
  sync_bad_cleanup,
  sync_raise,
)

# ---------------------------------------------------------------------------
# Except handler fixtures (1-arg convention: current_value = exc)
# ---------------------------------------------------------------------------


def sync_except_consume(exc: Any) -> str:
  return 'recovered'


async def async_except_consume(exc: Any) -> str:
  return 'recovered'


def sync_except_noop(exc: Any) -> None:
  pass


async def async_except_noop(exc: Any) -> None:
  pass


def sync_except_fails(exc: Any) -> Any:
  raise RuntimeError('handler boom')


async def async_except_fails(exc: Any) -> Any:
  raise RuntimeError('handler boom')


# ---------------------------------------------------------------------------
# Finally handler fixtures
# ---------------------------------------------------------------------------


def sync_finally_ok(rv: Any) -> None:
  pass


async def async_finally_ok(rv: Any) -> None:
  pass


# sync_bad_cleanup / async_bad_cleanup are in fixtures


# ---------------------------------------------------------------------------
# Kwargs-only handler fixtures (Rule 1: kwargs trigger explicit-args path)
# ---------------------------------------------------------------------------


def sync_except_kwargs(*, sentinel: bool = True) -> int:
  return 42


async def async_except_kwargs(*, sentinel: bool = True) -> int:
  return 42


def sync_finally_kwargs(*, sentinel: bool = True) -> None:
  pass


async def async_finally_kwargs(*, sentinel: bool = True) -> None:
  pass


# ---------------------------------------------------------------------------
# Error injection fixtures
# ---------------------------------------------------------------------------


def sync_raise_base(x: Any) -> Any:
  raise CustomBaseError('base error')


async def async_raise_base(x: Any) -> Any:
  raise CustomBaseError('base error')


def sync_return_signal(x: Any) -> Any:
  return Chain.return_(x * 100)


async def async_return_signal(x: Any) -> Any:
  return Chain.return_(x * 100)


def sync_break_signal(x: Any) -> Any:
  return Chain.break_(x * 100)


async def async_break_signal(x: Any) -> Any:
  return Chain.break_(x * 100)


# ---------------------------------------------------------------------------
# HandlerConfig and handler setup functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandlerConfig:
  """Configuration for attaching error handlers to a chain.

  The ``oracle`` callable computes the expected Result for a given
  (error_type, happy_value, partial_value) combination:
    - error_type: 'none', 'exception', 'base_exception', 'return_signal', 'break_signal'
    - happy_value: the composed oracle value when all bricks succeed
    - partial_value: the composed oracle value up to error_pos (for return_signal)
  Returns None to skip oracle checking (e.g. break_signal).
  """

  name: str
  apply: Callable[..., Any]  # (chain, is_handler_async: bool) -> None
  has_finally: bool = False  # True if this config attaches a finally_ handler
  oracle: Callable[..., Result | None] | None = None


def _apply_no_handler(chain: Any, is_handler_async: bool) -> None:
  pass


def _apply_except_consume(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_consume if is_handler_async else sync_except_consume
  chain.except_(handler)


def _apply_except_reraise(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_noop if is_handler_async else sync_except_noop
  chain.except_(handler, reraise=True)


def _apply_except_fails(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_fails if is_handler_async else sync_except_fails
  chain.except_(handler)


def _apply_finally_ok(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.finally_(handler)


def _apply_finally_fails(chain: Any, is_handler_async: bool) -> None:
  handler = async_bad_cleanup if is_handler_async else sync_bad_cleanup
  chain.finally_(handler)


def _apply_except_consume_finally_ok(chain: Any, is_handler_async: bool) -> None:
  exc_handler = async_except_consume if is_handler_async else sync_except_consume
  fin_handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.except_(exc_handler)
  chain.finally_(fin_handler)


def _apply_except_reraise_finally_ok(chain: Any, is_handler_async: bool) -> None:
  exc_handler = async_except_noop if is_handler_async else sync_except_noop
  fin_handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.except_(exc_handler, reraise=True)
  chain.finally_(fin_handler)


def _apply_except_consume_finally_fails(chain: Any, is_handler_async: bool) -> None:
  exc_handler = async_except_consume if is_handler_async else sync_except_consume
  fin_handler = async_bad_cleanup if is_handler_async else sync_bad_cleanup
  chain.except_(exc_handler)
  chain.finally_(fin_handler)


def _apply_except_with_args(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_consume if is_handler_async else sync_except_consume
  chain.except_(handler, 'injected_arg')


def _apply_except_nested_chain(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_consume if is_handler_async else sync_except_consume
  chain.except_(Chain().then(handler))


def _apply_finally_with_args(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.finally_(handler, 'injected_arg')


def _apply_finally_nested_chain(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.finally_(Chain().then(handler))


def _apply_except_kwargs(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_kwargs if is_handler_async else sync_except_kwargs
  chain.except_(handler, sentinel=True)


def _apply_finally_kwargs(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_kwargs if is_handler_async else sync_finally_kwargs
  chain.finally_(handler, sentinel=True)


# ---------------------------------------------------------------------------
# Handler config oracle functions
# ---------------------------------------------------------------------------
#
# Each oracle: (error_type, happy_value, partial_value) -> Result | None
#   error_type: 'none', 'exception', 'base_exception', 'return_signal', 'break_signal'
#   happy_value: composed oracle value (all bricks succeed)
#   partial_value: composed oracle value up to error_pos (for return_signal)
#   Returns None to skip oracle checking (e.g. break_signal).


def _oracle_no_handler(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_consume(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    # BaseException not caught by except_ (which uses Exception)
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    # _Return caught by engine before except_
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_reraise(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=RuntimeError)
  if error_type == 'base_exception':
    # BaseException not caught by except_
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_ok: no except handler, finally runs but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_fails: finally raises RuntimeError, always overrides outcome."""
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError regardless of error_type
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_consume_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_reraise_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_consume_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_reraise_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_fails_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=RuntimeError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_fails_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_with_args(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """except_with_args: handler('injected_arg') -> 'recovered' (Rule 1: args override cv)."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_nested_chain(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """except_nested_chain: Chain().then(consume) -> 'recovered'."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_with_args(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_with_args: no except, finally runs (ok) but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_nested_chain(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_nested_chain: no except, finally runs (ok) but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_kwargs(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """except_kwargs: handler(sentinel=True) -> 42 (Rule 1: kwargs override cv)."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value=42)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_kwargs(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_kwargs: no except, finally runs (ok) but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


HANDLER_CONFIGS: list[HandlerConfig] = [
  HandlerConfig(name='no_handler', apply=_apply_no_handler, oracle=_oracle_no_handler),
  HandlerConfig(name='except_consume', apply=_apply_except_consume, oracle=_oracle_except_consume),
  HandlerConfig(name='except_reraise', apply=_apply_except_reraise, oracle=_oracle_except_reraise),
  HandlerConfig(name='except_fails', apply=_apply_except_fails, oracle=_oracle_except_fails),
  HandlerConfig(name='finally_ok', apply=_apply_finally_ok, has_finally=True, oracle=_oracle_finally_ok),
  HandlerConfig(name='finally_fails', apply=_apply_finally_fails, has_finally=True, oracle=_oracle_finally_fails),
  HandlerConfig(
    name='except_consume+finally_ok',
    apply=_apply_except_consume_finally_ok,
    has_finally=True,
    oracle=_oracle_except_consume_finally_ok,
  ),
  HandlerConfig(
    name='except_reraise+finally_ok',
    apply=_apply_except_reraise_finally_ok,
    has_finally=True,
    oracle=_oracle_except_reraise_finally_ok,
  ),
  HandlerConfig(
    name='except_consume+finally_fails',
    apply=_apply_except_consume_finally_fails,
    has_finally=True,
    oracle=_oracle_except_consume_finally_fails,
  ),
  HandlerConfig(
    name='except_reraise+finally_fails',
    apply=lambda chain, is_async: chain.except_(
      async_except_noop if is_async else sync_except_noop, reraise=True
    ).finally_(async_bad_cleanup if is_async else sync_bad_cleanup),
    has_finally=True,
    oracle=_oracle_except_reraise_finally_fails,
  ),
  HandlerConfig(
    name='except_fails+finally_ok',
    apply=lambda chain, is_async: chain.except_(async_except_fails if is_async else sync_except_fails).finally_(
      async_finally_ok if is_async else sync_finally_ok
    ),
    has_finally=True,
    oracle=_oracle_except_fails_finally_ok,
  ),
  HandlerConfig(
    name='except_fails+finally_fails',
    apply=lambda chain, is_async: chain.except_(async_except_fails if is_async else sync_except_fails).finally_(
      async_bad_cleanup if is_async else sync_bad_cleanup
    ),
    has_finally=True,
    oracle=_oracle_except_fails_finally_fails,
  ),
  HandlerConfig(name='except_with_args', apply=_apply_except_with_args, oracle=_oracle_except_with_args),
  HandlerConfig(name='except_nested_chain', apply=_apply_except_nested_chain, oracle=_oracle_except_nested_chain),
  HandlerConfig(
    name='finally_with_args', apply=_apply_finally_with_args, has_finally=True, oracle=_oracle_finally_with_args
  ),
  HandlerConfig(
    name='finally_nested_chain',
    apply=_apply_finally_nested_chain,
    has_finally=True,
    oracle=_oracle_finally_nested_chain,
  ),
  HandlerConfig(name='except_kwargs', apply=_apply_except_kwargs, oracle=_oracle_except_kwargs),
  HandlerConfig(name='finally_kwargs', apply=_apply_finally_kwargs, has_finally=True, oracle=_oracle_finally_kwargs),
]


# ---------------------------------------------------------------------------
# Error injection types
# ---------------------------------------------------------------------------


ERROR_INJECTION_TYPES: list[tuple[str, Any, Any]] = [
  ('exception', sync_raise, async_raise),
  ('base_exception', sync_raise_base, async_raise_base),
  ('return_signal', sync_return_signal, async_return_signal),
  ('break_signal', sync_break_signal, async_break_signal),
]
