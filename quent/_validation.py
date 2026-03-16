# SPDX-License-Identifier: MIT
"""Validation helpers for pipeline builder methods."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Executor
from typing import Any

from ._types import QuentException


def _require_callable(v: Any, method: str, chain: Any = None) -> None:
  """Raise TypeError if *v* is not callable."""
  if not callable(v):
    chain_suffix = (
      f' (in chain {chain._name!r})' if chain is not None and getattr(chain, '_name', None) is not None else ''
    )
    msg = f'{method}() requires a callable, got {type(v).__name__}{chain_suffix}'
    raise TypeError(msg)


def _validate_concurrency(concurrency: int | None, method: str, chain: Any = None) -> None:
  """Validate the concurrency parameter for iteration and gather operations."""
  if concurrency is not None:
    chain_suffix = (
      f' (in chain {chain._name!r})' if chain is not None and getattr(chain, '_name', None) is not None else ''
    )
    if isinstance(concurrency, bool) or not isinstance(concurrency, int):
      msg = (
        f'{method}() concurrency must be a positive integer or -1 (unbounded), '
        f'got {type(concurrency).__name__}{chain_suffix}'
      )
      raise TypeError(msg)
    if concurrency != -1 and concurrency < 1:
      msg = f'{method}() concurrency must be -1 (unbounded) or a positive integer, got {concurrency}{chain_suffix}'
      raise ValueError(msg)


def _validate_executor(executor: Executor | None, method: str) -> None:
  """Validate the executor parameter for concurrent operations."""
  if executor is not None and not isinstance(executor, Executor):
    msg = f'{method}() executor must be a concurrent.futures.Executor instance, got {type(executor).__name__}'
    raise TypeError(msg)


def _normalize_exception_types(
  exc_types: Any,
  method: str,
  default: tuple[type[BaseException], ...] | None = None,
) -> tuple[type[BaseException], ...]:
  """Validate and normalize exception types to a tuple of BaseException subclasses."""
  if exc_types is None:
    if default is not None:
      return default
    msg = f'{method}() requires exception types'
    raise TypeError(msg)
  if isinstance(exc_types, str):
    msg = f"{method}() expects exception types, not string '{exc_types}'"
    raise TypeError(msg)
  if isinstance(exc_types, type):
    if not issubclass(exc_types, BaseException):
      msg = f'{method}() expects exception types (subclasses of BaseException), got {exc_types!r}'
      raise TypeError(msg)
    return (exc_types,)
  if isinstance(exc_types, Iterable):
    result = tuple(exc_types)
    if not result:
      msg = f'{method}() requires at least one exception type when exceptions is provided.'
      raise QuentException(msg)
    for exc_type in result:
      if not isinstance(exc_type, type) or not issubclass(exc_type, BaseException):
        msg = f'{method}() expects exception types (subclasses of BaseException), got {exc_type!r}'
        raise TypeError(msg)
    return result
  msg = f'{method}() expects exception types (subclasses of BaseException), got {exc_types!r}'
  raise TypeError(msg)
