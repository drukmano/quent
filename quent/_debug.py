# SPDX-License-Identifier: MIT
"""Debug infrastructure — execution tracing via Q.debug()."""

from __future__ import annotations

import asyncio
import io
import sys
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

_T = TypeVar('_T')

# Lazily created on first use to avoid circular import (_debug → _q → _debug).
_DebugQ: type | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
  """A single captured step from pipeline execution."""

  step_name: str
  input_value: Any
  result: Any
  elapsed_ns: int
  exception: BaseException | None = None

  @property
  def ok(self) -> bool:
    return self.exception is None


@dataclass(slots=True)
class DebugResult(Generic[_T]):
  """Result of a debug execution, capturing value and step trace."""

  value: _T
  steps: list[StepRecord] = field(default_factory=list)
  elapsed_ns: int = 0

  @property
  def succeeded(self) -> bool:
    return all(s.ok for s in self.steps)

  @property
  def failed(self) -> bool:
    return any(not s.ok for s in self.steps)

  def print_trace(self, file: Any = None) -> None:
    """Print the trace table. file defaults to sys.stderr."""
    if file is None:
      file = sys.stderr
    buf = io.StringIO()
    _render_trace(buf, self.steps, self.elapsed_ns)
    file.write(buf.getvalue())


def _truncate(value: Any, max_len: int = 60) -> str:
  try:
    s = repr(value)
  except Exception:
    s = f'<{type(value).__name__}>'
  if len(s) > max_len:
    return s[: max_len - 3] + '...'
  return s


def _format_elapsed(ns: int) -> str:
  if ns < 1_000:
    return f'{ns}ns'
  if ns < 1_000_000:
    return f'{ns / 1_000:.1f}us'
  if ns < 1_000_000_000:
    return f'{ns / 1_000_000:.1f}ms'
  return f'{ns / 1_000_000_000:.2f}s'


def _render_trace(buf: io.StringIO, steps: list[StepRecord], total_ns: int) -> None:
  headers = ('#', 'Step', 'Input', 'Result', 'Elapsed', 'Status')

  rows: list[tuple[str, ...]] = []
  for i, s in enumerate(steps):
    status = 'OK' if s.ok else 'FAIL'
    result_str = _truncate(s.result) if s.ok else _truncate(s.exception)
    rows.append(
      (
        str(i),
        s.step_name,
        _truncate(s.input_value),
        result_str,
        _format_elapsed(s.elapsed_ns),
        status,
      )
    )

  widths = [len(h) for h in headers]
  for row in rows:
    for j, cell in enumerate(row):
      widths[j] = max(widths[j], len(cell))

  sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

  buf.write(sep + '\n')
  header_line = '|'
  for j, h in enumerate(headers):
    header_line += ' ' + h.ljust(widths[j]) + ' |'
  buf.write(header_line + '\n')
  buf.write(sep + '\n')

  for row in rows:
    line = '|'
    for j, cell in enumerate(row):
      line += ' ' + cell.ljust(widths[j]) + ' |'
    buf.write(line + '\n')

  buf.write(sep + '\n')
  buf.write(f'Total: {_format_elapsed(total_ns)}\n')


def _on_step_recorder(
  q: Any,
  step_name: str,
  input_value: Any,
  result: Any,
  elapsed_ns: int,
  exception: BaseException | None,
) -> None:
  q._debug_steps.append(
    StepRecord(
      step_name=step_name,
      input_value=input_value,
      result=result,
      elapsed_ns=elapsed_ns,
      exception=exception,
    )
  )


def _get_debug_q_cls() -> type:
  global _DebugQ
  if _DebugQ is not None:
    return _DebugQ

  from ._q import Q

  class _DC(Q):  # type: ignore[type-arg]
    """Q subclass with independent on_step — engine reads type(q).on_step."""

    __slots__ = ('_debug_steps',)
    on_step = staticmethod(_on_step_recorder)  # type: ignore[assignment]

  _DebugQ = _DC
  return _DC


def _make_debug_q(q: Any) -> Any:
  """Clone into a _DebugQ with built-in step capture; original is not modified."""
  from ._q import Q

  dc_cls = _get_debug_q_cls()
  cloned = q.clone()
  dc: Any = object.__new__(dc_cls)
  for slot in Q.__slots__:
    setattr(dc, slot, getattr(cloned, slot))
  dc._debug_steps = []
  return dc


def _debug_run(
  q: Any, v: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> DebugResult[Any] | Coroutine[Any, Any, DebugResult[Any]]:
  """Run a debug pipeline; wraps the result (sync or async) in DebugResult."""
  t0 = time.perf_counter_ns()

  raw = q.run(v, *args, **kwargs)

  if asyncio.iscoroutine(raw):

    async def _await_and_wrap() -> DebugResult[Any]:
      val = await raw
      elapsed = time.perf_counter_ns() - t0
      return DebugResult(
        value=val,
        steps=list(q._debug_steps),
        elapsed_ns=elapsed,
      )

    return _await_and_wrap()

  elapsed = time.perf_counter_ns() - t0
  return DebugResult(
    value=raw,
    steps=list(q._debug_steps),
    elapsed_ns=elapsed,
  )
