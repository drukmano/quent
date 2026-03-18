# SPDX-License-Identifier: MIT
"""Debug infrastructure -- execution tracing via chain.debug()."""

from __future__ import annotations

import asyncio
import io
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

_T = TypeVar('_T')

# The _DebugChain subclass is created lazily on first use to avoid
# circular imports (_debug -> _chain -> _debug).
_DebugChain: type | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
  """A single captured step from chain execution."""

  step_name: str
  input_value: Any
  result: Any
  elapsed_ns: int
  exception: BaseException | None = None

  @property
  def ok(self) -> bool:
    """True if the step completed without error."""
    return self.exception is None


@dataclass(slots=True)
class DebugResult(Generic[_T]):
  """Result of a debug execution, capturing value and step trace."""

  value: _T
  steps: list[StepRecord] = field(default_factory=list)
  elapsed_ns: int = 0

  @property
  def succeeded(self) -> bool:
    """True if all steps completed without error."""
    return all(s.ok for s in self.steps)

  @property
  def failed(self) -> bool:
    """True if any step raised an exception."""
    return any(not s.ok for s in self.steps)

  def print_trace(self, file: Any = None) -> None:
    """Print a formatted execution trace table.

    Args:
      file: Output stream (defaults to sys.stderr).
    """
    if file is None:
      file = sys.stderr
    buf = io.StringIO()
    _render_trace(buf, self.steps, self.elapsed_ns)
    file.write(buf.getvalue())


def _truncate(value: Any, max_len: int = 60) -> str:
  """Truncate a repr to max_len characters."""
  try:
    s = repr(value)
  except Exception:
    s = f'<{type(value).__name__}>'
  if len(s) > max_len:
    return s[: max_len - 3] + '...'
  return s


def _format_elapsed(ns: int) -> str:
  """Format nanoseconds as a human-readable duration."""
  if ns < 1_000:
    return f'{ns}ns'
  if ns < 1_000_000:
    return f'{ns / 1_000:.1f}us'
  if ns < 1_000_000_000:
    return f'{ns / 1_000_000:.1f}ms'
  return f'{ns / 1_000_000_000:.2f}s'


def _render_trace(buf: io.StringIO, steps: list[StepRecord], total_ns: int) -> None:
  """Render the step trace table into a StringIO buffer."""
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
  chain: Any,
  step_name: str,
  input_value: Any,
  result: Any,
  elapsed_ns: int,
  exception: BaseException | None = None,
) -> None:
  """on_step callback that appends StepRecords to the chain's capture list."""
  chain._debug_steps.append(
    StepRecord(
      step_name=step_name,
      input_value=input_value,
      result=result,
      elapsed_ns=elapsed_ns,
      exception=exception,
    )
  )


def _get_debug_chain_cls() -> type:
  """Return the _DebugChain subclass, creating it on first call."""
  global _DebugChain
  if _DebugChain is not None:
    return _DebugChain

  from ._chain import Chain

  class _DC(Chain):  # type: ignore[type-arg]
    """Chain subclass with independent on_step for debug capture.

    The engine reads on_step via type(chain).on_step, so this subclass
    gets its own instrumentation without affecting Chain.on_step.
    """

    __slots__ = ('_debug_steps',)
    on_step = staticmethod(_on_step_recorder)  # type: ignore[assignment]

  _DebugChain = _DC
  return _DC


def _make_debug_chain(chain: Any) -> Any:
  """Clone a chain into a _DebugChain with built-in step capture.

  The original chain is NOT modified -- clone() creates an independent copy.
  A _DebugChain instance is constructed and all slot values from the clone
  are transferred, so the engine reads _DebugChain.on_step instead of
  Chain.on_step.
  """
  from ._chain import Chain

  dc_cls = _get_debug_chain_cls()
  cloned = chain.clone()
  # Build a _DebugChain instance and copy all Chain slots from the clone.
  dc: Any = object.__new__(dc_cls)
  for slot in Chain.__slots__:
    setattr(dc, slot, getattr(cloned, slot))
  dc._debug_steps = []
  return dc


def _debug_run(chain: Any, v: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
  """Execute a debug chain and wrap the result in DebugResult.

  Handles both sync and async execution transparently.
  """
  from ._types import Null

  t0 = time.perf_counter_ns()

  if v is not Null:
    raw = chain.run(v, *args, **kwargs)
  else:
    raw = chain.run()

  if asyncio.iscoroutine(raw):

    async def _await_and_wrap() -> DebugResult[Any]:
      val = await raw
      elapsed = time.perf_counter_ns() - t0
      return DebugResult(
        value=val,
        steps=list(chain._debug_steps),
        elapsed_ns=elapsed,
      )

    return _await_and_wrap()

  elapsed = time.perf_counter_ns() - t0
  return DebugResult(
    value=raw,
    steps=list(chain._debug_steps),
    elapsed_ns=elapsed,
  )
