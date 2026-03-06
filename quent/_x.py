"""Placeholder X object for lambda-free expressions in chain pipelines.

Usage:
    from quent import X

    Chain([1, 2, 3, 4])
      .filter(X % 2 == 0)    # instead of lambda x: x % 2 == 0
      .foreach(X * 10)        # instead of lambda x: x * 10

    Chain('hello')
      .then(X.upper())        # instead of lambda x: x.upper()
      .then(X[0])             # instead of lambda x: x[0]

Note: X.method(single_arg) is ambiguous with runtime replay and will be
treated as replay. Use a lambda for single-argument method calls:
    .then(lambda x: x.replace('a'))  # instead of X.replace('a')
Multi-argument and zero-argument method calls work fine:
    .then(X.strip())           # works (0 args)
    .then(X.replace('a', 'b')) # works (2 args)
"""

from __future__ import annotations

import operator
from typing import Any

# Operator symbols for repr display
_OP_SYMBOLS: dict[Any, str] = {
  operator.add: '+',
  operator.sub: '-',
  operator.mul: '*',
  operator.truediv: '/',
  operator.floordiv: '//',
  operator.mod: '%',
  operator.pow: '**',
  operator.and_: '&',
  operator.or_: '|',
  operator.xor: '^',
  operator.lshift: '<<',
  operator.rshift: '>>',
  operator.eq: '==',
  operator.ne: '!=',
  operator.lt: '<',
  operator.le: '<=',
  operator.gt: '>',
  operator.ge: '>=',
  operator.neg: '-',
  operator.pos: '+',
  operator.abs: 'abs',
  operator.invert: '~',
}


class _XExpr:
  """Deferred expression proxy that records operations for replay.

  Each operation returns a new _XExpr with the operation appended.
  When called with a value, replays all recorded operations on it.
  """

  __slots__ = ('_ops',)
  _ops: tuple[tuple[str, Any], ...]

  def __init__(self, ops: tuple[tuple[str, Any], ...] = ()) -> None:
    object.__setattr__(self, '_ops', ops)

  def _chain(self, op_type: str, op_args: Any) -> _XExpr:
    return _XExpr((*self._ops, (op_type, op_args)))

  def __call__(self, value: Any) -> Any:
    """Replay all recorded operations on the given value."""
    result: Any = value
    for op_type, op_args in self._ops:
      if op_type == 'attr':
        result = getattr(result, op_args)
      elif op_type == 'item':
        result = result[op_args]
      elif op_type == 'call':
        call_args, call_kwargs = op_args
        result = result(*call_args, **call_kwargs)
      elif op_type == 'binop':
        fn, other = op_args
        result = fn(result, other)
      elif op_type == 'rbinop':
        fn, other = op_args
        result = fn(other, result)
      elif op_type == 'unop':
        result = op_args(result)
    return result

  # --- Attribute access ---
  def __getattr__(self, name: str) -> _XAttr:
    if name.startswith('_'):
      raise AttributeError(name)
    return _XAttr((*self._ops, ('attr', name)))

  def __setattr__(self, name: str, value: Any) -> None:
    raise AttributeError('Cannot set attributes on X expressions')

  # --- Containment / iteration guards ---
  def __contains__(self, item: Any) -> bool:
    raise TypeError(
      'Cannot use `in` operator on X expressions. '
      'Use a lambda instead: lambda x: item in x'
    )

  def __iter__(self) -> Any:
    raise TypeError(
      'Cannot iterate over X expressions. '
      'Use a lambda instead: lambda x: iter(x)'
    )

  # --- Copy / pickle guards ---
  def __copy__(self) -> _XExpr:
    raise TypeError('X expressions cannot be copied. Construct a new expression instead.')

  def __deepcopy__(self, memo: Any) -> _XExpr:
    raise TypeError('X expressions cannot be deep-copied. Construct a new expression instead.')

  def __reduce__(self) -> tuple[Any, ...]:
    raise TypeError('X expressions cannot be pickled. Construct a new expression instead.')

  # --- Item access ---
  def __getitem__(self, key: Any) -> _XExpr:
    return self._chain('item', key)

  # --- Repr ---
  def __repr__(self) -> str:
    if not self._ops:
      return 'X'
    parts: list[str] = ['X']
    for op_type, op_args in self._ops:
      if op_type == 'attr':
        parts.append(f'.{op_args}')
      elif op_type == 'item':
        parts.append(f'[{op_args!r}]')
      elif op_type == 'call':
        call_args, call_kwargs = op_args
        arg_parts = [repr(a) for a in call_args]
        arg_parts.extend(f'{k}={v!r}' for k, v in call_kwargs.items())
        parts.append(f'({", ".join(arg_parts)})')
      elif op_type == 'binop':
        fn, other = op_args
        sym = _OP_SYMBOLS.get(fn, fn.__name__)
        parts = [f'({"".join(parts)} {sym} {other!r})']
      elif op_type == 'rbinop':
        fn, other = op_args
        sym = _OP_SYMBOLS.get(fn, fn.__name__)
        parts = [f'({other!r} {sym} {"".join(parts)})']
      elif op_type == 'unop':
        sym = _OP_SYMBOLS.get(op_args, op_args.__name__)
        parts = [f'{sym}({"".join(parts)})']
    return ''.join(parts)

  # --- Arithmetic operators ---
  def __add__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.add, other))

  def __radd__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.add, other))

  def __sub__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.sub, other))

  def __rsub__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.sub, other))

  def __mul__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.mul, other))

  def __rmul__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.mul, other))

  def __truediv__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.truediv, other))

  def __rtruediv__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.truediv, other))

  def __floordiv__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.floordiv, other))

  def __rfloordiv__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.floordiv, other))

  def __mod__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.mod, other))

  def __rmod__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.mod, other))

  def __pow__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.pow, other))

  def __rpow__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.pow, other))

  # --- Bitwise operators ---
  def __and__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.and_, other))

  def __rand__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.and_, other))

  def __or__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.or_, other))

  def __ror__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.or_, other))

  def __xor__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.xor, other))

  def __rxor__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.xor, other))

  def __lshift__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.lshift, other))

  def __rlshift__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.lshift, other))

  def __rshift__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.rshift, other))

  def __rrshift__(self, other: Any) -> _XExpr:
    return self._chain('rbinop', (operator.rshift, other))

  # --- Comparison operators ---
  def __eq__(self, other: Any) -> _XExpr:  # type: ignore[override]
    return self._chain('binop', (operator.eq, other))

  def __ne__(self, other: Any) -> _XExpr:  # type: ignore[override]
    return self._chain('binop', (operator.ne, other))

  def __lt__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.lt, other))

  def __le__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.le, other))

  def __gt__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.gt, other))

  def __ge__(self, other: Any) -> _XExpr:
    return self._chain('binop', (operator.ge, other))

  # --- Unary operators ---
  def __neg__(self) -> _XExpr:
    return self._chain('unop', operator.neg)

  def __pos__(self) -> _XExpr:
    return self._chain('unop', operator.pos)

  def __abs__(self) -> _XExpr:
    return self._chain('unop', operator.abs)

  def __invert__(self) -> _XExpr:
    return self._chain('unop', operator.invert)

  # __bool__ intentionally NOT overridden — must return bool
  # __hash__ is implicitly None because __eq__ is overridden — _XExpr instances are unhashable


class _XAttr(_XExpr):
  """An _XExpr whose last operation is an attribute access.

  Handles the dual role of attribute access and method calls:
  - X.attr used in a chain: called with (value,) -> replays as getattr(value, attr)
  - X.method(): called with () -> records a zero-arg method call, returns _XExpr
  - X.method(a, b): called with (a, b) -> records a method call, returns _XExpr
  - X.method(a): AMBIGUOUS — treated as chain replay (getattr(value, method))
    Use a lambda for single-argument method calls.
  """

  __slots__ = ()

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if len(args) == 1 and not kwargs:
      # Treat as runtime replay (chain engine calling with current_value)
      return super().__call__(args[0])
    if not args and not kwargs:
      # Build-time: X.method() — record zero-arg call
      return _XExpr((*self._ops, ('call', ((), {}))))
    # Build-time: X.method(a, b, ...) or X.method(k=v) — record method call
    return _XExpr((*self._ops, ('call', (args, kwargs))))


# Module-level singleton
X: _XExpr = _XExpr()
