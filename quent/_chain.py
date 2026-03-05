"""Chain and FrozenChain: the primary execution engine."""
from __future__ import annotations

from typing import Any, NoReturn
from collections.abc import Callable, Iterable
import warnings
import functools
import collections.abc
from inspect import isawaitable

from ._core import (
  Null, _evaluate_value, _resolve_value,
  _handle_return_exc, _handle_break_exc,
  _ControlFlowSignal, _Return, _Break, QuentException,
  Link, _ensure_future,
)
from ._ops import _Generator, _make_with, _make_foreach, _make_filter, _make_gather
from ._traceback import _modify_traceback


async def _await_run(result: Any, chain: Chain | None = None, link: Link | None = None, root_link: Link | None = None) -> Any:
  """Await a coroutine with traceback support for fire-and-forget paths.

  Used when a sync handler (except/finally) returns a coroutine that gets
  scheduled as a task. Ensures the traceback is properly modified if the
  coroutine raises.
  """
  try:
    return await result
  except BaseException as exc:
    raise _modify_traceback(exc, chain, link, root_link)


def _except_handler_body(exc: BaseException, chain: Chain, link: Link, root_link: Link | None) -> Any:
  """Shared except handler logic: modify traceback, evaluate handler, return result."""
  _modify_traceback(exc, chain, link, root_link)
  if chain.on_except_link is None or not isinstance(exc, chain.on_except_exceptions):
    raise exc
  try:
    result = _evaluate_value(chain.on_except_link, exc)
  except _ControlFlowSignal:
    raise QuentException('Using control flow signals inside except handlers is not allowed.')
  except BaseException as exc_:
    _modify_traceback(exc_, chain, chain.on_except_link, root_link)
    raise exc_ from exc
  return result


def _finally_handler_body(chain: Chain, root_value: Any, root_link: Link | None) -> Any:
  """Shared finally handler logic: evaluate handler, return result."""
  try:
    return _evaluate_value(chain.on_finally_link, root_value)
  except _ControlFlowSignal:
    raise QuentException('Using control flow signals inside finally handlers is not allowed.')
  except BaseException as exc_:
    _modify_traceback(exc_, chain, chain.on_finally_link, root_link)
    raise exc_


class Chain:
  """Sequential pipeline that transparently bridges synchronous and asynchronous operations."""

  _is_chain = True

  __slots__ = (
    'root_link', 'first_link', 'on_finally_link', 'on_except_link',
    'current_link', 'on_except_exceptions', 'is_nested'
  )

  # TODO rename `v` globally to a more appropriate term. there must be a term
  #  for something that can be multiple things (a value, a function, or a class, for example).
  def __init__(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> None:
    self.is_nested = False
    self.root_link = Link(v, args, kwargs) if v is not Null else None
    self.first_link = None
    self.current_link = None
    self.on_finally_link = None
    self.on_except_link = None
    self.on_except_exceptions = None

  def _then(self, link: Link) -> Chain:
    if self.current_link is not None:
      self.current_link.next_link = link
      self.current_link = link
    elif self.first_link is not None:
      self.first_link.next_link = link
      self.current_link = link
    else:
      self.first_link = link
      if self.root_link is not None:
        self.root_link.next_link = link
    return self

  def _run(self, v: Any, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> Any:
    link = self.root_link
    root_link = link
    current_value = Null
    root_value = Null
    has_run_value = v is not Null
    has_root_value = has_run_value or link is not None
    ignore_finally = False
    set_initial_values = False

    try:
      if has_run_value:
        link = Link(v, args, kwargs)
        link.next_link = self.first_link
        root_link = link
      elif not has_root_value:
        link = self.first_link

      while link is not None:
        result = _evaluate_value(link, current_value)
        if isawaitable(result):
          ignore_finally = True
          return self._run_async(result, link, current_value, root_value, has_root_value, root_link)
        if not set_initial_values:
          set_initial_values = True
          if has_root_value and root_value is Null:
            root_value = result
          if current_value is Null:
            current_value = result
        if not link.ignore_result:
          current_value = result
        link = link.next_link

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      return _handle_return_exc(exc, self.is_nested)

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      if getattr(exc, '__quent_source_link__', None) is None:
        exc.__quent_source_link__ = link
      result = _except_handler_body(exc, self, link, root_link)
      if isawaitable(result):
        try:
          result = _ensure_future(_await_run(result, self, self.on_except_link, root_link))
        except RuntimeError:
          result.close()
          raise QuentException(
            'An except handler returned a coroutine but no event loop is running.'
          ) from exc
        warnings.warn(
          'An except handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning,
        )
        return result
      if result is Null:
        return None
      return result

    finally:
      if not ignore_finally and self.on_finally_link is not None:
        result = _finally_handler_body(self, root_value, root_link)
        if isawaitable(result):
          try:
            _ensure_future(_await_run(result, self, self.on_finally_link, root_link))
          except RuntimeError:
            result.close()
            raise QuentException(
              'A finally handler returned a coroutine but no event loop is running.'
            )
          warnings.warn(
            'A finally handler returned a coroutine from a synchronous execution path. '
            'It was scheduled as a fire-and-forget Task via ensure_future().',
            category=RuntimeWarning,
          )

  async def _run_async(
    self, awaitable: Any, link: Link, current_value: Any = Null, root_value: Any = Null,
    has_root_value: bool = False, root_link: Link | None = None,
  ) -> Any:
    try:
      result = await awaitable
      if has_root_value and root_value is Null:
        root_value = result
      if current_value is Null:
        current_value = result
      if not link.ignore_result:
        current_value = result
      link = link.next_link
      while link is not None:
        result = _evaluate_value(link, current_value)
        if isawaitable(result):
          result = await result
        if not link.ignore_result:
          current_value = result
        link = link.next_link

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      result = _handle_return_exc(exc, self.is_nested)
      if isawaitable(result):
        return await result
      return result

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      result = _except_handler_body(exc, self, link, root_link)
      if isawaitable(result):
        result = await result
      if result is Null:
        return None
      return result

    finally:
      if self.on_finally_link is not None:
        result = _finally_handler_body(self, root_value, root_link)
        if isawaitable(result):
          await result

  def decorator(self) -> Callable[..., Callable[..., Any]]:
    """Wrap the chain as a function decorator."""
    chain = self
    def _decorator(fn):
      @functools.wraps(fn)
      def _wrapper(*args, **kwargs):
        try:
          return chain._run(fn, args, kwargs)
        except _ControlFlowSignal:
          # TODO is that even possible?
          raise QuentException('A control flow signal escaped the chain.') from None
      return _wrapper
    return _decorator

  def run(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> Any:
    """Execute the chain and return the result."""
    try:
      return self._run(v, args, kwargs)
    except _ControlFlowSignal:
      # TODO is that even possible?
      raise QuentException('A control flow signal escaped the chain.') from None

  def then(self, v: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Append a step. The result replaces the current chain value."""
    return self._then(Link(v, args, kwargs))

  def do(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Chain:
    """Append a side-effect step. The result is discarded."""
    return self._then(Link(fn, args, kwargs, ignore_result=True))

  def except_(self, fn: Any, /, *args: Any, exceptions: type[BaseException] | Iterable[type[BaseException]] | None = None, **kwargs: Any) -> Chain:
    """Register an exception handler. Receives the caught exception."""
    if self.on_except_link is not None:
      raise QuentException('You can only register one \'except\' callback.')
    if exceptions is not None:
      if isinstance(exceptions, str):
        raise TypeError(f"except_() expects exception types, not string '{exceptions}'")
      if isinstance(exceptions, collections.abc.Iterable):
        self.on_except_exceptions = tuple(exceptions)
        if not self.on_except_exceptions:
          raise QuentException('except_() requires at least one exception type when exceptions is provided.')
      else:
        self.on_except_exceptions = (exceptions,)
    else:
      self.on_except_exceptions = (Exception,)
    self.on_except_link = Link(fn, args, kwargs)
    return self

  def finally_(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Register a cleanup handler. Receives the root value."""
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = Link(fn, args, kwargs)
    return self

  def iterate(self, fn: Callable[[Any], Any] | None = None) -> _Generator:
    """Return a sync/async iterator over the chain's output."""
    return _Generator(self._run, fn, ignore_result=False)

  def iterate_do(self, fn: Callable[[Any], Any] | None = None) -> _Generator:
    """Return a sync/async iterator, discarding fn's return values."""
    return _Generator(self._run, fn, ignore_result=True)

  def foreach(self, fn: Callable[[Any], Any], /) -> Chain:
    """Apply fn to each element of the current iterable value."""
    # original_value must be the inner Link (not fn directly) so that:
    # 1) the traceback formatter can drill through via isinstance(original_value, Link)
    # 2) _set_link_temp_args keys by id(inner), which the formatter matches after drill-through
    inner = Link(fn)
    return self._then(Link(_make_foreach(inner, False), original_value=inner))

  def foreach_do(self, fn: Callable[[Any], Any], /) -> Chain:
    """Apply fn to each element as a side-effect, keeping original elements."""
    inner = Link(fn)
    return self._then(Link(_make_foreach(inner, True), original_value=inner))

  def filter(self, fn: Callable[[Any], Any], /) -> Chain:
    """Filter the current iterable, keeping elements where fn returns truthy."""
    inner = Link(fn)
    return self._then(Link(_make_filter(inner), original_value=inner))

  def gather(self, *fns: Callable[[Any], Any]) -> Chain:
    """Run multiple functions concurrently on the current value."""
    return self._then(Link(_make_gather(fns)))

  def with_(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Enter current value as context manager, run fn with the context."""
    inner = Link(fn, args, kwargs)
    return self._then(Link(_make_with(inner, False), ignore_result=False, original_value=inner))

  def with_do(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Enter current value as context manager, run fn as side-effect."""
    inner = Link(fn, args, kwargs)
    return self._then(Link(_make_with(inner, True), ignore_result=True, original_value=inner))

  def freeze(self) -> _FrozenChain:
    """Compile the chain into a frozen form.

    The chain must not be modified after freezing. Adding links, except
    handlers, or finally handlers after freeze() produces undefined behavior.
    """
    return _FrozenChain(self)

  @classmethod
  def return_(cls, v: Any = Null, /, *args: Any, **kwargs: Any) -> NoReturn:
    """Signal early return from the chain."""
    raise _Return(v, args, kwargs)

  @classmethod
  def break_(cls, v: Any = Null, /, *args: Any, **kwargs: Any) -> NoReturn:
    """Signal break from an iteration."""
    raise _Break(v, args, kwargs)

  __call__ = run

  def __bool__(self) -> bool:
    return True

  def __repr__(self) -> str:
    from ._traceback import _get_link_name, _get_obj_name
    parts = []
    if self.root_link is not None:
      parts.append(_get_obj_name(self.root_link.v))
    result = f'Chain({", ".join(parts)})'
    link = self.first_link
    while link is not None:
      result += f'.{_get_link_name(link)}(...)'
      link = link.next_link
    return result


class _FrozenChain:
  """Frozen chain: delegates to the underlying Chain.
  The chain must not be modified after freezing.
  """
  __slots__ = ('_chain',)

  def __init__(self, chain: Chain) -> None:
    self._chain = chain

  def run(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> Any:
    return self._chain.run(v, *args, **kwargs)

  __call__ = run

  def __bool__(self) -> bool:
    return True

  def __repr__(self) -> str:
    return f'Frozen({repr(self._chain)})'
