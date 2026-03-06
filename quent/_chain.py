"""Chain and FrozenChain: the primary execution engine."""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import functools
import time
import warnings
from collections.abc import Callable, Iterable
from inspect import isawaitable
from typing import Any, NoReturn

from ._core import (
  Link,
  Null,
  QuentException,
  _Break,
  _ControlFlowSignal,
  _ensure_future,
  _evaluate_value,
  _handle_return_exc,
  _Return,
  _set_link_temp_args,
)
from ._ops import _Generator, _make_filter, _make_foreach, _make_gather, _make_if, _make_with
from ._traceback import _modify_traceback


async def _await_run(
  result: Any, chain: Chain | None = None, link: Link | None = None, root_link: Link | None = None
) -> Any:
  """Await a coroutine with traceback support for fire-and-forget paths.

  Used when a sync handler (except/finally) returns a coroutine that gets
  scheduled as a task. Ensures the traceback is properly modified if the
  coroutine raises.
  """
  try:
    return await result
  except BaseException as exc:
    raise _modify_traceback(exc, chain, link, root_link) from None


def _except_handler_body(exc: BaseException, chain: Chain, link: Link, root_link: Link | None) -> Any:
  """Shared except handler logic: modify traceback, evaluate handler, return result."""
  with contextlib.suppress(Exception):
    _modify_traceback(exc, chain, link, root_link)
  if chain.on_except_link is None or not isinstance(exc, chain.on_except_exceptions):  # type: ignore[arg-type]
    raise exc
  try:
    result = _evaluate_value(chain.on_except_link, exc)
  except _ControlFlowSignal:
    raise QuentException('Using control flow signals inside except handlers is not allowed.') from None
  except BaseException as exc_:
    _set_link_temp_args(exc_, chain.on_except_link, exc=exc)
    _modify_traceback(exc_, chain, chain.on_except_link, root_link)
    raise exc_ from exc
  return result


def _finally_handler_body(chain: Chain, root_value: Any, root_link: Link | None) -> Any:
  """Shared finally handler logic: evaluate handler, return result."""
  try:
    return _evaluate_value(chain.on_finally_link, root_value)  # type: ignore[arg-type]
  except _ControlFlowSignal:
    raise QuentException('Using control flow signals inside finally handlers is not allowed.') from None
  except BaseException as exc_:
    if root_value is not Null:
      _set_link_temp_args(exc_, chain.on_finally_link, root_value=root_value)  # type: ignore[arg-type]
    with contextlib.suppress(Exception):
      _modify_traceback(exc_, chain, chain.on_finally_link, root_link)
    raise exc_


class Chain:
  """Sequential pipeline that transparently bridges synchronous and asynchronous operations."""

  # Duck-typing marker. Checked via getattr() in Link.__init__ and elsewhere
  # to detect Chain instances without importing Chain (avoids circular imports).
  _is_chain = True

  __slots__ = (
    '_retry_backoff',
    '_retry_max_attempts',
    '_retry_on',
    'current_link',
    'first_link',
    'is_nested',
    'on_except_exceptions',
    'on_except_link',
    'on_finally_link',
    'root_link',
  )

  _retry_backoff: Callable[[int], float] | float | None
  _retry_max_attempts: int | None
  _retry_on: tuple[type[BaseException], ...] | None
  current_link: Link | None
  first_link: Link | None
  is_nested: bool
  on_except_exceptions: tuple[type[BaseException], ...] | None
  on_except_link: Link | None
  on_finally_link: Link | None
  root_link: Link | None

  # `v` is intentional shorthand for "value" in the broad sense — any Python object
  #  (literal, callable, class, etc.). This generality is by design.
  def __init__(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> None:
    self.is_nested = False
    self.root_link = Link(v, args, kwargs) if v is not Null else None
    self.first_link = None
    self.current_link = None
    self.on_finally_link = None
    self.on_except_link = None
    self.on_except_exceptions = None
    self._retry_max_attempts = None
    self._retry_on = None
    self._retry_backoff = None

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
    # When the sync path discovers an awaitable and delegates to _run_async,
    # the finally block here must NOT fire — it would run before the async work
    # completes. _run_async has its own finally handling.
    ignore_finally = False
    _active_exc = None
    # One-shot flag: on the first link evaluation, capture the result as root_value
    # (for finally handlers) and initialize current_value (the pipeline value).
    set_initial_values = False
    max_attempts = self._retry_max_attempts or 1
    retry_on = self._retry_on or ()

    try:
      for _retry_attempt in range(max_attempts):
        try:
          # Reset state for each attempt
          root_link = self.root_link
          root_value = Null
          current_value = Null
          set_initial_values = False

          # When run(v) is called with a value, wrap it in a temporary Link and splice
          # it before first_link, so it becomes the root of this execution.
          if has_run_value:
            link = Link(v, args, kwargs)
            link.next_link = self.first_link
            root_link = link
          elif has_root_value:
            link = self.root_link
          else:
            link = self.first_link

          while link is not None:
            result = _evaluate_value(link, current_value)
            if isawaitable(result):
              ignore_finally = True
              return self._run_async(
                result,
                link,
                current_value,
                root_value,
                has_root_value,
                root_link,
                v,
                args,
                kwargs,
                _retry_attempt,
              )
            if not set_initial_values:
              set_initial_values = True
              if has_root_value and root_value is Null:
                root_value = result
              if current_value is Null and not link.ignore_result:
                current_value = result
            if not link.ignore_result:
              current_value = result
            link = link.next_link

          break  # Success — exit retry loop

        except _ControlFlowSignal:
          raise  # NEVER retry control flow signals
        except BaseException as exc:
          if _retry_attempt < max_attempts - 1 and isinstance(exc, retry_on):
            delay = self._get_retry_delay(_retry_attempt)
            if delay > 0:
              time.sleep(delay)
            continue
          raise  # Last attempt or non-retryable — propagate to outer except

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      return _handle_return_exc(exc, self.is_nested)

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('Chain.break_() cannot be used outside of a foreach iteration.') from None

    except BaseException as exc:
      _active_exc = exc
      # Stamp the failing link onto the exception (first-write-wins) so the
      # traceback formatter can highlight where the error originated.
      if getattr(exc, '__quent_source_link__', None) is None:
        exc.__quent_source_link__ = link  # type: ignore[attr-defined]
      if (
        current_value is not Null
        and not link.args  # type: ignore[union-attr]
        and not link.kwargs  # type: ignore[union-attr]
        and not link.is_chain  # type: ignore[union-attr]
        and not getattr(link.v, '_quent_op', None)  # type: ignore[union-attr]
      ):
        _set_link_temp_args(exc, link, current_value=current_value)  # type: ignore[arg-type]
      result = _except_handler_body(exc, self, link, root_link)  # type: ignore[arg-type]
      if isawaitable(result):
        # _ensure_future may raise RuntimeError if no event loop is running.
        # In that case, close both coroutines to avoid ResourceWarning.
        coro = _await_run(result, self, self.on_except_link, root_link)
        try:
          result = _ensure_future(coro)
        except RuntimeError:
          coro.close()
          result.close()
          raise QuentException('An except handler returned a coroutine but no event loop is running.') from exc
        warnings.warn(
          'An except handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning,
          stacklevel=2,
        )
        return result
      if result is Null:
        return None
      return result

    finally:
      if not ignore_finally and self.on_finally_link is not None:
        result = _finally_handler_body(self, root_value, root_link)
        if isawaitable(result):
          coro = _await_run(result, self, self.on_finally_link, root_link)
          try:
            _ensure_future(coro)
          except RuntimeError:
            coro.close()
            result.close()  # type: ignore[attr-defined]
            raise QuentException(
              'A finally handler returned a coroutine but no event loop is running.'
            ) from _active_exc
          warnings.warn(
            'A finally handler returned a coroutine from a synchronous execution path. '
            'It was scheduled as a fire-and-forget Task via ensure_future().',
            category=RuntimeWarning,
            stacklevel=2,
          )

  async def _run_async(
    self,
    awaitable: Any,
    link: Link,
    current_value: Any = Null,
    root_value: Any = Null,
    has_root_value: bool = False,
    root_link: Link | None = None,
    run_v: Any = Null,
    run_args: tuple[Any, ...] | None = None,
    run_kwargs: dict[str, Any] | None = None,
    retry_attempt: int = 0,
  ) -> Any:
    _active_exc = None
    max_attempts = self._retry_max_attempts or 1
    retry_on = self._retry_on or ()
    has_run_value = run_v is not Null

    try:
      for _attempt in range(retry_attempt, max_attempts):
        try:
          if _attempt == retry_attempt:
            # First time through — complete the current attempt
            result = await awaitable
            if has_root_value and root_value is Null:
              root_value = result
            if current_value is Null and not link.ignore_result:
              current_value = result
            if not link.ignore_result:
              current_value = result
            link = link.next_link  # type: ignore[assignment]
          else:
            # Retry — restart from scratch
            root_link = self.root_link
            root_value = Null
            current_value = Null
            has_root_value = has_run_value or self.root_link is not None
            if has_run_value:
              link = Link(run_v, run_args, run_kwargs)
              link.next_link = self.first_link
              root_link = link
            elif has_root_value:
              link = self.root_link  # type: ignore[assignment]
            else:
              link = self.first_link  # type: ignore[assignment]
            set_initial_values = False

          while link is not None:
            result = _evaluate_value(link, current_value)
            if isawaitable(result):
              result = await result
            if _attempt > retry_attempt and not set_initial_values:
              set_initial_values = True
              if has_root_value and root_value is Null:
                root_value = result
              if current_value is Null and not link.ignore_result:
                current_value = result
            if not link.ignore_result:
              current_value = result
            link = link.next_link  # type: ignore[assignment]

          if current_value is Null:
            return None
          return current_value

        except _ControlFlowSignal:
          raise  # NEVER retry control flow signals
        except BaseException as exc:
          if _attempt < max_attempts - 1 and isinstance(exc, retry_on):
            delay = self._get_retry_delay(_attempt)
            if delay > 0:
              await asyncio.sleep(delay)
            continue
          raise  # Last attempt or non-retryable — propagate to outer except

    except _Return as exc:
      result = _handle_return_exc(exc, self.is_nested)
      if isawaitable(result):
        return await result
      return result

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('Chain.break_() cannot be used outside of a foreach iteration.') from None

    except BaseException as exc:
      _active_exc = exc
      if getattr(exc, '__quent_source_link__', None) is None:
        exc.__quent_source_link__ = link  # type: ignore[attr-defined]
      if (
        current_value is not Null
        and not link.args
        and not link.kwargs
        and not link.is_chain
        and not getattr(link.v, '_quent_op', None)
      ):
        _set_link_temp_args(exc, link, current_value=current_value)
      result = _except_handler_body(exc, self, link, root_link)
      if isawaitable(result):
        result = await result
      if result is Null:
        return None
      return result

    finally:
      if self.on_finally_link is not None:
        try:
          rv = _finally_handler_body(self, root_value, root_link)
          if isawaitable(rv):
            try:
              await rv
            except _ControlFlowSignal:
              raise QuentException('Using control flow signals inside finally handlers is not allowed.') from None
            except BaseException as exc_:
              if root_value is not Null:
                _set_link_temp_args(exc_, self.on_finally_link, root_value=root_value)
              _modify_traceback(exc_, self, self.on_finally_link, root_link)
              raise exc_
        except BaseException as finally_exc:
          if finally_exc.__context__ is None and _active_exc is not None:
            finally_exc.__context__ = _active_exc
          raise

  def decorator(self) -> Callable[..., Callable[..., Any]]:
    """Wrap the chain as a function decorator.

    Warning: The decorator captures the chain by reference. Modifying
    the chain after calling decorator() affects the decorated function.
    Use freeze() first if you need an immutable snapshot.
    """
    chain = self

    def _decorator(fn):
      @functools.wraps(fn)
      def _wrapper(*args, **kwargs):
        try:
          return chain._run(fn, args, kwargs)
        except _ControlFlowSignal:
          # Defensive: _ControlFlowSignal should be caught inside _run, but this
          # guard prevents leaking internal signals if a future code change breaks
          # that invariant.
          raise QuentException('A control flow signal escaped the chain.') from None

      return _wrapper

    return _decorator

  def run(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> Any:
    """Execute the chain and return the result."""
    try:
      return self._run(v, args, kwargs)
    except _ControlFlowSignal:
      # Defensive: _ControlFlowSignal should be caught inside _run, but this
      # guard prevents leaking internal signals if a future code change breaks
      # that invariant.
      raise QuentException('A control flow signal escaped the chain.') from None

  def then(self, v: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Append a step. The result replaces the current chain value."""
    return self._then(Link(v, args, kwargs))

  def do(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Append a side-effect step. The result is discarded."""
    return self._then(Link(fn, args, kwargs, ignore_result=True))

  def except_(
    self,
    fn: Any,
    /,
    *args: Any,
    exceptions: type[BaseException] | Iterable[type[BaseException]] | None = None,
    **kwargs: Any,
  ) -> Chain:
    """Register an exception handler. Receives the caught exception."""
    if self.on_except_link is not None:
      raise QuentException("You can only register one 'except' callback.")
    if exceptions is not None:
      if isinstance(exceptions, str):
        raise TypeError(f"except_() expects exception types, not string '{exceptions}'")
      if isinstance(exceptions, collections.abc.Iterable):
        self.on_except_exceptions = tuple(exceptions)
        if not self.on_except_exceptions:
          raise QuentException('except_() requires at least one exception type when exceptions is provided.')
      else:
        self.on_except_exceptions = (exceptions,)
      for exc_type in self.on_except_exceptions:
        if not isinstance(exc_type, type) or not issubclass(exc_type, BaseException):
          raise TypeError(f'except_() expects exception types (subclasses of BaseException), got {exc_type!r}')
    else:
      self.on_except_exceptions = (Exception,)
    self.on_except_link = Link(fn, args, kwargs)
    return self

  def finally_(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Register a cleanup handler. Receives the root value."""
    if self.on_finally_link is not None:
      raise QuentException("You can only register one 'finally' callback.")
    self.on_finally_link = Link(fn, args, kwargs)
    return self

  def retry(
    self,
    max_attempts: int = 3,
    on: tuple[type[BaseException], ...] | type[BaseException] = (Exception,),
    backoff: Callable[[int], float] | float | None = None,
  ) -> Chain:
    """Configure retry for this chain.

    Retries the entire chain execution from scratch on failure.
    One retry config per chain (like except_/finally_).
    For per-link retry, use nested chains.

    Args:
      max_attempts: Total attempts (3 = initial + 2 retries).
      on: Exception types that trigger retry.
      backoff: None (no delay), float (flat delay in seconds),
        or callable(attempt_index) -> delay in seconds.
    """
    self._retry_max_attempts = max_attempts
    self._retry_on = on if isinstance(on, tuple) else (on,)
    self._retry_backoff = backoff
    return self

  def _get_retry_delay(self, attempt: int) -> float:
    """Compute the delay before the next retry attempt."""
    b = self._retry_backoff
    if b is None:
      return 0.0
    if callable(b):
      return b(attempt)
    return b  # flat float delay

  def iterate(self, fn: Callable[[Any], Any] | None = None) -> _Generator:
    """Return a sync/async iterator over the chain's output."""
    link = Link(fn) if fn is not None else None
    return _Generator(self._run, fn, ignore_result=False, chain=self, link=link)

  def iterate_do(self, fn: Callable[[Any], Any] | None = None) -> _Generator:
    """Return a sync/async iterator, discarding fn's return values."""
    link = Link(fn) if fn is not None else None
    return _Generator(self._run, fn, ignore_result=True, chain=self, link=link)

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

  def if_(self, predicate: Callable[..., Any], fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Conditionally apply fn if predicate(current_value) is truthy.

    If predicate returns truthy, fn is evaluated and its result replaces
    the current value. If falsy, the current value passes through unchanged.
    Both predicate and fn can be sync or async.
    """
    fn_link = Link(fn, args, kwargs)
    return self._then(Link(_make_if(Link(predicate), fn_link), original_value=fn_link))

  def else_(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain:
    """Register an else branch for the preceding if_() step.

    Must be called immediately after if_(). If the preceding if_'s
    predicate was falsy, fn is evaluated instead.
    """
    last = self.current_link if self.current_link is not None else self.first_link
    if last is None or getattr(last.v, '_quent_op', None) != 'if':
      raise QuentException('else_() can only be used immediately after if_()')
    last.v._else_link = Link(fn, args, kwargs)
    return self

  def freeze(self) -> _FrozenChain:
    """Compile the chain into a frozen form.

    The chain must not be modified after freezing. Adding links, except
    handlers, or finally handlers after freeze() produces undefined behavior.

    Note: return_() and break_() inside a frozen sub-chain do NOT propagate
    to the outer chain. The frozen boundary acts as an opaque execution scope.
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

  # Alias so Chain instances are directly callable: chain(v) == chain.run(v)
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
    return f'Frozen({self._chain!r})'
