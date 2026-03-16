# SPDX-License-Identifier: MIT
"""Chain class -- pipeline definition, fluent API, and reuse utilities."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from concurrent.futures import Executor
from typing import Any, ClassVar, Generic, NoReturn, ParamSpec, TypeVar, overload

from ._engine import _run, _user_stacklevel
from ._gather_ops import _make_gather
from ._generator import ChainIterator
from ._if_ops import _IfOp
from ._iter_ops import _make_iter_op
from ._link import _CLONE_SENTINEL, Link, _clone_link
from ._types import (
  Null,
  QuentException,
  _Break,
  _ControlFlowSignal,
  _Return,
  _UnpicklableMixin,
)
from ._validation import _normalize_exception_types, _require_callable, _validate_concurrency, _validate_executor
from ._viz import _stringify_chain, _VizContext
from ._with_ops import _WithOp

if sys.version_info >= (3, 11):
  from typing import Self
else:
  from typing_extensions import Self

_P = ParamSpec('_P')
_R = TypeVar('_R')
T = TypeVar('T')
U = TypeVar('U')


# ---- Chain ----


class Chain(Generic[T], _UnpicklableMixin):
  """A sequential pipeline that transparently bridges sync and async execution.

  Chain is the core primitive of quent. It models a pipeline as a singly-linked
  list of Link nodes, where each link holds a callable (or value) and
  its arguments. The pipeline is built fluently via methods like .then(),
  .do(), .foreach(), etc., and executed via .run().

  **Sync/async bridging.** Execution always starts synchronously in
  _run() (``_engine.py``). After each link evaluation, if the result is
  awaitable, control immediately delegates to _run_async(), which picks
  up where the sync path left off. The caller sees either a plain value or a
  coroutine -- no ceremony required.

  **Single except/finally per chain.** This is enforced at registration
  time. It keeps the execution model simple and predictable. For per-link error
  handling, compose nested chains.

  **Thread safety (including free-threaded / no-GIL Python).** A fully
  constructed chain is safe to execute concurrently from multiple threads,
  including under free-threaded Python (PEP 703, ``python3.14t``).  Execution
  uses only function-local state; the chain's linked-list structure is never
  mutated after construction.

  **Caveats:** :attr:`on_step` is a class-level attribute — set it
  before any concurrent chain execution begins; mutating it while chains
  are running is a data race under free-threaded Python.

  Chain-*building* methods (``.then()``, ``.do()``, ``.except_()``, etc.)
  mutate the chain and are **not** thread-safe — always build chains in a
  single thread before sharing them.  Once ``.run()`` is called, the chain
  must not be further modified from any thread.

  Example::

    result = Chain(fetch_data).then(validate).do(log).run(url)
  """

  # Duck-typing marker used by Link.__init__ to detect Chain instances
  # without circular imports.
  _quent_is_chain = True

  # Optional class-level callback for chain execution instrumentation.
  # When set, called after each step with (chain, step_name, input_value, result, elapsed_ns).
  # Zero overhead when None.
  # Class-level only (intentional): use the chain argument to dispatch per-instance.
  #
  # Error handling: if the callback raises, the exception is logged at WARNING
  # level and emitted as a RuntimeWarning, then swallowed — chain execution
  # continues uninterrupted.  This ensures instrumentation bugs never break
  # the pipeline.  Monitor the 'quent' logger at WARNING level to detect
  # callback failures.
  #
  # Thread safety: ``on_step`` must be set *before* any chain execution begins
  # (i.e., at initialization time).  Mutating ``on_step`` while chains are
  # running concurrently is a data race under free-threaded Python (PEP 703).
  on_step: ClassVar[Callable[[Chain[Any], str, Any, Any, int], None] | None] = None

  __slots__ = (
    '_if_predicate_link',
    '_name',
    '_pending_if',
    'current_link',
    'first_link',
    'on_except_exceptions',
    'on_except_link',
    'on_except_reraise',
    'on_finally_link',
    'root_link',
  )

  _if_predicate_link: Link | None
  _name: str | None
  _pending_if: bool
  current_link: Link | None
  first_link: Link | None
  on_except_exceptions: tuple[type[BaseException], ...] | None
  on_except_link: Link | None
  on_except_reraise: bool
  on_finally_link: Link | None
  root_link: Link | None

  # ---- Construction ----

  @overload
  def __init__(self, v: Callable[..., Awaitable[T]], /, *args: Any, **kwargs: Any) -> None: ...
  @overload
  def __init__(self, v: Callable[..., T], /, *args: Any, **kwargs: Any) -> None: ...
  @overload
  def __init__(self, v: T, /) -> None: ...
  @overload
  def __init__(self) -> None: ...

  def __init__(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]  # mypy limitation: overload impl with positional-only default
    """Create a new chain.

    ``v`` is an optional root value or callable. When ``v`` is callable and
    ``args``/``kwargs`` are provided, they are passed when ``v`` is invoked.

    - ``Chain()`` — creates an empty chain with no root value.
    - ``Chain(value)`` — sets a root value that seeds the pipeline.
    - ``Chain(fn, *args)`` — sets a root callable with arguments.

    Type inference:
      - ``Chain(fn)`` where ``fn: () -> R`` → ``Chain[R]`` (callable return type)
      - ``Chain(5)`` → ``Chain[int]`` (literal value type)
      - ``Chain()`` → ``Chain[Any]`` (unbound)
    """
    self._name = None
    self.root_link = Link(v, args, kwargs) if v is not Null else None
    self.first_link = None
    self.current_link = None
    self.on_finally_link = None
    self.on_except_link = None
    self.on_except_exceptions = None
    self.on_except_reraise = False
    self._pending_if = False
    self._if_predicate_link = None

  # ---- Dunder methods ----

  def __repr__(self) -> str:
    ctx = _VizContext(source_link=None, link_temp_args=None)
    return _stringify_chain(self, nest_lvl=0, root_link=None, ctx=ctx)

  def __bool__(self) -> bool:
    """Always True — prevents chains from being treated as falsy in boolean contexts."""
    return True

  def __call__(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> T | Coroutine[Any, Any, T]:
    """Alias for .run(). Allows calling the chain directly."""
    return self.run(v, *args, **kwargs)

  # ---- Control flow (class methods) ----

  @classmethod
  def return_(cls, v: Any = Null, /, *args: Any, **kwargs: Any) -> NoReturn:
    """Signal an early return from the chain.

    Must be used as ``return Chain.return_(value)`` so the internal
    ``_Return`` signal propagates through the call stack.

    Args:
      v: Optional return value. Follows the standard calling conventions.

    Raises:
      _Return: Always (this is the mechanism, not an error).

    Example::

        result = Chain(5).then(lambda x: Chain.return_(x * 10) if x > 0 else x).then(str).run()
        # result = 50 (str step is skipped due to early return)
    """
    raise _Return(v, args, kwargs)

  @classmethod
  def break_(cls, v: Any = Null, /, *args: Any, **kwargs: Any) -> NoReturn:
    """Signal a break from a foreach/foreach_do iteration.

    Only valid inside iteration operations. Using ``break_()`` outside
    iteration raises QuentException.

    Args:
      v: Optional break value. Follows the standard calling conventions.

    Raises:
      _Break: Always (this is the mechanism, not an error).

    Example::

        result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_(x) if x == 3 else x * 2).run()
        # result = 3
    """
    raise _Break(v, args, kwargs)

  # ---- Labeling ----

  def name(self, label: str, /) -> Self:
    """Assign a user-provided label for traceback identification.

    The label appears in chain visualizations (``Chain[label](root)``),
    exception notes, and ``repr(chain)``.  It has no effect on execution
    semantics — purely for debuggability.

    Args:
      label: A short descriptive string identifying this chain.

    Returns:
      ``self`` for fluent chaining.

    Example::

        Chain(fetch).name('auth_pipeline').then(validate).run()
    """
    self._name = label
    return self

  # ---- Pipeline building: core ----

  # Linked list structure: root_link -> first_link -> ... -> current_link (tail)
  def _then(
    self,
    v: Any,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    ignore_result: bool = False,
    original_value: Any | None = None,
  ) -> Self:
    """Construct a Link from the given arguments and append it to the pipeline. O(1) via tail pointer."""
    link = Link(v, args, kwargs, ignore_result=ignore_result, original_value=original_value)
    if self.current_link is not None:  # 3+ links: append to tail
      self.current_link.next_link = link
      self.current_link = link
    elif self.first_link is not None:  # 2nd link: first_link exists, establish tail pointer
      self.first_link.next_link = link
      self.current_link = link
    else:  # 1st link: set first_link (and wire from root if present)
      self.first_link = link
      if self.root_link is not None:
        self.root_link.next_link = link
    return self  # fluent

  @overload
  def then(self, v: Callable[..., Awaitable[U]], /, *args: Any, **kwargs: Any) -> Chain[U]: ...
  @overload
  def then(self, v: Callable[..., U], /, *args: Any, **kwargs: Any) -> Chain[U]: ...
  @overload
  def then(self, v: U, /) -> Chain[U]: ...

  def then(self, v: Any, /, *args: Any, **kwargs: Any) -> Chain[Any]:
    """Append a pipeline step whose result replaces the current value.

    ``v`` can be a callable, a literal value, or a nested Chain.
    The calling convention is determined by ``_evaluate_value`` based on
    whether explicit args are provided and whether a current value exists.

    If called immediately after ``.if_()``, this step becomes the truthy
    branch of the conditional instead of a regular pipeline step.

    Type inference:
      - ``.then(fn)`` where ``fn: (T) -> R`` → ``Chain[R]``
      - ``.then(42)`` → ``Chain[int]``

    Args:
      v: Callable, value, or nested Chain to evaluate.
      *args: Positional arguments forwarded to the callable.
      **kwargs: Keyword arguments forwarded to the callable.

    Returns:
      This chain, for fluent method chaining.

    Example::

        Chain(5).then(lambda x: x * 2).run()  # 10
    """
    if self._pending_if:
      return self._build_if_step(v, args, kwargs, ignore_result=False)
    return self._then(v, args, kwargs)

  def do(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
    """Append a side-effect step whose result is discarded.

    Unlike .then(), ``fn`` must be callable. This is enforced to
    prevent bugs where a literal value is accidentally used as a
    side-effect (which would silently do nothing).

    If called immediately after ``.if_()``, this step becomes a
    side-effect conditional branch (result discarded, current value
    passes through).

    Args:
      fn: Callable to invoke for its side-effects.
      *args: Positional arguments forwarded to *fn*.
      **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.

    Example::

        Chain(5).do(print).then(lambda x: x * 2).run()
        # prints: 5
        # returns: 10 (print's None return is discarded)
    """
    _require_callable(fn, 'do', self)
    if self._pending_if:
      return self._build_if_step(fn, args, kwargs, ignore_result=True)
    return self._then(fn, args, kwargs, ignore_result=True)

  # ---- Pipeline building: iteration and concurrency ----

  def foreach(
    self, fn: Callable[[Any], U], /, *, concurrency: int | None = None, executor: Executor | None = None
  ) -> Chain[list[U]]:
    """Apply *fn* to each element of the current iterable, collecting results.

    The current pipeline value must be iterable. Each element is passed to
    *fn*, and the list of results replaces the current value.

    Args:
      fn: Callable to apply to each element.
      concurrency: Optional maximum number of concurrent executions.
        When provided, items are processed concurrently (sync: ThreadPoolExecutor,
        async: Semaphore-limited tasks). When None (default), items are processed
        sequentially. Concurrent variants eagerly materialize the entire input
        iterable into a list before processing; do not use with infinite or
        very large iterables.
      executor: Optional executor instance for sync concurrent execution.
        When provided, this executor is used instead of creating a new
        ThreadPoolExecutor, and it is NOT shut down after use (lifecycle is
        the caller's responsibility). Only applies when concurrency is set.
        When None (default), a new ThreadPoolExecutor is created per invocation.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.

    Note:
      This method eagerly collects all results into a list. For infinite
      or very large iterables, use .iterate() for lazy evaluation or
      .break_() to limit iteration.

    Example::

        Chain([1, 2, 3]).foreach(lambda x: x ** 2).run()  # [1, 4, 9]
    """
    _require_callable(fn, 'foreach', self)
    _validate_concurrency(concurrency, 'foreach', self)
    _validate_executor(executor, 'foreach')
    # Inner Link is stored for traceback drill-through and temp arg display.
    inner = Link(fn)
    return self._then(_make_iter_op(inner, 'foreach', concurrency, executor), original_value=inner)  # type: ignore[return-value]

  def foreach_do(
    self, fn: Callable[[Any], Any], /, *, concurrency: int | None = None, executor: Executor | None = None
  ) -> Self:
    """Apply *fn* to each element as a side-effect, keeping original elements.

    Like .foreach(), but *fn*'s return values are discarded. The original
    elements are collected into the result list.

    Args:
      fn: Callable to invoke for its side-effects on each element.
      concurrency: Optional maximum number of concurrent executions.
        When provided, items are processed concurrently (sync: ThreadPoolExecutor,
        async: Semaphore-limited tasks). When None (default), items are processed
        sequentially. Concurrent variants eagerly materialize the entire input
        iterable into a list before processing; do not use with infinite or
        very large iterables.
      executor: Optional executor instance for sync concurrent execution.
        When provided, this executor is used instead of creating a new
        ThreadPoolExecutor, and it is NOT shut down after use (lifecycle is
        the caller's responsibility). Only applies when concurrency is set.
        When None (default), a new ThreadPoolExecutor is created per invocation.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.

    Note:
      This method eagerly collects all elements into a list. For infinite
      or very large iterables, use .iterate_do() for lazy evaluation
      or .break_() to limit iteration.

    Example::

        Chain([1, 2, 3]).foreach_do(print).run()
        # prints: 1, 2, 3
        # returns: [1, 2, 3] (original elements preserved)
    """
    _require_callable(fn, 'foreach_do', self)
    _validate_concurrency(concurrency, 'foreach_do', self)
    _validate_executor(executor, 'foreach_do')
    inner = Link(fn)
    return self._then(_make_iter_op(inner, 'foreach_do', concurrency, executor), original_value=inner)

  def gather(
    self, *fns: Callable[[T], Any], concurrency: int = -1, executor: Executor | None = None
  ) -> Chain[tuple[Any, ...]]:
    """Run multiple functions concurrently on the current value.

    Each function receives the current pipeline value. If any function
    returns an awaitable, all awaitables are gathered via
    ``asyncio.gather``. Results are returned in the same order as *fns*.

    When multiple gathered tasks fail, exceptions are wrapped in an
    ``ExceptionGroup``. Single failures propagate directly.

    Sync callables execute sequentially in order; concurrency only applies
    when one or more functions return awaitables (via ``asyncio.gather``).

    Args:
      *fns: Callables to run concurrently.
      concurrency: Maximum number of concurrent executions.
        ``-1`` (default) means unbounded — all functions run concurrently
        with no limit (effective concurrency equals ``len(fns)``). A positive
        integer limits how many functions run simultaneously
        (sync: ThreadPoolExecutor, async: Semaphore-limited tasks).
      executor: Optional executor instance for sync concurrent execution.
        When provided, this executor is used instead of creating a new
        ThreadPoolExecutor, and it is NOT shut down after use (lifecycle is
        the caller's responsibility). Only applies when concurrency is set.
        When None (default), a new ThreadPoolExecutor is created per invocation.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If any function is not callable.

    Example::

        Chain('hello').gather(str.upper, len).run()  # ('HELLO', 5)
    """
    for fn in fns:
      _require_callable(fn, 'gather', self)
    if concurrency is None:
      raise TypeError('gather() concurrency must be -1 or a positive integer, not None')
    _validate_concurrency(concurrency, 'gather', self)
    _validate_executor(executor, 'gather')
    return self._then(_make_gather(fns, concurrency, executor))  # type: ignore[return-value]

  # ---- Pipeline building: context managers ----

  @overload
  def with_(self, fn: Callable[..., Awaitable[U]], /, *args: Any, **kwargs: Any) -> Chain[U]: ...
  @overload
  def with_(self, fn: Callable[..., U], /, *args: Any, **kwargs: Any) -> Chain[U]: ...

  def with_(self, fn: Any, /, *args: Any, **kwargs: Any) -> Chain[Any]:
    """Enter the current value as a context manager and run *fn* with it.

    The current pipeline value is used as the context manager. *fn* is called
    with the context value (the ``__enter__`` / ``__aenter__`` result), and
    its return value replaces the current pipeline value.

    Args:
      fn: Callable to invoke with the context value.
      *args: Positional arguments forwarded to *fn*.
      **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
      This chain, for fluent method chaining.

    Example::

        Chain(open('data.txt')).with_(lambda f: f.read()).run()
        # Returns file contents; file is properly closed
    """
    _require_callable(fn, 'with_', self)
    inner = Link(fn, args, kwargs)
    return self._then(_WithOp(inner, False), original_value=inner)

  def with_do(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
    """Enter the current value as a context manager and run *fn* as a side-effect.

    Like .with_(), but *fn*'s return value is discarded. The original
    pipeline value passes through unchanged.

    Args:
      fn: Callable to invoke for its side-effects.
      *args: Positional arguments forwarded to *fn*.
      **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
      This chain, for fluent method chaining.

    Example::

        Chain(open('log.txt', 'w')).with_do(lambda f: f.write('done')).run()
        # Returns the file object (write result discarded); file is closed
    """
    _require_callable(fn, 'with_do', self)
    inner = Link(fn, args, kwargs)
    return self._then(_WithOp(inner, True), ignore_result=True, original_value=inner)

  # ---- Pipeline building: conditionals ----

  def if_(self, predicate: Any = None, /, *args: Any, **kwargs: Any) -> Self:
    """Begin a conditional branch. Must be followed by ``.then()`` or ``.do()``.

    When *predicate* is ``None``, the truthiness of the current pipeline
    value is used. When *predicate* is callable, it is invoked per the
    standard 2-rule calling convention and its result tested for truthiness.
    When *predicate* is a non-callable value, its truthiness is used
    directly.

    The next ``.then(v, ...)`` or ``.do(fn, ...)`` after ``if_()``
    becomes the truthy branch (instead of a regular pipeline step).
    Use ``.else_(v, ...)`` after the truthy branch for a falsy branch.

    Args:
      predicate: Callable, literal value, or ``None`` (uses current value).
      *args: Positional arguments forwarded to the predicate callable.
      **kwargs: Keyword arguments forwarded to the predicate callable.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      QuentException: If ``if_()`` is called while another ``if_()`` is
        already pending, or if args/kwargs are provided without a predicate.

    Example::

        Chain(5).if_(lambda x: x > 0).then(lambda x: x * 2).run()  # 10
        Chain(-5).if_(lambda x: x > 0).then(lambda x: x * 2).run()  # -5 (unchanged)
        Chain(5).if_().then(lambda x: x * 2).else_(0).run()  # 10 (truthy)
    """
    if self._pending_if:
      msg = 'if_() called while a previous if_() is still pending — add .then() or .do() first.'
      raise QuentException(msg)
    if predicate is None and (args or kwargs):
      msg = 'if_() received args/kwargs but no predicate — pass a callable or value as the first argument.'
      raise QuentException(msg)
    self._pending_if = True
    self._if_predicate_link = Link(predicate, args or None, kwargs or None) if predicate is not None else None
    return self

  def _build_if_step(
    self,
    v: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    ignore_result: bool,
  ) -> Self:
    """Consume a pending ``if_()`` and build the conditional step.

    Called by ``then()`` / ``do()`` when ``_pending_if`` is True.
    """
    self._pending_if = False
    predicate_link = self._if_predicate_link
    self._if_predicate_link = None
    fn_link = Link(v, args, kwargs, ignore_result=ignore_result)
    return self._then(_IfOp(predicate_link, fn_link), original_value=fn_link)

  def else_(self, v: Any, /, *args: Any, **kwargs: Any) -> Self:
    """Register an else branch for the preceding ``.if_().then()`` step.

    Must be called immediately after the ``.then()`` or ``.do()`` that
    follows ``.if_()``, with no other operations in between. If the
    preceding ``if_``'s predicate was falsy, *v* is evaluated instead.

    Args:
      v: Callable or value to evaluate in the else branch.
      *args: Positional arguments forwarded to *v*.
      **kwargs: Keyword arguments forwarded to *v*.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      QuentException: If the chain has no steps, a pending ``if_()`` has
        not yet been consumed, or the last step is not an ``if_()`` operation.

    Example::

        Chain(-5).if_(lambda x: x > 0).then(str).else_(abs).run()  # 5
    """
    if self._pending_if:
      msg = (
        'else_() called while if_() is still pending — '
        'add .then() or .do() between if_() and else_(). '
        'Usage: chain.if_(pred).then(v).else_(alt)'
      )
      raise QuentException(msg)
    last = self.current_link if self.current_link is not None else self.first_link
    if last is None:
      msg = (
        'else_() requires a preceding if_().then() — the chain has no steps yet. '
        'Usage: chain.if_(pred).then(v).else_(alt)'
      )
      raise QuentException(msg)
    if not isinstance(last.v, _IfOp):
      msg = (
        'else_() must follow immediately after if_().then() with no operations in between. '
        'The last operation in this chain is not if_(). '
        'Usage: chain.if_(pred).then(v).else_(alt)'
      )
      raise QuentException(msg)
    else_link = Link(v, args, kwargs)
    last.v.set_else(else_link)
    return self  # fluent

  def else_do(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
    """Register a side-effect else branch for the preceding ``.if_().then()`` step.

    Like ``.else_()``, but *fn*'s return value is discarded. The current
    pipeline value passes through unchanged when the else branch is taken.

    Must be called immediately after the ``.then()`` or ``.do()`` that
    follows ``.if_()``, with no other operations in between.

    Args:
      fn: Callable to invoke for its side-effects in the else branch.
      *args: Positional arguments forwarded to *fn*.
      **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.
      QuentException: If the chain has no steps, a pending ``if_()`` has
        not yet been consumed, or the last step is not an ``if_()`` operation.

    Example::

        Chain(-5).if_(lambda x: x > 0).then(str).else_do(print).run()
        # prints: -5
        # returns: -5 (print's return value discarded, original value passes through)
    """
    _require_callable(fn, 'else_do', self)
    if self._pending_if:
      msg = (
        'else_do() called while if_() is still pending — '
        'add .then() or .do() between if_() and else_do(). '
        'Usage: chain.if_(pred).then(v).else_do(fn)'
      )
      raise QuentException(msg)
    last = self.current_link if self.current_link is not None else self.first_link
    if last is None:
      msg = (
        'else_do() requires a preceding if_().then() — the chain has no steps yet. '
        'Usage: chain.if_(pred).then(v).else_do(fn)'
      )
      raise QuentException(msg)
    if not isinstance(last.v, _IfOp):
      msg = (
        'else_do() must follow immediately after if_().then() with no operations in between. '
        'The last operation in this chain is not if_(). '
        'Usage: chain.if_(pred).then(v).else_do(fn)'
      )
      raise QuentException(msg)
    else_link = Link(fn, args, kwargs, ignore_result=True)
    last.v.set_else(else_link)
    return self  # fluent

  # ---- Exception handling ----

  def except_(
    self,
    fn: Callable[..., Any],
    /,
    *args: Any,
    exceptions: type[BaseException] | Iterable[type[BaseException]] | None = None,
    reraise: bool = False,
    **kwargs: Any,
  ) -> Self:
    """Register an exception handler. At most one per chain.

    The handler receives a ``ChainExcInfo(exc, root_value)`` instance as its
    current value (standard 2-rule calling convention). See
    _except_handler_body() for the full calling convention.

    **Calling convention details:**

    - ``chain.except_(handler)`` → ``handler(ChainExcInfo(exc, root_value))``
    - ``chain.except_(handler, arg1, arg2)`` → ``handler(arg1, arg2)``
    - ``chain.except_(handler, key=val)`` → ``handler(key=val)``
      (kwargs-only: ``ChainExcInfo`` is **not** passed)

    When ``reraise=False`` (default), the handler's return value becomes
    the chain's result. When ``reraise=True``, the handler runs for
    side-effects only and the original exception is re-raised.

    .. warning:: Handler failure with ``reraise=True``

       When ``reraise=True`` and the handler itself raises an ``Exception``,
       the handler's exception is **discarded** — a warning is emitted and
       a note is attached to the original exception (Python 3.11+), but the
       original exception is re-raised, not the handler's. This means
       handler failures (e.g., alerting, logging) may go unnoticed unless
       the caller inspects warnings or exception notes. ``BaseException``
       subclasses (``KeyboardInterrupt``, ``SystemExit``) always propagate
       naturally regardless of ``reraise``.

    Args:
      fn: Callable exception handler.
      *args: Positional arguments forwarded to the handler.
      exceptions: Exception type(s) to catch. Defaults to ``Exception``.
      reraise: If True, re-raise the original exception after the handler.
      **kwargs: Keyword arguments forwarded to the handler.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.
      QuentException: If an except handler is already registered.

    Example::

        Chain(0).then(lambda x: 1 / x).except_(lambda e: -1).run()  # -1
    """
    _require_callable(fn, 'except_', self)
    if self.on_except_link is not None:
      msg = "You can only register one 'except' callback."
      raise QuentException(msg)
    self.on_except_exceptions = _normalize_exception_types(exceptions, 'except_', default=(Exception,))
    for exc_type in self.on_except_exceptions:
      if issubclass(exc_type, BaseException) and not issubclass(exc_type, Exception):
        warnings.warn(
          f'quent: except_() is configured to catch {exc_type.__name__}. '
          f'This may suppress critical system signals (KeyboardInterrupt, SystemExit). '
          f'Consider using Exception (the default) instead.',
          RuntimeWarning,
          stacklevel=_user_stacklevel(),
        )
        break
    self.on_except_link = Link(fn, args, kwargs)
    self.on_except_reraise = reraise
    return self  # fluent

  def finally_(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
    """Register a cleanup handler. At most one per chain.

    The finally handler always runs (success or failure) and receives
    the chain's root value -- not the current pipeline value. Its return
    value is always discarded. If it raises, that exception propagates.

    .. warning:: Exception suppression

       If the finally handler raises while an exception is already active
       (i.e., a chain step or except handler failed), the finally handler's
       exception replaces the active one.  The original exception is
       preserved in ``__context__`` (matching standard Python ``try/finally``
       semantics), but generic error handlers that do not inspect
       ``__context__`` may lose the original cause.

    Args:
      fn: Callable cleanup handler.
      *args: Positional arguments forwarded to the handler.
      **kwargs: Keyword arguments forwarded to the handler.

    Returns:
      This chain, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.
      QuentException: If a finally handler is already registered.

    Example::

        Chain(resource).then(process).finally_(cleanup).run()
        # cleanup(resource) always runs, even if process raises
    """
    _require_callable(fn, 'finally_', self)
    if self.on_finally_link is not None:
      msg = "You can only register one 'finally' callback."
      raise QuentException(msg)
    self.on_finally_link = Link(fn, args, kwargs)
    return self  # fluent

  # ---- Execution ----

  def _run(
    self,
    v: Any,
    args: tuple[Any, ...] | None,
    kwargs: dict[str, Any] | None,
    is_nested: bool = False,
  ) -> Any:
    """Delegate to the module-level execution engine.

    This thin wrapper exists so that ``_evaluate_value`` in ``_eval.py``
    can call ``v._run()`` on nested Chain instances without knowing about
    the ``_engine`` module.  All logic lives in ``_engine._run()``.

    *is_nested* controls whether ``_Return``/``_Break`` signals propagate
    up (nested) or get caught as errors (top-level).  Callers pass
    ``is_nested=True`` (from ``_evaluate_value`` for nested chains) or
    ``is_nested=False`` (from ``run()`` and ``decorator()`` for top-level
    execution).
    """
    return _run(self, v, args, kwargs, is_nested)

  def run(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> T | Coroutine[Any, Any, T]:
    """Execute the chain and return the final pipeline value.

    This is the public entry point. It delegates to _run() (in
    ``_engine.py``) and catches any escaped control flow signals
    (``_Return``, ``_Break``) that would otherwise leak into user code.

    The return type depends on whether an async transition occurred during
    execution: a plain value if everything was synchronous, or a coroutine
    if any link returned an awaitable. The caller decides whether to
    ``await``.

    When the chain contains async steps, this method returns a coroutine.
    The caller **must** ``await`` it -- unawaited coroutines skip
    ``finally_()`` handlers and leak resources (Python will emit a
    "coroutine was never awaited" warning).

    Args:
      v: Optional initial value injected into the pipeline, overriding the
        root link's value if both are present.
      *args: Positional arguments for the initial value's callable.
      **kwargs: Keyword arguments for the initial value's callable.

    Returns:
      The final pipeline value, or a coroutine if async transition
      occurred.

    Raises:
      QuentException: If a control flow signal escapes the chain (this is
        a bug in user code, e.g., ``break_()`` outside iteration).

    Example::

        Chain(lambda x: x * 2).run(5)  # 10
    """
    if self._pending_if:
      msg = (
        'run() called with a pending .if_() that was never consumed by .then() or .do(). '
        'Usage: chain.if_(pred).then(v).run()'
      )
      raise QuentException(msg)
    if v is not Null and (args or kwargs) and not callable(v):
      msg = f'run() received arguments but v is not callable (got {type(v).__name__})'
      raise TypeError(msg)
    try:
      return self._run(v, args, kwargs, is_nested=False)  # type: ignore[no-any-return]
    except _ControlFlowSignal as signal:
      msg = f'A {type(signal).__name__} signal escaped the chain via run().'
      raise QuentException(msg) from None

  # ---- Utilities and iteration ----

  def decorator(self) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap the chain as a function decorator.

    The decorated function's return value becomes the chain's input value.

    Returns:
      A decorator that wraps functions to execute through this chain.

    Example::

        @Chain().then(lambda x: x.strip()).then(str.upper).decorator()
        def get_name():
            return '  alice  '

        get_name()  # 'ALICE'
    """
    if self._pending_if:
      msg = (
        'decorator() called with a pending .if_() that was never consumed by .then() or .do(). '
        'Usage: chain.if_(pred).then(v).decorator()'
      )
      raise QuentException(msg)
    chain = self.clone()

    def _decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
      @functools.wraps(fn)
      def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        __tracebackhide__ = True
        try:
          # Thread safety: is_nested=False is passed explicitly rather than
          # relying on chain.is_nested.  The cloned chain is shared across all
          # invocations of the decorated function, so chain.is_nested is shared
          # mutable state.  The explicit parameter ensures correct behavior
          # under concurrent calls.
          return chain._run(fn, args, kwargs, is_nested=False)  # type: ignore[no-any-return]  # _run returns Any; decorator signature provides narrower _R for callers
        except _ControlFlowSignal as signal:
          msg = f'A {type(signal).__name__} signal escaped the chain via decorator().'
          raise QuentException(msg) from None

      return _wrapper

    return _decorator

  def iterate(self, fn: Callable[[Any], Any] | None = None) -> ChainIterator[Any]:
    """Return a sync/async iterator over the chain's output.

    Each element of the chain's iterable result is yielded. An optional
    *fn* transforms each element before yielding.

    Args:
      fn: Optional callable to transform each element.

    Returns:
      A ChainIterator supporting both ``for`` and ``async for``.

    Example::

        for item in Chain(range(5)).iterate(lambda x: x ** 2):
            print(item)  # 0, 1, 4, 9, 16
    """
    if self._pending_if:
      msg = (
        'iterate() called with a pending .if_() that was never consumed by .then() or .do(). '
        'Usage: chain.if_(pred).then(v).iterate()'
      )
      raise QuentException(msg)
    link = Link(fn) if fn is not None else None
    chain_run = functools.partial(_run, self)
    # Only retain chain/link references when fn is provided (needed for
    # traceback enhancement).  When fn is None, no per-item error handling
    # occurs, so the chain reference is unnecessary overhead.
    return ChainIterator(chain_run, fn, ignore_result=False, chain=self if fn is not None else None, link=link)

  def iterate_do(self, fn: Callable[[Any], Any] | None = None) -> ChainIterator[Any]:
    """Return a sync/async iterator, discarding *fn*'s return values.

    Like .iterate(), but *fn* runs as a side-effect. The original
    elements are yielded.

    Args:
      fn: Optional callable to invoke for side-effects on each element.

    Returns:
      A ChainIterator supporting both ``for`` and ``async for``.

    Example::

        for item in Chain(range(3)).iterate_do(print):
            pass  # prints 0, 1, 2; yields 0, 1, 2
    """
    if self._pending_if:
      msg = (
        'iterate_do() called with a pending .if_() that was never consumed by .then() or .do(). '
        'Usage: chain.if_(pred).then(v).iterate_do()'
      )
      raise QuentException(msg)
    link = Link(fn) if fn is not None else None
    chain_run = functools.partial(_run, self)
    return ChainIterator(chain_run, fn, ignore_result=True, chain=self if fn is not None else None, link=link)

  def clone(self) -> Self:
    """Create an independent copy of this chain for fork-and-extend patterns.

    Returns a new Chain with a cloned linked list. Callables, values,
    argument objects (args tuple elements, kwargs dict values), and handler
    callables (except/finally) are shared by reference -- only the list
    structure (Link nodes) and nested Chain instances are deep-copied.
    This allows forking a base chain and extending each fork independently.

    Returns:
      A new Chain with independent link structure.

    Example::

        base = Chain().then(validate).then(normalize)
        for_api = base.clone().then(to_json)  # base is not modified
    """
    cls = type(self)
    new = cls.__new__(cls)
    new._name = self._name
    new._pending_if = self._pending_if
    new._if_predicate_link = _clone_link(self._if_predicate_link) if self._if_predicate_link is not None else None
    new.on_except_reraise = self.on_except_reraise
    new.on_except_exceptions = self.on_except_exceptions
    new.on_except_link = _clone_link(self.on_except_link) if self.on_except_link is not None else None
    new.on_finally_link = _clone_link(self.on_finally_link) if self.on_finally_link is not None else None

    new.root_link = _clone_link(self.root_link) if self.root_link is not None else None

    if self.first_link is None:
      new.first_link = None
      new.current_link = None
    else:
      new.first_link = _clone_link(self.first_link)
      prev = new.first_link
      old_link = self.first_link.next_link
      while old_link is not None:
        cloned = _clone_link(old_link)
        prev.next_link = cloned
        prev = cloned
        old_link = old_link.next_link
      # current_link is the tail pointer; only needed when there are 2+ non-root links.
      new.current_link = prev if self.current_link is not None else None

    if new.root_link is not None and new.first_link is not None:
      new.root_link.next_link = new.first_link

    if __debug__:
      _missing = [s for s in self.__slots__ if getattr(new, s, _CLONE_SENTINEL) is _CLONE_SENTINEL]
      if _missing:
        raise QuentException(f'clone() missing slots: {_missing}')

    return new
