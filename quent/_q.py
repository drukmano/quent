# SPDX-License-Identifier: MIT
"""Q class -- pipeline definition, fluent API, and reuse utilities."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from concurrent.futures import Executor
from typing import Any, ClassVar, Generic, Literal, NoReturn, ParamSpec, TypeVar, overload

from ._context import _MISSING, _ctx_get, _ctx_set
from ._drive_gen_ops import _DriveGenOp
from ._engine import _run
from ._gather_ops import _make_gather
from ._generator import QuentIterator
from ._if_ops import _IfOp
from ._iter_ops import _foreach_identity, _make_iter_op
from ._link import _CLONE_SENTINEL, Link, _clone_link
from ._traceback import _user_stacklevel
from ._types import (
  Null,
  QuentException,
  _Break,
  _ControlFlowSignal,
  _Return,
  _UncopyableMixin,
)
from ._validation import _normalize_exception_types, _require_callable, _validate_concurrency, _validate_executor
from ._viz import _stringify_q, _VizContext
from ._while_ops import _WhileOp
from ._with_ops import _WithOp

if sys.version_info >= (3, 11):
  from typing import Self
else:
  from typing_extensions import Self

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T')
_U = TypeVar('_U')


class _SetDescriptor:
  """Descriptor enabling dual-mode Q.set().

  Instance access -- ``q.set(key)`` or ``q.set(key, value)``:
    Appends a pipeline step that stores a value under *key* in the
    execution context.  With one arg, stores the current pipeline value;
    with two, stores the explicit *value*.  Current value is unchanged
    (like ``.do()``).  Returns the pipeline for fluent chaining.

  Class access -- ``Q.set(key, value)``:
    Stores an explicit *value* under *key* in the execution context.
    Not a pipeline step -- takes effect immediately. Returns ``None``.
  """

  @overload
  def __get__(self, obj: Q[Any], objtype: type | None = None) -> Callable[..., Q[Any]]: ...
  @overload
  def __get__(self, obj: None, objtype: type) -> Callable[[str, Any], None]: ...

  def __get__(self, obj: Any, objtype: Any = None) -> Any:
    if obj is not None:
      # Instance access: q.set('key') or q.set('key', value) -> pipeline step
      def instance_set(key: str, value: Any = _MISSING) -> Any:
        if value is _MISSING:

          def _store(cv: Any = None) -> None:
            _ctx_set(key, cv)

          _store.__qualname__ = _store.__name__ = f'set({key!r})'
        else:

          def _store(cv: Any = None) -> None:
            _ctx_set(key, value)

          _store.__qualname__ = _store.__name__ = f'set({key!r}, ...)'

        return obj._then(_store, (), {}, ignore_result=True)

      return instance_set
    else:
      # Class access: Q.set('key', value) -> immediate store
      def static_set(key: str, value: Any) -> None:
        _ctx_set(key, value)

      return static_set


class _GetDescriptor:
  """Descriptor enabling dual-mode Q.get().

  Instance access -- ``q.get(key)`` or ``q.get(key, default)``:
    Appends a pipeline step that retrieves the value stored under *key*
    from the execution context. The retrieved value replaces the current
    value (like ``.then()``). Returns the pipeline for fluent chaining.

  Class access -- ``Q.get(key)`` or ``Q.get(key, default)``:
    Retrieves a value from the execution context immediately.
    Not a pipeline step.
  """

  @overload
  def __get__(self, obj: Q[Any], objtype: type | None = None) -> Callable[..., Q[Any]]: ...
  @overload
  def __get__(self, obj: None, objtype: type) -> Callable[..., Any]: ...

  def __get__(self, obj: Any, objtype: Any = None) -> Any:
    if obj is not None:
      # Instance access: q.get('key') -> pipeline step
      def instance_get(key: str, default: Any = _MISSING) -> Any:
        def _retrieve(cv: Any = None) -> Any:
          return _ctx_get(key, default)

        _retrieve.__qualname__ = _retrieve.__name__ = f'get({key!r})'
        return obj._then(_retrieve, (), {})

      return instance_get
    else:
      # Class access: Q.get('key') -> immediate retrieval
      return _ctx_get


class Q(Generic[_T], _UncopyableMixin):
  """A sequential pipeline that transparently bridges sync and async execution.

  Q is the core primitive of quent. It models a pipeline as a singly-linked
  list of Link nodes, where each link holds a callable (or value) and
  its arguments. The pipeline is built fluently via methods like .then(),
  .do(), .foreach(), etc., and executed via .run().

  **Sync/async bridging.** Execution always starts synchronously in
  _run() (``_engine.py``). After each link evaluation, if the result is
  awaitable, control immediately delegates to _run_async(), which picks
  up where the sync path left off. The caller sees either a plain value or a
  coroutine -- no ceremony required.

  **Single except/finally per pipeline.** This is enforced at registration
  time. It keeps the execution model simple and predictable. For per-link error
  handling, compose nested pipelines.

  **Thread safety (including free-threaded / no-GIL Python).** A fully
  constructed pipeline is safe to execute concurrently from multiple threads,
  including under free-threaded Python (PEP 703, ``python3.14t``).  Execution
  uses only function-local state; the pipeline's linked-list structure is never
  mutated after construction.

  **Caveats:** :attr:`on_step` is a class-level attribute — set it
  before any concurrent pipeline execution begins; mutating it while pipelines
  are running is a data race under free-threaded Python.

  Pipeline-*building* methods (``.then()``, ``.do()``, ``.except_()``, etc.)
  mutate the pipeline and are **not** thread-safe — always build pipelines in a
  single thread before sharing them.  Once ``.run()`` is called, the pipeline
  must not be further modified from any thread.

  Example::

    result = Q(fetch_data, url).then(validate).do(log).run()
  """

  # Duck-typing marker used by Link.__init__ to detect Q instances
  # without circular imports.
  _quent_is_q = True

  # Optional class-level callback for pipeline execution instrumentation.
  # When set, called after each step with
  # (q, step_name, input_value, result, elapsed_ns, exception).
  # The 6th `exception` parameter is optional for backward compatibility:
  # callbacks with 5 positional parameters are auto-detected via
  # inspect.signature and called without the exception argument.
  # On success: exception=None.  On failure: exception=<the exception>, result=None.
  # Zero overhead when None.
  # Class-level only (intentional): use the q argument to dispatch per-instance.
  #
  # Error handling: if the callback raises, the exception is logged at WARNING
  # level and emitted as a RuntimeWarning, then swallowed — pipeline execution
  # continues uninterrupted.  This ensures instrumentation bugs never break
  # the pipeline.  Monitor the 'quent' logger at WARNING level to detect
  # callback failures.
  #
  # Thread safety: ``on_step`` must be set *before* any pipeline execution begins
  # (i.e., at initialization time).  Mutating ``on_step`` while pipelines are
  # running concurrently is a data race under free-threaded Python (PEP 703).
  on_step: ClassVar[
    Callable[[Q[Any], str, Any, Any, int, BaseException | None], None]
    | Callable[[Q[Any], str, Any, Any, int], None]
    | None
  ] = None

  set = _SetDescriptor()
  get = _GetDescriptor()

  __slots__ = (
    '_buffer_size',
    '_current_link',
    '_first_link',
    '_if_predicate_link',
    '_name',
    '_on_except_exceptions',
    '_on_except_link',
    '_on_except_reraise',
    '_on_finally_link',
    '_pending_if',
    '_pending_while',
    '_root_link',
    '_while_predicate_link',
  )

  _buffer_size: int | None
  _if_predicate_link: Link | None
  _name: str | None
  _pending_if: bool
  _pending_while: bool
  _current_link: Link | None
  _first_link: Link | None
  _on_except_exceptions: tuple[type[BaseException], ...] | None
  _on_except_link: Link | None
  _on_except_reraise: bool
  _on_finally_link: Link | None
  _root_link: Link | None
  _while_predicate_link: Link | None

  # ---- Construction ----

  @overload
  def __init__(self, v: Callable[..., Awaitable[_T]], /, *args: Any, **kwargs: Any) -> None: ...
  @overload
  def __init__(self, v: Callable[..., _T], /, *args: Any, **kwargs: Any) -> None: ...
  @overload
  def __init__(self, v: _T, /) -> None: ...
  @overload
  def __init__(self) -> None: ...

  def __init__(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]  # mypy limitation: overload impl with positional-only default
    """Create a new pipeline.

    ``v`` is an optional root value or callable. When ``v`` is callable and
    ``args``/``kwargs`` are provided, they are passed when ``v`` is invoked.

    - ``Q()`` — creates an empty pipeline with no root value.
    - ``Q(value)`` — sets a root value that seeds the pipeline.
    - ``Q(fn, *args)`` — sets a root callable with arguments.

    Type inference:
      - ``Q(fn)`` where ``fn: () -> R`` → ``Q[R]`` (callable return type)
      - ``Q(5)`` → ``Q[int]`` (literal value type)
      - ``Q()`` → ``Q[Any]`` (unbound)
    """
    self._name = None
    if v is Null and (args or kwargs):
      raise TypeError('Q() keyword arguments require a root value as the first positional argument')
    self._root_link = Link(v, args, kwargs) if v is not Null else None
    self._first_link = None
    self._current_link = None
    self._on_finally_link = None
    self._on_except_link = None
    self._on_except_exceptions = None
    self._on_except_reraise = False
    self._pending_if = False
    self._if_predicate_link = None
    self._pending_while = False
    self._while_predicate_link = None
    self._buffer_size = None

  # ---- Dunder methods ----

  def __repr__(self) -> str:
    ctx = _VizContext(source_link=None, link_temp_args=None)
    return _stringify_q(self, nest_lvl=0, root_link=None, ctx=ctx)

  def __bool__(self) -> bool:
    """Always True — prevents pipelines from being treated as falsy in boolean contexts."""
    return True

  def __call__(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> _T | Coroutine[Any, Any, _T]:
    """Alias for .run(). Allows calling the pipeline directly."""
    return self.run(v, *args, **kwargs)

  # ---- Control flow (class methods) ----

  @classmethod
  def return_(cls, v: Any = Null, /, *args: Any, **kwargs: Any) -> NoReturn:
    """Signal an early return from the pipeline.

    Must be used as ``return Q.return_(value)`` so the internal
    ``_Return`` signal propagates through the call stack.

    Args:
      v: Optional return value. Follows the standard calling conventions.

    Raises:
      _Return: Always (this is the mechanism, not an error).

    Example::

        result = Q(5).then(lambda x: Q.return_(x * 10) if x > 0 else x).then(str).run()
        # result = 50 (str step is skipped due to early return)
    """
    raise _Return(v, args, kwargs)

  @classmethod
  def break_(cls, v: Any = Null, /, *args: Any, **kwargs: Any) -> NoReturn:
    """Signal a break from a foreach/foreach_do iteration or ``while_`` loop.

    Only valid inside iteration operations and while loops.
    Using ``break_()`` outside these contexts raises QuentException.

    Args:
      v: Optional break value. Follows the standard calling conventions.

    Raises:
      _Break: Always (this is the mechanism, not an error).

    Example::

        result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_(x) if x == 3 else x * 2).run()
        # result = 3
    """
    raise _Break(v, args, kwargs)

  # ---- Labeling ----

  def name(self, label: str, /) -> Self:
    """Assign a user-provided label for traceback identification.

    The label appears in pipeline visualizations (``Q[label](root)``),
    exception notes, and ``repr(q)``.  It has no effect on execution
    semantics — purely for debuggability.

    Args:
      label: A short descriptive string identifying this pipeline.

    Returns:
      ``self`` for fluent chaining.

    Example::

        Q(fetch).name('auth_pipeline').then(validate).run()
    """
    self._ensure_if_consumed()
    self._name = label
    return self

  # ---- Pipeline building: core ----

  # Linked list structure: _root_link -> _first_link -> ... -> _current_link (tail)
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
    self._ensure_if_consumed()
    link = Link(v, args, kwargs, ignore_result=ignore_result, original_value=original_value)
    if self._current_link is not None:  # 3+ links: append to tail
      self._current_link.next_link = link
      self._current_link = link
    elif self._first_link is not None:  # 2nd link: _first_link exists, establish tail pointer
      self._first_link.next_link = link
      self._current_link = link
    else:  # 1st link: set _first_link (and wire from root if present)
      self._first_link = link
      if self._root_link is not None:
        self._root_link.next_link = link
    return self  # fluent

  @overload
  def then(self, v: Q[_U], /) -> Q[_U]: ...
  @overload
  def then(self, v: Callable[..., Awaitable[_U]], /, *args: Any, **kwargs: Any) -> Q[_U]: ...
  @overload
  def then(self, v: Callable[..., _U], /, *args: Any, **kwargs: Any) -> Q[_U]: ...
  @overload
  def then(self, v: _U, /) -> Q[_U]: ...

  def then(self, v: Any, /, *args: Any, **kwargs: Any) -> Q[Any]:
    """Append a pipeline step whose result replaces the current value.

    ``v`` can be a callable, a literal value, or a nested Q.
    The calling convention is determined by ``_evaluate_value`` based on
    whether explicit args are provided and whether a current value exists.

    If called immediately after ``.if_()``, this step becomes the truthy
    branch of the conditional instead of a regular pipeline step.

    Type inference:
      - ``.then(fn)`` where ``fn: (T) -> R`` → ``Q[R]``
      - ``.then(42)`` → ``Q[int]``

    Args:
      v: Callable, value, or nested Q to evaluate.
      *args: Positional arguments forwarded to the callable.
      **kwargs: Keyword arguments forwarded to the callable.

    Returns:
      This pipeline, for fluent method chaining.

    Example::

        Q(5).then(lambda x: x * 2).run()  # 10
    """
    if self._pending_while:
      return self._build_while_step(v, args, kwargs, ignore_result=False)
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
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.

    Example::

        Q(5).do(print).then(lambda x: x * 2).run()
        # prints: 5
        # returns: 10 (print's None return is discarded)
    """
    _require_callable(fn, 'do', self)
    if self._pending_while:
      return self._build_while_step(fn, args, kwargs, ignore_result=True)
    if self._pending_if:
      return self._build_if_step(fn, args, kwargs, ignore_result=True)
    return self._then(fn, args, kwargs, ignore_result=True)

  # ---- Pipeline building: iteration and concurrency ----

  @overload
  def foreach(self, /, *, concurrency: int | None = None, executor: Executor | None = None) -> Q[list[Any]]: ...
  @overload
  def foreach(
    self, fn: Callable[[Any], _U], /, *, concurrency: int | None = None, executor: Executor | None = None
  ) -> Q[list[_U]]: ...
  def foreach(
    self, fn: Callable[[Any], _U] | None = None, /, *, concurrency: int | None = None, executor: Executor | None = None
  ) -> Q[list[_U]]:
    """Apply *fn* to each element of the current iterable, collecting results.

    The current pipeline value must be iterable. Each element is passed to
    *fn*, and the list of results replaces the current value.

    When *fn* is omitted, the identity function is used — the iterable's
    elements are collected into a list as-is. This is equivalent to
    ``foreach(lambda v: v)``.

    Args:
      fn: Callable to apply to each element. When ``None`` (default),
        elements are collected unchanged (identity).
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
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is provided and not callable.

    Note:
      This method eagerly collects all results into a list. For infinite
      or very large iterables, use .iterate() for lazy evaluation or
      .break_() to limit iteration.

    Example::

        Q([1, 2, 3]).foreach(lambda x: x ** 2).run()  # [1, 4, 9]
        Q([1, 2, 3]).foreach().run()  # [1, 2, 3]
    """
    if fn is not None:
      _require_callable(fn, 'foreach', self)
    else:
      fn = _foreach_identity
    return self._build_foreach(fn, 'foreach', concurrency, executor)  # type: ignore[return-value]

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
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.

    Note:
      This method eagerly collects all elements into a list. For infinite
      or very large iterables, use .iterate_do() for lazy evaluation
      or .break_() to limit iteration.

    Example::

        Q([1, 2, 3]).foreach_do(print).run()
        # prints: 1, 2, 3
        # returns: [1, 2, 3] (original elements preserved)
    """
    _require_callable(fn, 'foreach_do', self)
    return self._build_foreach(fn, 'foreach_do', concurrency, executor)

  def _build_foreach(
    self,
    fn: Callable[[Any], Any],
    method_name: Literal['foreach', 'foreach_do'],
    concurrency: int | None,
    executor: Executor | None,
  ) -> Self:
    """Shared implementation for foreach() and foreach_do()."""
    _validate_concurrency(concurrency, method_name, self)
    _validate_executor(executor, method_name)
    # Inner Link is stored for traceback drill-through and temp arg display.
    inner = Link(fn)
    return self._then(_make_iter_op(inner, method_name, concurrency, executor), original_value=inner)

  def gather(
    self, *fns: Callable[[_T], Any], concurrency: int = -1, executor: Executor | None = None
  ) -> Q[tuple[Any, ...]]:
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
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If any function is not callable.

    Example::

        Q('hello').gather(str.upper, len).run()  # ('HELLO', 5)
    """
    for fn in fns:
      _require_callable(fn, 'gather', self)
    if concurrency is None:
      raise TypeError('gather() concurrency must be -1 or a positive integer, not None')
    _validate_concurrency(concurrency, 'gather', self)
    _validate_executor(executor, 'gather')
    return self._then(_make_gather(fns, concurrency, executor))  # type: ignore[return-value]

  def buffer(self, n: int, /) -> Self:
    """Attach a backpressure-aware buffer of size *n* for iteration.

    When a subsequent iteration terminal (``iterate()``, ``iterate_do()``,
    ``flat_iterate()``, ``flat_iterate_do()``) consumes the pipeline's output,
    a bounded queue of size *n* is interposed between the producer (the
    pipeline's iterable result) and the consumer (the iteration loop).

    When the buffer is full, the producer blocks (backpressure).  When
    the buffer is empty, the consumer blocks.  For sync iteration a
    background thread drives the producer; for async iteration a
    background task is used.

    ``buffer()`` is a pipeline-level modifier — it does not add a pipeline
    step.  It only takes effect when the pipeline is consumed via an
    iteration terminal.  If ``run()`` is used instead, a
    ``QuentException`` is raised.

    Args:
      n: Maximum number of items the buffer can hold.  Must be a
        positive integer.

    Returns:
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *n* is not an integer.
      ValueError: If *n* is less than 1.

    Example::

        Q(produce).buffer(10).iterate()
        # Producer feeds items into a buffer of size 10.
        # Consumer reads from the buffer with backpressure.
    """
    self._ensure_if_consumed()
    if isinstance(n, bool) or not isinstance(n, int):
      msg = f'buffer() requires a positive integer, got {type(n).__name__}'
      raise TypeError(msg)
    if n < 1:
      msg = f'buffer() requires a positive integer, got {n}'
      raise ValueError(msg)
    self._buffer_size = n
    return self

  # ---- Pipeline building: context managers ----

  @overload
  def with_(self, fn: Callable[..., Awaitable[_U]], /, *args: Any, **kwargs: Any) -> Q[_U]: ...
  @overload
  def with_(self, fn: Callable[..., _U], /, *args: Any, **kwargs: Any) -> Q[_U]: ...

  def with_(self, fn: Any, /, *args: Any, **kwargs: Any) -> Q[Any]:
    """Enter the current value as a context manager and run *fn* with it.

    The current pipeline value is used as the context manager. *fn* is called
    with the context value (the ``__enter__`` / ``__aenter__`` result), and
    its return value replaces the current pipeline value.

    Args:
      fn: Callable to invoke with the context value.
      *args: Positional arguments forwarded to *fn*.
      **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
      This pipeline, for fluent method chaining.

    Example::

        Q(open('data.txt')).with_(lambda f: f.read()).run()
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
      This pipeline, for fluent method chaining.

    Example::

        Q(open('log.txt', 'w')).with_do(lambda f: f.write('done')).run()
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
      This pipeline, for fluent method chaining.

    Raises:
      QuentException: If ``if_()`` is called while another ``if_()`` is
        already pending, or if args/kwargs are provided without a predicate.

    Example::

        Q(5).if_(lambda x: x > 0).then(lambda x: x * 2).run()  # 10
        Q(-5).if_(lambda x: x > 0).then(lambda x: x * 2).run()  # -5 (unchanged)
        Q(5).if_().then(lambda x: x * 2).else_(0).run()  # 10 (truthy)
    """
    self._ensure_if_consumed()
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

  def _validate_else_precondition(self, method: str) -> Link:
    """Validate and return the last link for else_() / else_do()."""
    if self._pending_if:
      msg = (
        f'{method}() called while if_() is still pending — '
        f'add .then() or .do() between if_() and {method}(). '
        f'Usage: q.if_(pred).then(v).{method}(alt)'
      )
      raise QuentException(msg)
    last = self._current_link if self._current_link is not None else self._first_link
    if last is None:
      msg = (
        f'{method}() requires a preceding if_().then() — the pipeline has no steps yet. '
        f'Usage: q.if_(pred).then(v).{method}(alt)'
      )
      raise QuentException(msg)
    if not isinstance(last.v, _IfOp):
      msg = (
        f'{method}() must follow immediately after if_().then() with no operations in between. '
        f'The last operation in this pipeline is not if_(). '
        f'Usage: q.if_(pred).then(v).{method}(alt)'
      )
      raise QuentException(msg)
    return last

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
      This pipeline, for fluent method chaining.

    Raises:
      QuentException: If the pipeline has no steps, a pending ``if_()`` has
        not yet been consumed, or the last step is not an ``if_()`` operation.

    Example::

        Q(-5).if_(lambda x: x > 0).then(str).else_(abs).run()  # 5
    """
    last = self._validate_else_precondition('else_')
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
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.
      QuentException: If the pipeline has no steps, a pending ``if_()`` has
        not yet been consumed, or the last step is not an ``if_()`` operation.

    Example::

        Q(-5).if_(lambda x: x > 0).then(str).else_do(print).run()
        # prints: -5
        # returns: -5 (print's return value discarded, original value passes through)
    """
    _require_callable(fn, 'else_do', self)
    last = self._validate_else_precondition('else_do')
    else_link = Link(fn, args, kwargs, ignore_result=True)
    last.v.set_else(else_link)
    return self  # fluent

  # ---- Pipeline building: looping ----

  def while_(self, predicate: Any = None, /, *args: Any, **kwargs: Any) -> Self:
    """Begin a while loop. Must be followed by ``.then()`` or ``.do()``.

    When *predicate* is ``None``, the truthiness of the current pipeline
    value is used.  When *predicate* is callable, it is invoked per the
    standard 2-rule calling convention and its result tested for truthiness.
    When *predicate* is a non-callable value, its truthiness is used
    directly.

    The next ``.then(fn, ...)`` or ``.do(fn, ...)`` after ``while_()``
    becomes the loop body (instead of a regular pipeline step):

    - ``.then(fn)``: *fn*'s result feeds back as the loop value each
      iteration.  The final loop value replaces the current pipeline value.
    - ``.do(fn)``: *fn* runs for side-effects; the loop value is unchanged
      each iteration.  The current pipeline value passes through.

    ``break_()`` inside the body or predicate stops the loop.
    ``return_()`` propagates to the enclosing pipeline.

    Args:
      predicate: Callable, literal value, or ``None`` (uses current value).
      *args: Positional arguments forwarded to the predicate callable.
      **kwargs: Keyword arguments forwarded to the predicate callable.

    Returns:
      This pipeline, for fluent method chaining.

    Raises:
      QuentException: If ``while_()`` is called while another ``while_()``
        or ``if_()`` is already pending, or if args/kwargs are provided
        without a predicate.

    Example::

        Q(1).while_(lambda x: x < 10).then(lambda x: x + 1).run()  # 10
        Q(1).while_(lambda x: x < 10).do(print).run()  # prints 1 nine times, returns 1
    """
    self._ensure_if_consumed()
    if predicate is None and (args or kwargs):
      msg = 'while_() received args/kwargs but no predicate — pass a callable or value as the first argument.'
      raise QuentException(msg)
    self._pending_while = True
    self._while_predicate_link = Link(predicate, args or None, kwargs or None) if predicate is not None else None
    return self

  def _build_while_step(
    self,
    v: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    ignore_result: bool,
  ) -> Self:
    """Consume a pending ``while_()`` and build the loop step.

    Called by ``then()`` / ``do()`` when ``_pending_while`` is True.
    """
    self._pending_while = False
    predicate_link = self._while_predicate_link
    self._while_predicate_link = None
    body_link = Link(v, args, kwargs, ignore_result=ignore_result)
    return self._then(_WhileOp(predicate_link, body_link, ignore_result), original_value=body_link)

  # ---- Pipeline building: generator driving ----

  def drive_gen(self, fn: Callable[[Any], Any], /) -> Self:
    """Drive a sync/async generator with a step function.

    The current pipeline value must be a generator (or a callable that
    produces one). Each value yielded by the generator is passed to *fn*;
    *fn*'s return value is sent back into the generator via ``.send()``.
    When the generator stops, the last *fn* result becomes the pipeline value.

    Handles both sync and async generators transparently -- sync generators
    use ``next()``/``.send()``, async generators use ``__anext__()``/
    ``.asend()``. If *fn* returns an awaitable, it is awaited (standard
    async transition).

    Args:
      fn: Callable to invoke with each yielded value. Its return value is
        sent back into the generator.

    Returns:
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable, or if the pipeline value at
        execution time is not a generator.

    Example::

        def auth_flow(request):
            yield request           # yield request to driver
            response = yield        # receive response back
            yield response          # yield for further processing

        Q(auth_flow(initial_req)).drive_gen(send_request).run()
    """
    _require_callable(fn, 'drive_gen', self)
    inner = Link(fn)
    return self._then(_DriveGenOp(inner), original_value=inner)

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
    """Register an exception handler. At most one per pipeline.

    The handler receives a ``QuentExcInfo(exc, root_value)`` instance as its
    current value (standard 2-rule calling convention). See
    _except_handler_body() for the full calling convention.

    **Calling convention details:**

    - ``q.except_(handler)`` → ``handler(QuentExcInfo(exc, root_value))``
    - ``q.except_(handler, arg1, arg2)`` → ``handler(arg1, arg2)``
    - ``q.except_(handler, key=val)`` → ``handler(key=val)``
      (kwargs-only: ``QuentExcInfo`` is **not** passed)

    When ``reraise=False`` (default), the handler's return value becomes
    the pipeline's result. When ``reraise=True``, the handler runs for
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
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.
      QuentException: If an except handler is already registered.

    Example::

        Q(0).then(lambda x: 1 / x).except_(lambda e: -1).run()  # -1
    """
    self._ensure_if_consumed()
    _require_callable(fn, 'except_', self)
    if self._on_except_link is not None:
      msg = "You can only register one 'except' callback."
      raise QuentException(msg)
    self._on_except_exceptions = _normalize_exception_types(exceptions, 'except_', default=(Exception,))
    for exc_type in self._on_except_exceptions:
      if issubclass(exc_type, BaseException) and not issubclass(exc_type, Exception):
        warnings.warn(
          f'quent: except_() is configured to catch {exc_type.__name__}. '
          f'This may suppress critical system signals (KeyboardInterrupt, SystemExit). '
          f'Consider using Exception (the default) instead.',
          RuntimeWarning,
          stacklevel=_user_stacklevel(),
        )
        break
    self._on_except_link = Link(fn, args, kwargs)
    self._on_except_reraise = reraise
    return self  # fluent

  def finally_(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
    """Register a cleanup handler. At most one per pipeline.

    The finally handler always runs (success or failure) and receives
    the pipeline's root value -- not the current pipeline value. Its return
    value is always discarded. If it raises, that exception propagates.

    .. warning:: Exception suppression

       If the finally handler raises while an exception is already active
       (i.e., a pipeline step or except handler failed), the finally handler's
       exception replaces the active one.  The original exception is
       preserved in ``__context__`` (matching standard Python ``try/finally``
       semantics), but generic error handlers that do not inspect
       ``__context__`` may lose the original cause.

    Args:
      fn: Callable cleanup handler.
      *args: Positional arguments forwarded to the handler.
      **kwargs: Keyword arguments forwarded to the handler.

    Returns:
      This pipeline, for fluent method chaining.

    Raises:
      TypeError: If *fn* is not callable.
      QuentException: If a finally handler is already registered.

    Example::

        Q(resource).then(process).finally_(cleanup).run()
        # cleanup(resource) always runs, even if process raises
    """
    self._ensure_if_consumed()
    _require_callable(fn, 'finally_', self)
    if self._on_finally_link is not None:
      msg = "You can only register one 'finally' callback."
      raise QuentException(msg)
    self._on_finally_link = Link(fn, args, kwargs)
    return self  # fluent

  @staticmethod
  def _wrap_escaped_signal(signal: _ControlFlowSignal, source: str) -> QuentException:
    """Wrap a control flow signal that escaped the pipeline into a QuentException."""
    msg = f'A {type(signal).__name__} signal escaped the pipeline via {source}().'
    return QuentException(msg)

  def _ensure_if_consumed(self) -> None:
    """Raise if an if_() or while_() is pending without a matching then()/do()."""
    if self._pending_if:
      raise QuentException('if_() must be followed by .then() or .do() to register the conditional branch.')
    if self._pending_while:
      raise QuentException('while_() must be followed by .then() or .do() to register the loop body.')

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
    can call ``v._run()`` on nested Q instances without knowing about
    the ``_engine`` module.  All logic lives in ``_engine._run()``.

    *is_nested* controls whether ``_Return``/``_Break`` signals propagate
    up (nested) or get caught as errors (top-level).  Callers pass
    ``is_nested=True`` (from ``_evaluate_value`` for nested pipelines) or
    ``is_nested=False`` (from ``run()`` and ``as_decorator()`` for top-level
    execution).
    """
    return _run(self, v, args, kwargs, is_nested)

  def run(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> _T | Coroutine[Any, Any, _T]:
    """Execute the pipeline and return the final pipeline value.

    This is the public entry point. It delegates to _run() (in
    ``_engine.py``) and catches any escaped control flow signals
    (``_Return``, ``_Break``) that would otherwise leak into user code.

    The return type depends on whether an async transition occurred during
    execution: a plain value if everything was synchronous, or a coroutine
    if any link returned an awaitable. The caller decides whether to
    ``await``.

    When the pipeline contains async steps, this method returns a coroutine.
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
      QuentException: If a control flow signal escapes the pipeline (this is
        a bug in user code, e.g., ``break_()`` outside iteration).

    Example::

        Q(5).then(lambda x: x * 2).run()  # 10
    """
    self._ensure_if_consumed()
    if self._buffer_size is not None:
      raise QuentException(
        'buffer() requires an iteration terminal'
        ' (iterate, iterate_do, flat_iterate, flat_iterate_do);'
        ' run() is not supported with buffer()'
      )
    if v is Null and (args or kwargs):
      raise TypeError('run() keyword arguments require a root value as the first positional argument')
    if v is not Null and (args or kwargs) and not callable(v):
      msg = f'run() received arguments but v is not callable (got {type(v).__name__})'
      raise TypeError(msg)
    try:
      return self._run(v, args, kwargs, is_nested=False)  # type: ignore[no-any-return]
    except _ControlFlowSignal as signal:
      raise self._wrap_escaped_signal(signal, 'run') from None

  def debug(self, v: Any = Null, /, *args: Any, **kwargs: Any) -> Any:
    """Execute the pipeline with debug tracing and return a DebugResult.

    Clones the pipeline (the original is not modified) and runs the clone
    with step-level instrumentation.  The return value is a
    ``DebugResult`` with ``.value``, ``.steps``, ``.elapsed_ns``,
    ``.succeeded``, ``.failed``, and ``.print_trace()``.

    For async pipelines the returned coroutine resolves to a DebugResult.

    Args:
      v: Optional initial value (same semantics as ``run()``).
      *args: Positional arguments for the initial value's callable.
      **kwargs: Keyword arguments for the initial value's callable.

    Returns:
      A DebugResult (or coroutine resolving to one for async pipelines).

    Example::

        dr = Q(5).then(lambda x: x * 2).debug()
        print(dr.value)   # 10
        dr.print_trace()  # formatted table to stderr
    """
    from ._debug import _debug_run, _make_debug_q

    self._ensure_if_consumed()
    debug_q = _make_debug_q(self)
    return _debug_run(debug_q, v, args, kwargs)

  # ---- Utilities and iteration ----

  def as_decorator(self) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap the pipeline as a function decorator.

    The decorated function's return value becomes the pipeline's input value.

    Returns:
      A decorator that wraps functions to execute through this pipeline.

    Example::

        @Q().then(lambda x: x.strip()).then(str.upper).as_decorator()
        def get_name():
            return '  alice  '

        get_name()  # 'ALICE'
    """
    self._ensure_if_consumed()
    q = self.clone()

    def _decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
      @functools.wraps(fn)
      def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        __tracebackhide__ = True
        try:
          # Thread safety: is_nested=False is passed explicitly as a parameter
          # rather than as pipeline state.  This avoids shared mutable state when
          # the cloned pipeline is invoked concurrently from multiple threads.
          return q._run(fn, args, kwargs, is_nested=False)  # type: ignore[no-any-return]  # _run returns Any; decorator signature provides narrower _R for callers
        except _ControlFlowSignal as signal:
          raise Q._wrap_escaped_signal(signal, 'as_decorator') from None

      return _wrapper

    return _decorator

  def _build_iterator(
    self,
    method: str,
    fn: Callable[[Any], Any] | None,
    ignore_result: bool,
    flat: bool = False,
    flush: Callable[[], Any] | None = None,
  ) -> QuentIterator[Any]:
    """Shared implementation for iterate/iterate_do/flat_iterate/flat_iterate_do."""
    self._ensure_if_consumed()
    link = Link(fn) if fn is not None else None
    q_run = functools.partial(_run, self)
    last_link = self._current_link or self._first_link
    deferred_with = None
    if last_link is not None and isinstance(last_link.v, _WithOp):
      deferred_with = (last_link.v._link, last_link.v._ignore_result)
    has_deferred_with = deferred_with is not None
    return QuentIterator(
      q_run,
      fn,
      ignore_result=ignore_result,
      q=self if fn is not None or self._on_finally_link is not None or has_deferred_with else None,
      link=link,
      deferred_with=deferred_with,
      flat=flat,
      flush=flush,
      buffer_size=self._buffer_size,
    )

  def iterate(self, fn: Callable[[Any], Any] | None = None) -> QuentIterator[Any]:
    """Return a sync/async iterator over the pipeline's output.

    Each element of the pipeline's iterable result is yielded. An optional
    *fn* transforms each element before yielding.

    Args:
      fn: Optional callable to transform each element.

    Returns:
      A QuentIterator supporting both ``for`` and ``async for``.

    Example::

        for item in Q(range(5)).iterate(lambda x: x ** 2):
            print(item)  # 0, 1, 4, 9, 16
    """
    return self._build_iterator('iterate', fn, ignore_result=False)

  def iterate_do(self, fn: Callable[[Any], Any] | None = None) -> QuentIterator[Any]:
    """Return a sync/async iterator, discarding *fn*'s return values.

    Like .iterate(), but *fn* runs as a side-effect. The original
    elements are yielded.

    Args:
      fn: Optional callable to invoke for side-effects on each element.

    Returns:
      A QuentIterator supporting both ``for`` and ``async for``.

    Example::

        for item in Q(range(3)).iterate_do(print):
            pass  # prints 0, 1, 2; yields 0, 1, 2
    """
    return self._build_iterator('iterate_do', fn, ignore_result=True)

  def flat_iterate(
    self,
    fn: Callable[[Any], Any] | None = None,
    *,
    flush: Callable[[], Any] | None = None,
  ) -> QuentIterator[Any]:
    """Return a sync/async flatmap iterator over the pipeline's output.

    Each element of the pipeline's iterable result is passed to *fn*, and
    each element from *fn*'s returned iterable is yielded individually.
    After source exhaustion, *flush* (if provided) is called and its
    output yielded similarly.

    Args:
      fn: Optional callable that returns an iterable for each element.
      flush: Optional callable returning a final iterable after source exhaustion.

    Returns:
      A QuentIterator supporting both ``for`` and ``async for``.
    """
    return self._build_iterator('flat_iterate', fn, ignore_result=False, flat=True, flush=flush)

  def flat_iterate_do(
    self,
    fn: Callable[[Any], Any] | None = None,
    *,
    flush: Callable[[], Any] | None = None,
  ) -> QuentIterator[Any]:
    """Return a sync/async flatmap iterator, discarding *fn*'s return values.

    Like .flat_iterate(), but *fn* runs as a side-effect — its returned
    iterable is consumed but not yielded. The original elements are
    yielded instead. *flush* output is still yielded.

    Args:
      fn: Optional callable for side-effects on each element.
      flush: Optional callable returning a final iterable after source exhaustion.

    Returns:
      A QuentIterator supporting both ``for`` and ``async for``.
    """
    return self._build_iterator('flat_iterate_do', fn, ignore_result=True, flat=True, flush=flush)

  def clone(self) -> Self:
    """Create an independent copy of this pipeline for fork-and-extend patterns.

    Returns a new Q with a cloned linked list. Callables, values,
    argument objects (args tuple elements, kwargs dict values), and handler
    callables (except/finally) are shared by reference -- only the list
    structure (Link nodes) and nested Q instances are deep-copied.
    This allows forking a base pipeline and extending each fork independently.

    Returns:
      A new Q with independent link structure.

    Example::

        base = Q().then(validate).then(normalize)
        for_api = base.clone().then(to_json)  # base is not modified
    """
    self._ensure_if_consumed()
    cls = type(self)
    new = cls.__new__(cls)
    new._name = self._name
    new._buffer_size = self._buffer_size
    new._pending_if = self._pending_if
    new._if_predicate_link = _clone_link(self._if_predicate_link) if self._if_predicate_link is not None else None
    new._pending_while = self._pending_while
    new._while_predicate_link = (
      _clone_link(self._while_predicate_link) if self._while_predicate_link is not None else None
    )
    new._on_except_reraise = self._on_except_reraise
    new._on_except_exceptions = self._on_except_exceptions
    new._on_except_link = _clone_link(self._on_except_link) if self._on_except_link is not None else None
    new._on_finally_link = _clone_link(self._on_finally_link) if self._on_finally_link is not None else None

    new._root_link = _clone_link(self._root_link) if self._root_link is not None else None

    if self._first_link is None:
      new._first_link = None
      new._current_link = None
    else:
      new._first_link = _clone_link(self._first_link)
      prev = new._first_link
      old_link = self._first_link.next_link
      while old_link is not None:
        cloned = _clone_link(old_link)
        prev.next_link = cloned
        prev = cloned
        old_link = old_link.next_link
      # _current_link is the tail pointer; only needed when there are 2+ non-root links.
      new._current_link = prev if self._current_link is not None else None

    if new._root_link is not None and new._first_link is not None:
      new._root_link.next_link = new._first_link

    if __debug__:
      _missing = [s for s in self.__slots__ if getattr(new, s, _CLONE_SENTINEL) is _CLONE_SENTINEL]
      if _missing:
        raise QuentException(f'clone() missing slots: {_missing}')

    return new

  @classmethod
  def from_steps(cls, *steps: Any) -> Self:
    """Create a new pipeline from an iterable of steps.

    Accepts either a flat sequence of steps as positional arguments, or a
    single list/tuple argument containing the steps. Each step is appended
    to the pipeline via ``_then()``.

    Args:
      *steps: Steps to add to the pipeline. If a single list or tuple is
        passed, its contents are used as the steps.

    Returns:
      A new Q with all steps appended.

    Example::

        Q.from_steps(validate, normalize, str.upper).run('  hello  ')
        Q.from_steps([validate, normalize, str.upper]).run('  hello  ')
    """
    resolved: tuple[Any, ...] | list[Any] = steps
    if len(steps) == 1 and isinstance(steps[0], (list, tuple)):
      resolved = steps[0]
    q = cls()
    for step in resolved:
      q._then(step)
    return q
