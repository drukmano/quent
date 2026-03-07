import asyncio
import threading
from collections.abc import Callable, Coroutine
from typing import Any

class _Null:
  __slots__: tuple[()]
  def __repr__(self) -> str: ...
  def __copy__(self) -> _Null: ...
  def __deepcopy__(self, memo: dict[int, Any]) -> _Null: ...
  def __reduce__(self) -> str: ...

Null: _Null

class QuentException(Exception):
  __slots__: tuple[()]

class _ControlFlowSignal(Exception):
  __slots__: tuple[str, ...]
  value: Any
  args_: tuple[Any, ...]
  kwargs_: dict[str, Any]
  def __init__(self, v: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None: ...

class _Return(_ControlFlowSignal):
  __slots__: tuple[()]

class _Break(_ControlFlowSignal):
  __slots__: tuple[()]

def _resolve_value(v: Any, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> Any: ...
def _handle_break_exc(exc: _Break, fallback: Any) -> Any: ...
def _handle_return_exc(exc: _Return, propagate: bool) -> Any: ...
def _set_link_temp_args(exc: BaseException, link: Link, /, **kwargs: Any) -> None: ...

_create_task_fn: Callable[..., asyncio.Task[Any]]
_task_registry: set[asyncio.Task[Any]]
_task_registry_lock: threading.Lock

def _ensure_future(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]: ...
def _task_done_callback(task: asyncio.Task[Any]) -> None: ...

class Link:
  __slots__: tuple[str, ...]
  v: Any
  next_link: Link | None
  ignore_result: bool
  args: tuple[Any, ...] | None
  kwargs: dict[str, Any] | None
  original_value: Any
  is_chain: bool
  def __init__(
    self,
    v: Any,
    args: tuple[Any, ...] | None = ...,
    kwargs: dict[str, Any] | None = ...,
    ignore_result: bool = ...,
    original_value: Any | None = ...,
  ) -> None: ...

def _evaluate_value(
  link: Link,
  current_value: Any = ...,
  extra_args: tuple[Any, ...] = ...,
  extra_kwargs: dict[str, Any] | None = ...,
) -> Any: ...
