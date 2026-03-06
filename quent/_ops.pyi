from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from ._core import Link

def _make_with(link: Link, ignore_result: bool) -> Callable[[Any], Any]: ...
def _sync_generator(
  chain_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  chain: Any = ...,
  link: Any = ...,
) -> Iterator[Any]: ...
async def _aiter_wrap(sync_iter: Iterator[Any]) -> AsyncIterator[Any]: ...
async def _async_generator(
  chain_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  chain: Any = ...,
  link: Any = ...,
) -> AsyncIterator[Any]: ...

class _Generator:
  __slots__: tuple[str, ...]
  _chain_run: Callable[..., Any]
  _fn: Callable[[Any], Any] | None
  _ignore_result: bool
  _chain: Any
  _link: Any
  _run_args: tuple[Any, tuple[Any, ...], dict[str, Any]]
  def __init__(
    self,
    chain_run: Callable[..., Any],
    fn: Callable[[Any], Any] | None,
    ignore_result: bool,
    chain: Any = ...,
    link: Any = ...,
  ) -> None: ...
  def __call__(self, v: Any = ..., *args: Any, **kwargs: Any) -> _Generator: ...
  def __iter__(self) -> Iterator[Any]: ...
  def __aiter__(self) -> AsyncIterator[Any]: ...
  def __repr__(self) -> str: ...

def _make_foreach(link: Link, ignore_result: bool) -> Callable[[Any], Any]: ...
def _make_filter(link: Link) -> Callable[[Any], Any]: ...
def _make_gather(fns: tuple[Callable[[Any], Any], ...]) -> Callable[[Any], Any]: ...
def _make_if(predicate_link: Link, fn_link: Link) -> Callable[[Any], Any]: ...
