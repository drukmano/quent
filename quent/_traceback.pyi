import traceback
import types
from typing import Any

from ._chain import Chain
from ._core import Link

_quent_file: str
_RAISE_CODE: types.CodeType
_HAS_QUALNAME: bool
_TracebackType: type[types.TracebackType]

class _Ctx:
  __slots__: tuple[str, ...]
  source_link: Link | None
  link_temp_args: dict[int, dict[str, Any]] | None
  found: bool
  def __init__(self, source_link: Link | None, link_temp_args: dict[int, dict[str, Any]] | None) -> None: ...

def _clean_internal_frames(tb: types.TracebackType | None) -> types.TracebackType | None: ...
def _clean_chained_exceptions(exc: BaseException | None, seen: set[int]) -> None: ...
def _modify_traceback(
  exc: BaseException,
  chain: Chain | None = ...,
  link: Link | None = ...,
  root_link: Link | None = ...,
  extra_links: list[tuple[Link, str]] | None = ...,
) -> BaseException: ...
def _get_true_source_link(source_link: Link | None, root_link: Link | None) -> Link | None: ...
def _make_indent(nest_lvl: int) -> str: ...
def _get_link_name(link: Link) -> str: ...
def _get_obj_name(obj: Any) -> str: ...
def _format_call_args(args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> str: ...
def _resolve_nested_chain(
  link: Link,
  args: tuple[Any, ...] | None,
  kwargs: dict[str, Any] | None,
  nest_lvl: int,
  ctx: _Ctx,
  max_depth: int = ...,
) -> str: ...
def _stringify_chain(
  chain: Chain,
  nest_lvl: int = ...,
  root_link: Link | None = ...,
  *,
  ctx: _Ctx,
  extra_links: list[tuple[Link, str]] | None = ...,
  max_depth: int = ...,
) -> str: ...
def _format_link(
  link: Link, nest_lvl: int, ctx: _Ctx, method_name: str | None = ..., max_depth: int = ...
) -> str: ...

_original_excepthook: Any

def _quent_excepthook(
  exc_type: type[BaseException], exc_value: BaseException, exc_tb: types.TracebackType | None
) -> None: ...

_original_te_init: Any

def _patched_te_init(
  self: traceback.TracebackException,
  exc_type: type[BaseException],
  exc_value: BaseException | None = ...,
  exc_traceback: types.TracebackType | None = ...,
  **kwargs: Any,
) -> None: ...

_patching_enabled: bool

def disable_traceback_patching() -> None: ...
def enable_traceback_patching() -> None: ...
