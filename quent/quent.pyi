from typing import Callable, Any, Iterator, AsyncIterator, TypeVar, ParamSpec, Self, Type


P = ParamSpec('P')
#Self = TypeVar('Self', bound='Chain')

AnyValue = TypeVar('AnyValue')
RootValue = TypeVar('RootValue')
CurrentValue = TypeVar('CurrentValue')
NextValue = TypeVar('NextValue')
IgnoredValue = TypeVar('IgnoredValue')
Item = TypeVar('Item')
NewItem = TypeVar('NewItem')
IgnoredItem = TypeVar('IgnoredItem')
Context = TypeVar('Context')
AnyOrCoroutine = TypeVar('AnyOrCoroutine')

RootLink = RootValue | Callable[[P], RootValue]

LinkFunc = Callable[[CurrentValue | P], NextValue]
LinkAnyValue = AnyValue | Callable[[CurrentValue | P], NextValue]
LinkVoidFunc = Callable[[CurrentValue | P], IgnoredValue]

LinkRootFunc = Callable[[RootValue | P], NextValue]
LinkRootVoidFunc = Callable[[RootValue | P], IgnoredValue]

LinkIterFunc = Callable[[Item], NewItem]
LinkIterVoidFunc = Callable[[Item], IgnoredItem]

LinkWithFunc = Callable[[Context], NextValue]
LinkWithVoidFunc = Callable[[Context], IgnoredValue]


class Chain:
  def __init__(self: Self, v: RootLink = None, *args: P.args, **kwargs: P.kwargs): ...

  def config(self, *, autorun: bool = None) -> Self: ...

  def autorun(self, autorun: bool = True) -> Self: ...

  def clone(self) -> Self: ...

  def freeze(self) -> FrozenChain: ...

  def decorator(self) -> Callable[[Callable], Callable]: ...

  def run(self, v: RootLink = None, *args: P.args, **kwargs: P.kwargs) -> AnyOrCoroutine: ...

  def then(self, v: LinkAnyValue, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def do(self, fn: LinkVoidFunc, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def root(self, fn: LinkRootFunc, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def root_do(self, fn: LinkRootVoidFunc , *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def attr(self, name: str) -> Self: ...

  def attr_fn(self, name: str, *args, **kwargs) -> Self: ...

  def except_(
    self, fn: LinkRootVoidFunc, *args: P.args, exceptions: list[Type[BaseException]] | Type[BaseException] = None,
    raise_: bool = True, **kwargs: P.kwargs
  ) -> Self: ...

  def finally_(self, fn: LinkRootVoidFunc, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def iterate(self, fn: LinkIterFunc = None) -> Iterator | AsyncIterator: ...

  def iterate_do(self, fn: LinkIterVoidFunc = None) -> Iterator | AsyncIterator: ...

  def foreach(self, fn: LinkIterFunc) -> Self: ...

  def foreach_do(self, fn: LinkIterVoidFunc) -> Self: ...

  def with_(self, fn: LinkWithFunc, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def with_do(self, fn: LinkWithVoidFunc, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def if_(self, v: LinkAnyValue, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def else_(self, v: LinkAnyValue, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def if_not(self, v: LinkAnyValue, *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def if_raise(self, exc: BaseException) -> Self: ...

  def else_raise(self, exc: BaseException) -> Self: ...

  def if_not_raise(self, exc: BaseException) -> Self: ...

  def condition(self, fn: Callable[[CurrentValue | P], bool], *args: P.args, **kwargs: P.kwargs) -> Self: ...

  def not_(self) -> Self: ...

  def eq(self, value: Any) -> Self: ...

  def neq(self, value: Any) -> Self: ...

  def is_(self, value: Any) -> Self: ...

  def is_not(self, value: Any) -> Self: ...

  def in_(self, value: Any) -> Self: ...

  def not_in(self, value: Any) -> Self: ...

  def or_(self, value: Any) -> Self: ...

  def raise_(self, exc: BaseException) -> Self: ...

  def __or__(self, other: AnyValue | Callable[[CurrentValue], NextValue]) -> Self: ...

  def __call__(self, v: RootLink = None, *args: P.args, **kwargs: P.kwargs) -> AnyOrCoroutine: ...


class Cascade(Chain):
  ...


class ChainAttr(Chain):
  def __getattr__(self, name: str) -> Self: ...

  def __call__(self, *args, **kwargs) -> Self | AnyOrCoroutine: ...


class CascadeAttr(ChainAttr):
  ...


class FrozenChain:
  def decorator(self) -> Callable[[Callable], Callable]: ...

  def run(self, v: RootLink = None, *args: P.args, **kwargs: P.kwargs) -> AnyOrCoroutine: ...

  def __call__(self, v: RootLink = None, *args: P.args, **kwargs: P.kwargs) -> AnyOrCoroutine: ...
