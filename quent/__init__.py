from typing import TypeVar, Generic, Union, Awaitable
from .quent import Chain, Cascade, ChainAttr, CascadeAttr, run, QuentException


T = TypeVar('T')


class ResultOrAwaitable(Generic[T]):
  pass


__all__ = [
  'Chain', 'Cascade', 'ChainAttr', 'CascadeAttr', 'QuentException', 'run', 'ResultOrAwaitable'
]
