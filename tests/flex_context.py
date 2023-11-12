import threading
import logging
from contextlib import suppress
from contextvars import ContextVar
from typing import *


# TODO test this with nested dict / list; this class should never mutate anything (if user gives a list etc.).
FLEX_CONTEXT = ContextVar('flex_context', default={})


# TODO decorate with @final and add annotations


class ContextManagerReentryError(Exception):
  pass


class FlexContext:
  """
  https://gist.github.com/elidchan/2c944d46c2d857fd592b095b55d8a60f

  FlexContext
  FlexContext is a context manager for context variables. Unlike
  regular context variables that must be declared individually at the
  module level, flex context variables all share a single FLEX_CONTEXT
  variable by utilizing a dictionary.

  A copy of the current FLEX_CONTEXT is retrieved via get_context().

  A FlexContext instance is initialized with a `delta` dictionary of
  flex context variables and values to be applied upon context entry.
  As a context manager, a FlexContext instance is entered via `with`.

  Upon context entry, FLEX_CONTEXT is updated with a new `context` in
  which `delta` has been applied to the `baseline` FLEX_CONTEXT. At
  this point, `context` and `baseline` are established and stored on
  the FlexContext instance for reference.

  Upon context exit, the FLEX_CONTEXT is reset to the `baseline`. In
  addition, the `context` and `baseline` fields are reset to None.

  Multiple FlexContext instances may be applied by nesting `with`
  statements. A single FlexContext instance may be re-entered multiple
  times, but only after exiting after each use.

  Usage:
  >>> print(FlexContext.get_context())
  {}

  >>> with FlexContext(color='blue', number=42, obj=object()) as context_a:
          print(FlexContext.get_context())
          assert context_a.context == FlexContext.get_context()
          assert context_a.baseline == {}
          with FlexContext(color='yellow', obj=object()) as context_b:
              print(FlexContext.get_context())
              assert context_b.context == FlexContext.get_context()
              assert context_b.baseline == context_a.context
          print(FlexContext.get_context())
          assert context_b.context is None and context_b.baseline is None
  {'color': 'blue', 'number': 42, 'obj': <object object at 0x107b4ca60>}
  {'color': 'yellow', 'number': 42, 'obj': <object object at 0x107b4caa0>}
  {'color': 'blue', 'number': 42, 'obj': <object object at 0x107b4ca60>}

  >>> print(FlexContext.get_context())
  {}
  """

  # TODO make decorator `cls.on_exit('key')` which register a method to be called when a context-var
  #  is being exited (self._delta tracks the current variables, not all variables if nested contexts)
  @classmethod
  def on_exit(cls, **kwargs):
    """ Called before __exit__, kwargs hold the current variables """
    pass

  @classmethod
  def __format_key(cls, key: str) -> str:
    return f'{cls.__name__}:{key}'

  @classmethod
  def __from_formatted(cls, key: str) -> str:
    return key.split(':', maxsplit=1)[1]

  @classmethod
  def setter(cls, name, value):
    """ Customized setter used to set values by specific rules; override to customize """
    return value

  @classmethod
  def get(cls, **kwargs) -> Union[Tuple, Any, List]:
    """ Get variables, each kwarg is as follows: var-name=default-value """
    data = FLEX_CONTEXT.get({})
    values = [data.get(cls.__format_key(k), v) for k, v in kwargs.items()]
    return tuple(values)[0]

  @classmethod
  def get_context(cls):
    """Get context from flex_context ContextVar; always current state"""
    context = FLEX_CONTEXT.get({})
    return context.copy()

  @property
  def delta(self):
    """Delta context vars; dict to apply to baseline upon entry"""
    return self._delta.copy()

  @property
  def baseline(self):
    """Baseline (old) context dict upon entry; None outside context"""
    return None if self._baseline is None else self._baseline.copy()

  @property
  def context(self):
    """Context (new) dict upon entry; None outside context"""
    return None if self._context is None else self._context.copy()

  def __enter__(self):
    """Enter context, applying delta to baseline to form context"""
    if self._token is not None:
      raise ContextManagerReentryError(
        f'The same context cannot be re-entered until exiting; token: {self._token}'
      )
    self._baseline = self.get_context()
    self._context = {**self._baseline, **self._delta}
    self._token = FLEX_CONTEXT.set(self._context)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Exit context, resetting baseline and context to None"""
    self.on_exit(**{self.__from_formatted(k): v for k, v in self._delta.items()})
    if self._token is not None:
      FLEX_CONTEXT.reset(self._token)
    else:
      logging.warning('FlexContext.token is None upon __exit__')
    self._baseline = None
    self._context = None
    self._token = None

  def __getattr__(self, name):
    """Get delta vars and within context, context vars"""
    with suppress(KeyError):
      return self._delta[name]
    try:
      return self._context[name]
    except TypeError as e:
      raise AttributeError(
        f"'{self!r}' context vars are only available within context"
      ) from e
    except KeyError as e:
      raise AttributeError(f"'{name}' not found in '{self!r}' context") from e

  def __setattr__(self, name, value):
    """Setattr is disabled for context vars; contexts are immutable"""
    if name in self.__dict__:
      return super().__setattr__(name, value)
    raise AttributeError(f"'{self!r}' vars are immutable")

  def __delattr__(self, name):
    """Delattr is disabled for context vars; contexts are immutable"""
    if name in self.__dict__:
      return super().__delattr__(name)
    raise AttributeError(f"'{self!r}' vars cannot be deleted")

  def __contains__(self, item):
    """Contains item in delta vars or, within context, context vars"""
    try:
      return item in self._delta or item in self._context
    except TypeError:
      return False

  def __repr__(self):
    arguments = [f'{k}={v!r}' for k, v in self._delta.items()]
    return f"{self.__class__.__name__}({', '.join(arguments)})"

  def __init__(self, **delta):
    """Initialize instance with `delta` dict of context var names/values"""
    delta = {self.__format_key(k): self.setter(k, v) for k, v in delta.items()}
    self._initialize_attributes(_delta=delta, _baseline=None, _context=None, _token=None)

  def _initialize_attributes(self, **attributes):
    """Initialize attributes on instance given dict of attribute names/values"""
    for attribute, value in attributes.items():
      super().__setattr__(attribute, value)

  @classmethod
  def create_from_context(cls, context: dict):
    ctx = cls()
    ctx._delta = context
    return ctx

  @classmethod
  def to_thread(cls, fn, *, args=None, kwargs=None, **kwargs_):
    if args is None:
      args = []
    if kwargs is None:
      kwargs = {}

    def wrapper(context):
      with cls.create_from_context(context):
        return fn(*args, **kwargs)

    return threading.Thread(target=wrapper, args=(cls.get_context(),), kwargs=kwargs, **kwargs_)
