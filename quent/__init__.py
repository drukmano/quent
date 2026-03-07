"""quent -- Fluent chain/pipeline with transparent sync/async bridging."""

from importlib.metadata import version as _get_version

from . import _traceback  # noqa: F401  -- triggers exception hook installation
from ._chain import Chain
from ._core import Null, QuentException
from ._traceback import disable_traceback_patching, enable_traceback_patching
from ._x import X

__version__ = _get_version(__name__)

__all__ = [
  'Chain',
  'Null',
  'QuentException',
  'X',
  '__version__',
  'disable_traceback_patching',
  'enable_traceback_patching',
]
