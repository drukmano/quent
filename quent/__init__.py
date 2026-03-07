"""quent -- Fluent chain/pipeline with transparent sync/async bridging."""

from importlib.metadata import version as _get_version

from . import _traceback  # noqa: F401  -- triggers exception hook installation
from ._chain import Chain
from ._core import Null, QuentException

__version__ = _get_version(__name__)

__all__ = [
  'Chain',
  'Null',
  'QuentException',
  '__version__',
]
