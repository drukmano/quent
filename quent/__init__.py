# SPDX-License-Identifier: MIT
"""Fluent pipelines with transparent sync/async bridging."""

from importlib.metadata import PackageNotFoundError as _PNF
from importlib.metadata import version as _get_version

from . import _traceback  # noqa: F401  -- triggers exception hook installation
from ._generator import QuentIterator
from ._q import Q
from ._types import QuentException, QuentExcInfo

try:
  __version__ = _get_version(__name__)
except _PNF:
  __version__ = '0.0.0-dev'

__all__ = [
  'Q',
  'QuentExcInfo',
  'QuentException',
  'QuentIterator',
  '__version__',
]
