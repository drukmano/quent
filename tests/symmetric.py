"""SymmetricTestCase -- cartesian-product sync/async bridge testing base class."""

from __future__ import annotations

import itertools
from typing import Any
from unittest import IsolatedAsyncioTestCase

from tests.fixtures import _UNSET, Result, capture


class SymmetricTestCase(IsolatedAsyncioTestCase):
  """Base class for cartesian-product sync/async bridge testing.

  ``variant()`` computes the cartesian product of all axes, runs the
  builder with each combination, and asserts:
  1. ALL combinations produce the same result.
  2. The result matches the expected value or exception.
  """

  async def variant(
    self,
    builder: Any,
    *,
    expected: Any = _UNSET,
    expected_exc: Any = _UNSET,
    expected_msg: str | None = None,
    **axes: Any,
  ) -> list[Result]:
    axis_names = list(axes.keys())
    axis_values = list(axes.values())
    combos = list(itertools.product(*axis_values))

    results: list[Result] = []
    labels: list[str] = []

    for combo in combos:
      label_parts = []
      kwargs: dict[str, Any] = {}
      for name, (label, value) in zip(axis_names, combo):
        label_parts.append(f'{name}={label}')
        kwargs[name] = value
      label = ', '.join(label_parts) if label_parts else 'default'
      labels.append(label)

      kw = dict(kwargs)
      result = await capture(lambda _kw=kw: builder(**_kw))
      results.append(result)

    # Assert all combinations produce the same outcome
    first = results[0]
    for i in range(1, len(results)):
      with self.subTest(symmetry=f'{labels[0]} vs {labels[i]}'):
        self.assertEqual(
          first.success,
          results[i].success,
          f'{labels[0]} {"succeeded" if first.success else "failed"} but '
          f'{labels[i]} {"succeeded" if results[i].success else "failed"}',
        )
        if first.success:
          self.assertEqual(first.value, results[i].value, f'value mismatch: {labels[i]}')
        else:
          self.assertEqual(first.exc_type, results[i].exc_type, f'exc type mismatch: {labels[i]}')

    # Assert expected value/exception
    for result, label in zip(results, labels):
      with self.subTest(expected=label):
        if expected is not _UNSET:
          self.assertTrue(
            result.success,
            f'{label}: expected success but got '
            f'{result.exc_type.__name__ if result.exc_type else "?"}: {result.exc_message}',
          )
          self.assertEqual(result.value, expected, f'{label}: value mismatch')
        if expected_exc is not _UNSET:
          self.assertFalse(
            result.success,
            f'{label}: expected {expected_exc.__name__} but succeeded with {result.value!r}',
          )
          self.assertEqual(result.exc_type, expected_exc, f'{label}: exception type mismatch')
          if expected_msg is not None:
            self.assertIn(expected_msg, result.exc_message or '', f'{label}: message mismatch')

    return results
