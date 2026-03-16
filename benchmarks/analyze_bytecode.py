# SPDX-License-Identifier: MIT
"""Bytecode analysis of quent hot-path functions using the dis module.

Disassembles the key functions and flags potentially expensive bytecode
operations such as dynamic dispatch, dict allocation, or unpacking.

Usage:
    python benchmarks/analyze_bytecode.py
"""

from __future__ import annotations

import dis
import sys
from typing import Any

from quent._engine import _run, _run_async
from quent._eval import _evaluate_value
from quent._link import Link

# Bytecode operations that may indicate expensive operations in hot-path code.
FLAGGED_OPS: set[str] = {
  'CALL_FUNCTION_EX',
  'BUILD_MAP',
  'BUILD_MAP_UNPACK',
  'DICT_MERGE',
  'LIST_EXTEND',
  'BUILD_CONST_KEY_MAP',
  'COPY_DICT_WITHOUT_KEYS',
}


def analyze_function(fn: Any, name: str) -> int:
  """Disassemble *fn* and flag expensive bytecode operations.

  Returns the number of flagged instructions found.
  """
  print(f'\n{"=" * 80}')
  print(f'Disassembly: {name}')
  print(f'{"=" * 80}')

  instructions = list(dis.get_instructions(fn))
  flagged_count = 0

  for instr in instructions:
    marker = ''
    if instr.opname in FLAGGED_OPS:
      marker = '  <<< FLAGGED'
      flagged_count += 1

    offset_str = f'{instr.offset:>4}'
    arg_str = f'{instr.arg}' if instr.arg is not None else ''
    argval_str = f'({instr.argrepr})' if instr.argrepr else ''
    print(f'  {offset_str} {instr.opname:<30} {arg_str:<8} {argval_str}{marker}')

  if flagged_count:
    print(f'\n  >>> {flagged_count} flagged instruction(s) found')
  else:
    print('\n  >>> No flagged instructions (clean)')

  return flagged_count


def count_opcodes(fn: Any) -> dict[str, int]:
  """Return a dict mapping opname -> count for every instruction in *fn*."""
  counts: dict[str, int] = {}
  for instr in dis.get_instructions(fn):
    counts[instr.opname] = counts.get(instr.opname, 0) + 1
  return counts


def main() -> None:
  print(f'Python {sys.version}')
  print('Analyzing bytecode of quent hot-path functions...')

  targets = [
    (_evaluate_value, '_evaluate_value'),
    (_run, '_run'),
    (_run_async, '_run_async'),
    (Link.__init__, 'Link.__init__'),
  ]

  total_flagged = 0
  for fn, name in targets:
    total_flagged += analyze_function(fn, name)

  # Summary table of instruction mix across all targets.
  print(f'\n{"=" * 80}')
  print('Instruction count summary (all targets combined)')
  print(f'{"=" * 80}')
  combined: dict[str, int] = {}
  for fn, _ in targets:
    for op, n in count_opcodes(fn).items():
      combined[op] = combined.get(op, 0) + n
  for op, n in sorted(combined.items(), key=lambda x: -x[1])[:20]:
    flag = ' <<<' if op in FLAGGED_OPS else ''
    print(f'  {op:<30} {n:>6}{flag}')

  print(f'\n{"=" * 80}')
  print(f'Total flagged instructions across all functions: {total_flagged}')
  print(f'{"=" * 80}')


if __name__ == '__main__':
  main()
