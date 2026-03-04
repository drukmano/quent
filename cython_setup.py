import sys

from setuptools import setup, find_packages
from setup import build_extensions
from Cython.Build import cythonize
# PERF: Larger closure freelist (16 vs default 8) reduces allocation overhead for closures/lambdas.
# See PERFORMANCE.md #13.
import Cython.Compiler.Options; Cython.Compiler.Options.closure_freelist_size = 16


if __name__ == '__main__':
  is_tests = 'tests' in sys.argv[1]
  if is_tests:
    sys.argv.remove('tests')
  # PERF: Compiler directives for maximum performance. See PERFORMANCE.md #7-#12, #41.
  compiler_directives = {
    'binding': False,                    # PERF #7: Lighter CFunction wrappers
    'boundscheck': False,                # PERF #9: No array bounds checking
    'wraparound': False,                 # PERF #9: No negative-index handling
    'language_level': 3,
    'embedsignature': True,
    'annotation_typing': False,
    'always_allow_keywords': False,      # PERF #8: METH_NOARGS/METH_O calling conventions
    'infer_types': True,                 # PERF #12: Automatic C type inference
    'initializedcheck': False,           # PERF #10: No memoryview init checks
    'cdivision': True,                   # PERF #11: C-style division, no ZeroDivisionError
    'iterable_coroutine': False,
    'nonecheck': False,                  # PERF #41: No None attribute access checks
    'overflowcheck': False,              # PERF #41: No integer overflow checks
    'optimize.use_switch': True,         # Generates C switch statements for int comparisons
    'optimize.unpack_method_calls': True, # Optimizes method calls by unpacking bound methods
  }
  if is_tests:
    compiler_directives.update({
      'profile': True,
      'linetrace': True,
      'warn.undeclared': True,
    })
  else:
    compiler_directives.update({
      'emit_code_comments': False,
    })
  setup(
    name='quent',
    ext_modules=cythonize(
      build_extensions('.pyx', tests=is_tests),
      compiler_directives=compiler_directives,
      annotate=is_tests,
    ),
    packages=find_packages(),
  )
