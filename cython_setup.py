import sys

from setuptools import setup, find_packages
from setup import build_extensions
from Cython.Build import cythonize
import Cython.Compiler.Options; Cython.Compiler.Options.closure_freelist_size = 16
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#Cython.Compiler.Options.closure_freelist_size


if __name__ == '__main__':
  is_tests = 'tests' in sys.argv[1]
  if is_tests:
    sys.argv.remove('tests')
  compiler_directives = {
    'binding': False,
    'boundscheck': False,
    'wraparound': False,
    'language_level': 3,
    'embedsignature': True,
    'annotation_typing': False,
    'always_allow_keywords': False,
    'infer_types': True,
    'initializedcheck': False,
    'cdivision': True,
    'iterable_coroutine': False,
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
