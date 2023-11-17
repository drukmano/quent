import sys

from setuptools import setup, find_packages
from setup import build_extensions
from Cython.Build import cythonize


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
      #include_path=['quent/', 'quent/helpers/'],
      compiler_directives=compiler_directives,
      annotate=is_tests
    ),
    packages=find_packages(),
  )
