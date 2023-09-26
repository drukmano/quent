from setuptools import setup, Extension
from Cython.Build import cythonize


setup(
  ext_modules=cythonize(
    [Extension(name='quent', sources=['src/quent.pyx'], extra_compile_args=['-O3'])],
    include_path=['src/'],
    annotate=False
  )
)
