from setuptools import setup, Extension


setup(ext_modules=[Extension(name='quent', sources=['src/quent.c'])])
