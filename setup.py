import os
from setuptools import setup, Extension, find_packages


def build_extensions(file_ext, path='./quent', *, tests=False):
  extensions = []
  for root, dirs, filenames in os.walk(path):
    for fname in filenames:
      if fname.endswith(file_ext):
        file_path = os.path.join(root, fname)
        module_name = '.'.join(file_path.replace(file_ext, '').split('/')).lstrip('.')
        macros = []
        if tests:
          macros.append(('CYTHON_TRACE_NOGIL', 1))
        extensions.append(Extension(module_name, sources=[file_path], extra_compile_args=['-O3'], define_macros=macros))
  return extensions


if __name__ == '__main__':
  setup(
    name='quent',
    ext_modules=build_extensions('.c'),
    packages=find_packages(include=['quent*']),  # ['quent', 'quent.helpers']
    #include_dirs=['quent'],
    package_data={
      'quent': ['py.typed'],
    },
  )
