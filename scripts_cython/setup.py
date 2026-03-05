import os
from setuptools import setup, Extension, find_packages


def build_extensions(file_ext, path='./quent', *, tests=False):
  extensions = []
  lto = os.environ.get('QUENT_LTO')
  pgo = os.environ.get('QUENT_PGO')
  native = os.environ.get('QUENT_NATIVE')
  for root, dirs, filenames in os.walk(path):
    for fname in filenames:
      if fname.endswith(file_ext):
        file_path = os.path.join(root, fname)
        module_name = '.'.join(file_path.replace(file_ext, '').split('/')).lstrip('.')
        macros = []
        if tests:
          macros.append(('CYTHON_TRACE_NOGIL', 1))
          macros.append(('CYTHON_USE_SYS_MONITORING', 0))
        # PERF: -O3 enables maximum C compiler optimization (inlining, vectorization, loop transforms).
        # See PERFORMANCE.md #14.
        compile_args = ['-O3', '-Wno-unreachable-code', '-Wno-unused-function']
        link_args = []
        # PERF: LTO (Link-Time Optimization) enables cross-file inlining and dead code elimination.
        # See PERFORMANCE.md #15.
        if lto:
          compile_args.append('-flto')
          link_args.append('-flto')
        # PERF: PGO (Profile-Guided Optimization) uses runtime data for branch prediction and layout.
        # See PERFORMANCE.md #16.
        if pgo == 'generate':
          compile_args.append('-fprofile-generate')
          link_args.append('-fprofile-generate')
        elif pgo == 'use':
          compile_args.extend(['-fprofile-use', '-fprofile-correction'])
          link_args.extend(['-fprofile-use', '-fprofile-correction'])
        # PERF: -march=native (CPU-specific instructions), -funroll-loops (loop unrolling),
        # -fomit-frame-pointer (frees rbp register). See PERFORMANCE.md #24, #25, #26.
        if native:
          compile_args.extend(['-march=native', '-funroll-loops', '-fomit-frame-pointer'])
        extensions.append(Extension(module_name, sources=[file_path], extra_compile_args=compile_args, extra_link_args=link_args, define_macros=macros))
  return extensions


if __name__ == '__main__':
  setup(
    name='quent',
    ext_modules=build_extensions('.c'),
    packages=find_packages(include=['quent*']),
  )
