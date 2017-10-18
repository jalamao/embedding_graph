from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'graph kernels',
  ext_modules = cythonize("kernels.pyx"),
)
