from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = Extension(
    name="align",
    sources=["align.pyx"],
)
setup(
    name="align",
    ext_modules=cythonize([examples_extension])
)