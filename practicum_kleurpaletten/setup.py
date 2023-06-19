from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize([
        Extension(
            name="dither",
            sources=["dither.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=[],
            ),
    ], annotate=True),
)
