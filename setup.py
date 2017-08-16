import os
from sys import platform as _platform

if _platform == "darwin":
    os.environ["CC"] = "gcc-5"
    os.environ["CXX"] = "g++-5"

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

sourcefiles = ['src/c_velsolve.pyx']

if _platform == "darwin":
    extensions = [Extension("c_velsolve",
                    sourcefiles,
                    language="c++",
                    include_dirs=[".",  "./src",  "./src/include", numpy.get_include()],
                    library_dirs=[".", "./src"],
                    extra_compile_args=['-fopenmp', '-std=c++11'],
                    extra_link_args=['-fopenmp'],
                   )]
elif _platform == "win32":
    extensions = [Extension("c_velsolve",
                    sourcefiles,
                    language="c++",
                    include_dirs=[".", "./src", "./src/include", numpy.get_include()],
                    library_dirs=[".", "./src"],
                    extra_compile_args=['/openmp'],
                   )]

setup(
    ext_modules = cythonize(extensions)
)