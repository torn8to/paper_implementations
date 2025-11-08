from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = [Extension("operations_faster",
                        [
                       #"cython_modules/voxel_types.pyx",
                       "cython_modules/unique_cy.pyx"], 
                       include_dirs=[numpy.get_include()],
                       language="c++",
                       extra_compile_args=["-O3", "-std=c++17"]
                       )]

setup(
    ext_modules = cythonize(extension, language_level="3")
)
