"""build.py: for making lorenz_sde[...].so from lorenz_sde.pyx

To use:

$ python build.py build_ext --inplace
"""
import setuptools
import Cython.Build

import numpy

extensions = [
    setuptools.Extension("lorenz_sde", ["lorenz_sde.pyx"],
                         extra_compile_args=["-fopenmp"],
                         extra_link_args=["-fopenmp"],
                         include_dirs=[numpy.get_include()])
]

setuptools.setup(
    name="lorenz_sde",
    ext_modules=Cython.Build.cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        annotate=True  # Creates C.html
    ),
)

# I used code from the section "Configuring the C-Build" in
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
# as a template.
