from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("utils_cy", ["utils_cy.pyx"]),
                   Extension("numerics", ["numerics.pyx",]),
                   Extension("field_numerics", ["field_numerics.pyx",]),
                   Extension("walled_field_numerics", ["walled_field_numerics.pyx",]),
                   ],
    include_dirs = [numpy.get_include()],
)
