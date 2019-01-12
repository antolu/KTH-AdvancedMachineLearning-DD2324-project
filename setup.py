from distutils.core import setup
from Cython.Build import cythonize
import pyximport; pyximport.install()




pyximport.install(pyimport=True)

setup(
    ext_modules=cythonize("test.pyx"),
)