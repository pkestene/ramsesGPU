from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("powerSpectrum2", ["powerSpectrum2.pyx"], libraries=["m"])]

)

# build:
# python setup.py build_ext --inplace

