from distutils.core import setup, Extension
from Cython.Build import cythonize

import os

os.environ['CFLAGS'] = '-O3 -Wall -std=c++11 -fPIC'
setup(ext_modules = cythonize(Extension(
		   "_kfusion",
           sources=["_kfusion.pyx", "kfusion.cpp"],
           include_dirs = [
           		'/workspace/kinfu_remake/kfusion/include',
           ],
           libraries = ['kfusion'],
           library_dirs = ['/workspace/kinfu_remake/build/kfusion'],
           language="c++",
           extra_compile_args=['-fPIC'],
           extra_link_args=["-L/workspace/kinfu_remake/build/kfusion"]
      )))