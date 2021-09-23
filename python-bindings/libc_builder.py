import os
import sys
import pathlib
import ctypes


def cpp_build(whichlib):
    """

    :param str whichlib:
        - A string representing the name of the .so or .dll (prepended with lib)
          file which contains the C/C++ library to compile and build using invoke,
          and load into ctypes for use with python
    :return ctypes.CDLL object

    """

    print("Building libraries...")
    # Start by calling invoke using tasks.py to build C/C++ libraries with gcc/g++

    libname = pathlib.Path().absolute()
    print("libname: ", libname)

    # Load the shared library into ctypes.
    if sys.platform.startswith("win"):
        libstr = whichlib + ".dll"
        return_lib = ctypes.CDLL(libname / libstr[3:]) # Windows wont have 'lib' as prefix
    else:
        libstr = whichlib + ".so"
        return_lib = ctypes.CDLL(libname / libstr)

    print("Got C/C++ library: ", libstr)

    return return_lib
