import os
import ctypes
import sys
import pathlib
import numpy as np

import ccd
import libc_builder

if __name__ == '__main__':

    ccd_lib = libc_builder.cpp_build("libccd")

    scalelimit = 300
    looking_at = "/FITS/saturn2.fits"
    path_here = pathlib.Path().absolute()
    filepath = ccd.get_path(str(path_here) + looking_at)

    hdulist, header, imagedata = ccd.fits_handler(filepath, scalelimit)

    imagedata_list = (np.ndarray.flatten(imagedata)).tolist()
    imagedata_list_len = len(imagedata_list)
    array_to_c = (ctypes.c_int * imagedata_list_len)(*imagedata_list)

    class_ctor_wrapper = ccd_lib.constructor_wrapper
    class_ctor_wrapper.restype = ctypes.c_void_p
    return_ptr = ctypes.c_void_p(class_ctor_wrapper())

    # [outputs the pointer address of the element]
    ccd_lib.sum_array_wrapper.restype = ctypes.c_double
    value = ccd_lib.sum_array_wrapper(return_ptr, imagedata_list_len, array_to_c)
    print("Value from test function: ", value)
    print("Note to self: Result is overflown, but this was expected. Else works as intended.")

    # Class_ctor_wrapper_del = ccd_lib.DeleteInstanceOfClass
    # Class_ctor_wrapper_del.restype = ctypes.c_void_p
    # spam_del = ctypes.c_void_p(Class_ctor_wrapper_del())


