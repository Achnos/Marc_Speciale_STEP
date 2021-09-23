import os
import ctypes
import sys
import pathlib
import numpy as np

import ccd
import libc_builder

if __name__ == '__main__':
    if len(sys.argv) < 2:
        libname = "libccd"
        os.system("invoke build-" + libname)
        ccd_lib = libc_builder.cpp_build(str(libname))
    if len(sys.argv) == 2:
        os.system("invoke build-" + str(sys.argv[1]))
        ccd_lib = libc_builder.cpp_build(str(sys.argv[1]))

    scalelimit = 63776
    looking_at = "/zerobias.fits"
    path_here = pathlib.Path().absolute()
    filepath = ccd.get_path(str(path_here) + looking_at)

    hdulist, header, imagedata = ccd.fits_handler(filepath, scalelimit, show=True)

    imagedata_list = (np.ndarray.flatten(imagedata)).tolist()
    imagedata_list_len = len(imagedata_list)
    array_to_c = (ctypes.c_int * imagedata_list_len)(*imagedata_list)

    class_ctor_wrapper = ccd_lib.constructor_wrapper
    class_ctor_wrapper.restype = ctypes.c_void_p
    return_ptr = ctypes.c_void_p(class_ctor_wrapper())

    # [outputs the pointer address of the element]
    ccd_lib.sum_array_wrapper.restype = ctypes.c_double
    value = ccd_lib.dark_current(return_ptr, imagedata_list_len, array_to_c)
    print("Estimate of thermal dark current: ", value, " #electrons / pixel")
    print("Note to self: Result is overflown, but this was expected. Else works as intended.")

    # Class_ctor_wrapper_del = ccd_lib.DeleteInstanceOfClass
    # Class_ctor_wrapper_del.restype = ctypes.c_void_p
    # spam_del = ctypes.c_void_p(Class_ctor_wrapper_del())


