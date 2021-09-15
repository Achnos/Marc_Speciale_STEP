import os
import ctypes
import sys
import pathlib

import CCD

if __name__ == '__main__':
    os.system("invoke build-libccd")

    lookingAt = 'focus_3/focus_9.fits' # 'saturn2.fits'

    #test_camera = CCD.CCD("test_name")
    #test_camera.__linearity("/home/marc/Dropbox/STEP_Speciale_Marc/FITS/focus_3/")

    libname = pathlib.Path().absolute()
    print("libname: ", libname)

    # Load the shared library into c types.
    if sys.platform.startswith("win"):
        c_lib = ctypes.CDLL(libname / "ccd.dll")
    else:
        c_lib = ctypes.CDLL(libname / "libccd.so")

    # Sample data for our call:
    x, y = 6, 2.3

    # You need tell ctypes that the function returns a float
    c_lib.cppmult.restype = ctypes.c_float
    answer = c_lib.cppmult(x, ctypes.c_float(y))
    print(f"    In Python: int: {x} float {y:.1f} return val {answer:.1f}")
