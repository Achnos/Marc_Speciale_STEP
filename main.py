import os
import ctypes
import pathlib

import CCD

if __name__ == '__main__':

    lookingAt = 'focus_3/focus_9.fits' # 'saturn2.fits'

    test_camera = CCD.CCD("test_name")
    test_camera.__linearity("/home/marc/Dropbox/STEP_Speciale_Marc/FITS/focus_3/")
    

