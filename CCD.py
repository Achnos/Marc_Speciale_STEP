import os
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_file(path):
    """
    This function returns the absolute path of a file

    :param str path:
        - path to the file, specified as a string datatype
    :return: PureWindowsPath or PurePosixPath object
        - type depends on the operating system in use

    """

    def get_project_root() -> Path:
        """Returns project root folder."""
        return Path(__file__).parent.parent

    return get_project_root().joinpath(path)


def fits_handler(filepath, scalelimit):
    """
    This function handles the loading and plotting of a FITS file

    :param path filepath:
        - The absolute path to the file of interest
    :param dbl scalelimit:
        - Upper limit of the normalization of the image produced, numerical value
    :return:
        - hdulist, an HDU list, an astropy type
        - header, the header information from the HDU list
        - imagedata, 2D list (matrix) of image data numerical values

    """

    hdulist = astropy.io.fits.open(filepath)
    hdulist.info()
    header = hdulist[0].header
    imagedata = hdulist[0].data

    print(repr(header))  # For printing the header of the HDU
    print("The largest numerical value in the image is: " + str(repr(max(np.ndarray.flatten(imagedata)))))

    # norm_vmax = max(data[:])

    #plt.imshow(imagedata, vmin=0, vmax=scalelimit, cmap='gray')
    #plt.colorbar()
    #plt.show()

    return hdulist, header, imagedata


class CCD:
    name = []
    __name = name

    def __init__(self, name):
        print("Initiallizing CCD with name: " + name)
        self.__name = name

    def characterize(self):
        print("Initiallizing characterization of CCD...")

        self.noise()
        #self.linearity()
        self.charge_transfer_efficiency()
        self.charge_diffusion()
        self.quantum_efficiency()

    def noise(self):
        print("Characterizing noise...")
        self.dark_current()
        self.readout_noise()

    def dark_current(self):
        print("Computing dark current levels...")

    def readout_noise(self):
        print("Computing readout noise levels...")

    def linearity(self, path_of_data):
        data_series = os.listdir(path_of_data)
        print(data_series)

        for file in data_series:
            scaleLimit = 223
            filePath = get_file(path_of_data + file)

            hdul, hdr, data = fits_handler(filePath, scaleLimit)
            hdul.close()
        
        print("Characterizing linearity...")

    __linearity = linearity

    def charge_transfer_efficiency(self):
        print("Characterizing charge transfer efficiency...")

    def charge_diffusion(self):
        print("Characterizing charge diffusion rates")

    def quantum_efficiency(self):
        print("Testing quantum efficiency")


