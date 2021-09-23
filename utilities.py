"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           August    2021 ----- #
# ----- Last edit:         23. September 2021 ----- #
# ------------------------------------------------- #
#####################################################
"""
import os
import pathlib
import numpy as np
import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath, PurePosixPath


def get_path(path: str):
    """
    This function returns the absolute path of a file

    :parameter str path:
        - path to the file, specified as a string datatype
    :returns: PureWindowsPath or PurePosixPath object:
        - type depends on the operating system in use

    """

    def get_project_root() -> Path:
        """Returns project root folder."""
        return Path(__file__).parent.parent

    return get_project_root().joinpath(path)


def complete_path(dir_path: str):
    """
    Method to complete a path, given only a string that represents
    the name of a directory in the same folder as this program

    :parameter str dir_path:
        - path to the directory, specified as a string datatype
    :returns str completed_path:
        - The completed path as a string
    """
    path_here       =   str(pathlib.Path().absolute())
    completed_path  =   path_here + "/" + dir_path + "/"

    return completed_path


def fits_handler(filepath: PureWindowsPath or PurePosixPath, scalelimit: float=None, show: bool=False):
    """
    This function handles the loading and plotting of a FITS file

    :parameter path filepath:
        - The absolute path to the file of interest
    :parameter dbl scalelimit:
        - Upper limit of the normalization of the image produced, numerical value
    :returns:
        - hdulist, an HDU list, an astropy type
        - header, the header information from the HDU list
        - imagedata, 2D list (matrix) of image data numerical values

    """
    hdulist     =   astropy.io.fits.open(filepath)
    header      =   hdulist[0].header
    imagedata   =   hdulist[0].data

    if show:
        plt.imshow(imagedata, vmin=0, vmax=scalelimit, cmap='gray')
        plt.colorbar()
        plt.savefig("test.pdf")

    return hdulist, header, imagedata


def get_dims(filepath: PureWindowsPath or PurePosixPath):
    """
    This is a helper function that returns the dimensions of
    the data we are interested in, found at the location of
    filepath

    :parameter filepath:
        - The path of the (series of) data that we are
          interested in obtaining the dimensionality of
    :returns tuple extracted_dims:
        - A tuple of [a, b, ....] fit for numpy, that
          represents the dimensionality of the data
    """
    hdul, header, imagedata = fits_handler(filepath)
    extracted_dims = imagedata.shape
    return extracted_dims


def list_data(dirpath: PureWindowsPath or PurePosixPath):
    """
    This is a helper function that returns a list of filenames
    of the data, in the data series found in the directory
    specified by dirpath.

    :parameter dirpath:
        - The path of the directory containing the (series of)
          data that we wish to list the filenames of
    :returns list data_list:
        - A list of filenames found in the directory specified
          by dirpath
    """
    data_list = os.listdir(dirpath)
    data_list.sort()

    return data_list


def mean_image(filelist: list, dirpath: str):
    """
    This is a helper function that returns the mean image from
    a data series of images in .fit(s) file format. Use list_data()
    to generate parameter filelist

    :parameter filelist:
        - A list containing filenames to be imported by
          fits_handler() as images, type np.ndarray numpy arrays,
          which will then be meaned over. Generate with list_data()
    :parameter dirpath:
        - The path of the directory containing the images we wish
          to mean over. A sequence of data in the form of .fit(s)
          files. Use list_data() in conjunction to obtain filelist
    :returns np.ndarray mean_image_array:
        - The mean image constructed form the data series
    """
    dim_path            =   get_path(dirpath + filelist[0])
    image_shape         =   get_dims(dim_path)
    number_of_images    =   len(filelist)
    mean_image_array    =   np.zeros(image_shape)

    for imageid in filelist:
        filepath                    =   get_path(dirpath + imageid)
        hdul, header, imagedata     =   fits_handler(filepath)

        mean_image_array += imagedata

        hdul.close()

    mean_image_array /= (1 / number_of_images)
    return mean_image_array
