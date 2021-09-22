import os
import astropy
import numpy
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PureWindowsPath, PurePosixPath
import math


def get_path(path):
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


def fits_handler(filepath, scalelimit=None, show=False):
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
    #hdulist.info()
    header = hdulist[0].header
    imagedata = hdulist[0].data

    # print(repr(header))  # For printing the header of the HDU
    #print("The largest numerical value in the image is: " + str(repr(max(np.ndarray.flatten(imagedata)))))

    # norm_vmax = max(data[:])
    if show:
        plt.imshow(imagedata, vmin=0, vmax=scalelimit, cmap='gray')
        plt.colorbar()
        plt.savefig("test.pdf")

    return hdulist, header, imagedata


def get_dims(filepath: PureWindowsPath or PurePosixPath):
    hdul, header, imagedata = fits_handler(filepath)
    extracted_dims = imagedata.shape
    return extracted_dims


def list_data(dirpath: PureWindowsPath or PurePosixPath):
    data_list = os.listdir(dirpath)
    data_list.sort()

    return data_list


def mean_image(filelist: list, dirpath: str):
    dim_path = get_path(dirpath + filelist[0])
    image_shape = get_dims(dim_path)
    number_of_images = len(filelist)
    mean_image_array = np.zeros(image_shape)

    for imageid in filelist:
        filepath = get_path(dirpath + imageid)
        hdul, header, imagedata = fits_handler(filepath)
        mean_image_array += imagedata
        hdul.close()

    mean_image_array /= (1 / number_of_images)
    return mean_image_array


class CCD:
    name: str = []
    __name: str = name
    master_dark: np.ndarray = []
    master_bias: np.ndarray = []
    master_flat: np.ndarray = []
    linearity: np.ndarray = []
    dark_current_versus_temperature: np.ndarray = []
    readout_noise: np.ndarray = []

    def __init__(self, name):
        print("Initiallizing CCD with name: " + name)
        self.name = name
        self.__name = name

    def characterize(self):
        print("Initiallizing characterization of CCD...")

        #self.noise()
        # self.linearity()
        #self.charge_transfer_efficiency()
        #self.charge_diffusion()
        #self.quantum_efficiency()

    def noise(self):
        print("Characterizing noise...")
        # self.dark_current()
        # self.readout_noise()

    def dark_current_vs_temperature(self, path_of_data_series):
        print("Computing dark currents...")

        data_series = os.listdir(path_of_data_series)
        dark_current_versus_temperature_return = []
        tmplist = []

        for imageid in data_series:
            #imageid is a string with the filename ending with .fit

            filepath = get_path(path_of_data_series + imageid)
            hdul, header, imagedata = fits_handler(filepath)
            temp = float(imageid[-12:-10] + '.' + imageid[-9])  # The temperature as a string
            if imageid[-14] == "m":
                temp *= -1
            tmplist.append([temp, np.mean(imagedata)])

            hdul.close()

        tmparray = np.sort(np.asarray(tmplist), axis=0)
        self.dark_current_versus_temperature = tmparray
        dark_current_versus_temperature_return = tmparray
        return dark_current_versus_temperature_return

    def master_dark_current_image(self, path_of_data_series):
        data_series = list_data(path_of_data_series)
        mean_image_return = mean_image(data_series, path_of_data_series)

        self.master_dark = mean_image_return
        return mean_image_return

    def dark_current_correction(self, image_to_be_corrected):
        corrected_image = np.subtract(image_to_be_corrected, self.master_dark)
        return corrected_image

    def master_bias_image(self, path_of_data_series):
        print("Constructing mean bias image...")

        data_series = list_data(path_of_data_series)
        mean_image_return = mean_image(data_series, path_of_data_series)

        self.master_bias = mean_image_return
        return mean_image_return

    def bias_correction(self, image_to_be_corrected):
        corrected_image = np.subtract(image_to_be_corrected, self.master_bias)
        return corrected_image

    def master_flat_field_image(self, path_of_data_series):
        print("Constructing the flat field correction...")
        data_series = list_data(path_of_data_series)
        dim_path = get_path(path_of_data_series + data_series[0])
        image_shape = get_dims(dim_path)
        number_of_images = len(path_of_data_series)
        meaned_flat = np.zeros(image_shape)

        for imageid in data_series:
            filepath = get_path(path_of_data_series + imageid)
            hdul, header, imagedata = fits_handler(filepath)
            corrected_image = dark_current_correction(imagedata)
            meaned_flat += corrected_image
            hdul.close()

        meaned_flat /= (1 / number_of_images)

        self.master_flat = meaned_flat

    def flat_field_correction(self, image_to_be_corrected):
        corrected_image = np.divide(image_to_be_corrected, self.master_flat)
        return corrected_image

    def readout_noise_estimation(self, path_of_data_series):
        # Consider a row, plot the ADU as a function of
        # column number. Add all rows in an area that is
        # consistent, and mean. Fit power law, and noise
        # is then the width of the fit distribution.
        print("Computing readout noise levels...")

        data_series = list_data(path_of_data_series)

        first_filepath = get_path(path_of_data_series + data_series[0])
        image_shape = get_dims(first_filepath)
        tmp_image_array = np.zeros(image_shape)
        tmp_std = 0
        tmp_mean = 0

        for first_imageid in range(0, len(path_of_data_series)):
            first_hdul, first_header, first_imagedata = fits_handler(first_filepath)

            for next_imageid in range(first_imageid + 1, len(path_of_data_series)):
                next_filepath = get_path(path_of_data_series + data_series[next_imageid])
                hdul, header, next_imagedata = fits_handler(next_filepath)

                noise_deviation = np.subtract(first_imagedata, next_imagedata)
                tmp_std += np.std(noise_deviation) / np.sqrt(2)
                tmp_mean += np.mean(noise_deviation)

        tmp_std /= math.factorial(len(path_of_data_series))
        tmp_mean /= math.factorial(len(path_of_data_series))

        return tmp_std

    def linearity_estimation(self, path_of_data_series, num_of_exposures: int, num_of_repeats: int):
        print("Testing linearity...")

        tmplist = []

        bias_image = self.master_bias
        readout_noise = self.readout_noise

        data_series = list_data(path_of_data_series)
        reordered_data = numpy.empty([num_of_exposures, num_of_repeats], dtype=object)

        index = 0
        for imageid in data_series:
            repeat_num = int(imageid[10:13])-1
            reordered_data[index][repeat_num] = str(imageid)
            index += 1
            if index == 10:
                index = 0

        for repeat_sequence in reordered_data:
            repeat_sequence_meaned = self.flat_field_correction(self.bias_correction(mean_image(repeat_sequence, path_of_data_series)))
            exposure_time = float(repeat_sequence[0][14] + '.' + repeat_sequence[0][15:17])  # time in s
            tmplist.append([exposure_time, np.mean(repeat_sequence_meaned)])

        tmparray = np.asarray(tmplist)
        self.linearity = tmparray
        return tmparray

    def charge_transfer_efficiency(self):
        """
        Consider read out direction, plot mean ADU as function of
        rows (if readout is down/up-wards). Say it is linear, we
        can fit to this, and get the linear rate of loss per row
        """
        print("Characterizing charge transfer efficiency...")

    def charge_diffusion(self):
        print("Characterizing charge diffusion rates")

    def quantum_efficiency(self):
        print("Testing quantum efficiency")
