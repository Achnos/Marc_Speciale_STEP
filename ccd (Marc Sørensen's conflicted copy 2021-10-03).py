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
import numpy as np
import utilities as util


class CCD:
    """
    A general CCD class, that will contain a number of useful functions to do
    data analysis of image data from a Charge Coupled Device (CCD). Will be
    used to characterize CCD's.

    :parameter str name:
        - String representing the name of the CCD instance
    :parameter float gain_factor:
        - The gain factor of the CCD, either from specs or found experimentally
          default value is 1.
    :parameter np.ndarray master_bias:
        - Numpy array representing the master bias image
          for use in corrections
    :parameter np.ndarray master_dark:
        - Numpy array representing the master dark current image
          for use in corrections
    :parameter np.ndarray master_flat:
        - Numpy array representing the master flat field image
          for use in corrections
    :parameter np.ndarray linearity:
        - Numpy array of data from the CCD linearity test
    :parameter np.ndarray dark_current_versus_temperature:
        - Numpy array of data from the CCD dark current temperature test
    :parameter np.ndarray readout_noise_versus_temperature:
        - Numpy array of data from the CCD readout noise levels temperature test
    """
    name: str = []
    __name: str = name

    gain_factor: float = 1

    master_bias: np.ndarray = []
    master_dark: np.ndarray = []
    master_flat: np.ndarray = []

    linearity: np.ndarray = []

    dark_current_versus_temperature: np.ndarray = []
    readout_noise_versus_temperature: np.ndarray = []

    readout_noise_level: float = []

    def __init__(self, name: str, gain_factor: float):
        """
        Constructor member function
        :parameter str name:
            - String representing the name of the CCD instance
        :parameter float gain_factor:
            - Float representing the gain factor of the CCD instance
        """
        print("\nInitiallizing CCD with ")
        print(" Name:", name)
        print(" Gain:", gain_factor)

        self.name           =   name
        self.__name         =   name
        self.gain_factor    =   gain_factor

        print("")

    def characterize(self):
        print("Initializing characterization of CCD...")

        # self.noise()
        # self.linearity()
        # self.charge_transfer_efficiency()
        # self.charge_diffusion()
        # self.quantum_efficiency()

    def noise(self):
        print("Characterizing noise...")
        # self.dark_current()
        # self.readout_noise()

    def master_bias_image(self, path_of_data_series: str):
        """
        Method that will set the class member np.ndarray master_bias

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master bias image
        """
        print("Constructing the master bias correction image...")

        data_series         =   util.list_data(path_of_data_series)
        mean_image_return   =   util.mean_image(data_series, path_of_data_series)

        self.master_bias = mean_image_return

    def bias_correction(self, image_to_be_corrected: np.ndarray):
        """
        Method that will apply the bias correction to an image
        by subtracting the master bias image

        :parameter np.ndarray image_to_be_corrected:
            - A numpy array which is the image data to be corrected
        """
        corrected_image = np.subtract(image_to_be_corrected, self.master_bias)
        return corrected_image

    def master_dark_current_image(self, path_of_data_series: str, exposure_time: float):
        """
        Method that will set the class member np.ndarray master_dark

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master dark current image
        :param exposure_time:
            - The exposure time in seconds of the data_series
        """
        print("Constructing the master dark current correction image...")

        data_series         =   util.list_data(path_of_data_series)
        dim_path            =   util.get_path(path_of_data_series + data_series[0])
        image_shape         =   util.get_dims(dim_path)
        mean_image_array    =   np.zeros(image_shape)
        number_of_images    =   len(data_series)

        for imageid in data_series:
            filepath                    =   util.get_path(path_of_data_series + imageid)
            hdul, header, imagedata     =   util.fits_handler(filepath)
            imagedata                   =   self.bias_correction(imagedata)

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                imagedata   =   np.divide(imagedata, exposure_time)
            else:
                print("master_dark_current_image(): Error, exposure time does not match up")
                print("master_dark_current_image(): Exposure time was: ", header['EXPTIME'])

            mean_image_array += imagedata
            hdul.close()

        mean_image_array   /=   number_of_images

        self.master_dark    =   mean_image_array

    def dark_current_correction(self, image_to_be_corrected: np.ndarray):
        """
        Method that will apply the dark current correction to an image
        by subtracting the master dark current image

        :parameter np.ndarray image_to_be_corrected:
            - A numpy array which is the image data to be corrected
        """
        corrected_image = np.subtract(image_to_be_corrected, self.master_dark)
        return corrected_image

    def dark_current_vs_temperature(self, path_of_data_series: str, exposure_time: float):
        """
        Method which will compute the dark current levels as a function
        of temperature, which it returns as a list, and fills the member
        dark_current_versus_temperature with as well. Implies a certain
        file naming convention of the type

            nnn_s_tt_d_rrr.fit

              - where nnn is an arbitrary descriptive string, s is
                a character representing the sign of the tempreature
                either m or p, representing plus or minus (degrees),
                tt is the temperature in degrees, while d is the decimal.
                rrr represents the repeat number of the given data file.

            An example of this:

            thermal_noise_celcius_m_04_9_000.fit

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the data points
        :param exposure_time:
            - The exposure time in seconds of the data_series
        :returns np.ndarray dark_current_versus_temperature_return:
            - A numpy array of data points of the form (temperature, value)
        """
        print("Computing dark current as a function of temperature...")

        data_series = os.listdir(path_of_data_series)
        tmplist = []

        for imageid in data_series:
            filepath                    =   util.get_path(path_of_data_series + imageid)
            hdul, header, imagedata     =   util.fits_handler(filepath)
            imagedata                   =   self.bias_correction(imagedata)
            #print(imagedata)

            # The temperature as a string
            temp = float(imageid[-12:-10] + '.' + imageid[-9])
            if imageid[-14] == "m":
                temp *= -1

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                dark_per_time_per_pixel = (np.mean(imagedata) * self.gain_factor) / exposure_time
                tmplist.append([temp, dark_per_time_per_pixel])
            else:
                print("dark_current_vs_temperature(): Error, exposure times do not match up")
                print("dark_current_vs_temperature(): Exposure time was: ", header['EXPTIME'])

            hdul.close()

        tmparray = np.sort(np.asarray(tmplist), axis=0)

        self.dark_current_versus_temperature    = tmparray
        dark_current_versus_temperature_return  = tmparray

        return dark_current_versus_temperature_return

    def master_flat_field_image(self, path_of_data_series: str):
        """
        Method that will set the class member np.ndarray master_flat

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master flat field image
        """
        print("Constructing the master flat field correction image...")

        data_series         =   util.list_data(path_of_data_series)
        dim_path            =   util.get_path(path_of_data_series + data_series[0])
        image_shape         =   util.get_dims(dim_path)
        number_of_images    =   len(data_series)
        meaned_flat         =   np.zeros(image_shape)

        for imageid in data_series:
            filepath                    =   util.get_path(path_of_data_series + imageid)
            hdul, header, imagedata     =   util.fits_handler(filepath)
            imagedata                   =   self.bias_correction(imagedata)

            corrected_image             =   self.dark_current_correction(imagedata)
            meaned_flat                +=   corrected_image

            hdul.close()

        meaned_flat /= number_of_images
        meaned_flat /= np.max(meaned_flat)  # Normalize

        self.master_flat = meaned_flat

    def flat_field_correction(self, image_to_be_corrected: np.ndarray):
        """
        Method that will apply the flat field correction to an image
        by subtracting the master flat field image

        :parameter np.ndarray image_to_be_corrected:
            - A numpy array which is the image data to be corrected
        """
        corrected_image = np.divide(image_to_be_corrected, self.master_flat)
        return corrected_image

    def readout_noise_estimation(self, path_of_data_series: str, temperature: float):
        # Consider a row, plot the ADU as a function of
        # column number. Add all rows in an area that is
        # consistent, and mean. Fit power law, and noise
        # is then the width of the fit distribution.
        print("Computing readout noise level...")

        data_series         =   util.list_data(path_of_data_series)
        number_of_images    =   len(data_series)
        tmp_std             =   np.zeros(number_of_images)
        tmp_mean            =   np.zeros(number_of_images)

        it = 0
        for imageid in data_series:
            filepath                    =   util.get_path(path_of_data_series + imageid)
            hdul, header, imagedata     =   util.fits_handler(filepath)

            # Check for consistency
            if header['CCD-TEMP'] == temperature:
                noise_deviation     =   np.subtract(self.master_bias, imagedata) * self.gain_factor
                tmp_std[it]         =   np.std(noise_deviation) / np.sqrt(2)
                tmp_mean[it]        =   np.mean(noise_deviation)

            hdul.close()
            it += 1

        readout_noise  =  np.sqrt(np.mean(np.square(tmp_std)))

        print(f"The readout noise level is {readout_noise:.3f} RMS electrons per pixel.")
        self.readout_noise_level = readout_noise

    def readout_noise_vs_temperature(self):
        print("Computing readout noise as a function of temperature...")
        pass

    def linearity_estimation(self, path_of_data_series: str, num_of_exposures: int, num_of_repeats: int):
        """
        Method which will test the linearity, by plotting mean ADU in an image
        as a function of exposure time, which it returns as a list,
        and fills the member linearity with as well. Implies a certain
        file naming convention of the type

            nnn_rrr_eee.fit

              - where nnn is an arbitrary descriptive string, rrr
                is the number of repeats, and eee is the exposure
                time in seconds

            An example of this:

            linearity_001_008.fit

        :parameter path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the data points
        :param int num_of_repeats:
            - Integer representing the total number of repeats of the data sequence
        :param int num_of_exposures:
            - Integer representing the number of different exposure times
              in the data sequence
        :returns np.ndarray linearity_array:
            - A numpy array of data points of the form (exposure time, mean ADU)
        """
        print("Testing linearity...")

        tmplist = []

        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   np.empty([num_of_exposures, num_of_repeats], dtype=object)

        index = 0
        for imageid in data_series:
            repeat_num = int(imageid[10:13])
            reordered_data[index][repeat_num] = str(imageid)
            index += 1
            if index == 10:
                index = 0

        for repeat_sequence in reordered_data:
            repeat_sequence_meaned  =  util.mean_image(repeat_sequence, path_of_data_series) # self.flat_field_correction(self.bias_correction(util.mean_image(repeat_sequence, path_of_data_series)))

            filepath                    =   util.get_path(path_of_data_series + repeat_sequence[0])
            hdul, header, imagedata     =   util.fits_handler(filepath)
            exposure_time               =   float(repeat_sequence[0][-7:-4])  # time in s
            print(int(exposure_time), "%")
            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned)])
            else:
                print("linearity_estimation(): Error, exposure times do not match up")
                print("linearity_estimation(): Exposure time was: ", header['EXPTIME'], "should have been: ", exposure_time)

        linearity_array     =   np.asarray(tmplist)
        self.linearity      =   linearity_array

        return linearity_array

    def linearity_precision(self):


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
