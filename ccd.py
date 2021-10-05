"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           August    2021 ----- #
# ----- Last edit:          5. October   2021 ----- #
# ------------------------------------------------- #
#####################################################
"""
import math
import os
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import pubplot as pp


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
    :parameter np.ndarray hot_pixel_mask:
        - A mask array of indices of hot pixels in the camera
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

    hot_pixel_mask: np.ndarray = []

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
        :parameter exposure_time:
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

    def dark_current_vs_temperature(self, path_of_data_series: str, exposure_time: float, num_of_repeats: int, num_of_temperatures: int,):
        """
        Method which will compute the dark current levels as a function
        of temperature, which it returns as a list, and fills the member
        dark_current_versus_temperature with as well. Implies a certain
        file naming convention of the type

            noise_rrr_dc_s_tt_d.fit

              - where nnn is an arbitrary descriptive string, s is
                a character representing the sign of the temperature
                either m or p, representing plus or minus (degrees),
                tt is the temperature in degrees, while d is the decimal.
                rrr represents the repeat number of the given data file.

            An example of this:

            noise_000_dc_m_04_9.fit

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the data points
        :parameter exposure_time:
            - The exposure time in seconds of the data_series
        :parameter int num_of_repeats:
            - Integer representing the total number of repeats of the data sequence
        :parameter int num_of_temperatures:
            - Integer representing the number of different temperatures
              in the data sequence
        :returns np.ndarray dark_current_versus_temperature_return:
            - A numpy array of data points of the form (temperature, value)
        """
        print("Computing dark current as a function of temperature...")

        tmplist         =   []
        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=num_of_temperatures,
                                                                num_of_repeats_input=num_of_repeats,
                                                                where_is_repeat_num_in_string=[6, 9],
                                                                data_series_list=data_series)

        for repeat_sequence in reordered_data:
            repeat_sequence_meaned      =   self.bias_correction(util.mean_image(repeat_sequence, path_of_data_series))

            filepath                    =   util.get_path(path_of_data_series + repeat_sequence[0])
            hdul, header, imagedata     =   util.fits_handler(filepath)
            temperature                 =   float(float(repeat_sequence[0][15:17] + '.' + repeat_sequence[0][18]))  # time in s
            if repeat_sequence[0][13] == "m":
                temperature *= -1

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                dark_per_time_per_pixel = np.mean(np.divide(np.multiply(repeat_sequence_meaned, self.gain_factor), exposure_time))
                tmplist.append([temperature, dark_per_time_per_pixel])
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

    def hot_pixel_estimation(self, path_of_data_series: str, num_of_repeats: int, exposure_time: list):
        """
        Method to find hot pixels qualitatively, and then construct a mask used to remove them

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master flat field image
        :parameter int num_of_repeats:
            - Integer representing the total number of repeats of the data sequence
        :parameter list exposure_time:
            - list of shape [exposure time of short exposure image, exposure time of long exposure image]
        """
        print("Looking for hot pixels...")

        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=2,
                                                                num_of_repeats_input=num_of_repeats,
                                                                where_is_repeat_num_in_string=[7, 8],
                                                                data_series_list=data_series          )

        # Construct the two images from the data files in the path
        short_exposure_image    =   util.mean_image(reordered_data[0].tolist(), path_of_data_series)
        long_exposure_image     =   util.mean_image(reordered_data[1].tolist(), path_of_data_series)

        # Apply bias correction
        short_exposure_image    =   self.bias_correction(short_exposure_image)
        long_exposure_image     =   self.bias_correction(long_exposure_image)

        # Convert ADU to dark current
        short_exposure_image    =   np.divide(np.multiply(short_exposure_image, self.gain_factor), exposure_time[0])
        long_exposure_image     =   np.divide(np.multiply(long_exposure_image , self.gain_factor), exposure_time[1])

        smallest_measurable_dark_current = 2 * ((self.readout_noise_level / math.sqrt(num_of_repeats)) / exposure_time[0])
        potential_hot_pixels = long_exposure_image > smallest_measurable_dark_current

        plt.plot(short_exposure_image[potential_hot_pixels].flatten(), long_exposure_image[potential_hot_pixels].flatten(), '.', c="k", label='Data')
        plt.plot([0, 1e3], [0, 1e3], c="dodgerblue", label='Ideal relationship')
        pp.pubplot("Hot pixels", "dark current ($e^-$/sec), 90 sec exposure time", "dark current ($e^-$/sec), 1000 sec exposure time", "hot_pixels_test.png", xlim=[0.5, 100.0], ylim=[0.5, 20], legendlocation="lower right")

        hot_pixels = (short_exposure_image > 7.5)
        print("No. of hot pixels:", hot_pixels.sum())

        self.hot_pixel_mask = hot_pixels

    def hot_pixel_correction(self, image_to_be_corrected: np.ndarray):
        """
        Method that will apply the hot pixel correction to an image
        by subtracting the hot pixel correction mask

        :parameter np.ndarray image_to_be_corrected:
            - A numpy array which is the image data to be corrected
        """
        corrected_image = np.subtract(image_to_be_corrected, image_to_be_corrected[self.hot_pixel_mask])
        return corrected_image

    def readout_noise_estimation(self, path_of_data_series: str, temperature: float):
        """
        Method to compute the readout noise level at a given temperature

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master flat field image
        :parameter float temperature:
            - Integer representing the temperature at which the data series
              was acquired
        """
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
            check_temperature = (temperature*1.1 <= header['CCD-TEMP'] <= temperature*0.9) or (temperature*1.1 >= header['CCD-TEMP'] >= temperature*0.9)
            if check_temperature:
                noise_deviation     =   np.subtract(self.master_bias, imagedata) * self.gain_factor
                tmp_std[it]         =   np.std(noise_deviation) / np.sqrt(2)
                tmp_mean[it]        =   np.mean(noise_deviation)
            else:
                print(header['CCD-TEMP'])

            hdul.close()
            it += 1

        readout_noise  =  np.sqrt(np.mean(np.square(tmp_std)))

        print(f"The readout noise level is {readout_noise:.3f} RMS electrons per pixel.")
        self.readout_noise_level = readout_noise

    def readout_noise_vs_temperature(self, path_of_data_series, num_of_temperatures, num_of_repeats, exposure_time):
        print("Computing readout noise as a function of temperature...")
        data_series         =   util.list_data(path_of_data_series)
        tmp_std             =   np.zeros(num_of_repeats)
        tmp_mean            =   np.zeros(num_of_repeats)
        reordered_data      = util.repeat_sequence_ordered_data(num_of_temperatures, num_of_repeats,
                                                                where_is_repeat_num_in_string=[6, 9],
                                                                data_series_list=data_series)

        readout_noise = []
        for repeat_sequence in reordered_data:

            temperature = float(repeat_sequence[0][16:18] + '.' + repeat_sequence[0][19])  # time in s
            if repeat_sequence[0][14] == "m":
                temperature *= -1

            it = 0
            for imageid in repeat_sequence:
                filepath = util.get_path(path_of_data_series + imageid)
                hdul, header, imagedata = util.fits_handler(filepath)

                noise_deviation     =   np.subtract( np.mean(imagedata), imagedata) * self.gain_factor # self.master_bias
                tmp_std[it]         =   np.std(noise_deviation)  # / np.sqrt(2)
                tmp_mean[it]        =   np.mean(noise_deviation)

                hdul.close()
                it += 1

                if it == num_of_repeats:
                    it = 0

            readout_noise.append([temperature, np.sqrt(np.mean(np.square(tmp_std)))])

        tmparray = np.sort(np.asarray(readout_noise), axis=0)

        # self.ron_vs_temperature = tmparray
        redout_noise_versus_temperature_return = tmparray
        return redout_noise_versus_temperature_return

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
        :parameter int num_of_repeats:
            - Integer representing the total number of repeats of the data sequence
        :parameter int num_of_exposures:
            - Integer representing the number of different exposure times
              in the data sequence
        :returns np.ndarray linearity_array:
            - A numpy array of data points of the form (exposure time, mean ADU)
        """
        print("Testing linearity...")

        tmplist = []

        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=num_of_exposures,
                                                                num_of_repeats_input=num_of_repeats,
                                                                where_is_repeat_num_in_string=[10, 13],
                                                                data_series_list=data_series)

        stabillity_data =   self.test_lightsource_stabillity(path_of_data_series, num_of_exposures, num_of_repeats)

        first_it = 0
        second_it = 0
        for repeat_sequence in reordered_data:
            # repeat_sequence_meaned      =   util.mean_image(repeat_sequence, path_of_data_series)
            dim_path = util.get_path(path_of_data_series + repeat_sequence[0])
            image_shape = util.get_dims(dim_path)
            number_of_images = len(repeat_sequence)
            mean_image_array = np.zeros(image_shape)

            for imageid in repeat_sequence:
                filepath                        =   util.get_path(path_of_data_series + imageid)
                hdul, header, imagedata         =   util.fits_handler(filepath)
                imagedata_meaned                =   self.bias_correction(imagedata)
                imagedata_meaned_and_corrected  =   np.divide(imagedata_meaned, (stabillity_data[first_it, second_it] / 100 + 1))

                mean_image_array += imagedata_meaned_and_corrected

                hdul.close()
                second_it += 1
                if second_it == num_of_repeats:
                    second_it = 0

            first_it += 1

            mean_image_array /= number_of_images

            # repeat_sequence_meaned      =   self.bias_correction(repeat_sequence_meaned)  # self.flat_field_correction(  ,   util.mean_image(repeat_sequence, path_of_data_series)

            repeat_sequence_meaned      =   mean_image_array
            filepath                    =   util.get_path(path_of_data_series + repeat_sequence[0])
            hdul, header, imagedata     =   util.fits_handler(filepath)
            exposure_time               =   float(repeat_sequence[0][-7:-4])  # time in s

            # Treat hot pixels
            repeat_sequence_meaned[np.where(self.hot_pixel_mask)] = np.mean(repeat_sequence_meaned[np.where(np.logical_not(self.hot_pixel_mask))])

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned)])
            else:
                print("linearity_estimation(): Error, exposure times do not match up")
                print("linearity_estimation(): Exposure time was: ", header['EXPTIME'], "should have been: ", exposure_time)

                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned)])\

        linearity_array     =   np.asarray(tmplist)

        print("Done! Data constructed from linearity measurements:")
        print(linearity_array)

        self.linearity      =   linearity_array

        return linearity_array

    def test_lightsource_stabillity(self, path_of_data_series: str, num_of_data_points: int, num_of_repeats:int):
        print("Testing light source stabillity...")
        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=num_of_data_points,
                                                                num_of_repeats_input=num_of_repeats,
                                                                where_is_repeat_num_in_string=[10, 13],
                                                                data_series_list=data_series                )

        plot_data   =   np.empty([num_of_data_points, num_of_repeats])

        data_point  =   0
        repeat      =   0
        for sequence in reordered_data:
            for image in sequence:
                filepath = util.get_path(path_of_data_series + image)
                hdul, header, imagedata = util.fits_handler(filepath)
                if data_point > 0:
                    self.bias_correction(imagedata)
                mean_adu = np.mean(imagedata)

                plot_data[data_point, repeat] = mean_adu
                repeat += 1
                if repeat == num_of_repeats:
                    repeat = 0

            data_point += 1

        thismean = np.mean(plot_data, axis=1, keepdims=True)
        plot_data = np.multiply(np.divide(np.abs(plot_data - thismean), thismean), 100)

        return plot_data

    def linearity_precision(self):
        print("Testing the precision of the linearity measurement...")
        linearity_data      =   self.linearity[:, 1]
        query_points        =   self.linearity[:, 0]
        reference_point     =   self.linearity[1, 1]

        ideal_slope         =   reference_point / query_points[1]
        ideal_offset        =   0

        ideal_linearity     =   np.add(np.multiply(query_points, ideal_slope), ideal_offset)
        deviations          =   np.multiply(np.divide(np.subtract(ideal_linearity, linearity_data), linearity_data), 100)
        deviations[0]       =   np.subtract(ideal_linearity[0], linearity_data[0])

        return ideal_linearity, deviations

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
