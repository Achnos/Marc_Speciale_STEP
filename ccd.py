"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           August    2021 ----- #
# ----- Last edit:          8. April     2022 ----- #
# ------------------------------------------------- #
#####################################################
"""
import math
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import pubplot as pp

plt.style.use(['science', 'ieee', 'vibrant'])


class DataSequence:
    """
    A general class to store informations about a data sequence.

    :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the data points
    :parameter int num_of_datapoints:
            - Integer representing the number of different datapoints
              in the data sequence
    :parameter int num_of_repeats:
            - Integer representing the total number of repeats of each datapoint
              in the data series
    :parameter exposure_time:
            - The exposure time in seconds of the data_series
    """
    path_of_data_series: str
    num_of_data_points: int
    num_of_repeats: int
    exposure_time: float or list
    cutoff: float
    exposure_list: np.ndarray
    milliseconds: bool = False

    def __init__(self,  path_of_data_series_input: str                  ,
                        num_of_data_points_input: int           = None  ,
                        num_of_repeats_input: int               = None  ,
                        exposure_time_input: float or list      = None  ,
                        cutoff_input: float                     = None  ,
                        exposure_list_input: np.ndarray         = None  ,
                        milliseconds_input: bool                = False ):
        """
            Constructor member function
            :parameter str path_of_data_series_input:
            - A string representing the path to the directory
              containing the data series used to construct the data points
            :parameter int num_of_datapoints_input:
            - Integer representing the number of different datapoints
              in the data sequence
            :parameter int num_of_repeats_input:
            - Integer representing the total number of repeats of each datapoint
              in the data series
            :parameter exposure_time_input:
            - The exposure time in seconds of the data_series
        """
        self.path_of_data_series    =   path_of_data_series_input

        if num_of_data_points_input is not None:
            self.num_of_data_points     =   num_of_data_points_input
        if num_of_repeats_input is not None:
            self.num_of_repeats         =   num_of_repeats_input
        if exposure_time_input is not None:
            self.exposure_time          =   exposure_time_input
        if cutoff_input is not None:
            self.cutoff                 =   cutoff_input
        if exposure_list_input is not None:
            self.exposure_list          =   exposure_list_input
        if milliseconds_input:
            self.milliseconds           =   milliseconds_input


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

    gain_factor: float = 1
    time_calibration_factor: float = 0  # 1/15

    master_bias: np.ndarray = []
    master_dark: np.ndarray = []
    master_flat: np.ndarray = []

    hot_pixel_mask: np.ndarray = []
    hot_pixel_data: np.ndarray = []

    linearity: np.ndarray = []

    dark_current_versus_temperature:  np.ndarray = []
    readout_noise_versus_temperature: np.ndarray = []
    gain_vs_temp:                     np.ndarray = []

    readout_noise_level: float = []

    analysis_data_storage_directory_path:  str = []
    master_frames_storage_directory_path:  str = []
    datastorage_filename_append:           str = []
    figure_directory_path:                 str = []

    construct_master_bias:   bool = True
    construct_master_dark:   bool = True
    construct_master_flat:   bool = True
    do_noise_estimation:     bool = True
    do_time_calibration:     bool = True
    do_linearity_estimation: bool = True
    do_gain_factor_estimation: bool = True

    def __init__(self, name: str, gain_factor: float, analysis_data_path: str, master_frame_path: str, datastorage_filename_append: str, figure_directory_path: str):
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
        self.gain_factor    =   gain_factor

        self.analysis_data_storage_directory_path = analysis_data_path
        self.master_frames_storage_directory_path = master_frame_path
        self.datastorage_filename_append          = datastorage_filename_append
        self.figure_directory_path                = figure_directory_path

        print("")

    def load_ccd_characterization_data( self                                     ,
                                        construct_master_bias:      bool = True  ,
                                        construct_master_dark:      bool = True  ,
                                        construct_master_flat:      bool = True  ,
                                        do_noise_estimation:        bool = True  ,
                                        do_time_calibration:        bool = True  ,
                                        do_linearity_estimation:    bool = True  ,
                                        do_gain_factor_estimation:  bool = True,
                                        path_of_master_bias_frame:   str = None  ,
                                        path_of_master_dark_frame:   str = None  ,
                                        path_of_master_flat_frame:   str = None  ,
                                        path_of_linearity_data:      str = None  ,
                                        path_of_dark_current_data:   str = None  ,
                                        path_of_readout_noise_data:  str = None   ):

        self.construct_master_bias   = construct_master_bias
        self.construct_master_dark   = construct_master_dark
        self.construct_master_flat   = construct_master_flat
        self.do_noise_estimation     = do_noise_estimation
        self.do_time_calibration     = do_time_calibration
        self.do_linearity_estimation = do_linearity_estimation
        self.do_gain_factor_estimation = do_gain_factor_estimation

        if (construct_master_bias is False) and (path_of_master_bias_frame is not None):
            fullpath_master_bias = self.master_frames_storage_directory_path + path_of_master_bias_frame
            print(" Reading master bias frame from:\n ", fullpath_master_bias)
            self.master_bias                        =   np.loadtxt(fullpath_master_bias)
        if (construct_master_dark is False) and (path_of_master_dark_frame is not None):
            fullpath_master_dark = self.master_frames_storage_directory_path + path_of_master_dark_frame
            print(" Reading master dark frame from:\n ", fullpath_master_dark)
            self.master_dark                        =   np.loadtxt(fullpath_master_dark)
        if (construct_master_flat is False) and (path_of_master_flat_frame is not None):
            fullpath_master_flat = self.master_frames_storage_directory_path + path_of_master_flat_frame
            print(" Reading master flat frame from:\n ", fullpath_master_flat)
            self.master_flat                        =   np.loadtxt(fullpath_master_flat)
        if (do_linearity_estimation is False) and (path_of_linearity_data is not None):
            fullpath_linearity = self.analysis_data_storage_directory_path + path_of_linearity_data
            print(" Reading linearity data from:\n ", fullpath_linearity)
            self.linearity                          =   np.loadtxt(fullpath_linearity)
        if (do_noise_estimation is False) and (path_of_readout_noise_data is not None) and (path_of_readout_noise_data is not None):
            fullpath_dark_current = self.analysis_data_storage_directory_path + path_of_dark_current_data
            fullpath_readout_noise = self.analysis_data_storage_directory_path + path_of_readout_noise_data
            print(" Reading dark current temperature data from:\n ", fullpath_dark_current)
            print(" Reading readout noise temperature data from:\n ", fullpath_readout_noise)
            self.dark_current_versus_temperature    =   np.loadtxt(self.analysis_data_storage_directory_path + path_of_dark_current_data)
            self.readout_noise_versus_temperature   =   np.loadtxt(self.analysis_data_storage_directory_path + path_of_readout_noise_data)

    def characterize(self,
                     bias_data_sequence:                DataSequence,
                     flat_data_sequence:                DataSequence,
                     dark_current_data_sequence:        DataSequence,
                     readout_noise_data_sequence:       DataSequence,
                     linearity_data_sequence:           DataSequence,
                     hot_pixel_data_sequence:           DataSequence,
                     zero_point_data_sequence:          DataSequence,
                     time_calibration_data_sequence:    DataSequence,
                     gain_data_sequence:                DataSequence,
                     old_timecal_data_sequence:         DataSequence = None):
        """
            The main interface for the characterization procedure. Calling this
            method will fully characterize the ccd in question. The method will
            call all of the members below, to first construct master frames of
            the bias, flat field and dark current. A preliminary estimation of
            the readout noise levels is computed by readout_noise_estimation().
            A hot pixel mask is then constructed by hot_pixel_estimation().
            The noise of the CCD is then characterized by noise_versus_temperature()
            and the linearity data analyzed and characterized by the methods
            linearity_estimation() and linearity_precision(). Finally the light
            source stability and the CCD preliminary zero point estimation are
            both treated by, respectively, the lightsource_stability() and
            zeropoint_estimation() methods.

            :param DataSequence bias_data_sequence:
                - A datasequence instance representing the bias data series
            :param DataSequence flat_data_sequence:
                - A datasequence instance representing the flat field data series
            :param DataSequence dark_current_data_sequence:
                - A datasequence instance representing the dark current data series
            :param DataSequence readout_noise_data_sequence:
                - A datasequence instance representing the readout noise data series
            :param DataSequence linearity_data_sequence:
                - A datasequence instance representing the linearity data series
            :param DataSequence hot_pixel_data_sequence:
                - A datasequence instance representing the hot pixel data series
            :param DataSequence zero_point_data_sequence:
                - A datasequence instance representing the zero point estimation data series
        """

        print("\nInitializing characterization of CCD: " + self.name + " ...")

        if self.construct_master_bias:
            self.master_bias_image(         path_of_data_series =   bias_data_sequence.path_of_data_series       )
        if self.construct_master_dark:
            self.master_dark_current_image( path_of_data_series =   bias_data_sequence.path_of_data_series      ,
                                            exposure_time       =   bias_data_sequence.exposure_time             )
        if self.construct_master_flat:
            self.master_flat_field_image(   path_of_data_series =   flat_data_sequence.path_of_data_series       )

        self.readout_noise_estimation(  path_of_data_series =   bias_data_sequence.path_of_data_series      ,
                                        temperature         =   -10                                          )

        self.gain_factor_estimation(    path_of_data_series =   flat_data_sequence.path_of_data_series)
        self.hot_pixel_estimation(      path_of_data_series =   hot_pixel_data_sequence.path_of_data_series ,
                                        num_of_repeats      =   hot_pixel_data_sequence.num_of_repeats      ,
                                        exposure_time       =   hot_pixel_data_sequence.exposure_time       ,
                                        hot_pixel_cutoff    =   hot_pixel_data_sequence.cutoff               )
        self.test_zero_point(           path_of_data_series =   zero_point_data_sequence.path_of_data_series,
                                        num_of_data_points  =   zero_point_data_sequence.num_of_data_points,
                                        num_of_repeats      =   zero_point_data_sequence.num_of_repeats)

        if self.do_gain_factor_estimation:
            gain_data = self.gain_vs_temperature(path_of_data_series=gain_data_sequence.path_of_data_series,
                                                 num_of_data_points=gain_data_sequence.num_of_data_points,
                                                 num_of_repeats=gain_data_sequence.num_of_repeats
                                                 )
        else:
            gain_data = []

        if self.do_noise_estimation:
            dark_current_data, readout_noise_data, ron_dists_vs_temp = self.noise_vs_temperature(dark_current_data_sequence, readout_noise_data_sequence)

        else:
            dark_current_data = self.dark_current_versus_temperature
            readout_noise_data = self.readout_noise_versus_temperature

        # ron_dists_vs_temp = []
        if self.do_time_calibration and old_timecal_data_sequence is not None:
            time_calibration = self.time_calibration(path_of_data_series =   old_timecal_data_sequence.path_of_data_series,
                                                     num_of_exposures    =   old_timecal_data_sequence.num_of_data_points,
                                                     num_of_repeats      =   old_timecal_data_sequence.num_of_repeats)
        else:
            time_calibration = []

        self.new_time_calibration( path_of_data_series = time_calibration_data_sequence.path_of_data_series   ,
                                   num_of_repeats      = time_calibration_data_sequence.num_of_repeats        ,
                                   exposures           = time_calibration_data_sequence.exposure_list          )

        if self.do_linearity_estimation:
            linearity_data = self.linearity_estimation_with_reference( path_of_data_series =   linearity_data_sequence.path_of_data_series ,
                                                                       num_of_exposures    =   linearity_data_sequence.num_of_data_points  ,
                                                                       num_of_repeats      =   linearity_data_sequence.num_of_repeats      ,
                                                                       reference_exposure  =   linearity_data_sequence.exposure_time       ,
                                                                       exposures           =   linearity_data_sequence.exposure_list       ,
                                                                       milliseconds        =   linearity_data_sequence.milliseconds         )
        else:
            linearity_data  = self.linearity

        # ideal_linear_relation, linearity_deviations, linearity_dev_err = self.linearity_precision()

        # stabillity_data = self.test_lightsource_stability(path_of_data_series  =   linearity_data_sequence.path_of_data_series,
        #                                                   num_of_data_points   =   linearity_data_sequence.num_of_data_points,
        #                                                   num_of_repeats       =   linearity_data_sequence.num_of_repeats)
        # stabillity_data = []
        return [dark_current_data, readout_noise_data, time_calibration, linearity_data, gain_data]
        #, ideal_linear_relation, linearity_deviations, linearity_dev_err, stabillity_data, ron_dists_vs_temp]

    def noise_vs_temperature(self, dark_current_vars: DataSequence, readout_noise_vars: DataSequence ):
        """
            Fully characterizes the noise of the CCD, both thermal (dark current) and
            readout noise (RON) as a function of temperature

            :param DataSequence dark_current_vars:
                - A datasequence instance representing the dark current data series
            :param DataSequence readout_noise_vars:
                - A datasequence instance representing the readout noise data series
        """

        print(" Characterizing the noise levels of the CCD...")

        path_of_data_series_dark_current    =   dark_current_vars.path_of_data_series
        exposure_time_dark_current          =   dark_current_vars.exposure_time
        num_of_repeats_dark_current         =   dark_current_vars.num_of_repeats
        num_of_temperatures_dark_current    =   dark_current_vars.num_of_data_points

        dark_current_data = self.dark_current_vs_temperature(   path_of_data_series   =   path_of_data_series_dark_current,
                                                                exposure_time         =   exposure_time_dark_current      ,
                                                                num_of_repeats        =   num_of_repeats_dark_current     ,
                                                                num_of_temperatures   =   num_of_temperatures_dark_current  )

        path_of_data_series_readout_noise   =   readout_noise_vars.path_of_data_series
        num_of_repeats_readout_noise        =   readout_noise_vars.num_of_repeats
        num_of_temperatures_readout_noise   =   readout_noise_vars.num_of_data_points

        readout_noise_data, ron_dists_vs_temp = self.readout_noise_vs_temperature(  path_of_data_series  =   path_of_data_series_readout_noise,
                                                                                    num_of_repeats       =   num_of_repeats_readout_noise     ,
                                                                                    num_of_temperatures  =   num_of_temperatures_readout_noise  )
        return dark_current_data, readout_noise_data, ron_dists_vs_temp

    def master_bias_image(self, path_of_data_series: str):
        """
        Method that will set the class member np.ndarray master_bias

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master bias image
        """
        print(" Constructing the master bias correction image...")

        data_series         =   util.list_data(path_of_data_series)
        mean_image_return   =   util.mean_image(data_series, path_of_data_series)


        util.print_txt_file("master_bias" + self.datastorage_filename_append + ".txt", mean_image_return, which_directory=self.master_frames_storage_directory_path)
        self.master_bias = mean_image_return  # np.flip( , axis = 1)  # np.flip( , axis = 0)

    def bias_correction(self, image_to_be_corrected: np.ndarray):
        """
        Method that will apply the bias correction to an image
        by subtracting the master bias image

        :parameter np.ndarray image_to_be_corrected:
            - A numpy array which is the image data to be corrected
        """
        corrected_image = np.subtract(image_to_be_corrected, self.master_bias)  # np.flip(np.subtract(image_to_be_corrected, self.master_bias), axis=0)
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
        print(" Constructing the master dark current correction image...")

        data_series         =   util.list_data(path_of_data_series)
        dim_path            =   util.get_path(path_of_data_series + data_series[0])
        image_shape         =   util.get_dims(dim_path)     # Need the shape to prepare dummy array
        mean_image_array    =   np.zeros(image_shape)       # Prepare dummy array
        number_of_images    =   len(data_series)

        for imageid in data_series:
            filepath                    =   util.get_path(path_of_data_series + imageid)
            hdul, header, imagedata     =   util.fits_handler(filepath)
            imagedata                   =   self.bias_correction(imagedata)

            # Check for consistency
            #if header['EXPTIME'] == exposure_time:
            imagedata   =   np.divide(imagedata, exposure_time)
            #else:
             #   print("  master_dark_current_image(): Error, exposure time does not match up")
              #  print("  master_dark_current_image(): Exposure time was: ", header['EXPTIME'])

            mean_image_array += imagedata
            hdul.close()

        mean_image_array   /=   number_of_images

        util.print_txt_file("master_dark" + self.datastorage_filename_append + ".txt", mean_image_array, which_directory=self.master_frames_storage_directory_path)

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
        print("  Computing dark current as a function of temperature...")

        tmplist         =   []
        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=num_of_temperatures,
                                                                num_of_repeats_input=num_of_repeats,
                                                                where_is_repeat_num_in_string=[6, 9],
                                                                data_series_list=data_series)
        tempid = 0
        for repeat_sequence in reordered_data:
            repeat_sequence_meaned      =   self.bias_correction(util.mean_image(repeat_sequence, path_of_data_series))

            # Get temperatures from the filename
            filepath                    =   util.get_path(path_of_data_series + repeat_sequence[0])
            hdul, header, imagedata     =   util.fits_handler(filepath)
            temperature                 =   float(float(repeat_sequence[0][15:17] + '.' + repeat_sequence[0][18]))  # time in s
            if repeat_sequence[0][13] == "m":
                temperature *= -1

            # Error bars
            errorbar = util.compute_errorbar(repeat_sequence, path_of_data_series)

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                dark_per_time_per_pixel = np.mean(np.divide(np.multiply(repeat_sequence_meaned,  self.gain_vs_temp[tempid, 1]), exposure_time))
                tmplist.append([temperature, dark_per_time_per_pixel, errorbar])
            else:
                print("   dark_current_vs_temperature(): Error, exposure times do not match up")
                print("   dark_current_vs_temperature(): Exposure time was: ", header['EXPTIME'])

            hdul.close()
            tempid += 1

        #tmparray = np.sort(
        tmplist = np.asarray(tmplist)
        tmplist2D = tmplist.reshape(-1, tmplist.shape[-1])
        tmparray = (tmplist2D[np.lexsort(tmplist2D.T[::-1])]).reshape(tmplist.shape)
        #       , axis=0)

        self.dark_current_versus_temperature    = tmparray
        dark_current_versus_temperature_return  = tmparray

        util.print_txt_file("dark_current_versus_temperature" + self.datastorage_filename_append + ".txt", dark_current_versus_temperature_return, which_directory=self.analysis_data_storage_directory_path)

        return dark_current_versus_temperature_return

    def master_flat_field_image(self, path_of_data_series: str):
        """
        Method that will set the class member np.ndarray master_flat

        :parameter str path_of_data_series:
            - A string representing the path to the directory
              containing the data series used to construct the
              master flat field image
        """
        print(" Constructing the master flat field correction image...")

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
        meaned_flat /= np.mean(meaned_flat)  # Normalize

        util.print_txt_file("master_flat" + self.datastorage_filename_append + ".txt", meaned_flat, which_directory=self.master_frames_storage_directory_path)

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

    def hot_pixel_estimation(self, path_of_data_series: str, num_of_repeats: int, exposure_time: list, hot_pixel_cutoff: float):
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
        print(" Looking for hot pixels...")

        data_series                         =   util.list_data(path_of_data_series)
        reordered_data                      =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=2,
                                                                                    num_of_repeats_input=num_of_repeats,
                                                                                    where_is_repeat_num_in_string=[7, 8],
                                                                                    data_series_list=data_series          )
        # Construct the two images from the data files in the path
        short_exposure_image                =   util.mean_image(reordered_data[0].tolist(), path_of_data_series)
        long_exposure_image                 =   util.mean_image(reordered_data[1].tolist(), path_of_data_series)

        # Apply bias correction
        short_exposure_image                =   self.bias_correction(short_exposure_image)
        long_exposure_image                 =   self.bias_correction(long_exposure_image)

        # Convert ADU to dark current
        short_exposure_image                =   np.divide(np.multiply(short_exposure_image, self.gain_factor), exposure_time[0])
        long_exposure_image                 =   np.divide(np.multiply(long_exposure_image , self.gain_factor), exposure_time[1])

        # Find potential hot pixels
        smallest_measurable_dark_current    =   2 * ((self.gain_factor * self.readout_noise_level / math.sqrt(num_of_repeats)) / exposure_time[0])
        potential_hot_pixels                =   long_exposure_image > smallest_measurable_dark_current

        # Plot to qualitatively decide upon cutoff
        self.hot_pixel_data = [short_exposure_image[potential_hot_pixels].flatten(), long_exposure_image[potential_hot_pixels].flatten()]

        hot_pixels = (short_exposure_image > hot_pixel_cutoff)
        print("  No. of hot pixels:", hot_pixels.sum())

        self.hot_pixel_mask = hot_pixels

    def hot_pixel_correction(self, image_to_be_corrected: np.ndarray):
        """
        Method that will apply the hot pixel correction to an image.

        :parameter np.ndarray image_to_be_corrected:
            - A numpy array which is the image data to be corrected
        """
        image_to_be_corrected[np.where(self.hot_pixel_mask)] = np.mean(image_to_be_corrected[np.where(np.logical_not(self.hot_pixel_mask))])
        return image_to_be_corrected

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
        print(" Computing readout noise level...")

        data_series         =   util.list_data(path_of_data_series)
        number_of_images    =   len(data_series)
        tmp_std             =   np.zeros(number_of_images)
        tmp_mean            =   np.zeros(number_of_images)

        it = 0
        for imageid in data_series:
            filepath                    =   util.get_path(path_of_data_series + imageid)
            hdul, header, imagedata     =   util.fits_handler(filepath)

            # Check for consistency
            # check_temperature = (temperature*1.1 <= header['CCD-TEMP'] <= temperature*0.9) or (temperature*1.1 >= header['CCD-TEMP'] >= temperature*0.9)
            # if check_temperature:
            # noise_deviation     =   np.subtract(np.mean(imagedata), imagedata) * self.gain_factor # self.master_bias
            noise_deviation     =   np.subtract(self.master_bias, imagedata)  # * self.gain_factor
            tmp_std[it]         =   np.std(noise_deviation) # * self.gain_factor * np.sqrt(8 * np.log(2))
            tmp_mean[it]        =   np.mean(noise_deviation)
            #else:
            #    print(header['CCD-TEMP'])

            hdul.close()
            it += 1

        readout_noise                   =  np.sqrt(np.mean(np.square(tmp_std)))
        readout_noise_electrons         =  np.sqrt(np.mean(np.square(np.multiply(tmp_std, self.gain_factor))))
        num_of_rons                     =  len(tmp_std)
        readout_noise_error             =  np.std(tmp_std)
        readout_noise_electrons_error   =  readout_noise_error * self.gain_factor
        ron_mean                        =  np.mean(tmp_mean)



        plt.plot(np.linspace(0, num_of_rons, num_of_rons), tmp_std, 'k.', markersize=3, label="")
        plt.plot(np.linspace(0, num_of_rons, num_of_rons), np.ones(num_of_rons) * readout_noise, ls='-', c='k', lw=1, label="Mean value")
        plt.plot(np.linspace(0, num_of_rons, num_of_rons), np.ones(num_of_rons) * ((readout_noise_error / np.sqrt(len(data_series))) +  readout_noise ) , ls='--', c='k', lw=1, label="$\sigma / \sqrt{N}$")
        plt.plot(np.linspace(0, num_of_rons, num_of_rons), np.ones(num_of_rons) * (-(readout_noise_error / np.sqrt(len(data_series))) + readout_noise ) , ls='--', c='k', lw=1)
        pp.pubplot("$\mathbf{Readout\;Noise}$" + self.name, "Measurement no.", "Readout Noise [RMS $\mathbf{e}^-$/pixel]", self.figure_directory_path + "readout_noise_measurement" + self.datastorage_filename_append + ".png", legend=True, legendlocation="upper right")

        print(f"  The readout noise level is {readout_noise:.3f} ± ", readout_noise_error / np.sqrt(len(data_series)), " RMS ADU per pixel")
        print(f"  The readout noise level is {readout_noise_electrons:.3f} ± ", readout_noise_electrons_error / np.sqrt(len(data_series)), " RMS electrons per pixel")
        print(f"  The mean of the distribution is {ron_mean:.3f}, and should be equal to 0")

        self.readout_noise_level = readout_noise
        return np.mean(tmp_std)

    def readout_noise_vs_temperature(self, path_of_data_series, num_of_temperatures, num_of_repeats):
        """
                Method which will compute the readout noise levels as a function
                of temperature, which it returns as a list, and fills the member
                readout_noise_versus_temperature with as well. Implies a certain
                file naming convention of the type

                    noise_rrr_ron_s_tt_d.fit

                      - where nnn is an arbitrary descriptive string, s is
                        a character representing the sign of the temperature
                        either m or p, representing plus or minus (degrees),
                        tt is the temperature in degrees, while d is the decimal.
                        rrr represents the repeat number of the given data file.

                    An example of this:

                    noise_000_ron_m_04_9.fit

                :parameter str path_of_data_series:
                    - A string representing the path to the directory
                      containing the data series used to construct the data points
                :parameter int num_of_repeats:
                    - Integer representing the total number of repeats of the data sequence
                :parameter int num_of_temperatures:
                    - Integer representing the number of different temperatures
                      in the data sequence
                :returns np.ndarray dark_current_versus_temperature_return:
                    - A numpy array of data points of the form (temperature, value)
        """

        print("  Computing readout noise as a function of temperature...")
        data_series         =   util.list_data(path_of_data_series)
        tmp_std             =   np.zeros(num_of_repeats)
        tmp_mean            =   np.zeros(num_of_repeats)
        reordered_data      =   util.repeat_sequence_ordered_data(  num_of_temperatures, num_of_repeats,
                                                                    where_is_repeat_num_in_string=[6, 9],
                                                                    data_series_list=data_series            )

        readout_noise = []
        ron_dists_vs_temp = []
        tempid = 0
        for repeat_sequence in reordered_data:
            # Get temperatures from the filename
            temperature = float(repeat_sequence[0][16:18] + '.' + repeat_sequence[0][19])  # time in s
            if repeat_sequence[0][14] == "m":
                temperature *= -1

            dist_at_this_temperature = util.mean_image(repeat_sequence, path_of_data_series)
            ron_dists_vs_temp.append(dist_at_this_temperature.flatten())

            it = 0
            for imageid in repeat_sequence:
                """
                    For each image in each repeat sequence, find noise in electrons 
                    by subtracting the mean of the image from the image itself, and 
                    multiplying with the gain. This yields a noise image, that is a 
                    gaussian distribution. The noise is found as the  width of said 
                    distribution. In addition we check that the mean is equal to 0.
                """
                filepath = util.get_path(path_of_data_series + imageid)
                hdul, header, imagedata = util.fits_handler(filepath)

                noise_deviation     =   np.subtract(np.mean(imagedata), imagedata) * self.gain_vs_temp[tempid, 1]  # self.master_bias
                tmp_std[it]         =   np.std(noise_deviation)  # / np.sqrt(2)
                tmp_mean[it]        =   np.mean(noise_deviation)

                hdul.close()
                it += 1

                if it == num_of_repeats:
                    it = 0

            # Errorbars
            errorbar = util.compute_errorbar(repeat_sequence, path_of_data_series)

            readout_noise.append([temperature, np.sqrt(np.mean(np.square(tmp_std))), errorbar])
            tempid += 1

        #tmparray = np.sort(

        tmplist = np.asarray(readout_noise)
        tmplist2D = tmplist.reshape(-1, tmplist.shape[-1])
        tmparray = (tmplist2D[np.lexsort(tmplist2D.T[::-1])]).reshape(tmplist.shape)
            # tmparray=np.asarray(readout_noise)
            #, axis=0)

        self.readout_noise_versus_temperature = tmparray
        readout_noise_versus_temperature_return = tmparray

        util.print_txt_file("readout_noise_versus_temperature" + self.datastorage_filename_append + ".txt", readout_noise_versus_temperature_return,
                            which_directory=self.analysis_data_storage_directory_path)
        util.print_txt_file("readout_noise_distribution_versus_temperature" + self.datastorage_filename_append + ".txt", ron_dists_vs_temp,
                            which_directory=self.analysis_data_storage_directory_path)

        return readout_noise_versus_temperature_return, ron_dists_vs_temp

    def gain_factor_estimation(self, path_of_data_series: str):
        data_series         =   util.list_data(path_of_data_series)

        gain_factors = []
        for id in range(0, len(data_series), 2):
            filepath_first = util.get_path(path_of_data_series + data_series[id])
            filepath_second = util.get_path(path_of_data_series + data_series[id + 1])
            hdul, header, imagedata_first = util.fits_handler(filepath_first)
            hdul, header, imagedata_second = util.fits_handler(filepath_second)
            imagedata_first = self.bias_correction(imagedata_first)
            imagedata_second = self.bias_correction(imagedata_second)

            image_flux_ratios = np.divide(np.mean(imagedata_first), np.mean(imagedata_second))
            numerator = np.mean(np.add(imagedata_first, imagedata_second))
            denominator = np.std(np.subtract(imagedata_first, np.multiply(imagedata_second, image_flux_ratios)))**2 - 2 * (self.readout_noise_level**2)

            gain_factors.append(np.divide(numerator, denominator))

        gain_factor         = np.mean(np.asarray(gain_factors))
        num_of_gain_factors = len(gain_factors)
        gain_factor_error   = np.std(np.asarray(gain_factors))
        gain_factor_relative_error = gain_factor_error / np.sqrt(1/2 * len(data_series))

        plt.plot(np.linspace(0, num_of_gain_factors, num_of_gain_factors), gain_factors, 'k.', markersize=3, label="Measured value")
        plt.plot(np.linspace(0, num_of_gain_factors, num_of_gain_factors), np.ones(num_of_gain_factors) * gain_factor, ls='-', c='k', lw=1, label="Mean value")
        plt.plot(np.linspace(0, num_of_gain_factors, num_of_gain_factors), np.ones(num_of_gain_factors) * (  gain_factor_relative_error +  gain_factor ) , ls='--', c='k', lw=1, label="$\sigma / \sqrt{N}$")
        plt.plot(np.linspace(0, num_of_gain_factors, num_of_gain_factors), np.ones(num_of_gain_factors) * (- gain_factor_relative_error + gain_factor ) , ls='--', c='k', lw=1)
        pp.pubplot("$\mathbf{Gain\;factor}-10.0^\circ$C " + self.name, "Measurement no.", "Gain factor, $g$", self.figure_directory_path + "gain_factor_measurement" + self.datastorage_filename_append + ".png", legend=True, legendlocation="upper right")

        print("  The estimated gain factor is ", gain_factor, " ± ", gain_factor_relative_error, " electrons/ADU, while the tabulated (input) value was ", self.gain_factor, " electrons/ADU")
        self.gain_factor = np.float(gain_factor)

    def gain_vs_temperature(self, path_of_data_series: str, num_of_data_points: int, num_of_repeats: int):
        print("  Computing gain factor as a function of temperature...")
        data_series     =   util.list_data(path_of_data_series)
        reordered_data  =   util.repeat_sequence_ordered_data(  num_of_data_points, num_of_repeats,
                                                                where_is_repeat_num_in_string=[5, 8],
                                                                data_series_list=data_series            )

        gain_vs_temp = []
        for repeat_sequence in reordered_data:
            # Get temperatures from the filename
            temperature = float(repeat_sequence[0][11:13] + '.' + repeat_sequence[0][14])  # time in s
            if repeat_sequence[0][9] == "m":
                temperature *= -1

            gain_factors = []
            for id in range(0, len(repeat_sequence), 2):
                filepath_first = util.get_path(path_of_data_series + repeat_sequence[id])
                filepath_second = util.get_path(path_of_data_series + repeat_sequence[id + 1])
                hdul, header, imagedata_first = util.fits_handler(filepath_first)
                hdul, header, imagedata_second = util.fits_handler(filepath_second)
                imagedata_first = self.bias_correction(imagedata_first)
                imagedata_second = self.bias_correction(imagedata_second)

                image_flux_ratios = np.divide(np.mean(imagedata_first), np.mean(imagedata_second))
                numerator = np.mean(np.add(imagedata_first, imagedata_second))
                denominator = np.std(
                    np.subtract(imagedata_first, np.multiply(imagedata_second, image_flux_ratios))) ** 2 - 2 * (
                                      self.readout_noise_level ** 2)

                gain_factors.append(np.divide(numerator, denominator))

            gain_factor = np.mean(np.asarray(gain_factors))
            gain_factor_error = np.std(np.asarray(gain_factors))

            gain_vs_temp.append([temperature, gain_factor, gain_factor_error])

        #gain_vs_temp = np.sort(gain_vs_temp, axis=0)

        tmplist = np.asarray(gain_vs_temp)
        tmplist2D = tmplist.reshape(-1, tmplist.shape[-1])
        gain_vs_temp = (tmplist2D[np.lexsort(tmplist2D.T[::-1])]).reshape(tmplist.shape)
        # gain_vs_temp = np.asarray(gain_vs_temp)
        self.gain_vs_temp = gain_vs_temp

        util.print_txt_file("gain_factor_versus_temperature" + self.datastorage_filename_append + ".txt",
                            gain_vs_temp,
                            which_directory=self.analysis_data_storage_directory_path)

        return gain_vs_temp

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
        print(" Testing linearity...")

        tmplist = []

        data_series         =   util.list_data(path_of_data_series)
        reordered_data      =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=num_of_exposures,
                                                                    num_of_repeats_input=num_of_repeats,
                                                                    where_is_repeat_num_in_string=[10, 13],
                                                                    data_series_list=data_series)

        # stability_data     =   self.test_lightsource_stability(path_of_data_series, num_of_exposures, num_of_repeats)

        first_it = 0
        second_it = 0
        for repeat_sequence in reordered_data:
            # repeat_sequence_meaned      =   util.mean_image(repeat_sequence, path_of_data_series)
            dim_path            =   util.get_path(path_of_data_series + repeat_sequence[0])
            image_shape         =   util.get_dims(dim_path)
            number_of_images    =   len(repeat_sequence)
            mean_image_array    =   np.zeros(image_shape)

            for imageid in repeat_sequence:
                filepath                        =   util.get_path(path_of_data_series + imageid)
                hdul, header, imagedata         =   util.fits_handler(filepath)
                imagedata_meaned                =   self.bias_correction(imagedata)
                imagedata_meaned_and_corrected  =   imagedata_meaned  # np.divide(imagedata_meaned, (stability_data[first_it, second_it] / 100 + 1))

                mean_image_array += imagedata_meaned_and_corrected

                hdul.close()
                second_it += 1
                if second_it == num_of_repeats:
                    second_it = 0

            first_it += 1

            mean_image_array /= number_of_images

            # repeat_sequence_meaned      =   self.bias_correction(repeat_sequence_meaned)  # self.flat_field_correction(  ,   util.mean_image(repeat_sequence, path_of_data_series)

            # Get exposure time from filename within a given repeat sequence
            repeat_sequence_meaned      =   mean_image_array
            filepath                    =   util.get_path(path_of_data_series + repeat_sequence[0])
            hdul, header, imagedata     =   util.fits_handler(filepath)
            exposure_time               =   float(repeat_sequence[0][-7:-4])  # time in s

            # Treat hot pixels
            repeat_sequence_meaned[np.where(self.hot_pixel_mask)] = np.mean(repeat_sequence_meaned[np.where(np.logical_not(self.hot_pixel_mask))])

            # Compute errorbars
            errorbar = util.compute_errorbar(repeat_sequence, path_of_data_series)

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned), errorbar])
            else:
                print("  linearity_estimation(): Error, exposure times do not match up")
                print("  linearity_estimation(): Exposure time was: ", header['EXPTIME'], "should have been: ", exposure_time)

                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned), errorbar])

        linearity_array     =   np.asarray(tmplist)

        # print("  Done! Data constructed from linearity measurements:")
        # print(linearity_array)

        self.linearity      =   linearity_array

        util.print_txt_file("linearity" + self.datastorage_filename_append + ".txt", linearity_array,
                            which_directory=self.analysis_data_storage_directory_path)

        return linearity_array

    def linearity_estimation_with_reference(self, path_of_data_series: str, num_of_exposures: int, num_of_repeats: int, reference_exposure: float, exposures: np.ndarray, milliseconds: bool = False):
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
        :parameter float reference_exposure:
            - The reference measurement exposure time
        :returns np.ndarray linearity_array:
            - A numpy array of data points of the form (exposure time, mean ADU)
        """
        print(" Testing linearity...")

        tmplist = []

        data_series         =   util.list_data(path_of_data_series)
        where_is_repeat_num_in_string = [10, 13]

        reordered_data      =   np.empty((num_of_exposures, num_of_repeats, 3), dtype=object)
        from_id_in_str      =   where_is_repeat_num_in_string[0]
        to_id_in_str        =   where_is_repeat_num_in_string[1]

        index = 0
        for imageid in data_series:
            if imageid[-7:-4] == "(2)":
                this_actual_exposure_time = 10.0
            else:
                # this_actual_exposure_time = float(imageid[-9:-7] + "." + imageid[-7:-6])
                # print(this_actual_exposure_time)
                this_actual_exposure_time = float(imageid[-9:-6])  # -7:-4
                if milliseconds:
                    this_actual_exposure_time /= 10
            exposure_index      =   (np.where(exposures == this_actual_exposure_time))[0][0]
            repeat_num          =   int(imageid[from_id_in_str:to_id_in_str])
            reference_index     =   int(imageid[-5])

            reordered_data[exposure_index][repeat_num][reference_index] = str(imageid)

            index += 1
            if index == num_of_exposures:
                index = 0

        for repeat_sequence_id in range(0, num_of_exposures):
            this_actual_exposure_time   =   exposures[repeat_sequence_id]
            tmp_mean    =  0
            mean_ADU    =  0
            distribution_of_image_means = []
            for repeat_id in range(0, num_of_repeats):
                filepath_first_ref                           =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][0])
                filepath_actual_image                        =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][1])
                filepath_next_ref                            =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][2])

                hdul, header_actual, imagedata_actual_image  =   util.fits_handler(filepath_actual_image)
                hdul, header_first, imagedata_first_ref      =   util.fits_handler(filepath_first_ref)
                hdul, header_next, imagedata_next_ref        =   util.fits_handler(filepath_next_ref)

                imagedata_actual_bias_corrected_and_meaned   =   np.mean((self.bias_correction(imagedata_actual_image)))  # [350:400, 350:400])
                imagedata_first_bias_corrected_and_meaned    =   np.mean((self.bias_correction(imagedata_first_ref)))  # [350:400, 350:400])
                imagedata_next_bias_corrected_and_meaned     =   np.mean((self.bias_correction(imagedata_next_ref)))  # [350:400, 350:400])

                mean_lightsource_change                      =   (1/2) * (imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned)
                time_calibration                             =   (reference_exposure + self.time_calibration_factor) / (this_actual_exposure_time + self.time_calibration_factor)

                imagedata_meaned_normed_and_corrected        =   np.divide(np.multiply(imagedata_actual_bias_corrected_and_meaned, time_calibration), mean_lightsource_change)
                imagedata_converted_to_deviation             =   np.subtract(imagedata_meaned_normed_and_corrected, 1) * 100

                tmp_mean += imagedata_converted_to_deviation
                mean_ADU += imagedata_actual_bias_corrected_and_meaned

                distribution_of_image_means.append((np.mean(imagedata_meaned_normed_and_corrected) * 100) / float(np.sqrt(num_of_repeats)))

            tmp_mean /= num_of_repeats
            mean_ADU /= num_of_repeats

            # Compute errorbars
            errorbar = np.std(np.asarray(distribution_of_image_means))

            # Check for consistency
            #if header_actual['EXPTIME'] == this_actual_exposure_time:
            tmplist.append([this_actual_exposure_time, tmp_mean, errorbar, mean_ADU])
            #else:
            #    print("  linearity_estimation(): Error, exposure times do not match up")
            #    print("  linearity_estimation(): Exposure time was: ", header_actual['EXPTIME'], "should have been: ", this_actual_exposure_time)

            #    tmplist.append([this_actual_exposure_time, tmp_mean, errorbar, mean_ADU])

        linearity_array     =   np.asarray(tmplist)
        self.linearity      =   linearity_array

        query_points = self.linearity[:, 3]
        linearity_data = self.linearity[:, 1]
        fitted_linear_model = np.polyfit(query_points[6:19], linearity_data[6:19], deg=1)
        fitted_slope = fitted_linear_model[0]
        fitted_offset = fitted_linear_model[1]
        fitted_linearity = np.add(np.multiply(query_points, fitted_slope), fitted_offset)
        util.print_txt_file("fitted_linearity" + self.datastorage_filename_append + ".txt", fitted_linearity,
                            which_directory=self.analysis_data_storage_directory_path)

        util.print_txt_file("linearity" + self.datastorage_filename_append + ".txt", linearity_array,
                            which_directory=self.analysis_data_storage_directory_path)

        return linearity_array

    def time_calibration(self, path_of_data_series: str, num_of_exposures: int, num_of_repeats: int):
        """

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
        print(" Performing time calibration...")

        tmplist = []

        data_series         =   util.list_data(path_of_data_series)


        num_of_datapoints_input = num_of_exposures
        num_of_repeats_input = num_of_repeats
        data_series_list = data_series
        where_is_repeat_num_in_string = [8, 11]

        reordered_data      =   np.empty((num_of_exposures, num_of_repeats, 2), dtype=object)
        from_id_in_str      =   where_is_repeat_num_in_string[0]
        to_id_in_str        =   where_is_repeat_num_in_string[1]

        exposures = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])
        index = 0
        for imageid in data_series_list:
            if imageid[-7:-4] == "(2)":
                exposure_time = 10.0
            else:
                exposure_time = float(imageid[-8:-5] + "." + imageid[-5])

            exposure_index  =   (np.where(exposures == exposure_time))[0][0]
            repeat_num      =   int(imageid[from_id_in_str:to_id_in_str])

            reference_string = imageid[12:16]
            is_reference = (reference_string == "0100")
            if is_reference:
                if imageid[-7:-4] == "(2)":
                    reordered_data[exposure_index][repeat_num][0] = str(imageid)
                reordered_data[exposure_index][repeat_num][1] = str(imageid)
            else:
                reordered_data[exposure_index][repeat_num][0] = str(imageid)

            index += 1
            if index == num_of_datapoints_input:
                index = 0

        dim_path            =   util.get_path(path_of_data_series + reordered_data[0][0][0])
        image_shape         =   util.get_dims(dim_path)

        scaling = 1
        for repeat_sequence_id in range(0, num_of_exposures):
            mean_image_array    =   np.zeros(image_shape)

            distribution_of_image_means = []
            for repeat_id in range(0, num_of_repeats):
                if repeat_sequence_id == num_of_exposures - 1:
                    filepath_actual_image   =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][0])
                    filepath_first_ref      =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][1])
                    filepath_next_ref       =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][1])
                else:
                    filepath_actual_image   =   util.get_path( path_of_data_series + reordered_data[repeat_sequence_id    ][repeat_id][0])
                    filepath_first_ref      =   util.get_path( path_of_data_series + reordered_data[repeat_sequence_id    ][repeat_id][1])
                    filepath_next_ref       =   util.get_path( path_of_data_series + reordered_data[repeat_sequence_id + 1][repeat_id][1])

                hdul, header, imagedata_actual_image    =   util.fits_handler(filepath_actual_image)
                hdul, header, imagedata_first_ref       =   util.fits_handler(filepath_first_ref)
                hdul, header, imagedata_next_ref        =   util.fits_handler(filepath_next_ref)

                imagedata_actual                 =   self.bias_correction(imagedata_actual_image)
                imagedata_first                  =   self.bias_correction(imagedata_first_ref)
                imagedata_next                   =   self.bias_correction(imagedata_next_ref)

                scaling                          =   np.mean(imagedata_next) / np.mean(imagedata_first)

                imagedata_meaned_and_corrected   =   np.multiply(imagedata_actual, scaling)

                mean_image_array += imagedata_meaned_and_corrected

                distribution_of_image_means.append(np.mean(imagedata_meaned_and_corrected))

            mean_image_array /= num_of_repeats

            # Get exposure time from filename within a given repeat sequence
            repeat_sequence_meaned      =   mean_image_array
            filepath                    =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][0][0])
            hdul, header, imagedata     =   util.fits_handler(filepath)
            if reordered_data[repeat_sequence_id][0][0][-7:-4] == "(2)":
                exposure_time   =     10.0
            else:
                exposure_time   =     float(reordered_data[repeat_sequence_id][0][0][-8:-5] + "." + reordered_data[repeat_sequence_id][0][0][-5]) # time in s

            # Treat hot pixels
            repeat_sequence_meaned[np.where(self.hot_pixel_mask)] = np.mean(repeat_sequence_meaned[np.where(np.logical_not(self.hot_pixel_mask))])

            # Compute errorbars
            errorbar = np.std(np.asarray(distribution_of_image_means))

            # Check for consistency
            if header['EXPTIME'] == exposure_time:
                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned), errorbar])
            else:
                print("  linearity_estimation(): Error, exposure times do not match up")
                print("  linearity_estimation(): Exposure time was: ", header['EXPTIME'], "should have been: ", exposure_time)

                tmplist.append([exposure_time, np.mean(repeat_sequence_meaned), errorbar])

        linearity_array     =   np.asarray(tmplist)

        query_points        =   linearity_array[:, 0]
        linearity_data      =   linearity_array[:, 1]
        error_data          =   linearity_array[:, 2]

        linear_model        =   np.polyfit(query_points, linearity_data, deg=1)
        linear_model_func   =   np.poly1d(linear_model)
        time_offset         =   np.roots(linear_model)
        linear_model_data   =   linear_model_func(query_points)

        # Apply time calibration to the exposure times
        self.time_calibration_factor = float(time_offset)
        print(time_offset, self.time_calibration_factor)
        print("  The time offset (time calibration factor) has been estimated to ", self.time_calibration_factor, " s ...")

        corrected_data          =   np.add(query_points, self.time_calibration_factor)
        new_linear_model        =   np.polyfit(corrected_data, linearity_data, deg=1)
        new_linear_model_func   =   np.poly1d(new_linear_model)
        new_linear_data         =   new_linear_model_func(corrected_data)

        deviations              =   np.multiply(np.divide(np.subtract(linearity_data, linear_model_data), linear_model_data), 100)
        errors                  =   np.multiply(np.divide(error_data, linearity_data), 100)
        new_deviations          =   np.multiply(np.divide(np.subtract(linearity_data, new_linear_data), new_linear_data), 100)

        return [linearity_array, linear_model_data, corrected_data, new_linear_data, deviations, errors, new_deviations]

    def new_time_calibration(self, path_of_data_series: str, num_of_repeats: int, exposures: np.ndarray):
        print("Computing time calibration factor...")
        data_series = util.list_data(path_of_data_series)
        where_is_repeat_num_in_string = [8, 11]

        reordered_data = np.empty((3, num_of_repeats, 3), dtype=object)
        from_id_in_str = where_is_repeat_num_in_string[0]
        to_id_in_str = where_is_repeat_num_in_string[1]

        for imageid in data_series:
            repeat_num = int(imageid[from_id_in_str:to_id_in_str])
            reference_index = int(imageid[-7])
            intensity_index = int(imageid[-5])

            reordered_data[intensity_index][repeat_num][reference_index] = str(imageid)


        offsets = []
        one_second_correction = 1  #  (1 - 2.826937348128014116e-01)  # 3.269314122356812846e-01)  # 2.335026485015368747e-01)  # 4.061413920020190416e-01)
        two_second_correction = 1  #  (1 - 2.470296021393934560e-01)  # 2.862068501265830345e-01)  # 2.034781685231622506e-01)  # 3.563827129498361446e-01)
        for repeat_sequence_id in range(0, 3):
            time_offset = 0
            for repeat_id in range(0, num_of_repeats):
                filepath_first_ref                           =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][0])
                filepath_actual_image                        =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][1])
                filepath_next_ref                            =   util.get_path(path_of_data_series + reordered_data[repeat_sequence_id][repeat_id][2])

                hdul, header_actual, imagedata_actual_image  =   util.fits_handler(filepath_actual_image)
                hdul, header_first, imagedata_first_ref      =   util.fits_handler(filepath_first_ref)
                hdul, header_next, imagedata_next_ref        =   util.fits_handler(filepath_next_ref)

                imagedata_actual_bias_corrected_and_meaned   =   one_second_correction * np.mean((self.bias_correction(imagedata_actual_image)))
                imagedata_first_bias_corrected_and_meaned    =   two_second_correction * np.mean((self.bias_correction(imagedata_first_ref)))
                imagedata_next_bias_corrected_and_meaned     =   one_second_correction * np.mean((self.bias_correction(imagedata_next_ref)))

                numerator = 1 - ((imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned) / imagedata_actual_bias_corrected_and_meaned)
                denominator = ((imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned) / (2 * imagedata_actual_bias_corrected_and_meaned)) - 1

                # numerator   = ((imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned) / imagedata_actual_bias_corrected_and_meaned) - 1
                # denominator = 1 - ((imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned) / (2 * imagedata_actual_bias_corrected_and_meaned))

                #time_offset += (numerator)    / denominator

                numerator    = (((imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned) / imagedata_actual_bias_corrected_and_meaned) * (exposures[1] / 2)) - exposures[0]
                denominator  = 1 - ((imagedata_first_bias_corrected_and_meaned + imagedata_next_bias_corrected_and_meaned) / (2 * imagedata_actual_bias_corrected_and_meaned))

                time_offset += numerator / denominator

            time_offset /= num_of_repeats
            offsets.append(time_offset)


        final_offset = np.float(np.mean(np.asarray(offsets)))
        final_offset_error = np.float(np.std(np.asarray(offsets)))
        self.time_calibration_factor = final_offset
        print("  Result: " + str(self.time_calibration_factor), "±", final_offset_error / np.sqrt(len(offsets)), " s")


    def test_lightsource_stability(self, path_of_data_series: str, num_of_data_points: int, num_of_repeats: int):
        """
            Method to analyze the stabillity of the lightsource used to acquire the
            linearity data sequences. All of the images are analyzed. For a given
            exposure sequence, a mean ADU value is computed from the entire sequence
            and this value is the same as what is ultimately used for the linearity
            analysis above. For each image in a sequence the mean ADU value is computed
            and it's deviation from the sequence mean is computed as a percentage.
            This is hence interpreted as the temporal lightsource (in)stabillity.

            Returns an array of datapoints. A temporal series for each exposure sequence
            of size (num_of_data_points, num_of_repeats)

            :parameter path_of_data_series:
                - A string representing the path to the directory
                  containing the data series used to construct the data points
            :parameter int num_of_data_points:
                - Integer representing the number of different exposure times
                  in the data sequence
            :parameter int num_of_repeats:
                - Integer representing the total number of repeats of the data sequence
            :returns np.ndarray stability_array:
                - A numpy array of data points of size (num_of_data_points, num_of_repeats)
        """
        print(" Testing light source stabillity...")
        data_series         =   util.list_data(path_of_data_series)
        reordered_data      =   util.repeat_sequence_ordered_data(  num_of_datapoints_input=num_of_data_points,
                                                                    num_of_repeats_input=num_of_repeats,
                                                                    where_is_repeat_num_in_string=[10, 13],
                                                                    data_series_list=data_series                )

        stability_array     =   np.empty([num_of_data_points, num_of_repeats])

        data_point          =   0
        repeat              =   0

        for sequence in reordered_data:
            for image in sequence:
                filepath = util.get_path(path_of_data_series + image)
                hdul, header, imagedata = util.fits_handler(filepath)

                if data_point > 0:
                    self.bias_correction(imagedata)

                mean_adu = np.mean(imagedata)

                stability_array[data_point, repeat] = mean_adu
                repeat += 1
                if repeat == num_of_repeats:
                    repeat = 0

            data_point += 1

        sequence_mean       =   np.mean(stability_array, axis=1, keepdims=True)
        stability_array     =   np.multiply(np.divide(np.subtract(stability_array, sequence_mean), sequence_mean), 100)

        util.print_txt_file("lightsource_stability" + self.datastorage_filename_append + ".txt", stability_array,
                            which_directory=self.analysis_data_storage_directory_path)

        return stability_array

    def linearity_precision(self):
        """
            A method to complete the linearity analysis by computing the deviations
            of the measured mean ADU (linearity datapoints acquired from the method
            linearity_estimation()), from an ideal linear relation found from linear
            regression.

            Does not take any parameters, but uses filled members from other methods
            in the class definition. Assumes that the method linearity_estimation()
            has been completed successfully.
        """

        print(" Testing the precision of the linearity measurement...")
        query_points            =   self.linearity[:, 0]
        linearity_data          =   self.linearity[:, 1]
        error_data              =   self.linearity[:, 2]

        # Apply time calibration to the exposure times
        query_points            =   np.add(query_points, self.time_calibration_factor)
        self.linearity[:, 0]    =   query_points

        # Preliminary estimation of ideal linearity curve
        ideal_slope = self.linearity[9, 1] / query_points[9] # - self.linearity[6, 1]) / (query_points[7] - query_points[6])
        ideal_offset = 0

        # ideal_linear_model            =   np.polyfit(query_points[:14], linearity_data[:14], deg=1)

        ideal_linear_model        =   np.polyfit(np.concatenate([query_points[0:14]]), np.concatenate([linearity_data[0:14]]), deg=1)
        # linear_model_func       =   np.poly1d(ideal_linear_model)
        # time_offset             =   np.roots(ideal_linear_model)
        # query_points            =   np.subtract(query_points, time_offset)
        # self.linearity[:, 0]    =   query_points
        ideal_slope = ideal_linear_model[0]
        ideal_offset = ideal_linear_model[1]
        # ideal_offset = 0
        print(ideal_slope, ideal_offset)

        # Construct the ideal linear relation and compute deviations from that
        ideal_linearity         =   np.add(np.multiply(query_points, ideal_slope), ideal_offset)  # linear_model_func(query_points)  #
        deviations              =   np.multiply(np.divide(np.subtract(linearity_data, ideal_linearity), ideal_linearity), 100)
        errors                  =   np.multiply(np.divide(error_data, ideal_linearity), 100)

        util.print_txt_file("ideal_linearity" + self.datastorage_filename_append + ".txt", ideal_linearity,
                            which_directory=self.analysis_data_storage_directory_path)
        util.print_txt_file("linearity_deviations" + self.datastorage_filename_append + ".txt", deviations,
                            which_directory=self.analysis_data_storage_directory_path)
        util.print_txt_file("linearity_deviation_errors" + self.datastorage_filename_append + ".txt", errors,
                            which_directory=self.analysis_data_storage_directory_path)

        return ideal_linearity, deviations, errors

    @staticmethod
    def test_zero_point(path_of_data_series: str, num_of_data_points: int, num_of_repeats: int):
        """
            Method that will test the ground assumption that the zero exposure time, is actually
            an exposure time of zero seconds. This will be used in the time calibration. The working
            idea is to test whether the following is true:

                [ IM(11s) - IM(1s) ]         [ IM(10s) - Bias ]
                --------------------    =    ------------------
                [ IM(21s) - IM(1s) ]         [ Im(20s) - Bias ]

            Which geometrically means that the slope is such that the ideal linear relation of ADU
            from exposures passes through zero. The method assumes that data acquired, is then
            reordered/restructured in the following order (basically sorting of the data seq.):

                Bias, Bias, 1s, 1s, 10s, 11s, 20s, 21s

            :parameter path_of_data_series:
                - A string representing the path to the directory
                  containing the data series used to construct the data points
            :parameter int num_of_data_points:
                - Integer representing the number of different exposure times
                  in the data sequence
            :parameter int num_of_repeats:
                - Integer representing the total number of repeats of the data sequence

            Returns nothing, but prints the result of the test below.
        """
        print(" Testing zero point assumption...")
        data_series = util.list_data(path_of_data_series)
        reordered_data = util.repeat_sequence_ordered_data(num_of_datapoints_input=num_of_data_points,
                                                           num_of_repeats_input=num_of_repeats,
                                                           where_is_repeat_num_in_string=[10, 13],
                                                           data_series_list=data_series)

        repeat_sequence_meaned  =   []
        for repeat_sequence in reordered_data:
            repeat_sequence_meaned.append(util.mean_image(repeat_sequence, path_of_data_series))

        # The following assumes the reordered data on the form as specified in the docstring above
        # Construct numerators and denominators on both sides
        lhs_numerator       =   np.mean(np.subtract(repeat_sequence_meaned[5], repeat_sequence_meaned[2]))
        lhs_denominator     =   np.mean(np.subtract(repeat_sequence_meaned[7], repeat_sequence_meaned[3]))
        rhs_numerator       =   np.mean(np.subtract(repeat_sequence_meaned[4], repeat_sequence_meaned[0]))
        rhs_denominator     =   np.mean(np.subtract(repeat_sequence_meaned[6], repeat_sequence_meaned[1]))

        # Construct fractions on both sides
        lhs_final           =   np.divide(lhs_numerator, lhs_denominator)
        rhs_final           =   np.divide(rhs_numerator, rhs_denominator)

        # Subtract both sides from each other, to check if result is zero
        result              =   np.subtract(lhs_final, rhs_final)

        print("  The result of the test is that the zeropoint assumption is valid to a precision of", result)
