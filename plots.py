import ccd
import matplotlib.pyplot as plt
import pubplot as pp
import utilities as util
import numpy as np


def gauss_dist_plot(detector: ccd, figure_directory: str, bias_sequence: str):
    """
        A method for plotting the distribution in bias frames
        to show that there is a 1/sqrt(N) reduction in the noise
        for the master bias frame, in comparison to the individual
        arbitrarily chosen bias frame
    """

    # Start by plotting the distribution of an individual bias frame
    data_series                 =   util.list_data(bias_sequence)
    filepath                    =   util.get_path(bias_sequence + data_series[0])
    hdul, header, imagedata     =   util.fits_handler(filepath)

    bias_dist           =   imagedata.flatten()
    n, bins, patches    =   plt.hist(bias_dist, bins=1000, color='dodgerblue', width=0.8, label="Arbitrary individual bias frame")
    gaussdata           =   np.linspace(240, 375, 1000)
    gaussheight         =   np.amax(n)
    gaussmean           =   float(np.mean(imagedata))
    gausswidth          =   float(np.std(bias_dist))

    plt.plot(gaussdata, util.gaussian(gaussdata, gaussheight, gaussmean, gausswidth), c='navy', label="Gauss. dist.")

    individual_gauss_width = gausswidth
    print("\nThe found width of the ADU distribution is ", individual_gauss_width, " ADUs")
    print("This corresponds to a readout noise of ", individual_gauss_width * 0.28 * np.sqrt(8 * np.log(2)))
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Then plot the distribution of the master bias frame with reduced noise
    bias_dist           =   detector.master_bias.flatten()
    n, bins, patches    =   plt.hist(bias_dist, bins=350, color='steelblue', width=0.4, label="Master bias frame")
    gaussheight         =   np.amax(n)
    gaussmean           =   float(np.mean(bias_dist))
    gausswidth          =   float(np.std(bias_dist))

    plt.plot(gaussdata, util.gaussian(gaussdata, gaussheight, gaussmean, gausswidth), c='k', ls="--", label="Reduced wdith gauss. dist.")

    print("The found, reduced width of the ADU distribution is ", gausswidth, " ADUs")
    print("It should be equal to ",  individual_gauss_width/np.sqrt(300))
    print("This corresponds to a readout noise of ", gausswidth * 0.28 * np.sqrt(8 * np.log(2)))
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    pp.pubplot("$\mathbf{Bias\;\;distribution:}$ " + detector.name, "Bias value", "Arb. units.", figure_directory + "gauss_bias.png", legend=False, xlim=[260, 350])


def master_frame_plot(image: np.ndarray, figurename: str, title: str, label: str, filename: str, raisetitle: bool = False, scaling = None):
    """
        A method that will produce a plot of a given master frame (bias, dark or flat).

        :param np.ndarray image:
            - np.ndarray representing the image
        :param str figurename:
            - str representing the figurename
        :param str title:
            - str representing the title of the figure for printing
        :parameter str label:
            - The detector name used as a label in the plot
        :param str filename:
            - str representing the name of the file to be printed to
        :param bool raisetitle:
            - bool that toggles raising of the title in the figure
    """

    pp.plot_image(image, title, "x", "y", "detector: " + label, filename, figurename, raisetitle=raisetitle, scale=scaling)


def noise_plot(detector: ccd, figure_directory: str, dark_current_data = None, readout_noise_data = None):
    """
        A method that will produce a plot of the stability data
        from the ccd.dark_current_versus_temperature() and ccd.readout_noise_versus_temperature() methods
    """

    plt.plot(dark_current_data[:, 0], dark_current_data[:, 1], ls='--', c='k', lw=1, marker='o', markersize=3, label="Dark current")
    plt.ylim(-0.1, 3)
    pp.pubplot("$\mathbf{Dark\;current\;}$ " + detector.name, "Temperature [$^\circ$C]", "$\mathbf{e}^-$/s/pixel", figure_directory + "darkcurrent_versus_temperature" + detector.datastorage_filename_append + ".png", legend=True, legendlocation="upper left")

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    plt.plot(readout_noise_data[:, 0], readout_noise_data[:, 1], ls='--', c='k', lw=1, marker='o', markersize=3, label="Readout noise")
    plt.ylim(3, 14)
    pp.pubplot("$\mathbf{Readout\;noise\;}$" + detector.name, "Temperature [$^\circ$C]", "Readout Noise [RMS $\mathbf{e}^-$/pixel]", figure_directory + "readoutnoise_versus_temperature" + detector.datastorage_filename_append + ".png", legend=True, legendlocation="upper left")


def gain_plot(detector: ccd, figure_directory: str, gain_data = None):
    plt.errorbar(gain_data[:, 0], gain_data[:, 1], yerr=gain_data[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label="Gain factor", capsize=2)
    pp.pubplot("$\mathbf{Gain\;Factor\;}$ " + detector.name, "Temperature [$^\circ$C]", "Gain factor", figure_directory + "gain_versus_temperature" + detector.datastorage_filename_append + ".png", legend=True)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #


def linearity_plot(detector: ccd, figure_directory: str, linearity_data = None):
    """
        A method that will produce a plot of the stability data
        from the ccd.linearity_estimation() and ccd.linearity_deviations() methods
    """

    # Plot the linearity data as a function of exposure times, and plot the ideal linear relation
    plt.errorbar(linearity_data[:, 3], linearity_data[:, 1], yerr=linearity_data[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label=detector.name, capsize=2)
    plt.plot(linearity_data[:, 3], np.zeros(len(linearity_data[:, 3])), ls='-', c='dodgerblue', lw=1, label="Ideal linear relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU / pixel", "Deviation in \%", figure_directory + "linearity_notimecal" + detector.datastorage_filename_append + ".png", legendlocation="lower left")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    plt.errorbar(linearity_data[:, 3], linearity_data[:, 1], yerr=linearity_data[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label=detector.name, capsize=2)
    plt.plot(linearity_data[:, 3], np.zeros(len(linearity_data[:, 3])), ls='-', c='dodgerblue', lw=1, label="Ideal linear relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU / pixel", "Deviation in \%", figure_directory + "linearity_zoom_notimecal" + detector.datastorage_filename_append + ".png", legendlocation="lower left", xlim=[0, 60e3], ylim=[-7.5, 3])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #


"""
def lightsource_stability_plot():
    """
        # A method that will produce a plot of the stability data
        # from the ccd.lightsource_stability() method
"""

    params = {'legend.fontsize': 7, 'legend.handlelength': 2}
    plt.rcParams.update(params)

    new = 0
    for exposure in range(0, 29):
        if exposure <= 10:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='-', lw=1, label="No. " + str(exposure))
        if 20 > exposure > 10:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='--', lw=1, label="No. " + str(exposure))
        if exposure >= 20:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='-.', lw=1, label="No. " + str(exposure))

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    pp.pubplot("$\mathbf{Lightsource\;\;stabillity}$", "Repeat no.", "\%- dev. from seq. mean", figure_directory + "lightsource.png", legend=False)
"""


def lightsource_stability_from_datafile_plot(detector: ccd, figure_directory: str, analysis_data_path: str):
    """
            A method that will produce a plot of the stability data
            from the ccd.lightsource_stability() method
        """

    filename = "lightsource_stability" + detector.datastorage_filename_append + ".txt"
    fullpath_lightsource_stability_data = analysis_data_path + filename
    stability_data = np.loadtxt(fullpath_lightsource_stability_data)

    for exposure in range(0, 10):
        plt.plot(np.asarray(range(0, 100)), stability_data[exposure, :], ls='-', lw=1, label= str(exposure + 1) + " s")

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fancybox=False, framealpha=1, edgecolor='inherit', fontsize = 10)
    pp.pubplot("$\mathbf{Lightsource\;\;stabillity}$", "Repeat no.", "\%- dev. from seq. mean", figure_directory + "lightsource_1to10" + detector.datastorage_filename_append + ".png", legend=False)

    exp_time = 0
    for exposure in range(11, 20):
        plt.plot(np.asarray(range(0, 99)), stability_data[exposure, :-1], ls='-', lw=1, label= str(10*(exp_time + 1) + 10) + " s")
        exp_time += 1

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fancybox=False, framealpha=1, edgecolor='inherit', fontsize = 10)
    pp.pubplot("$\mathbf{Lightsource\;\;stabillity}$", "Repeat no.", "\%- dev. from seq. mean", figure_directory + "lightsource_20to100" + detector.datastorage_filename_append + ".png", legend=False, xlim=[0, 100])

    exp_time = 0
    for exposure in range(20, 29):
        plt.plot(np.asarray(range(0, 100)), stability_data[exposure, :], ls='-', lw=1, label=str(100 + (exp_time + 1)) + " s")
        exp_time += 1

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fancybox=False, framealpha=1, edgecolor='inherit', fontsize = 10)
    pp.pubplot("$\mathbf{Lightsource\;\;stabillity}$", "Repeat no.", "\%- dev. from seq. mean",
               figure_directory + "lightsource_101to110" + detector.datastorage_filename_append + ".png", legend=False, xlim=[0, 100])


"""
def ron_dist_plot():
    plt.figure()
    i = 0
    temperatures = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    for dist in ron_dists_vs_temp:
        n, bins, patches = plt.hist(dist, bins=500, width=0.8, label= str(temperatures[i]) + "s")
        i += 1

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    pp.pubplot("$\mathbf{Readout\;\;noise\;\;distributions}$ ", "Bias value", "Arb. units.", figure_directory + "ron_dists.png", legend=False, xlim=[300, 475])
"""


def time_calibration_plot(detector: ccd, time_calibration, figure_directory: str):
    time_cal_linearity_array    = time_calibration[0]
    time_cal_linear_model_data  = time_calibration[1]
    time_cal_corrected_data     = time_calibration[2]
    time_cal_new_linear_model   = time_calibration[3]
    time_cal_deviations         = time_calibration[4]
    time_cal_errors             = time_calibration[5]
    time_cal_new_deviations     = time_calibration[6]

    plt.plot(time_cal_linearity_array[:, 0], time_cal_linear_model_data, ls='-', c='dodgerblue', lw=1, label="Ideal relationship")
    plt.errorbar(time_cal_linearity_array[:, 0], time_cal_linearity_array[:, 1], yerr=time_cal_linearity_array[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label=detector.name, capsize=2)  # + "$-10.0^\circ $ C")
    plt.plot(time_cal_corrected_data, time_cal_linearity_array[:, 1], ls='--', c='r', lw=1, marker='o', markersize=3, label="Corrected data")
    plt.plot(time_cal_corrected_data, time_cal_new_linear_model, ls='-', c='red', lw=1, label="New ideal relationship")

    pp.pubplot("$\mathbf{Time calibration}$ ", "Exposure time [s]", "Mean ADU/pixel", "time_calibration" + detector.datastorage_filename_append + ".png", legendlocation="lower right")  # xlim=[0, 2], ylim=[0,   1])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Plot the relative linearity deviations and the ideal relation
    plt.errorbar(time_cal_linearity_array[:, 1], time_cal_deviations, yerr=time_cal_errors[:], ls='--', c='k', lw=1, marker='o', markersize=3, label=detector.name, capsize=2)
    plt.plot(time_cal_linearity_array[:, 1], np.zeros(len(time_cal_linearity_array[:, 1])), ls='-', c='dodgerblue', lw=1, label="Ideal relation")
    plt.plot(time_cal_linearity_array[:, 1], time_cal_new_deviations,  ls='--', c='r', lw=1, marker='o', markersize=3, label=detector.name)

    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU/pixel", "Percentage deviation ", figure_directory + "time_calibration_deviations" + detector.datastorage_filename_append + ".png", legendlocation="upper left")


def shutter_test_plot(detector: ccd, shutter_test: str):
    filepath = util.get_path(shutter_test[:-1])
    hdul, header, imagedata = util.fits_handler(filepath)
    pp.plot_image(detector.bias_correction(imagedata), "Shuttertest", "x", "y", "detector: " + detector.name, "shutter_test" + detector.datastorage_filename_append + ".png", "shutter_test", scale=400)


def hot_pixels_plot(detector: ccd):
    hot_pixel_data = detector.hot_pixel_data
    plt.plot(hot_pixel_data[0], hot_pixel_data[1], '.', c="k", markersize = 2.5, label='Data')
    plt.plot([0, 1e3], [0, 1e3], c="dodgerblue", label='Ideal relationship')
    pp.pubplot("Hot pixels", "$e^-$/sec [90s]", "$e^-$/sec [1000s]", "hot_pixels_test" + detector.datastorage_filename_append + ".png", xlim=[0.5, 100.0], ylim=[0.5, 200], legendlocation="lower right")

    mask = detector.hot_pixel_mask
    padding = 10
    for first_id in range(padding, len(mask[:, 0])):
        for second_id in range(padding, len(mask[0, :])):
            if mask[first_id, second_id] == 1:
                for new_first_id in range(first_id - padding, first_id):
                    for new_second_id in range(second_id - padding, second_id):
                        mask[new_first_id, new_second_id] = 1

    pp.plot_image(mask, "Hot pixel mask", "x", "y", "detector: " + detector.name, "hot_pixel_mask" + detector.datastorage_filename_append + ".png", "hot_pixel_mask")


def produce_plots(detector: ccd,
                  figure_directory: str,
                  analysis_data_path: str,
                  linearity_data = None,
                  dark_current_data = None,
                  readout_noise_data = None,
                  gain_data = None,
                  time_calibration = None,
                  shutter_test: str = None,
                  hot_pixels = False,
                  lightsource_stabillity = False,
                  bias_sequence: str = None,
                  bias_frame_scaling:float = None
                  ):
    """
        A method that will produce all the relevant plots, from the data constructed
        from the characterization procedure.
    """
    print("\nProducing plots for CCD: " + detector.name + " ...")

    if bias_sequence is not None:
        gauss_dist_plot(detector, figure_directory, bias_sequence)
    if shutter_test is not None:
        shutter_test_plot(detector, shutter_test)
    if hot_pixels:
        hot_pixels_plot(detector)

    master_frame_plot(detector.master_bias, "master_bias_fig", "$\mathbf{Master\;bias\;frame}$ "          , detector.name, figure_directory + "master_bias" + detector.datastorage_filename_append + ".png"                    , scaling = 70)
    master_frame_plot(detector.master_dark, "master_dark_fig", "$\mathbf{Master\;dark\;current\;frame}$ " , detector.name, figure_directory + "master_dark" + detector.datastorage_filename_append + ".png",    raisetitle=True)
    master_frame_plot(detector.master_flat, "master_flat_fig", "$\mathbf{Master\;flat\;field\;frame}$ "   , detector.name, figure_directory + "master_flat" + detector.datastorage_filename_append + ".png"                    )

    if (dark_current_data is not None) and (readout_noise_data is not None):
        gain_plot(detector, figure_directory, gain_data)
        noise_plot(detector, figure_directory, dark_current_data, readout_noise_data)
    # if time_calibration is not None:
    # time_calibration_plot(detector, time_calibration, figure_directory)
    if linearity_data is not None:
        linearity_plot(detector, figure_directory, linearity_data)

    if lightsource_stabillity:
        lightsource_stability_from_datafile_plot(detector, figure_directory, analysis_data_path)
    # ron_dist_plot(detector)
