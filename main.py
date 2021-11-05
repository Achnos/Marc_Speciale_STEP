"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           August    2021 ----- #
# ----- Last edit:          1. November  2021 ----- #
# ------------------------------------------------- #
#####################################################
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pubplot as pp
import utilities as util
import numpy as np
import ccd


def gauss_dist_plot():
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
    bias_dist           =   atik_camera.master_bias.flatten()
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
    pp.pubplot("$\mathbf{Bias\;\;distribution:}$ " + atik_camera.name, "Bias value", "Arb. units.", figure_directory + "gauss_bias.png", legend=False, xlim=[260, 350])


def master_frame_plot(image: np.ndarray, figurename: str, title: str, label: str, filename: str, raisetitle: bool = False):
    """
        A method that will produce a plot of a given master frame (bias, dark or flat).

        :param np.ndarray image:
            - np.ndarray representing the image
        :param str figurename:
            - str representing the figurename
        :param str title:
            - str representing the title of the figure for printing
        :parameter str label:
            - The camera name used as a label in the plot
        :param str filename:
            - str representing the name of the file to be printed to
        :param bool raisetitle:
            - bool that toggles raising of the title in the figure
    """

    pp.plot_image(image, title, "x", "y", "Camera: " + label, filename, figurename, raisetitle=raisetitle)


def noise_plot():
    """
        A method that will produce a plot of the stability data
        from the ccd.dark_current_versus_temperature() and ccd.readout_noise_versus_temperature() methods
    """

    # Plot the dark current data, and adjust that axis' parameters
    dc_axis     =   plt.subplot()
    dc_line     =   dc_axis.plot(dark_current_data[:, 0], dark_current_data[:, 1], ls='--', c='k', lw=1, marker='o', markersize=3, label="Dark current")

    plt.ylim(-0.5, 14)
    plt.ylabel("Dark current [$\mathbf{e}^-$/sec]", {'fontweight': 'bold'})
    plt.xlabel("Temperature [$^\circ$C]", {'fontweight': 'bold'})
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Now plot the readout noise data and adjust the parameters of that axis
    ron_axis    =   dc_axis.twinx()
    ron_line    =   ron_axis.plot(readout_noise_data[:, 0], readout_noise_data[:, 1], ls='--', c='dodgerblue', lw=1, marker='o', markersize=3, label="Readout noise")

    plt.ylim(-0.5, 14)
    ron_axis.spines['right'].set_color('dodgerblue')
    ron_axis.tick_params(axis='y', colors='dodgerblue')
    ron_axis.yaxis.label.set_color('dodgerblue')
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Create two lines, that we may use to create a legend
    black_line  =   mlines.Line2D([], [], color='k', label="Dark current")
    blue_line   =   mlines.Line2D([], [], color='dodgerblue', label="Readout noise")

    plt.legend(handles=[black_line, blue_line])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    pp.pubplot("$\mathbf{Noise}$ " + atik_camera.name, "Temperature [$^\circ$C]", "Readout Noise [RMS $\mathbf{e}^-$/pixel]", figure_directory + "noise_versus_temperature.png", legend=False)


def linearity_plot():
    """
        A method that will produce a plot of the stability data
        from the ccd.linearity_estimation() and ccd.linearity_deviations() methods
    """

    # Plot the linearity data as a function of exposure times, and plot the ideal linear relation
    plt.plot(linearity_data[:, 0], ideal_linear_relation[:], ls='-', c='dodgerblue', lw=1, label="Ideal relationship")
    plt.errorbar(linearity_data[:, 0], linearity_data[:, 1], yerr=linearity_data[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)  # + "$-10.0^\circ $ C")

    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Exposure time [s]", "Mean ADU/pixel", "linearity.png", legendlocation="lower right")  # xlim=[0, 2], ylim=[0,   1])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Plot the relative linearity deviations and the ideal relation
    # plt.errorbar(linearity_data[:, 1], linearity_deviations[:], yerr=linearity_dev_err[:], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)
    plt.plot(linearity_data[:, 1], linearity_deviations[:], ls='--', c='k', lw=1,
                 marker='o', markersize=3, label=atik_camera.name)
    plt.plot(linearity_data[:, 1], np.zeros(len(linearity_data[:])), ls='-', c='dodgerblue', lw=1, label="Ideal relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU/pixel", "Percentage deviation ", figure_directory + "linearity_deviations.png", legendlocation="upper left", show=True)


def lightsource_stability_plot():
    """
        A method that will produce a plot of the stability data
        from the ccd.lightsource_stability() method
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


def ron_dist_plot():
    plt.figure()
    i = 0
    temperatures = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    for dist in ron_dists_vs_temp:
        n, bins, patches = plt.hist(dist, bins=500, width=0.8, label= str(temperatures[i]) + "s")
        i += 1

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    pp.pubplot("$\mathbf{Readout\;\;noise\;\;distributions}$ ", "Bias value", "Arb. units.", figure_directory + "ron_dists.png", legend=False, xlim=[300, 475])


def produce_plots():
    """
        A method that will produce all the relevant plots, from the data constructed
        from the characterization procedure.
    """

    gauss_dist_plot()

    master_frame_plot(atik_camera.master_bias, "master_bias_fig", "$\mathbf{Master\;\;bias\;\;frame}$ "            , atik_camera.name, figure_directory + "master_bias.png"                    )
    master_frame_plot(atik_camera.master_dark, "master_dark_fig", "$\mathbf{Master\;\;dark\;\;current\;\;frame}$ " , atik_camera.name, figure_directory + "master_dark.png",    raisetitle=True)
    master_frame_plot(atik_camera.master_flat, "master_flat_fig", "$\mathbf{Master\;\;flat\;\;field\;\;frame}$ "   , atik_camera.name, figure_directory + "master_flat.png"                    )

    noise_plot()
    linearity_plot()

    # lightsource_stability_plot()
    ron_dist_plot()


if __name__ == '__main__':
    # Initialize the camera in question
    atik_camera = ccd.CCD(  name                  =   "Atik 414EX mono",
                            gain_factor           =   0.28,
                            analysis_frame_path   =   "/home/marc/Dropbox/STEP_Speciale_Marc/data_from_characterization/",
                            master_frame_path     =   "/home/marc/Documents/Master_frames/")

    # Define the path of the data and where to put the figures
    file_directory      =   "/home/marc/Documents/FITS_files/"
    figure_directory    =   "/home/marc/Dropbox/STEP_Speciale_Marc/figures/"

    print("Directory of data: ", file_directory)
    print("Directory of figures: ", figure_directory, "\n")


    # Get the paths of the individual data sequences
    shutter_test                =    util.complete_path(file_directory + "shuttertest.fit"                                       , here=False)
    bias_sequence               =    util.complete_path(file_directory + "BIAS atik414ex 29-9-21 m10deg"                         , here=False)
    flat_sequence               =    util.complete_path(file_directory + "FLATS atik414ex 29-9-21 m10deg"                        , here=False)
    dark_current_sequence       =    util.complete_path(file_directory + "temp seq noise atik414ex 27-9-21"                      , here=False)
    readout_noise_sequence      =    util.complete_path(file_directory + "ron seq atik414ex 27-9-21"                             , here=False)
    # linearity_sequence          =    util.complete_path(file_directory + "total linearity  atik414ex"                            , here=False)
    linearity_sequence          =    util.complete_path(file_directory + "linearity with reference"                              , here=False)
    linearity_sequence_20C      =    util.complete_path(file_directory + "Linearity at 20 degrees celcius atik414ex 29-9-21"     , here=False)
    hot_pixel_sequence          =    util.complete_path(file_directory + "hotpix atik414ex 27-9-21"                              , here=False)
    zeropoint_sequence          =    util.complete_path(file_directory + "zeropoint value"                                       , here=False)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Construct data sequence class instances for use with the CCD class
    bias_dataseq                =    ccd.DataSequence(   path_of_data_series_input   =   bias_sequence           ,
                                                         exposure_time_input         =   0.001                    )
    flat_dataseq                =    ccd.DataSequence(   path_of_data_series_input   =   flat_sequence           ,
                                                         exposure_time_input         =   10                       )
    readout_noise_dataseq       =    ccd.DataSequence(   path_of_data_series_input   =   readout_noise_sequence  ,
                                                         num_of_data_points_input    =   16                      ,
                                                         num_of_repeats_input        =   100                      )
    dark_current_dataseq        =    ccd.DataSequence(   path_of_data_series_input   =   dark_current_sequence   ,
                                                         num_of_data_points_input    =   16                      ,
                                                         num_of_repeats_input        =   100                     ,
                                                         exposure_time_input         =   10                       )
    linearity_dataseq           =    ccd.DataSequence(   path_of_data_series_input   =   linearity_sequence      ,
                                                         num_of_data_points_input    =   29                      ,
                                                         num_of_repeats_input        =   100                      )
    hot_pixel_dataseq           =    ccd.DataSequence(   path_of_data_series_input   =   hot_pixel_sequence      ,
                                                         num_of_repeats_input        =   2                       ,
                                                         exposure_time_input         =   [90, 1000]               )
    zeropoint_dataseq           =    ccd.DataSequence(   path_of_data_series_input   =   zeropoint_sequence       ,
                                                         num_of_data_points_input    =   8                        ,
                                                         num_of_repeats_input        =   100                       )
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # atik_camera.master_bias_image(bias_sequence)
    # atik_camera.master_dark_current_image(bias_sequence, exposure_time=0.001)
    # atik_camera.master_flat_field_image(flat_sequence)

    # ADU_width = atik_camera.readout_noise_estimation(bias_sequence, temperature=-10)

    # atik_camera.hot_pixel_estimation(hot_pixel_sequence, num_of_repeats=2, exposure_time=[90, 1000])
    # atik_camera.test_zeropoint(zeropoint_sequence, num_of_data_points=8, num_of_repeats=100)

    # dark_current_data, readout_noise_data = atik_camera.noise_vs_temperature(dark_current_dataseq, readout_noise_dataseq)
    # dark_current_data       =   atik_camera.dark_current_vs_temperature(dark_current_sequence  , exposure_time=10   , num_of_repeats=100, num_of_temperatures=16)
    # readout_noise_data      =   atik_camera.readout_noise_vs_temperature(readout_noise_sequence, num_of_repeats=100, num_of_temperatures=16)
    # linearity_data          =   atik_camera.linearity_estimation(linearity_sequence, num_of_exposures=29, num_of_repeats=100)

    # ideal_linear_relation, linearity_deviations, linearity_dev_err = atik_camera.linearity_precision()

    # stabillity_data         =   atik_camera.test_lightsource_stabillity(linearity_sequence, num_of_data_points=29, num_of_repeats=100)
    # linearity_data_20C      =   atik_camera.linearity_estimation(linearity_sequence_20C, num_of_exposures=10, num_of_repeats=100)

    # Start characterization of the camera from using the data
    characterization = atik_camera.characterize(    bias_data_sequence          =   bias_dataseq            ,
                                                    flat_data_sequence          =   flat_dataseq            ,
                                                    dark_current_data_sequence  =   dark_current_dataseq    ,
                                                    readout_noise_data_sequence =   readout_noise_dataseq   ,
                                                    linearity_data_sequence     =   linearity_dataseq       ,
                                                    hot_pixel_data_sequence     =   hot_pixel_dataseq       ,
                                                    zero_point_data_sequence    =   zeropoint_dataseq        )

    dark_current_data       =   characterization[0]
    readout_noise_data      =   characterization[1]
    linearity_data          =   characterization[2]
    ideal_linear_relation   =   characterization[3]
    linearity_deviations    =   characterization[4]
    linearity_dev_err       =   characterization[5]
    stabillity_data         =   characterization[6]
    ron_dists_vs_temp       =   characterization[7]
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    produce_plots()
