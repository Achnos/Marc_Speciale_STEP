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
    plt.errorbar(np.concatenate((linearity_data[0:3, 3], linearity_data[6:, 3])), np.concatenate((linearity_data[0:3, 1], linearity_data[6:, 1])), yerr=np.concatenate((linearity_data[0:3, 2], linearity_data[6:, 2])), ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)
    plt.plot(np.concatenate((linearity_data[0:3, 3], linearity_data[6:, 3])), np.zeros(len(np.concatenate((linearity_data[0:3, 3], linearity_data[6:, 3])))), ls='-', c='dodgerblue', lw=1, label="Ideal linear relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU / pixel", "Deviation in \%", figure_directory + "linearity.png", legendlocation="lower right")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    plt.errorbar(np.concatenate((linearity_data[0:3, 3], linearity_data[6:, 3])), np.concatenate((linearity_data[0:3, 1], linearity_data[6:, 1])), yerr=np.concatenate((linearity_data[0:3, 2], linearity_data[6:, 2])), ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)
    plt.plot(np.concatenate((linearity_data[0:3, 3], linearity_data[6:, 3])), np.zeros(len(np.concatenate((linearity_data[0:3, 3], linearity_data[6:, 3])))), ls='-', c='dodgerblue', lw=1, label="Ideal linear relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU / pixel", "Deviation in \%", figure_directory + "linearity_zoom.png", legendlocation="lower right", xlim=[0, 60e3], ylim=[-7.5, 3])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

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

def time_calibration_plot():
    time_cal_linearity_array    = time_calibration[0]
    time_cal_linear_model_data  = time_calibration[1]
    time_cal_corrected_data     = time_calibration[2]
    time_cal_new_linear_model   = time_calibration[3]
    time_cal_deviations         = time_calibration[4]
    time_cal_errors             = time_calibration[5]
    time_cal_new_deviations     = time_calibration[6]

    plt.plot(time_cal_linearity_array[:, 0], time_cal_linear_model_data, ls='-', c='dodgerblue', lw=1, label="Ideal relationship")
    plt.errorbar(time_cal_linearity_array[:, 0], time_cal_linearity_array[:, 1], yerr=time_cal_linearity_array[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)  # + "$-10.0^\circ $ C")
    plt.plot(time_cal_corrected_data, time_cal_linearity_array[:, 1], ls='--', c='r', lw=1, marker='o', markersize=3, label="Corrected data")
    plt.plot(time_cal_corrected_data, time_cal_new_linear_model, ls='-', c='red', lw=1, label="New ideal relationship")

    pp.pubplot("$\mathbf{Time calibration}$ ", "Exposure time [s]", "Mean ADU/pixel", "time_calibration.png", legendlocation="lower right")  # xlim=[0, 2], ylim=[0,   1])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Plot the relative linearity deviations and the ideal relation
    plt.errorbar(time_cal_linearity_array[:, 1], time_cal_deviations, yerr=time_cal_errors[:], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)
    plt.plot(time_cal_linearity_array[:, 1], np.zeros(len(time_cal_linearity_array[:, 1])), ls='-', c='dodgerblue', lw=1, label="Ideal relation")
    plt.plot(time_cal_linearity_array[:, 1], time_cal_new_deviations,  ls='--', c='r', lw=1, marker='o', markersize=3, label=atik_camera.name)

    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Mean ADU/pixel", "Percentage deviation ", figure_directory + "time_calibration_deviations.png", legendlocation="upper left")


def produce_plots():
    """
        A method that will produce all the relevant plots, from the data constructed
        from the characterization procedure.
    """

    # gauss_dist_plot()

    master_frame_plot(atik_camera.master_bias, "master_bias_fig", "$\mathbf{Master\;\;bias\;\;frame}$ "            , atik_camera.name, figure_directory + "master_bias.png"                    )
    master_frame_plot(atik_camera.master_dark, "master_dark_fig", "$\mathbf{Master\;\;dark\;\;current\;\;frame}$ " , atik_camera.name, figure_directory + "master_dark.png",    raisetitle=True)
    master_frame_plot(atik_camera.master_flat, "master_flat_fig", "$\mathbf{Master\;\;flat\;\;field\;\;frame}$ "   , atik_camera.name, figure_directory + "master_flat.png"                    )

    # noise_plot()
    # time_calibration_plot()
    linearity_plot()

    # lightsource_stability_plot()
    # ron_dist_plot()


if __name__ == '__main__':
    # These bools can be changed in order to change the characterization procedure
    # For example  if "construct_master_bias" is set to "True", then the characterization
    # method will construct a new master bias frame from the data fed to the procedure.
    # If these are set to "False" data from previous runs are used instead.
    construct_master_bias       =   False
    construct_master_dark       =   False
    construct_master_flat       =   False
    do_noise_estimation         =   False
    do_time_calibration         =   True
    do_linearity_estimation     =   True

    # These are the paths at which to save the constructed master frames and data sets
    # from the analysis procedures. If these methods are not used in characterization,
    # these paths are used as paths at which to collect the data
    analysis_data_path          =   "/home/marc/Dropbox/STEP_Speciale_Marc/data_from_characterization/"
    master_frame_path           =   "/home/marc/Documents/Master_frames/"

    # Initialize the camera in question
    atik_camera = ccd.CCD( name                  =   "Atik 414EX mono"  ,
                           gain_factor           =   0.28               ,
                           analysis_data_path    =   analysis_data_path ,
                           master_frame_path     =   master_frame_path   )

    atik_camera.load_ccd_characterization_data( construct_master_bias       =   construct_master_bias                   ,
                                                construct_master_dark       =   construct_master_dark                   ,
                                                construct_master_flat       =   construct_master_flat                   ,
                                                do_noise_estimation         =   do_noise_estimation                     ,
                                                do_time_calibration         =   do_time_calibration                     ,
                                                do_linearity_estimation     =   do_linearity_estimation                 ,
                                                path_of_master_bias_frame   =   "master_bias.txt"                       ,
                                                path_of_master_dark_frame   =   "master_dark.txt"                       ,
                                                path_of_master_flat_frame   =   "master_flat.txt"                       ,
                                                path_of_linearity_data      =   "linearity.txt"                         ,
                                                path_of_dark_current_data   =   "dark_current_versus_temperature.txt"   ,
                                                path_of_readout_noise_data  =   "readout_noise_versus_temperature.txt"   )

    # Define the path of the data and where to put the figures
    file_directory      =   "/home/marc/Documents/FITS_files/"
    figure_directory    =   "/home/marc/Dropbox/STEP_Speciale_Marc/figures/"

    print("Directory of data:    ", file_directory        )
    print("Directory of figures: ", figure_directory, "\n")


    # Get the paths of the individual data sequences
    shutter_test                =    util.complete_path(file_directory + "shuttertest.fit"                                       , here=False)
    bias_sequence               =    util.complete_path(file_directory + "BIAS atik414ex 29-9-21 m10deg"                         , here=False)
    flat_sequence               =    util.complete_path(file_directory + "FLATS atik414ex 29-9-21 m10deg"                        , here=False)
    dark_current_sequence       =    util.complete_path(file_directory + "temp seq noise atik414ex 27-9-21"                      , here=False)
    readout_noise_sequence      =    util.complete_path(file_directory + "ron seq atik414ex 27-9-21"                             , here=False)
    # linearity_sequence          =    util.complete_path(file_directory + "total linearity with reference"                        , here=False)
    linearity_sequence_20C      =    util.complete_path(file_directory + "Linearity at 20 degrees celcius atik414ex 29-9-21"     , here=False)
    linearity_sequence          =    util.complete_path(file_directory + "linearity dimmed"     , here=False)
    time_calibration_sequence   =    util.complete_path(file_directory + "time calibration 15-11-21"                             , here=False)
    new_timecal_sequence        =    util.complete_path(file_directory + "new time calibration"                                  , here=False)
    hot_pixel_sequence          =    util.complete_path(file_directory + "hotpix atik414ex 27-9-21"                              , here=False)
    zeropoint_sequence          =    util.complete_path(file_directory + "zeropoint value"                                       , here=False)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    # Construct data sequence class instances for use with the CCD class
    bias_dataseq                =    ccd.DataSequence(   path_of_data_series_input   =   bias_sequence             ,
                                                         exposure_time_input         =   0.001                     )
    flat_dataseq                =    ccd.DataSequence(   path_of_data_series_input   =   flat_sequence             ,
                                                         exposure_time_input         =   10                        )
    readout_noise_dataseq       =    ccd.DataSequence(   path_of_data_series_input   =   readout_noise_sequence    ,
                                                         num_of_data_points_input    =   16                        ,
                                                         num_of_repeats_input        =   100                       )
    dark_current_dataseq        =    ccd.DataSequence(   path_of_data_series_input   =   dark_current_sequence     ,
                                                         num_of_data_points_input    =   16                        ,
                                                         num_of_repeats_input        =   100                       ,
                                                         exposure_time_input         =   10                        )
    linearity_dataseq           =    ccd.DataSequence(   path_of_data_series_input   =   linearity_sequence        ,
                                                         num_of_data_points_input    =   24                        ,
                                                         num_of_repeats_input        =   10                        ,
                                                         exposure_time_input         =   10                        )
    time_calibration_dataseq    =    ccd.DataSequence(   path_of_data_series_input   =   time_calibration_sequence ,
                                                         num_of_data_points_input    =   20                        ,
                                                         num_of_repeats_input        =   10                        )
    new_timecal_dataseq         =    ccd.DataSequence(   path_of_data_series_input   =   new_timecal_sequence      ,
                                                         num_of_repeats_input        =   10                        )
    hot_pixel_dataseq           =    ccd.DataSequence(   path_of_data_series_input   =   hot_pixel_sequence        ,
                                                         num_of_repeats_input        =   2                         ,
                                                         exposure_time_input         =   [90, 1000]                )
    zeropoint_dataseq           =    ccd.DataSequence(   path_of_data_series_input   =   zeropoint_sequence        ,
                                                         num_of_data_points_input    =   8                         ,
                                                         num_of_repeats_input        =   100                       )

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    characterization = atik_camera.characterize(    bias_data_sequence               =   bias_dataseq             ,
                                                    flat_data_sequence               =   flat_dataseq             ,
                                                    dark_current_data_sequence       =   dark_current_dataseq     ,
                                                    readout_noise_data_sequence      =   readout_noise_dataseq    ,
                                                    linearity_data_sequence          =   linearity_dataseq        ,
                                                    hot_pixel_data_sequence          =   hot_pixel_dataseq        ,
                                                    zero_point_data_sequence         =   zeropoint_dataseq        ,
                                                    # time_calibration_data_sequence   =   time_calibration_dataseq
                                                    time_calibration_data_sequence   =   new_timecal_dataseq      )

    dark_current_data       =   characterization[0]
    readout_noise_data      =   characterization[1]
    time_calibration        =   characterization[2]
    linearity_data          =   characterization[3]
    """
    ideal_linear_relation   =   characterization[3]
    linearity_deviations    =   characterization[4]
    linearity_dev_err       =   characterization[5]
    stabillity_data         =   characterization[6]
    ron_dists_vs_temp       =   characterization[7]
    """
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    produce_plots()
