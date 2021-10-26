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
import matplotlib.pyplot as plt
import pubplot as pp
import utilities as util
import numpy as np
import ccd


def produce_plots():
    data_series = util.list_data(linearity_sequence)
    filepath = util.get_path(linearity_sequence + data_series[0])
    hdul, header, imagedata = util.fits_handler(filepath)
    bias_dist = imagedata.flatten()
    gaussdata = np.linspace(240, 375, 1000)
    gaussmean = np.mean(imagedata)
    gausswidth = np.std(bias_dist)
    print("The found width of the ADU distribution is ", gausswidth, " ADUs")
    print("This corresponds to a readout noise of ", gausswidth * 0.28 * np.sqrt(8 * np.log(2)))
    n, bins, patches = plt.hist(bias_dist, bins=1000, color='dodgerblue', width=0.8, label="Individual bias frame")
    gaussheight = np.amax(n)
    plt.plot(gaussdata, util.gaussian(gaussdata, gaussheight, gaussmean, gausswidth), c='navy', label="Gaussian")
    bias_dist = atik_camera.master_bias.flatten()
    gaussmean = np.mean(bias_dist)
    gausswidth = np.std(bias_dist)
    n, bins, patches = plt.hist(bias_dist, bins=350, color='steelblue', width=0.4, label="Master bias frame")
    gaussheight = np.amax(n)
    print("The found, reduced width of the ADU distribution is ", gausswidth, " ADUs")
    print("This corresponds to a readout noise of ", gausswidth * 0.28 * np.sqrt(8 * np.log(2)))
    plt.plot(gaussdata, util.gaussian(gaussdata, gaussheight, gaussmean, gausswidth), c='k', ls="--", label="Poissonian")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    pp.pubplot("$\mathbf{Bias\;\;distribution:}$ " + atik_camera.name, "Bias value", "Counts", "gauss_bias.png", legend=False, xlim = [260, 350])

    pp.plot_image(atik_camera.master_bias, "master_bias_fig")
    pp.pubplot("$\mathbf{Master\;\;bias\;\;image:}$ " + atik_camera.name, "x", "y", "master_bias.png", legend=False, grid=False)

    pp.plot_image(atik_camera.master_dark, "master_dark_fig")
    pp.pubplot("$\mathbf{Master\;\;dark\;\;current\;\;image:}$ " + atik_camera.name, "x", "y", "master_dark.png", legend=False, grid=False)

    pp.plot_image(atik_camera.master_flat, "master_flat_fig")
    pp.pubplot("$\mathbf{Master\;\;flat\;\;field\;\;image:}$ " + atik_camera.name, "x", "y", "master_flat.png", legend=False, grid=False)

    plt.errorbar(dark_current_data[:, 0], dark_current_data[:, 1], yerr=dark_current_data[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label="Dark current", capsize=2)
    # pp.pubplot("$\mathbf{Dark\;\; current}$", "Temperature [$^\circ$C]", "($\mathbf{e}^-$/sec)/pixel", "dark_current_versus_temperature.png", legendlocation="upper left")
    plt.errorbar(readout_noise_data[:, 0], readout_noise_data[:, 1], yerr=readout_noise_data[:, 2], ls='--', c='dodgerblue', lw=1, marker='o', markersize=3, label="Readout noise", capsize=2)
    pp.pubplot("$\mathbf{Noise}$ " + atik_camera.name, "Temperature [$^\circ$C]", "RMS $\mathbf{e}^-$/pixel", "noise_versus_temperature.png", legendlocation="upper left")

    plt.plot(linearity_data[:, 0], ideal_linear_relation[:], ls='-', c='dodgerblue', lw=1, label="Ideal relationship")
    plt.errorbar(linearity_data[:, 0], linearity_data[:, 1], yerr=linearity_data[:, 2], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)  # + "$-10.0^\circ $ C")
    # plt.plot(linearity_data_20C[:, 0], linearity_data_20C[:, 1], ls='-.', c='mediumspringgreen', lw=1, marker='o', markersize=3, label=atik_camera.name + "$20.0^\circ $ C")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Exposure time [s]", "Mean ADU/pixel", "linearity.png", legendlocation="lower right")

    plt.errorbar(linearity_data[:, 0], linearity_deviations[:], yerr=linearity_dev_err[:], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name, capsize=2)
    plt.plot(linearity_data[:, 0], np.zeros(len(linearity_data[:])), ls='-', c='dodgerblue', lw=1, label="Ideal relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Exposure time [s]", "Percentage deviation ", "linearity_deviations.png", legendlocation="upper left")

    params = {'legend.fontsize': 7,
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    new = 0
    for exposure in range(0, 30):
        if exposure <= 10:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='-', lw=1, label="No. " + str(exposure))
        if 20 > exposure > 10:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='--', lw=1, label="No. " + str(exposure))
        if exposure >= 20:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='-.', lw=1, label="No. " + str(exposure))

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    pp.pubplot("$\mathbf{Lightsource\;\;stabillity}$", "Repeat no.", "\%- dev. from seq. mean", "lightsource.png", legend=False)
    return


if __name__ == '__main__':
    atik_camera = ccd.CCD("Atik 414EX mono", gain_factor=0.28)

    file_directory = "/home/marc/Documents/FITS_files/"

    shutter_test                =   util.complete_path(file_directory + "shuttertest.fit"                                       , here=False)
    bias_sequence               =   util.complete_path(file_directory + "BIAS atik414ex 29-9-21 m10deg"                         , here=False)
    flat_sequence               =   util.complete_path(file_directory + "FLATS atik414ex 29-9-21 m10deg"                        , here=False)
    dark_current_sequence       =   util.complete_path(file_directory + "temp seq noise atik414ex 27-9-21"                      , here=False)
    readout_noise_sequence      =   util.complete_path(file_directory + "ron seq atik414ex 27-9-21"                             , here=False)
    linearity_sequence          =   util.complete_path(file_directory + "total linearity  atik414ex"                            , here=False)
    linearity_sequence_20C      =   util.complete_path(file_directory + "Linearity at 20 degrees celcius atik414ex 29-9-21"     , here=False)
    hot_pixel_sequence          =   util.complete_path(file_directory + "hotpix atik414ex 27-9-21"                              , here=False)
    zeropoint_sequence          =   util.complete_path(file_directory + "zeropoint value"                                       , here=False)

    atik_camera.master_bias_image(bias_sequence)
    atik_camera.master_dark_current_image(bias_sequence, exposure_time=0.001)
    atik_camera.master_flat_field_image(flat_sequence)

    ADU_width = atik_camera.readout_noise_estimation(bias_sequence, temperature=-10)

    atik_camera.hot_pixel_estimation(hot_pixel_sequence, num_of_repeats=2, exposure_time=[90, 1000])
    atik_camera.test_zeropoint(zeropoint_sequence, num_of_data_points=8, num_of_repeats=100)

    dark_current_data       =   atik_camera.dark_current_vs_temperature(dark_current_sequence  , exposure_time=10   , num_of_repeats=100, num_of_temperatures=16)
    readout_noise_data      =   atik_camera.readout_noise_vs_temperature(readout_noise_sequence, exposure_time=0.001, num_of_repeats=100, num_of_temperatures=16)
    linearity_data          =   atik_camera.linearity_estimation(linearity_sequence, num_of_exposures=30, num_of_repeats=100)

    ideal_linear_relation, linearity_deviations, linearity_dev_err = atik_camera.linearity_precision()

    stabillity_data         =   atik_camera.test_lightsource_stabillity(linearity_sequence, num_of_data_points=30, num_of_repeats=100)
    # linearity_data_20C      =   atik_camera.linearity_estimation(linearity_sequence_20C, num_of_exposures=10, num_of_repeats=100)

    produce_plots()