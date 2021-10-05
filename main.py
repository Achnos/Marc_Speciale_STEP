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
    pp.plot_image(atik_camera.master_bias, "master_bias_fig")
    pp.pubplot("$\mathbf{Master\;\;bias\;\;image:}$ " + atik_camera.name, "x", "y", "master_bias.png", legend=False, grid=False)
    pp.plot_image(atik_camera.master_dark, "master_dark_fig")
    pp.pubplot("$\mathbf{Master\;\;dark\;\;current\;\;image:}$ " + atik_camera.name, "x", "y", "master_dark.png", legend=False, grid=False)
    pp.plot_image(atik_camera.master_flat, "master_flat_fig")
    pp.pubplot("$\mathbf{Master\;\;flat\;\;field\;\;image:}$ " + atik_camera.name, "x", "y", "master_flat.png", legend=False, grid=False)
    plt.plot(dark_current_data[:, 0], dark_current_data[:, 1], ls='--', c='k', lw=1, marker='o', markersize=3, label="Dark current")
    # pp.pubplot("$\mathbf{Dark\;\; current}$", "Temperature [$^\circ$C]", "($\mathbf{e}^-$/sec)/pixel", "dark_current_versus_temperature.png", legendlocation="upper left")
    plt.plot(readout_noise_data[:, 0], readout_noise_data[:, 1], ls='--', c='dodgerblue', lw=1, marker='o', markersize=3, label="Readout noise")
    pp.pubplot("$\mathbf{Noise}$ " + atik_camera.name, "Temperature [$^\circ$C]", "RMS $\mathbf{e}^-$/pixel", "noise_versus_temperature.png", legendlocation="upper left")
    plt.plot(linearity_data[:-3, 0], ideal_linear_relation[:-3], ls='-', c='dodgerblue', lw=1, label="Ideal relationship")
    plt.plot(linearity_data[:, 0], linearity_data[:, 1], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name)
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Exposure time [s]", "Mean ADU/pixel", "linearity.png", legendlocation="lower right")
    plt.plot(linearity_data[:-3, 0], linearity_deviations[:-3], ls='--', c='k', lw=1, marker='o', markersize=3, label=atik_camera.name)
    plt.plot(linearity_data[:-3, 0], np.zeros(len(linearity_data[:-3])), ls='-', c='dodgerblue', lw=1, label="Ideal relation")
    pp.pubplot("$\mathbf{Linearity}$ $-10.0^\circ $ C ", "Exposure time [s]", "Percentage deviation ", "linearity_deviations.png", legendlocation="upper left")

    params = {'legend.fontsize': 7,
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    new = 0
    for exposure in range(0, 11):
        if exposure - 7 >= 0:
            new = 1

        if new == 0:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='-', lw=1, label=str(exposure * 10) + "s")
        else:
            plt.plot(np.asarray(range(0, 100)), stabillity_data[exposure, :], ls='--', lw=1, label=str(exposure * 10) + "s")

    pp.pubplot("$\mathbf{Lightsource\;\;stabillity}$", "Repeat no.", "\%- dev. from seq. mean", "lightsource.png", legendlocation="upper right")
    return


if __name__ == '__main__':
    atik_camera = ccd.CCD("Atik 414EX mono", gain_factor=0.28)

    file_directory = "/home/marc/Documents/FITS_files/"

    bias_sequence           =   util.complete_path(file_directory + "BIAS atik414ex 29-9-21 m10deg"   , here=False)
    flat_sequence           =   util.complete_path(file_directory + "FLATS atik414ex 29-9-21 m10deg"  , here=False)
    dark_current_sequence   =   util.complete_path(file_directory + "temp seq noise atik414ex 27-9-21", here=False)
    readout_noise_sequence  =   util.complete_path(file_directory + "ron seq atik414ex 27-9-21"       , here=False)
    linearity_sequence      =   util.complete_path(file_directory + "linearity atik414ex 27-9-21"     , here=False)
    hot_pixel_sequence      =   util.complete_path(file_directory + "hotpix atik414ex 27-9-21"        , here=False)

    atik_camera.master_bias_image(bias_sequence)
    atik_camera.master_dark_current_image(bias_sequence, exposure_time=0.001)
    atik_camera.master_flat_field_image(flat_sequence)
    atik_camera.readout_noise_estimation(bias_sequence, temperature=-10)

    atik_camera.hot_pixel_estimation(hot_pixel_sequence, num_of_repeats=2, exposure_time=[90, 1000])

    dark_current_data       =   atik_camera.dark_current_vs_temperature(dark_current_sequence  , exposure_time=10   , num_of_repeats=100, num_of_temperatures=16)
    readout_noise_data      =   atik_camera.readout_noise_vs_temperature(readout_noise_sequence, exposure_time=0.001, num_of_repeats=100, num_of_temperatures=16)
    linearity_data          =   atik_camera.linearity_estimation(linearity_sequence, num_of_exposures=11, num_of_repeats=100)
    stabillity_data         =   atik_camera.test_lightsource_stabillity(linearity_sequence, num_of_data_points=11, num_of_repeats=100)

    ideal_linear_relation, linearity_deviations = atik_camera.linearity_precision()

    produce_plots()