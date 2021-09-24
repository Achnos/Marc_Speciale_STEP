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
import ccd


if __name__ == '__main__':
    atik_camera = ccd.CCD("Atik 414EX mono", gain_factor=0.28)

    bias_sequence           =   util.complete_path("bias_sequence")
    flat_sequence           =   util.complete_path("flats")
    dark_current_sequence   =   util.complete_path("temperature_sequence")
    linearity_sequence      =   util.complete_path("linearity")

    atik_camera.master_bias_image(bias_sequence)
    atik_camera.master_dark_current_image(bias_sequence, exposure_time=0.001)
    atik_camera.master_flat_field_image(flat_sequence)
    atik_camera.readout_noise_estimation(bias_sequence, temperature=23.6)

    dark_current_data       =   atik_camera.dark_current_vs_temperature(dark_current_sequence, exposure_time=10)
    linearity_data          =   atik_camera.linearity_estimation(linearity_sequence, 10, 10)

    pp.plot_image(atik_camera.master_bias, "master_bias_fig")
    pp.pubplot("Master bias image: " + atik_camera.name, "x", "y", "master_bias.png", legend=False, grid=False)
    pp.plot_image(atik_camera.master_dark, "master_dark_fig")
    pp.pubplot("Master dark current image: " + atik_camera.name, "x", "y", "master_dark.png", legend=False, grid=False)
    pp.plot_image(atik_camera.master_flat, "master_flat_fig")
    pp.pubplot("Master flat field image: " + atik_camera.name, "x", "y", "master_flat.png", legend=False, grid=False)
    plt.plot(dark_current_data[:, 0], dark_current_data[:, 1], label=atik_camera.name)
    pp.pubplot("Dark current", "Temperature [C°]", "Mean ADU/pixel", "dark_current_versus_temperature.png", legendlocation="upper left")
    plt.plot(linearity_data[:, 0], linearity_data[:, 1], ls='--', c='k', lw=1.5, marker='o', label=atik_camera.name)
    pp.pubplot("Linearity", "Exposure time [s]", "Mean ADU/pixel", "linearity.png", legendlocation="upper left")

