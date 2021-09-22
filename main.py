import os
import ctypes
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pubplot as pp

import ccd
import libc_builder

if __name__ == '__main__':
    path_here = pathlib.Path().absolute()
    atik_camera = ccd.CCD("Atik 414EX mono")

    bias_sequence = str(path_here) + "/bias_sequence/"
    bias_image = atik_camera.master_bias_image(bias_sequence)
    plt.figure("bias_image_fig")
    plt.imshow(bias_image, vmin=0, cmap='gray')
    plt.colorbar()
    pp.pubplot("Bias image: " + atik_camera.name, "x", "y", "bias_image.png", legend=False, grid=False)

    readout_noise_estimate = atik_camera.readout_noise_estimation(bias_sequence)
    print(readout_noise_estimate)

    dark_current_sequence = str(path_here) + "/temperature_sequence/"
    dark_current_data = atik_camera.dark_current_vs_temperature(dark_current_sequence)
    plt.plot(dark_current_data[:, 0], dark_current_data[:, 1], label=atik_camera.name)
    pp.pubplot("Dark current", "Temperature [C°]", "Mean ADU/pixel", "dark_current_versus_temperature.png", legendlocation="upper left")

    linearity_sequence = str(path_here) + "/linearity/"
    linearity_data = atik_camera.linearity_estimation(linearity_sequence, 10, 10)
    plt.plot(linearity_data[:, 0], linearity_data[:, 1], ls='--', c='k', lw=1.5, marker='o', label=atik_camera.name)
    pp.pubplot("Linearity", "Exposure time [s]", "Mean ADU/pixel", "linearity.png", legendlocation="upper left")

