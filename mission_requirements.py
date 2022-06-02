"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           August    2021 ----- #
# ----- Last edit:         8.  April     2022 ----- #
# ------------------------------------------------- #
#####################################################
"""
import matplotlib.pyplot as plt
import utilities as util
import ccd
import pubplot as pp
from scipy.linalg import lstsq
import numpy as np
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.centroids import centroid_sources, centroid_com

plt.style.use(['science', 'ieee', 'vibrant'])


def compute_noise(camera                : ccd,
                  aperture_counts_list  : np.ndarray,
                  annulus_counts_list   : np.ndarray,
                  aperture_areas        : np.ndarray,
                  annulus_areas         : np.ndarray
                  ):
    print("  Computing total noise...")
    # Total noise calculations
    # Compute the ratio of the areas of the apertures to annuli
    area_ratio  =  np.divide(aperture_areas, annulus_areas)

    # Compute the standard deviation of the photonic (and readout noise) distribution in the first aperture
    count_dist_width_first = np.sqrt(camera.gain_factor * aperture_counts_list[:, 0]
                                     + np.multiply(camera.gain_factor * annulus_counts_list[:, 0],
                                                   np.square(area_ratio) + area_ratio)
                                     + np.multiply(aperture_areas,
                                                   np.multiply(np.add(1, area_ratio),
                                                               camera.gain_factor * camera.readout_noise_level ** 2)
                                                   )
                                     )  # eq (9.11.28) in Detector book

    # Compute, from the standard deviation, the relative error
    count_dist_error_first = np.divide(count_dist_width_first, aperture_counts_list[:, 0] * camera.gain_factor)  # eq (9.11.29)

    # Compute the standard deviation of the photonic (and readout noise) distribution in the second aperture
    count_dist_width_second = np.sqrt(camera.gain_factor * aperture_counts_list[:, 1]
                                      + np.multiply(camera.gain_factor * annulus_counts_list[:, 1],
                                                    np.square(area_ratio) + area_ratio)
                                      + np.multiply(aperture_areas,
                                                    np.multiply(np.add(1, area_ratio),
                                                                camera.gain_factor * camera.readout_noise_level ** 2)
                                                    )
                                      )  # eq (9.11.28) in Detector book

    # Compute, from the standard deviation, the relative error
    count_dist_error_second = np.divide(count_dist_width_second, aperture_counts_list[:, 1] * camera.gain_factor)  # eq (9.11.29)

    total_noise = np.mean( np.sqrt( np.square(count_dist_error_first) + np.square(count_dist_error_second) ) )

    return total_noise


def differential_aperture_photometry(   camera:                 ccd     ,
                                        data_series:            list    ,
                                        path_of_data_series:    str     ,
                                        num_of_images:          int     ,
                                        aperture_positions:     list    ,
                                        aperture_radius:        float   ,
                                        annulus_radii:          list
                                        ):
    print("  Performing aperture photometry with aperture positions:")
    print("   ", aperture_positions)
    print("   Aperture radius: \n    ", aperture_radius)
    print("   Annulus radii:   \n    ", annulus_radii)

    # Prepare a set of lists for the data analysis below
    differential_flux_data              =   []
    timestamps                          =   []
    aperture_data_list                  =   []
    aperture_areas                      =   []
    annulus_areas                       =   []
    aperture_counts_list                =   []
    annulus_counts_list                 =   []

    id = 0
    for imageid in range(0, num_of_images):
        # Load the image, and correct it
        filepath                 =   util.get_path(path_of_data_series + data_series[imageid])
        hdul, header, imagedata  =   util.fits_handler(filepath)
        imagedata_corrected      =   camera.hot_pixel_correction(camera.flat_field_correction(camera.bias_correction(imagedata)))

        # Append to a list of timestamps for use in the figure below
        try:
            timestamp = id  # =   header['DATE-OBS']
            timestamps.append(str(timestamp[11:16]))
        except:
            timestamp = id
            timestamps.append(timestamp)

        # Initiallize the x and y positions of the apertures from the function call
        aperture_x_position = (aperture_positions[0][0], aperture_positions[1][0])  # (first aperture x position, second aperture x position)
        aperture_y_position = (aperture_positions[0][1], aperture_positions[1][1])  # (first aperture y position, second aperture y position)
        # Determine centroid positions of the light in the apertures
        box_size_for_centroid_search = int(2 * aperture_radius)
        aperture_centroid_x_pos, aperture_centroid_y_pos = centroid_sources(imagedata_corrected, aperture_x_position, aperture_y_position, box_size=box_size_for_centroid_search, centroid_func=centroid_com)

        # Construct two apertures of same radii and two annuli of the same radii
        aperture            =   CircularAperture( [aperture_positions[0], aperture_positions[1]], r=aperture_radius)
        annulus             =   CircularAnnulus(  [aperture_positions[0], aperture_positions[1]], r_in=annulus_radii[0], r_out=annulus_radii[1])
        aperatures          =   [aperture, annulus]  # Collect the objects in a list
        photometry_table    =   aperture_photometry(imagedata_corrected, aperatures)  # Do aperture photometry
        for col in photometry_table.colnames:
            photometry_table[col].info.format = '%.8g'

        # The aperture_sum_0 column refers to the first aperture in the list of input apertures (i.e., the circular aperture)
        # and the aperture_sum_1 column refers to the second aperture (i.e., the circular annulus).
        # Note that we cannot simply subtract the aperture sums because the apertures have different areas.
        annulus_counts                                          =   photometry_table['aperture_sum_1']
        aperture_counts                                         =   photometry_table['aperture_sum_0']
        background_counts_per_pixel                             =   annulus_counts / annulus.area
        background_counts_in_aperture                           =   background_counts_per_pixel * aperture.area
        aperture_counts_background_subtracted                   =   aperture_counts - background_counts_in_aperture
        # Collect the corrected fluxes in a new table
        photometry_table['residual_aperture_sum']               =   aperture_counts_background_subtracted / aperture.area  # The flux per pixel
        photometry_table['residual_aperture_sum'].info.format   =   '%.3g'  # for consistent table output
        differential_flux                                       =   photometry_table['residual_aperture_sum'][0] / photometry_table['residual_aperture_sum'][1]

        # Fill out lists
        differential_flux_data.append(differential_flux)
        aperture_data_list.append([photometry_table['residual_aperture_sum'][0], float(aperture_centroid_x_pos[0]), float(aperture_centroid_y_pos[0])])
        aperture_areas.append(aperture.area)
        annulus_areas.append(annulus.area)
        aperture_counts_list.append([aperture_counts[0], aperture_counts[1]])
        annulus_counts_list.append([annulus_counts[0], annulus_counts[1]])

        id += 1

    # Convert to numpy arrays
    aperture_areas          =   np.asarray(aperture_areas)
    annulus_areas           =   np.asarray(annulus_areas)
    aperture_counts_list    =   np.asarray(aperture_counts_list)
    annulus_counts_list     =   np.asarray(annulus_counts_list)
    aperture_data_list      =   np.asarray(aperture_data_list)
    timestamps              =   np.asarray(timestamps)

    # Compute total noise in the experiment
    total_noise = compute_noise(camera, aperture_counts_list, annulus_counts_list, aperture_areas, annulus_areas)

    return [differential_flux_data, aperture_data_list, timestamps, total_noise]


def drift_correct_data(dataset: list):
    num_of_datapoints = len(dataset)
    drift_corrected_data = []
    for data_id in range(1, num_of_datapoints - 1):
        new_datapoint = dataset[data_id] - ((dataset[data_id - 1] + dataset[data_id + 1]) / 2)
        drift_corrected_data.append(new_datapoint)
    return drift_corrected_data


def move_tol_pixel(requirement: float, linear_model_movement):
    return requirement / linear_model_movement[0]


def output_requirements(aperture_data_list: np.ndarray,
                        diffflux_percentage_dev: np.ndarray,
                        num_of_images: int,
                        requirement_input: float
                        ):

    # Fit a linear model to the flux variation as a function of positions
    linear_model_x_movement = np.polyfit(aperture_data_list[:, 1], diffflux_percentage_dev, 1)
    linear_model_y_movement = np.polyfit(aperture_data_list[:, 2], diffflux_percentage_dev, 1)
    linear_model_x_function = np.poly1d(linear_model_x_movement)
    linear_model_y_function = np.poly1d(linear_model_y_movement)
    fitquery_x = np.linspace(np.min(aperture_data_list[:, 1]), np.max(aperture_data_list[:, 1]), num_of_images)
    fitquery_y = np.linspace(np.min(aperture_data_list[:, 2]), np.max(aperture_data_list[:, 2]), num_of_images)

    print(" Linear model fit to x movement: \n ", linear_model_x_movement)
    print(" Linear model fit to y movement: \n ", linear_model_y_movement)

    print(" For the flux to change at most ΔF = ", requirement_input, "%")
    print(" We must require at most ")
    print("  Δx = ", move_tol_pixel(requirement_input, linear_model_x_movement), "\n  Δy = ",
          move_tol_pixel(requirement_input, linear_model_y_movement))

    return [linear_model_x_function, linear_model_y_function, fitquery_x, fitquery_y]


def plot_variations(camera:                         ccd,
                    aperture_data_list:             np.ndarray,
                    differential_flux_list:         np.ndarray,
                    diffflux_percentage_dev:        np.ndarray,
                    fit_query_pts_x:                np.ndarray,
                    fit_query_pts_y:                np.ndarray,
                    linear_model_x_function,
                    linear_model_y_function,
                    timestamp_ticks:                np.ndarray,
                    drift_corrected_diffflux_data,
                    ):

    # Plot the deviation in the differential flux as a function of x-movement
    plt.plot(aperture_data_list[:, 1], diffflux_percentage_dev, '.', markersize=3, color="k", label="Flux")
    plt.plot(fit_query_pts_x, linear_model_x_function(fit_query_pts_x), color="r", linewidth=1.5)
    pp.pubplot("Flux dep. on X pos.", "X pixel pos.", "Diff. corr. flux deviation (\%)", "fluxVsX" + camera.datastorage_filename_append + ".png", legend=False)

    # Plot the deviation in the differential flux as a function of y-movement
    plt.plot(aperture_data_list[:, 2], diffflux_percentage_dev, '.', markersize=3, color="k", label="Flux")
    plt.plot(fit_query_pts_y, linear_model_y_function(fit_query_pts_y), color="r", linewidth=1.5)
    pp.pubplot("Flux dep. on Y pos.", "Y pixel pos.", "Diff. corr. flux deviation (\%)", "fluxVsY" + camera.datastorage_filename_append + ".png", legend=False)

    # Plot the deviation in the differential flux as a function of variations in the raw flux
    plt.plot(aperture_data_list[:, 0], differential_flux_list, '.',  markersize=3, color="k", label="Flux")
    pp.pubplot("Diff. flux dep. on raw flux", "Differential corrected flux", "Raw flux corrected", "fluxVsB" + camera.datastorage_filename_append + ".png", legend=False)

    # Plot the differential flux as a function of time, as well as
    # the position as a function of time
    fig, ax = plt.subplots()

    # Construct extra axes to tidy up the plot a bit
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, host = plt.subplots()
    # Move axis out a bit
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.4))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)

    p1, = host.plot(timestamp_ticks[1:-1], drift_corrected_diffflux_data, '.', color="k", markersize=3, label="Flux vs time")
    p2, = par1.plot(timestamp_ticks, aperture_data_list[:, 1], '.', color="r", markersize=3, label="X pos. vs time")
    p3, = par2.plot(timestamp_ticks, aperture_data_list[:, 2], '.', color="b", markersize=3, label="Y pos. vs time")

    """host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)"""

    host.set_xlabel("Timestamp")
    host.set_ylabel("Flux")
    par1.set_ylabel("X pos.")
    par2.set_ylabel("Y pos.")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)
    lines = [p1, p2, p3]

    # Move legend outside
    host.legend(lines, [l.get_label() for l in lines], bbox_to_anchor=(1.5, 1), loc="upper left")
    # Print the plot nicely
    pp.pubplot("Flux and position as a function of time", "Timestamp", "Y pos.", "fluxposvtimecorr" + camera.datastorage_filename_append + ".png", legend=False)


def uncorrectable_variations_plot(camera:                   ccd,
                                  aperture_data_list:       np.ndarray,
                                  differential_flux_list:   np.ndarray,
                                  timestamp_ticks:          np.ndarray
                                  ):
    print(" Analyzing data, and constructing linear fit...")
    # Begin analysis of corrections
    dX = np.subtract(aperture_data_list[:, 1], np.mean(aperture_data_list[:, 1]))  # Changes in x positions
    dY = np.subtract(aperture_data_list[:, 2], np.mean(aperture_data_list[:, 2]))  # Changes in y positions
    dB = np.subtract(aperture_data_list[:, 0], np.mean(aperture_data_list[:, 0]))  # Changes in flux
    y = differential_flux_list

    # Construct a linear fit in four parameters (first list of ones is the offset)
    M = np.asarray(np.transpose(np.asarray([np.ones(len(dX)), dX, dY, dB])))
    p, res, rnk, s = lstsq(M, y)
    print("Found linear coefficients are: ", p, res, rnk, s)
    fit = p[0] + dX * p[1] + dY * p[2] + dB * p[3]

    mean = np.mean(np.subtract(differential_flux_list, fit))
    error = np.std(np.subtract(differential_flux_list, fit))
    precision = ((error + 1) / (mean + 1) - 1) * 100
    print("The precision is ", precision, "%")

    # plot the differential flux as a function of time
    # as well as the fit to the flux as a function of time
    fig, ax = plt.subplots()
    ax.plot(timestamp_ticks, differential_flux_list, label="Differential flux")
    ax.plot(timestamp_ticks, fit, label="Fit to diff. flux")
    ax.plot(timestamp_ticks, np.subtract(differential_flux_list, fit), label="Diff. flux minus fit")
    pp.pubplot("Diff. flux. vs image no.", "Timestamp", "Flux", "fluxfit" + camera.datastorage_filename_append + ".png", legend=True)


def pointing_requirements(camera: ccd, path_of_data_series: str, requirement_input: float, aperture_positions: list, aperture_radius: float, annulus_radii: list, num_of_images: int, exposure_time: float, cutoffs: list = None):
    print("\nAnalyzing the pointing requirements for detector: ", camera.name, "...")

    # Prepare the data set
    data_series     =   util.list_data(path_of_data_series)
    if cutoffs is not None:
        # These cutoffs are for only applying the analysis to a subset of the data
        first_cutoff    =   cutoffs[0]
        last_cutoff     =   cutoffs[1]
        # If a subset is defined, the number of images used will be less
        num_of_images   =   last_cutoff - first_cutoff
        data_series     =   data_series[first_cutoff:last_cutoff]

    # Perform differential aperture photometry, and gather data in a set of convenient arrays.
    diff_ap_phot_data               =   differential_aperture_photometry(camera, data_series, path_of_data_series, num_of_images, aperture_positions, aperture_radius, annulus_radii)
    differential_flux_list          =   diff_ap_phot_data[0]    # The list of computed differential fluxes (as a function of timestamps)
    aperture_data_list              =   diff_ap_phot_data[1]    # Aperture photometry data [actual flux in first aperture, x position of first aperture, y position of first aperture]
    timestamps                      =   diff_ap_phot_data[2]    # A list of timestamps
    total_noise                     =   diff_ap_phot_data[3]    # A float representing the computed total noise in the dataset

    # Correct the differential flux for drifts, so it is only noise, and we may compare with noise computed above
    drift_corrected_diffflux_data   =   drift_correct_data(differential_flux_list)
    # Convert differential flux to a percentage deviation from the first data point in the list
    diffflux_percentage_dev         =   np.multiply(np.divide(np.subtract(differential_flux_list, differential_flux_list[0]), differential_flux_list[0]), 100)


    print(" The relative photonic- (and readout-) noise is ", "{0:.10f}".format(total_noise))
    print(" The standard deviation of the drift corrected flux is ", "{0:.10f}".format(np.sqrt(2/3) * np.std(drift_corrected_diffflux_data) / np.mean(differential_flux_list) ))

    # Construct numpy array of data, and ready lists of timestamps and their labels for use in the figures
    timestamp_ticks         =   np.linspace(1, num_of_images, num_of_images)
    timestamp_ticks_labels  =   timestamps

    output_requirements_data    =   output_requirements(aperture_data_list, diffflux_percentage_dev, num_of_images, requirement_input)
    linear_model_x_function     =   output_requirements_data[0]
    linear_model_y_function     =   output_requirements_data[1]
    fit_query_pts_x             =   output_requirements_data[2]
    fit_query_pts_y             =   output_requirements_data[3]

    plot_variations(camera, aperture_data_list, differential_flux_list, diffflux_percentage_dev, fit_query_pts_x, fit_query_pts_y, linear_model_x_function, linear_model_y_function, timestamp_ticks, drift_corrected_diffflux_data)
    uncorrectable_variations_plot(camera, aperture_data_list, differential_flux_list, timestamp_ticks)
