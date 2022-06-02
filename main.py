"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           August    2021 ----- #
# ----- Last edit:       6th   January   2021 ----- #
# ------------------------------------------------- #
#####################################################
"""

import utilities as util
import numpy as np
import ccd
import plots
import mission_requirements as mreq


if __name__ == '__main__':
    # These bools can be changed in order to change the characterization procedure
    # For example  if "construct_master_bias" is set to "True", then the characterization
    # method will construct a new master bias frame from the data fed to the procedure.
    # If these are set to "False" data from previous runs are used instead.
    construct_master_bias_atik       =   False
    construct_master_dark_atik       =   False
    construct_master_flat_atik       =   False
    do_noise_estimation_atik         =   False
    do_gain_factor_estimation_atik   =   True
    do_old_time_calibration_atik     =   False
    do_linearity_estimation_atik     =   False
    produce_plots_atik               =   False

    construct_master_bias_AVT        =   False
    construct_master_dark_AVT        =   False
    construct_master_flat_AVT        =   False
    do_noise_estimation_AVT          =   False
    do_gain_factor_estimation_AVT    =   False
    do_old_time_calibration_AVT      =   False
    do_linearity_estimation_AVT      =   False
    produce_plots_AVT                =   False

    # These are the paths at which to save the constructed master frames and data sets
    # from the analysis procedures. If these methods are not used in characterization,
    # these paths are used as paths at which to collect the data
    analysis_data_path          =   "/home/marc/Dropbox/STEP_Speciale_Marc/data_from_characterization/"
    master_frame_path           =   "/home/marc/Documents/Master_frames/"

    # Define the path of the data and where to put the figures
    file_directory      =   "/home/marc/Documents/FITS_files/"
    figure_directory    =   "/home/marc/Dropbox/STEP_Speciale_Marc/figures/"

    print("Directory of data:    ", file_directory        )
    print("Directory of figures: ", figure_directory, "\n")

    # Initialize the camera in question
    atik_camera = ccd.CCD( name                         =   "Atik 414EX mono"  ,
                           gain_factor                  =   0.28               ,
                           analysis_data_path           =   analysis_data_path ,
                           master_frame_path            =   master_frame_path  ,
                           datastorage_filename_append  =   "_atikcam"         ,
                           figure_directory_path        =   figure_directory    )

    atik_camera.load_ccd_characterization_data(construct_master_bias       =   construct_master_bias_atik,
                                               construct_master_dark       =   construct_master_dark_atik,
                                               construct_master_flat       =   construct_master_flat_atik,
                                               do_noise_estimation         =   do_noise_estimation_atik,
                                               do_time_calibration         =   do_old_time_calibration_atik,
                                               do_linearity_estimation     =   do_linearity_estimation_atik,
                                               do_gain_factor_estimation   =   do_gain_factor_estimation_atik,
                                               path_of_master_bias_frame   =   "master_bias" + atik_camera.datastorage_filename_append + ".txt",
                                               path_of_master_dark_frame   =   "master_dark" + atik_camera.datastorage_filename_append + ".txt",
                                               path_of_master_flat_frame   =   "master_flat" + atik_camera.datastorage_filename_append + ".txt",
                                               path_of_linearity_data      =   "linearity" + atik_camera.datastorage_filename_append + ".txt",
                                               path_of_dark_current_data   =   "dark_current_versus_temperature" + atik_camera.datastorage_filename_append + ".txt",
                                               path_of_readout_noise_data  =   "readout_noise_versus_temperature" + atik_camera.datastorage_filename_append + ".txt")

    # Get the paths of the individual data sequences
    shutter_test                     =    util.complete_path(file_directory + "laser_000(2).fit"                                      , here=False)
    bias_sequence_AVT                =    util.complete_path(file_directory + "AVT_camera/Bias"                                       , here=False)
    bias_sequence_atik               =    util.complete_path(file_directory + "BIAS atik414ex 29-9-21 m10deg"                         , here=False)
    flat_sequence_AVT                =    util.complete_path(file_directory + "AVT_camera/Flats"                                      , here=False)
    flat_sequence_atik               =    util.complete_path(file_directory + "FLATS atik414ex 29-9-21 m10deg"                        , here=False)
    dark_current_sequence_atik       =    util.complete_path(file_directory + "temp seq noise atik414ex 27-9-21"                      , here=False)
    # dark_current_sequence       =    util.complete_path(file_directory + "preliminary dark current"                              , here=False)
    readout_noise_sequence_atik      =    util.complete_path(file_directory + "ron seq atik414ex 27-9-21"                             , here=False)
    # linearity_sequence          =    util.complete_path(file_directory + "total linearity with reference"                        , here=False)
    # linearity_sequence_20C      =    util.complete_path(file_directory + "Linearity at 20 degrees celcius atik414ex 29-9-21"     , here=False)
    linearity_sequence_AVT           =    util.complete_path(file_directory + "AVT_camera/linearity"                                  , here=False)
    linearity_sequence_atik          =    util.complete_path(file_directory + "linearity dimmed"                                      , here=False)
    time_calibration_sequence_atik   =    util.complete_path(file_directory + "time calibration 15-11-21"                             , here=False)
    new_timecal_sequence_AVT         =    util.complete_path(file_directory + "AVT_camera/timecal"                                    , here=False)
    new_timecal_sequence_atik        =    util.complete_path(file_directory + "new time calibration"                                  , here=False)
    hot_pixel_sequence_AVT           =    util.complete_path(file_directory + "AVT_camera/hotpix"                                     , here=False)
    hot_pixel_sequence_atik          =    util.complete_path(file_directory + "hotpix atik414ex 27-9-21"                              , here=False)
    zeropoint_sequence_atik          =    util.complete_path(file_directory + "zeropoint value"                                       , here=False)
    gaintemp_sequence_atik           =    util.complete_path(file_directory + "gain vs temp"                                          , here=False)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # exposures = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110])
    exposures_atik = np.array([2, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240])

    # Construct data sequence class instances for use with the CCD class
    bias_dataseq_atik               =    ccd.DataSequence(  path_of_data_series_input   =   bias_sequence_atik             ,
                                                            exposure_time_input         =   0.001                           )
    flat_dataseq_atik               =    ccd.DataSequence(  path_of_data_series_input   =   flat_sequence_atik             ,
                                                            exposure_time_input         =   10                              )
    readout_noise_dataseq_atik      =    ccd.DataSequence(  path_of_data_series_input   =   readout_noise_sequence_atik    ,
                                                            num_of_data_points_input    =   16                             ,
                                                            num_of_repeats_input        =   100                             )
    dark_current_dataseq_atik       =    ccd.DataSequence(  path_of_data_series_input   =   dark_current_sequence_atik     ,
                                                            num_of_data_points_input    =   16                             ,
                                                            num_of_repeats_input        =   100                            ,
                                                            exposure_time_input         =   10                              )
    linearity_dataseq_atik          =    ccd.DataSequence(  path_of_data_series_input   =   linearity_sequence_atik        ,
                                                            num_of_data_points_input    =   27                             ,
                                                            num_of_repeats_input        =   10                             ,
                                                            exposure_time_input         =   10                             ,
                                                            exposure_list_input         =   exposures_atik                 ,
                                                            milliseconds_input          =   False                           )
    old_time_calibration_dataseq_atik =  ccd.DataSequence(  path_of_data_series_input   =   time_calibration_sequence_atik,
                                                            num_of_data_points_input    =   20,
                                                            num_of_repeats_input        =   10)
    new_timecal_dataseq_atik        =    ccd.DataSequence(  path_of_data_series_input   =   new_timecal_sequence_atik      ,
                                                            num_of_repeats_input        =   10                             ,
                                                            exposure_list_input         =   np.array([1, 2])                )
    hot_pixel_dataseq_atik          =    ccd.DataSequence(  path_of_data_series_input   =   hot_pixel_sequence_atik        ,
                                                            num_of_repeats_input        =   2                              ,
                                                            exposure_time_input         =   [90, 1000]                     ,
                                                            cutoff_input                =   7.5                             )
    zeropoint_dataseq_atik          =    ccd.DataSequence(  path_of_data_series_input   =   zeropoint_sequence_atik        ,
                                                            num_of_data_points_input    =   8                              ,
                                                            num_of_repeats_input        =   100                             )
    gaintemp_dataseq_atik           =    ccd.DataSequence(  path_of_data_series_input   =   gaintemp_sequence_atik        ,
                                                            num_of_data_points_input    =   16                              ,
                                                            num_of_repeats_input        =   20                              )

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    characterization_atik = atik_camera.characterize(bias_data_sequence               =   bias_dataseq_atik,
                                                     flat_data_sequence               =   flat_dataseq_atik,
                                                     dark_current_data_sequence       =   dark_current_dataseq_atik,
                                                     readout_noise_data_sequence      =   readout_noise_dataseq_atik,
                                                     linearity_data_sequence          =   linearity_dataseq_atik,
                                                     hot_pixel_data_sequence          =   hot_pixel_dataseq_atik,
                                                     zero_point_data_sequence         =   zeropoint_dataseq_atik,
                                                     old_timecal_data_sequence        =   old_time_calibration_dataseq_atik,
                                                     time_calibration_data_sequence   =   new_timecal_dataseq_atik,
                                                     gain_data_sequence               =   gaintemp_dataseq_atik             )

    dark_current_data_atik       =   characterization_atik[0]
    readout_noise_data_atik      =   characterization_atik[1]
    time_calibration_atik        =   characterization_atik[2]
    linearity_data_atik          =   characterization_atik[3]
    gain_data_atik               =   characterization_atik[4]
    """
    ideal_linear_relation_atik   =   characterization_atik[3]
    linearity_deviations_atik    =   characterization_atik[4]
    linearity_dev_err_atik       =   characterization_atik[5]
    stabillity_data_atik         =   characterization_atik[6]
    ron_dists_vs_temp_atik       =   characterization_atik[7]
    """
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    if produce_plots_atik:
        plots.produce_plots(atik_camera, figure_directory, analysis_data_path, linearity_data_atik, dark_current_data_atik, readout_noise_data_atik, gain_data_atik, time_calibration_atik, hot_pixels=True, shutter_test=shutter_test, lightsource_stabillity=True, bias_sequence=bias_sequence_atik)


    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    AVT_camera  = ccd.CCD( name                         =   "AVT GC660M"       ,
                           gain_factor                  =   5.3                ,
                           analysis_data_path           =   analysis_data_path ,
                           master_frame_path            =   master_frame_path  ,
                           datastorage_filename_append  =   "_AVT"             ,
                           figure_directory_path        =   figure_directory    )

    AVT_camera.load_ccd_characterization_data(  construct_master_bias      =   construct_master_bias_AVT,
                                                construct_master_dark      =   construct_master_dark_AVT,
                                                construct_master_flat      =   construct_master_flat_AVT,
                                                do_noise_estimation        =   do_noise_estimation_AVT,
                                                do_time_calibration        =   do_old_time_calibration_AVT,
                                                do_linearity_estimation    =   do_linearity_estimation_AVT,
                                                do_gain_factor_estimation  =   do_gain_factor_estimation_AVT,
                                                path_of_master_bias_frame  =   "master_bias" + AVT_camera.datastorage_filename_append + ".txt",
                                                path_of_master_dark_frame  =   "master_dark" + AVT_camera.datastorage_filename_append + ".txt",
                                                path_of_master_flat_frame  =   "master_flat" + AVT_camera.datastorage_filename_append + ".txt",
                                                path_of_linearity_data     =   "linearity" + AVT_camera.datastorage_filename_append + ".txt")

    # Define the exposure series used for the AVT camera
    exposures_AVT = []
    for number in range(0, 4):
        for decimal in range(0, 10):
            if number == 0 and decimal == 0:
                continue
            exposures_AVT.append(float(str(number) + "." + str(decimal)))
    exposures_AVT = np.asarray(exposures_AVT)

    # Construct data sequence class instances for use with the CCD class
    bias_dataseq_AVT            = ccd.DataSequence( path_of_data_series_input   =   bias_sequence_AVT           ,
                                                    exposure_time_input         =   0.001                        )
    flat_dataseq_AVT            = ccd.DataSequence( path_of_data_series_input   =   flat_sequence_AVT           ,
                                                    exposure_time_input         =   1                            )
    readout_noise_dataseq_AVT   = ccd.DataSequence( path_of_data_series_input   =   readout_noise_sequence_atik ,
                                                    num_of_data_points_input    =   16                          ,
                                                    num_of_repeats_input        =   100                          )
    dark_current_dataseq_AVT    = ccd.DataSequence( path_of_data_series_input   =   dark_current_sequence_atik  ,
                                                    num_of_data_points_input    =   16                          ,
                                                    num_of_repeats_input        =   100                         ,
                                                    exposure_time_input         =   10                           )
    linearity_dataseq_AVT       = ccd.DataSequence( path_of_data_series_input   =   linearity_sequence_AVT      ,
                                                    num_of_data_points_input    =   39                          ,
                                                    num_of_repeats_input        =   10                          ,
                                                    exposure_time_input         =   1                           ,
                                                    exposure_list_input         =   exposures_AVT               ,
                                                    milliseconds_input          =   True                         )
    new_timecal_dataseq_AVT     = ccd.DataSequence( path_of_data_series_input   =   new_timecal_sequence_AVT    ,
                                                    num_of_repeats_input        =   10                          ,
                                                    exposure_list_input         =   np.array([0.1, 0.2])         )
    hot_pixel_dataseq_AVT       = ccd.DataSequence( path_of_data_series_input   =   hot_pixel_sequence_AVT      ,
                                                    num_of_repeats_input        =   2                           ,
                                                    exposure_time_input         =   [5, 50]                     ,
                                                    cutoff_input                =   49.5                         )
    zeropoint_dataseq_AVT       = ccd.DataSequence( path_of_data_series_input   =   zeropoint_sequence_atik     ,
                                                    num_of_data_points_input    =   8                           ,
                                                    num_of_repeats_input        =   100                          )

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    characterization_AVT = AVT_camera.characterize( bias_data_sequence              =   bias_dataseq_AVT            ,
                                                    flat_data_sequence              =   flat_dataseq_AVT            ,
                                                    dark_current_data_sequence      =   dark_current_dataseq_AVT    ,
                                                    readout_noise_data_sequence     =   readout_noise_dataseq_AVT   ,
                                                    linearity_data_sequence         =   linearity_dataseq_AVT       ,
                                                    hot_pixel_data_sequence         =   hot_pixel_dataseq_AVT       ,
                                                    zero_point_data_sequence        =   zeropoint_dataseq_AVT       ,
                                                    time_calibration_data_sequence  =   new_timecal_dataseq_AVT     ,
                                                    gain_data_sequence=gaintemp_dataseq_atik)

    dark_current_data_AVT   =   characterization_AVT[0]
    readout_noise_data_AVT  =   characterization_AVT[1]
    time_calibration_AVT    =   characterization_AVT[2]
    linearity_data_AVT      =   characterization_AVT[3]
    """
    ideal_linear_relation_AVT   =   characterization_AVT[3]
    linearity_deviations_AVT    =   characterization_AVT[4]
    linearity_dev_err_AVT       =   characterization_AVT[5]
    stabillity_data_AVT         =   characterization_AVT[6]
    ron_dists_vs_temp_AVT       =   characterization_AVT[7]
    """
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #

    if produce_plots_AVT:
        plots.produce_plots(AVT_camera, figure_directory, analysis_data_path, linearity_data_AVT, hot_pixels=True)

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #
    print("\n\n---\nMission requirements:\n---")
    requirement_input = 0.01  # Allowable flux deviaion in percent

    print(" The input mission requirement is an allowable flux change of ", requirement_input, "%")
    print("  (", requirement_input / 100, " absolute flux change )")

    difftest_ATIK = util.complete_path(file_directory + "difftest tidsvar lang ATIK", here=False)
    mreq.pointing_requirements(atik_camera, path_of_data_series=difftest_ATIK,
                               requirement_input = requirement_input,
                               aperture_positions=[(643., 676.), (434., 503.)], aperture_radius=65.,
                               annulus_radii=[75., 150.], num_of_images=500, exposure_time=40, cutoffs=[60, 499])
    print("Linearity errors for detector :", atik_camera.name, "\n ", atik_camera.linearity[:, 2])

    difftest_AVT = util.complete_path(file_directory + "AVT_camera/difftest_tid_2", here=False)
    mreq.pointing_requirements(AVT_camera, path_of_data_series=difftest_AVT,
                               requirement_input = requirement_input,
                               aperture_positions=[(359., 229.), (204., 104.)], aperture_radius=65., annulus_radii=[70., 100.], num_of_images=4000 - 1751, cutoffs=[1750, 3999], exposure_time=5)
    print("Linearity errors for detector :", AVT_camera.name, "\n ", AVT_camera.linearity[:, 2])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------- #


    # exit()
    import pubplot as pp
    filepath = util.get_path(difftest_ATIK + "differentialtest_033.fit")
    hdul, header, imagedata = util.fits_handler(filepath)
    pp.plot_image(imagedata, "test", "x", "y", "test ", "test.png", "test", raisetitle=False)
    pp.plot_image(imagedata, "Pointing test Atik 414EX", "x", "y", "Atik 414EX", "pointingtest_atik.png", "test",
                  raisetitle=False)

    filepath = util.get_path(difftest_AVT + "difftest_time_0013.fit")
    hdul, header, imagedata = util.fits_handler(filepath)
    pp.plot_image(imagedata, "test", "x", "y", "test ", "test.png", "test", raisetitle=False)
    pp.plot_image(imagedata, "Pointing test AVT GC660M", "x", "y", "AVT GC660M", "pointingtest_avt.png", "test",
                  raisetitle=False)
