"""
#####################################################
# ------------------------------------------------- #
# -----       By Marc Breiner Sørensen        ----- #
# ------------------------------------------------- #
# ----- Implemented:           October   2019 ----- #
# ----- Last edit:         23. September 2021 ----- #
# ------------------------------------------------- #
#####################################################
"""
import matplotlib.pyplot as plt
import numpy as np

kwargs = {'fontweight': 'bold'}


def plot_image(image: np.ndarray , figure_name: str = None):
    """
    A helper function to pubplot() in case we are plotting 2D images
    in stead of ordinary functions etc.

    :parameter np.ndarray image:
        - The image which is to be plotted as a figure.
    :parameter str figure_name:
        - A figure name
    """
    plt.figure(figure_name)
    plt.imshow(image, vmin=0, cmap='gray')
    plt.colorbar()


# For publication worthy plotting ----------------------------------------------
def pubplot(title           : str           ,
            xlabel          : str           ,
            ylabel          : str           ,
            filename        : str           ,
            show            : bool  = False ,
            xlim            : tuple = None  ,
            ylim            : tuple = None  ,
            legend          : bool  = True  ,
            legendlocation  : str   = None  ,
            grid            : bool  = True      ):
    """

    :parameter str title:
        - The title of the figure
    :parameter str xlabel:
        - The label on the first axis
    :parameter str ylabel:
        - The label on the second axis
    :parameter str filename:
        - The filename to be printed to
    :parameter bool show:
        - Toggle showing of the plot during runtime
    :parameter tuple xlim:
        - A tuple of form [a, b] which is the interval on the first axis
          within which data is to be plotted
    :parameter tuple ylim:
        - A tuple of form [a, b] which is the interval on the second axis
          within which data is to be plotted
    :parameter bool legend:
        - Toggle showing of the legend
    :parameter str legendlocation:
        - A string with the location specification for the legend
    :parameter bool grid:
        - Toggle figure grid
    """

    # Adjust a few plot parameters
    # plt.rc('font', family='sans-serif')
    plt.rc('font', family='serif')

    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    # plt.rc('font', serif='Helvetica')
    plt.rcParams.update({'font.size': 12})
    # plt.tick_params(axis='x', which='minor', bottom=False)

    # plt.axes().yaxis.set_minor_locator(MultipleLocator(5))
    # plt.axes().xaxis.set_minor_locator(MultipleLocator(5))

    # kwargs = {'fontweight': 'bold'}  # Bold font on axes

    plt.title(title, **kwargs)
    plt.xlabel(xlabel, **kwargs)
    plt.ylabel(ylabel, **kwargs)
    if legend:
        plt.legend(loc=legendlocation, fancybox=False, framealpha=1, edgecolor='inherit')
    if grid:
        plt.grid(b=True, which='major', axis='both', alpha=0.3)  # Include a grid!
    plt.ticklabel_format(axis='both', style='sci', scilimits=(5, 6), useOffset=True)  # set scientific notation.
    if show:
        plt.show()
    else:
        pass

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.savefig(filename)
    plt.close()
    return
