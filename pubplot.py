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


def plot_image(image: np.ndarray , figure_name: str = None, scale: float = None):
    """
    A helper function to pubplot() in case we are plotting 2D images
    in stead of ordinary functions etc.

    :parameter np.ndarray image:
        - The image which is to be plotted as a figure.
    :parameter str figure_name:
        - A figure name.
    :parameter float scale:
        - A cutoff above which data will not be included in the plot.
    """
    plt.figure(figure_name, figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.colorbar()


# For publication worthy plotting ----------------------------------------------
def pubplot(title           : str           ,
            xlabel          : str           ,
            ylabel          : str           ,
            filename        : str           ,
            show            : bool  = False ,
            xlim            : list  = None  ,
            ylim            : list  = None  ,
            legend          : bool  = True  ,
            legendlocation  : str   = None  ,
            grid            : bool  = True   ):
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
    plt.style.use(['science', 'ieee', 'vibrant'])
    font = {'family': 'serif',
            'serif': 'helvet',
            'weight': 'bold',
            'size': 10}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    plt.title(title, {'fontsize': 12, 'fontweight': 'black'})
    plt.xlabel(xlabel, **kwargs)
    plt.ylabel(ylabel, **kwargs)

    if legend:
        plt.legend(loc=legendlocation, fancybox=False, framealpha=1, edgecolor='inherit')
    if grid:
        plt.grid(b=True, which='major', axis='both', alpha=0.3)  # Include a grid!
    if show:
        plt.show()
    else:
        pass

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.savefig(filename, dpi=200)
    plt.close()
    return
