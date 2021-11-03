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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines

kwargs = {'fontweight': 'bold'}


def plot_image( image           : np.ndarray    ,
                title           : str           ,
                xlabel          : str           ,
                ylabel          : str           ,
                input_label     : str           ,
                filename        : str           ,
                figure_name     : str   = None  ,
                scale           : float = None  ,
                show            : bool  = False ,
                raisetitle      : bool  = False ):

    """
    A helper function to pubplot() in case we are plotting 2D images
    in stead of ordinary functions etc.

    :parameter np.ndarray image:
        - The image which is to be plotted as a figure.
    :parameter str title:
        - The title of the figure
    :parameter str xlabel:
        - The label on the first axis
    :parameter str ylabel:
        - The label on the second axis
    :parameter str input_label:
        - The camera name used as a label in the plot
    :parameter str filename:
        - The filename to be printed to
    :parameter str figure_name:
        - A figure name.
    :parameter float scale:
        - A cutoff above which data will not be included in the plot.
    :parameter bool show:
        - Toggle showing of the plot during runtime
    :param bool raisetitle:
        - bool that toggles raising of the title in the figure
    """

    plt.figure(figure_name)
    ax = plt.gca()
    im = ax.imshow(image, cmap='gray')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.rcParams['image.cmap'] = "winter"
    plt.colorbar(im, cax=cax)

    plt.style.use(['science', 'ieee', 'vibrant'])
    font = {'family': 'serif',
            'serif': 'helvet',
            'weight': 'bold',
            'size': 10}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    if raisetitle:
        ax.set_title(title, {'fontsize': 12, 'fontweight': 'black'},  y=1.06)
    else:
        ax.set_title(title, {'fontsize': 12, 'fontweight': 'black'})

    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)

    white_line = mlines.Line2D([], [], color='white', label=input_label)
    plt.legend(handles=[white_line], bbox_to_anchor=(1, 1), loc="upper left", fancybox=False, framealpha=1, edgecolor='inherit')

    if show:
        plt.show()

    plt.savefig(filename, dpi=200)
    plt.close()


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
