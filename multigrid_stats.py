import argparse as ap
from casacore import images
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

def get_args():
    """Generates appropriate variables from input arguments.
    
    Returns
    -------
    argparse.Namespace
        The parsed arguments"""
        
    parser = ap.ArgumentParser()
    parser.add_argument("grid_location",
                        help="The location of the grids to plot to generate the statistics plots from.")
    parser.add_argument("plot_config_yaml",
                        help="A yaml file that describes the relations between the files and how they need to be plotted. See example in config/plot-config.yaml")
    parser.add_argument("-o",
                        "--out_path",
                        default="stats.png",
                        help="The output path of the stats plot.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.grid_location):
        raise FileNotFoundError("The specified grid location does not exist")
    if not os.path.exists(args.plot_config_yaml):
        raise FileExistsError("Could not find the config file at the specified location")
    if os.path.isdir(args.out_path):
        args.out_path = os.path.join(args.out_path, "stats.png")
    
    return args

def read_yaml(filepath: str) -> dict:
    """Reads the yaml file to a dictionary to be used for plotting.
    
    Parameters
    ----------
    filepath: str
        The path to the yaml file
        
    Returns
    -------
    dict
        A dictionary detailing the plotting config"""
    
    with open(filepath, 'r') as yamlfile:
        try:
            plot_config = yaml.safe_load(yamlfile, Loader=yaml.Loader)
        except yaml.YAMLError as err:
            print(f"There was an error reading the yaml file see below: ", err)
    return plot_config

def make_plots(plot_config: dict, grid_location: str, out_path: str) -> None:
    """Creates the statistics plots for abs(vis/pcf.real) and abs(vis/sm).
    
    Parameters
    ----------
    plot_config: dict
        A dictionary of the form {'plots': [{'labels': ..., 'cubes':{'visgrid':..., 'pcfgrid': ...}}]}
    grid_location: str
        The path to the input grids
    out_path: str
        The path to the desired output plot location"""
        
    fig = plt.figure(figsize=(20,10))
    ax = fig.subplots(1,2)
    for plot in plot_config["plots"]:
        label = plot["label"]
        vis, pcf, sm = read_grids(plot["cubes"], grid_location)
        vphist = create_hist(np.abs(vis/pcf.real))
        plot_histogram(ax[0], vphist, "abs(vis/pcf.real)", label)
        vshist = create_hist(np.abs(vis/sm))
        plot_histogram(ax[1], vshist, "abs(vis/sm)", label=label)
    fig.legend()
    fig.savefig(out_path)
        
        
        
def plot_histogram(ax: plt.Axes, data: np.ndarray, title: str, label: str) -> None:
    """Plots the histogram provided
    
    Parameters
    ----------
    ax: plt.Axes
        An axes object to which to plot the histogram
    data: np.ndarray
        Data of the form (xvals, hist)
    title: str
        The title of the plot
    label: str
        The label to attach to the plot data"""
        
    ax.plot(data[0], data[1], '.', markeredgecolor='k', label=label)
    ax.step(data[0], data[1], where='mid', alpha=0.2)
    ax.set(yscale='log', title=title)

def create_hist(data: np.ndarray, nbins: int=500) -> tuple:
    """Creates histogram data given an input array
    
    Parameters
    ----------
    data: np.ndarray
        The data to create the histogram from
    nbins: int
        The number of bins to use
    
    Returns
    -------
    tuple
        The output is of the form (xvals, hist)"""
        
    hist, bins = np.histogram(data[np.isfinite(data)], nbins)
    xvals = (bins[1:] + bins[:-1])/2
    return xvals, hist
        
def read_grids(cubes: dict, grid_location: str) -> tuple:
    """Reads in the grids as casa images and converts them to ndarrays
    
    Parameters
    ----------
    cubes: dict
        A dict of the form {'visgrid':..., 'pcfgrid':...}
    grid_location: str
        The place to look for the grids specified in cubes
        
    Returns
    -------
    tuple
        a tuple of np.ndarrays"""
        
    with images.image(os.path.join(grid_location, cubes["visgrid"])) as image:
        visgrid = image.getdata()
        visgrid[visgrid == 0] = np.nan
    with images.image(os.path.join(grid_location, cubes["pcfgrid"])) as image:
        pcfgrid = image.getdata()
        pcfgrid[pcfgrid == 0] = np.nan
    smgrid = smear_pcf(pcfgrid)
    return visgrid, pcfgrid, smgrid
    
def smear_pcf(pcf: np.ndarray) -> np.ndarray:
    """Smears the pcf to spread the specified weights to have equal grid occupancy to the visgrid
    
    Parameters
    ----------
    pcf: np.ndarray
        The input PCF with zero values set to NaN
    
    Returns
    -------
    np.ndarray
        The resultant smeared grid with zero values set to NaN"""
        
    s = np.zeros(pcf.shape, dtype=np.float32)
    ksize = np.abs(pcf.imag/pcf.real).astype(int)
    pcfinite = np.isfinite(pcf)
    for f in range(pcf.shape[0]):
        for x in range(pcf.shape[1]):
            for y in range(pcf.shape[2]):
                if pcfinite[f,x,y]:
                    s[f,x-ksize[f,x,y]:x+ksize[f,x,y]+1, y-ksize[f,x,y]:y+ksize[f,x,y]+1] += pcf[f,x,y].real
    s[s == 0] = np.nan
    return s

        
    
if __name__ == '__main__':
    args = get_args()
    plot_config = read_yaml(args.plot_config_yaml)
    make_plots(plot_config, args.grid_location, args.out_path)