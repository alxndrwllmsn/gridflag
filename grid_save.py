from casacore import images
import numpy as np
import os
import argparse as ap

def get_args() -> ap.Namespace:
    """parses the argparse arguments
    
    Returns
    -------
    args: argparse.Namespace
        The argparse arguments
    """
    argparser = ap.ArgumentParser()
    argparser.add_argument("visibility_grid", 
                           help= "The path to the visibility grid (use -d to specify working directory).")
    argparser.add_argument("psf_grid", 
                           help= "The path to the psf grid (use -d to specify working directory).")
    argparser.add_argument("pcf_grid", 
                           help= "The path to the pcf grid (use -d to specify working directory).")
    argparser.add_argument("-d", "--directory", 
                           help="The path to the working directory, all other specified paths will be relative to this one.")
    args = argparser.parse_args()
    return args

def write_npy_grids(gridpath: str, npy_path: str):
    """writes a casa.image grid into a .npy file.
    
    Parameters
    ----------
    gridpath: str
        The path to the casa.image grid
    npy_path: str
        The path to the desired .npy file
    """
    grid = images.image(gridpath).getdata()
    np.save(npy_path, grid)

if __name__ == '__main__':
    args = get_args()
    if args.directory is None:
        vispath = args.visibility_grid
        psfpath = args.psf_grid
        pcfpath = args.pcf_grid
    else:
        vispath = os.path.join(args.directory, args.visibility_grid)
        psfpath = os.path.join(args.directory, args.psf_grid)
        pcfpath = os.path.join(args.directory, args.pcf_grid)
    
    for path, outpath in ((vispath, "vis.npy"), (psfpath, "psf.npy"), (pcfpath, "pcf.npy")):
        write_npy_grids(path, outpath)