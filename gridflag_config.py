# Simple script to prepare DINGO gridded visibilities for flagging.

# Some notes:

#     - The casaimage indexing is: [channel, pol, ra, dec]
#     - the last two axis is saved as a 'linear' object and so not as 'direction'

import sys, os
import logging
import numpy as np
import copy

import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

from casacore import images as casaimage

import argparse as ap

#=== Set up logging
logger = logging.getLogger()

logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
#handler = logging.NullHandler()

formatter = logging.Formatter('%(asctime)s -- %(levelname)s: %(message)s')

handler.setFormatter(formatter)

logger.addHandler(handler)

#=== Functions ===
#*******************************************************************************
#=== Some helper functions to make life easier
def echo_for_loop_counter(start: str,end: str,count: int,loop_string: str='Loop state'):
    """Simple routine to print the status of a for loop to the stdout and update
    it dynamically.
    
    It is useful for testing and running kinda long code on a screen session
    The code should be called within the loop, but it only works with for loops
    as one needs to know the start and end values of the loop.
    NOTE if it is logged every loop cycle will print a line to the logfile (!)
    NOTE this code is not compatible with the logging module
    
    NOTE no other print statements should be in the for loop (!)
    
    Parameters
    ----------
    start: int
        The start value of the loop
    end: int
        The end value of the loop
    count: int
        The current vale of the loop
    loop string: str
        A string to name the loop. The counter is printed in the following format:
        `loop string`: [===.......] x% y/zzzz
        
    Returns
    -------
        Prints the loop status to stderr
    """

    #=== Compute percentage and object counter ===
    #Cause int() always round downwards == np.floor()
    loop_range = int(end - start)
    floor_percentage = int(100 * (count - start) / loop_range)

    #=== Build percentage bar string ===
    floor_10_percentage_bar = int(floor_percentage * 0.1)

    bar = ''.join(['=' for bar in range(floor_10_percentage_bar)])
    bar_left = ''.join(['.' for bar in range(10 - floor_10_percentage_bar)])
    floor_percentage_bar = bar + bar_left

    #=== Display ===
    print('{4:s}: [{0:s}] {1:d}% {2:d}/{3:d}'.format(
        floor_percentage_bar,floor_percentage,count - start,loop_range,
        loop_string),
        end='\r')

    if count == end:
        print('{4:s}: [{0:s}] {1:d}% {2:d}/{3:d}'.format(
        floor_percentage_bar,floor_percentage,count - start,loop_range,
        loop_string))


#=== Minimum functions needed to interact with the grids as casaimages (read, write etc.)
def create_CIM_object(cimpath: str) -> casaimage.image:
    """This is a solution I use to interact with MS as well. Should solve avoiding
    opening images multiple times.

    NOTE: no close function exist, the read in images need to be deleted from
        memory!

    Parameters
    ----------
    cimpath: str
        The full path to the casa image to be opened

    Returns
    -------
    CIM:  ``casacore.images.image.image`` object
        The in-memory CASAImage

    """
    # create an empty image in-memory to check the object type: the working
    # solution
    CIM_type_tmp = casaimage.image(imagename='',shape=np.ones(1))

    # If already an image
    if isinstance(cimpath, type(CIM_type_tmp)):
        return cimpath

    else:
        # We know it is a string now
        logger.debug('Open image: {0:s}'.format(str(cimpath))) #We know it is a string in this case
        CIM = casaimage.image(cimpath)

        return CIM

def check_input_grids(visgrid: casaimage.image, psfgrid: casaimage.image, pcfgrid: casaimage.image):
    """Simple rouitine to perform a threeway check on the input grids.

    The code check for:
        - equity of dimensions
        - equity of image units
        - equity of spectral frame AND increment AND unit
        - equity of stokes frame AND increment AND unit
        - equity of linear increments AND units

    NOTE: the code does not check if any of the axis are covering the same physical
    range! Ergo, two grids with same xHz channel resolutions and 100 channels, but
    covering different bandwidth will pass this test!

    TO DO: fix this issue

    This is not important for now, and would just slow the code a bit...

    NOTE: code only works with StokesI images

    TO DO: implement stokes check

    NOTE: the image axis can be 'direction' instead of 'linear' the code continues
    in this case but raises an error

    Parameters
    ----------
    visgrid: ``casacore.images.image.image`` object
        The visibility grid

    psfgrid: ``casacore.images.image.image`` object
        The psf grid

    pcfgrid: ``casacore.images.image.image`` object
        The pcf grid

    Returns
    -------
    The code halts if the three grids have different frame

    """

    vis_CIM = create_CIM_object(visgrid)
    psf_CIM = create_CIM_object(psfgrid)
    pcf_CIM = create_CIM_object(pcfgrid)   

    # Check for shape
    if (vis_CIM.shape() != psf_CIM.shape()) or (vis_CIM.shape() != pcf_CIM.shape()):
        raise ValueError('Input grids have different shape!')

    
    # Check if the cubes are 4 dimensional (we know they are the same shape)
    if vis_CIM.ndim() != 4:
        raise NotImplementedError('Code only works with casaimages of 4 axis!')

    # Read in the coordinate info for the images
    vis_coords = vis_CIM.coordinates()
    psf_coords = psf_CIM.coordinates()
    pcf_coords = pcf_CIM.coordinates()

    #=== Define some check functions to have a more compact code x+y===============
    def check_increment(visGC, psfGC, pcfGC, axis):
        # Expected inputs are float, list and numpy array so I need to add any
        if np.any(visGC[axis].get_increment()) != np.any(psfGC[axis].get_increment()) \
        or np.any(visGC[axis].get_increment()) != np.any(pcfGC[axis].get_increment()):
            raise ValueError('Input grids have different {0:s} resolution!'.format(axis))
        return 0

    def check_unit(visGC, psfGC, pcfGC, axis):
        # Some reason adding np.any() breaks this code, but it works without it just fine
        if visGC[axis].get_unit() != psfGC[axis].get_unit() \
        or visGC[axis].get_unit() != pcfGC[axis].get_unit():
            raise ValueError('Input grids have different {0:s} unit!'.format(axis))
        return 0
    #===========================================================================

    #=== Spectral axis
    coords_axis = 'spectral'

    if vis_coords[coords_axis].get_frame() != psf_coords[coords_axis].get_frame() \
    or vis_coords[coords_axis].get_frame() != pcf_coords[coords_axis].get_frame():
        raise ValueError('Input grids have different spectral frame!')

    check_unit(vis_coords, psf_coords, pcf_coords, coords_axis)
    check_increment(vis_coords, psf_coords, pcf_coords, coords_axis)

    #=== Stokes axis
    coords_axis = 'stokes'

    if vis_coords[coords_axis].get_stokes() != psf_coords[coords_axis].get_stokes() \
    or vis_coords[coords_axis].get_stokes() != pcf_coords[coords_axis].get_stokes():
        raise ValueError('Input grids have different spectral resolution!')

    if vis_coords[coords_axis].get_stokes() != ['I']:
        raise NotImplementedError('Code only works with StokesI data!')

    check_unit(vis_coords, psf_coords, pcf_coords, coords_axis)
    check_increment(vis_coords, psf_coords, pcf_coords, coords_axis)

    #=== Linear axis
    coords_axis = 'linear'
    
    try:
        image_ax = vis_coords[coords_axis]
    except:
        coords_axis = 'direction'
        image_ax = vis_coords[coords_axis]
        logger.warning("Image axis is 'direction' not 'linear', not all checks can be made!")

    check_unit(vis_coords, psf_coords, pcf_coords, coords_axis)
    check_increment(vis_coords, psf_coords, pcf_coords, coords_axis)

    return 0


def save_CIM_object_from_data_and_template(cim_data, new_cimpath, template_cimpath, overwrite=True):
    """Creates a casaimage based in an image template and a data matrix.

    The template *should* be the visibility grid

    """
    if os.path.isdir(new_cimpath) and overwrite == False: 
        raise TypeError('Output casa image already exist, and the overwrite parameters is set to False!')

    template_CIM = create_CIM_object(template_cimpath)

    if np.shape(cim_data) != tuple(template_CIM.shape()):
        raise ValueError('Input data has different shape than the template casa image!')

    logger.debug('Create new image: {0:s}'.format(new_cimpath))

    coordsys = template_CIM.coordinates()

    # Create the output image
    output_cim = casaimage.image(new_cimpath,
                    coordsys=coordsys,
                    values=cim_data,
                    overwrite=overwrite)

    del output_cim

    return 0

def adaptive_convolutional_smearing(initial_pcf_grid_array: np.ndarray,
                                    reference_grid_array: np.ndarray,
                                    echo_counter: bool=False,
                                    anti_aliasing_kernel_size: int=7):
    """The preparation, running and checking of the adaptive convolutional smearing used
    to create the Wiener-filters for weighting, and basically to estimate the
    smeared weight-density distribution.

    The input arrays should follow the casaimage format and have to be 4D:

    [channel, pol, ra, dec]

    NOTE: the code works only on StokesI polarisation

    The RE(initial_pcf_grid_array) contains the cell-averaged SNR weight
    The IM(initial_pcf_grid_array) contains the cell- and SNR-weight- averaged
    kernel sizes
    
    The function returns the PCF grid.

    NOTE: the imaginary part should be taken to the absolute value as half the
        uv-plane is set to be negative to retain the hermitian property of the images
    """
    # Create the output array
    smeared_grid = np.zeros(np.shape(initial_pcf_grid_array))

    # Generate a matrix with the kernel sizes
    # pcf_kernel_sizes = np.fabs(np.divide(pcf_imag,pcf_real, where=pcf_real!=0))

    # Get the maximum projection kernel (per channel)
    # Array containing the max kernels per channel rounded to % precision to get rid
    # of numerical errors, then a ceil() is called (i.e 1.1 => 2 ; but 1.00001 => 1)
    # C_max_array = np.ceil(np.round(np.amax(pcf_kernel_sizes, axis=(1,2)),2))
    # C_max_array = np.ceil(np.amax(pcf_kernel_sizes, axis=(1,2)))
    
    
        # Apply correction for the kernel sizes (to whole grid)
    # pcf_kernel_sizes[pcf_kernel_sizes != 0] += anti_aliasing_kernel_size

    pcf_kernel_sizes = np.fabs(np.divide(initial_pcf_grid_array.imag, initial_pcf_grid_array.real, where=initial_pcf_grid_array.real != 0))
    pcf_kernel_sizes = np.ceil(pcf_kernel_sizes)
    
    
    # Apply correction, but only for the grid cells with < anti_aliasing_kernel_size
    kernel_size_test = np.bitwise_and((pcf_kernel_sizes != 0),(pcf_kernel_sizes < anti_aliasing_kernel_size))
    pcf_kernel_sizes[kernel_size_test] += anti_aliasing_kernel_size
    
    # Get the max kernel size for the smearing
    boxWidth = np.ceil(np.amax(pcf_kernel_sizes,axis=(2,3)))
    

    # Perform the operations by channel
    for i in range(0,np.shape(initial_pcf_grid_array)[0]):
        
        # do the smearing
        if args.fast:
            smeared_grid[i,0,...] = smearing_fast(initial_pcf_grid_array[i, 0],
                                                  pcf_kernel_sizes[i,0],
                                                  boxWidth[i])
        else:
            smeared_grid[i,0,...] = smearing_slow(initial_pcf_grid_array[i,0],
                                                  pcf_kernel_sizes[i,0], 
                                                  boxWidth[i],
                                                  echo_counter)
        
        # Now check if the resultant grid occupancy is the same as the example grid

        # Plot smeared grid
        im = plt.matshow(smeared_grid[i,0,...], cmap='gray_r')
        plt.colorbar(im)
        plt.show()

        reference_grid_occupancy = np.where(np.abs(reference_grid_array[i,0,...]) > 0.0, 1, 0)
        smeared_pcf_grid_occupancy = copy.deepcopy(np.where(smeared_grid[i,0,...] > 0.0, 2, 0))

        # I used to do this for testing, but should be better to return the smeared grid maybe...
        diff_grid = np.subtract(reference_grid_occupancy,smeared_pcf_grid_occupancy)

        im = plt.matshow(diff_grid, cmap='Set3')
        plt.colorbar(im)
        plt.show()        


def smearing_slow(pcf: np.ndarray, pcf_kernel_sizes: np.ndarray, boxWidth: np.ndarray, echo_counter: bool) -> np.ndarray:
    """The core algorithm (non-vectorised) performing the the adaptive convolutional smearing used
    to create the Wiener-filters for weighting, and basically to estimate the
    smeared weight-density distribution.
    
    Parameters
    ----------
    pcf: np.ndarray
        The input pcf array for a single channel
    pcf_kernel_sizes: np.ndarray
        The corrected kernel sizes for the pcf
    boxWidth: np.ndarray
        A single valued array with the current channel's box width
    echo_counter: bool
        A switch to enable the progress counter
        
    Returns
    -------
    smeared_grid: np.ndarray
        The smeared grid for a single channel
    """
    smeared_grid = np.zeros(pcf.shape)
    
    # The maximum kenel width in the given channel
    # boxWidth = C_max_array[i]

    # Plot the input pcf grid
    im = plt.matshow(pcf_kernel_sizes)
    plt.colorbar(im)
    plt.show()  

    # OPTIONAL: this should be tested
    
    logger.info('Max kernel size: {0:.2f}'.format(float(boxWidth)))
    logger.info('Min kernel size: {0:.2f}'.format(np.amin(pcf_kernel_sizes[pcf_kernel_sizes != 0.])))

    # Set boxwidt to minimum kernel size
    # if boxWidth < anti_aliasing_kernel_size:
    #    boxWidth = anti_aliasing_kernel_size
    #    boxWidth = anti_aliasing_kernel_size + bowWidth
    #    logger.info('Max kernel size: {0:f}'.format(int(boxWidth)))


    # Get the local maximum convolutional grid matrix (this supposed to be fast)
    # This step could introduce rounding errors....
    C_max_local_matrix = ndimage.maximum_filter(pcf_kernel_sizes[...],
                            size=(boxWidth, boxWidth),
                            mode='constant',
                            cval=0)

    # Remove the rounding errors
    C_max_local_matrix = np.round(C_max_local_matrix)

    # Now loop trough the pixels of the sub-image assuming that no points are
    # gridded near the edges

    extra = 2. # Some helper value

    # The Wiener-filtering algorithm online that I try to replicate:
    # https://bitbucket.csiro.au/projects/ASKAPSDP/repos/yandasoft/browse/askap/measurementequation/WienerPreconditioner.cc#65,67,75,84,90,143,600,607,610,612

    # NOTE that I am loopint in order of x first, then y unlike the online code

    loop_count = 0
    # Loop trough the grid cells
    for x in range(int(extra*boxWidth/2), np.subtract(np.shape(pcf)[0],int(extra*boxWidth/2))):

        boxStart0 = int(np.floor(x - boxWidth/2))

        for y in range(int(extra*boxWidth/2), np.subtract(np.shape(pcf)[1],int(extra*boxWidth/2))):

            if echo_counter:
                echo_for_loop_counter(0,np.size(pcf),loop_count,
                                    'Grid cells processed')
                loop_count += 1

            region_count = 0.
            region_sum = 0.
            local_count = 0

            # Get the local max kernel manually
            boxStart1 = int(np.floor(y - boxWidth/2))

            kernelW = 0.
            for xb in range(boxStart0, int(np.ceil(boxStart0 + boxWidth))):
                for yb in range(boxStart1, int(np.ceil(boxStart1 + boxWidth))):
                    kernelW = np.max([kernelW, pcf_kernel_sizes[xb,yb]])

            kernelW = np.round(kernelW)

            if kernelW != C_max_local_matrix[x,y]:
                # print(kernelW, C_max_local_matrix[x,y])

                kernelW = np.amax([kernelW, C_max_local_matrix[x,y]])


            # If the local max kernel size is 0 pass
            if kernelW > 0:
                # kernelWidth = C_max_local_matrix[x,y]
                kernelWidth = kernelW

                regionWidth = 1 + extra * (kernelWidth - 1)

                boxStart0 = int(np.floor(x - regionWidth/2))  # Added floor
                boxStart1 = int(np.floor(y - regionWidth/2))

                localRadiusSq = 0.25 * kernelWidth*kernelWidth
                regionRadiusSq = 0.25 * regionWidth*regionWidth

                # Loop trough the sub-box
                for xb in range(boxStart0, int(boxStart0 + regionWidth)):
                    dx2 = np.power(xb - boxStart0 - regionWidth/2, 2)

                    for yb in range(boxStart1, int(boxStart1 + regionWidth)):
                        dy2 = np.power(yb - boxStart1 - regionWidth/2, 2)

                        val = pcf[xb,yb]

                        if val > 0:
                            rsq = dx2 + dy2


                            # This part should be different for the smearing, but good for the Wiener-filetring
                            if rsq <= regionRadiusSq:

                                region_count += 1
                                region_sum += val
                                # The rounding (?) is new compared to the C++ code (?)

                                # print(rsq, localRadiusSq) #There are possibly some errors here

                                # If the point is in the local kernel radius (slightly
                                # larger radius, actually)
                                if rsq <= localRadiusSq:
                                    local_count += 1

            if local_count > 0:
                # Now add the region_sum to the new array
                smeared_grid[x,y] = region_sum/region_count
    return smeared_grid

    
def smearing_fast(pcf: np.ndarray, pcf_kernel_sizes: np.ndarray, box_width: np.ndarray):
    """The core algorithm (vectorised) performing the the adaptive convolutional smearing used
    to create the Wiener-filters for weighting, and basically to estimate the
    smeared weight-density distribution.
    
    Parameters
    ----------
    pcf: np.ndarray
        The input pcf array for a single channel
    pcf_kernel_sizes: np.ndarray
        The corrected kernel sizes for the pcf
    boxWidth: np.ndarray
        A single valued array with the current channel's box width
        
    Returns
    -------
    smeared_grid: np.ndarray
        The smeared grid for a single channel
    """
    smeared_grid = np.zeros(pcf.shape)
    xrange = np.array([2*box_width/2, pcf.shape[0] - 2*box_width/2], dtype=np.int32)
    yrange = np.array([2*box_width/2, pcf.shape[1] - 2*box_width/2], dtype=np.int32)
    x,y = np.mgrid[xrange[0]:xrange[1],yrange[0]:yrange[1]]
    # # per grid cell
    # # kernel
    rxk, ryk = np.mgrid[0:box_width, 0:box_width]
    xbk = ((x-box_width/2)[..., None, None] + rxk[None, None, ...]).astype(np.int32)
    ybk = ((y-box_width/2)[..., None, None] + ryk[None, None, ...]).astype(np.int32)
    kernelW = np.round(np.max(pcf_kernel_sizes[xbk, ybk], axis=(2,3)))
    max_filter = ndimage.maximum_filter(pcf_kernel_sizes,
                                        size=(box_width, box_width),
                                        mode='constant',
                                        cval=0)
    kernelW = np.fmax(kernelW, max_filter[xrange[0,0]:xrange[1,0],yrange[0,0]:yrange[1,0]])
    mask = kernelW > 0
    d2mask = mask
    regionW = np.zeros(kernelW.shape)
    regionW[mask] = 1 + 2*(kernelW[mask]-1)
    local_radius_sq = np.zeros(kernelW.shape)
    local_radius_sq[mask] = 0.25 * kernelW[mask]**2
    region_radius_sq = np.zeros(regionW.shape)
    region_radius_sq[mask] = 0.25 * regionW[mask]**2
    # val
    rx,ry = np.mgrid[0:np.max(regionW[mask]), 0:np.max(regionW[mask])]
    xb = np.zeros((x.shape[0], x.shape[1], rx.shape[0], ry.shape[1]), dtype=np.int32)
    xb[mask, :,:] = ((x[mask]-regionW[mask]/2)[...,None,None]+rx[None,None,...]).astype(np.int32)
    yb = np.zeros((y.shape[0], y.shape[1], rx.shape[0], ry.shape[1]), dtype=np.int32)
    yb[mask, :,:] = ((y[mask]-regionW[mask]/2)[...,None,None]+ry[None,None,...]).astype(np.int32)
    mask = np.bitwise_and(mask[..., None, None], xb < (x-1+regionW/2)[..., None, None])
    mask = np.bitwise_and(mask, yb < (y-1+regionW/2)[..., None, None])
    val = np.zeros((x.shape[0], y.shape[1], rx.shape[0], ry.shape[1]))
    val[mask] = pcf[xb[mask], yb[mask]].real
    mask = np.bitwise_and(mask, val > 0)
    # rsq (These had to be split up so my kernel didn't die from memory saturation)
    dx = np.zeros(val.shape, dtype=np.float32)
    dx[..., 0 , 0][d2mask] = np.floor(x[d2mask] - regionW[d2mask]/2) + regionW[d2mask]/2
    dx[...] = dx[..., 0, 0][..., None, None]
    dx[~mask] = 0
    dx[mask] = xb[mask]-dx[mask]
    dx[mask] = np.power(dx[mask], 2)
    dy = np.zeros(val.shape, dtype=np.float32)
    dy[..., 0 , 0][d2mask] = np.floor(y[d2mask] - regionW[d2mask]/2) + regionW[d2mask]/2
    dy[...] = dy[..., 0, 0][..., None, None]
    dy[~mask] = 0
    dy[mask] = yb[mask]-dy[mask]
    dy[mask] = np.power(dy[mask], 2)
    
    rsq = dx + dy
    # region_count
    mask = np.bitwise_and(mask, rsq <= region_radius_sq[..., None, None])
    region_count = mask.sum(axis=(2,3))
    # # region_sum
    val[~mask] = 0
    region_sum = val.sum(axis=(2,3), where=mask, dtype=np.float32)
    # # local count
    mask = np.bitwise_and(mask, rsq <= local_radius_sq[..., None, None])
    local_count = mask.sum(axis=(2,3))
    local_count_cond = np.zeros(pcf.shape,dtype=np.bool_)
    local_count_cond[xrange[0,0]:xrange[1,0], yrange[0,0]:yrange[1,0]] = local_count > 0
    smeared_grid[local_count_cond] = (region_sum/region_count)[local_count > 0]
    return smeared_grid


def plot_array(array: np.ndarray):
    """A helper function to assist in debugging arrays by plotting either a line
    graph (1d) or a matshow (2d)
    
    Paramters
    ---------
    array: np.ndarray
        The array to be plot
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    if np.ndim(array) == 1:
        ax.plot(array)
    elif np.ndim(array) == 2:
        ax.matshow(array)
    else:
        raise Exception("The array has neither 1 or 2 dimensions, can't plot")
    plt.show()
    
    
def get_args() -> ap.Namespace:
    """Encapsulates the argument parser
    
    Returns
    -------
    args: ap.Namespace
        The command line arguments
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
    argparser.add_argument("-f", "--fast", 
                           action='store_true', 
                           help="Use this flag to use the fast (vectorised) algorithm.")
    argparser.add_argument("-a", "--aa-kernel-size", 
                           type=int, 
                           default=7, 
                           help="The anti-aliasing kernel size")
    args = argparser.parse_args()
    return args


# TO DO: add a put_data_to_cim function which writes a 4D array to an image ondisc
# The plan is: read in the images and create the final images (vis, psf) with
# empty complex arrays. Then read in the input data to arrays and delete the in-
# memory images to free up memory space. Perform the inverse smearing and write
# the resultant arrays to the already existing files.


#*******************************************************************************
#=== The grid normalisation functionality operating on 2D numpy arrays


#*******************************************************************************
#=== Data selection and deploynment functions


#*******************************************************************************
#=== The MAIN function


# === MAIN ===
if __name__ == "__main__":
    args = get_args()
    
    if args.directory is None:
        vispath = args.visibility_grid
        psfpath = args.psf_grid
        pcfpath = args.pcf_grid
    else:
        vispath = os.path.join(args.directory, args.visibility_grid)
        psfpath = os.path.join(args.directory, args.psf_grid)
        pcfpath = os.path.join(args.directory, args.pcf_grid)

    # Open images
    pcf_CIM = create_CIM_object(cimpath=pcfpath)
    psf_CIM = create_CIM_object(cimpath=psfpath)
    vis_CIM = create_CIM_object(cimpath=vispath)

    
    # vis_example_map = np.fabs(np.abs(vis_CIM.getdata()[0,0,...]))
    # pcf_example_map = np.fabs(np.real(pcf_CIM.getdata()[0,0,...]))


    # print(np.sum(np.ones(np.shape(vis_example_map))[vis_example_map != 0]))
    # print(np.sum(np.ones(np.shape(vis_example_map))[pcf_example_map != 0]))

    # im = plt.matshow(vis_example_map - pcf_example_map, cmap=plt.cm.binary)
    # plt.colorbar(im)
    # plt.show()

    # del vis_CIM, vis_example_map

    # exit()  
    

    # del vis_CIM

    # check_input_grids(vis_CIM, psf_CIM, pcf_CIM)

    pcfGD = pcf_CIM.getdata()
    visGD = vis_CIM.getdata()

    del pcf_CIM, vis_CIM

    # This bit finds the min/max kernels in the whole pcf grid cube (no correction for the anti-aliasing kernel)
    
    # kernel_max = 0.
    # kernel_min = 10000.

    # #print(np.shape(pcfGD))
    # for i in range(0,np.shape(pcfGD)[2]):
    #     for j in range(0,np.shape(pcfGD)[3]):

    #         if pcfGD[0,0,i,j] != 0.+0.j:

    #             kernel_size =  np.fabs(np.divide(np.imag(pcfGD[0,0,i,j]),np.real(pcfGD[0,0,i,j])))
    #             #print(pcfGD[0,0,i,j],kernel_size)

    #             if kernel_size  > kernel_max:
    #                 kernel_max = kernel_size

    #             if kernel_size < kernel_min:
    #                 kernel_min = kernel_size

    # print(kernel_max, kernel_min)


    adaptive_convolutional_smearing(pcfGD, visGD, echo_counter=True, anti_aliasing_kernel_size=args.aa_kernel_size)