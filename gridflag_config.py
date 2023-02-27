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
def echo_for_loop_counter(start,end,count,loop_string='Loop state'):
    """Simple routine to print the status of a for loop to the stdout and update
    it dynamically.
    
    It is useful for testing and running kinda long code on a screen session
    The code should be called within the loop, but it only works with for loops
    as one needs to know the start and end values of the loop.
    NOTE if it is logged every loop cycle will print a line to the logfile (!)
    NOTE this code is not compatible with the logging module
    
    NOTE no other print statements should be in the for loop (!)
    Parameters:
    ===========
    start: int
        The start value of the loop
    end: int
        The end value of the loop
    count: int
        The current vale of the loop
    loop string: str
        A string to name the loop. The counter is printed in the following format:
        `loop string`: [===.......] x% y/zzzz
    Returns:
    ========
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
def create_CIM_object(cimpath):
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

def check_input_grids(visgrid, psfgrid, pcfgrid):
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

def adaptive_convolutional_smearing(initial_pcf_grid_array,
                                    reference_grid_array,
                                    echo_counter=False,
                                    anti_aliasing_kernel_size=7):
    """The core algorithm performing the the adaptive convolutional smearing used
    to create the Wiener-filters for weighting, and basically to estimate the
    smeared weight-density distribution.

    The input arrays should follow the casaimage format and have to be 4D:

    [channel, pol, ra, dec]

    NOTE: the code works only on StokesI polarisation

    The computations, should be vectorised, and so *should* have an okay performance.

    The bottleneck could be the size of the input arrays as the code need to store

    The RE(initial_pcf_grid_array) contains the cell-averaged SNR weight
    The IM(initial_pcf_grid_array) contains the cell- and SNR-weight- averaged
    kernel sizes


    The function returns the PCF grid.



    NOTE: the imaginary part should be taken to the absolute value as half the
        uv-plane is set to be negative to retain the hermitian property of the images



    """
    # Create the output array
    smeared_grid = np.zeros(np.shape(initial_pcf_grid_array))

    # Split the data to real and imaginary part
    pcf_real = np.real(initial_pcf_grid_array[:,0,...])
    pcf_imag = np.imag(initial_pcf_grid_array[:,0,...])

    # Generate a matrix with the kernel sizes
    # pcf_kernel_sizes = np.fabs(np.divide(pcf_imag,pcf_real, where=pcf_real!=0))

    # Get the maximum projection kernel (per channel)
    # Array containing the max kernels per channel rounded to % precision to get rid
    # of numerical errors, then a ceil() is called (i.e 1.1 => 2 ; but 1.00001 => 1)
    # C_max_array = np.ceil(np.round(np.amax(pcf_kernel_sizes, axis=(1,2)),2))
    # C_max_array = np.ceil(np.amax(pcf_kernel_sizes, axis=(1,2)))

    # Perform the operations by channel
    for i in range(0,np.shape(pcf_imag)[0]):
        # The maximum kenel width in the given channel
        # boxWidth = C_max_array[i]

        pcf_kernel_sizes = np.fabs(np.divide(pcf_imag[i,...],
                                            pcf_real[i,...],
                                            where=pcf_real[i,...]!=0))

        # Plot the input pcf grid
        im = plt.matshow(pcf_kernel_sizes)
        plt.colorbar(im)
        plt.show()  

        # OPTIONAL: this should be tested

        pcf_kernel_sizes = np.ceil(pcf_kernel_sizes) # correcting for numerics

        # Apply correction for the kernel sizes (to whole grid)
        # pcf_kernel_sizes[pcf_kernel_sizes != 0] += anti_aliasing_kernel_size

        # Apply correction, but only for the grid cells with < anti_aliasing_kernel_size
        pcf_kernel_sizes[(pcf_kernel_sizes != 0) & (pcf_kernel_sizes < anti_aliasing_kernel_size)] = \
        anti_aliasing_kernel_size + pcf_kernel_sizes[(pcf_kernel_sizes != 0) & (pcf_kernel_sizes < anti_aliasing_kernel_size)]


        # Get the max kernel size for the smearing
        boxWidth = np.ceil(np.amax(pcf_kernel_sizes))
        
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
                                size=boxWidth,
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
        for x in range(int(extra*boxWidth/2), np.subtract(np.shape(pcf_real)[1],int(extra*boxWidth/2))):

            boxStart0 = int(np.floor(x - boxWidth/2))

            for y in range(int(extra*boxWidth/2), np.subtract(np.shape(pcf_real)[2],int(extra*boxWidth/2))):

                if echo_counter:
                    echo_for_loop_counter(0,np.size(pcf_real[i,...]),loop_count,
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

                            val = pcf_real[i,xb,yb]

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
                    smeared_grid[i,0,x,y] = region_sum/region_count

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

        # print(np.unique(E))



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

    pcfgrid = 'pcfgrid.dal_test0.dumpgrid'
    psfgrid = 'psfgrid.dal_test0.dumpgrid'
    visgrid = 'visgrid.dal_test0.dumpgrid'

    grid_dir_path = '/home/krozgonyi/Desktop/playground/grid_flagging/blob/'

    pcf_grid_path = grid_dir_path + pcfgrid
    psf_grid_path = grid_dir_path + psfgrid
    vis_grid_path = grid_dir_path + visgrid

    # Open images
    pcf_CIM = create_CIM_object(cimpath=pcf_grid_path)
    psf_CIM = create_CIM_object(cimpath=psf_grid_path)
    vis_CIM = create_CIM_object(cimpath=vis_grid_path)

    
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


    adaptive_convolutional_smearing(pcfGD, visGD, echo_counter=True, anti_aliasing_kernel_size=0)