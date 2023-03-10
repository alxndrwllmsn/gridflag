# gridflag

A general description should go here...

## Requirements
- python >= 3.10.9
- numpy >= 1.24.2
- jupyter notebook (if you want to use the ipynb)
- matplotlib >= 3.7.0
- python-casacore == 3.5.2
- scipy >= 1.10.0

## Usage
```
gridflag_config.py [-h] [-d DIRECTORY] [-f] visibility_grid psf_grid pcf_grid
```

positional arguments:
- visibility_grid: The path to the visibility grid (use -d to specify working directory).
- psf_grid: The path to the psf grid (use -d to specify working directory).
- pcf_grid: The path to the pcf grid (use -d to specify working directory).

options:
```
-h, --help          show this help message and exit

-d DIRECTORY, --directory DIRECTORY
                    The path to the working directory, all other specified paths will be relative to this one.

-f, --fast            use the fast (vectorised) algorithm.
```
