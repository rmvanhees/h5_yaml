# H5YAML
[![image](https://img.shields.io/pypi/v/h5yaml.svg?label=release)](https://github.com/rmvanhees/h5yaml/)
[![image](https://img.shields.io/pypi/l/h5yaml.svg)](https://github.com/rmvanhees/h5yaml/LICENSE)
[![image](https://img.shields.io/pypi/dm/h5yaml.svg)](https://pypi.org/project/h5yaml/)
[![image](https://img.shields.io/pypi/status/h5yaml.svg?label=status)](https://pypi.org/project/h5yaml/)

## Description
This package let you generate [HDF5](https://docs.h5py.org/en/stable/)/[netCDF4](https://unidata.github.io/netcdf4-python/)
formatted files as defined in a [YAML](https://yaml.org/) configuration file. This has several advantages: 

 * you define the layout of your HDF5/netCDF4 file using YAML which is human-readable and has intuitive syntax.
 * you can reuse the YAML configuration file to to have all your product have a consistent layout.
 * you can make updates by only changing the YAML configuration file
 * you can have the layout of your HDF5/netCDF4 file as a python dictionary, thus without accessing any HDF5/netCDF4 file

The `H5YAML` package has two classes to generate a HDF5/netCDF4 formatted file.

 1. The class `H5Yaml` uses the [h5py](https://pypi.org/project/h5py/) package, which is a Pythonic interface to
    the HDF5 binary data format.
    Let 'h5_def.yaml' be your YAML configuration file then ```H5Yaml("h5_def.yaml").create("foo.h5")``` will create
	the HDF5 file 'foo.h5'. This can be read by netCDF4 software, because it uses dimension-scales to each dataset.
 2. The class `NcYaml` uses the [netCDF4](https://pypi.org/project/netCDF4/) package, which provides an object-oriented
    python interface to the netCDF version 4 library.
    Let 'nc_def.yaml' be your YAML configuration file then ```NcYaml("nc_def.yaml").create("foo.nc")``` will create
	the netCDF4/HDF5 file 'foo.nc'

The class `NcYaml` must be used when strict conformance to the netCDF4 format is required.
However, package `netCDF4` has some limitations, which `h5py` has not, for example it does
not allow variable-length variables to have a compound data-type.

## Installation
Releases of the code, starting from version 0.1, will be made available via PyPI.

## Usage

The YAML file should be structured as follows:

 * The top level are: 'groups', 'dimensions', 'compounds' and 'variables'
 * The section 'groups' are optional, but you should provide each group you want to use
   in your file. The 'groups' section in the YAML file may look like this:

   ```
   groups:
     - engineering_data
     - image_attributes
     - navigation_data
     - processing_control
     - science_data
   ```

 * The section 'dimensions' is obligatory, you should define the dimensions for each
   variable in your file. The 'dimensions' section may look like this:

   ```
   dimensions:
     days:
       _dtype: u4
       _size: 0
       long_name: days since 2024-01-01 00:00:00Z
     number_of_images:             # an unlimited dimension
       _dtype: u2
       _size: 0
     samples_per_image:            # a fixed dimension
       _dtype: u4
       _size: 307200
     /navigation_data/att_time:    # an unlimited dimension in a group with attributes
       _dtype: f8
       _size: 0
       _FillValue: -32767
       long_name: Attitude sample time (seconds of day)
       calendar: proleptic_gregorian
       units: seconds since %Y-%m-%d %H:%M:%S
       valid_min: 0
       valid_max: 92400
     n_viewport:                   # a fixed dimension with fixed values and attributes
       _dtype: i2
       _size: 5
       _values: [-50, -20, 0, 20, 50]
       long_name: along-track view angles at sensor
       units: degrees
   ```

 * The 'compounds' are optional, but you should provide each compound data-type which
   you want to use in your file. For each compound element you have to provide its
   data-type and attributes: units and long_name. The 'compound' section may look like
   this:

   ```
   compounds:
     stats_dtype:
       time: [u8, seconds since 1970-01-01T00:00:00, timestamp]
       index: [u2, '1', index]
       tbl_id: [u1, '1', binning id]
       saa: [u1, '1', saa-flag]
       coad: [u1, '1', co-addings]
       texp: [f4, ms, exposure time]
       lat: [f4, degree, latitude]
       lon: [f4, degree, longitude]
       avg: [f4, '1', '$S - S_{ref}$']
       unc: [f4, '1', '\u03c3($S - S_{ref}$)']
       dark_offs: [f4, '1', dark-offset]
   ```

   Alternatively, provide a list with names of YAML files which contain the definitions
   of the compounds.

   compounds:
     - h5_nomhk_tm.yaml
     - h5_science_hk.yaml

 * The 'variables' are defined by their data-type ('_dtype') and dimensions ('_dims'),
   and optionally chunk sizes ('_chunks'), compression ('_compression'), variable length
   ('_vlen'). In addition, each variable can have as many attributes as you like,
   defined by its name and value. The 'variables' section may look like this:

   ```
   variables:
     /image_attributes/nr_coadditions:
       _dtype: u2
       _dims: [number_of_images]
       _FillValue: 0
       long_name: Number of coadditions
       units: '1'
       valid_min: 1
     /image_attributes/exposure_time:
       _dtype: f8
       _dims: [number_of_images]
       _FillValue: -32767
       long_name: Exposure time
       units: seconds
     stats_163:
       _dtype: stats_dtype
       _vlen: True
       _dims: [days]
       comment: detector map statistics (MPS=163)
   ```

### Notes and ToDo:

 * The usage of older versions of h5py may result in broken netCDF4 files
 * Explain usage of parameter '_chunks', which is currently not correctly implemented.
 * Explain that the usage of variable length data-sets may break netCDF4 compatibility

## Support [TBW]

## Roadmap

 * Release v0.1 : stable API to read your YAML files and generate the HDF5/netCDF4 file


## Authors and acknowledgment
The code is developed by R.M. van Hees (SRON)

## License

* Copyright: SRON (https://www.sron.nl).
* License: BSD-3-clause
