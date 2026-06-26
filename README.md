# H5Yaml
[![image](https://img.shields.io/pypi/v/h5yaml.svg?label=release)](https://github.com/rmvanhees/h5_yaml/)
[![image](https://img.shields.io/pypi/l/h5yaml.svg)](https://github.com/rmvanhees/h5_yaml/LICENSE)
[![image](https://img.shields.io/pypi/dm/h5yaml.svg)](https://pypi.org/project/h5_yaml/)
[![image](https://img.shields.io/pypi/status/h5yaml.svg?label=status)](https://pypi.org/project/h5_yaml/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


## Description

This Python package let you design the abstract data model of your [HDF5](https://docs.h5py.org/en/stable/)/[netCDF4](https://unidata.github.io/netcdf4-python/) files.
Where the Abstract Data Model is a conceptual model of data, data types, and data organization. We choose the human-readable data serialization language [YAML](https://yaml.org/) to define the Abstract Data Model with the following components:

 * *Groups* which define its Hierarchical structure
 * *Dimensions* the size(s) of the variables
 * *Variables* the datasets to hold your data
 * *Attributes* the meta data of the *File*, *Groups*, or *Variables*

From the YAML files, you can create HDF5 or netCDF4 files, which are small because the Variables may still be empty. Thus the abstract data model and storage model can be shared among your colleagues for review, or for metadata compliance checks. For example the [CF conventions](https://cfconventions.org/) or the [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3), by using:

 * https://mcc.podaac.earthdatacloud.nasa.gov

And finally, you can implement the file-structure in JAVA, C++ or Fortran.
But of course you can also simply generate the empty products and fill the datasets using Python.

In short, this approach has the following advantages:

 * you define the layout of your HDF5/netCDF4 file using YAML which is human-readable and has intuitive syntax.
 * you can reuse the YAML configuration file to to have all your product have a consistent layout.
 * you can make updates by only changing the YAML configuration file.
 * you can have the layout of your HDF5/netCDF4 file as a Python dictionary, thus without accessing any HDF5/netCDF4 file.

The package `h5yaml` provides the classes `YamlToH5` and `YamlToNc` to generate a HDF5/netCDF4 formatted file from a Python dictionary.

 1. The class `YamlToH5` uses the [h5py](https://pypi.org/project/h5py/) package, which is a Pythonic interface to
    the HDF5 binary data format. The generated HDF5 file should be compatible with the netCDF4 format.
    YamlToH5 is slightly faster than the netCDF4 implementation and generates smaller files.
 2. The class `YamlToNc` uses the [netCDF4](https://pypi.org/project/netCDF4/) package, which provides an object-oriented
    python interface to the netCDF version 4 library. You should use this class when strict conformance with the netCDF4 format
    is required. However, package `netCDF4` has some limitations, which `h5py` has not, for example it does
    not allow variable-length variables to have a compound data-type.
 3. Both classes inherit the class ReadNcYaml, which read the Yaml files and combine the data in to a Python dictionary.

## Installation
The package `h5yaml` is available from PyPI. To install it use `pip`:

> $ pip install [--user] h5yaml

The module `h5yaml` requires Python3.10+ and Python modules: h5py (v3.14+), netCDF4 (v1.7+) and numpy (v2.0+).


## Usage

Example 1) Use the class `YamlToNc` to generate a template netCDF4 file:
```
from importlib.resources import files

from h5yaml.yaml_to_nc import YamlToNc

yaml_list = [
   files("h5yaml.Data") / "nc_testing.yaml"),
   files("h5yaml.Data") / "h5_global_attrs.yaml",
]
\# show the YAML configuration as a Python dictionary using pprint
aa = ReadNcYaml(yaml_list)
print(repr(aa))

\# generate an in-memory HDF5 file
aa = YamlToNc(yaml_list)
fid = aa.diskless()

\# Optional, set any unlimited dimension to a fixed length
aa.set_dims()

\# write data to datasets of the file
\# ...

\# write netCDF4 file to disk
res.to_disk(fid, filename)
```

Example 2) Use the class `YamlToH5` to generate a template HDF5 file:
```
from importlib.resources import files

from h5yaml.yaml_to_h5 import YamlToH5

yaml_list = [
   files("h5yaml.Data") / "h5_testing.yaml"),
   files("h5yaml.Data") / "h5_global_attrs.yaml",
]

# show the YAML configuration as a Python dictionary using pprint
aa = ReadNcYaml(yaml_list)
print(repr(aa))

# generate an in-memory HDF5 file
aa = YamlToH5(yaml_list)
fid = aa.diskless()

# Optional, set any unlimited dimension to a fixed length
aa.set_dims()

# write data to datasets of the file
# ...

# write HDF5 file to disk
res.to_disk(fid, filename)
```

Example 3) Use the class `YamlToH5` to generate a template netCDF4 file on disk and add data later.
```
from importlib.resources import files

from netCDF4 import Dataset

from h5yaml.nc_from_yaml import NcFromYaml

res = YamlToNc(files("h5yaml.Data") / "nc_testing.yaml").create(filename)
with Dataset(filename, "r+") as fid
  # write data to variables of the file
  ...
```

The YAML file should be structured as follows:

 * The top level are: 'groups', 'dimensions', 'compounds', 'variables', 'attrs\_global' and 'attrs\_groups'.
 * > 'attrs\_global' and 'attrs\_groups' are added in version 0.3.0
 * The names of the attributes, groups, dimensions, compounds and variable should be specified as PosixPaths, however:
   * The names of groups should never start with a slash (always relative to root);
   * All other elements which are stored in root should also not start with a slash;
   * Hoewever the non-group elements require a starting slash (absolute paths) when they are stored not the root. 
 * The section 'groups' are optional, but you should provide each group you want to use
   in your file. The 'groups' section in the YAML file may look like this:
   ```
   groups:
     - engineering_data
     - image_attributes
     - navigation_data
     - science_data
     - processing_control/input_data
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
   you want to use in your file. For each compound element you have to provide a list with its
   data-type and optionally its number of elements. The 'compound' section may look like
   this:

   ```
   compounds:
     stats_dtype:
       time: [u8]
       index: [u2]
       tbl_id: [u1]
       saa: [u1]
       coad: [u1]
       texp: [f4]
       lat: [f4]
       lon: [f4]
       avg: [f4]
       unc: [f4, [2,]]
       dark_offs: [f4]
   ```

 * The 'variables' are defined by their data-type ('_dtype') and dimensions ('_dims'),
   and optionally chunk sizes ('_chunks'), compression ('_compression'), variable length
   ('_vlen'). In addition, each variable can have as many attributes as you like,
   defined by its name and value. The 'variables' section may look like this:

   ```
   variables:
     /science_data/detector_images:
       _dtype: u2
       _dims: [number_of_images, samples_per_image]
	   _compression: 3
       _FillValue: 65535
       long_name: Detector pixel values
       coverage_content_type: image
       units: '1'
       valid_min: 0
       valid_max: 65534
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
       _dims: [days]
       _vlen: True
       comment: detector map statistics (MPS=163)
   ```

### Notes and ToDo

* The layout of a HDF5 or netCDF4 file can be complex. From version 0.3, you can split the file definition over several YAML files and provide a list with the names of YAML files as input to H5Yaml and NcYaml.
* From version 0.4, the classes `H5Yaml` and `NcYaml` are replaced by `NcFromYaml`. You can use module `h5py` to write the netCDF4 file or `netCDF4` using the keyword 'module'. Then the classes `H5Create` or `NcCreate` perform the `dict` to netCDF4 conversion.

## Support [TBW]

## Road map

 * Release v0.1 : stable API to read your YAML files and generate the HDF5/netCDF4 file


## Authors and acknowledgment
The code is developed by R.M. van Hees (SRON)

## License

* Copyright: Richard van Hees (SRON) (https://www.sron.nl).
* License: Apache-2.0
