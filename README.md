# H5_YAML

## Description
Use YAML configuration file to generate HDF5/netCDF4 formated files.

The class `NcYaml` must be used when strict conformance to the netCDF4 format is
required. However, the python netCDF4 implementation does not allow variable-length
data to have a compound data-type. The class `H5Yaml` does not have this restiction
and will generate HDF5 formated files which can be read by netCDF4 software.

## Installation
Relases of the code, starting from version 0.1, will be made available via PyPi.

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

 * The section 'dimensions' is obligatory, you shouold define the dimensions for each
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
 * Explain usage of parameter '_chunks', which is currently not correcly implemented.
 * Explain that the usage of variable length data-sets may break netCDF4 compatibility

## Support [TBW]

## Roadmap

 * Release v0.1 : stable API to read your YAML files and generate the HDF5/netCDF4 file


## Authors and acknowledgment
The code is developed by R.M. van Hees (SRON)

## License

* Copyright: SRON (https://www.sron.nl).
* License: BSD-3-clause

## Project status
Beta
