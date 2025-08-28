"""
Author: Brian Breitsch
Date: 2025-01-02
"""

import h5py
import numpy as np


def read_hdf5_into_dict(
        group: h5py.Group,
        read_datasets: str = "all",
        read_groups: str = "all",
        read_paths: str = "all",
        read_attrs: bool = True,
        convert_numpy_scalar_attrs: bool = True,
        ) -> dict:
    """
    ----------------------------------------------------------------------------
    Given an open HDF5 group (i.e. from open `h5py.File`), reads the data
    stucture recursively and stores in nested Python `dict` objects.
    
    `group` -- HDF5 group object, e.g. the root HDF5 file `h5py.File`
    `read_datasets` -- given a list of datasets to read, if a `Dataset.name`
        matches an entry in `read_datasets`, then it will be read.  Otherwise,
        if `"all"` (default) this function will parse all datasets
    `read_groups` -- given a list of group names to read, if a `Group.name`
        matches an entry in `read_groups`, then that group will be read.
        Otherwise, if `"all"` (default) this function will parse all datasets
    `read_paths` -- given a list of paths to either groups or datasets, this
        function will read the data starting from these paths.  Otherwise, if
        `"all"`, then all paths will be read.
    `read_attrs` -- whether to read attributes.  This function only reads and
        sets values for attributes of groups, since there is no clear key using
        this schema for attributes of datasets.
    `convert_numpy_scalar_attrs` -- whether to convert numpy scalar values inside
        `attrs` to Python scalars.
    """ 
    data = {}
    # read group attributes
    if read_attrs:
        attrs = {}
        for key, attr in group.attrs.items():
            if convert_numpy_scalar_attrs and isinstance(attr, np.generic):
                attrs[key] = attr.item()
            else:
                attrs[key] = attr
        if len(attrs) > 0:
            data["attrs"] = attrs
    if read_paths == "all":
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset) and (read_datasets == "all" or key in read_datasets):
                if group[key].shape == ():
                    data[key] = group[key][()]
                else:
                    data[key] = group[key][:]
            elif isinstance(group[key], h5py.Group) and (read_groups == "all" or key in read_groups):
                data[key] = read_hdf5_into_dict(group[key], read_datasets, read_groups)
    else:
        for path in read_paths:
            if path not in group:
                continue  # path must be valid within group
            # Note: valid HDF5 paths do NOT begin with "/"
            keys = path.split("/")  # there must be at least one valid string in the path
            assert(len(keys) >= 1)
            if len(keys) == 1: 
                if isinstance(group[path], h5py.Dataset):
                    if group[path].shape == ():
                        data[keys[0]] = group[path][()]
                    else:
                        data[keys[0]] = group[path][:]
                elif isinstance(group[path], h5py.Group):
                    data[keys[0]] = read_hdf5_into_dict(group[path], read_datasets, read_groups)
            else:
                data[keys[0]] = d = {}
                for k in keys[1:-1]:
                    d[k] = {}
                    d = d[k]
                if isinstance(group[path], h5py.Dataset):
                    if group[path].shape == ():
                        data[keys[-1]] = group[path][()]
                    else:
                        data[keys[-1]] = group[path][:]
                elif isinstance(group[path], h5py.Group):
                    d[keys[-1]] = read_hdf5_into_dict(group[path], read_datasets, read_groups)
    return data


def write_dict_to_hdf5(data: dict, output_group: h5py.Group, path: str = "", ignore_objects: bool = True, write_scalars_as_attributes: bool = True):
    """
    ----------------------------------------------------------------------------
    Given a dictionary-like structure, writes to an HDF5 file.  `ndarray`
    objects are written as HDF5 datasets, while scalars, strings, etc. are
    written as group attributes.  The HDF5 path structure follows the tree
    structure of the dictionary.
    
    `data` -- dictionary object containing data
    `output_group` -- handle to HDF5 output group
    """
    for key, item in data.items():
        if np.isscalar(item):
            if write_scalars_as_attributes:
                output_group.attrs[key] = item
            else:
                output_group.create_dataset(path + str(key), data=item)
        elif isinstance(item, np.ndarray) or isinstance(item, list) or isinstance(item, tuple):
            output_group.create_dataset(path + str(key), data=np.array(item))
        elif isinstance(item, dict):
            write_dict_to_hdf5(item, output_group, path + str(key) + "/")
        else:
            to_dict = getattr(item, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                if isinstance(d, dict):
                    write_dict_to_hdf5(d, output_group, path + str(key) + "/")
            if not ignore_objects:
                raise ValueError("Cannot save {0} type".format(type(item)))

            
            
def summarize_dataset(f: h5py.File, num_levels=None, max_num_lines=None):
    """
    `num_levels` -- number of levels deep to print group names (default None; prints all levels)
    """
    def print_recursive(f: h5py.File, indent: str, level: int, num_lines: int):
        if num_levels is not None and level == num_levels:
            return num_lines
        for key in f:
            if num_lines == max_num_lines:
                return num_lines
            if isinstance(f[key], h5py.Group):
                print(f"{indent}{key}:")
                num_lines += 1
                num_lines = print_recursive(f[key], indent + " ", level + 1, num_lines)
            else:
                print(f"{indent}{key}")
                num_lines += 1
        return num_lines
    indent = ""
    level = 0
    num_lines = 0
    num_lines = print_recursive(f, indent, level, num_lines)