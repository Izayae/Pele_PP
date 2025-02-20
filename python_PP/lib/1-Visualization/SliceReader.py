import cantera as ct
import numpy as np
import glob
import h5py as h5py

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import interpolate

import sys
import pathlib
import copy
import math
import time
import os

class SliceReader:
    """
    Class to read 2D slices into a standardized structures. The plan is to implement it for:
    - PeleAnalysis (Martin version) binary files for 2D contours
    - Amrex_kitchen for numpy 2D array files
    - Maybe a h5 file that would allow to group many slices in a single file
    This structure should contain 
    """
    def __init__(self, path_file, file_type, load_list=None):
        # file type and name:
        # - for PeleAnalysis binary: path_file leads to directory with all variable/fields
        # - for numpy format of amrex_kitchen: path_file leads to the exact npz file
        # - for h5py format:  path_file leads to the exact h5 file
        self.file_type = file_type    # type of file to be read ("binary" for PeleAnalysis, "npz" for amrex_kitchen)
        self.path_file = path_file    # full path to slice file
        
        # Slice metadata
        self.slice_name = os.path.basename(path_file)    # basename of the slice file
        #self.plt_file = plt_file      # name of the original plotfile (something like plt?????)
        self.time = None               # time of the plotfile (float)
        self.amr_level = None          # AMR level of the slice
        self.normal_dir = None         # normal direction to the slice ("x", "y", "z")
        self.normal_val = None         # distance of the slice in the normal direction (between 0 and 1 if possible)
        self.dx = None                 # grid spacing of the slice
        self.num_dim = np.zeros(2, dtype="int")     # numerical dimensions of the slice
        self.real_dim = np.zeros(2)    # real dimensions of the slice
        
        # Slice data
        self.field_available = []     # list of available fields
        self.load_list = load_list  # list of fields to actually load
        self.data = {}                # slice data (not initialized yet), one numpy array per field
        #self.min_var = 0.0            # minimum of the loaded slice (specific to Pele)
        #self.max_var = 0.0            # maximum of the loaded slice (specific to Pele)
        #self.verbose = 0              # verbosity (why ?)
        
        # Initialize object with appropriate reader
        if self.file_type == "binary":
            # Reader for PeleAnalysis binary files
            self.readBinary()
        elif self.file_type == "npz":
            # Reader for npz files from amrex_kitchen
            self.readNpz()
        elif self.file_type == "h5py":
            # Reader for h5py file from this class
            raise Error("h5py not implemented yet")
        else:
            raise Error("file_type {} not supported".format(self.file_type))
        
        return
    
    #-------------------------------------#
    #            Input section            #
    #-------------------------------------#
    
    # ------------ PeleAnalysis reader ------------ #
    def readBinary(self):
        # Reader for binary files from PeleAnalysis
        # Scan available fields
        self.field_available = [os.path.basename(var_path) for var_path in glob.glob(self.path_file+"/*")]
        
        # Check the field to load
        if self.load_list is None:
            print("Loading all available fields")
            self.load_list = self.field_available
        else:
            # check that all fields are indeed in available fields
            for field in self.load_list:
                if field not in self.field_available:
                    raise TypeError("{} is not an available field".format(field))
        
        # Read matedata for the first field
        self.readHeader(self.load_list[0])
        
        # Load data
        for field in self.load_list:
            # load each field slice into an entry of data with field as name
            var_data = self.readBinaryData(field)
            # special case for mass fraction by renaming Y_<species> in Y(<species>) for uniformity
            if "Y_" in field:
                final_field = "Y({})".format(field[2:-1])
            else:
                final_field = field
            self.data[final_field] = var_data
        
        # Read metadata from the header file
        #self.readHeader()
        
        # Then read slice data
        #self.readBinaryData()
        
        return
        
    def readHeader(self, field):
        # read header file in each directory
        header_path = self.path_file + "/{}/".format(field)
        with open(header_path + "header.txt", "r") as f:
            self.slice_name = f.readline().strip()        # plt_name
            self.time = float(f.readline().strip())       # time
            f.readline().strip()                          # variable name
            self.normal_dir = f.readline().strip()        # axis
            self.normal_val = float(f.readline().strip()) # value
            self.amr_level = int(f.readline().strip())    # level
            self.num_dim = np.array([int(n) for n in f.readline().strip().split()])    # numerical dimension
            self.real_dim = np.array([float(n) for n in f.readline().strip().split()]) # real dimension
            min_max = np.array([float(n) for n in f.readline().strip().split()])       # min/max on the slice
            #self.min = min_max[0]
            #self.max = min_max[1]
        return
        
    def readBinaryData(self, field):
        # Construct full path to binary
        binary_path = self.path_file + "/{}/".format(field)
        
        # Extract the data from binary file
        read_data = time.time()
        data_1d = np.fromfile(binary_path + "data.bin", dtype = np.dtype('<f8'))
        data_2d = data_1d.reshape(self.num_dim[1], self.num_dim[0])
        read_data = time.time() - read_data
        #if self.verbose>0:
        #print("It took a total of {:.5f}s to read binary data".format(read_data))
        return data_2d
    
    # ------------ amrex_kitchen npz reader ------------ #
    def readNpz(self):
        # Reader for npz files from amrex_kitchen
        # amrex_kitchen files do not come with metadata yet so find a way to deal with it
        # or talk about this with Olivier
        
        # simply load with numpy
        np_arr = np.load(self.path_file)
        
        # fill what little metadata we have
        self.field_available = list(np_arr.keys())
        if "x" in self.field_available and "y" in self.field_available:
            self.normal_dir = "z"
            self.dx = np_arr["x"][1]-np_arr["x"][0]
            self.num_dim[0] = len(np_arr["y"])
            self.num_dim[1] = len(np_arr["x"])
        elif "y" in self.field_available and "z" in self.field_available:
            self.normal_dir = "x"
            self.dx = np_arr["y"][1]-np_arr["y"][0]
            self.num_dim[0] = len(np_arr["z"])
            self.num_dim[1] = len(np_arr["y"])
        elif "z" in self.field_available and "x" in self.field_available:
            self.normal_dir = "y"
            self.dx = np_arr["z"][1]-np_arr["z"][0]
            self.num_dim[0] = len(np_arr["x"])
            self.num_dim[1] = len(np_arr["z"])
        self.real_dim = self.num_dim*self.dx                # deduce real dimensions from dx and size of slices
        
        # fill data with selected fields
        if self.load_list is None:
            print("Loading all available fields")
            self.load_list = self.field_available
        else:
            # check that all fields are indeed in available fields
            for field in self.load_list:
                if field not in self.field_available:
                    raise TypeError("{} is not an available field".format(field))
        
        for field in self.load_list:
            # load each field slice into an entry of data with field as name
            self.data[field] = np_arr[field]
        
        return
    
    #--------------------------------------#
    #            Output section            #
    #--------------------------------------#
    
    def saveh5(self):
        # Save everything loaded into a single h5py file containing efficiently both metadata and data
        # NOT implemented yet
        return
    
    #----------------------------------------#
    #            Operation section           #
    #----------------------------------------#
    
    # ------------ Information function ------------ #
    def writeInfo(self):
        # Just write to console main slice information
        print("-------------- Slice Reader --------------")
        print("Slice loaded from {}".format(self.slice_name))
        print("Direction: {}, distance: {}".format(self.normal_dir, self.normal_val))
        print("Lx: {:.5f}, Ly: {:.5f}".format(self.real_dim[0], self.real_dim[1]))
        print("Nx: {}, Ny: {}".format(self.num_dim[0], self.num_dim[1]))
        print("Available fields: {}".format(self.field_available))
        #print("\nLoaded fields: {}".format(self.load_list))
        print("------------------------------------------")
        return
    
    def getStateList(self, spec_names, filter_mask=None):
        ### Return a list of all datapoints states ordered by state_fields
        ### (used for anything that needs state as input: TSR/CSP/CEMA, cantera, ignition delay)
        # Check that all field are in loaded fields
        if "temp" not in self.load_list:
            raise TypeError("temp is not loaded")
        for field in spec_names:
            if field not in self.load_list:
                raise TypeError("{} is not loaded".format(field))
        
        # if no mask in input, select all datapoints
        if filter_mask is None:
            filter_mask = np.ones(self.num_dim)
        
        # Loop on all datapoints in filter_mask
        list_state = []
        list_index = []
        Nx = self.num_dim[0]
        Ny = self.num_dim[1]
        for ix in range(Nx):
            for iy in range(Ny):
                if filter_mask[ix, iy]:
                    # state vector (dictionnary)
                    state = {}
                    state["T"] = self.data["temp"][ix, iy]
                    state["Y"] = np.zeros(len(spec_names))
                    for i, spec in enumerate(spec_names):
                        state["Y"][i] = self.data[spec][ix, iy]

                    # Add to list
                    list_state.append(state)
                    list_index.append((ix, iy))
        return list_state, list_index
    
    def getGradient(self, field):
        # Return 2D gradient magnitude of desired field
        return gradient2d(self.data[field], self.dx)
    
### ------------ Processing functions ------------- ###
def gradient2d(data, dx):
    # Compute the 2D gradient magnitude (just for quick test purpose, 3D is much better)
    grad_x = np.gradient(data, dx, axis=0)
    grad_y = np.gradient(data, dx, axis=1)
    grad_mag = np.sqrt(grad_x**2+grad_y**2)
    return grad_mag
    
### ------------ Plot functions ------------- ###
def overbar_plot(fig, ax, data, cmap="RdBu_r", norm=colors.Normalize(), 
                 extent=None, origin="lower", colorbar=True, out_white=False):
    im_bin = ax.imshow(data, origin = origin, cmap = cmap, 
                       norm=norm, extent=extent, aspect="equal", interpolation="nearest")
    
    # turn value under min of colormap white
    if out_white:
        im_bin.cmap.set_under('white')
    
    # print the colorbar on top of the contour
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='10%', pad=0.01)
        fig.colorbar(im_bin, cax=cax, orientation="horizontal")
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')
    return

def simple_plot(fig, ax, data, cmap="RdBu_r", norm=colors.Normalize(), extent=None, origin="lower", colorbar=True):
    im_bin = ax.imshow(data, origin = origin, cmap = cmap, norm=norm, extent=extent, aspect="equal")
    
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='10%', pad=0.01)
        fig.colorbar(im_bin, cax=cax, orientation="horizontal")
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')
    return
    