import numpy as np
from math import *

from itertools import product

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import *

from scipy import interpolate

import glob
import sys
import pathlib
import copy
import math
import time
import os
import copy

# |-----------------------------|
# |          Bin class          |
# |-----------------------------|F
class Bin:
    # Class for single bin parameters and values
    def __init__(self, name_var, n_bin, is_log, count_ext, min_var, max_var):
        self.name_var = name_var
        self.n_bin = n_bin
        self.is_log = is_log
        self.count_ext = count_ext
        self.min_var = min_var
        self.max_var = max_var
        self.value_bin = self.binValues()
        
    def binValues(self):
        value_bin = np.zeros(self.n_bin)
        for i_bin in range(self.n_bin):
            if self.is_log==1:
                dbin = (log10(self.max_var)-log10(self.min_var))/self.n_bin;
                value_bin[i_bin] = self.min_var * pow(10, dbin*(0.5+i_bin));
            else:
                dbin = (self.max_var-self.min_var)/self.n_bin;
                value_bin[i_bin] = self.min_var + dbin*(0.5+i_bin);
        return value_bin
    
    def getIdBin(self, value):
        # Find the index of the bin where the datapoint is situated
        # require later check if returned id is outside of the bin range
        id_bin = 0;
        if (self.is_log==1):
            id_bin = int((self.n_bin*(log10(value)-log10(self.min_var))/(log10(self.max_var)-log10(self.min_var))))
        else:
            id_bin = int((self.n_bin*(value-self.min_var)/(self.max_var-self.min_var)))
            
        # Count the values outside as extreme index
        if (id_bin<0):
            id_bin=0
        if (id_bin>=self.n_bin):
            id_bin=self.n_bin-1
        
        #print("min: {}, max:{}".format(self.min_var, self.max_var))
        #print("value: {}, id: {}".format(value, id_bin))
        
        return id_bin;

# |-----------------------------------|
# |          MultiStat class          |
# |-----------------------------------|
class MultiStat:
    # Class for the statistics
    # |-----------------------------------|
    # |            Constructor            |
    # |-----------------------------------|
    def __init__(self, path_file, var, verbose=1, case=None):
        # miscellaneous
        self.verbose = verbose
        self.case=case
        
        # Header information
        self.plt_file = " "
        self.time = 0.0
        self.n_lev = 0
        self.n_dims = 1
        self.list_bin = []
        self.name_bin = []
        self.dim_bin = []
        self.n_dim_prod = []
        
        path_header = path_file + "/header.txt"
        self.readHeader(path_header)
        
        if self.verbose == 1:
            print("\n|-------------------- Reading {} --------------------|".format(self.plt_file))
            self.printGlobalInformation()
            
        # Pdf information
        path_pdf = path_file + "/data_pdf.bin"
        self.pdf = np.zeros(tuple(self.dim_bin))
        self.readPdf(path_pdf)
        
        if self.verbose == 1:
            self.printPdfInformation()
        
        # Data information
        self.var = var
        path_data = path_file + "/{}/".format(var)
        self.data_min = np.zeros(tuple(self.dim_bin))
        self.data_max = np.zeros(tuple(self.dim_bin))
        self.data_mean = np.zeros(tuple(self.dim_bin))
        self.readData(path_data)
    
        if self.verbose == 1:
            self.printDataInformation()
            
        return
    
    # |----------------------------------------|
    # |          Convenience functions         |
    # |----------------------------------------|
    def getValueBin(self, name_bin):
        id_bin = self.name_bin.index(name_bin)
        value_bin = self.list_bin[id_bin].value_bin
        return value_bin
    
    def getCleanPdf(self, threshold=1):
        pdf = self.pdf
        pdf[self.pdf < threshold] = None
        return pdf
    
    def getCleanMean(self, threshold=1):
        clean_mean = copy.deepcopy(self.data_mean)
        clean_mean[self.pdf < threshold] = None
        return clean_mean
    
    def getCleanMin(self, threshold=1):
        clean_min = copy.deepcopy(self.data_min)
        clean_min[self.pdf < threshold] = None
        clean_min[clean_min>1e18] = None
        return clean_min
    
    def getCleanMax(self, threshold=1):
        clean_max = copy.deepcopy(self.data_max)
        clean_max[self.pdf < threshold] = None
        return clean_max
    
    # |-------------------------------------------|
    # |          General output functions         |
    # |-------------------------------------------|
    def printGlobalInformation(self):
        print("Global information:")
        print("    Simulation time: {:.4e}s".format(self.time))
        print("    AMR levels: {}".format(self.n_lev))
        print("    Statistics dimensions: {}".format(self.n_dims))
        print("    Names of each bin: {}".format(self.name_bin))
        print("    Dimension of each bin: {}".format(self.dim_bin))
        print("")
        return
    
    def printPdfInformation(self):
        print("Pdf information:")
        print("    Total points extracted: {}".format(int(np.sum(self.pdf))))
        print("")
        return
    
    def printDataInformation(self):
        print("Data information:")
        print("    Global min of {}: {:.3e}".format(self.var, np.min(self.data_min)))
        print("    Global max of {}: {:.3e}".format(self.var, np.max(self.data_max)))
        print("    Global mean of {}: {:.3e}".format(self.var, np.sum(self.data_mean*self.pdf)/np.sum(self.pdf)))
        print("")
        return
    
    # |-----------------------------------|
    # |          Input functions          |
    # |-----------------------------------|
    def updateNProd(self):
        for i_dim in range(self.n_dims):
            n_prod=1
            for j_dim in range(i_dim):
                n_prod = n_prod*self.dim_bin[j_dim]
            self.n_dim_prod.append(n_prod)
        return
    
    def readHeader(self, path_header):
        with open(path_header, "r") as f:
            # Read plotfiles parameters from global header
            self.plt_file = f.readline().strip()      # plt_file
            self.time = float(f.readline().strip())   # time
            self.n_lev = int(f.readline().strip())    # level
            self.n_dims = int(f.readline().strip())   # n_dims
            for i_dim in range(self.n_dims):
                name_var = f.readline().strip()       # name of bin variable
                n_bin = int(f.readline().strip())     # number of bins
                is_log = int(f.readline().strip())    # are bins logarithmic
                min_var = float(f.readline().strip())   # minimum value bin
                max_var = float(f.readline().strip())   # maximum value bin
                count_ext = int(f.readline().strip()) # points outside of bin range counted ?
                self.list_bin.append(Bin(name_var, n_bin, is_log, 
                                         count_ext, min_var, max_var))
                
                self.name_bin.append(name_var)
                self.dim_bin.append(n_bin)
                
            # Update the multiplicative n_dim_prod for later use
            self.updateNProd()
                
        return
    
    def calcIdGlobal(self, index):
        id_global = 0
        for i_dim in range(self.n_dims):
            id_global += self.n_dim_prod[i_dim]*index[i_dim]
        
        return id_global
    
    def readPdf(self, path_pdf):
        # Extract the data from binary file
        read_data = time.time()
        
        # weird change from int to float
        #data_1d = np.fromfile(path_pdf, dtype = np.dtype('<uint32'))
        data_1d = np.fromfile(path_pdf, dtype = np.dtype('<f8'))
        
        list_dim = [list(range(self.dim_bin[self.n_dims-1-i])) for i in range(self.n_dims)]
        for index in list(product(*list_dim)):
            index=index[::-1]
            id_global = self.calcIdGlobal(index)
            #if id_global >= len(data_1d)-1:
            #    print(self.dim_bin)
            #    print(index)
            #    print(id_global)
            self.pdf[index]=data_1d[id_global]
        
        read_data = time.time() - read_data
        
        if self.verbose == 1:
            print("It took a total of {:.5f}s to read pdf".format(read_data))
        return
    
    def readData(self, path_data):
        # Extract the data from binary file
        read_data = time.time()
        
        # min stats
        data_1d = np.fromfile(path_data + "/data_min.bin", dtype = np.dtype('<f8'))
        
        list_dim = [list(range(self.dim_bin[self.n_dims-1-i])) for i in range(self.n_dims)]
        for index in list(product(*list_dim)):
            index=index[::-1]
            id_global = self.calcIdGlobal(index)
            self.data_min[index]=data_1d[id_global]
        
        # max stats
        data_1d = np.fromfile(path_data + "/data_max.bin", dtype = np.dtype('<f8'))
        
        list_dim = [list(range(self.dim_bin[self.n_dims-1-i])) for i in range(self.n_dims)]
        for index in list(product(*list_dim)):
            index=index[::-1]
            id_global = self.calcIdGlobal(index)
            self.data_max[index]=data_1d[id_global]
            
        # mean stats
        data_1d = np.fromfile(path_data + "/data_mean.bin", dtype = np.dtype('<f8'))
            
        list_dim = [list(range(self.dim_bin[self.n_dims-1-i])) for i in range(self.n_dims)]
        for index in list(product(*list_dim)):
            index=index[::-1]          # reverse tuple to get the right indexing order
            id_global = self.calcIdGlobal(index)
            self.data_mean[index]=data_1d[id_global]
        
        read_data = time.time() - read_data
        
        if self.verbose == 1:
            print("It took a total of {:.5f}s to read data".format(read_data))
        return
    
    # |-----------------------------------|
    # |       Processing functions        |
    # |-----------------------------------|
    ### Series of reduce functions
    def reduceSingleDimension(self, name_bin_reduc):
        # Print some useful information
        if self.verbose == 1:
            print("\n|-------------------- Reducing data in dimension {} --------------------|".format(name_bin_reduc))
        
        # Start timer
        reduc_data = time.time()
        
        # Reduce the data by 1 dimension indicated by name_bin_reduc
        # Verify that the variable to reduce is available in the bins loaded
        name_bins = [self.list_bin[i].name_var for i in range(self.n_dims)]
        if name_bin_reduc not in name_bins:
            raise Exception("{} is not in the bins".format(name_bin_reduc))
            
        # Detect which bin is associated with name_bin_reduc
        i_bin_reduc = name_bins.index(name_bin_reduc)
        
        # First compute the new pdf by summing in the direction of the i_bin_reduc dimension
        list_dim = []
        new_dim_bin = []
        for i_dim in range(self.n_dims):
            if (self.n_dims-1-i_dim) == i_bin_reduc:
                list_dim.append([slice(None)])
            else:
                list_dim.append(list(range(self.dim_bin[self.n_dims-1-i_dim])))
                new_dim_bin.append(self.dim_bin[i_dim])
        
        # Reduce pdf
        new_pdf = np.sum(self.pdf, axis=i_bin_reduc)
        
        # Then compute the reduced statistics
        # Reduce min
        new_min = np.min(self.data_min, axis=i_bin_reduc)
        
        # Reduce max
        new_max = np.max(self.data_max, axis=i_bin_reduc)
        
        # Reduce mean
        new_mean = np.sum(self.data_mean*self.pdf, axis=i_bin_reduc)/new_pdf
        if len(new_mean.shape)!=0:
            new_mean[new_pdf == 0] = 0
        
        
        # Create a new Multistat object by copying the original object and changing its attributes
        # Updating dimension attributes
        new_MultiStat = copy.deepcopy(self)
        new_MultiStat.n_dims = self.n_dims-1
        new_MultiStat.list_bin = [self.list_bin[i] for i in range(self.n_dims) if i != i_bin_reduc]
        new_MultiStat.name_bin = [self.name_bin[i] for i in range(self.n_dims) if i != i_bin_reduc]
        new_MultiStat.dim_bin = [self.dim_bin[i] for i in range(self.n_dims) if i != i_bin_reduc]
        # Update the multiplicative n_dim_prod for later use
        new_MultiStat.updateNProd()
        
        if new_MultiStat.verbose == 1:
            new_MultiStat.printGlobalInformation()
        
        # Update data
        new_MultiStat.pdf = new_pdf
        new_MultiStat.data_min = new_min
        new_MultiStat.data_max = new_max
        new_MultiStat.data_mean = new_mean
        
        # Stop timer
        reduc_data = time.time() - reduc_data
        
        if new_MultiStat.verbose == 1:
            new_MultiStat.printPdfInformation()
            new_MultiStat.printDataInformation()
            print("It took a total of {:.5f}s to reduce data by one dimension".format(reduc_data))
        
        return new_MultiStat
    
    def reduceToNDim(self, list_var):
        # Return a Multistats object reduced to variables asked in list_var
        
        # Check that MultiStat object is big enough for this reduction
        n_dims = len(list_var)
        if self.n_dims <= n_dims:
            raise Exception("This MultiStat object has not enough dimension for the reduction to {}".format(list_var))
        
        # Check that asked variables are in the list of bins
        name_bins = [self.list_bin[i].name_var for i in range(self.n_dims)]
        for name_bin_keep in list_var:
            if name_bin_keep not in name_bins:
                raise Exception("{} is not in the bins".format(name_bin_keep))
        
        # Complementary list of given list_var
        list_reduc = list(set(self.name_bin)-set(list_var))
        
        # Reduce the remaining dimensions one by one
        reduced_MS = copy.deepcopy(self)
        for name_bin_reduc in list_reduc:
            reduced_MS = reduced_MS.reduceSingleDimension(name_bin_reduc)
        
        return reduced_MS
    
    ### Filter functions
    def filterSingleDimension(self, field, valid_range, type_filter="constant", param=None):
        # Remove the data not verifying the condition: 'field' value not in 'valid_range'
        # field must be one of the variable in the kernel files
        # valid_range indicate a valid range for field value
        # Find which rows verify the condition:
        ## type_filter indicates the type of filtering applied:
        ### constant: remove datapoints with values of var outside of [lo, hi]
        ### interpolation: remove datapoints with values outside of [lo_interp(param), hi_interp(param)]
        if self.verbose == 1:
            print("\n|-------------------- Filtering data in dimension {} --------------------|".format(name_bin_filter))
            print("Range of valid values: {}".format(condition_ranges))
            
        # Start timer
        reduc_data = time.time()
        
        # Field check and get the bins values
        name_bins = [self.list_bin[i].name_var for i in range(self.n_dims)]
        if field in name_bins:
            i_bin_f = name_bins.index(field)
            bin_value_field = self.list_bin[i_bin_f].value_bin
        else:
            raise Exception("Filter {} is not an available field from the kernel files".format(field))
            
        if param!=None:
            if param in name_bins:
                i_bin_p = name_bins.index(param)
                bin_value_param = self.list_bin[i_bin_p].value_bin
            else:
                raise Exception("Parameter {} is not an available field from the kernel files".format(param))
                
        # Filtering
        filter_pdf = copy.deepcopy(self.pdf)
        filter_min = copy.deepcopy(self.data_min)
        filter_max = copy.deepcopy(self.data_max)
        filter_mean = copy.deepcopy(self.data_mean)
        if type_filter=="constant":
            lo = valid_range[0]
            hi = valid_range[1]
            for i_f in range(len(bin_value_field)):
                if (bin_value_field[i_f] < lo or bin_value_field[i_f] > hi):
                    # Modify data to remove it
                    index = tuple([i_f if i==i_bin_f else slice(None) for i in range(self.n_dims)])
                    filter_pdf[index] = np.zeros(self.pdf[index].shape)
                    filter_min[index] = np.ones(self.pdf[index].shape)*(1e38)
                    filter_max[index] = np.ones(self.pdf[index].shape)*(-1e38)
                    filter_mean[index] = np.zeros(self.pdf[index].shape)
        elif type_filter=="function":
            for i_p in range(len(bin_value_param)):
                lo = valid_range[0](bin_value_param[i_p])
                hi = valid_range[1](bin_value_param[i_p])
                for i_f in range(len(bin_value_field)):
                    if not(bin_value_field[i_f] > lo and bin_value_field[i_f] < hi):
                        # Modify data to remove it
                        index = tuple([i_f if i==i_bin_f else i_p if i==i_bin_p else slice(None) for i in range(self.n_dims)])
                        filter_pdf[index] = np.zeros(self.pdf[index].shape)
                        filter_min[index] = np.ones(self.pdf[index].shape)*(1e38)
                        filter_max[index] = np.ones(self.pdf[index].shape)*(-1e38)
                        filter_mean[index] = np.zeros(self.pdf[index].shape)
        else:
            raise Exception("Filter type {} is not an available filter".format(filter_type))
            
        # Create a new filtered object
        filter_MS = copy.deepcopy(self)
        
        # Update data
        filter_MS.pdf = filter_pdf
        filter_MS.data_min = filter_min
        filter_MS.data_max = filter_max
        filter_MS.data_mean = filter_mean
        
        # Stop timer
        reduc_data = time.time() - reduc_data
        
        if filter_MS.verbose == 1:
            filter_MS.printPdfInformation()
            filter_MS.printDataInformation()
            print("It took a total of {:.5f}s to filter data".format(reduc_data))
        
        return filter_MS
    
    # |-----------------------------------|
    # |          Graphical plots          |
    # |-----------------------------------|
    def plot1d(self, ax, type_stat="pdf", is_log=False,
               color="black", linestyle="-", linewidth=1, label=None):
        # Check dimension is 1
        if self.n_dims != 1:
            raise Exception("plot1d plot is available only for MultiStat in 1D")
        
        # Plot 1d stat
        X = self.list_bin[0].value_bin
        if type_stat == "pdf":
            sol = self.pdf
        elif type_stat == "mean":
            sol = self.data_mean
        elif type_stat == "min":
            sol = self.data_min
        elif type_stat == "max":
            sol = self.data_max
        
        if is_log:
            ax.plot(X, np.log10(sol), color=color, linestyle=linestyle, linewidth=linewidth, label=label, zorder=10)
        else:
            ax.plot(X, sol, color=color, linestyle=linestyle, linewidth=linewidth, label=label, zorder=10)
        # Some formatting
        ax.set_xlabel(self.name_bin[0])
        return
    
    def plot1dfromNd(self, ax, var_1d, type_stat="mean", filtering=(None, None), is_log=False, 
                     color="black", linestyle="-", linewidth=1, label=None):
        # given a MultiStat of any dimension, plot the 1d solution reduced on the right quantities and the right filtering
        
        # 1) Filtering
        print(filtering)
        if filtering[0] in self.name_bin:
            # filter field is valid
            filter_var = filtering[0]
            filter_range = filtering[1]
            MS = self.filterSingleDimension(filter_var, filter_range)
        else:
            # No filtering applied
            MS = self
        
        # 2) Reduce to 1d quantity of interest
        if var_1d in self.name_bin:
            MS = MS.reduceToNDim([var_1d])
        else:
            raise ValueError("{} is not a valid variable in bins".format(var_1d))
        
        # 3) plot and format
        print(MS.var)
        MS.plot1d(ax, type_stat="mean", color=color, is_log=is_log,
                  linestyle=linestyle, linewidth=linewidth, label=label)
        #ax.set_title("{} {}".format(type_stat, self.var))
        
        return
    
    def contourLine2D(self, ax, values, type_stat="pdf"):
        # Plot the isocontour lines of the values in values in a 2D plot (must be 2 dimensional)
        
        # Check dimension is 2
        if self.n_dims != 2:
            raise Exception("plot2d is available only for MultiStat in 2D")
        
        # load variable abnd bins information for the lines
        x = self.list_bin[0].value_bin
        y = self.list_bin[1].value_bin
        X, Y = np.meshgrid(x, y)
        if type_stat == "pdf":
            Z = np.transpose(self.getCleanPdf())
        elif type_stat == "mean":
            Z = np.transpose(self.getCleanMean())
        elif type_stat == "min":
            Z = np.transpose(self.getCleanMin())
        elif type_stat == "max":
            Z = np.transpose(self.getCleanMax())
        
        cont = ax.contour(X, Y, Z, levels=values, color="black", linestyle="--")
        
        return cont
    
    def plot2d(self, fig, ax, type_stat="pdf", cmap="jet", norm=None, pdf_threshold=1):
        # Plot the 2D graph of loaded variable against the 2 variables in the bins (must be 2 dimensional)
        
        # Check dimension is 2
        if self.n_dims != 2:
            raise Exception("plot2d is available only for MultiStat in 2D")
        
        # Variables redefinition for clarity
        main_var = self.var
        x_var = self.name_bin[0]
        if self.list_bin[0].is_log:
            x_min = log10(self.list_bin[0].min_var)
            x_max = log10(self.list_bin[0].max_var)
        else:
            x_min = self.list_bin[0].min_var
            x_max = self.list_bin[0].max_var
        y_var = self.name_bin[1]
        if self.list_bin[1].is_log:
            y_min = log10(self.list_bin[1].min_var)
            y_max = log10(self.list_bin[1].max_var)
        else:
            y_min = self.list_bin[1].min_var
            y_max = self.list_bin[1].max_var
        
        extent = [x_min, x_max, y_min, y_max]
        
        # 2d plots
        if type_stat=="pdf":
            if norm==None:
                norm = LogNorm(1, np.nanmax(self.getCleanPdf()))
            im = ax.imshow(np.transpose(self.getCleanPdf(pdf_threshold)), interpolation="nearest",
                           extent=extent, norm = norm, origin="lower", 
                           cmap=cmap, aspect = "auto")
        elif type_stat=="mean":
            if norm==None:
                norm = colors.Normalize()
            im = ax.imshow(np.transpose(self.getCleanMean()), interpolation="nearest",
                           extent=extent, norm = norm, origin="lower", 
                           cmap=cmap, aspect = "auto")
        elif type_stat=="min":
            if norm==None:
                norm = colors.Normalize()
            im = ax.imshow(np.transpose(self.getCleanMin()), interpolation="nearest",
                           extent=extent, norm = norm, origin="lower", 
                           cmap=cmap, aspect = "auto")
        elif type_stat=="max":
            if norm==None:
                norm = colors.Normalize()
            im = ax.imshow(np.transpose(self.getCleanMax()), interpolation="nearest",
                           extent=extent, norm = norm, origin="lower", 
                           cmap=cmap, aspect = "auto")
        else:
            raise("{} is not an option for the possible plots".format(type_stat))
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        # Additional plot customization
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        
        return
    
    def marginalPlot(self, log=False, norm=None):
        # for this plot, the MultiStat object need to be in 2D
        if self.n_dims != 2:
            raise Exception("Marginal plot is available only for MultiStat in 2D")
        
        fig = plt.figure(figsize=(12, 12), dpi = 200)
        
        # Variables redefinition for clarity
        main_var = self.var
        x_var = self.name_bin[0]
        if self.list_bin[0].is_log:
            x_min = log10(self.list_bin[0].min_var)
            x_max = log10(self.list_bin[0].max_var)
        else:
            x_min = self.list_bin[0].min_var
            x_max = self.list_bin[0].max_var
        y_var = self.name_bin[1]
        if self.list_bin[1].is_log:
            y_min = log10(self.list_bin[1].min_var)
            y_max = log10(self.list_bin[1].max_var)
        else:
            y_min = self.list_bin[1].min_var
            y_max = self.list_bin[1].max_var
            
        print(x_min, x_max)
        print(y_min, y_max)
        
        # definitions for the axes
        left, width = 0.1, 0.70
        bottom, height = 0.1, 0.70
        spacing = 0.05

        rect_main = [left, bottom, width-spacing, height-spacing]
        rect_histx = [left, bottom + height + spacing, width-spacing, 1-(height+spacing)]
        rect_histy = [left + width + spacing, bottom, 1-(width+spacing), height-spacing]
        rect_jpdf = [left + width + spacing, bottom + height + spacing, 1-(width+spacing), 1-(height+spacing)]
        
        # manage norm
        if norm==None:
            norm=colors.Normalize()
        
        ## main mean or max plot
        ax = fig.add_axes(rect_main)
        extent = [x_min, x_max, y_min, y_max]
        im_mean = ax.imshow(np.transpose(self.getCleanMean()), extent=extent, 
                            norm = norm, origin="lower", 
                            cmap="jet", aspect = "auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_mean, cax=cax, orientation='vertical')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

            # 1D reduced x plot for pdf and mean/max
        ax_x = fig.add_axes(rect_histx, sharex=ax)
        MS_1d_x = self.reduceToNDim([x_var])
        if self.list_bin[0].is_log:
            x_values = np.log10(MS_1d_x.getValueBin(x_var))
        else:
            x_values = MS_1d_x.getValueBin(x_var)
        ax_x.plot(x_values, MS_1d_x.getCleanPdf(), 
                  label="pdf", color="grey", linestyle = "--")
        twin_x = ax_x.twinx()
        twin_x.plot(x_values, MS_1d_x.getCleanMean(),
                    label="mean", color="black", linestyle = "-")
        ax_x.set_ylabel("pdf")
        twin_x.set_ylabel(main_var)
        ax_x.set_yscale("log")

            # 1D reduced y plot for pdf and mean/max
        ax_y = fig.add_axes(rect_histy, sharey=ax)
        MS_1d_y = self.reduceToNDim([y_var])
        if self.list_bin[1].is_log:
            y_values = np.log10(MS_1d_y.getValueBin(y_var))
        else:
            y_values = MS_1d_y.getValueBin(y_var)
        ax_y.plot(MS_1d_y.getCleanPdf(), y_values,
                  label="pdf", color="grey", linestyle = "--")
        twin_y = ax_y.twiny()
        twin_y.plot(MS_1d_y.getCleanMean(), y_values,
                    label="mean", color="black", linestyle = "-")
        ax_y.set_xlabel("pdf")
        twin_y.set_xlabel("\n"+ main_var)
        ax_y.set_xscale("log")

            # JPDF on the remaining square
        ax_jpdf = fig.add_axes(rect_jpdf)
        im_pdf = ax_jpdf.imshow(np.transpose(self.getCleanPdf()), extent=extent, 
                                norm = colors.LogNorm(1, None), origin="lower", 
                                cmap="jet", aspect = "auto")
        divider = make_axes_locatable(ax_jpdf)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_pdf, cax=cax, orientation='vertical')

        fig.tight_layout()
        plt.show()
        return fig