#ifndef MultiStat_H
#define MultiStat_H

#include <AMReX_ParmParse.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_DataServices.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <utility>

#include <AMReX_BLFort.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLABecLaplacian.H>

#include <string>
#include <iostream>
#include <cstdlib>

// Basic structure for a single bin
typedef struct Bin {
    std::string name_var;
    int n_bin;
    int is_log;
    int count_ext;            // Should value outside of min/max range be counted?
    amrex::Real min_var;
    amrex::Real max_var;
    amrex::Vector<amrex::Real> value_bin;
    
    Bin(std::string name, int n, int log,
        amrex::Real min, amrex::Real max, int ext)
    {
        // Initializer list doesn't work...
        name_var = name;
        n_bin = n;
        is_log = log;
        min_var = min;
        max_var = max;
        count_ext = ext;
        
        // Compute and fill the values of each bin
        value_bin.resize(n_bin);
        for (int i_bin=0; i_bin<n_bin; i_bin++){
            amrex::Real dbin;
            if (is_log==1){
                dbin = (log10(max_var)-log10(min_var))/(amrex::Real)n_bin;
                value_bin[i_bin] = min_var * pow(10, dbin*(0.5+(amrex::Real)i_bin));
            } else {
                dbin = (max_var-min_var)/(amrex::Real)n_bin;
                value_bin[i_bin] = min_var + dbin*(0.5+(amrex::Real)i_bin);
            }
        }
    }
    
    int getIdBin(const amrex::Real value) {
        // Find the index of the bin where the datapoint is situated
        // require later check if returned id is outside of the bin range
        int id_bin;
        if (is_log==1){
            id_bin = (int)(n_bin*(log10(value)-log10(min_var))/(log10(max_var)-log10(min_var)));
        } else {
            id_bin = (int)(n_bin*(value-min_var)/(max_var-min_var));
        }
        // Do we count the value if it's outside
        if (count_ext==1) {
            if (id_bin<0)      { id_bin=0; }
            if (id_bin>=n_bin) { id_bin=n_bin-1; }
        }
        
        return id_bin;
    }
} Bin;


// Class for multi-dimensional statistics
typedef struct MultiStat {
    // Constructor
    explicit MultiStat(const std::string path_output, const std::string plt_file,
                       const amrex::Vector<std::string> name_vars, const int level, const amrex::Real time);
    void readInputStat();                         // Read parameters for every combination of bins specified
    
    // Utility
    int getNCombination(const int i_bin_comb);    // Compute the product of number of bins in each dimension for this combination
    int getIdVar(const std::string name_var);
    std::string remove_parenthesis(std::string name);
    void statName(const int i_bin_comb);
    
    // Generate stats
    void initDataStructure(const int i_bin_comb); // Initialize data structure for the current combination
    
    // Manipulate the currently loaded data structure (not many ideas now, just for testing)
    int getGlobalNPoint();
    amrex::Real getGlobalMin(const int i_comp);
    amrex::Real getGlobalMin(const std::string name_comp);
    amrex::Real getGlobalMax(const int i_comp);
    amrex::Real getGlobalMax(const std::string name_comp);
    
    // Write data to file
    void writeStatToFile(const int i_bin_comb, const std::string type_file);
    void writeStatHeader(const int i_bin_comb);
    void writePdfToBinary();
    void writeStatToBinary(const int i_comp);
    void writeStatToAscii(const int i_comp);
    
    // General information for the statistics
    int n_dims;                                   // Number of dimensions for the statistics
    int n_bin_comb;                               // Number of bin combinations
    int n_comp;                                   // Number of components for the statistics
    amrex::Vector<std::string> name_vars;         // Name of each component for the statistics
    
    // Weight for weighted average (unity by default)
    std::string name_weight = "unity";
    
    // Bins list of parameters
    amrex::Vector<amrex::Vector<Bin>> bins;       // Vectors of bins object in each dimension and for each combination
    
    // Plotfile information
    int n_lev;
    std::string plt_file;                    // name of the original plt (plt?????)
    amrex::Real time = 0.0;                  // time of the simulation
    std::string stat_name;                   // name of the slice written to file
    std::string path_output;                 // directory where the slices are written
    
    // Data structures loaded one combination of bins at a time
    // first vector for each variable, second vector for multi-dimension bins
    amrex::Vector<amrex::Real> data_pdf;                   // n_dims-dimension probability density function
    amrex::Vector<amrex::Vector<amrex::Real>> data_min;    // min value
    amrex::Vector<amrex::Vector<amrex::Real>> data_max;    // max value
    amrex::Vector<amrex::Vector<amrex::Real>> data_mean;   // mean value
    amrex::Vector<amrex::Vector<amrex::Real>> data_std;    // std value
} MultiStat;

#endif