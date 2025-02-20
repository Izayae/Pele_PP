#ifndef SLICE2D_H
#define SLICE2D_H

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


class Slice2d
{
public:
    // Constructor
    explicit Slice2d(const std::string path_output, const std::string plt_file,
                     amrex::MultiFab& data, const amrex::Geometry geom, const amrex::Vector<std::string> name_vars,
                     const int level, const std::string axis, const amrex::Real value, const amrex::Real time);
    
    void extractSlice(amrex::MultiFab& data, const int i_comp);
    
    // ------------------- Data processing -------------------- //
    amrex::Real getMin(const std::string name_var);
    amrex::Real getMin(const int i_comp);
    amrex::Real getMax(const std::string name_var);
    amrex::Real getMax(const int i_comp);
    amrex::Real getMean(const std::string name_var);
    amrex::Real getMean(const int i_comp);
    
    // ------------------- Data input/output -------------------- //
    std::string remove_parenthesis(std::string name);
    void sliceHeader(const int i_comp);
    void sliceToAscii(const int i_comp);
    void sliceToBinary(const int i_comp);
    void writeSingleSlice(const std::string type_data, const int i_comp);
    void writeAllSlice(const std::string type_data);
    
    // ------------------- Utility -------------------- //
    int getIdVar(const std::string name_var);
    
    int getNComp() {return n_comp;};
    std::string getPltFile() {return plt_file;};
    amrex::Vector<std::string> getNameVars() {return name_vars;};
    int getLev() {return level;};
    amrex::Real getTime() {return time;};
    amrex::Vector<int> getNumDim(int lev) {return num_dim;};
    amrex::Vector<amrex::Real> getRealDim() {return real_dim;};
    
protected:
    // Slice parameters
    int level = 0;                           // level in the original file of the slice
    std::string axis = "x";                  // axis for the normal to the slice
    amrex::Real value = 0.5;                 // value for the position of the slice on the axis (relative to axis length)
    int i_value = 0;                         // index of the layer of cells corresponding to value
    amrex::Vector<std::string> name_vars;    // name of the variables
    
    // Slice data
    amrex::Vector<amrex::Vector<amrex::Real>> slice_data;   // first vector is for variables, second vector for 2d array
    
    // Slice information
    std::string plt_file;                    // name of the original plt (plt?????)
    amrex::Real time = 0.0;                  // time of the simulation
    amrex::Vector<int> num_dim;              // numerical dimension of the slice
    amrex::Vector<amrex::Real> real_dim;     // real dimensions of the slice
    int n_comp = 0;                          // number of components in the slice
    std::string slice_name;                  // name of the slice written to file
    std::string path_output;                 // directory where the slices are written
    
};

#endif