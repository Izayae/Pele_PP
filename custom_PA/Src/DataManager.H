#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <Slice2d.H>
#include <MultiStat.H>
#include <HeaderReader.H>

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <InputParam.H>
#include <ConditionalParam.H>
#include <AMReX_DataServices.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <PltFileManager.H>
#include <AMReX_VisMF.H>
#include <utility>

#include <string>
#include <iostream>
#include <cstdlib>

class DataManager
{
public:
    // Constructor
    explicit DataManager(InputParam input, 
                         ConditionalParam conditions,
                         const int i_plt,
                         const bool load_data);
    
    // Initialize geometry from AmrData
    void geomInit(const int i_plt);
    
    // -------------------- Data loading -------------------- //
    // Selected variables management
    void checkVariables(const amrex::Vector<std::string> plot_var_names,
                        const amrex::Vector<std::string> selected_var_names,
                        amrex::Vector<std::string>& checked_name_vars,
                        int& checked_n_comp);
                        
    void manageVariables(const int i_plt,
                         amrex::Vector<amrex::Vector<std::string>>& names_multi,
                         amrex::Vector<int>& n_comp_selected_multi);
                             
    void reduceMulti(const amrex::Vector<amrex::Vector<std::string>>& names_multi, 
                     amrex::Vector<std::string>& name_vars,
                     const amrex::Vector<int>& n_comp_selected_multi,
                     int& n_comp_selected,
                     amrex::Vector<amrex::Vector<int>>& dest_fill_comps_multi);
    
    // Load selected data from separated plotfiles to a global structure
    void loadMultiData(const int i_plt,
                       const amrex::Vector<amrex::Vector<std::string>>& names_multi,
                       const amrex::Vector<amrex::Vector<int>>& dest_fill_comps_multi);
    
    // Manage and compute the new quantities derived from plotfile data
    void manageNewQuantities();
    void addNewQuantities();
    
    // Load all data (selected + computed quantities)
    void loadData(const int i_plt);
    
    // Utility functions
    int getIdVar(const std::string name_var);
    
    // ------------------- Data conditioning ------------------- //
    void printConditions();
    int checkConditions(int i, int j, int k, amrex::Array4<amrex::Real>& fab);
    
    // -------------------- Write plotfile -------------------- //
    void writePlotFile();
    
    // -------------------- Data processing -------------------- //
    // common operations
    amrex::MultiFab regridUniform(amrex::Geometry& uniform_geom);
    void regridData(int level,
                    amrex::Geometry& lev_geom,
                    amrex::MultiFab& new_data);
    amrex::Vector<amrex::MultiFab> getGradient(const std::string name_var);
    
    // statistics
    amrex::Real getMin(std::string name_var);
    amrex::Real getMax(std::string name_var);
    amrex::Real getMean(std::string name_var);
    void extractAllStats(MultiStat& stat);
    void extractSingleStat(const int i_bin_comb, MultiStat& stat);
    int calcGlobalId(const int i_bin_comb, MultiStat& stat,
                     const int i, const int j, const int k, const amrex::Array4<const amrex::Real> arr_var);
    
    // write conditions in case directory
    void writeConditions();
    
    // extracting datapoints to file for kernels
    void writeDatapointsToFile();
    void pointsExtracted(int& n_extracted, 
                         const amrex::MultiFab& uniform_data);
    void extractDatapoints(const int n_extracted,
                           const amrex::MultiFab& uniform_data, 
                           amrex::Vector<amrex::Real>& extracted_var_data);
    void writeExtractHeader(const int n_extracted);
    void writeExtractToBinary(const int n_extracted, 
                              const amrex::Vector<amrex::Real>& extracted_var_data);
    //void writeExtractToAscii();
    
    // ---------------------- Data display --------------------- //
    amrex::Vector<std::string> getNameVars() {return name_vars; };
    std::string getPltFile() {return plt_file; };
    int getNLev() {return n_levs; };
    amrex::Real getTime() {return time; };
    amrex::Vector<int> getNumDim(int lev) {return {num_dim[lev].length(0), num_dim[lev].length(1), num_dim[lev].length(2)}; };
    amrex::Vector<amrex::Real> getRealDim() {return {real_dim.length(0), real_dim.length(1), real_dim.length(2)}; };
    
    // get functions
    amrex::MultiFab& getData(int lev) {return DM_data[lev]; };
    amrex::Vector<amrex::MultiFab*> getDataPtr() {return amrex::GetVecOfPtrs(DM_data); };
    amrex::Vector<const amrex::MultiFab*> getDataConstPtr() {return amrex::GetVecOfConstPtrs(DM_data); };
    amrex::Geometry& getGeom(int lev) {return DM_geoms[lev]; };
    amrex::BoxArray& getGrid(int lev) {return DM_grids[lev]; };
    amrex::DistributionMapping& getDmap(int lev) {return DM_dmaps[lev]; };
    
    // Modify variable names
    void varRename (amrex::Vector<std::string> new_names) {name_vars=new_names;};
    
    // return an entire 
    
protected:
    // Input parameters
    InputParam input;
    
    // Conditional parameters
    ConditionalParam conditions;
    
    // PltFileManager for the original data (not yet added)
    // pele::physics::pltfilemanager::PltFileManager*
    
    // Data stuctures
    amrex::Vector<amrex::MultiFab> DM_data;
    amrex::Vector<amrex::Geometry> DM_geoms;              // Geometries at each level
    amrex::Vector<amrex::IntVect> DM_ref_ratio;                      // refinement ratio between each level
    amrex::Vector<amrex::BoxArray> DM_grids;              // BoxArray at each level
    amrex::Vector<amrex::DistributionMapping> DM_dmaps;   // DistributionMap at each level
    
    // Metadata about the current pltfile
    std::string plt_file;                     // pltfile name
    amrex::Vector<std::string> name_vars;     // list of variable names loaded
    int n_levs{0};                            // number of levels
    amrex::Real time{0.0};                    // Simulation time
    amrex::Vector<amrex::Box> num_dim;        // Domain numerical dimension at each level
    amrex::RealBox real_dim;                  // Domain real dimension
    int n_comp_selected = 0;                  // Number of original quantities selected by user
    int n_comp_total = 0;                     // Number of total quantities loaded
};

#endif