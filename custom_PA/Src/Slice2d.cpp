#include <Slice2d.H>

#include <DataManager.H>

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_DataServices.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

#include <AMReX_VisMF.H>
#include <PltFileManagerBCFill.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <utility>

#include <AMReX_BLFort.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLABecLaplacian.H>

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <sstream>

#include <chrono>
using namespace std::chrono;

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 2)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

Slice2d::Slice2d(const std::string path_output, const std::string plt_file,
                 amrex::MultiFab& data, const amrex::Geometry geom, const amrex::Vector<std::string> name_vars,
                 const int level, const std::string axis, const amrex::Real value, const amrex::Real time):
                 path_output{path_output}, plt_file{plt_file}, level{level}, 
                 axis{axis}, value{value}, name_vars{name_vars}, time{time}
{   
    // Extract information about the slice
    n_comp = name_vars.size();
    num_dim.resize(2);
    real_dim.resize(2);
    if (axis == "x") {
        num_dim = {geom.Domain().length(1), geom.Domain().length(2)};
        real_dim = {geom.ProbLength(1), geom.ProbLength(2)};
        i_value = round(value*geom.Domain().length(0));
    } else if (axis == "y") {
        num_dim = {geom.Domain().length(0), geom.Domain().length(2)};
        real_dim = {geom.ProbLength(0), geom.ProbLength(2)};
        i_value = round(value*geom.Domain().length(1));
    } else if (axis == "z") {
        num_dim = {geom.Domain().length(0), geom.Domain().length(1)};
        real_dim = {geom.ProbLength(0), geom.ProbLength(1)};
        i_value = round(value*geom.Domain().length(2));
    } else {
        amrex::Print() << axis << " not supported as an axis\n";
        amrex::Error();
    }
    amrex::Print() << "slice of dimension: " << num_dim[0] << "x" << num_dim[1] << "\n";
    slice_name = axis + "_" + std::to_string(value) + "_" + std::to_string(level);            // name is <axis>_<value>_<level>
    
    // Extract the slice actual data
    slice_data.resize(n_comp);
    for (int i_comp=0; i_comp<n_comp; i_comp++){
        //amrex::Print() << "Extracting slice for " << name_vars[i_comp] << "\n";
        //auto start_extract = high_resolution_clock::now();
        extractSlice(data, i_comp);
        /*amrex::Print() << "Extracting slice took "
                       <<  duration_cast<microseconds>(high_resolution_clock::now() - start_extract).count() << "mus\n";*/
        
        // Reduce the data to the main process
        //auto start_reduce = high_resolution_clock::now();
        amrex::Real *slice_ptr = slice_data[i_comp].dataPtr();
        amrex::ParallelDescriptor::ReduceRealSum(slice_ptr, slice_data[i_comp].size(), 
                                                 amrex::ParallelDescriptor::IOProcessorNumber());
        /*amrex::Print() << "Reducing slice took "
                       <<  duration_cast<microseconds>(high_resolution_clock::now() - start_reduce).count() << "mus\n";*/
    }
}

void
Slice2d::extractSlice(amrex::MultiFab& data, const int i_comp) {
    // Fill slice with 2D data
    slice_data[i_comp].resize(num_dim[0]*num_dim[1]);
    
    for (amrex::MFIter mfi(data, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // kind of inefficient way with ParallelFor iterate over all axis
        /*
        //amrex::Real *slice_ptr=slice_data[i_comp].dataPtr();
        const amrex::Box& bx = mfi.validbox();
        if (bx.ok()) {
            const auto& data_arr = data.const_array(mfi, i_comp);
            
            amrex::ParallelFor(bx, [=]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {   
                if (axis == "x" and i==i_value) { 
                    slice_data[i_comp][k*num_dim[0]+j] = data_arr(i_value, j, k);
                } else if (axis == "y" and j==i_value) {
                    slice_data[i_comp][k*num_dim[0]+i] = data_arr(i, i_value, k);
                } else if (axis == "z" and k==i_value) {
                    slice_data[i_comp][j*num_dim[0]+i] = data_arr(i, j, i_value);
                }
            });
        }*/
        
        // Attempt at more efficient way to do that by skipping the irrelevant dimension
        const amrex::Box& bx = mfi.validbox();
        if (bx.ok()) {
            const auto& data_arr = data.const_array(mfi, i_comp);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);

            if (axis == "x" and (i_value >= lo.x and i_value <= hi.x)) { 
                for (int k = lo.z; k <= hi.z; ++k) {
                    for (int j = lo.y; j <= hi.y; ++j) {
                        slice_data[i_comp][k*num_dim[0]+j] = data_arr(i_value, j, k);
                    }
                }
            } else if (axis == "y" and (i_value >= lo.y and i_value <= hi.y)) { 
                for (int k = lo.z; k <= hi.z; ++k) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        slice_data[i_comp][k*num_dim[0]+i] = data_arr(i, i_value, k);
                    }
                }
            } else if (axis == "z" and (i_value >= lo.z and i_value <= hi.z)) { 
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        slice_data[i_comp][j*num_dim[0]+i] = data_arr(i, j, i_value);
                    }
                }
            }
        }
    } // end for
}

// ------------------------------------------------------ //
// ------------------- Data processing ------------------ //
// ------------------------------------------------------ //

amrex::Real
Slice2d::getMin(const std::string name_var) {
    // Get the right slice
    int id_var = getIdVar(name_var);
    amrex::Vector<amrex::Real> slice_comp = slice_data[id_var];
    
    // Compute stat
    amrex::Real min_comp = std::numeric_limits<float>::max();;
    for (int j=0; j<num_dim[1]; j++) {
        for (int i=0; i<num_dim[0]; i++) {
            min_comp = std::min(min_comp, slice_comp[j*num_dim[0]+i]);
        }
    }
    
    return min_comp;
}

amrex::Real
Slice2d::getMin(const int i_comp) {
    // Get the right slice
    amrex::Vector<amrex::Real> slice_comp = slice_data[i_comp];
    
    // Compute stat
    amrex::Real min_comp = std::numeric_limits<float>::max();;
    for (int j=0; j<num_dim[1]; j++) {
        for (int i=0; i<num_dim[0]; i++) {
            min_comp = std::min(min_comp, slice_comp[j*num_dim[0]+i]);
        }
    }
    
    return min_comp;
}

amrex::Real
Slice2d::getMax(const std::string name_var) {
    // Get the right slice
    int id_var = getIdVar(name_var);
    amrex::Real *slice_comp = slice_data[id_var].dataPtr();
    
    // Compute stat
    amrex::Real max_comp = std::numeric_limits<float>::min();;
    for (int j=0; j<num_dim[1]; j++) {
        for (int i=0; i<num_dim[0]; i++) {
            max_comp = std::max(max_comp, slice_comp[j*num_dim[0]+i]);
        }
    }
    
    return max_comp;
}

amrex::Real
Slice2d::getMax(const int i_comp) {
    // Get the right slice
    amrex::Real *slice_comp = slice_data[i_comp].dataPtr();
    
    // Compute stat
    amrex::Real max_comp = std::numeric_limits<float>::min();;
    for (int j=0; j<num_dim[1]; j++) {
        for (int i=0; i<num_dim[0]; i++) {
            max_comp = std::max(max_comp, slice_comp[j*num_dim[0]+i]);
        }
    }
    
    return max_comp;
}

amrex::Real
Slice2d::getMean(const std::string name_var) {
    // Get the right slice
    int id_var = getIdVar(name_var);
    amrex::Real *slice_comp = slice_data[id_var].dataPtr();
    
    // Compute stat
    amrex::Real mean_comp = 0.0;
    for (int j=0; j<num_dim[1]; j++) {
        for (int i=0; i<num_dim[0]; i++) {
            mean_comp += slice_comp[j*num_dim[0]+i];
        }
    }
    mean_comp = mean_comp/(num_dim[0]*num_dim[1]);
    
    return mean_comp;
}

amrex::Real
Slice2d::getMean(const int i_comp) {
    // Get the right slice
    amrex::Real *slice_comp = slice_data[i_comp].dataPtr();
    
    // Compute stat
    amrex::Real mean_comp = 0.0;
    for (int j=0; j<num_dim[1]; j++) {
        for (int i=0; i<num_dim[0]; i++) {
            mean_comp += slice_comp[j*num_dim[0]+i];
        }
    }
    mean_comp = mean_comp/(num_dim[0]*num_dim[1]);
    
    return mean_comp;
}

// ------------------------------------------------------ //
// -------------------- input/output -------------------- //
// ------------------------------------------------------ //
std::string
Slice2d::remove_parenthesis(std::string name) {
    // Directory names cannot contain parenthesis, so this code rewrite names (species with Y(...)) without the parenthesis
    std::string new_name = name;
    std::size_t found_sp = name.find("Y(");
    std::string species;
    if (found_sp < 10000){
        std::copy(name.begin()+2, name.end()-1, std::back_inserter(species));
        new_name = "Y_" + species;
    }
    
    std::size_t found_a = name.find("(S)");
    std::string annoying;
    if (found_a < 10000){
        std::copy(name.begin(), name.end()-4, std::back_inserter(annoying));
        new_name = annoying + "_S";
    }
    return new_name;
}

void
Slice2d::writeAllSlice(const std::string type_file) {
    for (int i_comp=0; i_comp<n_comp; i_comp++) {
        writeSingleSlice(type_file, i_comp);
    }
}

void
Slice2d::writeSingleSlice(const std::string type_file, const int i_comp) {
    // Write slice to file only on IOProcessor 
    // (don't know how to share the writing load between processes)
    if (amrex::ParallelDescriptor::IOProcessor()){
        // Write the header file in ascii format
        sliceHeader(i_comp);

        // Write the data with the specified format
        if (type_file == "ascii") {
            sliceToAscii(i_comp);
        } else if (type_file == "binary") {
            sliceToBinary(i_comp);
        } else {
            amrex::Print() << type_file << " is not supported\n";
            amrex::Error();
        }
    }
}

void
Slice2d::sliceHeader(const int i_comp) {
    // Create the directory for the full path to the files for the variable
    std::string full_path = path_output + "/" + plt_file + "/" + axis + "_" + to_string_with_precision(value, 2) +
                            "_" + std::to_string(level) + "/" + 
                            remove_parenthesis(name_vars[i_comp]) + "/";
    std::filesystem::create_directories(full_path);
    
    // Open the file
    try {
        //open file for writing
        std::ofstream fw(full_path + "header.txt", std::ofstream::out);
        //check if file was successfully opened for writing
        if (fw.is_open())
        {
            //amrex::Print() << "Writing header file to " << full_path << "\n";
            // Write information in header file
            fw << plt_file << "\n";
            fw << time << "\n";
            fw << name_vars[i_comp] << "\n";
            fw << axis << "\n";
            fw << value << "\n";
            fw << level << "\n";
            fw << num_dim[0] << " " << num_dim[1] << "\n";
            fw << real_dim[0] << " " << real_dim[1] << "\n";
            fw << getMin(i_comp) << " " << getMax(i_comp) << "\n";
            
            fw.close();
        }
        else amrex::Print() << "Problem with opening file\n";
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
}

void
Slice2d::sliceToAscii(const int i_comp) {
    // Create the directory for the full path to the files for the variable
    std::string full_path = path_output + "/" + axis + "_" + to_string_with_precision(value, 2) +
                            "_" + std::to_string(level) + "/" + 
                            remove_parenthesis(name_vars[i_comp]) + "/" + plt_file + "/";
    
    // Open ascii file
    try {
        FILE *file;
        file = fopen((full_path + "data.txt").c_str(), "w");
        if (file != NULL)
        {
            //amrex::Print() << "Writing ascii file to " << full_path << "\n";
        } else {
            amrex::Print() << "File not opened, crash incoming\n";
        }
        for (int j=0; j<num_dim[1]; j++) {
                for (int i=0; i<num_dim[0]; i++) {
                    fprintf(file,"%e ", slice_data[i_comp][j*num_dim[0]+i]);
            }
            fprintf(file,"\n");
        }

        fclose(file);
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
}

void
Slice2d::sliceToBinary(const int i_comp) {
    // Create the directory for the full path to the files for the variable
    std::string full_path = path_output + "/" + plt_file + "/" + axis + "_" + to_string_with_precision(value, 2) +
                            "_" + std::to_string(level) + "/" + 
                            remove_parenthesis(name_vars[i_comp]) + "/";
    
    // Open binary file
    try {
        std::ofstream binfile;
        binfile.open(full_path + "data.bin", std::ios::binary);
        if (binfile.is_open())
        {
            //amrex::Print() << "Writing binary file to " << full_path << "\n";
        } else {
            amrex::Print() << "Binary file not opened, crash incoming\n";
            amrex::Error();
        }
        // Converting and writing data to binary
        // ----- Data ----- //
        for (int j=0; j<num_dim[1]; j++) {
            for (int i=0; i<num_dim[0]; i++) {
                binfile.write(reinterpret_cast<const char*>(&slice_data[i_comp][j*num_dim[0]+i]), 
                              sizeof(slice_data[i_comp][j*num_dim[0]+i]));
            }
        }

        binfile.close();
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
}

// ---------------------------------------------- //
// ------------------- Utility ------------------ //
// ---------------------------------------------- //

int
Slice2d::getIdVar(const std::string name_var) {
    int id_var;
    auto itr = std::find(name_vars.begin(), name_vars.end(), name_var);
    if (itr != name_vars.cend()){
        id_var = std::distance(name_vars.begin(), itr);
    } else {
        std::cout << "variable " << name_var << " not loaded" << "\n";
        amrex::Error();
    }
    return id_var;
}