#include <MultiStat.H>

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

MultiStat::MultiStat(const std::string path_output, const std::string plt_file,
                     const amrex::Vector<std::string> name_vars, const int level, const amrex::Real time):
                     path_output{path_output}, plt_file{plt_file}, name_vars{name_vars},
                     n_lev{level}, time{time}, n_comp{name_vars.size()}
{
    amrex::Print() << "\n---------------------------------------------\n";
    amrex::Print() <<   "|   Managing multi-dimensional statistics   |\n";
    amrex::Print() <<   "---------------------------------------------\n";
    
    // Read input parameters for all bins combination and initialize data structures
    readInputStat();
    
    // Print some informations about the bins to validate initialization
    amrex::Print() << n_bin_comb << " combinations specified by the user with " << n_dims << " dimensions:\n";
    for (int i_bin_comb=0; i_bin_comb<n_bin_comb; i_bin_comb++) {
        amrex::Print() << "Combination " << i_bin_comb << ": ";
        for (int i_dim=0; i_dim<n_dims; i_dim++) {
            amrex::Print() << bins[i_bin_comb][i_dim].name_var << " ";
        }
        amrex::Print() << "\n";
    }
}

void
MultiStat::readInputStat() {
    // Parameter parser
    amrex::ParmParse pp;

    // Number of stats dimension
    pp.get("n_dims", n_dims);
    
    // Get number of combination specified with first bins combination
    std::string var = "bin_var_0";
    n_bin_comb = pp.countval(var.c_str());
    bins.resize(n_bin_comb);
    
    for (int i_bin_comb=0; i_bin_comb<n_bin_comb; i_bin_comb++) {
        //bins[i_bin_comb].resize(n_dims);
        for (int i_dim=0; i_dim<n_dims; i_dim++) {
            // Number of bins and verification of the arguments
            std::string var = "bin_var_";
            var += std::to_string(i_dim);

            // Check that all bins have the same number of possibility
            if (n_bin_comb != pp.countval(var.c_str())) {
                amrex::Print() << "Not the same number of possibility in bin " << i_dim << "as in bin 0\n";
                amrex::Error();
            }

            // Formatting of string for reading stat parameters and check number of parameter is correct
            std::string n = "n_bin_";
            std::string min = "min_var_";
            std::string max = "max_var_";
            std::string log = "is_log_";
            std::string ext = "count_ext_";
            n += std::to_string(i_dim);
            min += std::to_string(i_dim);
            max += std::to_string(i_dim);
            log += std::to_string(i_dim);
            ext += std::to_string(i_dim);
            if ((n_bin_comb != pp.countval(n.c_str()))   or
                (n_bin_comb != pp.countval(min.c_str())) or
                (n_bin_comb != pp.countval(max.c_str())) or
                (n_bin_comb != pp.countval(log.c_str())) )  {
                amrex::Print() << "Number of combinations should be: " << n_bin_comb << "\n";
                amrex::Print() << "n: " << pp.countval(n.c_str()) << "\n";
                amrex::Print() << "min: " << pp.countval(min.c_str()) << "\n";
                amrex::Print() << "max: " << pp.countval(max.c_str()) << "\n";
                amrex::Print() << "log: " << pp.countval(log.c_str()) << "\n";
                amrex::Abort("Wrong number of arguments for this bin\n");
            }

            // Create the bins for this combination
            std::string bin_var;
            int n_bin;
            int is_log;
            amrex::Real bin_min;
            amrex::Real bin_max;
            int count_ext=1;       // 1 by default
            pp.get(var.c_str(), bin_var, i_bin_comb);
            pp.get(n.c_str(), n_bin, i_bin_comb);
            pp.get(min.c_str(), bin_min, i_bin_comb);
            pp.get(max.c_str(), bin_max, i_bin_comb);
            pp.get(log.c_str(), is_log, i_bin_comb);
            pp.query(ext.c_str(), count_ext, i_bin_comb);
            bins[i_bin_comb].push_back(Bin(bin_var, n_bin, is_log, bin_min, bin_max, count_ext));
        }
    }
    
    // Get the weight if it's given (unity by default)
    pp.query("name_weight", name_weight);
}

int
MultiStat::getNCombination(const int i_bin_comb) {
    // Compute the product of number of bins in each dimension for this combination
    int n_product=1;
    for (int i_dim=0; i_dim<n_dims; i_dim++) {
        n_product = n_product*bins[i_bin_comb][i_dim].n_bin;
    }
    return n_product;
}

int
MultiStat::getIdVar(const std::string name_var) {
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

std::string
MultiStat::remove_parenthesis(std::string name) {
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
MultiStat::initDataStructure(const int i_bin_comb) {
    data_min.resize(n_comp);
    data_max.resize(n_comp);
    data_mean.resize(n_comp);
    data_std.resize(n_comp);
    for (int i_comp=0; i_comp<n_comp; i_comp++) {
        int n_prod = getNCombination(i_bin_comb);
        data_pdf.resize(n_prod);
        data_min[i_comp].resize(n_prod);
        data_max[i_comp].resize(n_prod);
        data_mean[i_comp].resize(n_prod);
        data_std[i_comp].resize(n_prod);
        for (int i_prod=0; i_prod<n_prod; i_prod++) {
            data_pdf[i_prod] = 0;
            data_min[i_comp][i_prod] = std::numeric_limits<float>::max();
            data_max[i_comp][i_prod] = -std::numeric_limits<float>::max();
            data_mean[i_comp][i_prod] = 0.0;
            data_std[i_comp][i_prod] = 0.0;
        }
    }
}

// Statistics manipulation
int 
MultiStat::getGlobalNPoint() {
    int n_points = data_pdf.size();
    int n_total_pdf;
    for (int i = 0; i < n_points; ++i) {
        n_total_pdf += data_pdf[i];
    }
    
    return n_total_pdf;
}

amrex::Real
MultiStat::getGlobalMin(const int i_comp) {
    // Iterate over all levels to find the min
    amrex::Real min_data = std::numeric_limits<float>::max();
    amrex::Vector<amrex::Real> data_comp_min = data_min[i_comp];
    int n_points = data_comp_min.size();
    for (int i = 0; i < n_points; ++i) {
        if (data_comp_min[i]!=0) {
            min_data = std::min(min_data, data_comp_min[i]);
        }
    }
    
    return min_data;
}

amrex::Real
MultiStat::getGlobalMin(const std::string name_comp) {
    // Get variable index in name_vars
    int i_comp= getIdVar(name_comp);

    // Iterate over all levels to find the min
    amrex::Real min_data = getGlobalMin(i_comp);
    
    return min_data;
}

amrex::Real
MultiStat::getGlobalMax(const int i_comp) {
    // Iterate over all levels to find the min
    amrex::Real max_data = std::numeric_limits<float>::min();
    amrex::Vector<amrex::Real> data_comp_max = data_max[i_comp];
    int n_points = data_comp_max.size();
    for (int i = 0; i < n_points; ++i) {
        if (data_comp_max[i]!=0) {
            max_data = std::max(max_data, data_comp_max[i]);
        }
    }
    
    return max_data;
}

amrex::Real
MultiStat::getGlobalMax(const std::string name_comp) {
    // Get variable index in name_vars
    int i_comp= getIdVar(name_comp);

    // Iterate over all levels to find the min
    amrex::Real max_data = getGlobalMax(i_comp);
    
    return max_data;
}

// Functions for writing statistics to file
void 
MultiStat::writeStatToFile(const int i_bin_comb, const std::string type_file="binary") {
    if (amrex::ParallelDescriptor::IOProcessor()){
        // First write the header
        writeStatHeader(i_bin_comb);
        
        // Write Pdf in same folder as header since its in common for all variables
        writePdfToBinary();
        
        // Then write the data to the type of file specified
        for (int i_comp=0; i_comp<n_comp; i_comp++) {
            if (type_file == "ascii") {
                //don't use that, unreadable anyway
                writeStatToAscii(i_comp);
            } else if (type_file == "binary"){
                writeStatToBinary(i_comp);
            } else {
                amrex::Print() << type_file << " is not supported\n";
                amrex::Error();
            }
        }
    }
}

void
MultiStat::statName(const int i_bin_comb) {
    // Append the name of each bin variable with the form X1-var1_X2-var2_.._XN-varN
    stat_name = "X1-" + remove_parenthesis(bins[i_bin_comb][0].name_var);
    for (int i_dim=1; i_dim<n_dims; i_dim++) {
        stat_name += "_X" + std::to_string(i_dim+1) + "-" + remove_parenthesis(bins[i_bin_comb][i_dim].name_var);
    }
}

void 
MultiStat::writeStatHeader(const int i_bin_comb) {
    // Write an ascii header with the metadata
    // Create the directory for the full path to the file
    std::string full_path = path_output + "/" + stat_name + "/" + plt_file + "/";
    std::filesystem::create_directories(full_path);
    amrex::Print() << "\nWrite combination n°" << i_bin_comb+1 << " with name " << stat_name << "\n";
    
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
            fw << n_lev << "\n";
            fw << n_dims << "\n";
            // Put each bin data one after another here
            for (int i_dim=0; i_dim<n_dims; i_dim++) {
                Bin bin(bins[i_bin_comb][i_dim]);
                fw << bin.name_var << "\n";
                fw << bin.n_bin << "\n";
                fw << bin.is_log << "\n";
                fw << bin.min_var << "\n";
                fw << bin.max_var << "\n";
                fw << bin.count_ext << "\n";
            }
            
            fw.close();
        }
        else amrex::Print() << "Problem with opening file\n";
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
}

void 
MultiStat::writePdfToBinary() {
    // Write the array of data in a binary file in the order of dimension given
    // Create the directory for the full path to the files for the variable
    std::string full_path = path_output + "/" + stat_name + "/" + plt_file + "/";
    std::filesystem::create_directories(full_path);
    
    // pdf statistics
    try {
        std::ofstream binfile;
        binfile.open(full_path + "data_pdf.bin", std::ios::binary);
        if (binfile.is_open())
        {
            //amrex::Print() << "Writing binary file to " << full_path << "\n";
        } else {
            amrex::Print() << "Binary file not opened, crash incoming\n";
            amrex::Error();
        }
        // Converting and writing data to binary
        // ----- Data ----- //
        int n_points = data_pdf.size();;
        for (int i = 0; i < n_points; ++i) {
            binfile.write(reinterpret_cast<const char*>(&data_pdf[i]), 
                          sizeof(data_pdf[i]));
        }

        binfile.close();
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
}

void 
MultiStat::writeStatToBinary(const int i_comp) {
    // Write the array of data in a binary file in the order of dimension given
    // Create the directory for the full path to the files for the variable
    std::string full_path = path_output + "/" + stat_name + "/" + plt_file + "/" + remove_parenthesis(name_vars[i_comp]) + "/";
    std::filesystem::create_directories(full_path);
    
    // Write all files separately
    // min statistics
    try {
        std::ofstream binfile;
        binfile.open(full_path + "data_min.bin", std::ios::binary);
        if (binfile.is_open())
        {
            //amrex::Print() << "Writing binary file to " << full_path << "\n";
        } else {
            amrex::Print() << "Binary file not opened, crash incoming\n";
            amrex::Error();
        }
        // Converting and writing data to binary
        // ----- Data ----- //
        int n_points = data_pdf.size();;
        for (int i = 0; i < n_points; ++i) {
            binfile.write(reinterpret_cast<const char*>(&data_min[i_comp][i]), 
                          sizeof(data_min[i_comp][i]));
        }

        binfile.close();
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
    
    //max statistics
    try {
        std::ofstream binfile;
        binfile.open(full_path + "data_max.bin", std::ios::binary);
        if (binfile.is_open())
        {
            //amrex::Print() << "Writing binary file to " << full_path << "\n";
        } else {
            amrex::Print() << "Binary file not opened, crash incoming\n";
            amrex::Error();
        }
        // Converting and writing data to binary
        // ----- Data ----- //
        int n_points = data_pdf.size();;
        for (int i = 0; i < n_points; ++i) {
            binfile.write(reinterpret_cast<const char*>(&data_max[i_comp][i]), 
                          sizeof(data_max[i_comp][i]));
        }

        binfile.close();
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
    
    //mean statistics
    try {
        std::ofstream binfile;
        binfile.open(full_path + "data_mean.bin", std::ios::binary);
        if (binfile.is_open())
        {
            //amrex::Print() << "Writing binary file to " << full_path << "\n";
        } else {
            amrex::Print() << "Binary file not opened, crash incoming\n";
            amrex::Error();
        }
        // Converting and writing data to binary
        // ----- Data ----- //
        int n_points = data_pdf.size();;
        for (int i = 0; i < n_points; ++i) {
            binfile.write(reinterpret_cast<const char*>(&data_mean[i_comp][i]), 
                          sizeof(data_mean[i_comp][i]));
        }

        binfile.close();
    }
    catch (const char* msg) {
        amrex::Print() << msg << "\n";
    }
}

void 
MultiStat::writeStatToAscii(const int i_comp) {
    // ---------------- PLEASE DON'T USE ASCII ... -------------------
    amrex::Print() << "PLEASE DON'T USE ASCII ...\n";
    amrex::Error();
}