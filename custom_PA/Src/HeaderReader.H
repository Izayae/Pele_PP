#ifndef HEADERREADER_H
#define HEADERREADER_H

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <math.h>

// Class inspired by Olivier's tool mandoline
// independent from AMReX just in case

typedef struct grid_coord {
    std::vector<int> lo;
    std::vector<int> hi;
    std::vector<int> grow;
    
    // constructor
    grid_coord() = default;
    grid_coord(int n_dim) {resize(n_dim);};
    
    // re-size
    void
    resize(int n_dim) {lo.resize(n_dim); 
                       hi.resize(n_dim);
                       grow.resize(n_dim);};
} grid_coord;

typedef struct box_coord {
    std::vector<double> real_lo;
    std::vector<double> real_hi;
    std::vector<int> num_lo;
    std::vector<int> num_hi;
    
    // constructor
    box_coord() = default;
    box_coord(int n_dim) {resize(n_dim);};
    
    // re-size
    void
    resize(int n_dim) {real_lo.resize(n_dim); 
                       real_hi.resize(n_dim);
                       num_lo.resize(n_dim);
                       num_hi.resize(n_dim);};
    
    // Count number of points inside this box
    int countPoint() {return (num_hi[0]-num_lo[0]+1)*(num_hi[1]-num_lo[1]+1)*(num_hi[2]-num_lo[2]+1);}
    
} box_coord;

typedef struct HeaderReader
{
    // Constructors
    HeaderReader() = default;
    HeaderReader(std::string header_path, int max_level);
    
    // print headers for validation
    void PrintHeader();
    
    // Header information
    int limit_level;
    std::string header_version; // header version (i.e. HyperCLaw-V1.1)
    int n_field; // number of fields in the plotfile
    std::vector<std::string> field_names; // array of field names of size n_field
    int n_dim; // Number of dimension
    double sim_time; // Simulation time
    int finest_level; // Finest AMR level
    std::vector<double> domain_lo; // lower domain boundaries
    std::vector<double> domain_hi; // Upper domain boundaries
    std::vector<int> ref_ratio; // Refinement ratio
    std::vector<grid_coord> grid_level; // grid numerical coordinates
    std::vector<double> dx; // dx of each level
    std::vector<int> n_box; // Number of boxes in each level
    std::vector<std::vector<int>> N_coord; // Numerical size of the domain for each level 
    std::vector<std::vector<box_coord>> box_list; // Box boundaries list
} HeaderReader;

#endif