#ifndef INPUTPARAM_H
#define INPUTPARAM_H

#include <AMReX_PhysBCFunct.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <libgen.h>

typedef struct InputParam{
    // directory parameters
    std::string path_output;
    std::string outfile = "plt";
    
    // Boundary parameters
    amrex::Vector<std::string> lo_BC;
    amrex::Vector<std::string> hi_BC;
    amrex::Vector<int> is_per;
    amrex::Vector<int> is_sym;
    
    // AMR parameters
    int max_grid_size = 32;
    int finest_level = 0;
    int n_lev;
    int n_grow = 1;
    int coord = 0;
    
    // Paramaters to manage multiple plotfiles reading
    int n_files=1;                                 // Number of plotfiles read
    amrex::Vector<std::string> file_paths;         // List of the file paths (as many as n_files)
    amrex::Vector<amrex::Vector<std::string>> file_vars;  // Names of the variables asked by the user in each plotfile read
    int n_comp = 0;                                // Number of selected components in total
    amrex::Vector<int> n_file_comp;                // Number of components in each file read
    int n_plt=1;
    amrex::Vector<std::string> plot_file_names;    // List of the plotfile names (supposed the same for every files read)
    amrex::Vector<amrex::Vector<std::string>> full_paths;  // list of full paths for loading data
    
    
    // New quantities to compute
    amrex::Vector<std::string> new_qt_names;       // Names of the new quantities to compute
    int n_new_qt = 0;                              // Number of new quantities computed
    
    InputParam(int n_dim, std::string type_plt="normal"){ // Constructor, is it the best way?
        // Parameter parser
        amrex::ParmParse pp;
        
        // Name and path of the output file
        pp.get("path_output", path_output);
        pp.query("outfile", outfile);
        
        // AMR parameters
        pp.query("max_grid_size", max_grid_size);
        pp.query("finest_level", finest_level);
        n_lev = finest_level+1;
        
        // Boundary conditions
        lo_BC.resize(n_dim);
        hi_BC.resize(n_dim);
        is_per.resize(n_dim);
        is_sym.resize(n_dim);
        pp.getarr("lo_BC", lo_BC, 0, n_dim);
        pp.getarr("hi_BC", hi_BC, 0, n_dim);
        // Fill the BC structures for later with the correct boundaries and verify that the given Boundaries are coherent
        for (int i_dim = 0; i_dim<n_dim; i_dim++){
            amrex::Print() << i_dim << " BC:" << lo_BC[i_dim] << " " << hi_BC[i_dim] << "\n";
            if (lo_BC[i_dim]=="Periodic" and hi_BC[i_dim]=="Periodic") { // Periodic boundaries, verify that lo and hi are both periodic
                is_per[i_dim] = 1;
                is_sym[i_dim] = 0;
            } else if ( (lo_BC[i_dim]=="Periodic" and hi_BC[i_dim]!="Periodic") or
                        (lo_BC[i_dim]!="Periodic" and hi_BC[i_dim]=="Periodic") ) { // lo and hi not both periodic
                amrex::Abort("lo and hi BCs are not both periodic");
            } else if (lo_BC[i_dim]=="Custom" or hi_BC[i_dim]=="Custom") { // custom boundaries
                // What additional parameters should I put ?
            } else {
                // Do nothing
            }
        }
        
        if (type_plt=="avg_init") {
            amrex::Print() << "Initializing the grid for plt_avg\n";
            // Read the grid from the union grid for averaged file initialization
            file_paths.resize(1);
            file_vars.resize(1);
            n_file_comp.resize(1);
            plot_file_names.resize(1);
            plot_file_names[0] = "union_grid";
            // Build full paths for loading data of each file
            full_paths.resize(1);
            full_paths[0].resize(1);
            pp.get("union_grid_path", full_paths[0][0]);
            amrex::Print() << full_paths[0][0] << "\n";
            // Variables extracted for each plotfile
            n_file_comp[0] = 1;
            file_vars[0].resize(1);
            n_comp = 1;
        } else if (type_plt=="avg") {
            amrex::Print() << "Loading the grid for plt_avg\n";
            // Read the grid from the averaged file
            // Read the average file 
            file_paths.resize(1);
            file_vars.resize(1);
            n_file_comp.resize(1);
            plot_file_names.resize(1);
            plot_file_names[0] = "plt_avg";
            // Build full paths for loading data of each file
            full_paths.resize(1);
            full_paths[0].resize(1);
            pp.get("plt_avg_path", full_paths[0][0]);
            amrex::Print() << full_paths[0][0] << "\n";
            // Variables extracted for each plotfile
            n_file_comp[0] = 1;
            file_vars[0].resize(1);
            file_vars[0][0]="all";
            n_comp = 1;   // to update later for "all" quantities
        } else if (type_plt=="fluct") {
            amrex::Print() << "Loading the grid for plt_fluct\n";
            // Read the grid from the averaged file
            // Read the average file 
            file_paths.resize(1);
            file_vars.resize(1);
            n_file_comp.resize(1);
            plot_file_names.resize(1);
            plot_file_names[0] = "plt_fluct";
            // Build full paths for loading data of each file
            full_paths.resize(1);
            full_paths[0].resize(1);
            pp.get("plt_fluct_path", full_paths[0][0]);
            amrex::Print() << full_paths[0][0] << "\n";
            // Variables extracted for each plotfile
            n_file_comp[0] = 1;
            file_vars[0].resize(1);
            file_vars[0][0]="all";
            n_comp = 1;   // to update later for "all" quantities
        } else if (type_plt=="inst_fluct") {
            amrex::Print() << "Loading the grid for plt_inst_fluct\n";
            // Read the grid from the averaged file
            // Read the average file 
            file_paths.resize(1);
            file_vars.resize(1);
            n_file_comp.resize(1);
            plot_file_names.resize(1);
            plot_file_names[0] = "plt_inst_fluct";
            // Build full paths for loading data of each file
            full_paths.resize(1);
            full_paths[0].resize(1);
            pp.get("plt_fluct_path", full_paths[0][0]);
            std::string path_name = dirname(const_cast<char *>(full_paths[0][0].c_str()));
            full_paths[0][0] = path_name + "/" + plot_file_names[0];
            amrex::Print() << full_paths[0][0] << "\n";
            // Variables extracted for each plotfile
            n_file_comp[0] = 1;
            file_vars[0].resize(1);
            file_vars[0][0]="all";
            n_comp = 1;   // to update later for "all" quantities
        } else {
            // All other "normal" cases
            // Number of files read
            
            pp.query("n_files", n_files);
            file_paths.resize(n_files);
            file_vars.resize(n_files);
            n_file_comp.resize(n_files);
            
            // Variable names in each plotfile
            for (int i_files=0; i_files<n_files; i_files++){
                // Extract path of the files
                std::string file = "Path_file" + std::to_string(i_files+1);
                pp.get(file.c_str(), file_paths[i_files]);

                // Variables extracted for each plotfile
                std::string var = "Var" + std::to_string(i_files+1);
                n_file_comp[i_files] = pp.countval(var.c_str());
                file_vars[i_files].resize(n_file_comp[i_files]);
                pp.queryarr(var.c_str(), file_vars[i_files], 0, n_file_comp[i_files]);
                n_comp += n_file_comp[i_files];
            }
            
            // Which plotfiles are selected
            n_plt = pp.countval("Plt_file_names");
            plot_file_names.resize(n_plt);
            pp.getarr("Plt_file_names", plot_file_names);
            
            // Build full paths for loading data of each file
            full_paths.resize(n_files);
            for (int i_files=0; i_files<n_files; i_files++){
                full_paths[i_files].resize(n_plt);
                for (int i_plt=0; i_plt<n_plt; i_plt++){
                    full_paths[i_files][i_plt] = file_paths[i_files] + "/" + plot_file_names[i_plt];
                }
            }
            // New quantities to compute
            n_new_qt = pp.countval("new_quantities");
            pp.queryarr("new_quantities", new_qt_names, 0, n_new_qt);
        }
    }
} InputParam;

#endif