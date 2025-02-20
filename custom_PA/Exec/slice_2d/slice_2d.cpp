#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <DataManager.H>
#include <Slice2d.H>
#include <InputParam.H>

#include <chrono>
using namespace std::chrono;

static
void
print_usage (int,
             char* argv[])
{
  std::cerr << "Template to use the class DataManager";
  std::cerr << "usage:\n";
  std::cerr << argv[0] << "infiles=<s1 s2 s3> [options] \n\tOptions:\n";
  std::cerr << "\t     infiles=<s1 s2 s3> where <s1> <s2> and <s3> are pltfiles\n";
  std::cerr << "\t     outfile=<s> where <s> is the output pltfile\n";
  std::cerr << "\t     variables=<s1 s2 s3> where <s1> <s2> and <s3> are variable names to select for combined pltfile [DEF-> all possible]\n";
  std::cerr << "\t     output_max_level=<s> where <s> is the max refinement level to combine, zero-indexed [DEF->1000]\n";
  std::cerr << "\t     output_max_grid_size=<s> where <s> is the output max_grid_size. If all BoxArrays are the same, this is ignored. [DEF->32]\n";
exit(1);
}

int
main (int   argc,
      char* argv[])
{
    // Basic Initialization of amrex
    amrex::Initialize(argc,argv);
    {
        if (argc < 2) {
            print_usage(argc,argv);
        }
        
        // Initialize input parameters
        InputParam input(AMREX_SPACEDIM);
        
        // Initialize conditiong variables
        ConditionalParam conditional;
        conditional.printConditions();
        
        // Initialize data structure (add verbosity at some point)
        DataManager original_data(input, conditional, 0, true);
        int lev = original_data.getNLev()-1;
        
        // Regridding before slicing (don't touch)
        amrex::Geometry geom;
        amrex::MultiFab data = original_data.regridUniform(geom);
        
        // Putting slice input here for now
        amrex::Vector<std::string> axis_list  = {"x" , "y" , "z"};
        amrex::Vector<amrex::Real> value_list = {0.50, 0.50, 0.10};
        int n_slice = axis_list.size();
        for (int i_slice=0; i_slice<n_slice; i_slice++) {
            auto start_slice = high_resolution_clock::now();
            // Create the Slice2d object
            Slice2d slice(input.path_output, original_data.getPltFile(), data, geom,
                          original_data.getNameVars(), lev, axis_list[i_slice], value_list[i_slice], original_data.getTime());
            
            amrex::Print() << "Creating slice took "
                           << duration_cast<milliseconds>(high_resolution_clock::now() - start_slice).count() << "ms\n";
            
            auto start_binary = high_resolution_clock::now();
            // Write it to file in binary
            slice.writeAllSlice("binary");
            
            amrex::Print() << "Writing binary took "
                           << duration_cast<milliseconds>(high_resolution_clock::now() - start_binary).count() << "ms\n";
            
            //auto start_ascii = high_resolution_clock::now();
            // Write it to file in ascii
            //slice.writeAllSlice("ascii");
            
            //amrex::Print() << "Writing ascii took "
            //               << duration_cast<milliseconds>(high_resolution_clock::now() - start_ascii).count() << "ms\n";
            
        }
    }
    amrex::Finalize();
    return 0;
}

