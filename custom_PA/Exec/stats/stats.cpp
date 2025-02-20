#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <DataManager.H>
#include <Slice2d.H>
#include <MultiStat.H>
#include <InputParam.H>
#include <ConditionalParam.H>

#include <chrono>
using namespace std::chrono;

static
void
print_usage (int,
             char* argv[])
{
  std::cerr << "Template to use the class DataManager";
  std::cerr << "usage:\n";
  std::cerr << argv[0];
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
        ConditionalParam conditions;
        conditions.printConditions();
        
        // Initialize data structure (add verbosity at some point)
        DataManager original_data(input, conditions, 0, true);
        
        // Testing statistics
        MultiStat stats(input.path_output, original_data.getPltFile(),
                        original_data.getNameVars(), original_data.getNLev(), original_data.getTime());
        
        // Write a metadata file for the conditions applied to data (to remember which case did what)
        conditions.writeConditions(input.path_output);
        
        // Write all extracted statistics to a file on disk
        int n_combination = stats.n_bin_comb;
        for (int i_bin_comb=0; i_bin_comb<n_combination; i_bin_comb++) {
            original_data.extractSingleStat(i_bin_comb, stats);
            stats.writeStatToFile(i_bin_comb, "binary");
        }
    }
    amrex::Finalize();
    return 0;
}

