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

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 2)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return std::move(out).str();
}

struct FillExtDirDummy
{

  AMREX_GPU_HOST
  constexpr FillExtDirDummy() = default;

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& /*iv*/,
    amrex::Array4<amrex::Real> const& /*dummy*/,
    const int /*dcomp*/,
    const int /*numcomp*/,
    amrex::GeometryData const& /*geom*/,
    const amrex::Real /*time*/,
    const amrex::BCRec* /*bcr*/,
    const int /*bcomp*/,
    const int /*orig_comp*/) const
  {
  }
};
// ------------------------------------------------------ //
// -------------------- Data loading -------------------- //
// ------------------------------------------------------ //

DataManager::DataManager(InputParam input, 
                         ConditionalParam conditions,
                         const int i_plt,
                         const bool load_data=true): input{std::move(input)}, conditions{std::move(conditions)}
{
    // Initialize geometry from amrData
    geomInit(i_plt);
    
    // Load user-selected data from the different plotfiles and group them in one big MultiFab
    if (load_data) {
        loadData(i_plt);
    }
    
    // Check conditioning variables if conditioning is activated
    //conditions.printConditions();
}

void
DataManager::checkVariables(const amrex::Vector<std::string> plot_var_names,
                            const amrex::Vector<std::string> selected_var_names,
                            amrex::Vector<std::string>& checked_name_vars,
                            int& checked_n_comp){
    // Make sure that variables in selected_var_names are in plot_var_names and are not duplicate
    // Update list and length of names accordingly
    int n_plot_comp = plot_var_names.size();
    int n_selected_comp = selected_var_names.size();
    
    // Validity check of the selected components
    if (n_selected_comp == 0 or selected_var_names[0]=="all") {
        // No variables were asked by the user or var is named "all" so all variables in this plotfiles are loaded
        amrex::Print() << "Loading all data in this plotfile\n";
        for (int ivar=0; ivar<n_plot_comp; ++ivar)
        {
            std::string var = plot_var_names[ivar];
            checked_name_vars.push_back(var);
            checked_n_comp += 1;
            amrex::Print() << checked_n_comp << ": found " << var << "\n";
        }
    } else {   
        checked_name_vars.resize(0);    // names of the selected variables in the order of loading
        for(int ivar = 0; ivar < n_selected_comp; ++ivar) { 
            // Index of variables extracted
            std::string var = selected_var_names[ivar];

            // Duplicate check
            if (std::find(checked_name_vars.begin(), checked_name_vars.end(), var) != checked_name_vars.end()) 
            {
                amrex::Print() << var << " is a duplicate and is ignored\n";
            }
            else
            {
                // Verify that the selected component name is indeed in the plotfile
                if (std::find(plot_var_names.begin(), plot_var_names.end(), var) != plot_var_names.end()) {
                    checked_name_vars.push_back(var);
                    checked_n_comp += 1;
                    amrex::Print() << checked_n_comp << ": found " << var << "\n";
                }
                else
                {
                    amrex::Print() << "Cannot find " << var << " so it's ignored\n";
                }
            }
        }
    }
    
    // If User asked variables but none were found, stop here
    if (checked_n_comp == 0) {
        amrex::Print() << "No valid variable found in this plotfile, please select one in:\n";
        for (int ivar=0; ivar<n_plot_comp; ++ivar) {
            amrex::Print() << plot_var_names[ivar] << " ";
        }
        amrex::Error();
    }
}

void
DataManager::manageVariables(const int i_plt,
                             amrex::Vector<amrex::Vector<std::string>>& names_multi,
                             amrex::Vector<int>& n_comp_selected_multi){
    amrex::Print() << "\n\n--------------------------\n";
    amrex::Print() <<     "|   Managing variables   |\n";
    amrex::Print() <<     "--------------------------\n";
    
    const amrex::Vector<amrex::Vector<std::string>> full_paths = input.full_paths;
    const amrex::Vector<amrex::Vector<std::string>> selected_var_names = input.file_vars;
    
    // Check that requested components are in the plotfiles and if no component is given take all of them
    for (int i_files=0; i_files<input.n_files; i_files++){
        // Get AmrData of this plotfile
        amrex::DataServices::SetBatchMode();
        amrex::Amrvis::FileType fileType(amrex::Amrvis::NEWPLT);

        // Set up for reading pltfile
        amrex::DataServices dataServices(full_paths[i_files][i_plt], fileType);
        if (!dataServices.AmrDataOk())
            amrex::DataServices::Dispatch(amrex::DataServices::ExitRequest, NULL);
        amrex::AmrData& amr_data = dataServices.AmrDataRef();

        // Get information from AmrData
        amrex::Vector<std::string> plot_var_names = amr_data.PlotVarNames();
        n_comp_selected_multi[i_files] = 0;
        names_multi[i_files].resize(0);
        amrex::Print() << "Search components in " << full_paths[i_files][i_plt] << "\n";
        checkVariables(plot_var_names, selected_var_names[i_files], 
                       names_multi[i_files], n_comp_selected_multi[i_files]);
        
        // Some cleaning
        amr_data.FlushGrids();
    }
}

void
DataManager::reduceMulti(const amrex::Vector<amrex::Vector<std::string>>& names_multi, 
                         amrex::Vector<std::string>& name_vars, 
                         const amrex::Vector<int>& n_comp_selected_multi, 
                         int& n_comp_selected,
                         amrex::Vector<amrex::Vector<int>>& dest_fill_comps_multi){
    // Reduce the multi file structures to a global structure
    
    // Get number of files from size of vector
    int i_dest_fill_comps=0;
    for (int i_files=0; i_files<input.n_files; i_files++){
        n_comp_selected += n_comp_selected_multi[i_files];
        for (int i_comp_files=0; i_comp_files<n_comp_selected_multi[i_files]; i_comp_files++){
            // Data reducing
            name_vars.push_back(names_multi[i_files][i_comp_files]);

            // Fill order of destination copy in the global data MultiFab
            dest_fill_comps_multi[i_files].push_back(i_dest_fill_comps);
            i_dest_fill_comps++;
        }
    }
}

void
DataManager::loadMultiData(const int i_plt,
                           const amrex::Vector<amrex::Vector<std::string>>& names_multi,
                           const amrex::Vector<amrex::Vector<int>>& dest_fill_comps_multi){
    amrex::Print() << "\n--------------------------------------\n";
    amrex::Print() <<   "|   Loading data for each plotfile   |\n";
    amrex::Print() <<   "--------------------------------------\n";
    
    // Fill the global data MultiFab from the succesive plotfiles/amrData
    for (int i_files=0; i_files<input.n_files; i_files++){
        amrex::Print() << "load from file " << i_files << "\n";
        
        // Get AmrData
        amrex::DataServices::SetBatchMode();
        amrex::Amrvis::FileType fileType(amrex::Amrvis::NEWPLT);
        
        // Set up for reading pltfile
        amrex::DataServices dataServices(input.full_paths[i_files][i_plt], fileType);
        if (!dataServices.AmrDataOk())
            amrex::DataServices::Dispatch(amrex::DataServices::ExitRequest, NULL);
        amrex::AmrData& amr_data = dataServices.AmrDataRef();

        // Fill global data MultiFab vector from AmrData
        amrex::Print() << "    ";
        for (int lev=0; lev<n_levs; ++lev) {
            amrex::Print() << lev << " ";
            amr_data.FillVar(DM_data[lev], lev, names_multi[i_files], dest_fill_comps_multi[i_files]);
        }
        amrex::Print() << " \n";
        
        // Some cleaning
        amr_data.FlushGrids();
    }
}

void
DataManager::geomInit(const int i_plt) {
    //

    // Name of the plt (plt????? or whatever it is)
    plt_file = input.plot_file_names[i_plt];
    amrex::Print() << plt_file << "\n";

    // Initialize geometry structures with AmrData of the first plotfiles
    
    // Initialize AmrData to prepare loading data
    /*for (int if==0; if<1; if++) {
        AmrDataInit(input.plotFileNames[if])
        // Add verification that all geometries are exactly the same in all plotfiles
    }*/
    amrex::DataServices::SetBatchMode();
    amrex::Amrvis::FileType fileType(amrex::Amrvis::NEWPLT);
    
    // Set up for reading pltfile
    amrex::DataServices dataServices(input.full_paths[0][i_plt], fileType);
    if (!dataServices.AmrDataOk())
        amrex::DataServices::Dispatch(amrex::DataServices::ExitRequest, NULL);
    amrex::AmrData& amr_data_base = dataServices.AmrDataRef();
    
    // Extract the useful data from armData
    
    n_levs = std::min(amr_data_base.FinestLevel()+1, input.n_lev); // minimum level between user asked and finest level in plt
    time = amr_data_base.Time();
    num_dim = amr_data_base.ProbDomain();
    real_dim = amrex::RealBox(&(amr_data_base.ProbLo()[0]), 
                              &(amr_data_base.ProbHi()[0]));
    // Resize and fill the multi-level structure from amrData
    DM_grids.resize(n_levs);
    DM_dmaps.resize(n_levs);
    DM_geoms.resize(n_levs);
    DM_data.resize(n_levs);
    DM_ref_ratio.resize(n_levs-1);
    for (int lev=0; lev<n_levs; ++lev) {
        amrex::Print() << lev << " ";
        const amrex::BoxArray ba = amr_data_base.boxArray(lev);
        DM_grids[lev] = ba;
        DM_dmaps[lev] = amrex::DistributionMapping(ba);
        DM_geoms[lev] = amrex::Geometry(amr_data_base.ProbDomain()[lev], &real_dim, input.coord, &(input.is_per[0]));
        if (lev > 0) {
            DM_ref_ratio[lev-1] = {AMREX_D_DECL(2,2,2)};
        }
    }
    amrex::Print() << "\n";
}

int
DataManager::getIdVar(const std::string name_var){
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





// Here was the code for new quantities before

void
DataManager::loadData(const int i_plt) {
    //-------------------------------------------------------------------//
    //            Load selected quantities from each plotfile            //
    //-------------------------------------------------------------------//
    // Create the structures to store variable information of each separate plotfile
    amrex::Vector<amrex::Vector<std::string>> names_multi(input.n_files);                 // Names of selected variables in each plotfile
    amrex::Vector<int> n_comp_selected_multi(input.n_files);                       // Effective number of selected variables in each plotfile
    
    // Manage variable indexing with the multi structures
    manageVariables(i_plt, names_multi, n_comp_selected_multi);
    
    // Reduce the multi structures and produce the final indexing of the variables in the global MultiFab
    amrex::Vector<amrex::Vector<int>> dest_fill_comps_multi(input.n_files);         // structure useful for fiLLVar to show what components of global MF to fill with each file
    reduceMulti(names_multi, name_vars, 
                n_comp_selected_multi, n_comp_selected, 
                dest_fill_comps_multi);
    
    // Update the total number of components
    n_comp_total = n_comp_selected + input.n_new_qt;
    
    // Define the final structure of the global data MF
    for (int lev=0; lev<n_levs; ++lev) {
        DM_data[lev].define(DM_grids[lev], DM_dmaps[lev], n_comp_total, input.n_grow);
    }
    
    // Load data in global MF
    loadMultiData(i_plt, names_multi, dest_fill_comps_multi);
    
    //-------------------------------------------------------------------//
    //                Compute new quantities in global MF                //
    //-------------------------------------------------------------------//
    
    // Manage new quantities to compute (not needed?)
    manageNewQuantities();
        
    // Compute the new quantites and add them to global data MF (if valid)
    addNewQuantities();
}

//------------------------------------------------------------//
//       Write new plotfile with all loaded quantities        //
//------------------------------------------------------------//
void
DataManager::writePlotFile(){
    // Write a plotfile with the loaded data and the output path from input
    std::string output_file = input.path_output + plt_file;
    amrex::Vector<int> stepidx(input.n_lev, 0);
    amrex::WriteMultiLevelPlotfile(output_file, input.n_lev, GetVecOfConstPtrs(DM_data),
                                   name_vars, DM_geoms, 0.0, stepidx, DM_ref_ratio);
}

// ------------------------------------------------------ //
// ------------------- Data processing ------------------ //
// ------------------------------------------------------ //
amrex::MultiFab
DataManager::regridUniform(amrex::Geometry& uniform_geom) {
    amrex::Print() << "\n--------------------------------------\n";
    amrex::Print() <<   "|   Regridding data to uniform grid  |\n";
    amrex::Print() <<   "--------------------------------------\n";
    
    //  If lev is higher the maximum level loaded, change it to finest_level
    int uniform_lev = n_levs-1;
    if (uniform_lev > n_levs-1) {
        amrex::Print() << "Level asked(" << uniform_lev << ") for regridding is higher than finest level loaded\n";
        amrex::Print() << "Change regridding level to " << n_levs-1 << "\n";
        uniform_lev = n_levs-1;
    }
    
    
    // Interpolator
    amrex::InterpBase* mapper = &amrex::mf_pc_interp;
    
    // BC record for interpolator
    amrex::Vector<amrex::BCRec> dummyBCRec(n_comp_total);
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        if (DM_geoms[0].isPeriodic(idim)) {
            for (int n = 0; n < n_comp_total; n++) {
                dummyBCRec[n].setLo(idim, amrex::BCType::int_dir);
                dummyBCRec[n].setHi(idim, amrex::BCType::int_dir);
            }
        } else {
            for (int n = 0; n < n_comp_total; n++) {
                dummyBCRec[n].setLo(idim, amrex::BCType::foextrap);
                dummyBCRec[n].setHi(idim, amrex::BCType::foextrap);
            }
        }
    }
    
    // Initialize data structures for uniform grid
    amrex::BoxArray uniform_ba(num_dim[uniform_lev]);
    uniform_ba.maxSize(input.max_grid_size);
    amrex::DistributionMapping uniform_dmap(uniform_ba);
    amrex::MultiFab uniform_data(uniform_ba, uniform_dmap, n_comp_total, 0);
    uniform_data.setVal(0); // this line created a Segfault because no variable was initialized in input
    uniform_geom = amrex::Geometry(num_dim[uniform_lev], &real_dim, input.coord, &(input.is_per[0]));
    
    // iterate over the different levels to regrid to a single level uniform grid
    for (int lev=0; lev<uniform_lev+1; ++lev) {
        // Just print some information to verify regridding is working correctly
        amrex::Print() << "Regridding level " << lev << "\n";
        
        // refinement ratio between finestLevel and the current level lev
        amrex::IntVect levrr = uniform_geom.Domain().size() / DM_geoms[lev].Domain().size();
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<FillExtDirDummy>> fine_bndry_func(uniform_geom, {dummyBCRec}, FillExtDirDummy{});
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<FillExtDirDummy>> crse_bndry_func(DM_geoms[lev], {dummyBCRec}, FillExtDirDummy{});
        
        if (lev==uniform_lev){
            amrex::MultiFab temp(DM_data[lev].boxArray(), DM_data[lev].distributionMap, n_comp_total, 0);
            amrex::FillPatchSingleLevel(temp, amrex::IntVect(0), 0.0, {&DM_data[lev]}, {0.0}, 0, 0, n_comp_total,
                                        DM_geoms[lev], fine_bndry_func, 0);
            uniform_data.ParallelCopy(temp, 0, 0, n_comp_total);
        } else {
            amrex::MultiFab temp(refine(DM_data[lev].boxArray(),levrr), DM_data[lev].distributionMap, n_comp_total, 0);
            amrex::InterpFromCoarseLevel(temp, amrex::IntVect(0), 0.0, DM_data[lev], 0, 0, n_comp_total,
                                         DM_geoms[lev], uniform_geom, crse_bndry_func, 0, fine_bndry_func, 0,
                                         levrr, mapper, {dummyBCRec}, 0);
            uniform_data.ParallelCopy(temp, 0, 0, n_comp_total);
        }
        
    }   // end for levels
    
    return uniform_data;
}

void
DataManager::regridData(int new_lev,
                        amrex::Geometry& new_geom,
                        amrex::MultiFab& new_data) {
    amrex::Print() << "\n----------------------------------\n";
    amrex::Print() <<   "|   Regridding data to new grid  |\n";
    amrex::Print() <<   "----------------------------------\n";
    
    // Interpolator
    amrex::InterpBase* mapper = &amrex::mf_pc_interp;
    
    // BC record for interpolator
    amrex::Vector<amrex::BCRec> dummyBCRec(n_comp_total);
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        if (DM_geoms[0].isPeriodic(idim)) {
            for (int n = 0; n < n_comp_total; n++) {
                dummyBCRec[n].setLo(idim, amrex::BCType::int_dir);
                dummyBCRec[n].setHi(idim, amrex::BCType::int_dir);
            }
        } else {
            for (int n = 0; n < n_comp_total; n++) {
                dummyBCRec[n].setLo(idim, amrex::BCType::foextrap);
                dummyBCRec[n].setHi(idim, amrex::BCType::foextrap);
            }
        }
    }
    
    // Just print some information to verify regridding is working correctly
    amrex::Print() << "Regridding for level " << new_lev << "\n";    
    
    for (int plt_lev = 0; plt_lev <= new_lev; plt_lev++) {
        amrex::IntVect levrr = new_geom.Domain().size() / DM_geoms[plt_lev].Domain().size();
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<FillExtDirDummy>> fine_bndry_func(new_geom, {dummyBCRec}, FillExtDirDummy{});
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<FillExtDirDummy>> crse_bndry_func(DM_geoms[plt_lev], {dummyBCRec}, FillExtDirDummy{});
        if (plt_lev == new_lev) {
            amrex::MultiFab temp(DM_data[plt_lev].boxArray(), DM_data[plt_lev].distributionMap, n_comp_total, 0);
            amrex::FillPatchSingleLevel(temp, amrex::IntVect(0), 0.0, {&DM_data[plt_lev]}, {0.0}, 0, 0, n_comp_total,
                                        DM_geoms[plt_lev], fine_bndry_func, 0);
            new_data.ParallelCopy(temp, 0, 0, n_comp_total);
        } else {
            amrex::MultiFab temp(refine(DM_data[plt_lev].boxArray(), levrr), DM_data[plt_lev].distributionMap, n_comp_total, 0);
            amrex::InterpFromCoarseLevel(temp, amrex::IntVect(0), 0.0, DM_data[plt_lev], 0, 0, n_comp_total,
                                         DM_geoms[plt_lev], new_geom, crse_bndry_func, 0, fine_bndry_func, 0,
                                         levrr, mapper, {dummyBCRec}, 0);
            new_data.ParallelCopy(temp, 0, 0, n_comp_total);
        }
    }
    
    return;
}

amrex::Vector<amrex::MultiFab>
DataManager::getGradient( const std::string name_var) {
    
    // Get variable index in DM_data
    int id_var = getIdVar(name_var);
    
    // Create the data structure for the multi-level gradient
    amrex::Vector<amrex::MultiFab> gradient_data(n_levs);
    for (int lev=0; lev<n_levs; ++lev) {
        gradient_data[lev].define(DM_data[lev].boxArray(), DM_data[lev].distributionMap, AMREX_SPACEDIM+1, 0);
    }
    
    // Get face-centered gradients from MLMG
    amrex::LPInfo info;
    info.setAgglomeration(1);
    info.setConsolidation(1);
    info.setMetricTerm(false);
    info.setMaxCoarseningLevel(0);
    amrex::MLPoisson poisson({DM_geoms}, {DM_grids}, {DM_dmaps}, info);
    poisson.setMaxOrder(4);
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> lo_bc;
    std::array<amrex::LinOpBCType, AMREX_SPACEDIM> hi_bc;
    for (int idim = 0; idim< AMREX_SPACEDIM; idim++){
       if (input.is_per[idim] == 1) {
          lo_bc[idim] = hi_bc[idim] = amrex::LinOpBCType::Periodic;
       } else {
          if (input.is_sym[idim] == 1) {
             lo_bc[idim] = hi_bc[idim] = amrex::LinOpBCType::reflect_odd;
          } else {
             lo_bc[idim] = hi_bc[idim] = amrex::LinOpBCType::Neumann;
          }
       }
    }
    poisson.setDomainBC(lo_bc, hi_bc);
    
    // Need to apply the operator to ensure CF consistency with composite solve
    int n_grow_grad = 0;                   // No need for ghost cell on gradient
    amrex::Vector<amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>> grad(n_levs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi;
    amrex::Vector<amrex::MultiFab> laps;
    for (int lev = 0; lev < n_levs; ++lev) {
        for (int idim = 0; idim <AMREX_SPACEDIM; idim++) {
            const auto& ba = DM_grids[lev];
            grad[lev][idim].define(amrex::convert(ba,amrex::IntVect::TheDimensionVector(idim)), DM_dmaps[lev], 1, n_grow_grad);
        }
        phi.push_back(std::make_unique<amrex::MultiFab> (DM_data[lev], amrex::make_alias, id_var, 1));
        poisson.setLevelBC(lev, phi[lev].get());
        laps.emplace_back(DM_grids[lev], DM_dmaps[lev], 1, 1);
    }
    amrex::MLMG mlmg(poisson);
    mlmg.apply(amrex::GetVecOfPtrs(laps), amrex::GetVecOfPtrs(phi));
    mlmg.getFluxes(amrex::GetVecOfArrOfPtrs(grad), amrex::GetVecOfPtrs(phi), amrex::MLMG::Location::FaceCenter);

    for (int lev = 0; lev < n_levs; ++lev) {
        // Convert to cell avg gradient
        amrex::MultiFab gradAlias(gradient_data[lev], amrex::make_alias, 0, AMREX_SPACEDIM+1);
        average_face_to_cellcenter(gradAlias, 0, amrex::GetArrOfConstPtrs(grad[lev]));
        gradAlias.mult(-1.0);
        for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {    
           const amrex::Box& bx = mfi.tilebox();
           auto const& grad_a   = gradAlias.array(mfi);
           amrex::ParallelFor(bx, [=]
           AMREX_GPU_DEVICE (int i, int j, int k) noexcept
           {    
              grad_a(i,j,k,3) = std::sqrt(AMREX_D_TERM( grad_a(i,j,k,0) * grad_a(i,j,k,0),
                                                      + grad_a(i,j,k,1) * grad_a(i,j,k,1),
                                                      + grad_a(i,j,k,2) * grad_a(i,j,k,2)));
           });  
        }
    }
    
    return gradient_data;
}

amrex::Real 
DataManager::getMin(std::string name_var) {
    // Get variable index in DM_data
    int id_var = getIdVar(name_var);
    
    // Iterate over all levels to find the min
    amrex::Real min_data = std::numeric_limits<float>::max();
    for (int lev = 0; lev < n_levs; ++lev) {
        // Minimum at this level
        min_data = std::min(min_data, DM_data[lev].min(id_var));
    }
    
    return min_data;
}

amrex::Real 
DataManager::getMax(std::string name_var) {
    // Get variable index in DM_data
    int id_var = getIdVar(name_var);
    
    // Iterate over all levels to find the min
    amrex::Real max_data = std::numeric_limits<float>::min();
    for (int lev = 0; lev < n_levs; ++lev) {
        // Minimum at this level
        max_data = std::max (max_data, DM_data[lev].max(id_var));
    }
    
    return max_data;
}

// Specific functions for multi-dimensional statistics

int
DataManager::calcGlobalId(const int i_bin_comb, MultiStat& stat,
                          const int i, const int j, const int k, const amrex::Array4<const amrex::Real> arr_var) {
    // Get the values from the original data
    int inside_range=1;          // this parameter is set to 0 if on of the i_bin returned by getIdBin is not in the range
    amrex::Vector<int> index_bins;
    for (int i_dim=0; i_dim<stat.n_dims; i_dim++) {
        int id_orig = stat.getIdVar(stat.bins[i_bin_comb][i_dim].name_var);
        amrex::Real val_bin = arr_var(i, j, k, id_orig);
        int i_bin = stat.bins[i_bin_comb][i_dim].getIdBin(val_bin);
        if (i_bin < 0 or i_bin > (stat.bins[i_bin_comb][i_dim].n_bin-1)) {
            inside_range = 0;
        }
        index_bins.push_back(i_bin);
    }
    
    // Get the global index with the list of bins index
    int id_global=0;
    if (inside_range) {
        for (int i_dim=0; i_dim<stat.n_dims; i_dim++) {
            int n_product=1;
            for (int j_dim=0; j_dim<i_dim; j_dim++) {
                n_product = n_product*stat.bins[i_bin_comb][j_dim].n_bin;
            }
            id_global += index_bins[i_dim]*n_product;
        }
    } else {
        id_global == -1;
    }
    
    return id_global;
}

void
DataManager::extractAllStats(MultiStat& stat) {
    for (int i_bin_comb=0; i_bin_comb<stat.n_bin_comb; i_bin_comb++) {
        extractSingleStat(i_bin_comb, stat);
    }
}

void
DataManager::extractSingleStat(const int i_bin_comb, MultiStat& stat) {
    amrex::Print() << "\nExtracting data for statistics combination " << i_bin_comb << "\n";
    
    // Update the name of the statistics for later use
    stat.statName(i_bin_comb);
    
    // Get the index of the weight function if valid (else unity)
    int id_weight = -1;
    if (std::count(name_vars.begin(), name_vars.end(), stat.name_weight)) {
        amrex::Print() << "Weighting field " << stat.name_weight << " taken\n";
        id_weight = getIdVar(stat.name_weight);
    } else {
        amrex::Print() << "Weighting field " << stat.name_weight << " not found, unity weight taken\n";
        id_weight = -1;
    }
    
    // Extract the stats for the i_bin_comb-th combination of bins
    stat.initDataStructure(i_bin_comb);
    //amrex::Print() << stat.n_lev << "\n";
    
    for (int lev = stat.n_lev-1; lev >= 0; --lev) {
        //amrex::Print() << lev << "\n";
        amrex::iMultiFab mask;
        //amrex::MultiFab& data = DM_data[lev];
        amrex::Geometry geom = DM_geoms[lev];
        amrex::BoxArray grid = DM_grids[lev];
        amrex::DistributionMapping dmap = DM_dmaps[lev];
        if (lev == stat.n_lev-1) {
            mask.define(grid, dmap, 1, 0);
            mask.setVal(0);
        } else {
            amrex::IntVect ratio{AMREX_D_DECL(2, 2, 2)};
            mask = makeFineMask(grid, dmap, DM_grids[lev+1], ratio);
        }

        // cell iterations
        for (amrex::MFIter mfi(DM_data[lev]); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const& arr_var    = DM_data[lev].const_array(mfi);
            auto const& mask_array = mask.const_array(mfi);
            amrex::ParallelFor(bx, [&]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                //Starting the loop over the variables
                for (int i_comp=0; i_comp<stat.n_comp; i_comp++) {
                    // Conditioning variable
                    int contribute = conditions.checkConditions(i, j, k, arr_var, name_vars);

                    // if the datapoint verifies the conditions
                    if (mask_array(i,j,k)==0 and contribute==1) {
                        // Compute the equivalent global index to fill data structures in the right bins
                        int id_global = calcGlobalId(i_bin_comb, stat, 
                                                     i, j, k, arr_var);

                        // Fill the different values
                        if ( id_global >= 0) {
                            // field of interest value
                            amrex::Real cell_val = arr_var(i, j, k, i_comp);
                            // weight value (one by default)
                            amrex::Real weight_val = 1;
                            if (id_weight >= 0) {
                                weight_val = arr_var(i, j, k, id_weight);
                            }
                            stat.data_min[i_comp][id_global] = std::min(cell_val, stat.data_min[i_comp][id_global]);
                            stat.data_max[i_comp][id_global] = std::max(cell_val, stat.data_max[i_comp][id_global]);
                            stat.data_mean[i_comp][id_global] += weight_val*cell_val*pow(8, (stat.n_lev-1)-lev);
                            if (i_comp==0) {
                                // the power here is to weight the values added to the mean and pdf by the volume of the cells at this level
                                stat.data_pdf[id_global] += weight_val*pow(8,(stat.n_lev-1)-lev);
                            }
                        }
                    }
                } // end of variable loop
            });
        } // end of mfiter loop
    } // end of level loop
    
    //Reduce all the data to have the complete dataset on all processes (and not fragmented)
    int n_prod = stat.getNCombination(i_bin_comb);
    amrex::ParallelDescriptor::ReduceRealSum(stat.data_pdf.data(), n_prod);
    for (int i_comp=0; i_comp<stat.n_comp; i_comp++) {
        amrex::ParallelDescriptor::ReduceRealMin(stat.data_min[i_comp].data(), n_prod);
        amrex::ParallelDescriptor::ReduceRealMax(stat.data_max[i_comp].data(), n_prod);
        amrex::ParallelDescriptor::ReduceRealSum(stat.data_mean[i_comp].data(), n_prod);
        for (int i_global=0; i_global<n_prod; i_global++) {
            if (stat.data_pdf[i_global]>0){
                stat.data_mean[i_comp][i_global] = stat.data_mean[i_comp][i_global]/stat.data_pdf[i_global];
            } else {
                stat.data_mean[i_comp][i_global] = 0.;
            }        
        }
    }
}


//--------------------------------------------------------------//
//             Data extraction for kernel analysis              //
//--------------------------------------------------------------//
void
DataManager::writeDatapointsToFile() {
    // Regrid DataManager to a uniform grid at the finest level loaded
    amrex::Geometry geom;
    int finest_level = n_levs-1;
    amrex::MultiFab uniform_data = regridUniform(geom);
    
    // Count the number of points extracted
    int n_extracted=0;
    pointsExtracted(n_extracted, uniform_data);
    
    // Extract Datapoints verifying the conditions and write them to file (numerical coordinates + the rest)
    amrex::Vector<amrex::Real> extracted_var_data(n_extracted*(AMREX_SPACEDIM+n_comp_total));
    
    if (n_extracted > 0) {
        // print info for process with points found
        std::cout << "Process n°" << amrex::ParallelDescriptor::MyProc() << ": " << n_extracted << " points found\n"; 
        
        // Extract datapoints
        extractDatapoints(n_extracted, uniform_data, extracted_var_data);

        // Write the file headers
        writeExtractHeader(n_extracted);

        // Write the file data
        writeExtractToBinary(n_extracted, extracted_var_data);
    }
        
}


void
DataManager::pointsExtracted(int& n_extracted, const amrex::MultiFab& uniform_data) {
    // Extract the datapoints that verify the conditions for ignition kernel in mf_full
    for (amrex::MFIter mfi(uniform_data); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        if (bx.ok()) {
            const auto& arr_var = uniform_data.array(mfi);
            amrex::ParallelFor(bx, [&]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Conditioning variable
                int contribute = conditions.checkConditions(i, j, k, arr_var, name_vars);
              
                if (contribute==1) {
                    //std::cout << fab(i,j,k,idCond[0]) << "\n";
                    n_extracted++;
                }
           });
        }
    }
}

void 
DataManager::extractDatapoints(const int n_extracted, const amrex::MultiFab& uniform_data, amrex::Vector<amrex::Real>& extracted_var_data){
    // Extract the datapoints that verify the conditions for ignition kernel in mf_full
    int i_datapoint=0;
    int n_comp_final = 3 + n_comp_total;
    for (amrex::MFIter mfi(uniform_data); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        if (bx.ok()) {
            const auto& arr_var = uniform_data.array(mfi);
            amrex::ParallelFor(bx, [&]
            AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Conditioning variable
                int contribute = conditions.checkConditions(i, j, k, arr_var, name_vars);
                
                if (contribute==1) {
                    // Get the numerical coordinates of the point
                    extracted_var_data[i_datapoint*n_comp_final+0] = i; // add ix
                    extracted_var_data[i_datapoint*n_comp_final+1] = j; // add iy
                    extracted_var_data[i_datapoint*n_comp_final+2] = k; // add iz
                    for (int ivar=3; ivar<n_comp_final; ivar++) {
                        extracted_var_data[i_datapoint*n_comp_final+ivar] = arr_var(i,j,k,ivar-3); // add n°ivar component
                    }
                    i_datapoint++;
                }
           });
        }
    }
}

void 
DataManager::writeExtractHeader(int n_extracted) {
    // Get the useful information
    amrex::Vector<int> N = getNumDim(0);        // Get numerical dimension at level 0 (only level)
    
    // Create the directory tree for the extracted data if it doesn't exist
    std::string path_final = input.path_output + "/" + plt_file + "/";
    amrex::Print() << "Writing datapoints in " << path_final << std::endl;
    const int dir_err = std::system(("mkdir -p " + path_final).c_str());
    if (-1 == dir_err)
    {
        printf("Error creating directory!n");
        exit(1);
    }
    
    // Create a separate header file for metadata
    //Open and write the file
    std::string header_name = path_final + "kernel_" + std::to_string(amrex::ParallelDescriptor::MyProc()) + ".header";
    FILE *file;
    file = fopen((header_name).c_str(),"w");
    // Check that the file is correctly opened
    if (file != NULL)
    {
        //std::cout << "Header file successfully open\n";
    } else {
        std::cout << "Header file not opened, crash incoming\n";
        std::cout << (header_name).c_str() << "\n";
        perror("fopen");
    }
    
    // 1: number of the plotfile
    fprintf(file, "%s\n", plt_file.c_str());
    // 2: time of the plotfile
    fprintf(file, "%.12f\n", time);
    // 3: numerical dimension of the domain
    fprintf(file, "%i %i %i\n", N[0], N[1], N[2]);
    // 4: number of points extracted
    fprintf(file, "%i\n", n_extracted);
    // 5: name of the extracted components (including ix/iy/iz)
    fprintf(file, "ix iy iz");
    for (int i = 0; i < n_comp_total; ++i) {
        fprintf(file, " %s", name_vars[i].c_str());
    }
    fclose(file);
}
    
void 

DataManager::writeExtractToBinary(const int n_extracted,
                                  const amrex::Vector<amrex::Real>& extracted_var_data) {
    // Open binary file
    std::string path_final = input.path_output + "/" + plt_file + "/";
    std::string datafile_name = path_final + "kernel_" + std::to_string(amrex::ParallelDescriptor::MyProc()) + ".bin"; 
    std::ofstream binfile;
    binfile.open(datafile_name, std::ios::binary);
    if (binfile.is_open())
    {
        //amrex::Print() << "Binary file successfully open\n";
    } else {
        std::cout << "Binary file not opened, crash incoming\n";
        std::cout << (datafile_name).c_str() << "\n";
    }
    
    // Converting and writing data to binary
    // ----- Data ----- //
    int n_comp_final = 3 + n_comp_total;
    for (int n = 0; n < n_extracted; ++n) {
        // Write the data with one datapoint per line
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            binfile.write(reinterpret_cast<const char*>(&extracted_var_data[n*n_comp_final+idim]), sizeof(extracted_var_data[n*AMREX_SPACEDIM+idim]));
        }
        for (int ivar = 3; ivar < n_comp_final; ++ivar) {
            
            binfile.write(reinterpret_cast<const char*>(&extracted_var_data[n*n_comp_final+ivar]), sizeof(extracted_var_data[n*n_comp_final+ivar]));
        }
    }

    binfile.close();
    return;
}

/*void 
writeExtractToAscii() {

}*/