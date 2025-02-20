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

amrex::Real
linearInterpolation(amrex::Vector<amrex::Real> X_arr,
                    amrex::Vector<amrex::Real> Y_arr,
                    amrex::Real x_val){
    
    amrex::Real min_X = *std::min_element(X_arr.begin(), X_arr.end());
    amrex::Real max_X = *std::max_element(X_arr.begin(), X_arr.end());
    amrex::Real min_Y = *std::min_element(Y_arr.begin(), Y_arr.end());
    amrex::Real max_Y = *std::max_element(Y_arr.begin(), Y_arr.end());
    amrex::Real y_val;
    // below X_arr range
    if (x_val <= min_X) {
        y_val = min_Y;
    }
    // above X_arr range
    else if (x_val >= max_X) {
        y_val = max_Y;
    } 
    // in X_arr range
    else {
        // Index of the point just below x_val in X_arr/segmentwhere x_val is situated
        int i_val = int(std::lower_bound(X_arr.begin(), X_arr.end(), x_val) - X_arr.begin()) - 1;
        
        // linear interpolation in the segment found
        amrex::Real a = (Y_arr[i_val+1] - Y_arr[i_val])/(X_arr[i_val+1] - X_arr[i_val]);
        amrex::Real b = Y_arr[i_val];
        y_val = a*(x_val-X_arr[i_val]) + b;
    }
    
    return y_val;
}

void
DataManager::manageNewQuantities(){
    amrex::Print() << "\n-------------------------------\n";
    amrex::Print() <<   "|   Managing new quantities   |\n";
    amrex::Print() <<   "-------------------------------\n";
    // Just add the new quantities to list of final names
    // Validity is checked later
    for (int n=0; n<input.n_new_qt; n++){
        name_vars.push_back(input.new_qt_names[n]);
    }
}

void
DataManager::addNewQuantities(){
    // Compute and add the new quantities if they are valid
    // TODO implement multi-components in one go by adding argument or thingy for number of components
    for (int n=0; n<input.n_new_qt; n++){
        std::string qt = input.new_qt_names[n];
        int id_new_qt = getIdVar(qt);
        if ((qt == "x") or (qt == "y") or (qt == "z")){
            for (int lev=0; lev<n_levs; ++lev) {
                amrex::Real dx=DM_geoms[lev].CellSize(0);
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr = DM_data[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute new quantity here
                        if (qt=="x"){
                            data_arr(i,j,k,id_new_qt) = i*dx + dx/2;
                        } else if (qt=="y") {
                            data_arr(i,j,k,id_new_qt) = j*dx + dx/2;
                        } else if (qt=="z") {
                            data_arr(i,j,k,id_new_qt) = k*dx + dx/2;
                        }
                   });  
                }
            }
        }
        // Numerical coordinates
        else if ((qt == "ix") or (qt == "iy") or (qt == "iz")){
            for (int lev=0; lev<n_levs; ++lev) {
                amrex::Real dx=DM_geoms[lev].CellSize(0);
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr   = DM_data[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute new quantity here
                        if (qt=="ix"){
                            data_arr(i,j,k,id_new_qt) = i;
                        } else if (qt=="iy") {
                            data_arr(i,j,k,id_new_qt) = j;
                        } else if (qt=="iz") {
                            data_arr(i,j,k,id_new_qt) = k;
                        }
                   });  
                }
            }
        }
        // Velocity magnitude of the velocity field
        else if (qt == "mag_velocity"){
            int id_vel_x = getIdVar("x_velocity");
            int id_vel_y = getIdVar("y_velocity");
            int id_vel_z = getIdVar("z_velocity");
            for (int lev=0; lev<n_levs; ++lev) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr = DM_data[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute velocity magnitude here
                        data_arr(i,j,k,id_new_qt) = std::sqrt(data_arr(i,j,k,id_vel_x)*data_arr(i,j,k,id_vel_x)+
                                                              data_arr(i,j,k,id_vel_y)*data_arr(i,j,k,id_vel_y)+
                                                              data_arr(i,j,k,id_vel_z)*data_arr(i,j,k,id_vel_z));
                    });  
                }
            }
        }
        // Compute kinematic viscosity
        else if (qt == "kinematic_viscosity"){
            int id_rho = getIdVar("density");
            int id_mu = getIdVar("viscosity");
            for (int lev=0; lev<n_levs; ++lev) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr = DM_data[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute velocity magnitude here
                        data_arr(i,j,k,id_new_qt) = data_arr(i,j,k,id_mu)/data_arr(i,j,k,id_rho);
                    });  
                }
            }
        }
        // Strain rate magnitude of the velocity field
        else if (qt == "strain_rate"){
            // Compute the velocity gradients for the strain rate tensor
            amrex::Vector<amrex::MultiFab> Ux = getGradient("x_velocity");
            amrex::Vector<amrex::MultiFab> Uy = getGradient("y_velocity");
            amrex::Vector<amrex::MultiFab> Uz = getGradient("z_velocity");
            
            // Compute the strain rate components
            for (int lev=0; lev<n_levs; ++lev) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr = DM_data[lev].array(mfi);
                    auto const& Ux_arr = Ux[lev].array(mfi);
                    auto const& Uy_arr = Uy[lev].array(mfi);
                    auto const& Uz_arr = Uz[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute the strain rate components Dij
                        amrex::Real Dxx = Ux_arr(i,j,k,0);
                        amrex::Real Dyy = Uy_arr(i,j,k,1);
                        amrex::Real Dzz = Uz_arr(i,j,k,2);
                        amrex::Real Dxy = 0.5*(Ux_arr(i,j,k,1) + Uy_arr(i,j,k,0));
                        amrex::Real Dxz = 0.5*(Ux_arr(i,j,k,2) + Uz_arr(i,j,k,0));
                        amrex::Real Dyx = 0.5*(Uy_arr(i,j,k,0) + Ux_arr(i,j,k,1));
                        amrex::Real Dyz = 0.5*(Uy_arr(i,j,k,2) + Uz_arr(i,j,k,1));
                        amrex::Real Dzx = 0.5*(Uz_arr(i,j,k,0) + Ux_arr(i,j,k,2));
                        amrex::Real Dzy = 0.5*(Uz_arr(i,j,k,1) + Uy_arr(i,j,k,2));
                        
                        // Compute the strain rate: sqrt(2*Dij:Dij)
                        data_arr(i,j,k,id_new_qt) = std::sqrt(2*(Dxx*Dxx + Dxy*Dxy + Dxz*Dxz
                                                                +Dyx*Dyx + Dyy*Dyy + Dyz*Dyz
                                                                +Dzx*Dzx + Dzy*Dzy + Dzz*Dzz));
                    });  
                }
            }
        }
        // Vorticity magnitude of the velocity field
        else if (qt == "vorticity"){       
            // Compute the velocity gradients for the strain rate tensor
            amrex::Vector<amrex::MultiFab> Ux = getGradient("x_velocity");
            amrex::Vector<amrex::MultiFab> Uy = getGradient("y_velocity");
            amrex::Vector<amrex::MultiFab> Uz = getGradient("z_velocity");
            
            // Compute the strain rate components
            for (int lev=0; lev<n_levs; ++lev) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr = DM_data[lev].array(mfi);
                    auto const& Ux_arr = Ux[lev].array(mfi);
                    auto const& Uy_arr = Uy[lev].array(mfi);
                    auto const& Uz_arr = Uz[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute the vorticity tensor components Wij
                        amrex::Real Wxx = 0;
                        amrex::Real Wyy = 0;
                        amrex::Real Wzz = 0;
                        amrex::Real Wxy = 0.5*(Ux_arr(i,j,k,1) - Uy_arr(i,j,k,0));
                        amrex::Real Wxz = 0.5*(Ux_arr(i,j,k,2) - Uz_arr(i,j,k,0));
                        amrex::Real Wyx = 0.5*(Uy_arr(i,j,k,0) - Ux_arr(i,j,k,1));
                        amrex::Real Wyz = 0.5*(Uy_arr(i,j,k,2) - Uz_arr(i,j,k,1));
                        amrex::Real Wzx = 0.5*(Uz_arr(i,j,k,0) - Ux_arr(i,j,k,2));
                        amrex::Real Wzy = 0.5*(Uz_arr(i,j,k,1) - Uy_arr(i,j,k,2));
                        
                        // Compute the strain rate: sqrt(2*Wij:Wij)
                        data_arr(i,j,k,id_new_qt) = std::sqrt(2*(Wxx*Wxx + Wxy*Wxy + Wxz*Wxz
                                                                +Wyx*Wyx + Wyy*Wyy + Wyz*Wyz
                                                                +Wzx*Wzx + Wzy*Wzy + Wzz*Wzz));
                    });  
                }
            }
        }
        // Q-criterion
        else if (qt == "Q_criterion"){       
            // Compute the velocity gradients for the strain rate tensor
            amrex::Vector<amrex::MultiFab> Ux = getGradient("x_velocity");
            amrex::Vector<amrex::MultiFab> Uy = getGradient("y_velocity");
            amrex::Vector<amrex::MultiFab> Uz = getGradient("z_velocity");
            
            // Compute the strain rate components
            for (int lev=0; lev<n_levs; ++lev) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& data_arr = DM_data[lev].array(mfi);
                    auto const& Ux_arr = Ux[lev].array(mfi);
                    auto const& Uy_arr = Uy[lev].array(mfi);
                    auto const& Uz_arr = Uz[lev].array(mfi);
                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        // Compute the strain rate components Dij
                        amrex::Real Dxx = Ux_arr(i,j,k,0);
                        amrex::Real Dyy = Uy_arr(i,j,k,1);
                        amrex::Real Dzz = Uz_arr(i,j,k,2);
                        amrex::Real Dxy = 0.5*(Ux_arr(i,j,k,1) + Uy_arr(i,j,k,0));
                        amrex::Real Dxz = 0.5*(Ux_arr(i,j,k,2) + Uz_arr(i,j,k,0));
                        amrex::Real Dyx = 0.5*(Uy_arr(i,j,k,0) + Ux_arr(i,j,k,1));
                        amrex::Real Dyz = 0.5*(Uy_arr(i,j,k,2) + Uz_arr(i,j,k,1));
                        amrex::Real Dzx = 0.5*(Uz_arr(i,j,k,0) + Ux_arr(i,j,k,2));
                        amrex::Real Dzy = 0.5*(Uz_arr(i,j,k,1) + Uy_arr(i,j,k,2));
                        
                        amrex::Real strain_mag_sq = 2*(Dxx*Dxx + Dxy*Dxy + Dxz*Dxz
                                                      +Dyx*Dyx + Dyy*Dyy + Dyz*Dyz
                                                      +Dzx*Dzx + Dzy*Dzy + Dzz*Dzz);
                        
                        // Compute the vorticity tensor components Wij
                        amrex::Real Wxx = 0;
                        amrex::Real Wyy = 0;
                        amrex::Real Wzz = 0;
                        amrex::Real Wxy = 0.5*(Ux_arr(i,j,k,1) - Uy_arr(i,j,k,0));
                        amrex::Real Wxz = 0.5*(Ux_arr(i,j,k,2) - Uz_arr(i,j,k,0));
                        amrex::Real Wyx = 0.5*(Uy_arr(i,j,k,0) - Ux_arr(i,j,k,1));
                        amrex::Real Wyz = 0.5*(Uy_arr(i,j,k,2) - Uz_arr(i,j,k,1));
                        amrex::Real Wzx = 0.5*(Uz_arr(i,j,k,0) - Ux_arr(i,j,k,2));
                        amrex::Real Wzy = 0.5*(Uz_arr(i,j,k,1) - Uy_arr(i,j,k,2));
                        
                        amrex::Real vort_mag_sq = 2*(Wxx*Wxx + Wxy*Wxy + Wxz*Wxz
                                                    +Wyx*Wyx + Wyy*Wyy + Wyz*Wyz
                                                    +Wzx*Wzx + Wzy*Wzy + Wzz*Wzz);
                        
                        // Compute the Q-criterion
                        data_arr(i,j,k,id_new_qt) = (vort_mag_sq-strain_mag_sq)/2;
                    });  
                }
            }
        }
        // Flame index (you may add your version for other fuel/oxidizer)
        else if (qt == "FI_DME_O2" or qt == "FI_CH4_O2" or qt == "FI_H2_O2"){
            // Management of the index and names
            int id_FI = getIdVar(qt);
            std::string name_fuel;
            std::string name_oxidizer = "Y(O2)";
            if (qt == "FI_DME_O2"){
                name_fuel = "Y(CH3OCH3)";
            } else if (qt == "FI_CH4_O2"){
                name_fuel = "Y(CH4)";
            } else if (qt == "FI_H2_O2"){
                name_fuel = "Y(H2)";
            }

            // Compute the gradient of the fuel species and oxidizer species
            amrex::Vector<amrex::MultiFab> grad_fuel = getGradient(name_fuel);
            amrex::Vector<amrex::MultiFab> grad_oxidizer = getGradient(name_oxidizer);
            
            // Compute the flame index with fuel and oxidizer gradients
            for (int lev = 0; lev < n_levs; ++lev) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(DM_data[lev],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {    
                    const amrex::Box& bx = mfi.tilebox();
                    auto const& grad_fuel_arr   = grad_fuel[lev].const_array(mfi);
                    auto const& grad_oxi_arr    = grad_oxidizer[lev].const_array(mfi);
                    auto const& flame_index_arr = DM_data[lev].array(mfi, id_FI);

                    amrex::ParallelFor(bx, [=]
                    AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {   
                        if (grad_fuel_arr(i,j,k,3)*grad_oxi_arr(i,j,k,3) != 0.0) {
                            flame_index_arr(i,j,k,0) = grad_fuel_arr(i,j,k,0) * grad_oxi_arr(i,j,k,0)
                                                     + grad_fuel_arr(i,j,k,1) * grad_oxi_arr(i,j,k,1)
                                                     + grad_fuel_arr(i,j,k,2) * grad_oxi_arr(i,j,k,2);
                            flame_index_arr(i,j,k,0) = flame_index_arr(i,j,k,0)/(grad_fuel_arr(i,j,k,3)*grad_oxi_arr(i,j,k,3));
                        } else {
                            flame_index_arr(i,j,k,0) = 0.0;
                        }
                   });  
                } 
            }
        } else {
            std::cout << "Quantity " << qt << " not supported yet\n";
            amrex::Error();
        }
        amrex::Print() << qt << " was successfully added\n";
    } // end for qt
}