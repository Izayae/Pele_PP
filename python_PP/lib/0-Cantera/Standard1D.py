# for analytical jacobian
import pyjacob
import cantera as ct

from scipy import linalg
from scipy import interpolate
from scipy.signal import find_peaks

import numpy as np

import time
import os

import h5py

import CSP_class

### Create a class for 1D standard solution (easier access to everything)

class Standard1D():
    ## Class to manipulate my standard 1D solution
    def __init__(self, path_file, mechanism, sol_type, P=1):
        # initialization with 1D dictionnary and mechanism
        self.P = P
        self.sol_type = sol_type
        self.mechanism = mechanism
        self.gas = ct.Solution(mechanism)
        self.species_names = self.gas.species_names
        self.N_spec = len(self.species_names)
        self.reaction_names = self.gas.reaction_equations()
        self.N_reac = len(self.reaction_names)
        
        # file and look for groups in this file
        self.path_h5 = path_file
        self.group_names = self.h5pyGrp()
        
        # Load the first solution to get the list of field names
        self.dict_1D = {}
        self.dict_CSP = {}       # not really necessary in attributes
        self.current_group = self.group_names[0]
        self.loadGroup(self.current_group)
        
        return
        
    def h5pyGrp(self):
    ## Return names of the group in the specified h5 file (sorted in alphabetic order)
        with h5py.File(self.path_h5, 'r') as h5file: 
            group_names = [name for name in h5file]
        return sorted(group_names)
    
    def loadGroup(self, group):
        # h5 files can contain several groups, each with one solution
        # load the solution from given group
        self.current_group = group
        with h5py.File(self.path_h5, "r") as f:
            grp = f[group]
            list_fields = grp.attrs['list_fields']
            self.dict_1D = {}
            for dset_name in list_fields:
                self.dict_1D[dset_name] = grp[dset_name][:]
        self.list_fields = list(self.dict_1D.keys())
        self.N_sol = len(self.dict_1D[self.list_fields[0]])
        self.setupSolArray()
        return
    
    def resize_T(self, N_int, kind="linear", Tmin=None, Tmax=None):
        ### Resize solution by interpolating on temperature field
        # Adjust min and max
        if Tmin is None or Tmin < np.min(self.dict_1D["temp"]):
            Tmin = np.min(self.dict_1D["temp"])
        if Tmax is None or Tmax > np.max(self.dict_1D["temp"]):
            Tmax = np.max(self.dict_1D["temp"])
        # remove duplicates, find list of unique values
        non_dup = np.unique(self.dict_1D["temp"], return_index=True)[1]
        T_unique = self.dict_1D["temp"][non_dup]

        # perform the interpolation
        #print(Tmin, Tmax, np.max(dict_1D["temp"]))
        range_T = np.logical_and(T_unique>=0.99*Tmin, T_unique<=1.01*Tmax)
        new_T = np.linspace(Tmin, Tmax, N_int)
        list_fields = list(self.dict_1D.keys())
        new_dict = {}
        for field in list_fields:
            new_dict[field] = interpolate.interp1d(T_unique[range_T], self.dict_1D[field][non_dup][range_T], kind=kind)(new_T)
        
        # Update everything
        self.dict_1D = new_dict
        self.N_sol = len(self.dict_1D[self.list_fields[0]])
        self.setupSolArray()
        
        return
    
    def analytical_chem_source(dict_0D):
        ## Get g from pyjacob with T first and then Y in mechanism order
        ## Y is supposed to be formatted to match with pyjacob (N2 skipped)
        const_P = dict_0D["P"]
        state = np.append(dict_0D["T"],dict_0D["Y"])            # Temperature first, then mass fraction (without N2)
        g = np.zeros_like(state)
        pyjacob.py_dydt(0, const_P, state, g)  # compute source terms
        return g
    
    def getStateVector(self, i_extract):
        # Return the statevector numpy array at index i_extract (in pyjacob format keeping N2)
        state_vec = np.zeros(self.N_spec+1)
        state_vec[0] = self.dict_1D["temp"][i_extract]
        for i_spec, spec in enumerate(self.species_names):
            state_vec[i_spec+1] = self.dict_1D["Y({})".format(spec)][i_extract]
        return np.array(state_vec)
    
    ### Source terms stuff
    def setupSolArray(self):
        # Setup 1D cantera Solution array for the 1D solution
        # Create Cantera Solution
        self.gas_array = ct.SolutionArray(self.gas)
        # Populate the SolutionArray from the dictionary
        for i in range(self.N_sol):
            T = self.dict_1D["temp"][i]
            Y = np.array([self.dict_1D["Y({})".format(spec)][i] for spec in self.species_names])
            self.gas.TPY = T, self.P*ct.one_atm, Y
            self.gas_array.append(self.gas.state)
        
        return
    
    def setupThermoProp(self):
        # Thermochemical properties from cantera Solution Array
        # cp [J/kg/K]
        self.dict_1D["cp"] = self.gas_array.cp_mass
        # lambda [W/m/K]
        self.dict_1D["lambda"] = self.gas_array.thermal_conductivity
        # Diffusion coefficients [m^2/s]   (should it be mass or mole)
        self.dict_1D["Di"] = self.gas_array.mix_diff_coeffs_mole
        # molecular weights [kg/kmol]
        self.dict_1D["Wi"] = self.gas_array.molecular_weights
        # Density [kg/m^3] (density alone is in the basis of Cantera ...)
        self.dict_1D["rho"] = self.gas_array.density_mass
        # Species enthalpy [no unit] - units depends of R units
        self.dict_1D["hk"] = self.gas_array.standard_enthalpies_RT
        # Gas constant [J/kmol/K]
        R = ct.gas_constant
        # Conversion in [J/kg] = [J/kmol/K * K * kmol/kg]
        for j in range(self.N_spec):
            self.dict_1D["hk"][:, j] *= R*self.dict_1D["temp"]/self.dict_1D["Wi"][j]
            #self.dict_1D["hk"][:, j] *= R*self.dict_1D["temp"]
        # mean molecular weight
        self.dict_1D["W"] = self.gas_array.mean_molecular_weight
        return
    
    def computeHRRPI(self):
        ## Compute HRR participation indices (species AND reactions)
        
        return
        
    def computeSources(self):
        ## Compute all sources for the solution
        
        ## Update Solution array
        self.setupSolArray()
        
        ## thermodynamic properties with cantera
        self.setupThermoProp()
        
        ## Balance budget terms
        # Initialize source terms
        self.dict_1D["source_react"] = np.zeros((len(self.dict_1D["temp"]), self.N_spec+1))
        self.dict_1D["source_diff"] = np.zeros((len(self.dict_1D["temp"]), self.N_spec+1))
        self.dict_1D["source_adv"] = np.zeros((len(self.dict_1D["temp"]), self.N_spec+1))

        # Reaction source terms
        # Production rates of each species [kmol/m^3/s]
        prod_rates = self.gas_array.net_production_rates
        # Conversion in [1/s] = [kmol/m^3/s * kg/kmol * m^3/kg]
        for j in range(self.N_spec):
            prod_rates[:, j] *= self.dict_1D["Wi"][j]/self.dict_1D["rho"]   
            #prod_rates[:, j] *= 1/self.dict_1D["rho"] 
        self.dict_1D["prod_rates"] = prod_rates
        # Compute reaction source [1/s]
        self.dict_1D["source_react"][:, 1:] = prod_rates
        # special case for T: [K/s] = [kg*K/J * J/kg * 1/s]
        self.dict_1D["source_react"][:,0] = -1/(self.dict_1D["cp"])*np.sum(self.dict_1D["hk"]*prod_rates, axis=1)

        # Diffusion source terms (if transport is in)
        if self.sol_type == "1D_FreeFlame":
            np_gradient(self.dict_1D, self.gas_array)

            # 1/rho*div(Fk)
            for i_sp, sp_name in enumerate(self.species_names):
                self.dict_1D["source_diff"][:, 1+i_sp] = -(1/self.dict_1D["rho"])*self.dict_1D["div_Fk"][:, i_sp]

            # Diffusion source term for temperature: 
            # 1/(rho*cp)*[+div(lambda.grad(T))-div(sum(hk.Fk))+sum(hk.div(Fk))]
            # Thermal diffusion
            self.dict_1D["T_diff_therm"] = (1/(self.dict_1D["rho"]*self.dict_1D["cp"]))*self.dict_1D["div_Q"]
            self.dict_1D["source_diff"][:, 0] = self.dict_1D["T_diff_therm"]
            # Differential diffusion
            self.dict_1D["T_diff_diff"] = -(1/(self.dict_1D["rho"]*self.dict_1D["cp"]))*self.dict_1D["div_hk_Fk"]
            self.dict_1D["source_diff"][:, 0] += self.dict_1D["T_diff_diff"]
            # Diffusion
            self.dict_1D["T_diff"] = np.zeros(self.dict_1D["source_diff"][:, 0].shape)
            for i_sp, sp_name in enumerate(self.species_names):
                self.dict_1D["T_diff"] += (1/(self.dict_1D["rho"]*self.dict_1D["cp"])*
                                          (self.dict_1D["hk"][:,i_sp]*self.dict_1D["div_Fk"][:, i_sp]))
            self.dict_1D["source_diff"][:, 0] += self.dict_1D["T_diff"]

            ## Third get Advection terms
            for i_sp, sp_name in enumerate(self.species_names):
                self.dict_1D["source_adv"][:,1+i_sp] = (-self.dict_1D["x_velocity"]*
                                                   np.gradient(self.dict_1D["Y({})".format(sp_name)], 
                                                               self.dict_1D["X"], edge_order=2))
            self.dict_1D["source_adv"][:,0] = (-self.dict_1D["x_velocity"]*
                                             np.gradient(self.dict_1D["temp"], 
                                                         self.dict_1D["X"], edge_order=2))
        return
    
    ### CSP stuff
    def initializeCSP(self, fields, dict_CSP):
        ## Reinitialize CSP solution base on specified fields
        N_mode = self.N_spec
        if "full_CSP" in fields:
            dict_CSP["lam_log"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["h"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["API_h"] = np.zeros((self.N_sol, N_mode, N_mode))
            dict_CSP["fg"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["API_fg"] = np.zeros((self.N_sol, N_mode, N_mode))
            dict_CSP["fd"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["API_fd"] = np.zeros((self.N_sol, N_mode, N_mode))
            dict_CSP["RPI"] = np.zeros((self.N_sol, N_mode, self.N_reac))
        if "full_TSR" in fields:
            dict_CSP["full_TSR_orig"] = np.zeros(self.N_sol)
            dict_CSP["full_TSR_approx"] = np.zeros(self.N_sol)
            dict_CSP["full_TSR_weights"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["full_TSR_PI"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["full_TSR_SPI"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["full_TSR_RPI"] = np.zeros((self.N_sol, self.N_reac))
        if "chem_TSR" in fields:
            dict_CSP["chem_TSR_orig"] = np.zeros(self.N_sol)
            dict_CSP["chem_TSR_approx"] = np.zeros(self.N_sol)
            dict_CSP["chem_TSR_weights"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["chem_TSR_PI"] = np.zeros((self.N_sol, N_mode))
        if "diff_TSR" in fields:
            dict_CSP["diff_TSR_orig"] = np.zeros(self.N_sol)
            dict_CSP["diff_TSR_approx"] = np.zeros(self.N_sol)
            dict_CSP["diff_TSR_weights"] = np.zeros((self.N_sol, N_mode))
            dict_CSP["diff_TSR_PI"] = np.zeros((self.N_sol, N_mode))
    
    def runCSP(self, fields=["full_TSR"]):
        ## Run CSP in each individual point and produce a new solution dictionnary
        ## for each structure, first index is point index, then mode index, then others
        dict_CSP = {}
        self.initializeCSP(fields, dict_CSP)
        CSP_0D = CSP_class.CSPpyjac(self.gas, self.mechanism)
        for i in range(self.N_sol):
            #print("{}/{}".format(i, self.N_sol))
            statevec = self.getStateVector(i)
            if self.sol_type == "1D_FreeFlame":
                Ld = self.dict_1D["source_diff"][i, :]
            else:
                Ld = np.zeros(self.N_spec)
            CSP_0D.updateDiffSource(Ld)
            CSP_0D.setThermoState(self.P, statevec[0], statevec[1:])
            CSP_0D.updateJacDecomposition()
            if "full_CSP" in fields:
                dict_CSP["lam_log"][i, :] = CSP_0D.lam_log
                dict_CSP["h"][i, :] = CSP_0D.h
                dict_CSP["API_h"][i, :, :] = CSP_0D.API_h
                dict_CSP["fg"][i, :] = CSP_0D.fg
                dict_CSP["API_fg"][i, :, :] = CSP_0D.API_fg
                dict_CSP["fd"][i, :] = CSP_0D.fd
                dict_CSP["API_fd"][i, :, :] = CSP_0D.API_fd
                dict_CSP["RPI"][i, :, :] = CSP_0D.RPI
            if "full_TSR" in fields:
                dict_TSR = CSP_0D.computeTSR(method="full")
                dict_CSP["full_TSR_orig"][i] = dict_TSR["TSR_orig"]
                dict_CSP["full_TSR_approx"][i] = dict_TSR["TSR_approx"]
                dict_CSP["full_TSR_weights"][i,:] = dict_TSR["TSR_weights"]
                dict_CSP["full_TSR_PI"][i,:] = dict_TSR["TSR_PI"]
                dict_CSP["full_TSR_SPI"][i,:] = dict_TSR["TSR_SPI"]
                dict_CSP["full_TSR_RPI"][i,:] = dict_TSR["TSR_RPI"]
            if "chem_TSR" in fields:
                dict_TSR = CSP_0D.computeTSR(method="chemical")
                dict_CSP["chem_TSR_orig"][i] = dict_TSR["TSR_orig"]
                dict_CSP["chem_TSR_approx"][i] = dict_TSR["TSR_approx"]
                dict_CSP["chem_TSR_weights"][i,:] = dict_TSR["TSR_weights"]
                dict_CSP["chem_TSR_PI"][i,:] = dict_TSR["TSR_PI"]
            if "diff_TSR" in fields:
                dict_TSR = CSP_0D.computeTSR(method="diffusion")
                dict_CSP["diff_TSR_orig"][i] = dict_TSR["TSR_orig"]
                dict_CSP["diff_TSR_approx"][i] = dict_TSR["TSR_approx"]
                dict_CSP["diff_TSR_weights"][i,:] = dict_TSR["TSR_weights"]
                dict_CSP["diff_TSR_PI"][i,:] = dict_TSR["TSR_PI"]
        return dict_CSP
    
    def balance_plot(self, ax, spec, x_axis="T"):
        # Plot the ADR balance of selected species or T (requires to have Pele Balance terms and run thermo_qt)
        gas = self.gas
        dict_1D = self.dict_1D
        # look index to use in state vector
        if spec=="T":
            i_spec = 0
        else:
            i_spec = gas.species_index("{}".format(spec)) + 1
            
        # which x axis to use
        if x_axis == "X" or x_axis == "x":
            X = dict_1D["X"]
        else:
            X = dict_1D["temp"]

        # Small validation before plot
        list_keys = list(dict_1D.keys())
        if ("source_react".format(spec) in list_keys):
            ax.set_title("{} balance".format(spec))

            ax.plot(X, dict_1D["source_adv"][:,i_spec], 
                          color="blue", label="A({})".format(spec))
            ax.plot(X, dict_1D["source_diff"][:,i_spec], 
                          color="green", label="D({})".format(spec))
            ax.plot(X, dict_1D["source_react"][:,i_spec], 
                          color="darkred", label="R({})".format(spec))

            sum_spec = (dict_1D["source_adv"][:,i_spec] + 
                        dict_1D["source_diff"][:,i_spec] + 
                        dict_1D["source_react"][:,i_spec])
            ax.plot(X, sum_spec, color="gray", linestyle="-")
            ax.set_xlabel(r"$T$")
        else:
            print("Balance plot is not available for this solution")

        return

def sortModes(dict_CSP, sorting_field="full_TSR_PI", is_CEMA=False):
    ## Sort the CSP dictionnary in descending order of TSR participation index
    ## is_CEMA put explosive modes first if True
    all_fields = list(dict_CSP.keys()) 
    N_point = len(dict_CSP[all_fields[0]])
    
    sorted_dict = {}
    for field in all_fields:
        sorted_dict[field] = np.zeros(dict_CSP[field].shape)
        
    for i in range(N_point):
        # Sort in descending order of TSR_PI
        sort = np.argsort(np.abs(dict_CSP[sorting_field][i]))[::-1]
        # Then put explosive modes first if necessary
        if is_CEMA:
            pos_sort = sort[dict_CSP["lam_log"][i] > 1]
            neg_sort = sort[dict_CSP["lam_log"][i] < 1]
            sort = np.concatenate((pos_sort, neg_sort))
        
        # Order modes in all structures
        for field in all_fields:
            if ("TSR_orig" in field or "TSR_approx" in field or 
                "TSR_SPI" in field or "TSR_RPI" in field):
                sorted_dict[field][i] = dict_CSP[field][i]
            else:
                sorted_dict[field][i] = dict_CSP[field][i, sort]
    
    return sorted_dict
    
def np_gradient(data, gas_array):
    # Perform an interpolation on a uniform grid before computing the second derivative
    # Re-interpolate on original grid afterward (not that smooth :/)
    Xi = gas_array.X
    N, N_spec = Xi.shape
    print(Xi.shape)
    type_int = "quadratic"
    type_int = "cubic"
    
    # Finalize necessary structures
    matrix_Wi_W = np.zeros(data["Di"].shape)
    matrix_rho = np.zeros(data["Di"].shape)
    for i in range(N):
        matrix_Wi_W[i,:] = data["Wi"]
    for j in range(N_spec):
        matrix_Wi_W[:,j] /= data["W"]
        matrix_rho[:,j] = data["rho"]
    
    # Diffusion fluxes: -rho*D*Y*grad(X)
    Xi_grad = np.zeros(Xi.shape)
    for j in range(N_spec):
        Xi_grad[:, j] = np.gradient(Xi[:, j], data["X"])
    data["Fk"] = -(matrix_rho*data["Di"]*matrix_Wi_W*Xi_grad)
    
    # Thermal diffusion flux
    T_grad = np.gradient(data["temp"], data["X"])
    data["Q"] = -data["lambda"]*T_grad
    
    # Sum hk*Fk
    data["sum_hk_Fk"] = np.sum(data["hk"]*data["Fk"], axis=1)
    
    # Take gradient
    data["div_Fk"] = np.zeros(data["Fk"].shape)
    for j in range(N_spec):
        data["div_Fk"][:, j] = np.gradient(data["Fk"][:, j], data["X"])
    data["div_Q"] = -np.gradient(data["Q"], data["X"])
    data["div_hk_Fk"] = np.gradient(data["sum_hk_Fk"], data["X"])
    return

# Functions I do not know where to put ...
def load_reactor(standard_1D, Z, do_CSP=False, N_pts=200, Tmin=None, Tmax=None):
    # Load the 0D solution closest to specified Z and return the useful things
    Z_group = standard_1D.h5pyGrp()
    Z_list = np.array([float(group[2:]) for group in Z_group])
    i_Z = np.argmin(np.abs(Z_list-Z))
    standard_1D.loadGroup(Z_group[i_Z])
    standard_1D.resize_T(N_pts, kind="cubic", Tmin=Tmin, Tmax=Tmax)
    standard_1D.computeSources()
    dict_1D = standard_1D.dict_1D
    if do_CSP:
        dict_modes = standard_1D.runCSP(fields=["full_CSP", "full_TSR"])
        print(Z_group[i_Z])
        dict_CSP = sortModes(dict_modes)
        dict_CEMA = sortModes(dict_CSP, is_CEMA=True)
    else:
        dict_CEMA={}
    return dict_1D, dict_CEMA

def load_freeflame(standard_1D, V, do_CSP=False, N_pts=200, Tmin=None, Tmax=None):
    # Load the 1D solution closest to specified V and return the useful things
    groups = standard_1D.h5pyGrp()
    V_groups = np.array([float(grp[2:]) for grp in groups])
    i_grp = np.argmin(np.abs(V*1e4-V_groups)) 
    V_grp = groups[i_grp]
    V_load = float(V_grp[2:])/1e4
    standard_1D.loadGroup(V_grp)
    standard_1D.resize_T(N_pts, kind="cubic", Tmin=Tmin, Tmax=Tmax)
    standard_1D.computeSources()
    standard_1D.computeSources()
    dict_1D = standard_1D.dict_1D
    if do_CSP:
        dict_modes = standard_1D.runCSP(fields=["full_CSP", "full_TSR"])
        print(V_load)
        dict_CSP = sortModes(dict_modes)
        dict_CEMA = sortModes(dict_CSP, is_CEMA=True)
    else:
        dict_CEMA={}
    return dict_1D, dict_CEMA