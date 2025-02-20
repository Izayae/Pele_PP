# Data manipulation
import numpy as np
import scipy
from scipy import optimize
from scipy import interpolate
from scipy.signal import find_peaks

# Graph library
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Cantera
import cantera as ct

# File management library
import pathlib
import h5py as h5
import glob
import os

# ------------------ Actual code ------------------
class Cantera0dGenerator:
    # This class is to finally create my 0d homogeneous reactor easily
    def __init__(self, mixture_fraction,
                 T_oxi=900, T_fuel=300, P=1,
                 oxi="O2:0.21, N2:0.79", fuel="CH4:0.5, CH3OCH3:0.5", 
                 offset=0.0, case=None, mechanism="dme38.yaml"):
        # Initialize parameters
        self.Z = mixture_fraction       # mixture fraction of the mixture
        self.offset = offset            # offset in noramalized time
        self.case = case                # which case is it
        self.mechanism = mechanism      # mechanism used for the reactor (must be in the code directory)
        
        # Initialize thermochemical parameters
        self.T_oxi = T_oxi                    # oxidizer temperature
        self.T_fuel = T_fuel                  # fuel temperature
        self.P = P*ct.one_atm                 # pressure (in atm)
        self.oxi = oxi                        # oxidizer composition (molar) in cantera format
        self.fuel = fuel                      # fuel composition (molar) in cantera format
        self.gas = ct.Solution(mechanism)     # load the solution for later use (saves time)
        
        # Initialize solution as a function of the specified case
        self.states = self.initializeState()
        
        return
    
    # |--------------------------------------------|
    # |          Initialization functions          |
    # |--------------------------------------------|
    
    def initializeState(self):
        # Initialize the state for the beginning of the main homogeneous reactor simulation
        # for the case specified
        if self.case == None:
            # LTC is easy because the initial mixture is used
            # Do the mixing and initialize SolutionArray
            self.gas.TPX = self.T_fuel, self.P, self.fuel
            self.adiabaticMixing(self.Z)
            states = ct.SolutionArray(self.gas, extra=['tres'])
        # Can add different scenario here if necessary
        elif self.case == "DME-HTC":
            # DME HTC needs to create a first mixture at z=0.13, advance it to a specific time
            # and then mix again to the desired mixture
            # Run a case from the start for long enough
            self.gas.TPX = self.T_fuel, self.P, self.fuel
            self.adiabaticMixing(0.13)
            states = ct.SolutionArray(self.gas, extra=['tres'])
            self.states = states
            self.runReactor(1)
            
            # Get the solution at the required time
            t_adv = 1.03727e-3
            adv_gas = self.getInterpolatedSolutionTime(t_adv + self.offset)
            
            self.gas.TPX = adv_gas.T, adv_gas.P, adv_gas.X
            self.adiabaticMixing(self.Z)
            states = ct.SolutionArray(self.gas, extra=['tres'])
        elif self.case == "CH4-HTC":
            # Same as DME-HTC but advance for longer to isolate CH4 flame front
            self.gas.TPX = self.T_fuel, self.P, self.fuel
            self.adiabaticMixing(0.13)
            self.runReactor(1)
            
            # Get the solution at HTC time and mix
            t_adv = 1.03727e-3
            adv_gas = self.getInterpolatedSolutionTime(t_adv)
            self.gas.TPX = adv_gas.T, adv_gas.P, adv_gas.X
            self.adiabaticMixing(self.Z)
            
            # Advance the reaction a bit more (until local minimum of HRR) and update gas
            self.runReactor(1e-1)
            peaks, _ = find_peaks(self.states.heat_release_rate[:], height=np.max(self.states.heat_release_rate)/20)
            mins, _ =find_peaks(self.states.heat_release_rate[peaks[0]:peaks[1]]*-1)
            mins += peaks[0]
            i_min = mins[np.argmax(self.states.heat_release_rate[mins])]
            t_adv_hot = self.states.tres[i_min]
            print("{:.3e}".format(t_adv_hot))
            adv_gas_hot = self.getInterpolatedSolutionTime(t_adv_hot)
            self.gas.TPX = adv_gas_hot.T, adv_gas_hot.P, adv_gas_hot.X
            states = ct.SolutionArray(self.gas, extra=['tres'])
        else:
            raise Exception("{} is not a case supported".format(self.case))          
        
        return states
    
    def findOxiQt(self, x, Z, sol_oxi, sol_fuel):
        # Function for Newton solver to find the required quantity of oxidizer for the right mixing
        sol_oxi.moles = x
        sol_mix = sol_oxi + sol_fuel
        sol_Z = sol_mix.mixture_fraction(self.fuel, self.oxi)
        return sol_Z - Z
    
    def adiabaticMixing(self, Z):
        # Provided oxidizer and fuel thermochemical states and a mixture fraction,
        # compute the mixture resulting from an adiabatic mixing (HP constant)
        
        # Oxidizer solution
        sol_oxi = ct.Quantity(ct.Solution(self.mechanism), constant='HP')
        sol_oxi.TPX = self.T_oxi, self.P, self.oxi
        
        # Fuel solution (from the gas created before the function
        sol_fuel = ct.Quantity(self.gas, constant='HP')
        
        # Find the right quantity of oxidizer to mix with fuel to obtain the specified mixture fraction
        if Z == 0.0:
            sol_fuel.moles = 0
            sol_oxi.moles = 1
            print("The mixture is full oxidizer")
        else:
            sol_fuel.moles = 1
            sol_oxi.moles = optimize.newton(self.findOxiQt, 0, tol=1e-8, maxiter=1000, args=(Z, sol_oxi, sol_fuel))
            print("{} moles of oxidizer for 1 mole of fuel".format(sol_oxi.moles))
        
        # Create the mixture
        sol_mix = sol_oxi + sol_fuel
        
        # Create a new Solution to initialize the SolutionArray for the run
        self.gas.TPX = sol_mix.T, self.P, sol_mix.X
        
        return
    
    # |--------------------------------------------|
    # |              Reactor functions             |
    # |--------------------------------------------|
    def runReactor(self, time_end):
        # Run the homogeneous reactor with base cantera parameters
        # gas 
        # Initialize reactor
        combustor = ct.ConstPressureReactor(self.gas)
        #combustor = ct.IdealGasConstPressureReactor(self.gas)
        combustor.volume = 1.0
        sim = ct.ReactorNet([combustor])
        
        # Add the initial state to the solution array (time 0)
        t = 0.0
        self.states = ct.SolutionArray(self.gas, extra=['tres'])
        self.states.append(combustor.thermo.state, tres=t)
        while (t<time_end):
            # Advance reactor
            t = sim.step()

            # Append everything useful to the solutionArray
            self.states.append(combustor.thermo.state, tres=t)
        
        return
    
    # |--------------------------------------------|
    # |          Transformation functions          |
    # |--------------------------------------------|
    def timeToSpace(self, V_in):
        # Convert the currently loaded solution from time to physical space given an inlet velocity
        # Return the time-converted spacing grid assuming full auto-ignitive 1D flame
        
        # Get velocity profile from inlet velocity and density
        # from momentum conservation (V*rho=Cst)
        rho = self.getFieldSolution("density")
        rho_0 = rho[0]
        V = V_in*rho_0/rho
        
        # convert time solution to physical space
        # from relation time-velocity: x(t) = int_0^t(V(t)dt)
        t = self.getFieldSolution("time")
        X = np.zeros(t.shape)
        for i in range(1, len(t)):
            X[i] = X[i-1] + (V[i]+V[i-1])/2*(t[i]-t[i-1])
        
        return X
    
    
    # |--------------------------------------------|
    # |              Output functions              |
    # |--------------------------------------------|
    def getInterpolatedSolutionTime(self, time):
        # Interpolate the solution to a single time within the range of simulation time
        solution = ct.Solution(self.mechanism)
        
        # Interpolate the temperature
        T = interpolate.interp1d(self.states.tres, self.states.T, kind="linear")(time)
              
        # Interpolate the molar fraction
        name_species = solution.species_names
        X = np.zeros(len(name_species))
        for i, species in enumerate(name_species):
            X[i] = interpolate.interp1d(self.states.tres, self.states(species).X[:,0], kind="cubic")(time)
        
        # Create the new solution at time
        solution.TPX = T, self.gas.P, X
        
        return solution
    
    def getFieldSolution(self, name_field):
        name_species = self.gas.species_names
        if name_field == "temp":
            sol_field = self.states.T
        elif name_field == "time":
            sol_field = self.states.tres
        elif name_field == "HRR":
            sol_field = self.states.heat_release_rate
        elif name_field == "density":
            sol_field = self.states.density_mass
        elif name_field in name_species:
            sol_field = self.states(name_field).Y[:,0]
        else:
            raise("{} is not a supported field".format(name_field))
            
        return sol_field
        
    def getUniformGridSolution(self, name_X, name_Y, n_points=10000, min_X=None, max_X=None, method="linear"):
        # Identify x and y
        sol_X = self.getFieldSolution(name_X)
        sol_Y = self.getFieldSolution(name_Y)
        
        # Get the extremas
        if min_X == None:
            min_X = np.min(sol_X)
        if max_X == None:
            max_X = np.max(sol_X)
        min_Y = np.min(sol_Y)
        max_Y = np.max(sol_Y)
        #print(min_X, max_X)
        #print(min_Y, max_Y)
        
        # Compute the interpolated solution on uniform grid of sol_X
        new_X = np.linspace(min_X, max_X, n_points)
        new_Y = interpolate.interp1d(sol_X, sol_Y, kind="linear", bounds_error=False, fill_value=None)(new_X)
        
        return new_X, new_Y
    
    def writeSolutionH5(self, path_output):
        # Build the final path for file
        if self.case == None:
            final_path = path_output + "/"
        else:
            final_path = path_output + self.case + "/"
        
        # Create the directory for the file
        pathlib.Path(final_path).mkdir(parents=True, exist_ok=True)
        
        # Add the data to the h5 file
        group = "Z_{:.4f}".format(self.Z)
        self.states.write_hdf(final_path + "solution_reactor.h5", group=group)
        
        print("Solution added to {} in group {}".format(final_path + "solution_reactor.h5", group))
        return
        