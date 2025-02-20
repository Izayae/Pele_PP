# for analytical jacobian
import pyjacob
import cantera as ct

import math
import cmath
import numpy as np
from scipy import linalg
import copy

import PyCSP . Functions as csp
import PyCSP . utils as utils

import sys
import time

# Create a class for all my CSP functions to simplify and prepare for the class in Pele

class CSPpyjac():
    def __init__(self, gas=None, mechanism=None):
        # Initialize all attributes with the right structures (just for reference in one place)
        # Cantera attributes
        self.mechanism = mechanism
        print(self.mechanism)
        if gas is not None:
            self.gas = gas
        elif mechanism is not None:
            self.gas = ct.Solution(mechanism)            # Cantera still very useful
        else:
            print("Please provide either a Cantera Solution or a chemical mechanism file")
            sys.exit()
        self.P = 1.0*ct.one_atm                          # constant pressure used (default to 1)
        #gas.species_names         # command useful for species and reaction names
        #gas.reaction_equations()
        self.N_mode = len(self.gas.species_names)           # Number of modes = number of species
        self.N_reac = len(self.gas.reaction_equations())    # Number of reactions
        # Matrix of net stoichiometric coefficients (negative for left side, positive for right side)
        self.net_stoichiometric_matrix = np.zeros((self.N_mode, self.N_reac)) # [J/m3/s for T, m3/kmol for Yi]
        # Matrix of net reaction rates (forward rates - reverse rates)
        self.net_reaction_rates = np.zeros((self.N_reac))   # [kmol/m3/s]
        
        # Input attributes
        self.statevector = np.zeros(self.N_mode)     # Input StateVector
        self.g = np.zeros(self.N_mode)               # Chemical source term
        self.Ld = np.zeros(self.N_mode)             # Diffusion source term (input by the user here)
        #self.L_therm = np.zeros(self.N_mode)         # Thermal contribution for T diffusion
        #self.L_diff = np.zeros(self.N_mode)          # Diffusion source term for T diffusion
        #self.L_diff_diff = np.zeros(self.N_mode)     # Differential diffusion source term for T diffusion
        
        # CSP attributes
        self.jacobian = np.zeros((self.N_mode, self.N_mode))      # Chemical jacobian matrix
        self.eig_val = np.zeros(self.N_mode)                      # Eigenvalues of the jacobian
        self.lam_log = np.zeros(self.N_mode)                      # Log Eigenvalues of the jacobian
        self.A = np.zeros((self.N_mode, self.N_mode))             # Matrix of right eigenvectors
        self.B = np.zeros((self.N_mode, self.N_mode))             # Inverse of A
        self.fg = np.zeros(self.N_mode)                           # Modes chemical amplitude
        self.fd = np.zeros(self.N_mode)                           # Modes diffusion amplitude
        self.h = np.zeros(self.N_mode)                            # Modes global amplitude
        self.API_fg = np.zeros((self.N_mode, self.N_mode))         # Amplitude Participation to fg
        self.API_fd = np.zeros((self.N_mode, self.N_mode))         # Amplitude Participation to fd
        self.API_h = np.zeros((self.N_mode, self.N_mode))          # Amplitude Participation to h
        self.CSP_pointer = np.zeros((self.N_mode, self.N_mode))   # CSP pointer
        self.RPI = np.zeros((self.N_mode, self.N_reac))           # Chemical CSP Reaction Participation Index
        #self.TPI_reac = np.zeros((self.N_mode, self.N_reac))      # Participation index to chemical amplitude
        
        # CSP in complex form
        self.complex_fg = np.zeros(self.N_mode, dtype="complex")                           # Modes chemical amplitude
        self.complex_fd = np.zeros(self.N_mode, dtype="complex")                           # Modes diffusion amplitude
        self.complex_h = np.zeros(self.N_mode, dtype="complex")                            # Modes global amplitude
        self.complex_API_fg = np.zeros((self.N_mode, self.N_mode), dtype="complex")        # Amplitude Participation to fg
        self.complex_API_fd = np.zeros((self.N_mode, self.N_mode), dtype="complex")        # Amplitude Participation to fd
        self.complex_API_h = np.zeros((self.N_mode, self.N_mode), dtype="complex")         # Amplitude Participation to h
        
        # TSR attributes
        # TSR based on source terms only
        self.TSR_orig = 0
        self.ext_TSR_orig = 0
        # TSR based on amplitude squared approximation
        self.TSR_weights = np.zeros(self.N_mode)           # TSR Wi: weight for timescales
        self.TSR_PI = np.zeros(self.N_mode)                # TSR mode participation index
        self.TSR_RPI = np.zeros(self.N_mode)               # TSR reaction participation index (TSR_PI*RPI)
        self.TSR_approx = 0
        self.ext_TSR_approx = 0
        
        # Number of characteristic modes
        self.N_explo = 0                                          # Number of explosive mode
        self.N_significant = 0                                    # Number of mode with significant TSR contribution
    
        return
    
    def setThermoState(self, P, T, Y):
        # Define the state variable vector [T, Y]
        # check which dimension Y has, and remove N2 species if needed
        self.P = P*ct.one_atm
        self.statevector[0] = T
        i_N2 = self.gas.species_index("N2")
        if len(Y) == self.N_mode:
            # N2 species must be removed
            #print("Removing N2 species at index", i_N2)
            if i_N2 == self.N_mode-1:
                self.statevector[1:] = Y[:i_N2]
            else:
                self.statevector[1:] = Y[:i_N2] + Y[i_N2+1:]
        elif len(Y) == self.N_mode-1:
            # Just put it as is and add the N2 value for cantera object
            self.statevector[1:]
            np.insert(Y, i_N2, 1-np.sum(Y))
        else:
            print("Y isn't the right size")
        
        # Update current Cantera Solution object and net production rate
        self.gas.TPY = T, self.P, Y
        self.net_reaction_rates = self.gas.net_rates_of_progress     # [kmol/m3/s]
        self.updateStoichioMatrix()                                  # [J/m3/s for T, m3/kmol for Yi]
        
        # Update chemical source term
        self.updateChemSource()
        
        return
    
    def updateStoichioMatrix(self):
        # Compute the net stoichiometric matrix with right side coefficients positive and left side negative
        # Matrix is of size (N_spec, N_reac) because N2 is removed and T is added
        # Thermodynamic properties
        i_N2 = self.gas.species_index("N2")
        rho = self.gas.density              # [kg/m3]
        cp = self.gas.cp_mass           # [J/kg/K]
        R = ct.gas_constant             # [J/kmol/K]
        if i_N2 == self.N_mode-1:
            nu_right = self.gas.product_stoich_coeffs[:i_N2]
            nu_left = self.gas.reactant_stoich_coeffs[:i_N2]
            W = self.gas.molecular_weights[:i_N2]             # [kg/kmol]
            hspec = self.gas.standard_enthalpies_RT[:i_N2]# non-dimensional
            Hspec = ct.gas_constant * self.statevector[0] * hspec # [J/kmol]
        else:
            nu_right = np.concatenate(self.gas.product_stoich_coeffs[:i_N2], self.gas.product_stoich_coeffs[i_N2+1:])
            nu_left = np.concatenate(self.gas.reactant_stoich_coeffs[:i_N2], self.gas.reactant_stoich_coeffs[i_N2+1:])
            W = np.concatenate(self.gas.molecular_weights[:i_N2], self.gas.molecular_weights[i_N2+1:])  # [kg/kmol]
            hspec = np.concatenate(self.gas.standard_enthalpies_RT[:i_N2], self.gas.standard_enthalpies_RT[i_N2+1:])  # non-dimensional
            Hspec = ct.gas_constant * self.statevector[0] * hspec # [J/kmol]
        
        # N_spec-1 terms (N2 removed)
        
        nu_net = nu_right-nu_left
        for i in range(self.N_mode-1):
            self.net_stoichiometric_matrix[i+1, :] = 1.0/rho*nu_net[i]*W[i]  # [m3/kmol]
        
        # T term
        for k in range(self.N_reac):
            self.net_stoichiometric_matrix[0, k] = -1.0/(rho*cp)*np.sum([nu_net[i, k] * Hspec[i] for i in range(self.N_mode-1)])  # [m3.K/kmol]
        
        return
    
    def updateChemSource(self):
        # Update chemical source term
        pyjacob.py_dydt(0, self.P, self.statevector, self.g)
        return
    
    def updateDiffSource(self, Ld):
        # Update diffusion source term (supposed to be in the pyjacob format already!)
        # T first and no N2
        self.Ld[0] = Ld[0]
        if len(Ld) == self.N_mode+1:
            # N2 species must be removed
            i_N2 = self.gas.species_index("N2")+1
            #print("Removing N2 species at index", i_N2)
            if i_N2 == self.N_mode:
                #print(self.Ld[1:].shape)
                #print(Ld[1:i_N2].shape)
                self.Ld[1:] = Ld[1:i_N2]
            else:
                self.Ld[1:] = Ld[1:i_N2] + Ld[i_N2+1:]
        elif len(Ld) == self.N_mode:
            self.Ld[1:] = Ld[1:]
        else:
            print("Ld isn't the right size")
        return
    
    def updateJacDecomposition(self):
        # update jacobian and run the eigenvalue decomposition of the chemical jacobian
        # Generate analytical jacobian
        jac = np.zeros(self.N_mode * self.N_mode)
        pyjacob.py_eval_jacobian(0, self.P, self.statevector, jac)
        # Unflatten it into an N x N array (need to be transposed to get the right form too)    
        self.jacobian = np.reshape(jac, (self.N_mode, self.N_mode))
        self.jacobian = np.transpose(self.jacobian)
        
        # Eigendecomposition: eigenvalues and right eigenvectors
        lam, R = linalg.eig(self.jacobian, right=True)
        
        # Sort modes by absolute value of eigenvalue
        idx = np.argsort(abs(lam))[::-1]   
        self.eig_val = lam[idx]
        A = R[:,idx]
        B = linalg.inv(A)
        
        # Compute log form for eigenvalues
        self.lam_log = np.sign(np.real(self.eig_val)) * np.log10(1.0 + np.abs(self.eig_val))
        
        # Normalization of A and B with criterion: B.g is real and positive
        L_norm = np.zeros(B.shape, dtype=complex)
        R_norm = np.zeros(A.shape, dtype=complex)
        for i in range(self.N_mode):
            ai = A[:, i]
            bi = B[i]
            bg = np.dot(bi, self.g)
            r, theta = cmath.polar(bg)
            EiTheta = cmath.exp(complex(0, theta))
            L_norm[i] = bi/EiTheta
            R_norm[:,i] = ai*EiTheta
            if np.abs(np.imag(bg))>1:
                #print(i, self.lam_log[i])
                #print(bg, np.dot(L_norm[i], self.g))
                #print("a: {}".format(ai))
                #print("a complex: {}".format(R_norm[:,i]))
                pass
        self.B = L_norm
        self.A = R_norm
        
        # Update amplitudes and participation indices
        self.computeAmplitudes()
        return
    
    def computeAmplitudes(self):
        ## mode amplitudes with the chemical or extended definition
        ## and Amplitude participation index
        time_API = 0
        time_RPI = 0
        i_N2 = self.gas.species_index("N2")
        next_complex = False
        for i in range(self.N_mode):
            # Prepare left eigenvector by taking imaginary part and real part separately
            bi = copy.deepcopy(self.B[i])
            # if bi is complex
            if next_complex == False:
                bi = np.real(bi)
                if np.imag(np.sum(bi))>1e-5:
                    next_complex = True
            else:
                #bi = np.abs(np.imag(bi))   # probably not how it works still..
                bi = np.imag(bi)
                next_complex = False
                           
            # THIS IS VERY WRONG
            #if np.sum(np.abs(np.imag(bi))) > 1e-5:
            #    print("bi before:", bi)
            #for j in range(self.N_mode):
            #    if np.imag(bi[j]) < -1e-5:
            #        bi[j] = -np.imag(bi[j])
            #    else:
                #elif np.imag(bi[j]) > 1e-5:
            #        bi[j] = np.real(bi[j])
            
            # API and amplitudes
            time_start = time.time()
            self.API_fg[i, :] = np.multiply(bi, self.g)
            self.API_fd[i, :] = np.multiply(bi, self.Ld)
            self.API_h[i, :] = self.API_fg[i, :] + self.API_fd[i, :]
            self.fg[i] = np.sum(self.API_fg[i, :])
            self.fd[i] = np.sum(self.API_fd[i, :])
            self.API_fg[i, :] = self.API_fg[i, :]/np.sum(np.abs(self.API_fg[i, :]))
            self.API_fd[i, :] = self.API_fd[i, :]/np.sum(np.abs(self.API_fd[i, :]))
            self.API_h[i, :] = self.API_h[i, :]/np.sum(np.abs(self.API_h[i, :]))
            time_API += time.time()-time_start
            
            # Same in complex form (for testing)
            self.complex_API_fg[i, :] = np.multiply(self.B[i], self.g)
            self.complex_API_fd[i, :] = np.multiply(self.B[i], self.Ld)
            self.complex_API_h[i, :] = self.complex_API_fg[i, :] + self.complex_API_fd[i, :]
            self.complex_fg[i] = np.sum(self.complex_API_fg[i, :])
            self.complex_fd[i] = np.sum(self.complex_API_fd[i, :])
            self.complex_API_fg[i, :] = self.complex_API_fg[i, :]/np.sum(np.abs(self.complex_API_fg[i, :]))
            self.complex_API_fd[i, :] = self.complex_API_fd[i, :]/np.sum(np.abs(self.complex_API_fd[i, :]))
            self.complex_API_h[i, :] = self.complex_API_h[i, :]/np.sum(np.abs(self.complex_API_h[i, :]))
            
            # Reaction participation index
            time_start = time.time()
            for k in range(self.N_reac):
                Sk = self.net_stoichiometric_matrix[:, k]
                Rk = self.net_reaction_rates[k]
                Cik = np.dot(bi, Sk)
                self.RPI[i, k] = Cik*Rk
            # Normalize for each mode
            self.RPI[i, :] = self.RPI[i, :]/np.sum(np.abs(self.RPI[i, :]))
            
            time_RPI += time.time()-time_start
        
        #print("API time: {:.4f}s".format(time_API))
        #print("RPI time: {:.4f}s".format(time_RPI))
        #print("PyCSP time: {:.4f}s".format(time.time()-time_start))
        self.h = self.fg + self.fd
        self.complex_h = self.complex_fg + self.complex_fd
        #for i in range(self.N_mode):
        #    if np.abs(self.h[i]-np.abs(self.complex_h[i])) > 1:
        #        print(i)
        #        print("API_h / complex_API_h: ", self.API_h[i, :], self.complex_API_h[i, :])
        #        print("h / R(h_complex) / I(h_complex)): ", self.h[i], np.real(self.complex_h[i]), np.imag(self.complex_h[i]))
        return
    
    def computeTSR(self, method="full"):
        ## Get chemical TSR and corresponding participation indices
        ## Available method: chemical, diffusion, full (default to full if method not recognized)
        if method == "chemical":
            source = self.g
            amps = self.fg
        elif method == "diffusion":
            source = self.Ld
            amps = self.fd
        else:
            # default behaviour with all source terms
            source = self.g + self.Ld
            amps = self.h
        
        # TSR based on source terms only
        tau_norm = source/linalg.norm(source)
        dict_TSR = {}
        #np.dot(np.matmul(np.transpose(tau_norm), jac2d), tau_norm)
        dict_TSR["TSR_orig"] = np.matmul(np.matmul(np.transpose(tau_norm), self.jacobian), tau_norm)
        dict_TSR["TSR_orig"] = np.sign(np.real(dict_TSR["TSR_orig"])) * np.log10(1.0 + np.abs(dict_TSR["TSR_orig"]))
        
        ## TSR based on amplitude squared approximation
        # Compute TSR weights
        source_norm = linalg.norm(source)
        dict_TSR["TSR_weights"] = np.zeros(self.N_mode)
        dict_TSR["TSR_PI"] = np.zeros(self.N_mode)
        next_complex = True
        for i in range(self.N_mode):
            ai = copy.deepcopy(self.A[:,i])
            if next_complex == False:
                ai = np.real(ai)
                if np.imag(np.sum(ai))>1e-5:
                    next_complex = True
            else:
                ai = np.imag(ai)
                next_complex = False
            
            """
            # Wrong way of doing it !
            for j in range(self.N_mode):
                if np.imag(ai[j]) < -1e-5:
                    ai[j] = -np.imag(ai[j])
                else:
                #elif np.imag(ai[j]) > 1:
                    ai[j] = np.real(ai[j])
            """
            
            # A lot of TSR definition try:
            #dict_TSR["TSR_weights"][i] = amps[i]/source_norm * np.dot(source, ai)/source_norm
            #dict_TSR["TSR_weights"][i] = np.abs(amps[i]/source_norm) * np.dot(source, ai)/source_norm
            #dict_TSR["TSR_weights"][i] = np.abs(amps[i]/source_norm * amps[i]/source_norm) * np.sign(np.dot(source, ai))
            dict_TSR["TSR_weights"][i] = np.abs(amps[i]/source_norm * amps[i]/source_norm)
            #dict_TSR["TSR_weights"][i] = np.abs(amps[i])/source_norm * np.abs(amps[i])/source_norm
        #print("TSR weights sum: {:.2f}".format(np.sum(np.abs(self.TSR_weights))))
        dict_TSR["TSR_weights"] = dict_TSR["TSR_weights"]/np.sum(np.abs(dict_TSR["TSR_weights"]))     # modified TSR weights, make sure sum is 1
        
        # Compute TSR participation index
        dict_TSR["TSR_PI"] = dict_TSR["TSR_weights"]*np.sign(np.real(self.eig_val))*np.abs(self.eig_val)   # individual terms in sum of TSR
        #dict_TSR["TSR_PI"] = dict_TSR["TSR_weights"]*np.abs(self.eig_val)   # individual terms in sum of TSR
        dict_TSR["TSR_approx"] = np.sum(dict_TSR["TSR_PI"])
        dict_TSR["TSR_PI"] = dict_TSR["TSR_PI"]/np.sum(np.abs(dict_TSR["TSR_PI"]))          # Normalize it aftwerward
        dict_TSR["TSR_approx"]  = np.sign(np.real(dict_TSR["TSR_approx"])) * np.log10(1.0 + np.abs(dict_TSR["TSR_approx"]))
        
        # Species participation index
        TSR_PI = dict_TSR["TSR_PI"]
        dict_TSR["TSR_SPI"] = np.zeros(self.N_mode)
        for j in range(self.N_mode):
            API_j = self.API_h[:, j]
            dict_TSR["TSR_SPI"][j] = np.dot(TSR_PI, API_j)
        dict_TSR["TSR_SPI"] = dict_TSR["TSR_SPI"]/np.sum(np.abs(dict_TSR["TSR_SPI"]))
        
        # Reaction participation index
        dict_TSR["TSR_RPI"] = np.zeros(self.N_reac)
        for k in range(self.N_reac):
            RPI_k = self.RPI[:, k]
            dict_TSR["TSR_RPI"][k] = np.dot(TSR_PI, RPI_k) 
        dict_TSR["TSR_RPI"] = dict_TSR["TSR_RPI"]/np.sum(np.abs(dict_TSR["TSR_RPI"]))
        
        return dict_TSR
    
    def computeRPI(self, method="full"):
        ## Get reaction participation index (to chemical amplitude)
        for i in range(self.N_mode):
            bi = self.B[i]
            for k in range(self.N_reac):
                Sk = self.net_stoichiometric_matrix[:, k]
                Rk = self.net_reaction_rates[k]
                self.RPI[i, k] = np.dot(bi, Sk)*Rk                
            # Normalize so that sum for each mode is 1
            self.RPI[i, :] = self.RPI[i, :]/np.sum(np.abs(self.RPI[i, :]))
        return