# Just a dump of function to simplify my post-processing codes
# It's very specific to my use cases

def labelTable(var):
    # Name to put in the plot labels from simulation name
    # Same for LTC and HTC
    label_table = {"mixture_fraction": "Mixture fraction",
                   "Scalar_diss": "Scalar dissipation rate",
                   "HeatRelease": "Heat release rate", 
                   "temp": "Temperature",
                   "CEM": "Chemical explosive mode",
                   "FI_DME_O2": "DME flame index",
                   "FI_CH4_O2": "CH4 flame index",
                   "Y_HO2": "Y(HO2)",
                   "Y_H2O2": "Y(H2O2)",
                   "Y_CH3OCH2O2": "Y(RO2)",
                   "Y_OH": "Y(OH)",
                   "Y_CH4": "Y(CH4)",
                   "Y_CH3OCH3": "Y(CH3OCH3)",
                   "enthalpy": "Enthalpy",
                   "time": "Time",
                   "x_velocity": "$U_x$",
                   "y_velocity": "$U_y$",
                   "z_velocity": "$U_z$",
                   "mag_velocity": "$U_{mag}$",
                   "x": "x/D",
                   "y": "y/D",
                   "z": "z/D",
                   "x_mean": "x/D",
                   "y_mean": "y/D",
                   "z_mean": "z/D",
                   "x_min": "x/D",
                   "y_min": "y/D",
                   "z_min": "z/D",
                   "x_max": "x/D",
                   "y_max": "y/D",
                   "z_max": "z/D",
                   "alpha": "Combustion mode index",
                   "c_temp": "Progress variable",
                   "c_DME": r"c_{1}",
                   "c_CH4": r"c_{2}",
                   "norm_HRR_DME": r"HRR$_{1}$/HRR$_{1,max}^{0D}$",
                   "norm_HRR_CH4": r"HRR$_{2}$/HRR$_{2,max}^{0D}$",
                   "norm_HRR_DME_1D": r"HRR$_{1}$/HRR$_{1,max}^{1D}$",
                   "norm_HRR_CH4_1D": r"HRR$_{2}$/HRR$_{2,max}^{1D}$",
                   "flame_thickness": "Flame thickness",
                   "c_SDR": "SDR of progress variable",
                   "c_DME_SDR": r"c-SDR$_1$",
                   "c_CH4_SDR": r"c-SDR$_2$",
                   "norm_c_DME_SDR_1D": r"c-SDR$_1$/SDR$_{1,max}^{1D}$",
                   "norm_c_CH4_SDR_1D": r"c-SDR$_2$/SDR$_{2,max}^{1D}$",
                   "z_SDR": "SDR",
                   "cross_SDR": "SDR of cross-scalar",
                   "strain_rate": "Strain rate",
                   "vorticity" : "Vorticity",
                   "diff_proj": r"$\phi_{s}$",
                   "react_proj": r"$\phi_{\omega}$"}
    if var not in label_table:
        label = var
    else:
        label = label_table[var]
    return label

def normTable(var, case="HTC"):
    # value used to normalize the variables
    # different for LTC and HTC
    if case=="HTC":
        tau_flame_sto = 3.5e-6
        tau_MR = 1.1105194806230339e-4
        D=3e-4
        Uin=12
        Ujet=53
        norm_table =  {"ix":11/528,
                       "iy":3/144,
                       "iz":4/192,
                       "mixture_fraction": 1,
                       "Scalar_diss": 1,
                       "HeatRelease": 1, 
                       "temp": 1/900,
                       "CEM": 1,
                       "FI_DME_O2": 1,
                       "FI_CH4_O2": 1,
                       "Y_HO2": 1,
                       "Y_H2O2": 1,
                       "Y_CH3OCH2O2": 1,
                       "Y_OH": 1,
                       "Y_CH4": 1,
                       "Y_CH3OCH3": 1,
                       "enthalpy": 1,
                       "time": 1,
                       "x_velocity": 1/Ujet,
                       "y_velocity": 1/Ujet,
                       "z_velocity": 1/Ujet,
                       "mag_velocity": "Umag",
                       "x": 1/D,
                       "y": 1/D,
                       "z": 1/D,
                       "x_mean": 1/D,
                       "y_mean": 1/D,
                       "z_mean": 1/D,
                       "x_min": 1/D,
                       "y_min": 1/D,
                       "z_min": 1/D,
                       "x_max": 1/D,
                       "y_max": 1/D,
                       "z_max": 1/D,
                       "alpha": 1,
                       "c_temp": 1,
                       "c_DME": 1,
                       "c_CH4": 1,
                       "flame_thickness": 1,
                       "c_SDR": 1,
                       "z_SDR": tau_MR,
                       "z_SDR_min": tau_MR,
                       "z_SDR_mean": tau_MR,
                       "z_SDR_max": tau_MR,
                       "norm_c_DME_SDR_1D": 1,
                       "norm_c_CH4_SDR_1D": 1,
                       "cross_SDR": 1,
                       "strain_rate": 1,
                       "vorticity" : 1,
                       "diff_proj": 1,
                       "react_proj": 1}
    elif case=="LTC":
        tau_flame = 2e-5
        D=5e-4
        Uin=6
        Ujet=30
        norm_table =  {"ix":11/528,
                       "iy":3/144,
                       "iz":4/192,
                       "mixture_fraction": 1,
                       "Scalar_diss": 1,
                       "HeatRelease": 1, 
                       "temp": 1/900,
                       "CEM": 1,
                       "FI_DME_O2": 1,
                       "FI_CH4_O2": 1,
                       "Y_HO2": 1,
                       "Y_H2O2": 1,
                       "Y_CH3OCH2O2": 1,
                       "Y_OH": 1,
                       "Y_CH4": 1,
                       "Y_CH3OCH3": 1,
                       "enthalpy": 1,
                       "time": 1,
                       "x_velocity": 1/Ujet,
                       "y_velocity": 1/Ujet,
                       "z_velocity": 1/Ujet,
                       "mag_velocity": "Umag",
                       "x": 1/D,
                       "y": 1/D,
                       "z": 1/D,
                       "alpha": 1}
    elif case==None:
        norm_table = {var:1}
    else:
        print("not yet implemented")
        norm_table = {var:1}
        
    if var in norm_table:
        norm = norm_table[var]
    else:
        norm = 1
    return norm

from matplotlib.colors import *
def scaleTable(var, case="HTC"):
    # Scales for 2d plot with imshow
    # different for LTC and HTC
    if case=="HTC":
        scale_table = {"mixture_fraction": Normalize(0, 0.13),
                       "Scalar_diss": LogNorm(1e-2, 5e3),
                       "HeatRelease": LogNorm(1e10, 4e12), 
                       "temp": Normalize(900, 2700),
                       "CEM": Normalize(-7, 7),
                       "extended_TSR": Normalize(-7, 7),
                       "CEM_MPI": Normalize(0, 1),
                       "source_reaction(T)": LogNorm(1e6, 6e8),
                       "source_diffusion(T)": Normalize(-3e8, 3e8),
                       "CEM_chem_amp": LogNorm(1e6, 1e9),
                       "CEM_diff_amp": Normalize(-3e8, 3e8),
                       "extended_TSR": Normalize(-7, 7),
                       "FI_DME_O2": Normalize(-1, 1),
                       "FI_CH4_O2": Normalize(-1, 1),
                       "Y_HO2": Normalize(),
                       "Y_H2O2": Normalize(),
                       "Y_CH3OCH2O2": Normalize(),
                       "Y_OH": Normalize(),
                       "Y_CH4": Normalize(),
                       "Y_CH3OCH3": LogNorm(1e-5, 1e-1),
                       "enthalpy": Normalize(),
                       "time": Normalize(),
                       "x_velocity": Normalize(),
                       "y_velocity": Normalize(),
                       "z_velocity": Normalize(),
                       "mag_velocity": Normalize(),
                       "x": Normalize(0, 0.0033),
                       "y": Normalize(0, 0.0009),
                       "z": Normalize(0, 0.0012),
                       "alpha": Normalize(-3, 3),
                       "c_temp": Normalize(0, 1),
                       "flame_thickness": LogNorm(1e-6, 1e-3),
                       "c_SDR": LogNorm(1, 1e7),
                       "z_SDR": LogNorm(1e-2, 1e4),
                       "norm_c_DME_SDR_1D": Normalize(),
                       "norm_c_CH4_SDR_1D": Normalize(),
                       "cross_SDR": LogNorm(1e-3, 1e6),
                       "strain_rate": LogNorm(1, 1e9),
                       "vorticity" : LogNorm(1, 1e9),
                       "diff_proj": Normalize(),
                       "react_proj": Normalize(),
                       "pdf": LogNorm(10, 1e7)}
    elif case=="LTC":
        scale_table = {"mixture_fraction": Normalize(0, 0.10),
                       "Scalar_diss": LogNorm(1e-3, 1e3),
                       "HeatRelease": LogNorm(1e6, 1e11), 
                       "temp": Normalize(750, 950),
                       "CEM": Normalize(-7, 7),
                       "extended_TSR": Normalize(-7, 7),
                       "CEM_MPI": Normalize(0, 1),
                       "source_reaction(T)": LogNorm(1e6, 6e8),
                       "source_diffusion(T)": Normalize(-3e8, 3e8),
                       "CEM_chem_amp": LogNorm(1e3, 1e6),
                       "CEM_diff_amp": Normalize(-1e6, 1e6),
                       "extended_TSR": Normalize(-7, 7),
                       "FI_DME_O2": Normalize(-1, 1),
                       "FI_CH4_O2": Normalize(-1, 1),
                       "Y_HO2": Normalize(),
                       "Y_H2O2": Normalize(),
                       "Y_CH3OCH2O2": Normalize(),
                       "Y_OH": Normalize(),
                       "Y_CH4": Normalize(),
                       "Y_CH3OCH3": LogNorm(),
                       "enthalpy": Normalize(),
                       "time": Normalize(),
                       "x_velocity": Normalize(),
                       "y_velocity": Normalize(),
                       "z_velocity": Normalize(),
                       "mag_velocity": Normalize(),
                       "x": Normalize(0, 0.0055),
                       "y": Normalize(0, 0.0015),
                       "z": Normalize(0, 0.0020),
                       "alpha": Normalize(-1, 1),
                       "pdf": LogNorm(1, 1e8)}
    elif case==None:
        scale_table = {var:Normalize()}
    else:
        print("not yet implemented")
        scale_table = {var:Normalize()}
        
    if var in scale_table:
        scale = scale_table[var]
    else:
        scale = Normalize()
    return scale