#!/bin/bash
#SBATCH -J slice_3
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=martin.vabre@polymtl.ca

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=40
#SBATCH --time=2:00:00

# Load the correspondent modules on your system if you're on a different supercomputer
module load NiaEnv/2019b gcc/9.2.0 openmpi gnu-parallel python/3.9

# Name of the current code (for compilation)
code=stats
# Compilation (in parallel)
make -j10 EBASE=${code}

# Parameters
lev=2         # maximum level to use
n_files=1

# Where to write the output (put it where you want, it's just an example)
rel_path="$(pwd)/../../Results/stats/tuto_bunsen/"
mkdir -p ${rel_path}
path_output=$(realpath "${rel_path}") 

echo
echo "output relative path: " $rel_path
echo "output absolute path: "$path_output
echo

# New quantities to compute (only "x", "y", "z", "mag_velocity", etc...)
new_quantities=("FI_CH4_O2" "z")

# Files 1
plt_name="plt04735"
path_plt_1=$(realpath "$(pwd)/../../../dataset_Bunsen_3D")
Var1=("mixture_fraction" "temp" "HeatRelease" "Y(CH4)" "Y(O2)")

# This command run the code for the selected plotfile (a bit overcomplicated but it works)
# if the input file name is changed, change it
name_input="input_stats"
mpiexec ${code}3d.gnu.MPI.ex ${name_input} n_files=${n_files} Path_file1=${path_plt_1} Var1=${Var1[@]} \
                             Plt_file_names=${plt_name} finest_level=${lev} \
                             new_quantities=${new_quantities[@]} path_output=${path_output} 

