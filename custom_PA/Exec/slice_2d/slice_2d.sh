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
code=slice_2d
# Compilation (in parallel)
make -j10 EBASE=${code}

# Parameters
lev=2
n_files=1

# Where to write the output (put it where you want, it's just an example)
rel_path="$(pwd)/../../Results/slice_2d/tuto_bunsen/"
mkdir -p ${rel_path}
path_output=$(realpath "${rel_path}") 

echo
echo "output relative path: " $rel_path
echo "output absolute path: "$path_output
echo

# New quantities to compute (only "x", "y", "z", "mag_velocity", etc...)
new_quantities=("FI_CH4_O2")

# Pattern of the pltfiles to find (can be adapted to only a subset of the plotfiles with pattern recognition)
#name_find="\( -name plt0???? \)"   # get all plotfiles in the format plt0????
#name_find="\( -name plt02508 -o -name plt01730\)"  # union of pattern: get plt00000 and plt01730 specifically
name_find="\( -name plt03??? \)"   # get all plotfiles from plt03000 to plt03999

# Function for the repetitive tasks to find all pltfiles matching the pattern
function find_plt() {
    plt_list=() 
    command="find $*"
    echo $command

    # Get the unsorted array of plotfiles
    while IFS=  read -r -d $'\0'; do
        plt_list+=("$REPLY")
    done < <(eval "${command}" -print0)
    
    # Print number of plotfiles in the directory
    eval "${command}" | wc -l
    
    # Sort the array
    IFS=$'\n' sort_plt=($(sort <<<"${plt_list[*]}")); unset IFS
}

sort_plt=()

# Files 1
path_list_1=$(realpath "$(pwd)/../../../dataset_Bunsen_3D")
Var1=("mixture_fraction" "temp" "HeatRelease" "Y(CH4)" "Y(O2)")

find_plt ${path_list_1} ${name_find}
plt_list_1=("${sort_plt[@]}")

# Files 2
#path_list_2="/scratch/b/bsavard/mavab/Reduced_data/plt_full/DNS_HTC_premixer/level_3/full_prog/"
#Var2=("CEM" "diff_proj" "react_proj" "c_temp" "alpha" "strain_rate" "vorticity"\
#      "FI_CH4_O2" "FI_DME_O2" "c_SDR" "z_SDR" "cross_SDR" "flame_thickness")
#find_plt ${path_list_2} ${name_find}
#plt_list_2=("${sort_plt[@]}")

# Format string for giving path_file and Var for each file selected
infile=""
for i_file in $(seq 1 $n_files) ; do
    path_file="path_list_$i_file"
    eval path=\( \${${path_file}} \)
    name_Var="Var$i_file"
    eval Var=\( \${${name_Var}[@]} \)
    echo ${path_file}
    echo "reading variables ${Var[@]} in ${path}"
    infile+="Path_file${i_file}=${path} Var${i_file}=${Var[@]} "
done

# Verify that list of plotfiles found for all files are matching
declare -a plt_file_names=()
#echo ${plt_list_1[@]}
echo ${#plt_list_1[@]}
for i_plt in "${!plt_list_1[@]}"; do
    name_plt_list_1="plt_list_$i_file"
    eval plt_1=\( \${${name_plt_list_1}[i_plt]} \)
    name_plt_1=$(basename "${plt_1}")
    for i_file in $(seq 2 $n_files); do
        name_plt_list_i="plt_list_$i_file"
        eval plt_i=\( \${${name_plt_list_i}[i_plt]} \)
        name_plt_i=$(basename "${plt_i}")
        if [[ "${name_plt_1}" != "${name_plt_i}" ]]
        then
            echo "Plot file names do not match!"
            echo ${name_plt_1}
            echo ${name_plt_i}
            exit 1
        fi
    done
    plt_file_names+=(${name_plt_1})
done

echo ${plt_file_names[@]}

# This command run the code with one plotfile given per run
for plt in "${plt_file_names[@]}"; do
    echo $plt
    mpiexec ${code}3d.gnu.MPI.ex input_${code} $infile Plt_file_names=${plt} finest_level=${lev} n_files=${n_files} \
                                 path_output=${path_output} new_quantities=${new_quantities[@]}
done
