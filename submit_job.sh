#!/bin/bash
#SBATCH --job-name=mhd_data_generation
#SBATCH --account=bbjo-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00 
#SBATCH --mem=240G 
#SBATCH --output=slurmlogs/mhd_datagen_%j.log
#SBATCH --error=slurmlogs/mhd_datagen_%j.err
#SBATCH --exclusive  # Highly recommended
# #SBATCH --nodelist=cn075  # Optional, but recommended if there's availability

module purge
module load anaconda3_cpu/23.7.4

source /sw/external/python/anaconda3/etc/profile.d/conda.sh # Just in case it's not included in .bashrc 
conda activate mhd  # Run data generation with independent data-generation environment to prevent dependency issues

# Example
python -m data_generation.main -N 10 -o "/work/hdd/bbjo/skacmaz/NODs_mhd_data_Re250" --Re 2.5e2 --ReM 2.5e2

conda deactivate
