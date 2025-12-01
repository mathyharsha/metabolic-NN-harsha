#!/bin/bash
#SBATCH --job-name=ecoli_samples
#SBATCH --account=project_2013496
#SBATCH --time=15:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load python-data
source /scratch/project_2013496/vflux/bin/activate

srun python /scratch/project_2013496/FluxFormer/metabolic-NN-harsha/ecoli_core_UB_combine_inps_all.py
