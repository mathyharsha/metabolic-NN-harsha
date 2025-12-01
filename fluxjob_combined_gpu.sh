#!/bin/bash
#SBATCH --job-name=ecoli_samples
#SBATCH --account=project_2013496
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:4

module load python-data
source /scratch/project_2013496/vflux/bin/activate

srun python /scratch/project_2013496/FluxFormer/metabolic-NN-harsha/ecoli_core_UB_combine_inps_all.py
