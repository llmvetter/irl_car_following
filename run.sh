#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=6000
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH -J jobname
#SBATCH --error="/home/leve469a/myjob-%J.out"
#SBATCH --output="/home/leve469a/myjob-%J.out"

ml purge 

ml release/24.04  GCC/13.2.0 OpenMPI/4.1.6 SciPy-bundle/2023.11
ml PyTorch/2.1.2

srun python /home/leve469a/car_following/policy.py