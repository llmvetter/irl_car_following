#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=6000
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH -J jobname
#SBATCH --error="/home/leve469a/maxent_train3-%J.out"
#SBATCH --output="/home/leve469a/maxent_train3-%J.out"

ml purge
ml release/24.04 GCCcore/11.3.0
ml Python/3.10.4

source /home/h6/leve469a/IQ-Learn/.venv/bin/activate

srun python /home/leve469a/car_following/runner.py