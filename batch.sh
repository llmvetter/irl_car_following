#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=9000
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --error="/home/leve469a/maxent_batch-%A_%a.out"
#SBATCH --output="/home/leve469a/maxent_batch-%A_%a.out"
#SBATCH --array=1-4

ml purge
ml release/24.04 GCCcore/11.3.0
ml Python/3.10.4

source /home/h6/leve469a/IQ-Learn/.venv/bin/activate

srun python /home/h6/leve469a/car_following/hp_search.py $SLURM_ARRAY_TASK_ID


