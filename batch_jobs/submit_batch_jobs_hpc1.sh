#!/bin/sh

## Set number of nodes for the job
#SBATCH -N 1

## Set number of tasks for the job
#SBATCH -n 1

## Specify to only use GPU nodes; no. per node
##SBATCH --gres=gpu:1

## Specify requested time
## day-hr
#SBATCH -t 7-00:00:00

## Specify stdout, stderr log; default is slurm-jobid.out
#SBATCH -o output.log
#SBATCH -e error.log

## Specify what to notify: BEGIN, END, FAIL...., ALL
##SBATCH --mail-type=FAIL
##SBATCH --mail-user arbint@bnl.gov

echo ""
echo "*********************************************************"
echo "This was run on:"
date
echo "*********************************************************"
echo ""
## Load cuda module for GPU job
module load anaconda2
module load cuda
source activate envDeepLearningWithProtoDUNE
cd ../

echo "*********************************************************"
echo "Running python train_model.py"
echo "JOB $SLURM_JOB_ID is running on $SLURM_JOB_NODELIST "
echo "*********************************************************"
echo ""
srun python train_model.py -o Training -e Default

echo "*********************************************************"
echo "Running python analyze_model.py"
echo "JOB $SLURM_JOB_ID is running on $SLURM_JOB_NODELIST "
echo "*********************************************************"
echo ""
srun python analyze_model.py -p 5 -s Development
