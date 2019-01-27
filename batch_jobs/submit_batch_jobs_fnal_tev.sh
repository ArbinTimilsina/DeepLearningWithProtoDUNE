#!/bin/sh

## Set number of nodes for the job
#SBATCH --nodes 1

## Set number of tasks for the job
#SBATCH --ntasks 1

## Request a specific partition
#SBATCH --partition gpu

## Specify to only use GPU nodes; no. per node
#SBATCH --gres=gpu:1

## Specify gpu4: Eight NVIDIA Tesla Pascal P100 GPUs with 17GB of memory each
#SBATCH --nodelist=gpu4

## Specify requested time
## day-hr
#SBATCH -t 1-00:00:00

## Specify stdout, stderr log; default is slurm-jobid.out
#SBATCH -o output.log
#SBATCH -e error.log

## Specify what to notify: BEGIN, END, FAIL...., ALL
##SBATCH --mail-type=FAIL
##SBATCH --mail-user arbint@bnl.gov

## Go to main directory
cd ../
echo ""
echo "*********************************************************"
echo "This was run on:"
date
echo "*********************************************************"
echo ""
echo "*********************************************************"
echo "Running python train_model.py"
echo "JOB $SLURM_JOB_ID is running on $SLURM_JOB_NODELIST "
echo "*********************************************************"
echo ""
singularity exec --bind /data/arbint --nv /home/arbint/DeepLearningWithProtoDUNE.img python train_model.py -o Training -e Default

echo "*********************************************************"
echo "Running python analyze_model.py"
echo "JOB $SLURM_JOB_ID is running on $SLURM_JOB_NODELIST "
echo "*********************************************************"
echo ""
singularity exec --bind /data/arbint --nv /home/arbint/DeepLearningWithProtoDUNE.img python analyze_model.py -p 5 -s Development

echo "*********************************************************"
echo "All done. Exiting"
echo "*********************************************************"
echo ""
exit
