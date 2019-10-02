#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --qos long
#SBATCH --mem 20GB
#SBATCH --time 4:00:00
#SBATCH --job-name senitivityn
#SBATCH --output log/%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --partition gpu

XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="curta"

echo -e "running on ${node} with ${user}"

TASK_FILE=$1
echo Got task file: $TASK_FILE
cat $TASK_FILE

ARGS=$(sed -n ${SLURM_ARRAY_TASK_ID}p $TASK_FILE)
ARGS=($ARGS)

echo
echo Running task at line ${SLURM_ARRAY_TASK_ID}:
echo ${ARGS[@]}
echo



echo "loading cuda and anaconda..."
module load  CUDA
module load  Anaconda3
echo "activating conda env..."
source activate py36

echo "starting evaluation..."
echo ""
cd ..

echo python ./scripts/eval_sensitivity_n.py ${ARGS[@]}
python ./scripts/eval_sensitivity_n.py ${ARGS[@]}
