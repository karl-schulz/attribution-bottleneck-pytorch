#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --qos long
#SBATCH --qos long
#SBATCH --mem=46GB
#SBATCH --time 3-12:00:00
#SBATCH --job-name bbox
#SBATCH --output log/%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --partition gpu

XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="curta"

TASK_FILE=$1
echo Got task file: $TASK_FILE
cat $TASK_FILE

echo Got task at line ${SLURM_ARRAY_TASK_ID}:

ARGS=$(sed -n ${SLURM_ARRAY_TASK_ID}p $TASK_FILE)
ARGS=($ARGS)

echo ${ARGS[@]}

echo -e "running on ${node} with ${user}"

echo "loading cuda and anaconda..."
module load  CUDA
module load  Anaconda3
echo "activating conda env..."
source activate py36

echo "starting evaluation..."
echo ""
cd ..


echo python ./scripts/eval_bounding_boxes.py ${ARGS[@]}
python ./scripts/eval_bounding_boxes.py ${ARGS[@]}
