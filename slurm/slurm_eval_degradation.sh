#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --qos long
#SBATCH --qos long
#SBATCH --mem=46GB
#SBATCH --time 4-00:00:00
#SBATCH --job-name eval
#SBATCH --output log/eval-%J.log
#SBATCH --gres=gpu:1
#SBATCH --partition gpu

XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="curta"

echo -e "running on ${node} with ${user}"

echo "loading cuda and anaconda..."
module load  CUDA
module load  Anaconda3
echo "activating conda env..."
source activate py36

echo "starting evaluation..."
echo ""
cd ..
python eval_degradation.py "$1" "$2" "$3" "$4"
