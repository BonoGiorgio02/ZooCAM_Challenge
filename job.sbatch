#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=20:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir=pwd
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 6a807f7df661f4aefddf615659ee3f293c637d8f"
cd $TMPDIR/code
git checkout 6a807f7df661f4aefddf615659ee3f293c637d8f


echo "Setting up the virtual environment"
/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1 $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main /usr/users/sdim/sdim_30/Deep Learning/deeplearning-team9_ancelle-hansen_bono_oneill/configs/tmp67ed71rj-config.yml train

# MODIFICATION : On stocke le code de sortie pour rapatrier les logs MÊME si ça a planté
TRAIN_EXIT_CODE=$?
 
# =========================================================
# NOUVEAU : RAPATRIEMENT DES LOGS ET MODÈLES
# =========================================================
echo "Retrieving logs from the compute node..."
mkdir -p $current_dir/logs
# On renvoie tout le dossier logs du GPU vers ton dossier d'origine
rsync -avz logs/ $current_dir/logs/
# =========================================================

if [[ $? != 0 ]]; then
    exit -1
fi
