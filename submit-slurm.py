#!/usr/bin/python

import os
import sys
import subprocess
import tempfile
import shutil


def makejob(commit_id, configpath, nruns):
    return f"""#!/bin/bash

#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --constraint=tx
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=$(pwd)
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir -p "$TMPDIR/code"
rsync -r --exclude logs --exclude logslurms . "$TMPDIR/code"

echo "Checking out the correct version of the code commit_id {commit_id}"
cd "$TMPDIR/code"
git checkout {commit_id}


echo "Setting up the virtual environment"
/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1 $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# Install the library
python -m pip install .

echo "Training"
python -m torchtmpl.main "{configpath}" train

# MODIFICATION : On stocke le code de sortie pour rapatrier les logs MÊME si ça a planté
TRAIN_EXIT_CODE=$?
 
# =========================================================
# NOUVEAU : RAPATRIEMENT DES LOGS ET MODÈLES
# =========================================================
echo "Retrieving logs from the compute node..."
mkdir -p "$current_dir/logs"
# On renvoie tout le dossier logs du GPU vers ton dossier d'origine
if [[ -d logs ]]; then
    rsync -avz logs/ "$current_dir/logs/"
    RSYNC_EXIT_CODE=$?
else
    echo "No logs/ directory found on compute node."
    RSYNC_EXIT_CODE=0
fi
# =========================================================

if [[ $RSYNC_EXIT_CODE != 0 ]]; then
    exit $RSYNC_EXIT_CODE
fi

exit $TRAIN_EXIT_CODE
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode().strip()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [2, 3]:
    print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
    sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) == 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
shutil.copy2(configpath, tmp_configfilepath)
tmp_config_relpath = os.path.relpath(tmp_configfilepath, start=os.getcwd())

# Launch the batch jobs
submit_job(makejob(commit_id, tmp_config_relpath, nruns))
