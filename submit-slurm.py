
#!/usr/bin/python

import os
import sys
import subprocess

WANDB_API_KEY = "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"
WANDB_MODE = "online"

def _get_run_name_from_config(configpath: str) -> str:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required on the login node: pip install pyyaml") from e

    with open(configpath, "r") as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model", {}).get("class", "model")
    configname = os.path.splitext(os.path.basename(configpath))[0]
    return f"{model}_{configname}"


def makejob(commit_id: str, configpath: str, run_root: str, nruns: int, install_torch_pascal: bool, wandb_mode: str,
    wandb_api_key: str,) -> str:
    run_name = _get_run_name_from_config(configpath)

    # IMPORTANT:
    # We use .format() to inject a few values (commit_id, configpath, etc.).
    # Therefore, every literal { } inside the generated bash/python must be escaped as {{ }}.
    torch_block = r"""
echo "Installing PyTorch compatible with Pascal (GTX 1080Ti sm_61)"
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2 torchvision==0.16.2
""" if install_torch_pascal else r"""
echo "Skipping torch override (using project dependencies)"
"""

    return """#!/bin/bash
#SBATCH --job-name=templatecode
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --array=1-{nruns}
#SBATCH --output={run_root}/slurm_logs/{run_name}_%A_%a.out
#SBATCH --error={run_root}/slurm_logs/{run_name}_%A_%a.err

set -u
set -o pipefail

echo "Session ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "Running on $(hostname)"
date

current_dir=$(pwd)

RUN_NAME="{run_name}"
RUN_DIR="{run_root}/${{RUN_NAME}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"

echo "Creating run directory: $RUN_DIR"
mkdir -p "$RUN_DIR"
mkdir -p "{run_root}/slurm_logs"

echo "Copying source code to scratch (TMPDIR fallback)"
TMPBASE="${{TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}}"
mkdir -p "$TMPBASE/code"
rsync -r --exclude logs --exclude logslurms --exclude runs . "$TMPBASE/code"

cd "$TMPBASE/code"
echo "Checkout commit {commit_id}"
git checkout {commit_id}

echo "Setting up DCE virtual environment"

/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1 $TMPDIR/venv
source $TMPDIR/venv/bin/activate

echo "Python version:"
python --version

echo "Torch version:"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# install your package in editable mode
python -m pip install -e .

# alcune dipendenze utili
python -m pip install wandb pyyaml

# ====
# WANDB SETUP
# ====

export WANDB_DIR="$RUN_DIR/wandb"
mkdir -p "$WANDB_DIR"

export WANDB_MODE="{wandb_mode}"

export WANDB_API_KEY="{wandb_api_key}"

echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"

# Method 2: Python explicit login
python3 - <<'PY'
import os

mode = os.environ.get("WANDB_MODE", "").strip().lower()
key = os.environ.get("WANDB_API_KEY", "").strip()

print("Detected WANDB_MODE =", mode)
print("WANDB_API_KEY present =", bool(key))

if mode == "online":
    if not key:
        raise RuntimeError(
            "WANDB_MODE=online ma WANDB_API_KEY e' vuota. "
            "Incolla la tua API key nella variabile WANDB_API_KEY in submit-slurm.py"
        )
    import wandb
    wandb.login(key=key, relogin=True)
    print("wandb login OK")
elif mode == "offline":
    print("wandb offline mode enabled")
elif mode == "disabled":
    os.environ["WANDB_DISABLED"] = "true"
    print("wandb disabled")
else:
    raise RuntimeError("WANDB_MODE deve essere uno tra: online, offline, disabled")
PY


PATCHED_CONFIG="$RUN_DIR/config.yaml"

echo "Patching config (logdir + checkpoint -> RUN_DIR)"
SRC_CONFIG="{configpath}" DST_CONFIG="$PATCHED_CONFIG" RUN_DIR="$RUN_DIR" python3 - <<'PY'
import os, yaml

src = os.environ["SRC_CONFIG"]
dst = os.environ["DST_CONFIG"]
run_dir = os.environ["RUN_DIR"]

best_path = os.path.join(run_dir, "best_model.pt")
last_path = os.path.join(run_dir, "last_model.pt")

with open(src, "r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("logging", {{}})
cfg["logging"]["logdir"] = run_dir
cfg["checkpoint"] = best_path

cfg.setdefault("checkpointing", {{}})
cfg["checkpointing"]["dir"] = run_dir
cfg["checkpointing"]["best_path"] = best_path
cfg["checkpointing"]["last_path"] = last_path
cfg["checkpointing"]["save_best"] = True
cfg["checkpointing"]["save_last"] = True

# Varianti comuni che alcuni trainer usano
cfg["best_checkpoint"] = best_path
cfg["last_checkpoint"] = last_path

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Config patched ->", dst)
print("logdir     ->", cfg["logging"]["logdir"])
print("checkpoint ->", cfg["checkpoint"])
print("best_checkpoint->", cfg["best_checkpoint"])
print("last_checkpoint->", cfg["last_checkpoint"])
print("checkpointing  ->", cfg["checkpointing"])
PY

echo "Starting training"

set +e
python -m torchtmpl.main "$PATCHED_CONFIG" train
exit_code=$?
set -e

echo "Training finished with code $exit_code"
echo "Final content of: $RUN_DIR"
ls -lah "$RUN_DIR" || true
exit $exit_code
""".format(
        nruns=nruns,
        run_root=run_root,
        run_name=run_name,
        commit_id=commit_id,
        configpath=configpath,
        torch_block=torch_block.strip(),
        wandb_mode=wandb_mode,
        wandb_api_key=wandb_api_key,
    )


def submit_job(job: str, run_root: str) -> None:
    os.makedirs(os.path.join(run_root, "slurm_logs"), exist_ok=True)
    with open("job.sbatch", "w") as f:
        f.write(job)
    os.system("sbatch job.sbatch")


def ensure_git_clean() -> None:
    dirty = int(
        subprocess.run(
            "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout.decode()
    )
    if dirty > 0:
        raise RuntimeError("Commit everything before submitting (git status must be clean)")


def get_commit_id() -> str:
    return subprocess.check_output(
        "git log --pretty=format:'%H' -n 1", shell=True
    ).decode().strip()


if __name__ == "__main__":
    # Usage:
    #   python submit-slurm.py config.yaml
    #   python submit-slurm.py config.yaml 1
    #   python submit-slurm.py config.yaml 3
    #   python submit-slurm.py config.yaml 1 --no-torch-fix
    if len(sys.argv) not in [2, 3, 4]:
        print(f"Usage: {sys.argv[0]} config.yaml <nruns|1> [--no-torch-fix]")
        sys.exit(-1)

    configpath = sys.argv[1]
    if not os.path.exists(configpath):
        raise FileNotFoundError(f"Config file {configpath} not found")

    nruns = 1 if len(sys.argv) == 2 else int(sys.argv[2])

    install_torch_pascal = True
    if len(sys.argv) == 4:
        if sys.argv[3] == "--no-torch-fix":
            install_torch_pascal = False
        else:
            raise ValueError("Unknown option. Use --no-torch-fix or omit the option.")

    ensure_git_clean()
    commit_id = get_commit_id()
    print("Using commit:", commit_id)

    run_root = os.path.join(os.getcwd(), "runs")
    os.makedirs(run_root, exist_ok=True)

    submit_job(makejob(commit_id, configpath, run_root, nruns, install_torch_pascal, wandb_mode=WANDB_MODE, wandb_api_key=WANDB_API_KEY), run_root)

