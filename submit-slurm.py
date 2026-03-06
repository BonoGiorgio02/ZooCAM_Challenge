#!/usr/bin/python

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


DEFAULT_PARTITION = "gpu_prod_long"
DEFAULT_DURATION = "20:00:00"


def build_sbatch_header(
    job_name,
    partition,
    duration,
    output_path,
    error_path,
    array_size=None,
    constraint=None,
    exclusive=True,
):
    lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        f"#SBATCH --partition={partition}",
    ]
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    if exclusive:
        lines.append("#SBATCH --exclusive")
    lines.extend(
        [
            f"#SBATCH --time={duration}",
            f"#SBATCH --output={output_path}",
            f"#SBATCH --error={error_path}",
        ]
    )
    if array_size is not None:
        lines.append(f"#SBATCH --array=1-{array_size}")
    return "\n".join(lines) + "\n\n"


def makejob(commit_id, configpath, nruns, partition, duration, constraint, exclusive):
    header = build_sbatch_header(
        job_name="templatecode",
        partition=partition,
        duration=duration,
        output_path="logslurms/slurm-%A_%a.out",
        error_path="logslurms/slurm-%A_%a.err",
        array_size=nruns,
        constraint=constraint,
        exclusive=exclusive,
    )
    return f"""{header}current_dir=$(pwd)
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
TRAIN_EXIT_CODE=$?

echo "Retrieving logs from the compute node..."
mkdir -p "$current_dir/logs"
if [[ -d logs ]]; then
    rsync -avz logs/ "$current_dir/logs/"
    RSYNC_EXIT_CODE=$?
else
    echo "No logs/ directory found on compute node."
    RSYNC_EXIT_CODE=0
fi

if [[ $RSYNC_EXIT_CODE != 0 ]]; then
    exit $RSYNC_EXIT_CODE
fi

exit $TRAIN_EXIT_CODE
"""


def make_reserve_job(duration, partition, constraint, exclusive):
    header = build_sbatch_header(
        job_name="reserve_node",
        partition=partition,
        duration=duration,
        output_path="logslurms/slurm-%j.out",
        error_path="logslurms/slurm-%j.err",
        array_size=None,
        constraint=constraint,
        exclusive=exclusive,
    )
    return f"""{header}echo "Reserved node:" $(hostname)
echo "Keeping allocation for {duration}"
date
sleep infinity
"""


def submit_job(job_script):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sbatch", prefix="tmp-submit-", dir=".", delete=False
    ) as fp:
        fp.write(job_script)
        script_path = fp.name

    try:
        return subprocess.run(
            ["sbatch", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def submit_or_raise(job_script):
    result = submit_job(job_script)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"sbatch submission failed: {detail}")
    print(result.stdout.strip())


def parse_slurm_time_to_seconds(value):
    token = value.strip().lower()
    if token in {"infinite", "unlimited"}:
        return float("inf")

    day_count = 0
    if "-" in token:
        day_token, token = token.split("-", 1)
        day_count = int(day_token)

    parts = token.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = map(int, parts)
    elif len(parts) == 1:
        hours = 0
        minutes = int(parts[0])
        seconds = 0
    else:
        raise ValueError(f"Unsupported SLURM time format: {value}")
    return day_count * 86400 + hours * 3600 + minutes * 60 + seconds


def discover_partitions_for_duration(duration):
    target_seconds = parse_slurm_time_to_seconds(duration)
    result = subprocess.run(
        ["sinfo", "-h", "-o", "%P|%l"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return []

    partitions = []
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        raw_name, max_time = line.split("|", 1)
        part_name = raw_name.replace("*", "").strip()
        if not part_name:
            continue
        try:
            max_seconds = parse_slurm_time_to_seconds(max_time)
        except ValueError:
            continue
        if max_seconds >= target_seconds:
            partitions.append(part_name)

    # Prioritize partitions that clearly indicate long jobs.
    return sorted(set(partitions), key=lambda name: (0 if "long" in name else 1, name))


def reserve_with_fallback(duration, partition, constraint, exclusive):
    candidates = []
    seen = set()

    def add_candidate(part_name, part_constraint):
        key = (part_name, part_constraint or "")
        if key in seen:
            return
        seen.add(key)
        candidates.append((part_name, part_constraint))

    add_candidate(partition, constraint)
    if constraint:
        add_candidate(partition, None)

    for discovered in discover_partitions_for_duration(duration):
        if discovered == partition:
            continue
        add_candidate(discovered, constraint)
        if constraint:
            add_candidate(discovered, None)

    failures = []
    for part_name, part_constraint in candidates:
        job_script = make_reserve_job(
            duration=duration,
            partition=part_name,
            constraint=part_constraint,
            exclusive=exclusive,
        )
        result = submit_job(job_script)
        if result.returncode == 0:
            print(result.stdout.strip())
            if part_constraint:
                print(
                    f"Reservation accepted on partition={part_name} with constraint={part_constraint}"
                )
            else:
                print(f"Reservation accepted on partition={part_name} without constraint")
            return

        detail = (result.stderr or result.stdout or "").strip()
        failures.append(
            f"- partition={part_name}, constraint={part_constraint or 'none'} -> {detail}"
        )

    raise RuntimeError(
        "Unable to reserve a node with current settings.\n"
        + "\n".join(failures)
        + "\nTry: sinfo -o '%P %l %D %f %t'"
    )


def ensure_clean_git_state():
    # Ensure all modified files have been staged and committed.
    unstaged = subprocess.run(
        ["git", "diff", "--name-only", "--", ".", ":(exclude)job.sbatch"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.splitlines()
    staged = subprocess.run(
        ["git", "diff", "--name-only", "--cached", "--", ".", ":(exclude)job.sbatch"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.splitlines()

    modified = [x for x in (unstaged + staged) if x.strip()]
    if modified:
        print(
            "We found modifications either not staged or not committed "
            f"(excluding job.sbatch): {', '.join(sorted(set(modified)))}"
        )
        raise RuntimeError(
            "You must stage and commit every modification before submission "
        )


def get_commit_id():
    return subprocess.check_output(
        ["git", "log", "--pretty=format:%H", "-n", "1"], text=True
    ).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?")
    parser.add_argument("nruns", nargs="?", type=int, default=1)
    parser.add_argument("--reserve-only", action="store_true")
    parser.add_argument("--duration", "--time", dest="duration", default=DEFAULT_DURATION)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--no-exclusive", action="store_true")
    return parser.parse_args()


def normalize_reserve_only_args(args):
    # Backward compatibility: submit-slurm.py --reserve-only 20:00:00
    if args.reserve_only and args.config and args.duration == DEFAULT_DURATION:
        args.duration = args.config
        args.config = None
    if args.reserve_only and args.nruns != 1:
        raise ValueError("nruns is not used with --reserve-only")
    return args


def main():
    args = normalize_reserve_only_args(parse_args())
    exclusive = not args.no_exclusive
    os.makedirs("logslurms", exist_ok=True)

    if args.reserve_only:
        reserve_with_fallback(
            duration=args.duration,
            partition=args.partition,
            constraint=args.constraint if args.constraint else None,
            exclusive=exclusive,
        )
        return 0

    if not args.config:
        raise ValueError(
            "Missing config file path. Usage: submit-slurm.py <config.yaml> [nruns]"
        )
    if not os.path.isfile(args.config):
        raise FileNotFoundError(
            f"Config file not found: {args.config}. "
            "Pass an existing config path (example: configs/my_run.yaml)."
        )

    ensure_clean_git_state()
    commit_id = get_commit_id()
    print(f"I will be using the commit id {commit_id}")

    os.makedirs("configs", exist_ok=True)
    tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
    shutil.copy2(args.config, tmp_configfilepath)
    tmp_config_relpath = os.path.relpath(tmp_configfilepath, start=os.getcwd())

    submit_or_raise(
        makejob(
            commit_id=commit_id,
            configpath=tmp_config_relpath,
            nruns=args.nruns,
            partition=args.partition,
            duration=args.duration,
            constraint=args.constraint if args.constraint else None,
            exclusive=exclusive,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
