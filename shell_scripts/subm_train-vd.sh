#! /bin/bash
experiment_name=variable-discharge

mkdir ../results/$experiment_name
mkdir ../results/${experiment_name}/models
mkdir ../results/${experiment_name}/predictions

sbatch --time=0:10:00 -J vd-train --export=script="../experiments/variable-discharge-train.py",kwargs="" run-battery-cpu
