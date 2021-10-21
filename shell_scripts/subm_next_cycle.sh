#! /bin/bash
experiment_name=variable-discharge

mkdir ../results/$experiment_name
mkdir ../results/${experiment_name}/models
mkdir ../results/${experiment_name}/predictions

sbatch --time=0:10:00 -J battery-cpu --export=script="../experiments/next-cycle-capacity.py",kwargs="" run-battery-cpu
