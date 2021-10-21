#! /bin/bash
experiment_name=variable-discharge

mkdir ../results/$experiment_name
mkdir ../results/${experiment_name}/models
mkdir ../results/${experiment_name}/predictions

sbatch --time=1:00:00 -J soap-ensemble --export=script="../experiments/next-cycle-capacity.py",kwargs="" run-battery-cpu
