#! /bin/bash
experiment_name=variable-discharge

mkdir ../results/$experiment_name
mkdir ../results/${experiment_name}/models
mkdir ../results/${experiment_name}/predictions

sbatch --time=0:10:00 -J n-steps --export=script="../experiments/n-step-lookahead.py",kwargs="" run-battery-cpu
