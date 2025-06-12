#!/bin/bash
#SBATCH --job-name=CS336_A5_grpo            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=128G                 # Memory per CPU core
#SBATCH --gres=gpu:2                    # Number of GPUs per node
#SBATCH --time 4:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

uv run cs336_alignment/grpo.py $@