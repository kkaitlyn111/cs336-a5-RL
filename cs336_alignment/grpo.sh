#!/bin/bash
#SBATCH --job-name=grpo_run
#SBATCH --output=grpo_run_%j.out
#SBATCH --error=grpo_run_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load any necessary modules (uncomment and edit as needed)
# module load cuda/12.1

# Activate your virtual environment if needed
source .venv/bin/activate

# Run your command
uv run cs336_alignment/grpo.py