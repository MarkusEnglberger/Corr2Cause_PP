#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=logs/Eval_%j.out
#SBATCH --error=logs/Eval_%j.err
#SBATCH --time=9:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14

# Load required modules
module purge
module load 2025
module load CUDA/12.8.0
module load Python/3.13.1-GCCcore-14.2.0

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

# Create logs directory


# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache

# Use unique port per job to avoid "address already in use" errors
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 500 ))
# Set your WandB API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"



# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of GPUs: $SLURM_GPUS"
echo "Working directory: $(pwd)"


torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=$MASTER_PORT \
    scripts/evaluate_models_locally.py configs/evaluation_config.yaml --run_name "evaluation${SLURM_JOB_ID}"


echo "Job finished at: $(date)"