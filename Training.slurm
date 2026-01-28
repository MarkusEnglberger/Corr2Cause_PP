#!/bin/bash
#SBATCH --job-name=train_c2cp
#SBATCH --output=logs/train_c2cp_%j.out
#SBATCH --error=logs/train_c2cp_%j.err
#SBATCH --time=60:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load required modules
module purge
module load 2025
module load CUDA/12.8.0
module load Python/3.13.1-GCCcore-14.2.0

# Create and activate virtual environment
if [ ! -d "venv" ] || [ ! -f "venv/pyvenv.cfg" ]; then
    echo "Creating new virtual environment with Python 3.13..."
    rm -rf venv
    python -m venv venv
    source venv/bin/activate
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate virtual environment"
        exit 1
    fi
fi

# Create logs directory


# Set environment variables
export TRANSFORMERS_CACHE=./data/cache
export HF_HOME=./data/cache

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
nvidia-smi

# Multi-node configuration
# Get the master node address (first node in the allocation)
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# Use unique port per job to avoid "address already in use" errors
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 500 ))

# Number of GPUs per node
GPUS_PER_NODE=4

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"

#pip uninstall flash-attn

#collect reasoning traces
python evaluate_models_via_API.py --for-sft --beginning 0 --max_samples 500 --split "c2cpSplit1" --model_name "gemini-3-flash-preview"
python scripts/prepare_sft_data.py

#train the 32b model
python scripts/train_sft.py configs/sft_config_32b.yaml
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=$MASTER_PORT \
   scripts/train_grpo.py  configs/grpo_config_32b.yaml --run_name "grpo-deepseek-torchrun-${SLURM_JOB_ID}"

#train the 7b model
python scripts/train_sft.py configs/sft_config_7b.yaml
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=$MASTER_PORT \
   scripts/train_grpo_simple.py  configs/grpo_config_7b.yaml --run_name "grpo-deepseek-torchrun-${SLURM_JOB_ID}"

# Launch distributed training across all nodes using srun

echo "Job finished at: $(date)"
