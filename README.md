# Corr2Cause++

Code and data for training and evaluating large language models on causal reasoning tasks. Given statistical relationships (correlations) and a hypothesis, models must:
1. Predict direct causal edges between variables
2. Answer yes/no to the causal hypothesis


## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Creating Dataset
```bash
python scripts\create_c2cp.py -n 3-6 -m 4000,4000,4000,280
python scripts\create_c2cp.py -n 3-6 -m 4000,4000,4000,280 -r
python scripts\create_c2cp.py -n 7 -m 180 -r
python scrips\data_preprocessing.py
```
## Training Pipeline
For HPC clusters with SLURM:
```bash
sbatch Training.slurm
sbatch Evaluation.slurm
```


### 1. Prepare SFT Data
```bash
python scripts/prepare_sft_data.py
```

### 2. Two-phase training pipeline (SFT+GRPO)


### 3. GRPO Training
Reinforcement learning with group relative policy optimization:
```
torchrun --nproc_per_node=4 scripts/train_grpo.py configs/grpo_config_7b.yaml
```

## Evaluation

### Local Models
See the slurm file Evaluation.slurm

### API-based Models
Set environment variables for the APIs you want to use:
```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
export MOONSHOT_API_KEY="your-key"

Then run 
```bash
python evaluate_models_via_API.py --split "c2cpSplit1" --model_name "MODEL_NAME"
```