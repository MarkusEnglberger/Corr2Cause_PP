# Corr2Cause++

Code and data for training and evaluating large language models on causal reasoning tasks.

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

## Evaluation

### Local Models
See the slurm file Evaluation.slurm

### API-based Models
Set environment variables for the APIs you want to use and then run
 
```bash
python evaluate_models_via_API.py --split "c2cpSplit1" --model_name "MODEL_NAME"
```