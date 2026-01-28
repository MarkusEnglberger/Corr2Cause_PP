
from datasets import load_dataset, DatasetDict
from typing import Dict, Optional
from functools import partial
import os
import shutil
from transformers import AutoTokenizer
import json
from datasets import Dataset, DatasetDict

# Initialize tokenizer for chat template formatting
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", trust_remote_code=True)



def format_c2cp_for_grpo(example: Dict, dataset_name: str = "c2cpSplit1") -> Dict:
    """Format C2CP example for GRPO training with chat template.

    Task: Given statistical relationships and a hypothesis:
    1. Predict the direct causal edges
    2. Answer whether the hypothesis is implied (yes/no)

    Output format:
    - For edges: "X has a direct causal effect on Y."
    - For hypothesis: "Therefore: Yes" or "Therefore: No"
    """
    instruction = (
        "Given a description of statistical relationships between variables and a hypothesis, determine if the hypothesis is implied. "
        'Conclude with your answer to the hypothesis as either "Therefore: Yes" or "Therefore: No".\n\n'
    )
    prompt = f"{instruction}Premise:\n{example['premise']}\n\nHypothesis: {example['hypothesis']}\n\n"

    # Format using chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Convert forced_edges to list of tuples
    edges = [tuple(edge) for edge in example['forced_edges']]

    return {
        "query": formatted_query,
        "forced_edges": edges,
        "label": example['label'],  # Binary yes/no for hypothesis
        "relation_type": example.get('relation_type', None),
        "num_edges": len(edges),
        "dataset": dataset_name
    }


def format_c2cp_for_flash(example: Dict, dataset_name: str = "c2cpSplit1") -> Dict:
    """Format C2CP example for GPT/external model evaluation (plain text)."""
    instruction = (
        "Given a description of statistical relationships between variables and a hypothesis, your job is to find out if the hypothesis is implied. "
        "To that end, first identify all direct causal effects.\n\n"
        "Show your reasoning step by step. You should roughly follow the PC-algorithm and explain the rules you apply (e.g. when finding V-structures or when applying Meek rules).\n\n"
        "For each direct causal relationship, state it in the format: "
        '"[Variable1] has a direct causal effect on [Variable2]."\n'
        "If there are no direct causal effects, state: \"There are no direct causal effects.\"\n\n"
        'Finally, conclude with your answer to the hypothesis as either "Therefore: Yes" or "Therefore: No".\n\n' )
    prompt = f"{instruction}Premise:\n{example['premise']}\n\nHypothesis: {example['hypothesis']}\n\n"

    edges = [tuple(edge) for edge in example['forced_edges']]

    return {
        "query": prompt,
        "forced_edges": edges,
        "label": example['label'],
        "relation_type": example.get('relation_type', None),
        "num_edges": len(edges),
        "dataset": dataset_name
    }

def format_c2cp_for_else(example: Dict, dataset_name: str = "c2cpSplit1") -> Dict:
    """Format C2CP example for GPT/external model evaluation (plain text)."""
    instruction = (
        "Given a description of statistical relationships between variables and a hypothesis, determine if the hypothesis is implied. "
        'Conclude with your answer to the hypothesis as either "Therefore: Yes" or "Therefore: No".\n\n'  )
    prompt = f"{instruction}Premise:\n{example['premise']}\n\nHypothesis: {example['hypothesis']}\n\n"

    edges = [tuple(edge) for edge in example['forced_edges']]

    return {
        "query": prompt,
        "forced_edges": edges,
        "label": example['label'],
        "relation_type": example.get('relation_type', None),
        "num_edges": len(edges),
        "dataset": dataset_name
    }

def format_c2cp_for_release(example: Dict, dataset_name: str = "c2cpSplit1") -> Dict:
    """Format C2CP example for public release (no instruction prefix)."""
    prompt = f"Premise:\n{example['premise']}\n\nHypothesis: {example['hypothesis']}"

    edges = [tuple(edge) for edge in example['forced_edges']]

    return {
        "query": prompt,
        "forced_edges": edges,
        "label": example['label'],
        "relation_type": example.get('relation_type', None),
        "num_edges": len(edges),
        "dataset": dataset_name
    }


def process_c2cp(cache_dir: str, output_base_dir: str, source_file: str, dataset_name: str, train_perc = 0.79):
    """Load and process C2CP dataset from JSON file."""

    print("\n" + "="*60)
    print(f"Processing {dataset_name} (from {source_file}.json)")
    print("="*60)

    json_path = os.path.join(cache_dir, source_file+".json")

    # Load from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {json_path}")

    # Create train/validation/test splits (80/10/10)
    total_size = len(data)
    train_size = int(train_perc * total_size)
    val_size = int(0.01 * total_size)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Create DatasetDict
    raw_dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

    # Format for GRPO training
    formatted_grpo = raw_dataset.map(
        partial(format_c2cp_for_grpo, dataset_name=dataset_name),
        remove_columns=['premise', 'hypothesis'],
        desc="Formatting C2CP for GRPO"
    )

    # Format for GPT evaluation
    formatted_flash = raw_dataset.map(
        partial(format_c2cp_for_flash, dataset_name=dataset_name),
        remove_columns=['premise', 'hypothesis'],
        desc="Formatting C2CP for Gemini Flash"
    )

    formatted_else = raw_dataset.map(
        partial(format_c2cp_for_else, dataset_name=dataset_name),
        remove_columns=['premise', 'hypothesis'],
        desc="Formatting C2CP for GPT 5.2 and Gemini 3"
    )

    # Format for public release (no instruction prefix)
    formatted_release = raw_dataset.map(
        partial(format_c2cp_for_release, dataset_name=dataset_name),
        remove_columns=['premise', 'hypothesis'],
        desc="Formatting C2CP for public release"
    )

    # Save datasets - use absolute paths to avoid Windows path issues
    save_path_grpo = os.path.abspath(os.path.join(output_base_dir, dataset_name, "grpo"))
    save_path_flash = os.path.abspath(os.path.join(output_base_dir, dataset_name, "flash"))
    save_path_else = os.path.abspath(os.path.join(output_base_dir, dataset_name, "else"))

    # Remove existing directories to avoid conflicts
    for path in [save_path_grpo, save_path_flash, save_path_else]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    formatted_grpo.save_to_disk(save_path_grpo)
    formatted_flash.save_to_disk(save_path_flash)
    formatted_else.save_to_disk(save_path_else)

    # Save release format to Corr2Cause++ folder at project root (one record per line, no instruction prefix)
    corr2cause_base = os.path.abspath(os.path.join(output_base_dir, "..", "..", "Corr2Cause++"))

    def save_json_records(records, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("[\n")
            for i, record in enumerate(records):
                f.write(json.dumps(record, ensure_ascii=False))
                if i < len(records) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]")

    # Map dataset names to output folders
    if dataset_name == "c2cpSplit1":
        # Save training data and test_split1
        save_json_records(list(formatted_release['train']), os.path.join(corr2cause_base, "training", "train.json"))
        save_json_records(list(formatted_release['test']), os.path.join(corr2cause_base, "test_split1", "test.json"))
    elif dataset_name == "c2cpSplit2":
        # Only save test_split2
        save_json_records(list(formatted_release['test']), os.path.join(corr2cause_base, "test_split2", "test.json"))
    elif dataset_name == "c2cpSplit3":
        # Only save test_split3
        save_json_records(list(formatted_release['test']), os.path.join(corr2cause_base, "test_split3", "test.json"))

    print(f"\nC2CP Dataset Statistics:")
    print(f"  Train: {len(formatted_grpo['train'])}")
    print(f"  Validation: {len(formatted_grpo['validation'])}")
    print(f"  Test: {len(formatted_grpo['test'])}")

    # Print label distribution
    label_counts = {0: 0, 1: 0}
    for split in ['train', 'validation', 'test']:
        for label in formatted_grpo[split]['label']:
            label_counts[label] = label_counts.get(label, 0) + 1
    print(f"\nLabel Distribution:")
    print(f"  No (0): {label_counts[0]}")
    print(f"  Yes (1): {label_counts[1]}")

    # Print edge distribution
    edge_counts = {}
    for split in ['train', 'validation', 'test']:
        for n in formatted_grpo[split]['num_edges']:
            edge_counts[n] = edge_counts.get(n, 0) + 1
    print(f"\nEdge Count Distribution:")
    for n_edges, count in sorted(edge_counts.items()):
        print(f"  {n_edges} edges: {count} samples")

    # Print relation type distribution
    if 'relation_type' in formatted_grpo['train'].column_names:
        rel_counts = {}
        for split in ['train', 'validation', 'test']:
            for rt in formatted_grpo[split]['relation_type']:
                if rt is not None:
                    rel_counts[rt] = rel_counts.get(rt, 0) + 1
        print(f"\nRelation Type Distribution:")
        for rt, count in sorted(rel_counts.items()):
            print(f"  {rt}: {count}")

    print(f"\nSaved to:")
    print(f"  GRPO: {save_path_grpo}")
    print(f"  Flash:  {save_path_flash}")
    print(f"  Else:  {save_path_else}")

    return formatted_grpo, formatted_flash, formatted_else


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    output_dir = "./data/processed"
    cache_dir = "./data/cache"

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Causal Reasoning Dataset Preprocessing")
    print("="*60)
    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")

    process_c2cp(cache_dir, output_dir, source_file="c2cp", dataset_name="c2cpSplit1", train_perc=0.79)
    process_c2cp(cache_dir, output_dir, source_file="c2cpR", dataset_name="c2cpSplit2", train_perc=0.79)
    process_c2cp(cache_dir, output_dir, source_file="c2cp7", dataset_name="c2cpSplit3", train_perc=0.01)

    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()