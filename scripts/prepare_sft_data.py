"""
Prepare SFT (Supervised Fine-Tuning) dataset from GPT/Gemini evaluation results.
This script filters correct reasoning traces and prepares them for fine-tuning.

Usage:
    python scripts/prepare_sft_data.py
"""
import json
import os
import argparse
import glob
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

C2CP_GPT_INSTRUCTION = (
     "Given a description of statistical relationships between variables and a hypothesis, your job is to find out if the hypothesis is implied. "
        "To that end, first identify all direct causal effects.\n\n"
        "Show your reasoning step by step. You should roughly follow the PC-algorithm and explain the rules you apply (e.g. when finding V-structures or when applying Meek rules).\n\n"
        "For each direct causal relationship, state it in the format: "
        '"[Variable1] has a direct causal effect on [Variable2]."\n'
        "If there are no direct causal effects, state: \"There are no direct causal effects.\"\n\n"
        'Finally, conclude with your answer to the hypothesis as either "Therefore: Yes" or "Therefore: No".\n\n'
)

C2CP_GRPO_INSTRUCTION = (
    "Given a description of statistical relationships between variables and a hypothesis, determine if the hypothesis is implied. "
    'Conclude with your answer to the hypothesis as either "Therefore: Yes" or "Therefore: No".\n\n'
)


# Initialize tokenizer for chat template formatting
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", trust_remote_code=True)

def load_gpt_predictions(predictions_file: str):
    """Load GPT predictions from JSON file."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def is_valid_response(generated_text: str) -> bool:
    """Check if a response is valid (not an API error or timeout)."""
    if not generated_text or len(generated_text) < 50:
        return False
    if generated_text.startswith('ERROR:'):
        return False
    if 'Deadline Exceeded' in generated_text:
        return False
    if 'Timeout' in generated_text and 'exceeded' in generated_text.lower():
        return False
    return True


def filter_correct_traces(predictions_data):
    predictions = predictions_data['predictions']
    correct_traces = [p for p in predictions if p['is_correct']]
    # Filter out API errors and invalid responses (safety check for older evaluation files)
    valid_traces = [p for p in correct_traces if is_valid_response(p.get('generated_text', ''))]
    filtered_count = len(correct_traces) - len(valid_traces)
    print(f"Total samples: {len(predictions)}")
    print(f"Correct samples: {len(correct_traces)}")
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} invalid responses (API errors/timeouts)")
    print(f"Valid samples: {len(valid_traces)}")
    return valid_traces

def format_for_sft(correct_traces):
    """Format correct traces for SFT training, preserving dataset-specific metadata."""
    formatted_data = []
    for trace in correct_traces:
        # Apply chat template to the input query
        # trace['input'] contains the raw user prompt
        #messages = [{"role": "user", "content": trace['input']}]
        input_text = trace['input']

        # For C2CP: replace GPT-style instruction with GRPO-style instruction
        if C2CP_GPT_INSTRUCTION in input_text:
            input_text = input_text.replace(C2CP_GPT_INSTRUCTION, C2CP_GRPO_INSTRUCTION)

        messages = [{"role": "user", "content": input_text}]
        formatted_query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # C2CP format: combined edge prediction + hypothesis verification
        ground_truth_edges = trace.get('ground_truth_edges', [])
        predicted_edges = trace.get('predicted_edges', [])

        formatted_item = {
            'query': formatted_query,  # Now properly formatted with chat template
            'response': trace['generated_text'],
            'ground_truth_edges': json.dumps(ground_truth_edges),
            'predicted_edges': json.dumps(predicted_edges),
            'num_edges': len(ground_truth_edges),
            'label': trace.get('ground_truth_label', trace.get('ground_truth')),
            'predicted_answer': trace.get('predicted_answer', trace.get('predicted')),
        }

        # Store component correctness if available
        if 'edges_correct' in trace:
            formatted_item['edges_correct'] = trace['edges_correct']
        if 'answer_correct' in trace:
            formatted_item['answer_correct'] = trace['answer_correct']

        # Preserve dataset-specific metadata if available
        metadata_keys = ['num_variables', 'template', 'rung', 'query_type', 'graph_id', 'story_id', 'dataset', 'relation_type']
        for key in metadata_keys:
            if key in trace and trace[key] is not None:
                formatted_item[key] = trace[key]

        formatted_data.append(formatted_item)
    return formatted_data

def create_train_val_split(data, val_ratio=0.01):
    """Split data into train and validation sets."""
    import random
    random.seed(42)

    indices = list(range(len(data)))
    random.shuffle(indices)

    val_size = int(len(data) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    return train_data, val_data

def process_prediction_files(prediction_files, model_name, output_dir):
    """Process prediction files from a specific model and save as SFT dataset.

    Args:
        prediction_files: List of prediction JSON file paths
        model_name: Name of the model (for logging)
        output_dir: Directory to save the SFT dataset
    """
    if not prediction_files:
        print(f"No {model_name} prediction files found.")
        return

    print(f"\n{'='*60}")
    print(f"Processing {model_name} predictions...")
    print(f"{'='*60}")

    all_correct_traces = []
    for predictions_file in prediction_files:
        print(f"Loading {os.path.basename(predictions_file)}...")
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        correct_traces = filter_correct_traces(predictions_data)
        all_correct_traces.extend(correct_traces)

    print(f"\nTotal correct traces from {model_name} files: {len(all_correct_traces)}")

    if len(all_correct_traces) == 0:
        print(f"No correct traces found. Skipping {model_name} dataset creation.")
        return

    # Format and save data
    formatted_data = format_for_sft(all_correct_traces)
    dataset = Dataset.from_list(formatted_data)
    dataset_dict = DatasetDict({'train': dataset})

    print(f"Saving {model_name} dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(f"{model_name} dataset saved successfully!")


def process_dataset(evaluation_dir, output_dir):
    """Process evaluation results for C2CP dataset.

    Args:
        evaluation_dir: Base directory for evaluation results
        output_dir: Base output directory for SFT datasets
    """
    # Evaluation results are in evaluation_results/c2cpSplit1/gemini-3-flash-preview/
    dataset_eval_dir = os.path.join(evaluation_dir, "c2cpSplit1", "gemini-3-flash-preview")

    # Find prediction files in the gemini-3-flash-preview subfolder
    gemini_files = glob.glob(os.path.join(dataset_eval_dir, '*predictions_gemini*.json'))

    print(f"\n{'='*60}")
    print(f"Processing c2cpSplit1 dataset")
    print(f"{'='*60}")
    print(f"Evaluation dir: {dataset_eval_dir}")
    print(f"Found {len(gemini_files)} Gemini prediction files")

    if not gemini_files:
        print(f"No prediction files found for c2cpSplit1. Skipping.")
        return

    # Output to dataset-specific SFT folder
    gemini_output_dir = os.path.join(output_dir, 'c2cpSplit1', 'sft_gemini')
    process_prediction_files(gemini_files, "Gemini", gemini_output_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from C2CP evaluation results")
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="./evaluation_results",
        help="Base directory containing evaluation result subfolders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Base output directory for SFT datasets"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("SFT Data Preparation (C2CP)")
    print("="*60)
    print(f"Evaluation results dir: {args.evaluation_dir}")
    print(f"Output directory: {args.output_dir}")

    process_dataset(args.evaluation_dir, args.output_dir)

    print(f"\n{'='*60}")
    print("SFT data preparation complete!")
    print(f"{'='*60}")

    # Print output locations
    print("\nSFT dataset saved to:")
    print(f"  C2CP Gemini: {os.path.join(args.output_dir, 'c2cpSplit1', 'sft_gemini')}")


if __name__ == "__main__":
    main()
