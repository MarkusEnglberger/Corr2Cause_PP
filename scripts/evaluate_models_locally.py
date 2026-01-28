from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Dict
import torch
import torch.distributed as dist
import yaml
import os
import sys
import json
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
from datetime import datetime


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(model_path: str, use_4bit: bool = False, local_rank: int = 0):
    """Load model and tokenizer.

    Args:
        model_path: Path to model or PEFT adapter checkpoint
        use_4bit: Whether to use 4-bit quantization
        local_rank: Local GPU rank for this process
    """
    print(f"[Rank {local_rank}] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    # Use specific GPU for this rank instead of auto device_map
    device_map = {"": local_rank}
    model_kwargs = {"trust_remote_code": True, "device_map": device_map, "attn_implementation": "sdpa"}
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_peft_model:
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")

        print(f"Detected PEFT model. Loading base model + adapter...")
        print(f"Base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    return model, tokenizer

def extract_answer(text: str) -> Optional[bool]:
    text_lower = text.lower()
    if "therefore: yes" in text_lower or "answer: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower or "answer: no" in text_lower:
        return False
    return None


def evaluate_dataset(model,tokenizer,dataset,max_new_tokens,batch_size,max_samples,show_samples,beginning=0,rank=0,world_size=1) -> Dict:
    # First select the requested range
    dataset = dataset.select(range(beginning, min(beginning+max_samples, len(dataset))))

    # Shard dataset across ranks - each rank processes a different subset
    total_samples = len(dataset)
    indices = list(range(rank, total_samples, world_size))  # e.g., rank 0: [0,2,4,...], rank 1: [1,3,5,...]
    dataset = dataset.select(indices)

    total = len(dataset)
    correct = 0
    predictions = []
    no_answer_count = 0
    samples_shown = 0

    if is_main_process(rank):
        print(f"Evaluating on {total_samples} total samples across {world_size} GPUs...")
        print(f"[Rank {rank}] Processing {total} samples")
    else:
        print(f"[Rank {rank}] Processing {total} samples")

    # Process in batches - only show progress bar on main process
    iterator = range(0, total, batch_size)
    if is_main_process(rank):
        iterator = tqdm(iterator, desc=f"Rank {rank}")
    for i in iterator:
        batch_end = min(i + batch_size, total)
        batch = dataset[i:batch_end]
        prompts = batch["query"]

        # Note: prompts are already formatted with chat template from preprocessing
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)

        # Decode and extract answers
        prompt_len = inputs["input_ids"].shape[1]      # padded batch length
        gen_only_ids = outputs[:, prompt_len:]         # generated tokens only (works with LEFT padding)
        gen_texts = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        for j, generated_text in enumerate(gen_texts):
            predicted_answer = extract_answer(generated_text)
            ground_truth_label = int(batch["label"][j]) == 1

            if predicted_answer is None:
                no_answer_count += 1
                is_correct = False
            else:
                is_correct = (predicted_answer == ground_truth_label)

            if is_correct:
                correct += 1

            pred_dict = {
                "input": batch["query"][j],
                "ground_truth_label": ground_truth_label,
                "predicted_answer": predicted_answer,
                "generated_text": generated_text,
                "is_correct": is_correct,
            }

            # Add dataset-specific metadata if available
            if "num_variables" in batch:
                pred_dict["num_variables"] = batch["num_variables"][j]
            if "template" in batch:
                pred_dict["template"] = batch["template"][j]
            if "rung" in batch:
                pred_dict["rung"] = batch["rung"][j]
            if "query_type" in batch:
                pred_dict["query_type"] = batch["query_type"][j]
            if "dataset" in batch:
                pred_dict["dataset"] = batch["dataset"][j]

            predictions.append(pred_dict)

            # Display sample predictions
            if samples_shown < show_samples:
                samples_shown += 1
                print(f"SAMPLE {samples_shown}/{show_samples}")
                print(f"\n[PROMPT]")
                print(prompts[j])
                print(f"\n[GENERATED ANSWER]")
                print(generated_text)
                print(f"[PREDICTED ANSWER]: {pred_dict['predicted_answer']}")
                print(f"[GROUND TRUTH LABEL]: {pred_dict['ground_truth_label']}")
                print(f"[CORRECT]: {'✓' if pred_dict['is_correct'] else '✗'}")

    accuracy = correct / total if total > 0 else 0

    # Calculate F1 score for binary labels
    tp = sum(1 for p in predictions if p['predicted_answer'] == True and p['ground_truth_label'] == True)
    fp = sum(1 for p in predictions if p['predicted_answer'] == True and p['ground_truth_label'] == False)
    fn = sum(1 for p in predictions if (p['predicted_answer'] == False or p['predicted_answer'] is None) and p['ground_truth_label'] == True)
    tn = sum(1 for p in predictions if p['predicted_answer'] == False and p['ground_truth_label'] == False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if is_main_process(rank):
        print(f"[Rank {rank}] Local results - Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  - F1 Score: {f1_score:.2%} (P={precision:.2%}, R={recall:.2%})")

    return {
        'predictions': predictions,
        'metrics': {
            'beginning': beginning,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'total_samples': total,
            'correct': correct,
            'no_answer_count': no_answer_count,
        }
    }


def merge_results(output_dir: str, world_size: int, timestamp: str) -> Dict:
    """Merge results from all rank files into a single result."""
    all_predictions = []

    for rank in range(world_size):
        rank_file = os.path.join(output_dir, f'results_rank{rank}_{timestamp}.json')
        with open(rank_file, 'r') as f:
            rank_results = json.load(f)
            all_predictions.extend(rank_results['predictions'])

    # Recalculate metrics from combined predictions
    total = len(all_predictions)
    correct = sum(1 for p in all_predictions if p['is_correct'])
    no_answer_count = sum(1 for p in all_predictions if p['predicted_answer'] is None)

    accuracy = correct / total if total > 0 else 0

    tp = sum(1 for p in all_predictions if p['predicted_answer'] == True and p['ground_truth_label'] == True)
    fp = sum(1 for p in all_predictions if p['predicted_answer'] == True and p['ground_truth_label'] == False)
    fn = sum(1 for p in all_predictions if (p['predicted_answer'] == False or p['predicted_answer'] is None) and p['ground_truth_label'] == True)
    tn = sum(1 for p in all_predictions if p['predicted_answer'] == False and p['ground_truth_label'] == False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'predictions': all_predictions,
        'metrics': {
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'total_samples': total,
            'correct': correct,
            'no_answer_count': no_answer_count,
        }
    }


def main():
    # Initialize distributed environment
    rank, world_size, local_rank = setup_distributed()

    # Accept config path from command line or use default
    config_path = sys.argv[1]
    config_path = os.path.abspath(config_path)
    if is_main_process(rank):
        print(f"Loading config from: {config_path}")
        print(f"Running distributed evaluation with {world_size} GPUs")

    config = load_config(config_path)

    # Setup output directory
    model_path = config['model_path']
    model_name = os.path.basename(model_path.rstrip("/\\"))
    output_dir = os.path.join(config['output_dir'], model_name)

    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)

    # Synchronize before loading model
    if dist.is_initialized():
        dist.barrier()

    # Create timestamp AFTER barrier so all ranks have synchronized
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model, tokenizer = load_model_and_tokenizer(config['model_path'], config['use_4bit'], local_rank)
    data_path = config['local_data_path']
    dataset = load_from_disk(data_path)['test']

    # Run evaluation on this rank's shard of the data
    local_results = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=config['max_new_tokens'],
        batch_size=config['batch_size'],
        max_samples=config['max_samples'],
        show_samples=config['show_samples'] if is_main_process(rank) else 0,
        beginning=config.get('beginning', 0),
        rank=rank,
        world_size=world_size,
    )

    # Each rank saves its own results
    rank_file = os.path.join(output_dir, f'results_rank{rank}_{timestamp}.json')
    with open(rank_file, 'w', encoding='utf-8') as f:
        json.dump(local_results, f, indent=2, ensure_ascii=False)
    print(f"[Rank {rank}] Saved results to {rank_file}")

    # Only rank 0 merges and prints final results
    # Poll for all rank files instead of using barrier (avoids NCCL timeout issues)
    if is_main_process(rank):
        import time
        expected_files = [os.path.join(output_dir, f'results_rank{r}_{timestamp}.json') for r in range(world_size)]
        print(f"Waiting for all {world_size} rank files to be written...")
        while True:
            missing = [f for f in expected_files if not os.path.exists(f)]
            if not missing:
                break
            print(f"  Still waiting for {len(missing)} files: {[os.path.basename(f) for f in missing]}")
            time.sleep(30)
        print("All rank files found. Merging results...")
        results = merge_results(output_dir, world_size, timestamp)

        print("\n" + "="*50)
        print("FINAL AGGREGATED RESULTS")
        print("="*50)
        print(f"Accuracy: {results['metrics']['accuracy']:.2%} ({results['metrics']['correct']}/{results['metrics']['total_samples']})")
        print(f"F1 Score: {results['metrics']['f1_score']:.2%} (P={results['metrics']['precision']:.2%}, R={results['metrics']['recall']:.2%})")

        # Save merged results
        complete_results_file = os.path.join(output_dir, f'complete_results_{timestamp}.json')
        with open(complete_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved complete results to {complete_results_file}")

        # Clean up rank files
        for r in range(world_size):
            rank_file = os.path.join(output_dir, f'results_rank{r}_{timestamp}.json')
            os.remove(rank_file)

        print(f"Evaluation complete! All outputs saved to {output_dir}")

    # Clean up distributed environment
    cleanup_distributed()
    

if __name__ == "__main__":
    main()
