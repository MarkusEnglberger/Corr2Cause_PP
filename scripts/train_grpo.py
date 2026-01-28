import os
import sys
import shutil
from transformers import (HfArgumentParser, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback)
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
from typing import Optional, List
import json
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

@dataclass
class ExtendedGRPOConfig(GRPOConfig):
    """Extended GRPO config with model loading and LoRA parameters."""
    # Model configuration
    model_name_or_path: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    # Dataset configuration
    dataset_path: str = field(default="./data/processed/grpo")
    max_train_samples: Optional[int] = field(default=None)
    use_4bit: bool = field(        default=False)
    use_8bit: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(  default=16)
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(
        default=0.05
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj"
    )

    # WandB configuration
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "WandB project name"}
    )


    # Completion logging configuration
    log_completions_to_file: bool = field(
        default=False,
        metadata={"help": "Whether to save completions and rewards to a JSON file"}
    )
    completions_log_interval: int = field(
        default=50,
        metadata={"help": "Log completions every N steps"}
    )

def extract_answer(text: str) -> Optional[bool]:
    """Extract the answer from generated text."""
    text_lower = text.lower()
    if "therefore: yes" in text_lower:
        return True
    elif "therefore: no" in text_lower:
        return False
    return None


def extract_edges(text: str) -> List[tuple]:
    """Extract predicted edges from generated text.

    Looks for patterns like "X has a direct causal effect on Y"
    Returns list of (source, target) tuples.
    """
    import re

    # Pattern to match "X has a direct causal effect on Y"
    # Case insensitive, allows for variable names with letters
    pattern = r'([A-Za-z][A-Za-z0-9]*)\s+has\s+a\s+direct\s+causal\s+effect\s+on\s+([A-Za-z][A-Za-z0-9]*)'

    matches = re.findall(pattern, text, re.IGNORECASE)

    # Normalize to uppercase for consistent comparison
    edges = [(m[0].upper(), m[1].upper()) for m in matches]

    # Remove duplicates while preserving order
    seen = set()
    unique_edges = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            unique_edges.append(edge)

    return unique_edges


_completion_log_file = None
_completion_log_step = [0]  # Use list to allow modification in nested function


def reward_fn_c2cp(prompts: List[str], completions: List[str], forced_edges: List[List], label: List[int], **kwargs) -> List[float]:
    """Reward function for C2CP task (combined edge prediction + hypothesis verification).

    Rewards:
    - Edge prediction: +0.2 per correct, -0.2 per missing, -0.2 per wrong
    - Hypothesis answer: +0.5 for correct, -0.5 for wrong, -0.25 for no answer

    The final reward is the sum of edge reward and hypothesis reward.
    """
    global _completion_log_file
    rewards = []

    for completion, true_edges_raw, true_label in zip(completions, forced_edges, label):
        # --- Edge Prediction Reward ---
        true_edges = set()
        for edge in true_edges_raw:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                true_edges.add((str(edge[0]).upper(), str(edge[1]).upper()))

        predicted_edges = set(extract_edges(completion))

        correct_edges = predicted_edges & true_edges
        missing_edges = true_edges - predicted_edges
        wrong_edges = predicted_edges - true_edges

        edge_reward = (
            0.2 * len(correct_edges)
            - 0.2 * len(missing_edges)
            - 0.2 * len(wrong_edges)
        )

        # --- Hypothesis Answer Reward ---
        pred_answer = extract_answer(completion)
        if pred_answer is None:
            answer_reward = -0.2
        elif pred_answer == bool(true_label):
            answer_reward = 0.3
        else:
            answer_reward = -0.3

        # --- Combined Reward ---
        reward = edge_reward + answer_reward

        # Mild length penalty to discourage overly long completions
        # Penalize completions longer than 800 tokens, scaling up to -0.1 at max length
        completion_len = len(completion)
        if completion_len > 3000:
            length_penalty = -0.00001 * (completion_len - 3000)  # -0.05 at 1300 chars
            reward += length_penalty

        rewards.append(reward)

    # Log completions and rewards to file (only on main process)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if _completion_log_file is not None and local_rank == 0:
        _completion_log_step[0] += 1
        if _completion_log_step[0] % 4 == 1:
            log_entry = {
                "step": _completion_log_step[0],
                "timestamp": datetime.now().isoformat(),
                "task_type": "c2cpSplit1",
                "samples": []
            }
            for i in range(len(completions)):
                log_entry["samples"].append({
                    "prompt": prompts[i] if i < len(prompts) else None,
                    "completion": completions[i] if i < len(completions) else None,
                    "reward": rewards[i] if i < len(rewards) else None,
                    "true_edges": list(forced_edges[i]) if i < len(forced_edges) else None,
                    "true_label": label[i] if i < len(label) else None,
                })
            with open(_completion_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return rewards


def merge_sft_adapters_to_base(sft_adapter_path: str, cache_dir: str = "./merged_models_cache") -> str:
    """
    Merge SFT adapters into the base model and save as a new 16-bit model.

    This is necessary because:
    1. 4-bit quantized models cannot have adapters merged directly
    2. We want fresh LoRA adapters for GRPO, not continuing SFT adapters

    The workflow:
    1. Load base model in 16-bit (requires ~15GB RAM for 7B model)
    2. Load and merge SFT adapters
    3. Save the merged model to cache
    4. Return the path to the merged model (to be reloaded in 4-bit)

    Args:
        sft_adapter_path: Path to the SFT adapter directory
        cache_dir: Directory to cache merged models

    Returns:
        Path to the merged model directory
    """
    # Read adapter config to get base model path
    adapter_config_path = os.path.join(sft_adapter_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config.get("base_model_name_or_path")

    # Create a unique cache path based on the SFT adapter path
    # This allows reusing merged models across runs
    sft_path_hash = hashlib.md5(sft_adapter_path.encode()).hexdigest()[:8]
    merged_model_name = f"merged_sft_{os.path.basename(sft_adapter_path)}_{sft_path_hash}"
    merged_model_path = os.path.join(cache_dir, merged_model_name)

    # Check if merged model already exists in cache
    if os.path.exists(os.path.join(merged_model_path, "config.json")):
        print(f"Found cached merged model at: {merged_model_path}")
        return merged_model_path

    print("="*80)
    print("MERGING SFT ADAPTERS INTO BASE MODEL")
    print("="*80)
    print(f"SFT adapters: {sft_adapter_path}")
    print(f"Base model: {base_model_path}")
    print(f"Merged model will be saved to: {merged_model_path}")
    print("\nNOTE: This requires ~15GB RAM for a 7B model. Loading in 16-bit...")
    print("="*80)

    # Step 1: Load base model in 16-bit (NOT quantized)
    print("\nStep 1/4: Loading base model in 16-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 16-bit for merging
        trust_remote_code=True,
        device_map="auto",  # Use available GPU/CPU
    )

    # Step 2: Load SFT adapters
    print("Step 2/4: Loading SFT adapters...")
    model_with_adapters = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        is_trainable=False  # Not training, just merging
    )

    # Step 3: Merge adapters into base model
    print("Step 3/4: Merging adapters (this may take a moment)...")
    merged_model = model_with_adapters.merge_and_unload()

    # Step 4: Save the merged model
    print(f"Step 4/4: Saving merged model to {merged_model_path}...")
    os.makedirs(merged_model_path, exist_ok=True)
    merged_model.save_pretrained(merged_model_path)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_model_path)

    # Clean up to free memory
    del merged_model
    del model_with_adapters
    del base_model
    torch.cuda.empty_cache()

    print("="*80)
    print(f"SFT adapters merged successfully!")
    print(f"Merged model saved to: {merged_model_path}")
    print("="*80 + "\n")

    return merged_model_path


def create_model_and_tokenizer(config: ExtendedGRPOConfig):
    """Create model and tokenizer based on config settings.

    Supports three modes:
    1. Resume from GRPO checkpoint: output_dir has checkpoint -> load and continue training
    2. Start from SFT adapters: Merge SFT adapters into base, then add fresh LoRA for GRPO
    3. Start fresh: Load from base model and create new LoRA adapters

    IMPORTANT: When starting from SFT adapters, we MERGE them into the base model
    and create NEW LoRA adapters for GRPO. This is because:
    - 4-bit models cannot have adapters merged directly
    - GRPO should train fresh adapters on top of the SFT-improved base

    Returns:
        tuple: (model, tokenizer, resume_from_checkpoint_path or None)
    """
    # Check for checkpoint: either adapter_config.json in output_dir (final save)
    # or checkpoint-* subdirectories (intermediate saves)
    resume_from_path = None
    if os.path.exists(os.path.join(config.output_dir, "adapter_config.json")):
        resume_from_path = config.output_dir
    elif os.path.exists(config.output_dir):
        # Check for checkpoint subdirectories
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(config.output_dir, latest_checkpoint)
            if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                resume_from_path = checkpoint_path
                print(f"Found intermediate checkpoint: {resume_from_path}")

    output_dir_has_checkpoint = resume_from_path is not None
    model_path_has_adapters = os.path.exists(os.path.join(config.model_name_or_path, "adapter_config.json"))

    # Configure model loading with optional quantization
    model_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"}

     # Configure model loading with optional quantization
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa", "quantization_config": bnb_config, "torch_dtype": torch.bfloat16}


    # Priority 1: Resume from checkpoint in output_dir
    if output_dir_has_checkpoint:
        print(f"Found checkpoint: {resume_from_path}")
        print("Resuming training...")

        with open(os.path.join(resume_from_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")

        tokenizer = AutoTokenizer.from_pretrained(resume_from_path, trust_remote_code=True, padding_side="left")
        print(f"Loading base model from: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, resume_from_path, is_trainable=True)

        print(f"Resumed from: {resume_from_path}")
        model.print_trainable_parameters()

        # Diagnostic: print adapter weight norms to verify weights are loaded
        print("\n" + "="*80)
        print("DIAGNOSTIC: Adapter weight norms after loading checkpoint")
        print("="*80)
        total_norm = 0.0
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param_norm = param.data.norm().item()
                total_norm += param_norm ** 2
                print(f"  {name}: norm = {param_norm:.6f}")
        print(f"\nTotal LoRA weight norm: {total_norm ** 0.5:.6f}")
        print("(If this is very small or zero, weights weren't loaded correctly)")
        print("="*80 + "\n")

        return model, tokenizer

    # Priority 2: Start from existing SFT adapters - MERGE them and create fresh GRPO adapters
    if model_path_has_adapters:
        print(f"Found SFT adapters in: {config.model_name_or_path}")
        print("Will MERGE SFT adapters into base model, then create fresh LoRA for GRPO")

        # Step 1: Merge SFT adapters into base model (in 16-bit)
        # This saves the merged model to cache for reuse
        merged_model_path = merge_sft_adapters_to_base(config.model_name_or_path)

        # Step 2: Load the merged model (now with SFT knowledge baked in)
        # This can be loaded in 4-bit since it's a regular model, not adapters
        print(f"\nLoading merged model from: {merged_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(merged_model_path, **model_kwargs)

        # Step 3: Create fresh LoRA adapters for GRPO training
        if config.use_lora:
            if config.use_4bit or config.use_8bit:
                model = prepare_model_for_kbit_training(model)
            target_modules = config.lora_target_modules.split(",")
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print("Created fresh LoRA adapters for GRPO (on top of merged SFT model)")
            model.print_trainable_parameters()
        else:
            print("WARNING: use_lora=False but starting from SFT adapters.")
            print("Consider enabling LoRA for GRPO training.")

        return model, tokenizer

    # Priority 3: Start fresh from base model (no SFT adapters)
    print(f"No adapters found. Starting fresh from: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)

    if config.use_lora:
        if config.use_4bit or config.use_8bit:
            model = prepare_model_for_kbit_training(model)
        target_modules = config.lora_target_modules.split(",")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        print("Created new LoRA adapters")
        model.print_trainable_parameters()

    return model, tokenizer
    
def main():
    parser = HfArgumentParser(ExtendedGRPOConfig)
    config = parser.parse_yaml_file(yaml_file=sys.argv[1], allow_extra_keys=True)[0]

    # Set wandb project via environment variable (Trainer will handle initialization)
    # This avoids conflicts with the Trainer's internal wandb setup
    if hasattr(config, 'wandb_project') and config.wandb_project:
        os.environ['WANDB_PROJECT'] = config.wandb_project

    # Initialize completion logging
    global _completion_log_file
    os.makedirs(config.output_dir, exist_ok=True)
    _completion_log_file = os.path.join(config.output_dir, "completions_rewards.jsonl")
    print(f"Logging completions and rewards to: {_completion_log_file}")

    # Load dataset
    dataset = load_from_disk(config.dataset_path)

    # Rename 'query' to 'prompt' for GRPO trainer compatibility
    dataset = dataset.rename_column('query', 'prompt')

    # Limit training samples if specified
    if config.max_train_samples is not None:
        dataset['train'] = dataset['train'].select(range(config.max_train_samples))

    # Select train/eval subsets
    train_dataset = dataset["train"]  #.select(range(0, 3100))
    eval_dataset = dataset["validation"].select(range(10, 20))

    # Compute and display expected training steps
    num_train_samples = len(train_dataset)
    num_gpus = int(os.environ.get('WORLD_SIZE', 1))
    effective_batch_size = config.per_device_train_batch_size * num_gpus * config.gradient_accumulation_steps

    # GRPO-specific: num_iterations controls how many times we update per batch of prompts
    num_iterations = getattr(config, 'num_iterations', 1)

    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        # Each "step" in GRPO processes one batch of prompts through num_iterations updates
        steps_per_epoch = (num_train_samples + effective_batch_size - 1) // effective_batch_size
        total_steps = int(steps_per_epoch * config.num_train_epochs)

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Training samples:           {num_train_samples}")
    print(f"Number of GPUs:             {num_gpus}")
    print(f"Per-device batch size:      {config.per_device_train_batch_size}")
    print(f"Gradient accumulation:      {config.gradient_accumulation_steps}")
    print(f"Effective batch size:       {effective_batch_size}")
    print(f"Number of epochs:           {config.num_train_epochs}")
    print(f"GRPO num_iterations:        {num_iterations}")
    print(f"GRPO num_generations:       {getattr(config, 'num_generations', 'N/A')}")
    print("-"*80)
    print(f"EXPECTED TOTAL STEPS:       {total_steps}")
    print(f"Eval steps:                 {getattr(config, 'eval_steps', 'N/A')}")
    print(f"Save steps:                 {getattr(config, 'save_steps', 'N/A')}")
    print("="*80 + "\n")

    # Create model and tokenizer from config
    model, tokenizer = create_model_and_tokenizer(config)

    # Enable gradient checkpointing if specified
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn_c2cp],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Ensure value head is in the correct dtype for 4-bit training
    if config.use_4bit and hasattr(trainer.model, 'v_head'):
        trainer.model.v_head = trainer.model.v_head.to(torch.bfloat16)

    # Fix: Reset "ref" adapter weights to zero so reference model = base model
    # GRPOTrainer copies "default" adapter to "ref", but on resume this means
    # both policy and reference have the same trained weights, causing KL ≈ 0.
    # Zeroing the "ref" adapter makes it a no-op, so reference = base model.
    ref_adapter_reset_count = 0
    for name, param in trainer.model.named_parameters():
        if ".ref." in name and "lora_" in name:
            param.data.zero_()
            ref_adapter_reset_count += 1
    if ref_adapter_reset_count > 0:
        print(f"Reset {ref_adapter_reset_count} 'ref' adapter parameters to zero (reference = base model)")

    # Diagnostic: Check adapter weights AFTER GRPOTrainer initialization
    # This verifies TRL didn't reset or modify the weights
    print("\n" + "="*80)
    print("DIAGNOSTIC: Adapter weight norms AFTER GRPOTrainer initialization")
    print("="*80)
    total_norm = 0.0
    for name, param in trainer.model.named_parameters():
        if "lora_" in name and param.requires_grad:
            param_norm = param.data.norm().item()
            total_norm += param_norm ** 2
    print(f"Total LoRA weight norm: {total_norm ** 0.5:.6f}")

    # Also check if there's a "ref" adapter and its weights
    if hasattr(trainer.model, 'peft_config'):
        print(f"PEFT adapters present: {list(trainer.model.peft_config.keys())}")
    if hasattr(trainer.model, 'active_adapter'):
        print(f"Active adapter: {trainer.model.active_adapter}")
    print("="*80 + "\n")

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Print loading instructions
    print("\n" + "="*80)
    print("Training completed! Model saved to:", config.output_dir)
    print("="*80)
    print("\nTo load this model for inference or resume training:")
    print("1. GRPO adapters: ", config.output_dir)
    print("2. Base model: ", config.model_name_or_path)
    print("\nLoading for inference:")
    print("  from transformers import AutoModelForCausalLM")
    print("  from peft import PeftModel")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{config.model_name_or_path}')")
    print(f"  model = PeftModel.from_pretrained(base, '{config.output_dir}')")
    print("\nResume GRPO training:")
    print(f"  Set model_name_or_path: '{config.output_dir}' in your config")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

