"""
Supervised Fine-Tuning (SFT) script for DeepSeek 32B with LoRA and 4-bit quantization.
Uses correct reasoning traces from GPT to finetune the model.
"""
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    HfArgumentParser
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig as TRLSFTConfig
import wandb
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SFTConfig(TRLSFTConfig):
    """Extended TRL SFTConfig with custom parameters."""

    # Model configuration
    model_name_or_path: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

    # Data configuration
    dataset_path: str = field(default="./data/processed/sft_deepseek")
    max_train_samples: Optional[int] = field(default=None)

    # LoRA configuration
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")

    # Quantization configuration
    use_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    use_nested_quant: bool = field(default=True)

    # Sequence length
    max_seq_length: int = field(default=4096)

    # WandB configuration
    wandb_project: str = field(default="corr2cause-sft") 


def create_peft_config(config: SFTConfig):
    """Create PEFT (LoRA) configuration."""
    target_modules = config.lora_target_modules.split(",")

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return peft_config


def load_model_and_tokenizer(config: SFTConfig):
    """Load model with 4-bit quantization and tokenizer."""
    print(f"Loading model: {config.model_name_or_path}")

    # Configure model loading (no device_map - let Trainer/Accelerate handle device placement for DDP)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
    
    model_kwargs = {"trust_remote_code": True, "quantization_config": bnb_config, "torch_dtype": torch.bfloat16    }

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs
    )

    # Prepare model for k-bit training if using quantization
    # use_gradient_checkpointing=False to avoid DDP + LoRA compatibility issues
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",  # Use right padding for training
        model_max_length=config.max_seq_length,  # Set max sequence length
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_formatting_func(tokenizer):
    """Create a formatting function with access to the tokenizer's EOS token."""
    eos_token = tokenizer.eos_token

    def formatting_prompts_func(example):
        """Format a single example for SFT training using chat template.

        The query field already contains the chat-formatted user prompt from preprocessing,
        which includes special tokens like <｜User｜> and <｜Assistant｜><think>.
        We need to append the assistant's response and close with the EOS token.
        """
        # Query already has: <bos><｜User｜>[prompt]<｜Assistant｜><think>\n
        # We need to add: [response]<｜end▁of▁sentence｜>
        text = f"{example['query']}{example['response']}{eos_token}"
        return text

    return formatting_prompts_func


def main():
    # Parse arguments
    parser = HfArgumentParser(SFTConfig)
    training_args = parser.parse_yaml_file(yaml_file=sys.argv[1])[0]

    # Initialize WandB if enabled
    if hasattr(training_args, 'report_to') and 'wandb' in training_args.report_to:
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name,
            config=vars(training_args)
        )

    # Load dataset
    print(f"\nLoading dataset from {training_args.dataset_path}...")
    dataset = load_from_disk(training_args.dataset_path)

    # Limit training samples if specified
    if training_args.max_train_samples is not None:
        dataset['train'] = dataset['train'].select(range(min(training_args.max_train_samples, len(dataset['train']))))
        print(f"Limited training to {len(dataset['train'])} samples")

    print(f"Train samples: {len(dataset['train'])}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(training_args)

    # Apply LoRA if enabled
    if training_args.use_lora:
        print("\nApplying LoRA configuration...")
        peft_config = create_peft_config(training_args)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Create trainer
    print("\nInitializing SFT Trainer...")
    formatting_func = create_formatting_func(tokenizer)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"].select(range(100, 105)),
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("\nTraining complete!")
    print(f"LoRA adapters saved to: {training_args.output_dir}")
    print(f"To continue with GRPO training, set model_name_or_path to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
