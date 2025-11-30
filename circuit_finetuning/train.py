import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    TrainerCallback, EarlyStoppingCallback
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
)
from datasets import load_dataset
import bitsandbytes as bnb

# ==========================================
# 1. Custom KD Trainer
# ==========================================
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha = alpha
        self.T = temperature
        
        # Move teacher to correct device and freeze
        if self.teacher:
            self.teacher.eval()
            self.teacher.to(self.args.device)
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        student_loss = outputs.loss
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Align logits length
        min_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
        
        loss_kl = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean"
        ) * (self.T ** 2)
        
        total_loss = (self.alpha * student_loss) + ((1 - self.alpha) * loss_kl)
        return (total_loss, outputs) if return_outputs else total_loss

# ==========================================
# 2. Main Training Logic
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--eval_data_path", type=str, default="data/test.jsonl") # Added Eval Set
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--mode", type=str, choices=["fft", "lora", "qlora", "distill"], required=True)
    parser.add_argument("--teacher_model", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--epochs", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print(f"--> Starting Training in {args.mode} mode...")
    
    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"--> BF16 Support Detected: {use_bf16}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Train AND Eval datasets
    data_files = {"train": args.data_path}
    if os.path.exists(args.eval_data_path):
        data_files["test"] = args.eval_data_path
        print(f"--> Found eval set: {args.eval_data_path}")
    
    dataset = load_dataset("json", data_files=data_files)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["task_prompt"], 
            truncation=True, 
            padding="max_length", 
            max_length=64
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Device Map Logic
    if args.mode in ["fft", "distill"]:
        device_map = None 
    else:
        device_map = "auto"

    print(f"--> Loading Model with device_map={device_map}...")

    if args.mode == "qlora":
        bnb_config = torch.utils.base_quantization_config = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            quantization_config=bnb_config, 
            device_map=device_map
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.float16, 
            device_map=device_map
        )

    # Setup LoRA
    if args.mode in ["lora", "qlora"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
        
        # LoRA + Gradient Checkpointing Fix
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
        model.print_trainable_parameters()

    # Teacher for Distillation
    teacher = None
    if args.mode == "distill":
        print(f"--> Loading Teacher: {args.teacher_model}")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.mode}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,      
        optim="paged_adamw_32bit",
        
        # Logging & Evaluation Strategy
        logging_steps=50,
        eval_strategy="steps" if "test" in tokenized_datasets else "no",
        eval_steps=100,             # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,             # Save every 100 steps (matches eval)
        save_total_limit=3,         # Keep fewer checkpoints
        load_best_model_at_end=True, # Always revert to best checkpoint
        metric_for_best_model="loss",
        report_to="none"
    )

    callbacks = []
    if "test" in tokenized_datasets:
        # Stop early if loss doesn't improve for 3 evals (300 steps)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    if args.mode == "distill":
        trainer = DistillationTrainer(
            model=model,
            teacher_model=teacher,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("test"),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=callbacks
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("test"),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=callbacks
        )

    # Enable GC on the model explicitly if needed
    model.gradient_checkpointing_enable()

    trainer.train()
    
    final_path = f"{args.output_dir}/{args.mode}/final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"--> Model saved to {final_path}")

if __name__ == "__main__":
    train()