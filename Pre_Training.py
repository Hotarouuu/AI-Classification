import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig, BitsAndBytesConfig
from utils import DataProcessor
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
import os
import polars as pl
import bitsandbytes as bnb
from dotenv import load_dotenv
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Pre-training script with configurable parameters')
parser.add_argument('--run_name', type=str, default='LoRA-PreTraining',
                    help='Name of the run for output directory and wandb')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training and evaluation')
args = parser.parse_args()

run_name = args.run_name


load_dotenv()

wandb_key= os.getenv("WANDB")
hf = os.getenv("HUGGINGFACE_TOKEN")



os.environ["WANDB_SILENT"] = "True"

import wandb
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

login(token=hf)
wandb.login(key=wandb_key)


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")



tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

processor = DataProcessor(tokenizer)
pre_train, pre_test = processor.pretraining_data()

total_tokens = sum(len(x) for x in pre_train["input_ids"])
print(f"Total de tokens: {total_tokens}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"  # ou float16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.pad_token_id = tokenizer.pad_token_id


config = LoraConfig(
    task_type = "CAUSAL_LM",
    r=8,
    lora_alpha=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["causal"],
)
model = get_peft_model(model, config)

print(model.print_trainable_parameters())

os.environ["WANDB_PROJECT"] = "SYA-AI"  


training_args = TrainingArguments(
    output_dir=run_name,
    learning_rate=1e-4,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=5,
    weight_decay=0.05,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps = 10000,
    save_steps = 10000,
    load_best_model_at_end=False,
    eval_accumulation_steps=1,
    push_to_hub=False,
    report_to="wandb",
    fp16= False,
    logging_steps=10,
    label_names=['labels'],
    remove_unused_columns=False,
    max_grad_norm=1.0
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pre_train,
    eval_dataset = pre_test,
    data_collator=data_collator)

trainer.train()