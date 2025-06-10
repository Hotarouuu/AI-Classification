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

load_dotenv()

wandb_key= os.getenv("WANDB")
hf = os.getenv("HUGGINGFACE_TOKEN")



os.environ["WANDB_SILENT"] = "True"

import wandb
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

login(token=hf)
wandb.login(key=wandb_key)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")



tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

processor = DataProcessor(tokenizer)
pre_train, pre_test = processor.pretraining_data()



# Define the quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True # Specify 8-bit loading within the config
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config, # Pass the BitsAndBytesConfig object
    device_map='auto',
)

model.config.pad_token_id = tokenizer.pad_token_id


config = LoraConfig(
    task_type = "CAUSAL_LM",
    r=8,
    lora_alpha=16,
    target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["causal"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

os.environ["WANDB_PROJECT"] = "SYA-AI"  


training_args = TrainingArguments(
    output_dir="LoRA-PreTraining_2",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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