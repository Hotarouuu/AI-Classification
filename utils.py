import polars as pl
from transformers import AutoTokenizer
from datasets import Dataset


class DataProcessor:
    def __init__(self, tokenizer):
        self.lora_train = None
        self.lora_test = None
        self.pre_train = None
        self.pre_test = None
        self.tokenizer = tokenizer

    def data_treatment(self):
        df = pl.read_parquet('hf://datasets/ahmadreza13/human-vs-Ai-generated-dataset/data/train-*.parquet')
        ai_generated = df.filter(df['generated'] == 1)
        self.lora_train = ai_generated[:100000]
        self.lora_test = ai_generated[100000:125000]
        self.lora_train = pl.DataFrame(self.lora_train['data'])
        self.lora_test = pl.DataFrame(self.lora_test['data'])
        return self.lora_train, self.lora_test

    def preprocess_function(self, examples):
        # Tokenize the texts. The data collator will handle padding and labels.
        tokenized_inputs = self.tokenizer(examples['data'], truncation=True, padding=True, return_tensors="pt")
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].clone()
        return tokenized_inputs

    def pretraining_data(self):
        self.lora_train, self.lora_test = self.data_treatment()
        self.lora_train = Dataset.from_polars(self.lora_train)
        self.lora_test = Dataset.from_polars(self.lora_test)
        self.pre_train = self.lora_train.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['data'] # Remove the original 'text' column
        )

        self.pre_test = self.lora_test.map(
            self.preprocess_function,
            batched=True,
            remove_columns=['data'] # Remove the original 'text' column
        )

        return self.pre_train, self.pre_test

