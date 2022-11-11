from transformers import AutoTokenizer
from datasets import load_dataset
import os

dataset_name = "yelp_review_full"
model_name = "bert-base-cased"

data_path = f'~/autodl-tmp/{dataset_name}/data'

if os.path.exists(data_path):
    print(f"{data_path} exists, skip data processing")

else:
    dataset = load_dataset(dataset_name)
    dataset["train"][100]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets.save_to_disk(data_path)
    print(f"save data to {data_path}")
