from transformers import AutoTokenizer
from datasets import load_dataset
import os

dataset_name = "yelp_review_full"  # $PARAM:dataset_name
pretrained_model = "bert-base-cased"  # $PARAM:pretrained_model

data_path = f'/tmp/dolphinscheduler/examples/{dataset_name}/{pretrained_model}/data'

if os.path.exists(data_path):
    print(f"{data_path} exists, skip data processing")

else:
    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets.save_to_disk(data_path)
    print(f"save data to {data_path}")
