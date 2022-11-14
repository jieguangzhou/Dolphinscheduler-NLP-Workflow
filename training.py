from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import json
from transformers import TrainingArguments, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
import os


dataset_name = "yelp_review_full"  # $PARAM:dataset_name
pretrained_model = "bert-base-cased"  # $PARAM:pretrained_model

tokenized_datasets = load_from_disk(
    f'/tmp/dolphinscheduler/examples/{dataset_name}/{pretrained_model}/data')

small_train_dataset = tokenized_datasets["train"].shuffle(
    seed=42).select(range(2000))

small_eval_dataset = small_train_dataset.select(range(1500))
small_train_dataset = small_train_dataset.select(range(1500, 2000))


small_test_dataset = tokenized_datasets["test"].shuffle(
    seed=42).select(range(1000))


model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

training_args = TrainingArguments(output_dir="test_trainer")

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model_path = f'/tmp/dolphinscheduler/examples/{dataset_name}/model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

log_metrics = trainer.evaluate(small_test_dataset)
log_params = {
    "dataset_name": dataset_name,
    "pretrained_model": pretrained_model,
}

json.dump(log_metrics, open(os.path.join(model_path, "log_metrics.json"), "w"))
json.dump(log_params, open(os.path.join(model_path, "log_params.json"), "w"))
