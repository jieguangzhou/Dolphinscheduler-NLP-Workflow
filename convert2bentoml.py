import bentoml
from transformers import pipeline

dataset_name = "yelp_review_full"

model_path = f"~/autodl-tmp/{dataset_name}/model"

classifier = pipeline(task="text-classification", model=model_path)


bento_model = bentoml.transformers.save_model(name="classifer", pipeline=classifier)
