from transformers import pipeline
import torch

from fastapi import FastAPI


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
classifier = None

app = FastAPI()


@app.get("/update_model")
def read_root():
    try:
        global classifier
        classifier = pipeline(task="text-classification",
                              model='~/autodl-tmp/yelp_review_full/service_model/artifact', device=device)
    except Exception as e:
        print(e)
        raise e

    return {"result": 1}


@app.get("/predict")
def read_item(text: str):
    assert classifier is not None
    print(text)
    return classifier.predict(text)
