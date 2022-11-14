from transformers import pipeline
import torch

from fastapi import FastAPI


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
classifier = None

app = FastAPI()


@app.get("/update_model")
def update_model():
    try:
        global classifier
        classifier = pipeline(task="text-classification",
                              model='/tmp/dolphinscheduler/examples/yelp_review_full/service_model', device=device)
    except Exception as e:
        print(e)
        raise e

    return {"result": 1}


@app.get("/health")
def health():
    return {"result": int(classifier is not None)}


@app.get("/predict")
def predict(text: str):
    assert classifier is not None
    print(text)
    return classifier.predict(text)
