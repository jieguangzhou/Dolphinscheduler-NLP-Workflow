import mlflow
import json
import os

remote_server_uri = "http://127.0.0.1:5000"  # $PARAM:
model_name = "yelp_review_full"  # $PARAM:dataset_name

model_path = f'/tmp/dolphinscheduler/examples/{model_name}/model'


mlflow.set_tracking_uri(remote_server_uri)

try:
    client = mlflow.MlflowClient()
    create_model_response = client.create_registered_model(model_name)
except mlflow.exceptions.MlflowException as e:
    print(
        "Registered model '%s' already exists. "
        % model_name
    )

mlflow.set_experiment(model_name)


with mlflow.start_run() as run:
    params = json.load(
        open(os.path.join(model_path, "log_params.json")))
    mlflow.log_params(params)

    metrics = json.load(
        open(os.path.join(model_path, "log_metrics.json")))
    mlflow.log_metrics(metrics)

    mlflow.log_artifacts(model_path)


model_uri = f"runs:/{run.info.run_id}/model"
mv = mlflow.register_model(model_uri, model_name)
print("Name: {}".format(mv.name))
print("Version: {}".format(mv.version))
