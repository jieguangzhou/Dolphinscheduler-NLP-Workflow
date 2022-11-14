import mlflow
import shutil

remote_server_uri = "http://127.0.0.1:5000"  # $PARAM:
model_name = "yelp_review_full"  # $PARAM:
model_version = "1"  # $PARAM:

save_model_path = f'/tmp/dolphinscheduler/examples/{model_name}/service_model'

print(f"delete {save_model_path} if exist")
shutil.rmtree(save_model_path, ignore_errors=True)


mlflow.set_tracking_uri(remote_server_uri)

client = mlflow.MlflowClient()


if model_version.isnumeric():
    def filter(x): return x.version == model_version
else:
    def filter(x): return x.current_stage == model_version

all_model = client.search_model_versions("name='yelp_review_full'")

specified_model = [model for model in all_model if filter(model)]

if not specified_model:
    assert f"Cant not find model {model_name} with version {model_version}"
else:
    specified_model = specified_model[0]


run_id = specified_model.run_id

print(f"download model: {model_name}:{model_version} to {save_model_path}")
mlflow.artifacts.download_artifacts(
    run_id=specified_model.run_id, dst_path=save_model_path)
