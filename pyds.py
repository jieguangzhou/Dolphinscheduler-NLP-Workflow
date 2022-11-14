from pydolphinscheduler.tasks import Python, Http
from pydolphinscheduler.core.process_definition import ProcessDefinition

CONVERT_TAG = "# $PARAM:"


def load_script(path):
    with open(path, 'r') as f:
        script_lines = []
        for line in f:
            if CONVERT_TAG not in line:
                script_lines.append(line)
                continue

            base_line, annotation = line.rstrip().split(CONVERT_TAG)
            param_name, param_value = base_line.split("=")
            param_value = param_value.strip()

            annotation = annotation or param_name.strip()
            annotation = "${%s}" % annotation.strip()

            if param_value.startswith('"') and param_value.endswith('"'):
                annotation = "\"" + annotation + "\""

            new_line = param_name + "= " + annotation + "\n"
            script_lines.append("# original: " + line)
            script_lines.append(new_line)

        script = "".join(script_lines)
        return script


environment_name = "transformers-textclassification"


# default params witch will be used in runing workflow
dataset_name = "yelp_review_full"
pretrained_model = "bert-base-cased"
remote_server_uri = "http://127.0.0.1:5000"
model_name = "yelp_review_full"
model_version = "1"


with ProcessDefinition(
    name="training",
    param={
        "dataset_name": dataset_name,
        "pretrained_model": pretrained_model,
        "remote_server_uri": remote_server_uri,
    }
) as pd:
    task_data_preprocessing = Python(name="data_preprocessing",
                                     definition=load_script(
                                         "data_preprocessing.py"),
                                     environment_name=environment_name)

    task_training = Python(name="training",
                           definition=load_script("training.py"),
                           environment_name=environment_name)

    task_mlflow_track = Python(name="mlflow_track",
                               definition=load_script("mlflow_track.py"),
                               environment_name=environment_name)

    task_data_preprocessing >> task_training >> task_mlflow_track

    pd.submit()


with ProcessDefinition(
    name="deploy",
    param={
        "remote_server_uri": remote_server_uri,
        "model_name": model_name,
        "model_version": model_version,
    }
) as pd:
    task_fetch_model = Python(name="fetch_model",
                              definition=load_script(
                                  "fetch_model.py"),
                              environment_name=environment_name)

    task_check_service = Http(
        name="check_service",
        url="http://localhost:8000/health",
        http_method="GET")

    task_update_model = Http(
        name="update_model",
        url="http://localhost:8000/update_model",
        http_method="GET")

    task_test_service = Http(
        name="test_service",
        url="http://localhost:8000/predict",
        http_method="GET",
        http_params=[
            {"prop": "text", "httpParametersType": "PARAMETER", "value": "test text"},
        ]

    )

    task_fetch_model >> task_check_service >> task_update_model >> task_test_service

    pd.submit()
