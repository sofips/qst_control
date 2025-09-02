import os
import configparser
import sys
import mlflow
from mlflow.tracking import MlflowClient
import datetime
import pandas as pd
import pickle
from tqdm import tqdm

directory_name = sys.argv[1]

# get list of files in directory
files = os.listdir(directory_name)

# find config file in directory
config_files = [file for file in files if file.endswith('.ini')]

if len(config_files) > 1:
    raise Exception("Multiple config files found in the directory.")
elif len(config_files) == 1:
    config_file = config_files[0]
else:
    raise Exception("No config files were found in the directory.")

# create associated config instance
config = configparser.ConfigParser()
config.read(os.path.join(directory_name, config_file))

# Convert the configuration to a dictionary
parameters = {section: dict(config.items(section)) for
              section in config.sections()}

experiment_tags = parameters["tags"]
system_parameters = parameters["system_parameters"]
learning_parameters = parameters["learning_parameters"]

# set associated mlflow instance
experiment_name = config.get("experiment", "experiment_alias")
tracking_uri = "http://localhost:5005"
client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.create_experiment(name=experiment_name,
                                      tags=experiment_tags)
print(f"Experiment created: {experiment_name}")
now = datetime.datetime.now()
date_str = now.strftime("%Y%m%d_%H%M%S")
run_name = f"{experiment_name}_{date_str}"

mlflow.set_tracking_uri(uri=tracking_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=run_name, nested=False):
    print(f"Logging to experiment: {experiment_name}")
    mlflow.log_artifacts(directory_name)
    mlflow.log_params(learning_parameters)
    mlflow.log_params(system_parameters)
    mlflow.set_tags(experiment_tags)

    result_files = [file for file in files if 'training_results' in file]

    if len(config_files) > 1:
        raise Exception("Multiple result files found in the directory.")
    elif len(config_files) == 1:
        results_file = result_files[0]
        results_file = os.path.join(directory_name, results_file)
    else:
        raise Exception("No result files were found in the directory.")
    print(f"Results file: {results_file}")

    # Cargar el DataFrame
    df = pd.read_csv(results_file, delimiter=" ", index_col=False, header=0)
    df.reset_index(drop=True, inplace=True)

    nbins = 100

    metrics = list(df.columns)

    # Crear una columna de binning basada en el índice (cada 100 filas)
    df["binned_step"] = df.index // nbins

    # Agrupar por binned_step y calcular el promedio de cada bin
    binned_df = df.groupby("binned_step").mean().reset_index()

    for index, row in tqdm(binned_df.iterrows(), total=len(binned_df),
                           desc="Logging binned metrics"):
        for metric in metrics:
            mlflow.log_metric(f"binned_{metric}", row[f"{metric}"], step=index)

    metrics_file = os.path.join(directory_name, 'training_metrics.pkl')
    # Load the dictionary back to verify
    with open(metrics_file, "rb") as f:
        loaded_metrics = pickle.load(f)

    # Log the metrics
    for key, value in loaded_metrics.items():
        mlflow.log_metric(key, value)

    validation_metrics_file = os.path.join(directory_name,
                                           'validation_metrics.pkl')

    if 'validation_metrics.pkl' in files:
        with open(validation_metrics_file, "rb") as f:
            validation_metrics = pickle.load(f)

        # Log the validation metrics
        for key, value in validation_metrics.items():
            mlflow.log_metric(f"{key}", value)
