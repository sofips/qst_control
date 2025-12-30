import configparser
import mlflow
from mlflow.tracking import MlflowClient
import datetime
import sys
import os
import pandas as pd
import pickle
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

parent_directory = sys.argv[1]

files = os.listdir(parent_directory)

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
config.read(os.path.join(parent_directory, config_file))

experiment_name = config.get("experiment", "experiment_alias")

# set mlflow uri and experiment
tracking_uri = "http://127.0.0.1:5005"
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.create_experiment(name=experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)
trial_dirs = [item for item in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, item))]
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name, nested=True):
    for trial_directory in trial_dirs:
        with mlflow.start_run(run_name=trial_directory, nested=True):
            trial_directory = os.path.join(parent_directory, trial_directory)
            files = os.listdir(trial_directory)
            print(files)
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
            config.read(os.path.join(trial_directory, config_file))

            # Convert the configuration to a dictionary
            parameters = {section: dict(config.items(section)) for section in 
                          config.sections()}

            experiment_tags = parameters["tags"]
            system_parameters = parameters["system_parameters"]
            learning_parameters = parameters["learning_parameters"]
            noise_parameters = parameters["noise_parameters"]
            
            mlflow.log_params(learning_parameters)
            mlflow.log_params(system_parameters)
            mlflow.log_params(noise_parameters)
            mlflow.set_tags(experiment_tags)
            
            # Log the artifacts from the trial directory
            mlflow.log_artifacts(trial_directory)

            result_files = [file for file in files if 'training_results' in file]
            print(f"Result files: {result_files}")

            if len(config_files) > 1:
                raise Exception("Multiple result files found in the directory.")
            elif len(config_files) == 1:
                results_file = result_files[0]
                results_file = os.path.join(trial_directory, results_file)
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
            
            for index, row in binned_df.iterrows():
                for metric in metrics:
                # Log binned metrics
                    print(f"Logging binned metrics for step {index}")
                    mlflow.log_metric(f"binned_{metric}", row[f"{metric}"], step=index)
                 

            metrics_file = os.path.join(trial_directory, 'training_metrics.pkl')
            # Load the dictionary back to verify
            with open(metrics_file, "rb") as f:
                loaded_metrics = pickle.load(f)

            # Log the metrics
            for key, value in loaded_metrics.items():
                mlflow.log_metric(key, value)

            if 'validation_metrics.pkl' in files:

                validation_metrics_file = os.path.join(trial_directory, 'validation_metrics.pkl')
                with open(validation_metrics_file, "rb") as f:
                    validation_metrics = pickle.load(f)
                
                # Log the validation metrics
                for key, value in validation_metrics.items():
                    mlflow.log_metric(f"{key}", value)
