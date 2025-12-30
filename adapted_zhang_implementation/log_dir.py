"""MLflow logging script for DQN training experiment results.

This script logs completed training runs to an MLflow tracking server. It reads
experiment configuration and results from a specified directory, creates an MLflow
experiment with appropriate tags, and logs all artifacts, parameters, and metrics
for visualization and comparison.

The script processes training results with binning for efficient metric logging
and includes both training and validation metrics when available.

Usage
-----
    python log_dir.py <experiment_directory>

Requirements
------------
The experiment directory must contain:
    - config.ini: Configuration file with experiment parameters
    - training_results.txt: Episode-level training metrics
    - training_metrics.pkl: Aggregated training statistics
    - validation_metrics.pkl: Validation statistics (optional)
    - Model checkpoints and other artifacts to log

MLflow Configuration
--------------------
The script connects to an MLflow tracking server at http://localhost:5005.
Ensure the server is running before executing this script.
"""
import os
import configparser
import sys
import mlflow
from mlflow.tracking import MlflowClient
import datetime
import pandas as pd
import pickle
from tqdm import tqdm

# Get experiment directory from command line argument
directory_name = sys.argv[1]

# =========================================================================
# CONFIGURATION FILE DISCOVERY AND PARSING
# =========================================================================

# Get list of files in directory
files = os.listdir(directory_name)

# Find config file in directory
config_files = [file for file in files if file.endswith('.ini')]

if len(config_files) > 1:
    raise Exception("Multiple config files found in the directory.")
elif len(config_files) == 1:
    config_file = config_files[0]
else:
    raise Exception("No config files were found in the directory.")

# Create associated config instance
config = configparser.ConfigParser()
config.read(os.path.join(directory_name, config_file))

# Convert the configuration to a dictionary
parameters = {section: dict(config.items(section)) for
              section in config.sections()}

experiment_tags = parameters["tags"]
system_parameters = parameters["system_parameters"]
learning_parameters = parameters["learning_parameters"]

# =========================================================================
# MLFLOW EXPERIMENT SETUP
# =========================================================================

# Set up MLflow tracking server and create experiment
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
    
    # Log all experiment artifacts (models, config, results)
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

    # Bin episodes for efficient metric logging (reduces number of logged points)
    nbins = 100
    metrics = list(df.columns)

    # Create binning column based on index (every 100 rows)
    df["binned_step"] = df.index // nbins

    # Group by binned_step and calculate average for each bin
    binned_df = df.groupby("binned_step").mean().reset_index()

    # Log binned metrics to MLflow for visualization
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
