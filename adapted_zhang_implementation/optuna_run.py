from state_env import State  # module with environment and dynamics
from RL_brain_pi_deep import DQNPrioritizedReplay  # sumtree version of DQN
import os
import numpy as np
import configparser
import time  # added to check running time
import optuna
import csv
import pandas as pd
import pickle
import tensorflow.compat.v1 as tf
import sys
from drl_training import qst_training, qst_validation
import multiprocessing

tf.disable_v2_behavior()
optuna.logging.set_verbosity(optuna.logging.ERROR)

new_config_file = True
config_file = sys.argv[1]  # input("Enter the config file name (default: config.ini): ") or "config.ini"

if new_config_file:
    os.system(f'python3 optuna_config.py {config_file}')

print(f"Using configuration from {config_file}")

# load config file
config_instance = configparser.ConfigParser()
config_instance.read(config_file + '.ini')

experiment_name = config_instance.get("experiment", "experiment_alias")
dirname = config_instance.get("experiment", "directory_name")

def objective(trial):
    print(f"Running trial {trial.number}")
    tf.reset_default_graph()

    # Create a fresh config instance for this trial
    local_config = configparser.ConfigParser()
    local_config.read(config_file + '.ini')
    local_dirname = local_config.get("experiment", "directory_name")

    # import parameters to optimize
    optimization_learning_parameters = dict(
        local_config.items('optimization_learning_parameters')
    )

    for key, value in optimization_learning_parameters.items():
        vals = value.split(',')
        type_str = ''.join(filter(str.isalpha, vals[3]))
        print(type_str)
        if type_str == "int":
            low = int(vals[0].lstrip('['))
            high = int(vals[1].rstrip(']'))
            trial_param = trial.suggest_int(key, low, high)
        elif type_str == "float":
            low = float(vals[0].lstrip('['))
            high = float(vals[1].rstrip(']'))
            log = vals[2].lower() == 'true'
            trial_param = trial.suggest_float(key, low, high, log=log)
        local_config.set("learning_parameters", key, str(trial_param))
        if key == "fc1_dims":
            local_config.set(
                "learning_parameters", "fc2_dims", str(trial_param // 3)
            )

    # Use Optuna's unique trial number for directory naming (parallel safe)
    trial_directory = f"{local_dirname}/trial_{trial.number}"
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)

    local_config.set("experiment", "experiment_alias", str(trial_directory))

    # Save the updated config_instance to a config file in the trial directory
    with open(os.path.join(trial_directory, "config.ini"), "w") as configfile:
        local_config.write(configfile)

    try:
        qst_training(
            config_instance=local_config, optuna_trial=trial, progress_bar=False
        )
    except Exception as e:
        print(f"[ERROR] Training failed for trial {trial.number}: {e}")
        return None

    noise = local_config.getboolean("noise_parameters", "noise")

    if not noise:
        print("Skipping validation as noise is not used.")
        best_fidelities_path = os.path.join(trial_directory, "best_fidelities.txt")
        if os.path.exists(best_fidelities_path):
            data = np.fromtxt(best_fidelities_path, dtype=float)
            if data.size > 0:
                top_10_mean = np.mean(data)
            else:
                print(f"Warning: {best_fidelities_path} is empty.")
                top_10_mean = 0.0
        else:
            print(f"Warning: {best_fidelities_path} does not exist.")
            top_10_mean = 0.0
        return top_10_mean

    selected_metric = local_config.get(
        "optuna_optimization", "optuna_metric"
    )

    try:
        value = qst_validation(
            config_instance=local_config,
            validation_metric=selected_metric,
            validation_episodes=100
        )
    except Exception as e:
        print(f"[ERROR] Validation failed for trial {trial.number}: {e}")
        return None

    return value  # Return the average Q-value of the last 100 episodes

# Limit number of threads for CPU optimization
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Use efficient sampler and pruner for CPU
sampler = optuna.samplers.TPESampler()
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner
)
ntrials = config_instance.getint("optuna_optimization", "ntrials")

# Use parallel execution for CPU optimization
n_jobs = min(multiprocessing.cpu_count(), 16)  # Limit to 16 or number of CPUs

study.optimize(
    objective,
    n_trials=ntrials,
    n_jobs=n_jobs,
    show_progress_bar=False,
)

best_trial = study.best_trial
best_params = study.best_params

best_results = {
    "soft_success_training_rate": best_trial.user_attrs.get(
        "soft_success_training_rate", 0
    ),
    "true_success_training_rate": best_trial.user_attrs.get(
        "true_success_training_rate", 0
    ),
    "max_fid": best_trial.user_attrs.get("max_fid", 0),
    "avg_Qvalue": best_trial.user_attrs.get("avg_Qvalue", 0),
}

# Save the best results and parameters to a pickle file
best_trial_file = os.path.join(dirname, "best_trial.pkl")

with open(best_trial_file, "wb") as f:
    pickle.dump({"best_params": best_params, "best_results": best_results}, f)
