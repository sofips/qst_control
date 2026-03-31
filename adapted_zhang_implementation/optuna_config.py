"""Configuration file generator for Optuna hyperparameter optimization.

Generates .ini configuration files for Optuna-based hyperparameter tuning of
DQN agents for quantum state transfer. Defines optimization ranges for learning
hyperparameters (learning rate, gamma, network dimensions) and system parameters.

This script is called automatically by optuna_run.py at the start of an
optimization study to create the base configuration that will be varied across
trials.

Usage
-----
    python optuna_config.py <experiment_name>
    python optuna_config.py interactive

Arguments
---------
    experiment_name : str
        Name for the optimization study. Creates directory 'opt_for_<name>'.
    interactive : str
        Enables interactive mode with prompts for all configuration options.

Examples
--------
    python optuna_config.py n16_optimization
    python optuna_config.py interactive

Configuration
-------------
    Modify optimization_learning_parameters and optimization_system_parameters
    dictionaries to define hyperparameter search spaces. Each parameter has:
    [lower_bound, upper_bound, use_log_scale, type]
"""

import configparser
import os
import sys
import shutil
config = configparser.ConfigParser()

# =========================================================================
# EXPERIMENT CONFIGURATION MODE
# =========================================================================

if sys.argv[1] == 'interactive':
    # -----------------------------------------------------------------#
    new_experiment = input("New experiment? (y/n): ")
    new_experiment = True if new_experiment == "y" else False if new_experiment == "n" else sys.exit("Error: y o n")
    experiment_alias =  input("Experiment alias/name: ")
    dirname = 'opt_for_' + experiment_alias
    experiment_description = input('Notes / Comments: ')
    optuna_metric = input("Optuna metric to optimize ('0 for general max fid', '1  for average max fid', '2 for average time max fid', '3 for average QValue'): ")
    
    if optuna_metric not in ["0", "1", "2", "3"]:
        raise ValueError(f"Error: '{optuna_metric}' is not a valid metric. Choose from ['0', '1', '2', '3'].")
    
    possible_metrics = [
    "general_val_fidelity",
    "average_val_fidelity",
    "average_time_max_fidelity",
    "average_QValue"
    ]

    optuna_metric = possible_metrics[int(optuna_metric)]

else:
    new_experiment =  True  
    experiment_alias = sys.argv[1]
    dirname = 'opt_for_' + experiment_alias
    optuna_metric = "average_val_fidelity" 
    experiment_description = 'optimization of fc1_dims, lr and gamma value for original actions and reward in chain of size 13 noise (0,25, =25)' 
    optuna_metric = "average_val_fidelity"


# =========================================================================
# OPTUNA OPTIMIZATION PARAMETERS
# =========================================================================

# Define hyperparameter search spaces
# Format: {parameter_name: [min, max, log_scale, type]}
optimization_system_parameters = {
    # ("learning_rate", "float"): [0.001, 0.01, True],
    # ("gamma", "float"): [0.90, 0.95, False],
    # ("batch_size", "int"): [16, 32, True],
}

optimization_learning_parameters = {
    ("gamma"): [0.9, 1., False, "float"],
    ("fc1_dims"): [1024, 4096, False, "int"],
    ("learning_rate"): [0.00001, 0.01, True, "float"],
}

ntrials = 64

# Display optimization configuration
print("Running optuna optimization for the following learning parameters:")
for param, values in optimization_learning_parameters.items():
    print(f"{param}: {values[0]} to {values[1]}, logscale: {values[2]}")

print("Running optuna optimization for the following system parameters:")
for param, values in optimization_system_parameters.items():
    print(f"{param}: {values[0]} to {values[1]}, logscale: {values[2]}")
#=========================================================================
# SYSTEM PARAMETERS
# =========================================================================

chain_length = 16
tstep_length = 0.15
tolerance = 0.05
max_t_steps = 5*chain_length
field_strength = 100
coupling = 1

# =========================================================================
# NOISE PARAMETERS
# =========================================================================

noise = True
noise_probability = 0.10
noise_amplitude = 0.10
#=========================================================================
# LEARNING HYPERPARAMETERS (Base values - will be optimized by Optuna)
# =========================================================================

prioritized = True
number_of_features = 2 * chain_length
number_of_episodes = 30000
step_learning_interval = 64

learning_rate = None
gamma = None

# memory
replace_target_iter = 200
memory_size = 40000
batch_size = 32

# epsilon
epsilon = 0.99 
epsilon_increment = 0.001
epsilon_increment = 0.0001

# dqn
fc1_dims = 0
fc2_dims = fc1_dims//3
dropout = 0.0 #not yet implemented in DQNPrioritizedReplay

reward_function = "original"     # "original" , "full reward", "ipr", "site evolution"
action_set = "oaps"   # "zhang", "oaps" (action per site)
n_actions = 16 if action_set == "zhang" else chain_length + 1  if action_set == "oaps" else 0

# ------------------------------------------------------------------------------
config["optuna_optimization"] = {

    "ntrials": str(ntrials),
    "optuna_metric": optuna_metric,

}

config["optimization_system_parameters"] = {param: str(values) for param, values in optimization_system_parameters.items()}
config["optimization_learning_parameters"] = {param: str(values) for param, values in optimization_learning_parameters.items()}


config["experiment"] = {"new_experiment": str(new_experiment),
                        "experiment_alias": experiment_alias,
                        "experiment_description": experiment_description,
                        "directory_name": dirname,
                       }

config["system_parameters"] = {
    "chain_length": str(chain_length),
    "tstep_length": str(tstep_length),
    "tolerance": str(tolerance),
    "max_t_steps": str(max_t_steps),
    "field_strength": str(field_strength),
    "coupling": str(coupling),
    "n_actions": str(n_actions),
    "action_set": action_set,
}

config["learning_parameters"] = {
    "prioritized_experience_replay": str(prioritized),
    "number_of_features": str(number_of_features),
    "number_of_episodes": str(number_of_episodes),
    "learning_rate": str(learning_rate),
    "gamma": str(gamma),
    "replace_target_iter": str(replace_target_iter),
    "memory_size": str(memory_size),
    "batch_size": str(batch_size),
    "epsilon": str(epsilon),
    "epsilon_increment": str(epsilon_increment),
    "fc1_dims": str(fc1_dims),
    "fc2_dims": str(fc2_dims),
    "dropout": str(dropout),
    "reward_function": reward_function,
}

config["noise_parameters"] = {
    "noise": str(noise),
    "noise_probability": str(noise_probability),
    "noise_amplitude": str(noise_amplitude),
}
# ---------------------------------------------------


config["tags"] = {
    "reward_function": reward_function,
    "prioritized": 'prioritized' if prioritized else 'not prioritized',
    "mlflow.note.content": experiment_description,
    "action set": action_set,
    "chain_length": str(chain_length),
    "noise": 'yes' if noise else 'no',
    "optuna_optimization": True
}
# ---------------------------------------------------

config_name = sys.argv[1] + '.ini'

try:
    os.mkdir(dirname)
except FileExistsError:
    if new_experiment==False:
        raise FileExistsError(f"Directory '{dirname}' already exists. Please choose a different experiment alias.")
    else:
        print(f"Directory '{dirname}' already exists. Overwriting...")

with open(config_name, "w") as configfile:
    config.write(configfile)

print(f"Config file {config_name} generated for experiment '{experiment_alias}'")

dest_path = os.path.join(dirname, config_name)
shutil.copy(config_name, dest_path)
print(f"Config file also copied to directory '{dirname}' as '{config_name}'")

