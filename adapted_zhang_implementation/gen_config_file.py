#####################################################################
# Generate configuration files for RL experiments with ODC
#####################################################################

import configparser
import os
import sys
import shutil
config = configparser.ConfigParser()

# -----------------------------------------------------------------#
#new_experiment = input("New experiment? (y/n): ")
#new_experiment = True if new_experiment == "y" else False if new_experiment == "n" else sys.exit("Error: y o n")
#experiment_alias = input("Experiment alias/name: ")
#experiment_description = input('Notes / Comments: ')

experiment_alias = 'n10_og_005'
experiment_description = 'n10 og parameters'

#run = input("Run experiment (y/n): ")
#run = True if run == "y" else False if run == "n" else sys.exit("Error: y o n")

new_experiment = True
#experiment_alias = 'n16_mse_gamma2'
#experiment_description = 'slow learning with og actions and reward and og gamma'
run = False
# -----------------------------------------------------------------#
#                         SYSTEM PARAMETERS                       #
# -----------------------------------------------------------------#


chain_length = 10
tstep_length = 0.15
tolerance = 0.05
max_t_steps = 5*chain_length
field_strength = 100
coupling = 1

# -----------------------------------------------------------------#
#                    NOISE PARAMETERS                              #
# -----------------------------------------------------------------#
noise = False
noise_probability = 0.
noise_amplitude = 0.

# -----------------------------------------------------------------#
#                    LEARNING HYPERPARAMETERS                     #
# -----------------------------------------------------------------#
prioritized = True
number_of_features = 2 * chain_length
number_of_episodes = 50000
step_learning_interval = 5

learning_rate = 0.01
gamma = 0.95 # 1 for no decay

# memory
replace_target_iter = 200
memory_size = 40000
batch_size = 32

# epsilon
epsilon = 0.99
epsilon_increment = 0.0001

# dqn
fc1_dims = 120
fc2_dims = fc1_dims//3
dropout = 0.0 #not yet implemented in DQNPrioritizedReplay

reward_function = "original"     # "original" , "full reward", "ipr", "site evolution"
action_set = "zhang"   # "zhang", "oaps" (action per site)

n_actions = 16 if action_set == "zhang" else chain_length + 1  if action_set == "oaps" else 0

# ---------------------------------------------------
config["experiment"] = {"new_experiment": str(new_experiment),
                        "experiment_alias": experiment_alias,
                        "experiment_description": experiment_description,
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
}
# ---------------------------------------------------

config_name = sys.argv[1] + '.ini' if len(sys.argv) > 1 else experiment_alias + '.ini'

try:
    os.mkdir(experiment_alias)
except FileExistsError:
    if new_experiment==False:
        raise FileExistsError(f"Directory '{experiment_alias}' already exists. Please choose a different experiment alias.")
    else:
        print(f"Directory '{experiment_alias}' already exists. Overwriting...")

with open(config_name, "w") as configfile:
    config.write(configfile)

print(f"Config file {config_name} generated for experiment '{experiment_alias}'")

dest_path = os.path.join(experiment_alias, config_name)
shutil.copy(config_name, dest_path)
print(f"Config file also copied to directory '{experiment_alias}' as '{config_name}'")

