"""
Generates configuration files for different runs of 
genetic algorithm implementation, using configparser
library. Creates directory and saves script in it.
"""

import configparser
import os
import sys

# Create instance of ConfigParser
config = configparser.ConfigParser()

# system parameters

initial_n = 8  # set initial n
final_n = 24   # set final n
n_step = 4
dt = 0.15      # length of temporal steps
b = 100        # magnetic field strength
coupling = 1.0  # coupling strength between nearest neighbors

# genetic algorithm parameters
fitness_function = 'reward_based'  # Options: '{reward,loc,fid}_based'
action_set = 'oaps'  # Options are: 'oaps', 'zhang'

num_generations = 1000
sol_per_pop = 4096
fidelity_tolerance = 0.05
reward_decay = 0.95  # time decay to achieve faster transmission
saturation = 30


# crossover and parent selection
num_parents_mating = sol_per_pop // 10
parent_selection_type = "sss"
keep_elitism = sol_per_pop // 10
crossover_type = "uniform"
crossover_probability = 0.8

# other mutation parameters

mutation_probability = 0.99
mutation_num_genes = 'n'

# execution and results saving
directory = sys.argv[1]
n_samples = 5


config["system_parameters"] = {
    "initial_n": str(initial_n),
    "final_n": str(final_n),
    "n_step": str(n_step),
    "dt": str(dt),
    "b": str(b),
    "coupling": str(coupling),
    "action_set": action_set,
}

config["ga_initialization"] = {
    "num_generations": str(num_generations),
    "num_genes": '5*n', # number of genes in the chromosome, set as 5n
    "sol_per_pop": str(sol_per_pop),
    "fidelity_tolerance": str(fidelity_tolerance),
    "saturation": str(saturation),
    "fitness_function": fitness_function,
    "reward_decay": str(reward_decay),
}


config["parent_selection"] = {
    "num_parents_mating": str(num_parents_mating),
    "parent_selection_type": parent_selection_type,
    "keep_elitism": str(keep_elitism),
}

config["crossover"] = {
    "crossover_type": crossover_type,
    "crossover_probability": str(crossover_probability),
}

config["mutation"] = {
    "mutation_probability": str(mutation_probability),
    "mutation_num_genes": 'n',
}

config["saving"] = {
    "directory": directory,
    "n_samples": str(n_samples),
}
script = "dc_ga.py"

isExist = os.path.exists(directory)
if not isExist:
    os.mkdir(directory)
    # Add directory to .gitignore if not already present
    gitignore_path = os.path.join(os.getcwd(), ".gitignore")
    try:
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r+") as f:
                lines = f.read().splitlines()
                if directory not in lines:
                    f.write(f"\n{directory}\n")
        else:
            with open(gitignore_path, "w") as f:
                f.write(f"{directory}\n")
    except Exception as e:
        print(f"Warning: Could not update .gitignore: {e}")
else:
    print("Warning: Directory already existed")

src = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.join(src, script)
mod_name = os.path.join(src, "dgamod.py")

cmd = f'cp "{script_name}" "{directory}"'
os.system(cmd)
cmd = f'cp "{mod_name}" "{directory}"'
os.system(cmd)
config_name = directory + "/" + directory + ".ini"

with open(config_name, "w") as configfile:
    config.write(configfile)

script_name = directory + "/" + script
config_name = directory + ".ini"

cmd = f'python3 "{script_name}" "{config_name}"'
os.system(cmd)
