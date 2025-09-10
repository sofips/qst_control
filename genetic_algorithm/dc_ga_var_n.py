import numpy as np
import os
import sys
from dgamod import  fitness_func_constructor, generation_func, \
                    generation_func_constructor

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, repo_root)

from shared import action_selector, gen_props, calc_max_fidelity
import pygad
import sys
import time
import os
import configparser
import pandas as pd

"""
Implementation of genetic algorithm to find optimal control sequences for
XX spin chains. Script is designed to run for a variety of chain lengths,
saving the best fidelities obtained for each length in a single file and 
the best action sequences for each length in an individual file.
"""

# access config file
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, str(sys.argv[1]))
print(initfile)
config = configparser.ConfigParser()
config.read(initfile)


# set chain lengths to be optimized
initial_n = config.getint("system_parameters", "initial_n") 
final_n = config.getint("system_parameters", "final_n")
n_step = config.getint("system_parameters", "n_step")

# system parameters (only n independent)
dt = config.getfloat("system_parameters", "dt")
b = config.getfloat("system_parameters", "b")
coupling = config.getfloat("system_parameters", "coupling")
action_set = config.get("system_parameters", "action_set")
speed_fraction = config.getfloat("system_parameters","speed_fraction")

# genetic algorithm parameters (only n independent)
num_generations = config.getint("ga_initialization", "num_generations")
sol_per_pop = config.getint("ga_initialization", "sol_per_pop")
fidelity_tolerance = config.getfloat("ga_initialization", "fidelity_tolerance")
saturation = config.getint("ga_initialization", "saturation")
stop_criteria = ["saturate_" + str(saturation)]

reward_decay = config.getfloat("ga_initialization", "reward_decay")
fitness_function = config.get("ga_initialization", "fitness_function")

# crossover and parent selection (only n independent)
num_parents_mating = config.getint("parent_selection", "num_parents_mating")
parent_selection_type = config.get("parent_selection", "parent_selection_type")
keep_elitism = config.getint("parent_selection", "keep_elitism")
crossover_type = config.get("crossover", "crossover_type")
crossover_probability = config.getfloat("crossover", "crossover_probability")

# mutation probability (only n independent)
mutation_probability = config.getfloat("mutation", "mutation_probability")

# saving data details
main_dirname = config.get("saving", "directory")
n_samples = config.getint("saving", "n_samples")


# Create a pandas DataFrame to store the collected data
columns = ["n",
           "sample_index",
           "max_fidelity",
           "ttime", "cpu_time",
           "generations"]

df = pd.DataFrame(columns=columns)


for n in range(initial_n, final_n + 1, n_step):

    # create config instance for each n
    import copy
    nconfig = copy.deepcopy(config)
    nconfig.set("system_parameters", "n", str(n))
    
    # generates actions and associated propagators
    actions = action_selector(action_set, n, b, coupling)
    props = gen_props(actions, dt)

    # set ga parameters that are proportional to chain length
    num_genes_expr = config.get("ga_initialization", "num_genes")
    num_genes = int(eval(num_genes_expr, {"n": n}))
    print(num_genes_expr, "->", num_genes)
    nconfig.set("ga_initialization", "num_genes", str(num_genes))
    print('num_genes: ', num_genes)
    # mutation
    mutation_num_genes_exp = config.get("mutation", "mutation_num_genes")
    mutation_num_genes = int(eval(mutation_num_genes_exp, {"n": n}))
    nconfig.set("mutation", "mutation_num_genes", str(mutation_num_genes))

    gene_space = np.arange(0, len(actions), 1)
    gene_type = int

    # call construction functions
    on_generation = generation_func_constructor(
        generation_func, [props, fidelity_tolerance]
    )

    fitness_func = fitness_func_constructor(
        fitness_function, props, tolerance=fidelity_tolerance, gamma=reward_decay, dt = dt, speed_fraction = speed_fraction)
    mutation_type = "swap"

    config_filename = f"{main_dirname}/n{n}_config.ini"

    with open(config_filename, "w") as configfile:
        nconfig.write(configfile)

    # store found solutions in a list
    nsolutions = []

    for i in range(n_samples):


        t1 = time.time()

        initial_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            parent_selection_type=parent_selection_type,
            keep_elitism=keep_elitism,
            gene_space=gene_space,
            gene_type=gene_type,
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            mutation_type=mutation_type,
            on_generation=on_generation,
            mutation_num_genes=mutation_num_genes,
            stop_criteria=stop_criteria,
            save_solutions=False,
            fitness_batch_size=sol_per_pop
        )

        initial_instance.run()

        t2 = time.time()
        trun = t2 - t1

        maxg = initial_instance.generations_completed

        solution, sol_fitness, sol_idx = initial_instance.best_solution()

        max_fidelity, time_max = calc_max_fidelity(solution, props, return_time=True)
        nsolutions.append(solution)
        
        row = {
            "n": n,
            "sample_index": i,
            "max_fidelity": format(max_fidelity),
            "ttime": "{:.8f}".format(time_max),
            "generations": maxg,
            "cpu_time": "{:.8f}".format(trun),
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    np.savetxt(main_dirname + "/n{}.txt".format(n),
               np.array(nsolutions, dtype=object),
               fmt="%s")

filename = main_dirname + "/results.dat"
df.to_csv(filename, index=False)
