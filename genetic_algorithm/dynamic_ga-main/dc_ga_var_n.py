import numpy as np
from dgamod import *
import csv
import pygad
import sys
import time
import os
import configparser
import pandas as pd



# get parameters from config file
thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, str(sys.argv[1]))
print(initfile)
config = configparser.ConfigParser()
config.read(initfile)

# system parameters
initial_n = config.getint("system_parameters", "initial_n")
final_n = config.getint("system_parameters", "final_n")
n_step = config.getint("system_parameters", "n_step")
dt = config.getfloat("system_parameters", "dt")
b = config.getfloat("system_parameters", "b")
speed_fraction = config.getfloat(
 "system_parameters", "speed_fraction"
 )  # fraction of qsl speed if loc based fitness
#max_optimization_time = config.getint("system_parameters", "max_optimization_time")

# genetic algorithm parameters
num_generations = config.getint("ga_initialization", "num_generations")
sol_per_pop = config.getint("ga_initialization", "sol_per_pop")
fidelity_tolerance = config.getfloat("ga_initialization", "fidelity_tolerance")
saturation = config.getint("ga_initialization", "saturation")
reward_decay = config.getfloat("ga_initialization", "reward_decay")

# crossover and parent selection
num_parents_mating = config.getint("parent_selection", "num_parents_mating")
parent_selection_type = config.get("parent_selection", "parent_selection_type")
keep_elitism = config.getint("parent_selection", "keep_elitism")
crossover_type = config.get("crossover", "crossover_type")
crossover_probability = config.getfloat("crossover", "crossover_probability")

# mutation probability
mutation_probability = config.getfloat("mutation", "mutation_probability")


# saving data details
dirname = config.get("saving", "directory")
n_samples = config.getint("saving", "n_samples")
filename = dirname + "/nvsmaxfid.dat"

stop_criteria = ["saturate_" + str(saturation)]  # , 'reach_'+str(fidelity_tolerance)]

# Create a pandas DataFrame from the collected data
columns = ["n", "sample_index", "max_fidelity", "ttime", "cpu_time", "generations"]
df = pd.DataFrame(columns=columns)

fitness_function = config.get("ga_initialization", "fitness_function")
action_set = config.get("ga_initialization", "action_set")

for n in range(initial_n, final_n + 1, n_step):
    
    config.set("system_parameters", "n", str(n))

    # create a directory for each n
    ndirname = config.get("saving", "directory") + "/n" + str(n)
    isExist = os.path.exists(ndirname)
    
    if not isExist:
        os.mkdir(ndirname)
    else:
        print("Warning: Directory already existed")

    # generates actions and associated propagators
    acciones = action_selector(
        action_set, n, b)
    props = gen_props(acciones, n, dt)

    # set ga parameters that are proportional to chain length
    num_genes = 5*n
    
    config.set("ga_initialization", "num_genes", str(num_genes))

    # mutation
    mutation_num_genes = n
    config.set("mutation", "mutation_num_genes", str(mutation_num_genes))

    gene_space = np.arange(0, len(acciones), 1)
    gene_type = int

    # call construction functions
    on_generation = generation_func_constructor(
        generation_func, [props, fidelity_tolerance]
    )

    fidelity_args = [
        props,
        fidelity_tolerance,
        reward_decay
    ] 

    fitness_func = fitness_selector(
        fitness_function, props,dt, speed_fraction=speed_fraction, tolerance=fidelity_tolerance, reward_decay=reward_decay)
    mutation_type = "swap"

    config_filename = ndirname + "/config.ini"
    
    with open(config_filename, "w") as configfile:
        config.write(configfile)
    
    for i in range(n_samples):

        solutions_fname = "{}/act_sequence_n{}_sample{}.dat".format(ndirname, n, i)
        fitness_history_fname = ndirname + '/fitness_history_sample'+ str(i) + '.dat'

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

        solution, solution_fitness, solution_idx = initial_instance.best_solution()

        evolution = time_evolution(solution, props, n, graph=False, filename=False)
        time_max_fidelity = np.argmax(evolution) * dt

        row = {
            "n": n,
            "sample_index": i,
            "max_fidelity": format(fidelity(solution, props)),
            "ttime": "{:.8f}".format(time_max_fidelity),
            "generations": maxg,
            "cpu_time": "{:.8f}".format(trun),
        }
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        
        actions_to_file(solution, solutions_fname, "w")

df.to_csv(filename, index=False)