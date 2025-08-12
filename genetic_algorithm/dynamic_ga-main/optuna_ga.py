import numpy as np
from dgamod import *
import csv
import pygad
import sys
import time
import os
import configparser
import optuna
import pickle
from datetime import datetime
import pandas as pd

def objective(trial,dirname):
    
    global ti
    ti += 1
    print("Trial number: ", ti)


    # get parameters from config file
    thisfolder = os.path.dirname(os.path.abspath(__file__))
    initfile = os.path.join(thisfolder, str(sys.argv[1]))
    print(initfile)
    config = configparser.ConfigParser()
    config.read(initfile)

    # system parameters
    n = config.getint("system_parameters", "n")

    dt = trial.suggest_float("dt", 0.1, 1.0)
    b = trial.suggest_float("b", 10, 100)

    num_genes = int(0.75*n/dt)
    print("n: ", n, "num_genes: ", num_genes)

    print("dt: ", dt, "b: ", b)
    
    speed_fraction = config.getfloat("system_parameters", "speed_fraction")
    
    fitness_function = config.get("ga_initialization", "fitness_function")
    action_set = config.get("ga_initialization", "action_set")
    
    # generates actions and associated propagators
    actions = action_selector(
        action_set, n, b)
    props = gen_props(actions, n, dt)

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

    # mutation
    mutation_probability = config.getfloat("mutation", "mutation_probability")
    mutation_num_genes = config.getint("mutation", "mutation_num_genes")

    # saving data details
    dirname = config.get("saving", "directory")
    n_samples = config.getint("saving", "n_samples")
    filename = dirname + "/nvsmaxfid.dat"

    gene_space = np.arange(0, len(actions), 1)
    gene_type = int

    stop_criteria = ["saturate_" + str(saturation)]

    # call construction functions
    on_generation = generation_func_constructor(
        generation_func, [props, fidelity_tolerance]
    )


    fitness_func = fitness_selector(
        fitness_function, props,dt,
        speed_fraction=speed_fraction, 
        tolerance=fidelity_tolerance,
        reward_decay=reward_decay)
    
    mutation_type = "swap"

    max_fidelity = 0
    
    trial_directory = dirname + "/trial_" + str(ti)

    # Create the directory if it doesn't exist
    if not os.path.exists(trial_directory):
        os.makedirs(trial_directory)
    
    # Create a pandas DataFrame from the collected data
    columns = ["n", "sample_index", "max_fidelity", "ttime", "cpu_time", "generations"]
    df = pd.DataFrame(columns=columns)
        
    for i in range(n_samples):
        solutions_fname = "{}/act_sequence_n{}_sample{}.dat".format(
            trial_directory, n, i
        )
        fitness_history_fname = trial_directory + "/fitness_history_sample" + str(i) + ".dat"

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
            fitness_batch_size=sol_per_pop,
        )

        initial_instance.run()

        t2 = time.time()
        trun = t2 - t1

        maxg = initial_instance.generations_completed

        solution, solution_fitness, solution_idx = initial_instance.best_solution()

        evolution = time_evolution(solution, props, n, graph=False, filename=False)
        time_max_fidelity = np.argmax(evolution) * dt

        max_fidelity = max(max_fidelity, fidelity(solution, props))

        # Report intermediate results to Optuna
        trial.report(solution_fitness, i)

        # Check if the trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        actions_to_file(solution, solutions_fname, "w")
        
        row = {
            "n": n,
            "trial_index": ti,
            "sample_index": i,
            "max_fidelity": format(fidelity(solution, props)),
            "ttime": "{:.8f}".format(time_max_fidelity),
            "generations": maxg,
            "cpu_time": "{:.8f}".format(trun),
            "max_fitness": "{:.8f}".format(solution_fitness),
            "dt": dt,
            "b": b,
        }
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(trial_directory, f"trial_{ti}_data.csv"), index=False)
    
    # Calculate mean, std, and max for all metrics
    metrics_summary = df[["max_fidelity", "ttime", "cpu_time", "generations","max_fitness","b","dt"]].astype(float).agg(["mean", "std", "max"])

    # Save the metrics summary to a CSV file
    metrics_summary.to_csv(os.path.join(trial_directory, "metrics_summary.csv"))

    # Access the mean fidelity from the metrics summary
    mean_fitness= metrics_summary.loc["mean", "max_fitness"]
    mean_max_fidelity=metrics_summary.loc["mean", "max_fidelity"]
    print("Mean fitness: ", mean_fitness)
    print("Max fidelity: ", mean_max_fidelity)
    # Save the fitness history to a CSV file
    return mean_fitness*mean_max_fidelity

n_trials = 10

if __name__ == "__main__":
    ti = 0
    
    # get parameters from config file
    thisfolder = os.path.dirname(os.path.abspath(__file__))
    initfile = os.path.join(thisfolder, str(sys.argv[1]))
    print(initfile)
    config = configparser.ConfigParser()
    config.read(initfile)

    dirname = config.get("saving", "directory")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, dirname), n_trials=n_trials)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best trial's metrics and parameters to a pickle file
    best_trial_data = {
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "trial_index": study.best_trial.number,  # Add trial index
    }

    with open(os.path.join(dirname, "best_trial.pkl"), "wb") as f:
        pickle.dump(best_trial_data, f)
