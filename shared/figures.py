import configparser
import math
import pandas as pd
from shared.metrics import fidelity_evolution
import os
import numpy as np

def print_parameters(config_file, ncols=3, col_width=30):
    config = configparser.ConfigParser()
    config.read(config_file)

    directory = config.get('saving','directory')
    nsamples = config.getint('saving','n_samples')
    print('# ------------------------------------------------------------')
    print(f"Printing parameters from {directory} \nNumber of samples: {nsamples}")
    print('# ------------------------------------------------------------')

    for section in config.sections():
        if section == 'saving':
            continue
        print(f"\n[{section}]")

        items = [f"{k} = {v}" for k, v in config.items(section)]
        nrows = math.ceil(len(items) / ncols)

        for r in range(nrows):
            row = items[r::nrows]
            print("".join(item.ljust(col_width) for item in row))
    print('# ------------------------------------------------------------')
    print('\n')


def extract_results(directory,
                    print_params=True,
                    filename='results.dat',
                    header=None,
                    delimiter=',',
                    skiprows=1,
                    columns=['chain_length',
                                'sample',
                                'fid',
                                'ttime',
                                'generations',
                                'cputime']):

    results = pd.read_csv(directory + filename,
                        header=header,
                        delimiter=delimiter,
                        skiprows=skiprows,
                        names=columns)
    if print_params:
        config_file = directory + directory.split('/')[-2] + '.ini'
        try:
            print_parameters(config_file, ncols=3, col_width=30)
        except FileNotFoundError:
            print(f"Could not find config file in {config_file}")
    return results


def access_evolutions(directory, chain_length, return_natural_evolution=False):

    config_file = directory + f'n{chain_length}_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    solutions_file = f'n{chain_length}.txt'
    solutions_path = os.path.join(directory, solutions_file)
    solutions = np.genfromtxt(solutions_path, delimiter=' ', dtype=int)

    evolutions = []  # <-- store here

    for row in range(solutions.shape[0]):
        action_sequence = solutions[row, :]

        forced_evolution, natural_evolution = fidelity_evolution(
            action_sequence,
            config_file,
            add_natural=True
        )

        evolutions.append(forced_evolution)

    if return_natural_evolution:
        return np.array(evolutions), natural_evolution
    else:
        return np.array(evolutions)
