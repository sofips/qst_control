from state_env import State  # module with environment and dynamics
from RL_brain_pi_deep import DQNPrioritizedReplay  # sumtree version of DQN
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()
from drl_training import *
import os
import sys


# Limitar los hilos internos
os.environ["OMP_NUM_THREADS"] = "8"  # o los cores que SLURM te da
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# (opcional) fijar afinidad si el cluster lo recomienda
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

# Comprobá la configuración
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(1)


new_config_file = True
config_file =  sys.argv[1] #input("Enter the config file name (default: config.ini): ") or "config.ini"

nruns = 5

for run in range(nruns):

    if new_config_file:
        os.system(f'python3 gen_config_file.py {config_file}_00{run}')

    config_instance = configparser.ConfigParser()
    config_instance.read(config_file + f'_00{run}.ini')

    dirname = config_instance.get('experiment','experiment_alias')

    os.mkdir(dirname) if not os.path.exists(dirname) else None

    print(f"Using configuration from {config_file}")

    trained_agent = qst_training(config_instance=config_instance, progress_bar=False)
    validate_fidelity = qst_validation(config_instance=config_instance)

    print(f"Run {run}. Validation fidelity: {validate_fidelity}")
