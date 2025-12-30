"""Single run pipeline for DQN quantum state transfer experiments.

Executes multiple independent training runs with the same hyperparameters to
assess statistical variability and reproducibility. Automatically generates
config files for each run, trains the DQN agent, and validates performance.

This pipeline is useful for:
- Evaluating training stability across random initializations
- Computing confidence intervals for hyperparameter configurations
- Generating multiple trained models for ensemble methods

Usage
-----
    python single_run_pipeline.py <base_experiment_name>

Example
-------
    python single_run_pipeline.py n8_test
    
    Creates and runs: n8_test_000.ini, n8_test_001.ini, ..., n8_test_004.ini

Configuration
-------------
    Modify nruns to change the number of independent runs (default: 5).
    Adjust thread settings for cluster/HPC environments as needed.
"""

import configparser
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

from drl_training import qst_training, qst_validation
from RL_brain_pi_deep import DQNPrioritizedReplay
from state_env import State

tf.disable_v2_behavior()

# =========================================================================
# TENSORFLOW THREAD CONFIGURATION FOR HPC/CLUSTER
# =========================================================================

# Limit internal threads for cluster environments (adjust based on SLURM allocation)
os.environ["OMP_NUM_THREADS"] = "8" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# Set thread affinity for optimal performance on cluster nodes
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

# Configure TensorFlow parallelism threads
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================================================================
# PIPELINE CONFIGURATION
# =========================================================================

new_config_file = True
config_file =  sys.argv[1] #input("Enter the config file name (default: config.ini): ") or "config.ini"

nruns = 5

# =========================================================================
# EXECUTE MULTIPLE INDEPENDENT RUNS
# =========================================================================

for run in range(nruns):

    # Generate configuration file for this run
    if new_config_file:
        os.system(f'python3 gen_config_file.py {config_file}_00{run}')

    # Load configuration instance
    config_instance = configparser.ConfigParser()
    config_instance.read(config_file + f'_00{run}.ini')

    dirname = config_instance.get('experiment','experiment_alias')

    # Create experiment directory if it doesn't exist
    os.mkdir(dirname) if not os.path.exists(dirname) else None

    print(f"Using configuration from {config_file}")

    # Train DQN agent and validate performance
    trained_agent = qst_training(config_instance=config_instance, progress_bar=False)
    validate_fidelity = qst_validation(config_instance=config_instance)

    print(f"Run {run}. Validation fidelity: {validate_fidelity}")
