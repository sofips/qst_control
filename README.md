# Quantum State Transfer Control Optimization
<a href="https://doi.org/10.5281/zenodo.18091211"><img src="https://zenodo.org/badge/1036925825.svg" alt="DOI"></a>

A comprehensive toolkit implementing **Genetic Algorithm (GA)** and **Deep Reinforcement Learning (DRL)** approaches for discovering optimal quantum control sequences in spin chains.



## Physical Problem

**Quantum State Transfer (QST)** is the faithful transmission of quantum states across qubit arrays. This is a fundamental task in quantum information processing and quantum computing architectures. The goal is to maximize transfer fidelity while minimizing control time and robustness to noise.

### Key Challenges
- High-dimensional control space
- Complex quantum dynamics and state evolution
- Trade-off between fidelity, control duration, and robustness
- Need for scalable optimization methods

## Project Overview

This repository contains two complementary approaches to solve quantum control optimization:

### 1. **Deep Reinforcement Learning (DRL)** Approach
- **Agent**: Deep Q-Network with prioritized experience replay
- **Framework**: TensorFlow with Optuna for hyperparameter optimization
- **Location**: `adapted_zhang_implementation/`
- **Key Features**:
  - Episodic training with noise injection for robustness
  - Prioritized sampling for efficient learning
  - Hyperparameter tuning via Optuna
  - Comprehensive metrics tracking

### 2. **Genetic Algorithm (GA)** Approach
- **Framework**: PyGAD evolutionary optimization
- **Location**: `genetic_algorithm/`
- **Key Features**:
  - Population-based search over action sequences
  - Multiple fitness function implementations (reward-based, fidelity-based, GPU-optimized)
  - Scalable to various chain lengths
  - Parallel population evaluation using PyTorch

## Project Structure

```
qst_control/
├── adapted_zhang_implementation/       # DRL-based approach
|   ├── optimized_models                # Results and models with optimized hyper-parameters
|   ├── original_drl_results            # Results and models with original parameters from ref. [1]
│   ├── drl_training.py                 # Main DQN training module
│   ├── gen_config_file.py              # Config file generator (for single experiment)
│   ├── log_dir.py                      # Logging utilities for single experiment using mlflow
│   ├── method_comparing.ipynb          # GA vs DRL robustness comparison notebook
│   ├── optuna_config.py                # Optuna configuration (called by optuna_run.py)
│   ├── optuna_log.py                   # Log results of optuna optimization using mlflow
│   ├── optuna_run.py                   # Hyperparameter optimization using optuna
│   ├── RL_brain_pi_deep.py             # DQN agent with prioritized replay
│   ├── single_run_pipeline.py          # Single training experiment pipeline
│   └── state_env.py                    # Quantum environment dynamics
│  
│
├── genetic_algorithm/                  # GA-based approach
│   ├── action_stats/                   # Experiments with a large number of samples to perform statistics over actions
│   ├── results/                        # Main results for different chain lengths
│   ├── dc_ga.py                        # Main GA implementation
│   ├── dgamod.py                       # GA utilities & fitness functions
│   └── generate_ga_run.py              # Experiment generator, creates config file and calls dc_ga.py
│
├── shared/                             # Shared utilities
│   ├── actions_module.py               # Action definitions (Zhang, OAPS)
│   ├── metrics.py                      # Fidelity & performance metrics
│   ├── figures.py                      # Analysis visualization
│   └── __init__.py                     # Package initialization
|
|
├── figures/                            # Resulting figures
├── main_figures.ipynb                  # Main analysis & results notebook
├── requirements                        # Main analysis & results notebook
└── log_env.yml                         # Conda environment file for mlflow
```

## Quick Start

### DRL Training

Run a single DRL training experiment (configure parameters in gen_config_file):

```bash
cd adapted_zhang_implementation
python3 single_run_pipeline.py my_experiment
```

This will:
- Create an experiment directory `my_experiment/`
- Train a DQN agent for quantum state transfer
- Save best and final models
- Generate training metrics and action sequences

**Key outputs:**
- `best_model/model.ckpt` - Best checkpoint
- `training_results.txt` - Per-episode metrics (CSV)
- `best_action_sequences.txt` - Top 10 control sequences
- `training_metrics.pkl` - Aggregated statistics

### DRL Hyperparameter Optimization

Optimize hyperparameters using Optuna (configure optimization in optuna_config.py):

```bash
cd adapted_zhang_implementation
python adapted_zhang_implementation/optuna_run.py optuna_experiment_name
```

### Genetic Algorithm Training

Run GA optimization:

```bash
cd genetic_algorithm
python genetic_algorithm/generate_ga_run.py dirname
```

This will generate and run a GA experiment and create a directory ```dirname``` to store its results


## Analysis & Visualization

### Main Figures Notebook
Generate publication-quality figures:

```bash
jupyter notebook main_figures.ipynb
```


## MLflow Experiment Tracking

### Post-Training Logging Workflow

This project uses **MLflow** for experiment tracking and comparison. To avoid complications with logging from HPC clusters (network restrictions, authentication issues), experiments are logged **after training completion** using dedicated logging scripts. This needs a different conda environment with requirements stated in log_env.yml.

#### Workflow:
1. **Train on HPC/Local**: Run training experiments (DRL or Optuna optimization) which save results to disk
2. **Transfer Results**: Copy experiment directories to a machine with MLflow server access
3. **Log to MLflow**: Use logging scripts to upload results to MLflow tracking server

#### MLflow Server Setup

Start the MLflow tracking server locally:
```bash
mlflow server --host 127.0.0.1 --port 5005
```

Access the UI at `http://localhost:5005`

#### Logging Single DRL Experiments

After training with `single_run_pipeline.py` or individual runs, log results:

```bash
python log_dir.py <experiment_directory>
```

**Example:**
```bash
python log_dir.py n8_25amp_25prob_oaps
```

This logs:
- All model checkpoints and artifacts
- Training hyperparameters and system parameters
- Binned episode metrics (every 100 episodes for efficient storage)
- Aggregated training statistics (success rates, average fidelity)
- Validation metrics (if validation was performed)

#### Logging Optuna Optimization Studies

After Optuna hyperparameter optimization, log all trials:

```bash
python optuna_log.py <optimization_directory>
```

**Example:**
```bash
python optuna_log.py opt_for_n16
```

This creates:
- Parent experiment for the optimization study
- Nested runs for each Optuna trial
- Comparative metrics across all trials for identifying best hyperparameters

#### Benefits of Post-Training Logging
- **HPC Compatibility**: No network/firewall issues during cluster training
- **Batch Processing**: Log multiple experiments at once after completion
- **Flexibility**: Choose when and where to log without interrupting training
- **Reliability**: Training continues even if logging fails

## License

MIT License - See LICENSE file for details

DRL optimization was adapted from the source code provided by the authors of [1]

## Citation

To cite this repo:

@software{sofia_peron_santana_2025_18091212,
  author       = {Sofía Perón Santana},
  title        = {sofips/qst\_control: qst\_control-v1},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {qst-control-v1},
  doi          = {10.5281/zenodo.18091212},
  url          = {https://doi.org/10.5281/zenodo.18091212},
}

Sofía Perón Santana. (2025). sofips/qst_control: qst_control-v1 (qst-control-v1). Zenodo. https://doi.org/10.5281/zenodo.18091212

## Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: sofia.peron@mi.unc.edu.ar (update as needed)

---

## References 

[1] Xiao-Ming Zhang, Zi-Wei Cui, Xin Wang, and Man-Hong Yung. Automatic spin chain learning to explore the quantum speed limit. Phys. Rev. A, 97:052333, 2018.

[2] Ahmed Fawzy Gad. Pygad: An intuitive genetic algorithm python library, 2023.
