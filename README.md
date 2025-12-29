# Quantum State Transfer Control Optimization

A comprehensive toolkit implementing **Genetic Algorithm (GA)** and **Deep Reinforcement Learning (DRL)** approaches for discovering optimal quantum control sequences in spin chains.

## Physical Problem

**Quantum State Transfer (QST)** is the faithful transmission of quantum states across qubit arrays. This is a fundamental task in quantum information processing and quantum computing architectures. The goal is to maximize transfer fidelity while minimizing control time and robustness to noise.

### Key Challenges
- High-dimensional control space (exponential in chain length)
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
  - Automatic hyperparameter tuning via Optuna
  - Model checkpointing (best and final models)
  - Comprehensive metrics tracking

### 2. **Genetic Algorithm (GA)** Approach
- **Framework**: PyGAD evolutionary optimization
- **Location**: `genetic_algorithm/`
- **Key Features**:
  - Population-based search over action sequences
  - Multiple fitness function implementations (reward-based, fidelity-based, GPU-optimized)
  - Scalable to various chain lengths
  - Parallel population evaluation

## Project Structure

```
qst_control/
├── adapted_zhang_implementation/        # DRL-based approach
│   ├── drl_training.py                 # Main DQN training module (PEP8 documented)
│   ├── RL_brain_pi_deep.py             # DQN agent with prioritized replay
│   ├── state_env.py                    # Quantum environment dynamics
│   ├── gen_config_file.py              # Config file generator (for single experiment)
│   ├── single_run_pipeline.py          # Single training experiment pipeline
│   ├── optuna_run.py                   # Hyperparameter optimization
│   ├── optuna_config.py                # Optuna configuration
│   ├── log_dir.py                      # Logging utilities (to use with mlflow)
│   ├── model_analizing.ipynb           # Model evaluation notebook
│   ├── method_comparing.ipynb          # GA vs DRL comparison notebook
│   └── modelos_exitosos/               # Successful trained models
│
├── genetic_algorithm/                  # GA-based approach
│   ├── dc_ga.py                        # Main GA implementation
│   ├── dgamod.py                       # GA utilities & fitness functions
│   ├── generate_ga_run.py              # Experiment generator
│
├── shared/                             # Shared utilities
│   ├── actions_module.py               # Action definitions (Zhang, OAPS)
│   ├── metrics.py                      # Fidelity & performance metrics
│   ├── figures.py                      # Analysis visualization
│   └── __init__.py                     # Package initialization
│
├── main_figures.ipynb                  # Main analysis & results notebook
├── requirements.yml                    # Conda environment file
```

## Quick Start

### DRL Training

Run a single DRL training experiment (configure parameters in gen_config_file):

```bash
python adapted_zhang_implementation/single_run_pipeline.py my_experiment
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

Optimize hyperparameters using Optuna:

```bash
python adapted_zhang_implementation/optuna_run.py --n_trials 100
```

### Genetic Algorithm Training

Run GA optimization:

```bash
python genetic_algorithm/generate_ga_run.py dirname
```

This will generate and run a GA experiment and create a directory ```dirname``` to store its results

## Results & Trained Models


Optuna optimized models are stored in `adapted_zhang_implementation/optimized_models/`:

```
optimized_models/
├── n10_10amp_10prob_zhang/   # Chain length 10, 10% noise, Zhang et.al. action set
├── n32_25amp_25prob_oaps/    # Chain length 32, 25% noise, OAPS action set
└── ...
```

Genetic algorithm results for different chain lengths are stored in `genetic_algorithm/results`

## Analysis & Visualization

### Main Figures Notebook
Generate publication-quality figures:

```bash
jupyter notebook main_figures.ipynb
```

### Model Analysis
Evaluate single trained model:

```bash
jupyter notebook adapted_zhang_implementation/model_analizing.ipynb
```

### Method Comparison
Compare DRL and GA approaches:

```bash
jupyter notebook adapted_zhang_implementation/method_comparing.ipynb
```

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{qst_control,
  title={Quantum State Transfer Control Optimization},
  author={Sofía Perón Santana},
  year={2025},
  url={https://github.com/sofips/qst_control}
}
```

## License

MIT License - See LICENSE file for details

DRL optimization was adapted from the source code provided by the authors of [1]
## Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: sofia.peralta@example.com (update as needed)

---

## References 

[1] Xiao-Ming Zhang, Zi-Wei Cui, Xin Wang, and Man-Hong Yung. Automatic spin chain learning to explore the quantum speed limit. Phys. Rev. A, 97:052333, 2018.

[2] Ahmed Fawzy Gad. Pygad: An intuitive genetic algorithm python library, 2023.
