# qst_control

This repository contains implementations of two optimization approaches: a Genetic Algorithm (GA) and a Deep Reinforcement Learning (DRL) agent to solve the Quantum State Transfer problem in spin chains or other equivalent qubit arrays.

## Physical Problem
Quantum State Transfer (QST) [1] is a fundamental task in quantum information processing, involving the faithful transmission of a quantum state across a network or chain of qubits. Efficient protocols to maximize transfer fidelity while minimizing errors and time are crucial.

## Project Structure

- `adapted_zhang_implementation/`: RL-based quantum control implementations, training scripts, configuration generators, and analysis notebooks.
- `genetic_algorithm/`: Genetic algorithm modules and scripts for quantum control optimization.
- `shared/`: Shared modules for actions and metrics used across RL and GA approaches.
- `modelos_exitosos/`: Results and models from successful training runs, organized by experiment.

## Key Files & Directories
- `adapted_zhang_implementation/single_run_pipeline.py`: Pipeline for single RL training runs.
- `adapted_zhang_implementation/drl_training.py`: Main RL training module.
- `adapted_zhang_implementation/optuna_run.py`: Hyperparameter optimization using Optuna.
- `adapted_zhang_implementation/model_analizing.ipynb`: Jupyter notebook for model analysis.
- `genetic_algorithm/dc_ga.py`: Genetic algorithm implementation.
- `genetic_algorithm/generate_ga_run.py`: Script to generate and run GA experiments.
- `shared/actions_module.py`: Action definitions for control tasks.
- `shared/metrics.py`: Metrics for evaluating control performance.
- `requirements.txt`: Python dependencies.

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/sofips/qst_control.git
	cd qst_control
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

## Usage

### RL Training
Run RL training with:
```bash
python adapted_zhang_implementation/single_run_pipeline.py <directory_to_store_results>

### Genetic Algorithm
Run GA optimization with:
```bash
python genetic_algorithm/generate_ga_run.py <directory_to_store_results>
```

### Analysis
Open and run the Jupyter notebooks in `adapted_zhang_implementation/` for model analysis and method comparison.

## Results
Successful models and experiment results are stored in `adapted_zhang_implementation/modelos_exitosos/`, organized by experiment name. Each folder contains:
- Best action sequences
- Fidelity scores
- Training/validation metrics
- Model checkpoints


## License
This project is licensed under the MIT License.

RL implementation was adapted from the one provided by the authors of [3].

## Contact
For questions or collaboration, contact [sofips](https://github.com/sofips).

[1] D.S. Acosta Coden, S.S. Gómez, A. Ferrón, and O. Osenda. Controlled quantum
state transfer in xx spin chains at the quantum speed limit. Physics Letters A,
387:127009, 2021.

[2] Ahmed Fawzy Gad. Pygad: An intuitive genetic algorithm python library, 2023.

[3] Xiao-Ming Zhang, Zi-Wei Cui, Xin Wang, and Man-Hong Yung. Automatic spin chain learning to explore the quantum speed limit. Phys. Rev. A, 97:052333, 2018.


