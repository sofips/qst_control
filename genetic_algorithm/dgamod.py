import numpy as np
import scipy.linalg as la
import csv
from actions import one_field_actions, zhang_actions, refined_cns
from metrics import fidelity

# import matplotlib.pyplot as plt
from scipy.linalg import expm

# from numba import njit
import torch as T

np.complex_ = np.complex128
np.mat = np.asmatrix


def fitness_func_constructor(fid_function, arguments):
    """
    Parameters:
        - fidelity function(can be either fidelity or en_fidelity)
        - arguments: arguments of fidelity functions
    Return:
        - lambda function: the fitness function as required by PyGAD
    """
    fitness = lambda vec: fid_function(vec, *arguments)

    return lambda ga_instance, solution, solution_idx: fitness(solution)


def generation_print(ga):

    solution, solution_fitness, solution_idx = ga.best_solution()

    print("Generation", ga.generations_completed)
    print("Solution: ", solution, "Fitness: ", solution_fitness)


def generation_func(ga, props, tol):
    """
    Function to be ran on every generation of the genetic algorithm.
    Prints relevant information on the best solution,
    and determines whether to stop the algorithm based on fidelity.

    
    Parameters:
        - ga (GeneticAlgorithm): An instance of the genetic algorithm.
        - props (dict): Propagators being used to calculate fidelity
        from action sequence.
        - tol (float): The tolerance level for the fidelity to determine
        if the algorithm should stop.

    Returns:
        str: Returns "stop" if the fidelity of the best solution is greater
        than or equal to (1 - tol).
    """

    solution, solution_fitness, solution_idx = ga.best_solution()

    fid, time = fidelity(solution, props, return_time=True)

    print("Generation", ga.generations_completed)
    print(
        "Solution: ",
        solution,
        "Fidelity: ",
        fid,
        "Time: ",
        time,
        "Fitness: ",
        solution_fitness,
    )

    if fid >= 1 - tol:
        return "stop"


def generation_func_constructor(gen_function, arguments):
    """
    Parameters:
        - generation function
        - arguments: arguments of generation function
    Return:
        - lambda function: the mutation function as required by PyGAD
    """

    on_gen = lambda ga_instance: gen_function(ga_instance, *arguments)

    return lambda ga_instance: on_gen(ga_instance)


def actions_to_file(solution, filename, condition):
    """
    Parameters:
        - solution: best solution obtained
        - filename
        - condition: write or append

    Return:
        - saves best action sequence in file = filename
    """
    with open(filename, condition) as f1:

        writer = csv.writer(f1, delimiter=" ")
        solution = np.asarray(solution)
        for i in range(len(solution)):
            row = [solution[i]]
            writer.writerow(row)

    return True



def calculate_reward(states, tolerance, reward_decay):

    """
    Reward function based on the fidelity of the final state adapted from
    Zhang et al. (2018) parallelized for multiple states representing the evolution
    from the initial state to the final state over discrete time episodes.

    Parameters:
    - states (numpy.ndarray): A 2D numpy array of shape (num_states, chain_length)
    where each row represents a quantum state at a given time step.
    - tolerance (float): A threshold value to determine the reward scaling.
    - reward_decay (float): A decay factor applied to the rewards over time.
    Returns:
    - float: The total fitness value calculated based on the rewards.
    """

    # Compute fidelity for all states
    n = np.shape(states)[1]
    fid = np.abs(states[:, n - 1]) ** 2  # Shape: (num_states,)

    # Compute rewards based on conditions
    rewards = np.zeros_like(fid)
    rewards[fid <= 0.8] = 10 * fid[fid <= 0.8]
    rewards[(fid > 0.8) & (fid <= 1 - tolerance)] = 100 / (
        1 + np.exp(10 * (1 - tolerance - fid[(fid > 0.8)
                                             & (fid <= 1 - tolerance)]))
    )
    rewards[fid > 1 - tolerance] = 2500

    # Compute fitness with decay
    decay_factors = reward_decay ** np.arange(len(fid))
    fitness = np.sum(rewards * decay_factors)

    return fitness


def generate_states(initial_state, action_sequence, props):

    """Generate a matrix where each row is the state at a given step."""
    num_elements = len(initial_state)
    steps = len(action_sequence)
    states = np.zeros((steps + 1, num_elements), dtype=initial_state.dtype)
    states[0] = initial_state  # Set the initial state

    # Sequentially calculate states
    for i in range(1, steps):
        states[i] = refined_cns(states[i - 1], action_sequence[i], props)

    return states


def ipr_based_fitness_gpu(
    action_sequences, props, tolerance, reward_decay, test_normalization=False
):
    device = "cuda"

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props, dtype=T.complex64, device=device, requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length), dtype=T.complex64, device=device
    )
    states[:, 0, 0] = 1.0  # Initial condition

    # Compute states using batched matrix multiplication
    for i in range(0, steps):
        states[:, i + 1, :] = T.bmm(
            props[action_sequences[:, i]], states[:, i, :].unsqueeze(-1)
        ).squeeze(-1)

    # Compute fidelity
    fid = states[:, :, -1].abs() ** 2  # Take absolute squared of last column

    ipr = 1. / T.sum(states.abs()**4, dim=2)
    
    rewards = fid/ipr

    # Apply decay and sum fitness
    decay_factors = reward_decay ** T.arange(steps + 1, device=device).unsqueeze(
        0
    )  # Shape: (1, steps)
    fitness = T.sum(rewards * decay_factors, dim=1)  # Sum over steps

    return fitness.cpu().numpy()  # Convert once at the end

def reward_based_fitness_gpu(
    action_sequences, props, tolerance, reward_decay, test_normalization=False
):
    device = "cuda" if T.cuda.is_available() else "cpu"

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props, dtype=T.complex64, device=device, requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length), dtype=T.complex64, device=device
    )
    states[:, 0, 0] = 1.0  # Initial condition

    # Compute states using batched matrix multiplication
    for i in range(0, steps):
        states[:, i + 1, :] = T.bmm(
            props[action_sequences[:, i]], states[:, i, :].unsqueeze(-1)
        ).squeeze(-1)

    # Compute fidelity
    fid = states[:, :, -1].abs() ** 2  # Take absolute squared of last column

    # Compute rewards in parallel
    rewards = T.zeros_like(fid, device=device)
    rewards[fid <= 0.8] = 10 * fid[fid <= 0.8]

    mask = (fid > 0.8) & (fid <= 1 - tolerance)
    rewards[mask] = 100 / (1 + T.exp(10 * (1 - tolerance - fid[mask])))

    rewards[fid > 1 - tolerance] = 2500

    # Apply decay and sum fitness
    decay_factors = reward_decay ** T.arange(steps + 1, device=device).unsqueeze(
        0
    )  # Shape: (1, steps)
    fitness = T.sum(rewards * decay_factors, dim=1)  # Sum over steps

    return fitness.cpu().numpy()  # Convert once at the end


def loc_based_fitness_gpu(
    action_sequences, dt, props, speed_fraction
):

    device = 'cuda' if T.cuda.is_available() else 'cpu'

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props, dtype=T.complex64, device=device, requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length), dtype=T.complex64, device=device
    )
    states[:, 0, 0] = 1.0  # Initial condition

    # Compute states using batched matrix multiplication
    for i in range(0, steps):
        states[:, i + 1, :] = T.bmm(
            props[action_sequences[:, i]], states[:, i, :].unsqueeze(-1)
        ).squeeze(-1)
    
    # Compute fidelity
    fid = states[:, :, -1].abs() ** 2  # Take absolute squared of last column


    loc_evolution = T.sum(
        T.real(states.abs() ** 2) * T.arange(1, chain_length + 1, device=device),
        dim=2
    )

    max_time = T.argmax(fid, dim=1)
    speed = speed_fraction * 2 * chain_length / (chain_length - 1)


    # Compute rewards in parallel for all sequences and time steps
    time_steps = T.arange(0, steps + 1, device=device).unsqueeze(0)  # Shape: (1, time_steps)
    expected_loc = speed * dt * time_steps  # Expected localization at each time step

    # Compute the reward for all sequences and time steps
    rewards = 1 / T.abs(loc_evolution[:, :steps + 1] - expected_loc) ** 2

    # Compute fitness by summing over time steps, weighted by fidelity
    fitness = T.sum(fid[:, :steps + 1] * rewards, dim=1)
    # fitness = np.max(fidelity_evolution)*(1+fitness-max_time)

    return fitness.cpu().numpy()  # Convert once at the end




def action_selector(actions, n,b):
    if actions == 'oaps':
        acciones = one_field_actions(b, n)
    elif actions == 'zhang':
        acciones = zhang_actions(b, n)
    else:
        raise ValueError("Invalid action set. Choose 'oaps' or 'zhang'.")
    
    return acciones

def fitness_selector(fitness,props,dt,speed_fraction=None,tolerance=None,reward_decay=None):
    
    if fitness == 'reward_based':
        fidelity_args = [props,tolerance, reward_decay]
        fitness_func = fitness_func_constructor(reward_based_fitness_gpu, fidelity_args)
    elif fitness == 'loc_based':
        fidelity_args = [dt,props,speed_fraction]
        fitness_func = fitness_func_constructor(loc_based_fitness_gpu, fidelity_args)
    elif fitness == 'ipr_based':
        fidelity_args = [props,tolerance, reward_decay]
        fitness_func = fitness_func_constructor(ipr_based_fitness_gpu, fidelity_args)
    else:
        raise ValueError("Invalid fitness function. Choose 'reward_based', 'loc_based' or 'ipr_based'.")
    return fitness_func

