import numpy as np
from actions import refined_cns
from metrics import max_fidelity
import torch as T

np.complex_ = np.complex128
np.mat = np.asmatrix


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

    fid, time = max_fidelity(solution, props, return_time=True)

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
    Generates a function that can be used as the on_generation
    PyGAD callback function.

    Parameters:
        - generation function
        - arguments: arguments of generation function
    Return:
        - lambda function: the mutation function as required by PyGAD
    """

    def on_gen(ga_instance):
        return gen_function(ga_instance, *arguments)

    return on_gen


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


def reward_based_fitness_gpu(action_sequences, props, tolerance, gamma):

    device = "cuda" if T.cuda.is_available() else "cpu"

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props, dtype=T.complex64,
                     device=device,
                     requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length),
        dtype=T.complex64,
        device=device
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
    decay_factors = gamma ** T.arange(steps + 1, device=device).unsqueeze(
        0
    )  # Shape: (1, steps)
    fitness = T.sum(rewards * decay_factors, dim=1)  # Sum over steps

    return fitness.cpu().numpy()  # Convert once at the end


def fidelity_fitness_gpu(action_sequences, props, tolerance, gamma):

    device = "cuda" if T.cuda.is_available() else "cpu"

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props,
                     dtype=T.complex64,
                     device=device,
                     requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length),
        dtype=T.complex64,
        device=device
    )
    states[:, 0, 0] = 1.0  # Initial condition

    # Compute states using batched matrix multiplication
    for i in range(0, steps):
        states[:, i + 1, :] = T.bmm(
            props[action_sequences[:, i]], states[:, i, :].unsqueeze(-1)
        ).squeeze(-1)

    # Compute fidelity
    fid = states[:, :, -1].abs() ** 2  # Take absolute squared of last column
    max_fid = T.max(fid, dim=1)[0]  # Max fidelity for each action sequence
    max_fid_times = T.argmax(fid, dim=1) / chain_length

    fitness = max_fid * (1 + max_fid_times)
    return fitness.cpu().numpy()  # Convert once at the end


def loc_based_fitness_gpu(
    action_sequences, dt, props, speed_fraction
):

    device = 'cuda' if T.cuda.is_available() else 'cpu'

    # Convert props to a CUDA tensor once (complex64 is faster)
    props = T.tensor(props,
                     dtype=T.complex64,
                     device=device,
                     requires_grad=False)

    # Convert action sequences to a tensor
    action_sequences = T.tensor(action_sequences, dtype=T.int64, device=device)

    num_sequences, steps = action_sequences.shape
    chain_length = props.shape[1]

    # Initialize states tensor (batch dimension added)
    states = T.zeros(
        (num_sequences, steps + 1, chain_length),
        dtype=T.complex64, 
        device=device
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
        T.real(states.abs() ** 2) * T.arange(1, chain_length + 1,
                                             device=device),
        dim=2
    )

    speed = speed_fraction * 2 * chain_length / (chain_length - 1)

    # Compute rewards in parallel for all sequences and time steps
    time_steps = T.arange(0, steps + 1, device=device).unsqueeze(0)
    expected_loc = speed * dt * time_steps  # Expected localization

    # Compute the reward for all sequences and time steps
    rewards = 1 / T.abs(loc_evolution[:, :steps + 1] - expected_loc) ** 2

    # Compute fitness by summing over time steps, weighted by fidelity
    fitness = T.sum(fid[:, :steps + 1] * rewards, dim=1)
    # fitness = np.max(fidelity_evolution)*(1+fitness-max_time)

    return fitness.cpu().numpy()  # Convert once at the end


def fitness_func_constructor(fitness_str, props, tolerance, gamma):

    """
    Used to correctly assign the fitness function to be used in PyGAD
    with the corresponding arguments.

    Parameters:
        - fitness (str): The type of fitness function to use ('reward_based',
        'fidelity_based', or 'site_based').
        - props (numpy.ndarray): The propagators associated with the actions.
        - tolerance (float, optional): The tolerance level for reward-based
        fitness functions.
        - gamma (float, optional): The decay factor for reward-based
        fitness functions.

    Returns:
        - function: A fitness function that can be used by PyGAD.
    """
    if fitness_str == 'reward_based':
        fid_args = [props, tolerance, gamma]
        fitness_func = fitness_func_constructor(reward_based_fitness_gpu,
                                                fid_args)
    elif fitness_str == 'fid_based':
        fid_args = [props]
        fitness_func = fitness_func_constructor(fidelity_fitness_gpu,
                                                fid_args)
    elif fitness_str == 'loc_based':
        fid_args = [props]
        fitness_func = fitness_func_constructor(loc_based_fitness_gpu,
                                                fid_args)
    else:
        raise ValueError("Invalid fitness function. Choose 'reward_based', "
                         "'loc_based' or 'fidelity'")

    def fitness(vec):
        return fitness_func(vec, *fid_args)

    def fitness_wrapper(ga_instance, solution, solution_idx):
        return fitness(solution)

    return fitness_wrapper
