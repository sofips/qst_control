import numpy as np
from scipy.linalg import expm


def zhang_actions(field_strength, chain_length, coupling):
    """
    Adapted from Zhang et. al work. Generates a set of action matrices
    corresponding to the actions described in the paper. 

    Parameters:
    field_strength (float): The maximum value of the field
    chain_length (int): Chain length (no. of sites)
    coupling (float): The coupling strength between nearest neighbors.

    Returns:
    action_hamiltonians(numpy.ndarray): A 3D numpy array of shape (16,
    chain_length, chain_length) representing the action matrices.
    """
    nc = 3  # number of control sites, nc=3,there are totally 16 actions.
    # defining action, each row of 'mag' corresponds to one configuration

    def binact(A):  # action label
        m = np.zeros(nc)
        for ii in range(
            nc
        ):  # transfer action to a binary list, for example: action=5,
            # x=[1, 0, 1, 0], the first and third magnetic is on

            m[nc - 1 - ii] = A >= 2 ** (nc - 1 - ii)
            A = A - 2 ** (nc - 1 - ii) * m[nc - 1 - ii]
        return m

    mag = []
    for ii in range(8):  # control at the beginning
        mag.append(list(np.concatenate((binact(ii) * field_strength,
                                        np.zeros(chain_length - nc)))))

    for ii in range(1, 8):  # control at the end
        mag.append(list(np.concatenate((np.zeros(chain_length - nc),
                                        binact(ii) * field_strength))))

    mag.append([field_strength for ii in range(chain_length)])

    action_hamiltonians = np.zeros((16, chain_length, chain_length), dtype=complex)

    for idx, actions in enumerate(mag):
        diag_coupling = [coupling for _ in range(chain_length - 1)]
        ham = (
            np.diag(diag_coupling, 1) * (1 - 0j)
            + np.diag(diag_coupling, -1) * (1 + 0j)
            + np.diag(actions)
        )
        action_hamiltonians[idx] = ham
    return action_hamiltonians


def one_field_actions(field_strength, chain_length, coupling):
    """
    Generates a set of action matrices corresponding to fields acting 
    on every individual site.

    Parameters:
    field_strength (float): The maximum value of the field,
    chain_length (int): Chain length (no. of sites)
    coupling (float): The coupling strength between nearest neighbors.

    Returns:
    action_hamiltonians(numpy.ndarray): A 3D numpy array of shape:
    (chain_length + 1, chain_length, chain_length) representing the actions.
    """

    action_hamiltonians = np.zeros((chain_length + 1,
                                    chain_length,
                                    chain_length), dtype=complex)

    for i in range(0, chain_length):

        for k in range(0, chain_length - 1):
            action_hamiltonians[i + 1, k, k + 1] = coupling
            action_hamiltonians[i + 1, k + 1, k] = action_hamiltonians[i + 1,
                                                                       k,
                                                                       k + 1]

        action_hamiltonians[i + 1, i, i] = field_strength

    for k in range(0, chain_length - 1):
        action_hamiltonians[0, k, k + 1] = coupling
        action_hamiltonians[0, k + 1, k] = action_hamiltonians[0, k, k + 1]

    return action_hamiltonians


def gen_props(action_hamiltonians, dt):

    """
    Generates the propagation matrices for each action.
    Parameters:
    - action_hamiltonians (numpy.ndarray): An array of shape (n_actions, n, n)
    representing the action Hamiltonians.
    - dt (float): Time step for the evolution.

    Returns:
    - propagators (list): A list of 2D numpy arrays, each representing the
    propagation matrix for the corresponding action.
    """

    propagators = [expm(-1j * action_hamiltonians[i] * dt)
                   for i in range(action_hamiltonians.shape[0])]

    return propagators


def refined_cns(state, action_index, props):

    """
    Applies the action corresponding to action_index on the given state
    using the precomputed propagation matrices (props).

    Parameters:
    - state (numpy.ndarray): The current state vector.
    - action_index (int): The index of the action to be applied.
    - props (numpy.ndarray): A 3D array where props[action] is the propagation
    matrix corresponding to that action.

    Returns:
    - next_state (numpy.ndarray): The state vector after applying the action.
    """

    # Retrieve the matrix corresponding to the action index
    p = props[action_index]

    # Perform matrix-vector multiplication directly
    next_state = p @ state

    # Return the result as a flat 1D array
    return next_state.ravel()


def action_selector(actions, n, b, coupling):
    """"Takes string determining action type and outputs the corresponding 
    matrices for desired system
    
    Parameters:
    - actions (str): The type of actions to generate ('oaps' or 'zhang').
    - n (int): The number of sites in the chain.
    - b (float): The strength of the external field.
    - coupling (float): The coupling strength between nearest neighbors.

    Returns:
    - action_hamiltonians (numpy.ndarray): A 3D array of shape (n_actions, n, n)
    representing the action Hamiltonians.
    """
    if actions == 'oaps':
        action_hamiltonians = one_field_actions(b, n, coupling)
    elif actions == 'zhang':
        action_hamiltonians = zhang_actions(b, n, coupling)
    else:
        raise ValueError("Invalid action set. Choose 'oaps' or 'zhang'.")

    return action_hamiltonians
