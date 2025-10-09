import numpy as np


def calc_max_fidelity(action_sequence, props, return_time=False):

    """
    Calculate the fidelity resulting of a given pulse sequence. The state is
    initialized to /10...0>

    Parameters:
    - action_sequence (list or array-like): A sequence of actions to be applied
    to the initial state.
    - props (ndarray): A 3D array where props[action] is the propagation matrix
    corresponding to that action.
    - return_time (bool, optional): If True, return the time step at which the
    maximum fidelity is achieved. Default is False.

    Returns:
    - float: The maximum fidelity achieved.
    - tuple: If return_time is True, returns a tuple (max_fid, imax) 
    where max_fid is the maximum fidelity and imax is the time step at which 
    it is achieved.
    """

    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.0
    max_fid = 0.0
    imax = 0
    i = 0
    for action in action_sequence:
        i += 1
        state = np.matmul(props[action], state)
        fid = np.real(state[n - 1] * np.conjugate(state[n - 1]))

        if fid > max_fid:
            imax = i
            max_fid = fid

    if return_time:
        return max_fid, imax

    return max_fid

def state_fidelity(state):

    return np.asarray((abs(state[-1]) ** 2)[0, 0])  # calculate fidelity


def calc_ipr(state):
    """
    Calculate the Inverse Participation Ratio (IPR) of a quantum state. The IPR
    is a measure of the localization of the state in the Hilbert space.
    Parameters:
    - state (numpy.ndarray): A 1D numpy array representing the quantum state.
    Returns:
    - float: The IPR value, which ranges from 1 (fully localized) to the
    dimension of the Hilbert space (fully delocalized).
    """
    nh = np.shape(state)[0]
    ipr = 0

    for i in range(nh):
        ipr += np.real(state[i] * np.conjugate(state[i])) ** 2

    return 1 / ipr


def mean_site(state):
    
    """
    Calculates the mean site index for a given quantum state vector.
    The mean site is computed as the weighted sum of the probability amplitudes
    at each site, where the weight is the (1-based) site index.
    Parameters
    - state (numpy.ndarray): A 1D numpy array representing the quantum state.
    Returns:
    - mean site (float): The mean site value, calculated as the sum over all
    sites of the probability at each site multiplied by its (1-based) index.
    
    Notes
    -----
    The function assumes that `state` is a 1D array-like object containing
    complex numbers.
    """

    chain_length = np.shape(state)[0]

    ms = np.sum(
        np.asarray([
            np.real(state[j] * np.conjugate(state[j])) * (j + 1)
            for j in range(0, chain_length)
        ])
    )
    return ms

