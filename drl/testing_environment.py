import numpy as np
from quantum_environment import QuantumStateEnv
from stable_baselines3.common.env_checker import check_env
import configparser
import scipy.linalg as la


def propagators_check(n, n_actions, dt, propagators, bases, energies):
    """
    Check the correctness of state propagation using propagators on the
    eigenvectors of each matrix.

    Args:
        n (int): Number of bases.
        n_actions (int): Number of actions.
        dt (float): Time step.
        propagators (ndarray): List of (n, n) arrays representing
        the propagators associated to each action
        bases (ndarray): Array of shape (n_actions, n, n)
        representing the eigenvectors.
        energies (ndarray): Array of shape (n_actions, n) '
        representing the energies / eigenvalues.

    Returns:
        bool: True if state propagation is correct, False otherwise.
    """
    check_prop = True
    comp_i = complex(0, 1)
    for a in np.arange(n_actions):
        for j in np.arange(0, n):
            errores = (
                np.matmul(propagators[a][:, :], bases[a, :, j])
                - np.exp(-comp_i * dt * energies[a, j]) * bases[a, :, j]
            )
            et = np.sum(errores)
            if la.norm(et) > 1e-8:
                check_prop = False
                raise Exception("Error in state propagation")

    if check_prop:
        print("State Propagation: checked")

    return check_prop


def check_sd(n, n_actions, actions, sd):
    """
    Check the spectral decomposition of actions.

    Parameters:
    - n (int): The size of the matrix.
    - n_actions (int): The number of actions.
    - actions (ndarray): The actions matrices.
    - sd (ndarray): The spectral decomposition matrices.

    Returns:
    - bool: True if the spectral decomposition is correct, False otherwise.
    """

    check_sd = True

    for k in np.arange(0, n_actions):
        for i in np.arange(0, n):
            for j in np.arange(0, n):
                if actions[k, i, j] - sd[k, i, j] > 1e-8:
                    print("error in spectral decomposition")
                    check_sd = False
                    raise Exception("Error in spectral decomposition")

    if check_sd:
        print("Correct Spectral Decomposition")

    return check_sd


def test_environment(env, episodes=10):
    print("========== SB3 check_env ==========")
    check_env(env, warn=True)
    print("✅ check_env passed!\n")

    print("========== Custom rollout tests ==========")
    norm_errors = 0
    nan_errors = 0
    obs_errors = 0
    total_steps = 0

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            total_steps += 1

            # Check state norm
            state_vec = env.state
            norm = np.linalg.norm(state_vec)

            if not np.isclose(norm, 1.0, atol=1e-9):
                print(
                    f"[WARN] Episode {ep} Step {step_count}: "
                    f"State norm = {norm:.6f}"
                )
                norm_errors += 1

            # Check for NaNs in state
            if np.isnan(state_vec).any():
                print(
                    f"[ERROR] Episode {ep} Step {step_count}: "
                    "NaN detected in state!"
                )
                nan_errors += 1

            # Check observation validity
            if len(obs) != env.n_features:
                print(
                    f"[ERROR] Episode {ep} Step {step_count}: "
                    f"Observation length mismatch! Expected {env.n_features}, "
                    f"got {len(obs)}"
                )
                obs_errors += 1

    # propagation check
    energies = np.zeros((env.n_actions, env.chain_length),
                        dtype=np.complex128)

    bases = np.zeros((env.n_actions, env.chain_length, env.chain_length),
                     dtype=np.complex128)

    sp_desc = np.zeros((env.n_actions, env.chain_length, env.chain_length),
                       dtype=np.complex128)

    for i in range(0, env.n_actions):
        energies[i, :], bases[i, :, :] = la.eig(env.action_matrices[i, :, :])

        for k in range(0, env.chain_length):
            p = np.outer(bases[i, :, k], bases[i, :, k])
            sp_desc[i, :, :] = sp_desc[i, :, :] + p * energies[i, k]

    prop_check = propagators_check(env.chain_length,
                                   env.n_actions,
                                   env.time_step,
                                   env.propagators,
                                   bases=bases,
                                   energies=energies)
    sd_check = check_sd(env.chain_length, env.n_actions,
                        env.action_matrices, sp_desc)

    print("\n========== Test Report ==========")
    print(f"Total episodes: {episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Norm errors: {norm_errors}")
    print(f"NaN errors: {nan_errors}")
    print(f"Observation errors: {obs_errors}")
    print(f"Propagation check: {'Passed' if prop_check else 'Failed'}")
    print(f"Spectral dec. check: {'Passed' if sd_check else 'Failed'}")
    print("✅ Test completed!\n")

# Main execution block to run tests with different configurations


if __name__ == "__main__":
    for config_file in ['test1.ini', 'test2.ini', 'test3.ini']:

        print(f"Loading configuration from {config_file}")
        config = configparser.ConfigParser()
        config.read(config_file)
        print(", ".join(f"{key}= {value}" for key,
                        value in config['tags'].items()))
        env = QuantumStateEnv(config)
        test_environment(env, episodes=50)
