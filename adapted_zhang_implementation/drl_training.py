"""Deep Q-Network training module for Quantum State Transfer optimization.

This module implements training and validation loops for a Deep Q-Network (DQN)
agent with prioritized experience replay. The agent learns to find optimal control
action sequences for quantum state transfer in spin chains, with support for
noise injection during training and integration with Optuna for hyperparameter
optimization.
"""
import configparser
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tqdm import tqdm

import optuna

from RL_brain_pi_deep import DQNPrioritizedReplay
from state_env import State

tf.disable_v2_behavior()


def qst_training(config_instance, optuna_trial=None, progress_bar=True):
    """Train a DQN agent for quantum state transfer control.

    Executes episodic training loop where the agent learns to maximize fidelity
    of quantum state transfer through sequential control actions. The agent uses
    prioritized experience replay for sample efficiency. Training is logged with
    episode metrics, and best/final models are saved.

    Configuration Requirements:
        The config_instance must contain the following sections:
        - experiment: experiment_alias (directory name for results)
        - system_parameters: chain_length, max_t_steps, tolerance
        - learning_parameters: number_of_episodes, learning_rate, gamma
        - dqn_parameters: epsilon, batch_size, memory_size
        - noise_parameters (optional): noise_amplitude, noise_probability

    Parameters
    ----------
    config_instance : configparser.ConfigParser
        Configuration object containing all training hyperparameters and
        environment settings.
    optuna_trial : optuna.trial.Trial, optional
        Optuna trial object for hyperparameter optimization. When provided,
        enables intermediate value reporting and trial pruning based on
        validation fidelity. Default is None.
    progress_bar : bool, optional
        If True, displays tqdm progress bar during training. Default is True.

    Returns
    -------
    agent : DQNPrioritizedReplay
        The trained DQN agent with learned policy.

    Outputs
    -------
    Saves to disk in directory specified by config_instance:
        - best_model/model.ckpt: Checkpoint with highest max fidelity
        - final_model/model.ckpt: Final checkpoint after all episodes
        - training_results.txt: CSV with per-episode metrics
        - training_metrics.pkl: Dictionary with aggregated training statistics
        - best_action_sequences.txt: Top 10 action sequences by fidelity
        - success_action_sequences.txt: Sequences with fidelity > 0.9
        - best_fidelities.txt: Fidelities of top 10 sequences
        - logs/: TensorFlow event files for visualization

    Notes
    -----
    - Training metrics include: Q-values, max/final fidelities, epsilon decay
    - Success rate computed at: soft (fidelity > 0.9) and true (fidelity >= 1-tol)
    - Learning starts after 500 steps with updates every 5 steps
    - Optuna pruning checks occur every 1000 episodes after episode 5000
    """

    experiment_name = config_instance.get("experiment", "experiment_alias")
    dirname = experiment_name

    env = State(config_instance=config_instance)
    RL = DQNPrioritizedReplay(config_instance=config_instance)

    number_of_episodes = config_instance.getint(
        "learning_parameters", "number_of_episodes"
    )

    max_t_steps = config_instance.getint(
        "system_parameters", "max_t_steps"
    )

    best_action_sequences = [[] for i in range(0, 10)]
    best_fidelities = np.zeros(10)

    results_df = pd.DataFrame(columns=[
        "episode", "Qvalue", "max_fidelity", "time_max_fidelity",
        "final_fidelity", "time_final_fidelity", "epsilon"
    ])

    tolerance = config_instance.getfloat("system_parameters", "tolerance")
    success_action_sequences = []
    stp = 0

    t1 = time.time()

    episode_iter = range(number_of_episodes)
    if progress_bar:
        episode_iter = tqdm(episode_iter, desc="Training episodes")

    for episode in episode_iter:

        observation = env.reset()
        newaction = []
        Q = 0
        fid_max = 0
        t_fid_max = 0

        for i in range(max_t_steps):

            action = RL.choose_action(observation)
            newaction.append(action)

            observation_, reward, fidelity = env.step(action)

            RL.store_transition(
                observation, action, reward, observation_
            )

            Q += reward
            if (stp > 500) and (stp % 5 == 0):
                RL.learn()

            observation = observation_

            if fidelity > fid_max:
                fid_max = fidelity
                t_fid_max = i

            stp += 1
            i += 1

        if episode % 1000 == 0 and episode > 5000:
            average_max_fid = np.mean(
                results_df["max_fidelity"][-1000:]
            )
            print(f"Episode {episode}, "
                  f"Average Max Fidelity: {average_max_fid:.4f}")
            if optuna_trial is not None:
                optuna_trial.report(average_max_fid, episode)
                if optuna_trial.should_prune():
                    print(f"Trial {optuna_trial.number} "
                          f"pruned at episode {episode}.")

                    t2 = time.time()

                    saver = tf.train.Saver()
                    saver.save(RL.sess,
                               dirname + "/final_model/model.ckpt")

                    results_df.to_csv(
                        dirname + "/training_results.txt",
                        sep=" ",
                        float_format="%.8f",
                        index=False,
                        header=True
                    )

                    soft_success_training_rate = 0
                    true_success_training_rate = 0

                    for fid in results_df["max_fidelity"]:
                        if fid > 0.9:
                            soft_success_training_rate += 1
                        if fid >= 1 - tolerance:
                            true_success_training_rate += 1

                    n_episodes = len(results_df["max_fidelity"])
                    soft_success_training_rate = (
                        soft_success_training_rate / n_episodes
                    )
                    true_success_training_rate = (
                        true_success_training_rate / n_episodes
                    )

                    metrics = {
                        "soft_success_training_rate": (
                            soft_success_training_rate
                        ),
                        "true_success_training_rate": (
                            true_success_training_rate
                        ),
                        "training max fidelity": np.max(
                            results_df["max_fidelity"]
                        ),
                        "training average fidelity": np.mean(
                            results_df["max_fidelity"]
                        ),
                        "average QValue": np.mean(
                            results_df["Qvalue"]
                        ),
                        "training_time": t2 - t1,
                        "number_of_episodes": number_of_episodes,
                        "pruned": False,
                    }
                    metrics_file = dirname + "/training_metrics.pkl"
                    with open(metrics_file, "wb") as f:
                        pickle.dump(metrics, f)

                    raise optuna.TrialPruned()

        if fid_max > 0.9:
            success_action_sequences.append(newaction)

        if fid_max > max(best_fidelities):
            saver = tf.train.Saver()
            saver.save(RL.sess, dirname + "/best_model/model.ckpt")

        if fid_max > min(best_fidelities):
            idx = np.argmin(best_fidelities)
            best_fidelities[idx] = fid_max
            best_action_sequences[idx] = newaction

        results_df = pd.concat([
            results_df,
            pd.DataFrame([{
                "episode": episode,
                "Qvalue": Q,
                "max_fidelity": fid_max,
                "time_max_fidelity": t_fid_max,
                "final_fidelity": fidelity,
                "time_final_fidelity": i,
                "epsilon": RL.epsilon
            }])
        ], ignore_index=True)

    t2 = time.time()

    results_df.to_csv(
        dirname + "/training_results.txt",
        sep=" ",
        float_format="%.8f",
        index=False,
        header=True
    )

    saver.save(RL.sess, dirname + "/final_model/model.ckpt")

    np.savetxt(dirname + "/success_action_sequences.txt",
               np.array(success_action_sequences, dtype=object),
               fmt="%s")
    np.savetxt(dirname + "/best_action_sequences.txt",
               np.array(best_action_sequences, dtype=object),
               fmt="%s")
    np.savetxt(dirname + "/best_fidelities.txt",
               best_fidelities,
               fmt="%.8f")

    logdirname = dirname + "/logs/"
    tf.summary.FileWriter(logdirname, RL.sess.graph)

    soft_success_training_rate = 0
    true_success_training_rate = 0

    for fid in results_df["max_fidelity"]:
        if fid > 0.9:
            soft_success_training_rate += 1
        if fid >= 1 - tolerance:
            true_success_training_rate += 1

    n_episodes = len(results_df["max_fidelity"])
    soft_success_training_rate = (
        soft_success_training_rate / n_episodes
    )
    true_success_training_rate = (
        true_success_training_rate / n_episodes
    )

    metrics = {
        "soft_success_training_rate": soft_success_training_rate,
        "true_success_training_rate": true_success_training_rate,
        "training max fidelity": np.max(
            results_df["max_fidelity"]
        ),
        "training average fidelity": np.mean(
            results_df["max_fidelity"]
        ),
        "average QValue": np.mean(results_df["Qvalue"]),
        "training_time": t2 - t1,
        "number_of_episodes": number_of_episodes,
        "pruned": False,
    }

    metrics_file = dirname + "/training_metrics.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    return RL


def qst_validation(
        config_instance,
        validation_metric='average_val_fidelity',
        validation_episodes=1000
    ):
    """Validate a trained DQN agent on quantum state transfer task.

    Evaluates a previously trained DQN agent without learning (greedy policy)
    over multiple episodes to obtain validation statistics. Uses the best model
    checkpoint saved during training. Results are saved to disk for analysis.

    Configuration Requirements:
        The config_instance must match the one used during training, with:
        - experiment: experiment_alias (directory containing trained models)
        - system_parameters: max_t_steps, tolerance (for success rate)
        - All environment settings (chain_length, noise parameters, etc.)

    Parameters
    ----------
    config_instance : configparser.ConfigParser
        Configuration object matching the training configuration. Must point to
        existing best_model/ directory.
    validation_metric : {'average_val_fidelity', 'general_val_fidelity', \
                         'average_time_max_fidelity', 'average_QValue'}, optional
        Metric to extract and return from validation results. Default is
        'average_val_fidelity' (mean max fidelity across all episodes).
    validation_episodes : int, optional
        Number of validation episodes to run. Default is 1000.

    Returns
    -------
    float
        Requested validation metric value.

    Outputs
    -------
    Saves to disk in directory specified by config_instance:
        - validation_results.txt: CSV with per-episode metrics
        - validation_metrics.pkl: Dictionary with all validation statistics
    """
    experiment_name = config_instance.get("experiment", "experiment_alias")
    dirname = experiment_name

    dir_path = dirname + '/best_model'

    tf.reset_default_graph()

    with tf.Session() as sess:
        RL_val = DQNPrioritizedReplay(
            config_instance=config_instance,
            sess=sess
        )

        saver = tf.train.Saver()
        checkpoint_path = f'{dir_path}/model.ckpt'
        if not tf.train.checkpoint_exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}"
            )
        saver.restore(sess, checkpoint_path)

        env = State(config_instance=config_instance)

        results_df = pd.DataFrame(columns=[
            "episode", "Qvalue", "max_fidelity", "time_max_fidelity",
            "final_fidelity", "time_final_fidelity", "epsilon"
        ])

        max_t_steps = config_instance.getint(
            "system_parameters", "max_t_steps"
        )

        for episode in range(validation_episodes):
            observation = env.reset()
            Q = 0
            fid_max = 0
            t_fid_max = 0

            for i in range(max_t_steps):

                action = RL_val.choose_action(observation, eval=True)

                observation_, reward, fidelity = env.step(action)

                observation = observation_.copy()
                Q += reward

                if fidelity > fid_max:
                    fid_max = fidelity
                    t_fid_max = i

            results_df = pd.concat([
                results_df,
                pd.DataFrame([{
                    "episode": episode,
                    "Qvalue": Q,
                    "max_fidelity": fid_max,
                    "time_max_fidelity": t_fid_max,
                    "final_fidelity": fidelity,
                    "time_final_fidelity": i,
                    "epsilon": RL_val.epsilon
                }])
            ], ignore_index=True)

        results_df.to_csv(
            dirname + "/validation_results.txt",
            sep=" ",
            float_format="%.8f",
            index=False,
            header=True
        )

        metrics = {
            "general_val_fidelity": np.max(results_df["max_fidelity"]),
            "average_val_fidelity": np.mean(results_df["max_fidelity"]),
            "average_time_max_fidelity": np.mean(
                results_df["time_max_fidelity"]
            ),
            "average_QValue": np.mean(results_df["Qvalue"]),
            "validation_episodes": validation_episodes,
        }

    sess.close()
    tf.reset_default_graph()

    metrics_file = dirname + "/validation_metrics.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    return metrics[validation_metric]

