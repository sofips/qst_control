from state_env import State  # module with environment and dynamics
from RL_brain_pi_deep import DQNPrioritizedReplay  # sumtree version of DQN
import numpy as np
import time  # added to check running time
import configparser
import pickle
import tensorflow.compat.v1 as tf
import pandas as pd
from tqdm import tqdm
import optuna  # for hyperparameter optimization

tf.disable_v2_behavior()


def qst_training(config_instance, optuna_trial=None, progress_bar=True):
    """
    Trains a Deep Q-Network (DQN) agent with prioritized experience replay for quantum state transfer tasks.
    This function initializes the environment and the DQN agent, then runs a training loop for a specified number of episodes.
    During training, it tracks and saves the best action sequences and fidelities, logs results, and computes training metrics.
    Results and metrics are saved to disk for further analysis.
    Args:
        config_instance: Configuration object containing parameters for the environment, agent, and training process.
        optuna_trial (optional): Boolean indicating if the training is being done inside an optuna optimization. Activates
        optuna specific features like trial reporting and pruning.
    Returns:
        RL: The trained DQNPrioritizedReplay agent session.
    Side Effects:
        - Saves model checkpoints, results, action sequences, and metrics to disk.
        - Writes TensorFlow summaries for visualization.
        - Prints updates when a new best fidelity is achieved.
    """

    experiment_name = config_instance.get("experiment", "experiment_alias")
    dirname = experiment_name

    env = State(config_instance=config_instance)  # create environment
    RL = DQNPrioritizedReplay(config_instance=config_instance)  # create RL agent

    number_of_episodes = config_instance.getint(
        "learning_parameters", "number_of_episodes"
    ) # number of training episodes

    max_t_steps = config_instance.getint(
        "system_parameters", "max_t_steps"
    )  # max. number of steps in each episode

    best_action_sequences = [[] for i in range(0, 10)]
    best_fidelities = np.zeros(10)

    results_df = pd.DataFrame(columns=[
        "episode","Qvalue", "max_fidelity", "time_max_fidelity",
        "final_fidelity", "time_final_fidelity", "epsilon"
    ])


    tolerance = config_instance.getfloat("system_parameters", "tolerance")
    success_action_sequences = []  # store successful success_action_seq
    stp = 0  # initialize TOTAL step counter

    t1 = time.time()  # start timer

    episode_iter = range(number_of_episodes)
    if progress_bar:
        episode_iter = tqdm(episode_iter, desc="Training episodes")

    for episode in episode_iter:

        observation = env.reset() # reset environment, get initial observation
        newaction = [] # list to store actions in this episode
        Q = 0
        fid_max = 0
        t_fid_max = 0

        for i in range(max_t_steps):  # episode maximum length

            action = RL.choose_action(
                observation
            )  # observation: input of network; action, the chosen action
            newaction.append(action)

            observation_, reward, fidelity = env.step(
                action
            )  # observation_:new action; done:break the episode or not

            RL.store_transition(
                observation, action, reward, observation_
            )

            Q += reward  # total reward
            if (stp > 500) and (stp % 5 == 0):
                RL.learn()  # update neural network

            observation = observation_  # update current state

            # save max. fidelity value
            if fidelity > fid_max:
                fid_max = fidelity
                t_fid_max = i

            stp += 1
            i += 1

        if episode % 1000 == 0 and episode > 5000:
            average_max_fid = np.mean(results_df["max_fidelity"][-1000:])
            print(f"Episode {episode}, Average Max Fidelity: {average_max_fid:.4f}")
            if optuna_trial is not None:
                optuna_trial.report(average_max_fid, episode)
                if optuna_trial.should_prune():
                    print(f"Trial {optuna_trial.number} pruned at episode {episode}.")

                    t2 = time.time()  # end timer

                    # Save the current state of the RL agent
                    saver = tf.train.Saver()
                    saver.save(RL.sess, dirname + "/final_model/model.ckpt")

                    results_df.to_csv(
                        dirname + "/training_results.txt",
                        sep=" ",
                        float_format="%.8f",
                        index=False,
                        header=True
                    )

                    # store success rate as metric
                    soft_success_training_rate = 0
                    true_success_training_rate = 0

                    for fid in results_df["max_fidelity"]:
                        if fid > 0.9:
                            soft_success_training_rate += 1
                        if fid >= 1 - tolerance:
                            true_success_training_rate += 1

                    soft_success_training_rate = soft_success_training_rate / len(results_df["max_fidelity"])
                    true_success_training_rate = true_success_training_rate / len(results_df["max_fidelity"])

                    metrics = {
                        "soft_success_training_rate": soft_success_training_rate,
                        "true_success_training_rate": true_success_training_rate,
                        "training max fidelity": np.max(results_df["max_fidelity"]),
                        "training average fidelity": np.mean(results_df["max_fidelity"]),
                        "average QValue": np.mean(results_df["Qvalue"]),
                        "training_time": t2 - t1,
                        "number_of_episodes": number_of_episodes,
                        "pruned": False,
                    }
                    # Save the dictionary to a file
                    metrics_file = dirname + "/training_metrics.pkl"
                    with open(metrics_file, "wb") as f:
                        pickle.dump(metrics, f)

                    raise optuna.TrialPruned()


        if fid_max > 0.9:
            success_action_sequences.append(newaction)

        if fid_max > max(best_fidelities):
            print("Best fidelity: ", fid_max)
            saver = tf.train.Saver()
            saver.save(RL.sess, dirname + "/best_model/model.ckpt")

        if fid_max > min(best_fidelities):
            idx = np.argmin(best_fidelities)
            best_fidelities[idx] = fid_max
            best_action_sequences[idx] = newaction

        # store results in dataframe

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

    t2 = time.time()  # end timer

    results_df.to_csv(
        dirname + "/training_results.txt",
        sep=" ",
        float_format="%.8f",
        index=False,
        header=True
    )

    saver.save(RL.sess, dirname + "/final_model/model.ckpt")

    # Save action sequences and fidelities using numpy
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


    # store success rate as metric
    soft_success_training_rate = 0
    true_success_training_rate = 0

    for fid in results_df["max_fidelity"]:
        if fid > 0.9:
            soft_success_training_rate += 1
        if fid >= 1 - tolerance:
            true_success_training_rate += 1

    soft_success_training_rate = soft_success_training_rate / len(results_df["max_fidelity"])
    true_success_training_rate = true_success_training_rate / len(results_df["max_fidelity"])

    metrics = {
        "soft_success_training_rate": soft_success_training_rate,
        "true_success_training_rate": true_success_training_rate,
        "training max fidelity": np.max(results_df["max_fidelity"]),
        "training average fidelity": np.mean(results_df["max_fidelity"]),
        "average QValue": np.mean(results_df["Qvalue"]),
        "training_time": t2 - t1,
        "number_of_episodes": number_of_episodes,
        "pruned": False,
    }

    # Save the dictionary to a file
    metrics_file = dirname + "/training_metrics.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    return RL


def qst_validation(
        config_instance,
        validation_metric='average_val_fidelity',
        validation_episodes=1000
    ):
    experiment_name = config_instance.get("experiment", "experiment_alias")
    dirname = experiment_name

    dir = dirname + '/best_model'

    tf.reset_default_graph()

    # Start a new TensorFlow session and restore variables from checkpoint
    
    with tf.Session() as sess:
        RL_val = DQNPrioritizedReplay(
            config_instance=config_instance,
            sess=sess
        )  # create RL agent with restored session

        saver = tf.train.Saver()
        checkpoint_path = f'{dir}/model.ckpt'
        if not tf.train.checkpoint_exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        saver.restore(sess, checkpoint_path)

        env = State(config_instance=config_instance)  # create environment

        results_df = pd.DataFrame(columns=[
            "episode", "Qvalue", "max_fidelity", "time_max_fidelity",
            "final_fidelity", "time_final_fidelity", "epsilon"
        ])

        max_t_steps = config_instance.getint(
            "system_parameters", "max_t_steps"
        )

        for episode in range(validation_episodes):
            # reset environment, get initial observation
            observation = env.reset()
            Q = 0
            fid_max = 0
            t_fid_max = 0

            for i in range(max_t_steps):  # episode maximum length
                
                action = RL_val.choose_action(observation, eval=True)

                observation_, reward, fidelity = env.step(action)

                observation = observation_.copy()  # update current state
                Q += reward  # total reward

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

    # Save the dictionary to a file
    metrics_file = dirname + "/validation_metrics.pkl"
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    return metrics[validation_metric]

