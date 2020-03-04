import tensorflow as tf
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import datetime
import time
import gym
import pdb
import random
import argparse
from tqdm import tqdm
import csv
import sys, os 
sys.path.append('../common')
from custom_losses import Agent_Wrapper
sys.path.append('../models')
import A2C_OD_SharedCNN as Model


class Agent(Agent_Wrapper):
    def __init__(self, model, total_steps, lr=1e-3, gamma=0.999, value_c=0.5, entropy_c=1e-2, reconstruction_c=1, clip_range=0.2, lam=0.95, num_PPO_epochs=3, batch_sz=1024):
        self.model = model
        self.value_c = value_c
        self.entropy_c = entropy_c
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate = lr, 
                decay_steps = total_steps // batch_sz * num_PPO_epochs,
                end_learning_rate = 1e-5,
                power = 1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-5)
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.num_PPO_epochs = num_PPO_epochs
        self.batch_size = batch_sz
        self.reconstruction_c = reconstruction_c
        self.peek_prob = 0.5
        
        #compile model
        self.model = model
        self.model.compile(
                optimizer=self.optimizer,
                loss=[
                    self._logits_loss_PPO, # actor loss
                    self._value_loss,   # critic loss
                    self._reconstruction_loss_2
                    ])

    def train(self, env, batch_sz=5000, updates=1, show_visual=True, random_action=False, show_first=False, tb_callback=None):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,))
        logits = np.empty((batch_sz, env.action_space.n))
        neglogprobs_prev = np.empty((batch_sz,))
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        true_observations_p1 = np.empty((batch_sz,) + env.observation_space.shape)
        reconstructions = np.zeros_like(observations)
        reconstr_mask = np.zeros((batch_sz,))
        observations = np.concatenate((observations,observations), axis=-1)


        # Training loop: collect samples, send to optimizer, repeat updates times.
        print('\n\nRunning episodes...')
        ep_rewards = [0.0]
        next_obs = env.reset().astype(np.float64)
        next_obs = (next_obs)/256.0
        prev_obs = next_obs.copy()
        first_run = True
        for update in range(updates):
            for step in tqdm(range(batch_sz)):
                combined_obs = np.concatenate((next_obs,prev_obs), axis=-1)
                prev_obs = next_obs.copy()
                observations[step] = combined_obs.copy()
                if(show_visual or (show_first and first_run)):
                    env.render()
                if(random_action):
                    _, values[step], neglogprobs_prev[step], reconstructions[step] = self.model.action_value_neglogprobs_obs(
                            combined_obs[None,:])
                    actions[step] = env.action_space.sample()
                else:
                    actions[step], values[step], neglogprobs_prev[step], reconstructions[step] = self.model.action_value_neglogprobs_obs(
                            combined_obs[None,:])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                next_obs = next_obs.astype(np.float64)
                next_obs = (next_obs)/256.0
                true_observations_p1[step] = next_obs.copy()

                # Observational Dropout
                if (np.random.uniform() > self.peek_prob):
                    next_obs = reconstructions[step]
                    #next_obs = next_obs + 1e-2*np.random.randn(*next_obs.shape)
                    next_obs = np.maximum(next_obs, -1)
                    next_obs = np.minimum(next_obs, 1)
                    reconstr_mask[step] = 1

                logits[step] = self.model.predict_on_batch(combined_obs[None,:])[0][0]

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    rewards[step] -= 1
                    _, next_value = self.model.action_value(
                            combined_obs[None,:])
                    if(not random_action):
                        # Don't bootstrap first (random) run 
                        rewards[step] += self.gamma*next_value
                    ep_rewards.append(0.0)
                    next_obs = env.reset().astype(np.float64)
                    next_obs = (next_obs)/256.0  
                    prev_obs = next_obs.copy()
                    first_run = False
            
            # Handle bootstrapped Critic value for last (unfinished) run
            if(not random_action):
                _, next_value = self.model.action_value(combined_obs[None,:])
            else:
                next_value = np.array([0])
            returns, advs = self._returns_GAE_advantages(rewards, dones, values, next_value)

            # # Remove outliers with respect to advantages
            # idx = self._normalize_advantages(advs)
            # advs = advs[idx]
            # actions = actions[idx]
            # neglogprobs_prev = neglogprobs_prev[idx]
            # returns = returns[idx]
            # values = values[idx]
            # observations = observations[idx]
            # reconstructions = reconstructions[idx]
            # reconstr_mask = reconstr_mask[idx]

            # Print out useful training metrics
            print('Average Advantage: ' + str(np.mean(advs)))
            print('Advantage STD: ' + str(np.std(advs)))
            # Print out distribution of Rewards-Value 
            print('Advantage distribution for batch:')
            print(np.histogram(advs)[0]) 
            print('Advantage histogram bins:')
            print(np.histogram(advs)[1]) 

            # Normalize advantages for numerical stability
            advs -= np.mean(advs)
            advs /= np.std(advs) + 1e-6

            # A trick to input actions, advantages, and log probabilities through same API
            acts_advs_and_neglogprobs = np.concatenate([actions[:, None], advs[:, None], neglogprobs_prev[:,None]], axis=-1)

            # A trick to input returns and previous predicted values through same API
            returns_and_prev_values = np.concatenate([returns[:, None], values[:, None]], axis=-1)

            # A trick to input reconstructions and advantages through same API.
            advs_reshaped = advs.reshape((-1,) + (1,)*(reconstructions.ndim-1))
            advs_broadcast = reconstructions*0 + advs_reshaped
            mask_reshaped = reconstr_mask.reshape((-1,) + (1,)*(reconstructions.ndim-1))
            mask_broadcast = reconstructions*0 + mask_reshaped
            #reconstr_advs_and_mask = np.concatenate([reconstructions[..., None], advs_broadcast[..., None], mask_broadcast[..., None]], axis=-1)
            obs_reconstr_advs_and_mask = np.concatenate([observations[..., :3, None], reconstructions[..., None], advs_broadcast[..., None], mask_broadcast[..., None]], axis=-1)

            # Disregard incomplete runs at head/tail
            idx = np.argwhere(dones==1)
            first_index = idx[0][0]
            last_index = idx[-1][0]

            # Trim/Decimate observations (if memory capacity is an issue)
            skip = 1
            observations = observations[first_index+1:last_index+1:skip]
            true_observations_p1 = true_observations_p1[first_index+1:last_index+1:skip]
            reconstructions = reconstructions[first_index+1:last_index+1:skip]
            returns_and_prev_values = returns_and_prev_values[first_index+1:last_index+1:skip]
            acts_advs_and_neglogprobs = acts_advs_and_neglogprobs[first_index+1:last_index+1:skip]
            #reconstr_advs_and_mask = reconstr_advs_and_mask[first_index+1:last_index+1:skip]
            obs_reconstr_advs_and_mask = obs_reconstr_advs_and_mask[first_index+1:last_index+1:skip]

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            print('training on batch')
            #print('Current optimizer learning rate: ', str(self.optimizer.learning_rate))
            if(tb_callback):
                self.model.fit(observations,
                        [acts_advs_and_neglogprobs, returns_and_prev_values, obs_reconstr_advs_and_mask],
                        shuffle=True,
                        batch_size=self.batch_size,
                        epochs=self.num_PPO_epochs,
                        callbacks=[tb_callback])
            else:
                self.model.fit(observations,
                        [acts_advs_and_neglogprobs, returns_and_prev_values, obs_reconstr_advs_and_mask],
                        shuffle=True,
                        batch_size=self.batch_size,
                        epochs=self.num_PPO_epochs)

            # Print out useful training progress information
            np.set_printoptions(precision=3)
            print('Action distribution for batch:')
            print(np.histogram(actions, bins=list(range(0,env.action_space.n)))[0])
            print('Policy Entropy:')
            policy_entropy = np.mean(scipy.stats.entropy(np.exp(logits.T)))
            print(policy_entropy)
            print('Policy logits peak to peak:')
            print(np.ptp(logits, axis=0))
            print('Mean probabilities of actions:' + str(np.mean(np.exp(-neglogprobs_prev))) +
                ', STD probabilities: ' + str(np.std(np.exp(-neglogprobs_prev))))

        return ep_rewards, policy_entropy

    def test(self, env, num_steps=5000, render=True):
        total_reward = 0
        next_obs = env.reset().astype(np.float64)
        next_obs = (next_obs)/256.0
        prev_obs = next_obs.copy()
        for step in tqdm(range(num_steps)):
            combined_obs = np.concatenate((next_obs,prev_obs), axis=-1)
            prev_obs = next_obs.copy()
            if(render):
                env.render()
            action, _, _ = self.model.action_value_neglogprob(combined_obs[None,:])
            next_obs, reward, done, _ = env.step(action)
            next_obs = next_obs.astype(np.float64)
            next_obs = (next_obs)/256.0

            total_reward += reward
            if done:
                break
        return reward

    def play_reconstructions(self, observations, reconstructions):
        for i in range(len(observations)):
            plt.subplot(2,3,4)
            plt.imshow((reconstructions[i,...,0]- np.min(reconstructions[i,...,0])) / np.ptp(reconstructions[i,...,0]))
            plt.subplot(2,3,5)
            plt.imshow((reconstructions[i,...,1]- np.min(reconstructions[i,...,1])) / np.ptp(reconstructions[i,...,1]))
            plt.subplot(2,3,6)
            plt.imshow((reconstructions[i,...,2]- np.min(reconstructions[i,...,2])) / np.ptp(reconstructions[i,...,2]))
            plt.subplot(2,3,1)
            plt.imshow((observations[i,...,0]- np.min(observations[i,...,0])) / np.ptp(observations[i,...,0]))
            plt.subplot(2,3,2)
            plt.imshow((observations[i,...,1]- np.min(observations[i,...,1])) / np.ptp(observations[i,...,1]))
            plt.subplot(2,3,3)
            plt.imshow((observations[i,...,2]- np.min(observations[i,...,2])) / np.ptp(observations[i,...,2]))
            plt.show(block=False)
            input('press enter to continue')
            plt.close()
        return



def main():
    parser = argparse.ArgumentParser(description='RL training parameters')
    parser.add_argument('-v', '--visual', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=5000)
    parser.add_argument('-sf', '--show_first', default=False, action='store_true')
    parser.add_argument('-l', '--load_model_path', default=None)
    parser.add_argument('-ts', '--total_steps', type=int, default=int(5e6))
    args = parser.parse_args()
    np.set_printoptions(precision=3)

    #random.seed(0)
    #tf.random.set_seed(0)

    # enable dynamic GPU memory allocation
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0
    sim_steps = 0
    batch_sz = args.batch_size
    # Initialize OpenAI Procgen environment 
    env = gym.make("procgen:procgen-starpilot-v0", num_levels=0, start_level=0, distribution_mode="easy") 
    with tf.Graph().as_default():
        #tf.compat.v1.disable_eager_execution()
        # set up tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/tensorboard/' + current_time
        tensorboard_callback = None #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model = Model.Model(env.action_space.n)
        obs = env.reset()
        agent = Agent(model, args.total_steps)
        if args.load_model_path:
            print('Loading pre-trained weights~~~')
            print('Loading Model from File: {}'.format(args.load_model_path))
            agent.model.load_weights(args.load_model_path)
        else:
            print('\nRandom Trajectories Cold Start...')
            rewards_history, _ = agent.train(
                    env, updates=1, 
                    batch_sz=batch_sz, 
                    random_action=True, 
                    show_visual=args.visual, 
                    show_first=args.show_first,
                    tb_callback=None)
        rewards_means = np.array([])
        rewards_stds = np.array([])
        rewards_max = np.array([])
        graph = tf.compat.v1.get_default_graph()
        iter_count = 0
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        clip_rate = 0.2
        with open('../logs/' + start_time + '_' + model.model_name + '.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['num_steps', 'num_episodes', 'rew_mean', 'rew_std', 'entropy'])
            for _ in range(args.total_steps // args.batch_size + 1):
                iter_count += 1
                rewards_history, policy_entropy = agent.train(env, 
                        batch_sz=batch_sz, 
                        show_visual=args.visual, 
                        show_first=args.show_first,
                        tb_callback=tensorboard_callback)
                agent.entropy_c *= 0.99  # reduce entropy over iterations
                #agent.peek_prob = (1 - 0.95*(1 - agent.peek_prob))
                agent.clip_range = clip_rate*(1 - sim_steps/args.total_steps)  # reduce clip range over iterations
                print('Current Entropy Coeff: {}, Current Clip Range: {}, Peek Probability: {}'.format(agent.entropy_c, agent.clip_range, agent.peek_prob))
                sim_steps += batch_sz
                rewards_means = np.append(rewards_means, np.mean(rewards_history[:-1]))
                rewards_stds = np.append(rewards_stds, np.std(rewards_history[:-1]))
                rewards_max = np.append(rewards_max, np.amax(rewards_history[:-1]))
                print('Total Sim Steps: ' + str(sim_steps))
                print('Number of levels: ' + str(len(rewards_history)))
                print('Epoch mean reward: ')
                print(rewards_means[-10:])
                print('Epoch std reward: ')
                print(rewards_stds[-10:])
                print('Epoch max reward: ')
                print(rewards_max[-10:])
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                csv_writer.writerow([str(sim_steps), str(len(rewards_history)), str(rewards_means[-1]), 
                    str(rewards_stds[-1]), str(policy_entropy)])
                if(iter_count%10 == 0):
                    model.save_weights('../log_weights/' + current_time + '_' + model.model_name + '_' + str(sim_steps), 
                            save_format='tf')
        print('Finished Training, testing now...')
        return


if __name__ == "__main__":
    main()


