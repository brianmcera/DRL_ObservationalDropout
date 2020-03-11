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
from src.common.custom_losses import Agent_Wrapper

class Agent(Agent_Wrapper):
    def __init__(self, model, total_steps, lr=1e-3, gamma=0.999, value_c=0.5, entropy_c=1e-2, entropy_decayrate=0.99, reconstruction_c=1, initial_clip_range=0.2, lam=0.95, num_PPO_epochs=3, batch_sz=1024, remove_outliers=False, initial_peek_prob=0.1, peekprob_decayrate=1.0):
        self.model = model
        self.total_steps = total_steps
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
        self.initial_clip_range = initial_clip_range
        self.clip_range = initial_clip_range
        self.num_PPO_epochs = num_PPO_epochs
        self.batch_size = batch_sz
        self.reconstruction_c = reconstruction_c
        self.initial_peek_prob = initial_peek_prob
        self.peekprob_decayrate = peekprob_decayrate
        self.peek_prob = self.initial_peek_prob
        self.remove_outliers = remove_outliers
        self.entropy_decayrate = entropy_decayrate
        
        #compile model
        self.model = model
        self.model.compile(
                optimizer=self.optimizer,
                loss=[
                    self._logits_loss_PPO, # actor loss
                    self._value_loss,   # critic loss
                    self._reconstruction_loss_2  # observational dropout loss
                    ])

    def train(self, env, current_sim_steps, sim_batch_sz=5000, updates=1, show_visual=True, random_action=False, show_first=False, tb_callback=None, v_recorder=None, streamlit=None):
        # Storage helpers for a single batch of data.
        actions = np.empty((sim_batch_sz,))
        logits = np.empty((sim_batch_sz, env.action_space.n))
        neglogprobs_prev = np.empty((sim_batch_sz,))
        rewards, dones, values = np.empty((3, sim_batch_sz))
        observations = np.empty((sim_batch_sz,) + env.observation_space.shape)
        true_observations_p1 = np.empty((sim_batch_sz,) + env.observation_space.shape)
        reconstructions = np.zeros_like(observations)
        reconstr_mask = np.zeros((sim_batch_sz,))
        observations = np.concatenate((observations,observations), axis=-1)


        # Training loop: collect samples, send to optimizer, repeat updates times.
        print('\n\nRunning episodes...')
        ep_rewards = [0.0]
        next_obs = env.reset().astype(np.float64)
        next_obs = (next_obs)/256.0
        prev_obs = next_obs.copy()
        first_run = True
        for update in range(updates):
            for step in tqdm(range(sim_batch_sz)):
                combined_obs = np.concatenate((next_obs,prev_obs), axis=-1)
                prev_obs = next_obs.copy()
                observations[step] = combined_obs.copy()
                if streamlit:
                    st_obj = streamlit[0]
                    render_fcn = streamlit[1]
                    render_fcn(st_obj, env, 0)
                if(v_recorder):
                    v_recorder.capture_frame()
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
                    next_obs = np.maximum(next_obs, -1)
                    next_obs = np.minimum(next_obs, 1)
                    reconstr_mask[step] = 1

                logits[step] = self.model.predict_on_batch(combined_obs[None,:])[0][0]

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    rewards[step] -= 0.1
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

            # Remove outliers with respect to advantages
            if(self.remove_outliers):
                idx = self._normalize_advantages(advs)
                advs = advs[idx]
                actions = actions[idx]
                neglogprobs_prev = neglogprobs_prev[idx]
                returns = returns[idx]
                values = values[idx]
                observations = observations[idx]
                reconstructions = reconstructions[idx]
                reconstr_mask = reconstr_mask[idx]

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
            obs_reconstr_advs_and_mask = np.concatenate([observations[..., :3, None], reconstructions[..., None],
                advs_broadcast[..., None], mask_broadcast[..., None]], axis=-1)

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
            #inputs_dataset = tf.data.Dataset.from_tensor_slices(observations)
            #outputs_dataset = tf.data.Dataset.from_tensor_slices([acts_advs_and_neglogprobs, returns_and_prev_values, obs_reconstr_advs_and_mask])
            #dataset = tf.data.Dataset.zip((inputs_dataset, outputs_dataset))

            #def generator():
            #    for ob, out1, out2, out3 in zip(observations,acts_advs_and_neglogprobs,returns_and_prev_values,obs_reconstr_advs_and_mask):
            #        yield ob, (out1, out2, out3)
            #dataset = tf.data.Dataset.from_generator(generator, output_types=tf.float64, output_shapes=((15),(1),(64,64,3))).batch(self.batch_size)
            #dataset = tf.data.Dataset.from_tensor_slices((observations,
            #    (acts_advs_and_neglogprobs, returns_and_prev_values, obs_reconstr_advs_and_mask))).shuffle(1024).batch(self.batch_size)
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
                #self.model.fit(dataset,
                #        epochs=self.num_PPO_epochs)

            metrics = self.print_metrics(env.action_space.n, actions, logits, neglogprobs_prev)

            # Update entropy, clip range, and peek probability
            self.entropy_c *= self.entropy_decayrate  # reduce entropy over iterations
            self.peek_prob = (1 - self.peekprob_decayrate*(1 - self.peek_prob))
            self.clip_range = self.initial_clip_range*(1 - current_sim_steps/self.total_steps)  # reduce clip range over iterations

        return ep_rewards, metrics

    def print_metrics(self, n, actions, logits, neglogprobs_prev):
        # Print out useful training progress information
        np.set_printoptions(precision=3)
        metrics ={}
        print('Action distribution for batch:')
        print(np.histogram(actions, bins=list(range(0,n)))[0])
        metrics['policy_entropy'] = np.mean(scipy.stats.entropy(np.exp(logits.T)))
        print('Policy Entropy: {}'.format(metrics['policy_entropy']))
        print('Policy logits peak to peak:')
        print(np.ptp(logits, axis=0))
        metrics['mean_ac_prob'] = np.mean(np.exp(-neglogprobs_prev))
        metrics['std_ac_prob'] = np.std(np.exp(-neglogprobs_prev))
        print('Mean probabilities of actions: {}, STD probabilities: {}'.format(metrics['mean_ac_prob'], metrics['std_ac_prob']))
        print('Current Entropy Coeff: {}, Current Clip Range: {}, Peek Probability: {}'.format(self.entropy_c, self.clip_range, self.peek_prob))
        return metrics

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
        for _ in range(10):
            i = np.random.randint(len(observations))
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
