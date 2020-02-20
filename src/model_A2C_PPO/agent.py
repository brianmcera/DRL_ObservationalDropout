import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn import preprocessing, decomposition
import scipy
import matplotlib.pyplot as plt
import datetime
import time
import gym
import pdb
import random
import argparse
from tqdm import tqdm
import csv

import model as Model
import utils


class Agent:
    def __init__(self, model, total_steps, lr=5e-4, gamma=0.999, value_c=0.5, entropy_c=1e-2, clip_range=0.2, lam=0.95, num_PPO_epochs=3, batch_sz=1024):
        self.model = model
        self.value_c = value_c
        self.entropy_c = entropy_c
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate = lr, 
                decay_steps = total_steps // batch_sz * num_PPO_epochs,
                end_learning_rate = 1e-6,
                power = 1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-5)
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.num_PPO_epochs = num_PPO_epochs
        self.batch_size = batch_sz
        
        #compile model
        self.model = model
        self.model.compile(
                optimizer=self.optimizer,
                loss=[
                    self._logits_loss_PPO, # actor loss
                    self._value_loss   # critic loss
                    ])

    def train(self, env, batch_sz=5000, updates=1, show_visual=True, random_action=False, show_first=False, tb_callback=None):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,))
        logits = np.empty((batch_sz, env.action_space.n))
        neglogprobs_prev = np.empty((batch_sz,))
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
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
                    _, values[step], neglogprobs_prev[step] = self.model.action_value_neglogprob(
                            combined_obs[None,:])
                    actions[step] = env.action_space.sample()
                else:
                    actions[step], values[step], neglogprobs_prev[step] = self.model.action_value_neglogprob(
                            combined_obs[None,:])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                next_obs = next_obs.astype(np.float64)
                next_obs = (next_obs)/256.0

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

            # Remove outliers with respect to advantage estimates 
            adv_mean = np.mean(advs)
            adv_std = np.std(advs)
            idx = np.logical_and(advs > adv_mean - 3*adv_std, advs < adv_mean + 3*adv_std)
            advs = advs[idx]
            actions = actions[idx]
            neglogprobs_prev = neglogprobs_prev[idx]
            returns = returns[idx]
            values = values[idx]
            observations = observations[idx]

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

            # A trick to input actions, advantages, and selected action log probability through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            acts_advs_and_neglogprobs = np.concatenate([actions[:, None], advs[:, None], neglogprobs_prev[:,None]], axis=-1)

            # A trick to input returns and previous predicted values through same API
            returns_and_prev_values = np.concatenate([returns[:, None], values[:, None]], axis=-1)

            # Disregard incomplete runs at head/tail
            idx = np.argwhere(dones==1)
            first_index = idx[0][0]
            last_index = idx[-1][0]

            # Trim/Decimate observations (if memory capacity is an issue)
            skip = 1
            observations = observations[first_index+1:last_index+1:skip]
            returns_and_prev_values = returns_and_prev_values[first_index+1:last_index+1:skip]
            acts_and_advs = acts_and_advs[first_index+1:last_index+1:skip]
            acts_advs_and_neglogprobs = acts_advs_and_neglogprobs[first_index+1:last_index+1:skip]

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            print('training on batch')
            #print('Current optimizer learning rate: ', str(self.optimizer.learning_rate))
            if(tb_callback):
                self.model.fit(observations,
                        [acts_advs_and_neglogprobs, returns_and_prev_values],
                        shuffle=True,
                        batch_size=self.batch_size,
                        epochs=self.num_PPO_epochs,
                        callbacks=[tb_callback])
            else:
                self.model.fit(observations,
                        [acts_advs_and_neglogprobs, returns_and_prev_values],
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

    def _returns_MonteCarlo_advantages(self, rewards, dones, values, next_value):
        # 'next value' is the bootstrapped returned value from the Critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1-dones[t])
        returns = returns[:-1]  # all but last
        advantages = returns - values  # advantage over Critic estimates
        return returns, advantages

    def _returns_GAE_advantages(self, rewards, dones, values, next_value):
        # 'next value' is the bootstrapped returned value from the Critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        values = np.append(values, next_value, axis=-1)
        advantages = np.zeros_like(values)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma*values[t+1]*(1-dones[t]) - values[t]
            advantages[t] = lastgaelam = delta + self.gamma*self.lam*(1-dones[t])*lastgaelam  # double check (1-dones) here
        returns = advantages + values
        returns = returns[:-1]  # all but last
        advantages = advantages[:-1]
        return returns, advantages

    def _value_loss(self, returns_and_prev_values, value):
        # this function calculates loss with value TD error
        returns, prev_values = tf.split(returns_and_prev_values, 2, axis=-1)
        # clip value to reduce variability during Critic training
        vpredclipped = prev_values + tf.clip_by_value(value - prev_values, -self.clip_range, self.clip_range)
        v_losses1 = tf.square(value - returns)
        v_losses2 = tf.square(vpredclipped - returns)
        return self.value_c*tf.reduce_mean(tf.maximum(v_losses1, v_losses2))

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        probs = tf.nn.softmax(logits)
        # Entropy calculated as crossentropy on itself
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        tf.print(entropy_loss, output_stream=sys.stdout)
        return policy_loss - self.entropy_c*entropy_loss

    def _logits_loss_PPO(self, actions_advantages_neglogprobs, logits):
        actions, advantages, neglogprobs_old = tf.split(actions_advantages_neglogprobs, 3, axis=-1)
        neglogprobs_old = tf.squeeze(neglogprobs_old, axis=-1)
        actions = tf.squeeze(tf.cast(actions, tf.int32), axis=-1)
        advantages = tf.squeeze(advantages, axis=-1)
        neglogprobs_new = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        ratio = tf.exp(neglogprobs_old - neglogprobs_new)  # note order of subtraction, due to *negative* log probabilities
        pg_loss1 = -advantages*ratio
        pg_loss2 = -advantages*tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
        # Calculate policy entropy - entropy calculated as crossentropy on same probability distribution
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        return policy_loss - self.entropy_c*entropy_loss


def main():
    parser = argparse.ArgumentParser(description='RL training parameters')
    parser.add_argument('-v', '--visual', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=5000)
    parser.add_argument('-sf', '--show_first', default=False, action='store_true')
    parser.add_argument('-l', '--local', default=False, action='store_true')
    parser.add_argument('-ts', '--total_steps', type=int, default=int(5e6))
    args = parser.parse_args()
    np.set_printoptions(precision=3)

    #random.seed(0)
    #tf.random.set_seed(0)

    # enable dynamic GPU memory allocation
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0
    if args.local:
        print('Training on local GPU, limit memory allocation')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
    sim_steps = 0
    batch_sz = args.batch_size
    # Initialize OpenAI Procgen environment 
    env = gym.make("procgen:procgen-bigfish-v0", num_levels=1, start_level=1, distribution_mode="easy") 
    with tf.Graph().as_default():
        #tf.compat.v1.disable_eager_execution()
        # set up tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/tensorboard/' + current_time
        tensorboard_callback = None #tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model = Model.Model(env.action_space.n)
        obs = env.reset()
        agent = Agent(model, args.total_steps)
        print('\nRandom Trajectories Cold Start...')
        rewards_history, _ = agent.train(
                env, updates=1, 
                batch_sz=batch_sz, 
                random_action=True, 
                show_visual=args.visual, 
                show_first=args.show_first,
                tb_callback=None)
        rewards_means = np.array([np.mean(rewards_history[:-1])])
        rewards_stds = np.array([np.std(rewards_history[:-1])])
        rewards_min = np.array([np.amin(rewards_history[:-1])])
        rewards_max = np.amax([np.amax(rewards_history[:-1])])
        graph = tf.compat.v1.get_default_graph()
        if False:
            print('Loading pre-trained weights~~~')
            agent.model.load_weights('weights/' + '20200219-165156_2800000')
        iter_count = 0
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open('logs/' + start_time + '.csv', mode='w') as csv_file:
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
                sim_steps += batch_sz
                rewards_means = np.append(rewards_means, np.mean(rewards_history[:-1]))
                rewards_stds = np.append(rewards_stds, np.std(rewards_history[:-1]))
                rewards_min = np.append(rewards_min, np.amin(rewards_history[:-1]))
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
                if(iter_count%10==0):
                    model.save_weights('weights/' + current_time + '_' + str(sim_steps), 
                            save_format='tf')
        print('Finished Training, testing now...')
        return


if __name__ == "__main__":
    main()



