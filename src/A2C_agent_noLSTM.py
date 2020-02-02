import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn import preprocessing, decomposition
import matplotlib.pyplot as plt
import datetime
import time
import gym
import pdb
import random
import argparse
from tqdm import tqdm

import A2C_model_noLSTM as A2C_model
import utils


class A2CAgent:
    def __init__(self, model, lr=1e-4, gamma=0.999, value_c=0.5, entropy_c=1e-2):
        self.model = model
        self.value_c = value_c
        self.entropy_c = entropy_c
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.img_mean = 0
        self.img_std = 1
        
        #compile model
        self.model = model
        self.model.compile(
                optimizer=self.optimizer,
                loss=[
                    self._logits_loss, # actor loss
                    self._value_loss   # critic loss
                    ])

    def train(self, env, batch_sz=15000, updates=1, show_visual=True, random_action=False, show_first=False):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,))
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        observations = np.concatenate((observations,observations), axis=-1)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = env.reset().astype(np.float64)
        next_obs = (next_obs-self.img_mean)/self.img_std
        prev_obs = next_obs.copy()
        print('\n\nRunning episodes...')
        first_run = True
        for update in range(updates):
            for step in tqdm(range(batch_sz)):
                combined_obs = np.concatenate((next_obs,prev_obs), axis=-1)
                prev_obs = next_obs
                observations[step] = combined_obs.copy()
                if(show_visual or (show_first and first_run)):
                    env.render()
                if(random_action):
                    _, values[step] = self.model.action_value(
                            combined_obs[None,:])
                    actions[step] = env.action_space.sample()
                else:
                    actions[step], values[step] = self.model.action_value(
                            combined_obs[None,:])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                next_obs = next_obs.astype(np.float64)
                next_obs = (next_obs-self.img_mean)/self.img_std

                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    _, next_value = self.model.action_value(
                            combined_obs[None,:])
                    #rewards[step] -= 20
                    if(not random_action):
                        # don't bootstrap first (random) run 
                        rewards[step] += next_value
                    ep_rewards.append(0.0)
                    next_obs = env.reset().astype(np.float64)
                    next_obs = (next_obs-self.img_mean)/self.img_std
                    prev_obs = next_obs.copy()
                    first_run = False
                    #logging.info("Episode: %03d, Reward: %03d" % (
                    #(ep_rewards) - 1, ep_rewards[-2]))
            
            # Handle bootstrapped Critic value for last (unfinished) run
            if(not random_action):
                _, next_value = self.model.action_value(combined_obs[None,:])
            else:
                next_value = np.array([0])

            returns, advs = self._returns_advantages(
                    rewards, dones, values, next_value)

            # normalize advantages for numerical stability
            advs -= np.mean(advs)
            advs /= np.std(advs) + 1e-3


            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Disregard incomplete runs at head/tail
            idx = np.argwhere(dones==1)
            first_index = idx[0][0]
            last_index = idx[-1][0]

            # Decimate observations (similar observations)
            skip = 1
            observations = observations[first_index+1:last_index:skip]
            returns = returns[first_index+1:last_index:skip]
            acts_and_advs = acts_and_advs[first_index+1:last_index:skip]

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            print('training on batch')
            if(random_action):
                for _ in range(1):
                    self.model.fit(observations, [acts_and_advs, returns], shuffle=True, batch_size=256, epochs=1)
            else:
                self.model.fit(observations, [acts_and_advs, returns], shuffle=True, batch_size=256, epochs=1)

            # Print out distribution of Actions, useful for debugging during training
            print('Action distribution for batch:')
            print(np.histogram(actions, bins=list(range(0,env.action_space.n)))[0])
            # Print out distribution of Rewards-Value 
            print('Advantage distribution for batch:')
            print(np.histogram(advs)[0]) 
            print('Advantage histogram bins:')
            print(np.histogram(advs)[1]) 

        if(random_action):
            num_channels = env.observation_space.shape[-1]
            self.img_mean = observations[...,:num_channels].mean(axis=(0,1,2), keepdims=True)[0]
            self.img_std = observations[...,:num_channels].std(axis=(0,1,2), keepdims=True)[0]
            pass
        return ep_rewards

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs.astype(np.float32)[None,:])
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # 'next value' is the bootstrapped returned value from the Critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1-dones[t])

        returns = returns[:-1] # all but last
        advantages = returns - values # advantage over Critic estimates
        return returns, advantages

    def _value_loss(self, returns, value):
        # this function calculates loss with value TD error
        return self.value_c*tf.keras.losses.mean_squared_error(returns,value)

    def _logits_loss(self, actions_and_advantages, logits):
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        probs = tf.nn.softmax(logits)
        # trick here - entropy can be calculated as crossentropy on itself
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)

        return policy_loss - self.entropy_c*entropy_loss

def main():
    parser = argparse.ArgumentParser(description='RL training parameters')
    parser.add_argument('-v', '--visual', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=15000)
    parser.add_argument('-sf', '--show_first', default=False, action='store_true')
    parser.add_argument('-l', '--local', default=False, action='store_true')
    args = parser.parse_args()

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


    np.set_printoptions(precision=2)

    sim_steps = 0
    batch_sz = args.batch_size

    # set up tensorboard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # initialize environment and deep model
    env = gym.make("procgen:procgen-starpilot-v0", num_levels=0, distribution_mode="easy") # define chosen environment here
    model = A2C_model.Model(env.action_space.n)
    obs = env.reset()
    agent = A2CAgent(model)
    #rewards_sum = agent.test(env)
    #print(rewards_sum)
    
    rewards_history = agent.train(env, updates=1, batch_sz=batch_sz, random_action=True, show_visual=args.visual, show_first=args.show_first)
    rewards_means = np.array([np.mean(rewards_history[:-1])])
    rewards_stds = np.array([np.std(rewards_history[:-1])])
    graph = tf.compat.v1.get_default_graph()
    graph.finalize()

    #agent.model.load_weights('pretrained_examples/' + '20200131-061824_2515000')
    iter_count = 0
    while True:
        iter_count += 1
        rewards_history = agent.train(env, batch_sz=batch_sz, show_visual=args.visual, show_first=args.show_first)
        rewards_means = np.append(rewards_means, np.mean(rewards_history[:-1]))
        rewards_stds = np.append(rewards_stds, np.std(rewards_history[:-1]))
        plt.plot(rewards_means)
        plt.plot(np.array(rewards_means)+np.array(rewards_stds))
        plt.plot(np.array(rewards_means)-np.array(rewards_stds))
        plt.draw()
        # plt.pause(1e-3)
        print('Total Sim Steps: ' + str(sim_steps))
        print('Number of levels: ' + str(len(rewards_history)))
        print('Epoch mean reward: ')
        print(rewards_means)
        print('Epoch std reward: ')
        print(rewards_stds)
        sim_steps += batch_sz
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if(iter_count%10==0):
            model.save_weights('weights/' + current_time + '_' + str(sim_steps), save_format='tf')
    print('Finished Training, testing now...')

    return


if __name__ == "__main__":
    main()



