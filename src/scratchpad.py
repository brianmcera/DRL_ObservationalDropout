"""
This scratchpad usess Streamlit which should be installed on your machine if
you follow the Insight Installation Instructions:

https://docs.google.com/presentation/d/1qo_MDz3iF0YRykuElF6I9WC4yAQIYzOA-GY16_NOuUM

Or by running:

pip install -r requirements.txt

from the top-level project folder.
"""

import tensorflow as tf
import streamlit as st
import numpy as np
import gym
import time
import sys
import os
import argparse
import csv
from tqdm import tqdm
import datetime
from procgen import interactive
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from PPO_Agent import Agent
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
import A2C_Shared_CNN as Model
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import custom_losses as Agent_Wrapper

@st.cache
def render_gym(st_object, env, episode, width=400):
    img = env.render(mode='rgb_array')
    caption = "Episode " + str(episode)
    st_object.image(img, caption = caption, width = width)
    return st_object

def streamlit_test(agent, env, streamlit_ob, num_steps=5000):
    total_reward = 0
    next_obs = env.reset().astype(np.float64)
    next_obs = (next_obs)/256.0
    prev_obs = next_obs.copy()
    for step in tqdm(range(num_steps)):
        combined_obs = np.concatenate((next_obs,prev_obs), axis=-1)
        prev_obs = next_obs.copy()
        render_gym(streamlit_ob, env, 0)
        action, _, _ = agent.model.action_value_neglogprob(combined_obs[None,:])
        next_obs, reward, done, _ = env.step(action)
        next_obs = next_obs.astype(np.float64)
        next_obs = (next_obs)/256.0

        total_reward += reward
        if done:
            break
    return reward

def main():
    st.title('Deep Reinforcement Learning with Observational Dropout')

    st.write('This is a *Streamlit* Demo of the Deep Reinforcement Learning policies with visualizations of some important training metrics')
    st_object = st.empty()

    parser = argparse.ArgumentParser(description='RL training parameters')
    parser.add_argument('-v', '--visual', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=5000)
    parser.add_argument('-sf', '--show_first', default=False, action='store_true')
    parser.add_argument('-l', '--local', default=False, action='store_true')
    parser.add_argument('-ts', '--total_steps', type=int, default=int(5e6))
    args = parser.parse_args()
    np.set_printoptions(precision=3)

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
    env = gym.make("procgen:procgen-starpilot-v0", num_levels=0, start_level=0, distribution_mode="easy") 
    with tf.Graph().as_default():
        # set up tensorboard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/tensorboard/' + current_time
        tensorboard_callback = None 
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
                tb_callback=None,
                sl_fcn=render_gym,
                sl_obj=st_object)
        streamlit_test(agent, env, st_object)
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
        with open('logs/' + start_time + '_' + model.model_name + '.csv', mode='w') as csv_file:
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
                if(iter_count%10 == 0):
                    model.save_weights('log_weights/' + current_time + '_' + model.model_name + '_' + str(sim_steps), 
                            save_format='tf')
        print('Finished Training, testing now...')
        return

def main2():
    st.line_chart(a2c_mean)
    st.line_chart(a2c_OD_mean)


    st.title('Deep Reinforcement Learning with Observational Dropout')

    st.write('This is a *Streamlit* Demo of the Deep Reinforcement Learning policies with visualizations of some important training metrics')

    #interactive.main()

    st_object = st.empty()
    env = gym.make('procgen:procgen-starpilot-v0')

    for episode in range(10):
        env.reset()
        j = 0
        done = False
        for step in tqdm(range(150)):
            j += 1
            render_gym(st_object, env, episode)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
            #time.sleep(0.02)
    env.close()

    st.subheader('A Numpy Array')

    st.write(np.random.randn(10, 10))

    st.subheader('A Graph!')

if __name__ == '__main__':
    main()

