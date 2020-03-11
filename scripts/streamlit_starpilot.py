import tensorflow as tf
import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import datetime
import csv
import gym
from gym.wrappers.monitoring import video_recorder

import sys, os
sys.path.append('../')
import src
from src.agents import PPO_Agent as Agent
from src.models import A2C_SharedCNN as Model


def update_print_rewards(rewards_last, rewards_record):
    rewards_record['means'] = np.append(rewards_record['means'], 
            np.mean(rewards_last[:-1]))
    rewards_record['stds'] = np.append(rewards_record['stds'], 
            np.std(rewards_last[:-1]))
    rewards_record['max'] = np.append(rewards_record['max'], 
            np.amax(rewards_last[:-1]))
    print('Epoch mean reward: ')
    print(rewards_record['means'][-10:])
    print('Epoch std reward: ')
    print(rewards_record['stds'][-10:])
    print('Epoch max reward: ')
    print(rewards_record['max'][-10:])
    return rewards_record

def render_gym(st_object, img, episode, width=300):
    st_object.image(img, width=width)
    return 



# Uncomment below for deterministic debugging
#random.seed(0)
#tf.random.set_seed(0)

st.title('Deep Reinforcement Learning with Observational Dropout')

st.write('This is a *Streamlit* Demo of the Deep Reinforcement Learning policies with visualizations of some important training metrics')

parser = argparse.ArgumentParser(description='RL training parameters')
parser.add_argument('-v', '--visual', default=False, action='store_true')
parser.add_argument('-bs', '--batch_size', type=int, default=5000)
parser.add_argument('-sf', '--show_first', default=False, action='store_true')
parser.add_argument('-l', '--load_model_path', default=None)
parser.add_argument('-ts', '--total_steps', type=int, default=int(5e6))
parser.add_argument('-rv', '--record_video', default=False, action='store_true')
args = parser.parse_args()
np.set_printoptions(precision=3)

# enable dynamic GPU memory allocation
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0

# Initialize OpenAI Procgen environment 
env = gym.make("procgen:procgen-starpilot-v0", 
        num_levels=0, start_level=0, distribution_mode="easy") 
if args.record_video:
    video = video_recorder.VideoRecorder(env, base_path='../data/videos/recent_video')
    video.frames_per_sec = 60
    args.total_steps = args.batch_size  # only run one recorded batch
else:
    video = None

# Initialize Arrays
sim_steps = 0
rewards_record = {}
rewards_record['means'] = np.array([])
rewards_record['stds'] = np.array([])
rewards_record['max'] = np.array([])

# Initialize streamlit objects
st.header('Visualize Current Simulation Batch')
st.subheader('Training Batch Progress')
st_bar = st.progress(0)
st_object = st.empty()
st.text('Action Logits:')
ac_prob_plot = st.empty()
st.text('State Value Estimate:')
values = pd.DataFrame([[0.0]], columns=['State Value'])
value_plot = st.line_chart(values)
st.header('Training Metrics')
st.write('These metrics will update after each training batch above.')
st.subheader('Batch Reward Monitoring')
st.text('Keep track of batch aggregate reward statistics to gauge overall training progress.')
reward_pd = pd.DataFrame([[0.0, 0.0, 0.0]], columns=('Batch Reward Means',
    'Batch Reward Variance','Batch Maximum Reward'))
st_plot = st.line_chart(reward_pd)
st.subheader('Distribution of Chosen Actions')
st.text('Monitor to see if the policy collapses to only selecting a few actions repeatedly.')
action_dist = st.empty()
st.subheader('Batch Entropy')
st.text('Overall entropy of the action probabilities. Should decrease as the RL agent learns.')
entropy = pd.DataFrame([[0.0]], columns=('Entropy',))
entropy_plot = st.line_chart(entropy)
st.subheader('Advantage Distribution')
st.text("The Critic model's estimate of the advantages from the last batch.")
advantage_dist = st.empty()
st_dict = {'render_obj':st_object, 'render_fcn':render_gym, 'progress_bar':st_bar, 
        'ac_prob_plot':ac_prob_plot, 'value':value_plot, 'action_dist':action_dist, 
        'advantage_dist':advantage_dist, 'entropy_plot':entropy_plot}


with tf.Graph().as_default():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/tensorboard/' + current_time
    tensorboard_callback = None
    model = Model.Model(env.action_space.n)
    obs = env.reset()
    agent = Agent.Agent(model, args.total_steps)
    if args.load_model_path:
        print('Loading pre-trained weights~~~')
        print('Loading Model from File: {}'.format(args.load_model_path))
        agent.model.load_weights(args.load_model_path)
    else:
        print('\nRandom Trajectories Cold Start...')
        rewards_last, _ = agent.train(
                env, sim_steps,
                sim_batch_sz=args.batch_size, 
                random_action=True, 
                show_visual=args.visual, 
                show_first=args.show_first,
                tb_callback=None,
                v_recorder=video,
                streamlit=st_dict)
        st.balloons()
        rewards_record = update_print_rewards(rewards_last, rewards_record)
        new_rows = pd.DataFrame([[rewards_record['means'][-1], rewards_record['stds'][-1], rewards_record['max'][-1]]], 
                columns=('Batch Reward Means','Batch Reward Variance','Batch Maximum Reward'))
        st_plot.add_rows(new_rows)
    iter_count = 0
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open('../data/training_logs/' + start_time + '_' + \
            model.model_name + '.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['num_steps', 'num_episodes', 'rew_mean',
            'rew_std', 'entropy'])
        for _ in range(args.total_steps // args.batch_size + 1):
            iter_count += 1
            rewards_last, metrics = agent.train(
                    env, sim_steps, 
                    sim_batch_sz=args.batch_size, 
                    show_visual=args.visual, 
                    show_first=args.show_first,
                    tb_callback=tensorboard_callback,
                    v_recorder=video,
                    streamlit=st_dict)
            st.balloons()
            sim_steps += args.batch_size
            print('Total Sim Steps: {}'.format(sim_steps))
            print('Number of levels: {}'.format(len(rewards_last)))
            rewards_record = update_print_rewards(rewards_last, rewards_record)
            new_rows = pd.DataFrame([[rewards_record['means'][-1], rewards_record['stds'][-1], rewards_record['max'][-1]]], 
                    columns=('Batch Reward Means','Batch Reward Variance','Batch Maximum Reward'))
            st_plot.add_rows(new_rows)
            csv_writer.writerow([str(sim_steps),
                str(len(rewards_last)),
                str(rewards_record['means'][-1]),
                str(rewards_record['stds'][-1]),
                str(metrics['policy_entropy'])])
            if(iter_count%10 == 0):
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                model.save_weights('../data/log_weights/' + current_time + '_' + \
                        model.model_name + '_' + str(sim_steps), save_format='tf')
    if args.record_video:
        video.close()
    # Save the final run weights
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save_weights('../data/log_weights/' + current_time + '_' + \
            model.model_name + '_' + str(sim_steps), save_format='tf')
print('Finished Training...')

