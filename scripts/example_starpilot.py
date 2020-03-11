import tensorflow as tf
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


# Uncomment below for deterministic debugging
#random.seed(0)
#tf.random.set_seed(0)

parser = argparse.ArgumentParser(description='RL training parameters')
parser.add_argument('-v', '--visual', default=False, action='store_true')
parser.add_argument('-bs', '--batch_size', type=int, default=10000)
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
    args.total_steps = args.batch_size  # only run two recorded batches
else:
    video = None

# Initialize Arrays
sim_steps = 0
rewards_record = {}
rewards_record['means'] = np.array([])
rewards_record['stds'] = np.array([])
rewards_record['max'] = np.array([])

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
        _, _ = agent.train(
                env, sim_steps,
                sim_batch_sz=args.batch_size, 
                random_action=True, 
                show_visual=args.visual, 
                show_first=args.show_first,
                tb_callback=None,
                v_recorder=video)
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
                    v_recorder=video)
            sim_steps += args.batch_size
            print('Total Sim Steps: {}'.format(sim_steps))
            print('Number of levels: {}'.format(len(rewards_last)))
            rewards_record = update_print_rewards(rewards_last, rewards_record)
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

