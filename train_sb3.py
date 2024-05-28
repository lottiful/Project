"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--episodes', default=1000, type=int)#default=100_000
    return parser.parse_args()

args = parse_args()

#if args.train is None or args.source_log_path is None or args.target_log_path is None:
if args.train is None:
    exit('Arguments required, run --help for more information')


N_ENVS = os.cpu_count()
MAX_EPS = args.episodes
ENV_EPS = int(np.ceil(MAX_EPS / N_ENVS))

def main():
    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='source', help='Specify training environment: source or target')
    args = parser.parse_args()

    source_env = gym.make('CustomHopper-source-v0')
    target_env = gym.make('CustomHopper-target-v0')

    print('State space:', source_env.observation_space)  # state-space
    print('Action space:', source_env.action_space)  # action-space
    print('Dynamics parameters:', source_env.get_parameters())  # masses of each link of the Hopper

    if args.train == 'source':
        train_env = source_env  # sets the train to source
    else:
        train_env = target_env


    model = PPO('MlpPolicy', n_steps=1024, batch_size=128, n_epochs=10, learning_rate=0.00025, env=train_env, verbose=1, device='cpu') #learning_rate=0.00025
    model.learn(total_timesteps=int(5000)) # total_timesteps=int(1e10)

    model.save("ppo_model_")

if __name__ == '__main__':
    main()