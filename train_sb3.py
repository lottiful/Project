"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env_randomization.custom_hopper import *
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

if args.train is None:
    exit('Arguments required, run --help for more information')


def main():
    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    #noi avevamo learning rate = 0.0025
    #learning_rate_values = [1e-2, 5e-3, 1e-3, 5e-4]
    learning_rate_values = [0.00025]

    models_performances = []

    #da capire se poi vogliamo passarglieli quando facciamo il train da linea di comando o che altro fare

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='source', help='Specify training environment: source or target')
    args = parser.parse_args()

    rand_masses = True

    for learning_rate in learning_rate_values:
        if args.train == 'source':
            train_env = gym.make('CustomHopper-source-v0')
        else:
            train_env = gym.make('CustomHopper-target-v0')

        print('State space:', train_env.observation_space)  # state-space
        print('Action space:', train_env.action_space)  # action-space
        print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

        #PARAMETRI 
        #decidere quale tipo di domain randomization implementare
        rand_masses = True
        rand_angle = True
        randomization_range = 0.5
        #decidere se utilizzare un angolo specifico di partenza
        inclination_angle = 0
        #decidere se fare la DDR
        performance_threshold=50
        dynamic_rand = True

        n_episodes = 600000 #500000

        train_env.modify_rand_paramether(rand_masses, rand_angle, inclination_angle, randomization_range, dynamic_rand, performance_threshold)

        model = PPO('MlpPolicy', n_steps=1024, batch_size=128, learning_rate=learning_rate, env=train_env, verbose=1, device='cpu') #learning_rate=0.00025
        model.learn(total_timesteps=int(n_episodes))

        #If the model is trained with the randomization of the angle, reset the xml file with starting angle uqual to 0.
        if train_env.rand_angle is True:
            train_env.inclination_angle = 0
            train_env.modify_xml_for_inclination()

        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=30)
        model_performance = []
        model_performance.append(learning_rate)
        model_performance.append(mean_reward)
        model_performance.append(std_reward)

        models_performances.append(model_performance)
    
    print(models_performances)

    """
    wandb.init(project="calGTT", name="PPO_train")
    for i in range(len(train_env.performance_history)):
        wandb.log({"Reward": train_env.performance_history[i]})
    wandb.finish()
    """

    model.save("ppo_model_")

    """
    #If the model is trained with the domain randomization, it is saved into a different file.
    if train_env.rand_masses is False:
        print("entra nel false")
        model.save("ppo_model_")
    else:
        print("Entra nel true")
        model.save("ppo_model_UDR_")
    """

if __name__ == '__main__':
    main()