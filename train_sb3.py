import gym
from env_randomization.custom_hopper import *
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--episodes', default=1000, type=int)
    return parser.parse_args()

args = parse_args()

if args.train is None:
    exit('Arguments required, run --help for more information')


def main():

    #Parameters
    rand_masses = True #if True -> UDR for masses
    rand_angle = False #if True -> UDR for inclination angle
    randomization_range = 1 #Uniform range

    inclination_angle = 0 
      
    performance_threshold=50
    dynamic_rand = False #if True -> DDR for masses and inclination angle

    n_timesteps = 1300000
    learning_rate_value = 0.00025

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='source', help='Specify training environment: source or target')
    args = parser.parse_args()


    if args.train == 'source':
        train_env = gym.make('CustomHopper-source-v0')
    else:
        train_env = gym.make('CustomHopper-target-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    train_env.initialize_parameter(rand_masses, rand_angle, inclination_angle, randomization_range, dynamic_rand, performance_threshold)

    model = PPO('MlpPolicy', n_steps=1024, batch_size=128, learning_rate=learning_rate_value, env=train_env, verbose=1, device='cpu')
    model.learn(total_timesteps=int(n_timesteps))

    #If the model is trained with the randomization of the angle, reset the xml file with starting angle equal to 0.
    if train_env.inclination_angle != 0:
        train_env.inclination_angle = 0
        train_env.modify_xml_for_inclination()

    #evaluation of the model
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=30)
    model_performance = []
    model_performance.append(mean_reward)
    model_performance.append(std_reward)
    print(model_performance)

    model.save("ppo_model_UDR")
 
    wandb.init(project="calGTT", name="train_noUDR_target")
    for i in range(len(train_env.performance_history)):
        wandb.log({"Train_Reward": train_env.performance_history[i]})
    wandb.finish()

if __name__ == '__main__':
    main()