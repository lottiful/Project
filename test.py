"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env_test.custom_hopper import *
from agent import Agent, Policy, Policy_critic

import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=50, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	inclination_angle = 0
	env.modify_inclination(inclination_angle)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	args.model = "model.mdl"
	policy = Policy(observation_space_dim, action_space_dim)
	policy_critic = Policy_critic(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	
	#Cambiare il name in base a cosa si sta testando, così rimangono tutte le tabelle con i nomi giusti.
	wandb.init(project="calGTT", name="REINFORCEb0_test")

	agent = Agent(policy, policy_critic, device=args.device)

	list_rewards = []

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward
		
		wandb.log({"Reward": test_reward})
		list_rewards.append(test_reward)
		print(f"Episode: {episode} | Return: {test_reward}")
	
	mean_reward = np.mean(list_rewards)
	std_reward = np.std(list_rewards)

	print(f"mean reward = {mean_reward}")
	print(f"std reward = {std_reward}")

	wandb.finish()

	wandb.init(project="calGTT", name="mean")
	for i in range(50):
		wandb.log({"Reward": mean_reward})
	wandb.finish()

if __name__ == '__main__':
	main()