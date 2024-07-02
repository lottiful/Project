import argparse

import torch
import gym

from env_test.custom_hopper import *
from agent import Agent, Policy, Policy_critic

from timeit import default_timer as timer
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=50000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=400, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	inclination_angle = 0
	env.modify_inclination(inclination_angle)

	#Training
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy_critic = Policy_critic(observation_space_dim, 1)
	agent = Agent(policy, policy_critic, device=args.device)

	#Reinforce -> critic = False; Actor-critic -> critic = True
	critic = True

	wandb.init(project="calGTT", name="AC_train")
	i = 0
	start_time = timer()

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over
			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward

			if critic == False:
				if done == True:
					agent.update_policy(critic)
			else:
				if ((i%30 == 0) and (i!=0)): #update the policy every 30 steps, regardless of the episode
					agent.update_policy(critic)
			i +=1

		wandb.log({"Reward_AC": train_reward})
		
	
	wandb.finish()

	end_time = timer()
	total_time = end_time-start_time
	print ('total time = ',total_time)

	torch.save(agent.policy.state_dict(), "modelAC.mdl")

	

if __name__ == '__main__':
	main()