"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env_test.custom_hopper import *
from agent import Agent, Policy, Policy_critic

from timeit import default_timer as timer
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy_critic = Policy_critic(observation_space_dim, 1)
	agent = Agent(policy, policy_critic, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	#Reinforce -> critic = False; Actor-critic -> critic = True
	critic = True
	i = 0

	#sistemare il nome ogni volta
	wandb.init(project="calGTT", name="AC_train")

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state
		start_time = timer()

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
				if (i%50 == 0) and (i!=0): #update the policy every 50 steps, a prescindere dall'episodio (i cumulativa esterna)
					agent.update_policy(critic)
			i +=1

		wandb.log({"Reward": train_reward})
		
	
	wandb.finish()
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

		end_time = timer()
		total_time = end_time-start_time
		print ('total time = ',total_time)

	torch.save(agent.policy.state_dict(), "model.mdl")

	

if __name__ == '__main__':
	main()