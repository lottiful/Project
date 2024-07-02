import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

#compute the discounted rewards
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#Actor network
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):

        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        
        return normal_dist

#Critic network
class Policy_critic(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        action_mean = self.fc3_critic_mean(x_critic)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        
        return action_mean


class Agent(object):
    def __init__(self, policy, policy_critic, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.policy_critic = policy_critic.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(policy_critic.parameters(), lr=0.00025)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, critic):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        loss = 0

        discounted_rewards = []

        #Parameter
        b = 0 #without baseline
        # b = 20 #with baseline
        adaptive_baseline = False #baseline equal to cumulative mean of rewards

        discounted_rewards = discount_rewards(rewards, self.gamma)
        actor_loss = 0
        critic_loss = 0

        # Policy gradient update
        if critic is False: #REINFORCE
            tot_r = 0
            i = 1
            for log_prob, actor_G, r in zip(action_log_probs, discounted_rewards, rewards):
                if adaptive_baseline is True:
                    tot_r += r
                    b = tot_r / i
                loss += -log_prob * (actor_G - b)
                i +=1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if critic is True: #AC
            j = 0
            for log_prob, state, next_state in zip(action_log_probs, states, next_states):
                v_t =  self.policy_critic(state)
                v_next = (1-done[j])*self.policy_critic(next_state)

                #actor
                actor_G = rewards[j] + self.gamma * v_next
                actor_loss += -log_prob * (actor_G - v_t).detach()
                #critic
                critic_loss += ((rewards[j] + self.gamma* v_next).detach() - v_t).pow(2)

                j+=1

            actor_loss = actor_loss/j
            critic_loss = critic_loss/j

            self.optimizer.zero_grad()
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
            actor_loss.backward()
            self.optimizer.step()
            

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        return 


    def get_action(self, state, evaluation=False):

        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

