from collections import deque
from dataclasses import dataclass
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import gymnasium as gym

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
	def __init__(self, n_observations, hidden, n_actions):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_observations, hidden)
		self.layer2 = nn.Linear(hidden, hidden)
		self.layer3 = nn.Linear(hidden, n_actions)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)

class ReplayMemory(object):

	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
	
	def clear(self):
		self.memory.clear()

class DDQN:
	def __init__(self, n_observations, action_space, params, device = "cuda"):
		if not isinstance(action_space, gym.spaces.Discrete):
			raise ValueError("Action space must be discrete.")
			
		n_actions = action_space.n

		self.action_space = action_space
		self.device = device
		self.batch_size = params["batch_size"]

		self.gamma = params["gamma"]
		self.learning_rate = params["learning_rate"]
		
		self.tau = params["tau"]
		self.target_update = params["target_update"]
		self.is_training = True

		self.eps_start = params["eps_start"]
		self.eps_end = params["eps_end"]
		self.eps_decay = params["eps_decay"]
		
		self.steps_done = 0 

		self.policy_net = DQN(n_observations, params["hidden_size"], n_actions).to(self.device)
		self.target_net = DQN(n_observations, params["hidden_size"], n_actions).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())

		self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
		self.memory = ReplayMemory(params["memory_capacity"])
		
		# used to store (state, action, next_state) transitions before having a reward
		self.tmp_memory = [] 

		# debugging
		self.q_values = []
		self.loss_values = []
	
	def reset(self):
		self.memory.clear()
		self.steps_done = 0 

	def save(self, path):
		torch.save(self.policy_net.state_dict(), path+".pth")
	
	def load(self, path):
		self.is_training = False
		self.policy_net.load_state_dict(torch.load(path+".pth"))
		self.policy_net.eval()  # Set to evaluation mode
		self.epsilon = 0.01

	def select_action(self, state):
		action = None
		eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
		self.steps_done += 1

		if np.random.rand() <= eps_threshold:
			action = self.action_space.sample().item()
		else:
			with torch.no_grad():
				output = self.policy_net(torch.Tensor(state).to(self.device))
				action = output.argmax().cpu().numpy().item()
				self.q_values.append(output.mean().item())

		return action
	
	def step(self, state, action, next_state, reward, done):
		if self.is_training:
			# Ensure the parameters are stored as torch tensors
			state = torch.tensor(state, dtype=torch.float32)
			action = torch.tensor([action], dtype=torch.int64)  # Action should be a single integer
			next_state = torch.tensor(next_state, dtype=torch.float32)
			reward = torch.tensor([reward], dtype=torch.float32)
			done = torch.tensor([done], dtype=torch.bool)

			self.memory.push(state, action, next_state, reward, done)
			self.update()

	def update(self):
		self.__optimize_model()
		self.__soft_update()

	def __optimize_model(self):	
		if len(self.memory) < self.batch_size:
			return
		transitions = self.memory.sample(self.batch_size)
		batch = Transition(*zip(*transitions))
		
		# Compute a mask of non-final states and concatenate the batch elements
		non_final_next_states_list = [next_state.unsqueeze(0) for next_state, is_done in zip(batch.next_state, batch.done) if not is_done]
		if len(non_final_next_states_list) == 0:
			return
		
		non_final_next_states = torch.cat(non_final_next_states_list).to(self.device)
		non_final_mask = ~torch.cat(batch.done).to(self.device)

		state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(self.device)
		action_batch = torch.cat([a.unsqueeze(0) for a in batch.action]).to(self.device)
		reward_batch = torch.cat([r.unsqueeze(0) for r in batch.reward]).to(self.device)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = self.policy_net(state_batch).gather(1, action_batch)
		
		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1).values
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.batch_size, device=self.device)
		with torch.no_grad():
			next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze(1)

		# Compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
		self.loss_values.append(loss.item())
		
		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
		self.optimizer.step()
	
	def __soft_update(self):
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
		self.target_net.load_state_dict(target_net_state_dict)

	def __hard_update(self):
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]
		self.target_net.load_state_dict(target_net_state_dict)