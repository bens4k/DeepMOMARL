from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.fc3 = nn.Linear(hidden, action_dim)

	def forward(self, state):
		x = torch.relu(self.fc1(state))
		x = torch.relu(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_dim + action_dim, hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.fc3 = nn.Linear(hidden, 1)
		
	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class OrnsteinUhlenbeckNoise:
	def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
		self.size = size
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(self.size) * self.mu

	def reset(self):
		self.state = np.ones(self.size) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
		self.state += dx
		return self.state
	
class DDPG:
	def __init__(self, state_dim, action_space: gym.spaces.Box, params, device: torch.device):
		self.device = device
		self.action_dim = action_space.shape[0]
		self.action_space = action_space
		
		# Initialize models and move them to the specified device
		self.actor = Actor(state_dim, self.action_dim, hidden=params["hidden_dim"]).to(self.device)
		self.critic = Critic(state_dim, self.action_dim, hidden=params["hidden_dim"]).to(self.device)
		self.target_actor = Actor(state_dim, self.action_dim, hidden=params["hidden_dim"]).to(self.device)
		self.target_critic = Critic(state_dim, self.action_dim, hidden=params["hidden_dim"]).to(self.device)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params["actor_lr"])
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params["critic_lr"])
		
		self.replay_buffer = []
		self.replay_buffer_size = params["replay_buffer_size"]  # Size of the replay buffer
		self.gamma = params["discount"]  # Discount factor for future rewards
		self.tau = params["tau"]  # Soft update parameter for target networks
		self.batch_size = params["batch_size"] # Batch size for training
		# noise
		self.noise = OrnsteinUhlenbeckNoise(self.action_dim, mu=params["ou_noise_mu"], theta=params["ou_noise_theta"], sigma=params["ou_noise_sigma"]) 
		self.noise_std_dev = params["noise_std_dev"] # Standard deviation for Gaussian noise

	def select_action(self, state):
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		action_np = self.actor(state).cpu().data.numpy().flatten()
		
		# gaussian noise
		#action_np = action_np + np.random.normal(0, self.noise_std_dev, size=action_np.shape) 
		
		# Ornstein-Uhlenbeck noise
		#action_np = action_np + self.noise.sample()

		# Scale action using the action space bounds
		scaled_action = self.action_space.low + (action_np + 1.0) * 0.5 * (self.action_space.high - self.action_space.low)
		return scaled_action
		#return np.clip(scaled_action, self.action_space.low, self.action_space.high)
	
	def step(self, state, action, next_state, reward, done):
		if len(self.replay_buffer) > self.replay_buffer_size:
			self.replay_buffer.pop(0)
		
		# Store transitions in replay buffers
		self.replay_buffer.append((state, action, reward, next_state))

		# Update agents
		self.update(self.batch_size)

	def update(self, batch_size):
		if len(self.replay_buffer) < batch_size:
			return
		
		batch = random.sample(self.replay_buffer, batch_size)
		states, actions, rewards, next_states = zip(*batch)

		# Convert to tensors and move to the specified device
		states = torch.FloatTensor(np.array(states)).to(self.device)
		actions = torch.FloatTensor(np.array(actions)).to(self.device)
		rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
		next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

		# Update Critic
		next_actions = self.target_actor(next_states)
		target_Q = self.target_critic(next_states, next_actions)
		expected_Q = rewards.view(-1, 1) + self.gamma * target_Q.detach()
		
		Q_values = self.critic(states, actions)
		
		critic_loss = nn.MSELoss()(Q_values, expected_Q)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update Actor
		actor_loss = -self.critic(states, self.actor(states)).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Soft update targets
		for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class DDPG_Centralized(DDPG):
	def __init__(self, state_dim_actor, state_dim_critic, action_space: gym.spaces.Box, params, device: torch.device):
		super().__init__(state_dim_actor, action_space, params, device)
		self.state_dim_critic = state_dim_critic

		# Redefine critic to handle the larger state dimension
		self.critic = Critic(state_dim_critic, self.action_dim, hidden=params["hidden_dim"]).to(self.device)
		self.target_critic = Critic(state_dim_critic, self.action_dim, hidden=params["hidden_dim"]).to(self.device)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params["critic_lr"])

	def update(self, batch_size):
		if len(self.replay_buffer) < batch_size:
			return
		
		batch = random.sample(self.replay_buffer, batch_size)
		states_actor, states_critic, actions, rewards, next_states_actor, next_states_critic = zip(*batch)

		# Convert to tensors and move to the specified device
		states_actor = torch.FloatTensor(np.array(states_actor)).to(self.device)
		states_critic = torch.FloatTensor(np.array(states_critic)).to(self.device)
		actions = torch.FloatTensor(np.array(actions)).to(self.device)
		rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
		next_states_actor = torch.FloatTensor(np.array(next_states_actor)).to(self.device)
		next_states_critic = torch.FloatTensor(np.array(next_states_critic)).to(self.device)

		# Update Critic
		next_actions = self.target_actor(next_states_actor)
		target_Q = self.target_critic(next_states_critic, next_actions)
		expected_Q = rewards.view(-1, 1) + self.gamma * target_Q.detach()
		
		Q_values = self.critic(states_critic, actions)
		
		critic_loss = nn.MSELoss()(Q_values, expected_Q)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update Actor
		actor_loss = -self.critic(states_critic, self.actor(states_actor)).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Soft update targets
		for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def step(self, state_actor, state_critic, action, next_state_actor, next_state_critic, reward, done):
		# Store transitions in replay buffer
		self.replay_buffer.append((state_actor, state_critic, action, reward, next_state_actor, next_state_critic))

		# Update agents
		self.update(self.batch_size)

def test_ddpg():
	# Create a simple environment
	env = gym.make("Pendulum-v1", render_mode="human")
	state_dim = env.observation_space.shape[0]
	action_space = env.action_space
	#print(f"action_space = {action_space}") 
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize DDPG agent
	agent = DDPG(state_dim, action_space, device)

	# Training loop
	num_episodes = 100
	for episode in range(num_episodes):
		state = env.reset()[0]
		episode_reward = 0
		done = False

		while not done:
			# Select action
			action = agent.select_action(state)
			#print(f"action = {action}")
			# Step environment
			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated

			# Store transition and train
			agent.step(state, action, next_state, reward, done)

			state = next_state
			episode_reward += reward

		print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

#if __name__ == "__main__":
#    test_ddpg()