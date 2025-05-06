from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

class CNNFeatureExtractor(nn.Module):
	def __init__(self, input_shape, output_dim=64, is_image = False):
		super(CNNFeatureExtractor, self).__init__()
		# Determine if we need CNN or just a flatten layer
		self.input_shape = input_shape
		self.is_image = is_image #len(input_shape) == 3  # Check if input is an image (height, width, channels)
		
		if self.is_image:
			c, h, w = input_shape
			# Simple CNN for feature extraction
			self.cnn = nn.Sequential(
				nn.Conv2d(c, 32, kernel_size=8, stride=4),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Flatten()
			)
			
			# Calculate the CNN output size
			with torch.no_grad():
				dummy_input = torch.zeros(1, c, h, w)
				cnn_output = self.cnn(dummy_input)
				cnn_output_size = cnn_output.shape[1]
			
			self.fc = nn.Linear(cnn_output_size, output_dim)
		else:
			# For non-image array states, just flatten and use FC layer
			flat_size = np.prod(input_shape)
			self.cnn = nn.Flatten()
			self.fc = nn.Linear(flat_size, output_dim)
		
	def forward(self, x):
		# Input shape handling
		if self.is_image and len(x.shape) == 3:  # Add batch dimension if missing
			x = x.unsqueeze(0)
			
		features = self.cnn(x)
		return torch.tanh(self.fc(features))

class ContinuousActor(nn.Module):
	def __init__(self, obs_shape, action_dim, hidden_dim=64):
		super(ContinuousActor, self).__init__()
		# Feature extractor for complex observations
		self.features = CNNFeatureExtractor(obs_shape, hidden_dim)
		
		# Policy head
		self.fc = nn.Linear(hidden_dim, hidden_dim)
		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Parameter(torch.zeros(action_dim))
		
		# Initialize weights
		nn.init.orthogonal_(self.fc.weight, gain=np.sqrt(2))
		nn.init.orthogonal_(self.mean.weight, gain=0.01)
		
	def forward(self, state):
		x = self.features(state)
		x = torch.tanh(self.fc(x))
		mean = self.mean(x)
		
		# Bound log_std for stability
		log_std = torch.clamp(self.log_std, -20, 2)
		
		return mean, log_std

class DiscreteActor(nn.Module):
	def __init__(self, obs_shape, action_dim, hidden_dim=64):
		super(DiscreteActor, self).__init__()
		# Feature extractor for complex observations
		self.features = CNNFeatureExtractor(obs_shape, hidden_dim)
		
		# Policy head
		self.fc = nn.Linear(hidden_dim, hidden_dim)
		self.logits = nn.Linear(hidden_dim, action_dim)
		
		# Initialize weights
		nn.init.orthogonal_(self.fc.weight, gain=np.sqrt(2))
		nn.init.orthogonal_(self.logits.weight, gain=0.01)
		
	def forward(self, state):
		x = self.features(state)
		x = torch.tanh(self.fc(x))
		action_logits = self.logits(x)
		
		return action_logits

class Critic(nn.Module):
	def __init__(self, obs_shape, hidden_dim=64):
		super(Critic, self).__init__()
		# Feature extractor for complex observations
		self.features = CNNFeatureExtractor(obs_shape, hidden_dim)
		
		# Value head
		self.fc = nn.Linear(hidden_dim, hidden_dim)
		self.value = nn.Linear(hidden_dim, 1)
		
		# Initialize weights
		nn.init.orthogonal_(self.fc.weight, gain=np.sqrt(2))
		nn.init.orthogonal_(self.value.weight, gain=1.0)
		
	def forward(self, state):
		x = self.features(state)
		x = torch.tanh(self.fc(x))
		value = self.value(x)
		return value

class RolloutMemory:
	def __init__(self):
		self.states = []
		self.actions = []
		self.logprobs = []
		self.rewards = []
		self.values = []
		self.dones = []

	def push(self, state, action, reward, logprob, value, done):
		self.states.append(state)
		self.actions.append(action)
		self.logprobs.append(logprob)
		self.rewards.append(reward)
		self.values.append(value)
		self.dones.append(done)
	
	def clear(self):
		self.states.clear()
		self.actions.clear()
		self.logprobs.clear()
		self.rewards.clear()
		self.values.clear()
		self.dones.clear()
	
	def __len__(self):
		return len(self.states)

class CentralizedCriticRolloutMemory:
	def __init__(self):
		self.states = []
		self.actions = []
		self.logprobs = []
		self.rewards = []
		self.values = []
		self.dones = []
		self.critic_states = []

	def push(self, state, action, reward, logprob, value, done, critic_state):
		self.states.append(state)
		self.actions.append(action)
		self.logprobs.append(logprob)
		self.rewards.append(reward)
		self.values.append(value)
		self.dones.append(done)
		self.critic_states.append(critic_state)
	
	def clear(self):
		self.states.clear()
		self.actions.clear()
		self.logprobs.clear()
		self.rewards.clear()
		self.values.clear()
		self.dones.clear()
		self.critic_states.clear()
	
	def __len__(self):
		return len(self.states)

class PPO:
	def __init__(self, observation_shape : np.ndarray, action_space : gym.spaces.Space, params, device="cuda"):
		
		self.device = torch.device(device)
		self.obs_shape = observation_shape

		# Determine if action space is discrete or continuous
		self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
		
		if self.is_discrete:
			self.action_dim = action_space.n
			self.actor = DiscreteActor(self.obs_shape, self.action_dim, params["hidden_dim"]).to(self.device)
		else:
			self.action_dim = action_space.shape[0]
			# Check if action space is bounded
			if np.isfinite(action_space.low).all():
				self.action_low = action_space.low
			if np.isfinite(action_space.high).all():
				self.action_high = action_space.high   
			# Initialize networks
			self.actor = ContinuousActor(self.obs_shape, self.action_dim, params["hidden_dim"]).to(self.device)
			
		self.critic = Critic(self.obs_shape, params["hidden_dim"]).to(self.device)
		
		# Initialize optimizer
		self.optimizer = optim.Adam([
			{'params': self.actor.parameters(), 'lr': params["lr"]},
			{'params': self.critic.parameters(), 'lr': params["lr"]}
		])
		
		# Set hyperparameters
		self.gamma = params["gamma"]
		self.gae_lambda = params["gae_lambda"]
		self.clip_ratio = params["clip_ratio"]
		self.target_kl = params["target_kl"]
		self.value_coef = params["value_coef"]
		self.entropy_coef = params["entropy_coef"]
		self.update_epochs = params["update_epochs"]
		self.horizon = params["horizon"]
		self.minibatch_size = params["minibatch_size"]
		self.max_grad_norm = params["max_grad_norm"]
		self.reward_scale = params["reward_scale"]
		
		# Initialize tracking variables
		self.episode_rewards = []
		self.memory = RolloutMemory()
		
	def preprocess_state(self, state):
		"""Convert numpy state to torch tensor with proper shape"""
		if isinstance(state, np.ndarray):
			# Convert to tensor
			state_tensor = torch.FloatTensor(state).to(self.device)
		elif isinstance(state, torch.Tensor):
			state_tensor = state
		else:
			raise ValueError("expected a numpy ndarray or torch tensor as state")
		
		# Handle shape for image inputs
		if len(self.obs_shape) == 3:  # Image observation
			# Check if batch dimension is needed
			if len(state_tensor.shape) == 3:
				state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
		else:
			# For flat observations, always ensure there's a batch dimension
			if len(state_tensor.shape) == len(self.obs_shape):
				state_tensor = state_tensor.unsqueeze(0)
				
		return state_tensor
	
	def scale_action(self, action_np):
		is_scalar = np.isscalar(action_np) or (isinstance(action_np, np.ndarray) and action_np.size == 1)
		
		if is_scalar:
			lower_bound_isscalar = np.isscalar(self.action_low) or (isinstance(self.action_low, np.ndarray) and self.action_low.size == 1)
			upper_bound_isscalar = np.isscalar(self.action_high) or (isinstance(self.action_high, np.ndarray) and self.action_high.size == 1)

			has_lower_bound = hasattr(self, 'action_low') and lower_bound_isscalar and self.action_low > float('-inf')
			has_upper_bound = hasattr(self, 'action_high') and upper_bound_isscalar and self.action_high < float('inf')

			# Handle scalar action
			action = float(action_np)
			if has_lower_bound and has_upper_bound:
				# Fully bounded: map from [-1, 1] to [low, high]
				scaled_action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
				#scaled_action = np.clip(scaled_action, self.action_low, self.action_high)
				#print(f"action={action}, scaled_action={scaled_action}")
			elif has_lower_bound:
				# Lower-bounded only: map from (-∞, ∞) to [low, ∞)
				scaled_action = self.action_low + np.exp(action)
			elif has_upper_bound:
				# Upper-bounded only: map from (-∞, ∞) to (-∞, high]
				scaled_action = self.action_high - np.exp(action)
			else:
				# Unbounded: no transformation needed
				scaled_action = action
			return np.array([scaled_action])
		else:
			# Handle different action space types
			scaled_action = np.zeros_like(action_np)
			
			# Process each action dimension separately to handle mixed bounds
			for i in range(len(action_np)):
				# Check bounds for this dimension
				has_lower_bound = hasattr(self, 'action_low') and self.action_low[i] > float('-inf')
				has_upper_bound = hasattr(self, 'action_high') and self.action_high[i] < float('inf')
				if has_lower_bound and has_upper_bound:
					# Fully bounded: map from [-1, 1] to [low, high]
					scaled_action[i] = self.action_low[i] + (action_np[i] + 1.0) * 0.5 * (self.action_high[i] - self.action_low[i])
					#scaled_action[i] = np.clip(scaled_action[i], self.action_low[i], self.action_high[i])
				
				elif has_lower_bound:
					# Lower-bounded only: map from (-∞, ∞) to [low, ∞)
					# Using exponential transformation: low + exp(action)
					scaled_action[i] = self.action_low[i] + np.exp(action_np[i])
				
				elif has_upper_bound:
					# Upper-bounded only: map from (-∞, ∞) to (-∞, high]
					# Using negative exponential: high - exp(action)
					scaled_action[i] = self.action_high[i] - np.exp(action_np[i])
				
				else:
					# Unbounded: no transformation needed
					scaled_action[i] = action_np[i]
			return scaled_action
	
	def select_action(self, obs):
		state_tensor = self.preprocess_state(obs)
		value = self.critic(state_tensor).squeeze().cpu().item()

		with torch.no_grad():
			if self.is_discrete:
				# Discrete action space
				logits = self.actor(state_tensor)
				dist = Categorical(logits=logits)
				action = dist.sample()
				log_prob = dist.log_prob(action)
				action_np = action.cpu().item()  # Convert to scalar
				return action_np, log_prob.item(), value
			else:
				# Continuous action space
				mean, log_std = self.actor(state_tensor)
				std = torch.exp(log_std)
				dist = Normal(mean, std)
				action = dist.sample()
				log_prob = dist.log_prob(action).sum(dim=-1)
				
				# Scale action to environment's action space
				action_np = action.squeeze().cpu().numpy()
				scaled_action = self.scale_action(action_np)
				
				return scaled_action, log_prob.item(), value
	
	def compute_gae(self, rewards, values, next_value, dones):
		values = values + [next_value]
		advantages = []
		gae = 0
		
		for t in reversed(range(len(rewards))):
			delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
			gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
			advantages.insert(0, gae)
			
		returns = [adv + val for adv, val in zip(advantages, values[:-1])]
		advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
		
		return returns, advantages
	
	def step(self, state, action, next_state, reward, log_prob, value, done):
		self.memory.push(state, action, reward, log_prob, value, done)
		
		if len(self.memory.states) >= self.horizon:
			self.update(next_state)

	def update(self, next_state):	
		scaled_rewards = np.array(self.memory.rewards) * self.reward_scale
		
		# Get final value estimate
		state_tensor = self.preprocess_state(next_state)

		with torch.no_grad():
			next_value = self.critic(state_tensor).squeeze().cpu().item()
		
		# Compute returns and advantages
		returns, advantages = self.compute_gae(scaled_rewards, self.memory.values, next_value, self.memory.dones)
		
		rollout_data = {
			'states': self.memory.states,  # Keep as list for varying array shapes
			'actions': np.array(self.memory.actions),
			'log_probs': np.array(self.memory.logprobs),
			'returns': np.array(returns),
			'advantages': np.array(advantages)
		}

		self.update_policy(rollout_data)
		self.memory.clear()

	def prepare_batch(self, states, indices):
		"""Prepare a batch of states with proper tensor dimensions"""
		# Select the states for this batch
		batch_states = [states[i] for i in indices]
		
		# Convert to tensors and stack
		if len(self.obs_shape) == 3:  # Image observations
			return torch.stack([self.preprocess_state(s).squeeze(0) for s in batch_states]).to(self.device)
		else:
			# For vector observations
			return torch.FloatTensor(np.array(batch_states)).to(self.device)
	
	def update_policy(self, rollout_data):
		states = rollout_data['states']
		actions = rollout_data['actions']
		old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
		returns = torch.FloatTensor(rollout_data['returns']).unsqueeze(1).to(self.device)
		advantages = torch.FloatTensor(rollout_data['advantages']).unsqueeze(1).to(self.device)
		
		# Convert actions to appropriate format
		if self.is_discrete:
			actions = torch.LongTensor(actions).to(self.device)
		else:
			actions = torch.FloatTensor(actions).to(self.device)
			
		# Make sure old_log_probs has right shape
		if old_log_probs.dim() == 1:
			old_log_probs = old_log_probs.unsqueeze(1)
		
		# Update policy for multiple epochs
		for _ in range(self.update_epochs):
			# Generate random indices
			indices = np.random.permutation(len(states))
			
			# Update in minibatches
			for start_idx in range(0, len(states), self.minibatch_size):
				# Get minibatch indices
				end_idx = min(start_idx + self.minibatch_size, len(states))
				idx = indices[start_idx:end_idx]
				
				# Extract minibatch data
				mb_states = self.prepare_batch(states, idx)
				mb_actions = actions[idx]
				mb_old_log_probs = old_log_probs[idx]
				mb_returns = returns[idx]
				mb_advantages = advantages[idx]
				
				# Get current policy distribution
				if self.is_discrete:
					logits = self.actor(mb_states)
					dist = Categorical(logits=logits)
					new_log_probs = dist.log_prob(mb_actions).unsqueeze(1)
					entropy = dist.entropy().mean()
				else:
					mean, log_std = self.actor(mb_states)
					std = torch.exp(log_std)
					dist = Normal(mean, std)
					new_log_probs = dist.log_prob(mb_actions).sum(dim=1, keepdim=True)
					entropy = dist.entropy().mean()
				
				# Calculate ratio between old and new policy
				ratio = torch.exp(new_log_probs - mb_old_log_probs)
				
				# Calculate surrogate objectives
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
				
				# Calculate actor loss
				actor_loss = -torch.min(surr1, surr2).mean()
				
				# Calculate critic loss
				value_pred = self.critic(mb_states)
				critic_loss = 0.5 * ((value_pred - mb_returns) ** 2).mean()
				
				# Calculate total loss
				loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
				
				# Perform optimization step
				self.optimizer.zero_grad()
				loss.backward()
				
				# Clip gradients
				nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				
				# Update parameters
				self.optimizer.step()
				
				# Check KL divergence
				with torch.no_grad():
					if self.is_discrete:
						logits_new = self.actor(mb_states)
						dist_new = Categorical(logits=logits_new)
						kl = torch.distributions.kl_divergence(dist, dist_new).mean().item()
					else:
						mean_new, log_std_new = self.actor(mb_states)
						std_new = torch.exp(log_std_new)
						dist_new = Normal(mean_new, std_new)
						kl = torch.distributions.kl_divergence(dist, dist_new).mean().item()
					
					if kl > 1.5 * self.target_kl:
						break

class PPO_Centralized(PPO):
	def __init__(self, observation_shape, critic_observation_shape, action_space, params, device="cuda"):
		super().__init__(observation_shape, action_space, params, device)
		
		# Update critic to use centralized observation
		self.critic_obs_shape = critic_observation_shape
		self.critic = Critic(self.critic_obs_shape, params["hidden_dim"]).to(self.device)

		self.memory = CentralizedCriticRolloutMemory()


	def preprocess_critic_state(self, critic_state):
		"""Convert centralized critic state to torch tensor with proper shape"""
		if isinstance(critic_state, np.ndarray):
			state_tensor = torch.FloatTensor(critic_state).to(self.device)
		elif isinstance(critic_state, torch.Tensor):
			state_tensor = critic_state
		else:
			raise ValueError("expected a numpy ndarray or torch tensor as critic state")
		
		if len(self.critic_obs_shape) == 3:  # Image observation
			if len(state_tensor.shape) == 3:
				state_tensor = state_tensor.unsqueeze(0)
		else:
			if len(state_tensor.shape) == len(self.critic_obs_shape):
				state_tensor = state_tensor.unsqueeze(0)
				
		return state_tensor

	def select_action(self, obs, critic_state):
		"""Select action based on observation and centralized critic state"""
		state_tensor = self.preprocess_state(obs)
		critic_state_tensor = self.preprocess_critic_state(critic_state)
		value = self.critic(critic_state_tensor).squeeze().cpu().item()

		with torch.no_grad():
			if self.is_discrete:
				# Discrete action space
				logits = self.actor(state_tensor)
				dist = Categorical(logits=logits)
				action = dist.sample()
				log_prob = dist.log_prob(action)
				action_np = action.cpu().item()  # Convert to scalar
				return action_np, log_prob.item(), value
			else:
				# Continuous action space
				mean, log_std = self.actor(state_tensor)
				std = torch.exp(log_std)
				dist = Normal(mean, std)
				action = dist.sample()
				log_prob = dist.log_prob(action).sum(dim=-1)
				
				# Scale action to environment's action space
				action_np = action.squeeze().cpu().numpy()
				scaled_action = self.scale_action(action_np)
				
				return scaled_action, log_prob.item(), value
	
	
	def step(self, state, action, reward, log_prob, value, done, state_critic, next_state_critic):
		self.memory.push(state, action, reward, log_prob, value, done, state_critic)
		
		if len(self.memory.states) >= self.horizon:
			self.update(next_state_critic)

	def update(self, next_critic_state):
		scaled_rewards = np.array(self.memory.rewards) * self.reward_scale
		
		# Get final value estimate using centralized critic state
		critic_state_tensor = self.preprocess_critic_state(next_critic_state)
		with torch.no_grad():
			next_value = self.critic(critic_state_tensor).squeeze().cpu().item()
		
		# Compute returns and advantages
		returns, advantages = self.compute_gae(scaled_rewards, self.memory.values, next_value, self.memory.dones)
		
		rollout_data = {
			'states': self.memory.states,
			'critic_states': self.memory.critic_states,  # Centralized critic states
			'actions': np.array(self.memory.actions),
			'log_probs': np.array(self.memory.logprobs),
			'returns': np.array(returns),
			'advantages': np.array(advantages)
		}

		self.update_policy(rollout_data)
		self.memory.clear()

	def update_policy(self, rollout_data):
		states = rollout_data['states']
		critic_states = rollout_data['critic_states']
		actions = rollout_data['actions']
		old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
		returns = torch.FloatTensor(rollout_data['returns']).unsqueeze(1).to(self.device)
		advantages = torch.FloatTensor(rollout_data['advantages']).unsqueeze(1).to(self.device)
		
		if self.is_discrete:
			actions = torch.LongTensor(actions).to(self.device)
		else:
			actions = torch.FloatTensor(actions).to(self.device)
			
		if old_log_probs.dim() == 1:
			old_log_probs = old_log_probs.unsqueeze(1)
		
		for _ in range(self.update_epochs):
			indices = np.random.permutation(len(states))
			for start_idx in range(0, len(states), self.minibatch_size):
				end_idx = min(start_idx + self.minibatch_size, len(states))
				idx = indices[start_idx:end_idx]
				assert len(states) == len(critic_states), f"States and critic states must have the same length len(state):{len(states)} != len(critic_states):{len(critic_states)}"
				mb_states = self.prepare_batch(states, idx)
				mb_critic_states = self.prepare_batch(critic_states, idx)
				mb_actions = actions[idx]
				mb_old_log_probs = old_log_probs[idx]
				mb_returns = returns[idx]
				mb_advantages = advantages[idx]
				
				if self.is_discrete:
					logits = self.actor(mb_states)
					dist = Categorical(logits=logits)
					new_log_probs = dist.log_prob(mb_actions).unsqueeze(1)
					entropy = dist.entropy().mean()
				else:
					mean, log_std = self.actor(mb_states)
					std = torch.exp(log_std)
					dist = Normal(mean, std)
					new_log_probs = dist.log_prob(mb_actions).sum(dim=1, keepdim=True)
					entropy = dist.entropy().mean()
				
				ratio = torch.exp(new_log_probs - mb_old_log_probs)
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
				actor_loss = -torch.min(surr1, surr2).mean()
				
				value_pred = self.critic(mb_critic_states)
				critic_loss = 0.5 * ((value_pred - mb_returns) ** 2).mean()
				
				loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
				
				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				self.optimizer.step()

def test_ppo():
	# Create a simple environment
	env = gym.make("CartPole-v1", render_mode='human')
	obs_shape = env.observation_space.shape
	action_space = env.action_space

	# Initialize PPO agent
	ppo_agent = PPO(observation_shape=obs_shape, action_space=action_space, device="cpu")

	num_episodes = 600
	for episode in range(num_episodes):
		obs = env.reset()[0]
		done = False
		episode_reward = 0

		while not done:
			action, log_prob, value = ppo_agent.select_action(obs)
			next_obs, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			ppo_agent.step(obs, action, next_obs, reward, log_prob, value, done)
			obs = next_obs
			episode_reward += reward

		print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# if __name__ == "__main__":
# 	test_ppo()