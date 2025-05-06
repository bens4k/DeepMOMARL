import inspect
from itertools import count
import warnings
import gymnasium as gym
import numpy as np
import torch
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import List
from enum import StrEnum
import random

from models.sac import SAC, SACDiscrete
from models.dqn import DDQN
from models.ppo import PPO, PPO_Centralized
from models.ddpg import DDPG, DDPG_Centralized

import user_code

plt.ion()

device = torch.device(
	"cuda" if torch.cuda.is_available() else
	"mps" if torch.backends.mps.is_available() else
	"cpu"
)

# Fix seeds for reproducibility
def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

set_seed(42)

class Optimization(StrEnum):
	maximize = "maximize"
	minimize = "minimize"


class SolverModelType(StrEnum):
	ddpg_centralized_global = "ddpg_centralized_global"
	ddpg_centralized_group = "ddpg_centralized_group"
	ppo_centralized_group = "ppo_centralized_group"
	ppo = "ppo"
	ddpg = "ddpg"
	ddqn = "ddqn",
	sac = "sac"


class UtilityMethod(StrEnum):
	weighted_sum = "weighted_sum"
	utopian_point = "utopian_point"


class Utility:
	def __init__(self, func: str, weight: float, utopian_point: float, optim: Optimization = Optimization.maximize, name: str = "obj"):
		if not hasattr(user_code, func):
			raise ValueError(f"function '{func}' not found in user_code")
		
		if not callable(getattr(user_code, func)):
			raise ValueError(f"'{func}' is not a callable function")
		
		self.func = getattr(user_code, func)
		self.weight = weight
		self.utopian_point = utopian_point
		self.optim = optim
		self.name = name


class UtilityGroup:
	def __init__(self, utils: List[Utility], method: UtilityMethod = UtilityMethod.weighted_sum, name: str = ""):
		self.utils = utils
		self.name = f"util-{len(utils)}" if len(name) == 0 else name
		self.utility_method = method
		self.min_reward = -1e3

	def group_reward(self, global_state):
		if len(self.utils) == 0:
			return 0
		
		if self.utility_method == UtilityMethod.weighted_sum:
			return self.weighted_sum(global_state)
		elif self.utility_method == UtilityMethod.utopian_point:
			return self.utopian_point(global_state)
		else:
			raise ValueError(f"Unknown utility type: {self.utility_method}")	
	
	def calc_objectives(self, global_state):
		objectives = [u.func(global_state) for u in self.utils]
		objectives = np.array([obj if np.isscalar(obj) else obj.item() for obj in objectives])
		return objectives
	
	def weighted_sum(self, global_state):
		objectives = self.calc_objectives(global_state)
		weights = np.array([u.weight for u in self.utils])
		optim = np.array([-1 if u.optim == Optimization.minimize else 1 for u in self.utils])

		scalarized_value = np.sum(weights * objectives * optim)
		return max(self.min_reward, scalarized_value), objectives

	def utopian_point(self, global_state):
		objectives = self.calc_objectives(global_state)
		weights = np.array([u.weight for u in self.utils])
		utopian_point = np.array([u.utopian_point for u in self.utils])

		scalarized_value = np.sum(weights * np.abs(objectives - utopian_point))
		return max(self.min_reward, -scalarized_value), objectives

	def __str__(self):
		s = f"utility group: {self.name}, method={self.utility_method}\n"
		s += "\n".join([f">> {u.name}: w={u.weight}, utp={u.utopian_point}, optim={u.optim}" for u in self.utils])
		return s
	
	def props(self):
		weights = np.array([u.weight for u in self.utils])
		utopian_point = np.array([u.utopian_point for u in self.utils])
		return weights, utopian_point


class Var:
	def __init__(
			self,
			var_i: int,
			state_size: int,
			action_space: gym.spaces.Space,
			is_chance: bool,
			util_group: UtilityGroup,
			action_values = None,
			sampling_space = None,
			name: str = "",
			action_labels=None
	):
		self.var_i = var_i
		self.state_size = state_size
		self.action_space = action_space
		self.action_shape = action_space.sample().shape
		self.is_continuous = isinstance(action_space, gym.spaces.Box)
		self.is_chance = is_chance
		self.sampling_space = sampling_space
		self.util_group = util_group
		self.name = f"var-{var_i}" if len(name) == 0 else name
		self.action_labels = action_labels
		self.action_values = action_values

		if self.state_size == 0 and not self.is_chance:
			self.is_chance = True
			warnings.warn(
				f"empty state for '{self.name}', setting is_chance to True")

		# if chance var & sampling method given -> validate
		if self.sampling_space is not None:
			self.validate_sampling_method()

	def __str__(self):
		is_ch = "[C]" if self.is_chance else "[M]"
		group = self.util_group.name if self.util_group is not None else "<none>"
		return f"var-{self.var_i} ({self.name}) {is_ch}, group={group}, state_size=({self.state_size})"

	def validate_sampling_method(self):
		if isinstance(self.action_space, gym.spaces.Discrete):
			# check type of parameter
			assert isinstance(self.sampling_space, np.ndarray), \
				f"Var({self.name}): unsupported tpye {type(self.sampling_space)} for a probability table, expected a numpy.ndarray"
			if self.state_size > 0:
				# ensure num of dimensions matches state size
				assert len(self.sampling_space.shape[:-1]) == self.state_size,\
					 f"Var({self.name}): probability table shape {self.sampling_space.shape} does not match the number of observable variables {self.state_size}"
			
			# ensure last dimension matches domain size
			assert self.sampling_space.shape[-1] == self.action_space.n, \
				f"Var({self.name}): probability table dim {len(self.sampling_space.shape)-1} of size {self.sampling_space.shape[-1]} does not match the domain space {self.action_space.n}"
			
			# ensure action_values have the same shape as sampling_space
			if self.action_values is not None:
				assert self.action_values.shape == self.sampling_space.shape, \
					f"Var({self.name}): probability table shape {self.sampling_space.shape} does not match the action_values shape {self.action_values.shape}"
		
		elif isinstance(self.action_space, gym.spaces.Box):
			assert callable(self.sampling_space), \
				f"Var({self.name}): {type(self.sampling_space)} is not callable"
			if self.state_size > 0:
				params = inspect.signature(self.sampling_space).parameters
				assert len(params) == self.state_size, \
					f"Var({self.name}): probability function parameter mismatch, expected {self.state_size} params, got {len(params)}"

	def sample_action_space(self, var_state):
		if self.sampling_space is not None:
			return self.__sample_prob(var_state)
		else:
			return self.__sample_uniform()
		
	def __sample_prob(self, var_state):
		sampled_action = None
		# sample using probability table
		if isinstance(self.action_space, gym.spaces.Discrete):
			# sample from action_space or from give action_values
			sample_from = self.action_space.n
			if self.action_values is not None:
				sample_from = self.action_values
			
			if self.state_size > 0:
				p_index = tuple(var_state.astype(np.int64))
				sampled_action = np.random.choice(sample_from, p=self.sampling_space[p_index])
			else:
				sampled_action = np.random.choice(sample_from, p=self.sampling_space)

		# sample using distribution func
		elif isinstance(self.action_space, gym.spaces.Box):
			p = self.sampling_space(*var_state)
			sampled_action = np.clip(p, self.action_space.low, self.action_space.high)
			# print(f"sampled action for {self.name} is {sampled_action} with p({var_state}) = {p}")

		return sampled_action

	def __sample_uniform(self):
		if self.action_values is not None:
			return random.choice(self.action_values)
		return self.action_space.sample()


class VarFactory:
	def __init__(self):
		self.var_configs = []

	def add(self,
			action_space: gym.spaces.Space,
			is_chance: bool,
			action_values : np.ndarray = None,
			util_group: UtilityGroup = None,
			sampling_space=None,
			name: str = "",
			action_labels=None):

		self.var_configs.append({
			"action_space": action_space,
			"is_chance": is_chance,
			"util_group": util_group,
			"sampling_space": sampling_space,
			"name": name,
			"action_values" : action_values,
			"action_labels": action_labels
		})

	def get_observed_state_size(self, adj_matrix: np.ndarray, var_i: int) -> int:
		observations = []
		for i, obs_flag in enumerate(adj_matrix[var_i]):
			if obs_flag != 0:
				action_shape = self.var_configs[i]["action_space"].sample().shape
				observations.append(
					MultiAgentEnv.empty_observation(obs_flag, action_shape).flatten())
		
		if len(observations) == 0:
			return 0
		#print(observations)
		return np.concat(observations).flatten().shape[0]

	def create_vars(self, links: np.ndarray):
		num_vars = len(self.var_configs)
		assert(num_vars, num_vars) == links.shape, f"expected adj matrix shape {(num_vars, num_vars)} but found {links.shape}"
		variables = []
		for i, config in enumerate(self.var_configs):
			observed_state_size = self.get_observed_state_size(links, i)
			variables.append(
				Var(
					var_i=i,
					state_size=observed_state_size,
					action_space=config["action_space"],
					is_chance=config["is_chance"],
					util_group=config["util_group"],
					action_values=config["action_values"],
					sampling_space=config["sampling_space"],
					name=config["name"],
					action_labels=config["action_labels"]
				)
			)
		return variables


class MultiAgentEnv:
	def __init__(self, all_vars, links_adj_matrix):
		self.all_vars = sorted(all_vars, key=lambda v: v.var_i)
		self.history_max_len = 10
		self.actions = dict([(v.var_i, deque(maxlen=self.history_max_len)) for v in self.all_vars])
		self.links_adj_matrix = links_adj_matrix
		self.stats = dict((v.var_i, {"rewards": [], "objectives" : [], "actions" : []}) for v in self.all_vars)

	def get_var_obs_state(self, var_i, observ_flag):
		acts = []
		recorded_actions = list(self.actions[var_i])

		if observ_flag < 0:
			# not including the current action
			index = max(observ_flag, -1*len(recorded_actions))
			acts = recorded_actions[index:]
		elif observ_flag > 0:
			# including the current action
			index = min(observ_flag, len(recorded_actions)-1)
			acts = recorded_actions[-index:]

		observation = MultiAgentEnv.empty_observation(
			observ_flag, self.all_vars[var_i].action_shape)
		
		for i, a in enumerate(acts):
			observation[i] = self.normalize(a, self.all_vars[var_i].action_space)

		return observation.flatten()

	def empty_observation(observ_flag, action_shape):
		num_actions = abs(observ_flag)
		return np.zeros(shape=(num_actions, *action_shape), dtype=np.float32)

	def get_var_state(self, var_i):
		if self.links_adj_matrix is None or self.all_vars[var_i].state_size == 0:
			var_state = np.array([])
		else:
			observable_vars_states = []
			for var_i, obs_flag in enumerate(self.links_adj_matrix[var_i]):
				if obs_flag != 0:
					var_i_observation = self.get_var_obs_state(var_i, obs_flag)
					observable_vars_states.append(var_i_observation)
			var_state = np.concat(observable_vars_states)
		return var_state

	def normalize(self, var_state, action_space):
		if isinstance(action_space, gym.spaces.Box):
			if np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high)):
				return (var_state - action_space.low) / (action_space.high - action_space.low)
			return var_state # no normalization if action space has infinite bounds
		elif isinstance(action_space, gym.spaces.Discrete):
			return var_state / (action_space.n - 1)
		else:
			raise ValueError(f"Unknown action space type: {type(action_space)}")
		
	def get_group_state(self, util_group: UtilityGroup):
		var_states = []
		for v in filter(lambda a: a.util_group == util_group, self.all_vars):
			var_states.append(self.get_var_state(v.var_i))
		return np.concat(var_states)

	def step(self, var_i: int, var_action) -> tuple[np.ndarray, float, bool, bool, dict]:
		self.actions[var_i].append(var_action)
		reward = 0
		terminated = var_i == (len(self.all_vars) - 1)  # terminal state
		# premature ending (e.g. time limit, or agent gets out of bounds)
		truncated = False
		info = {}
		return reward, terminated, truncated, info

	def get_global_state(self):
		global_state = []
		for v in self.all_vars:
			if len(self.actions[v.var_i]) > 0:
				last_action = self.actions[v.var_i][-1]
				#print(f"last_action: {last_action}")
				last_action = np.reshape(last_action, v.action_shape)
				global_state.append(last_action)
			else:
				global_state.append(np.zeros(v.action_shape))
		#print(f"global_state: {global_state}")
		## MUST BE A PYTHON LIST, IT CAN CONTAIN UNHOMOGENEOUS ELEMENTS
		return global_state

	def group_reward(self, util_group: UtilityGroup):
		return util_group.group_reward(self.get_global_state())

	def sol_summary(self, ep) -> str:
		s = f"--- solution details (ep-{ep+1}) ---\n"
		for v in self.all_vars:
			#print(f"stats = {self.stats[v.var_i]}")
			action = self.stats[v.var_i]["actions"][ep]
			if not v.is_continuous and v.action_labels is not None:
				action = v.action_labels[action]	
			s += f"{v.name}: a={action}"
			if not v.is_chance:
				scalarized_val = self.stats[v.var_i]["rewards"][ep]
				objectives = self.stats[v.var_i]["objectives"][ep]
				weights, utopian_point = v.util_group.props()
				s += f", r={scalarized_val:.2f}, obj={objectives}, w={weights}, utp={utopian_point}\n"
			else:
				s += "\n"
		#s += "-----------" * 2
		return s

TempTransition = namedtuple('TempTransition', ['state', 'action','value', 'logprob', 'group_state', 'global_state', 'done'])

class MultiAgentSolver:
	def __init__(self, vars, links, model : SolverModelType, params, device="cuda"):
		self.all_vars = vars
		self.env = MultiAgentEnv(vars, links)
		self.model_type = model
		self.model_params = params
		self.device = device

	def __str__(self):
		s = f"--- solver summary ---\n"
		s += f"model: {self.model_type}\n"
		s += f"device: {self.device}\n"
		s += f"actions history max len: {self.env.history_max_len}\n"
		s += f"variables:\n"
		s += "\n".join([str(v) for v in self.all_vars])
		s += "\n------------------"
		return s

	def reset_agent(self, var_i: int):
		v = self.all_vars[var_i]
		if v.is_chance:
			return
		global_state_size = np.concat(self.env.get_global_state()).size
		group_state_size = self.env.get_group_state(v.util_group).size

		# create models
		if self.model_type == SolverModelType.ddpg_centralized_global:
			self.models[var_i] = DDPG_Centralized(v.state_size, global_state_size, v.action_space, self.model_params, device=self.device)
		
		elif self.model_type == SolverModelType.ddpg_centralized_group:
			self.models[var_i] = DDPG_Centralized(v.state_size, group_state_size, v.action_space, self.model_params, device=self.device)
		
		elif self.model_type == SolverModelType.ppo_centralized_group:
			self.models[var_i] = PPO_Centralized(np.array([v.state_size]), np.array([group_state_size]), v.action_space, self.model_params, device=self.device)
		
		elif self.model_type == SolverModelType.ppo:
			self.models[var_i] = PPO(np.array([v.state_size]), v.action_space, self.model_params, device=self.device)
		
		elif self.model_type == SolverModelType.ddpg:
			self.models[var_i] = DDPG(v.state_size, v.action_space, self.model_params, device=self.device)
		
		elif self.model_type == SolverModelType.ddqn:
			self.models[var_i] = DDQN(v.state_size, v.action_space, self.model_params, device=self.device)
		
		elif self.model_type == SolverModelType.sac:
			self.models[var_i] = SAC(v.state_size, v.action_space, self.model_params, device=self.device) \
			if v.is_continuous \
			else SACDiscrete(v.state_size, v.action_space, self.model_params, device=self.device)
		else:
			raise ValueError(f"Unknown model type: {self.model_type}")
		
	def train(self, episodes: int, log_freq: int, plot_stats: bool, verbose : bool):
		non_chance_vars = list(filter(lambda x: not x.is_chance, self.env.all_vars))
		util_groups = list(set(map(lambda v: v.util_group, non_chance_vars)))
		# agents_not_reset = int(len(non_chance_vars)/2)
		# reset_freq = 1000

		# create models
		if self.model_type == SolverModelType.ddpg_centralized_global:
			global_state_size = np.concat(self.env.get_global_state()).size
			self.models = dict([(v.var_i, DDPG_Centralized(v.state_size, global_state_size, v.action_space, self.model_params, device=self.device)) for v in non_chance_vars])
		
		elif self.model_type == SolverModelType.ddpg_centralized_group:
			group_state_sizes = dict([(u, self.env.get_group_state(u).size) for u in util_groups])
			self.models = dict([(v.var_i, DDPG_Centralized(v.state_size, group_state_sizes[v.util_group], v.action_space, self.model_params, device=self.device)) for v in non_chance_vars])
		
		elif self.model_type == SolverModelType.ppo_centralized_group:
			group_state_sizes = dict([(u, self.env.get_group_state(u).size) for u in util_groups])
			self.models = dict([(v.var_i, PPO_Centralized(np.array([v.state_size]), np.array([group_state_sizes[v.util_group]]), v.action_space, self.model_params, device=self.device)) for v in non_chance_vars])
		
		elif self.model_type == SolverModelType.ppo:
			self.models = dict([(v.var_i, PPO(np.array([v.state_size]), v.action_space, self.model_params, device=self.device)) for v in non_chance_vars])
		
		elif self.model_type == SolverModelType.ddpg:
			self.models = dict([(v.var_i, DDPG(v.state_size, v.action_space, self.model_params, device=self.device)) for v in non_chance_vars])
		
		elif self.model_type == SolverModelType.ddqn:
			self.models = dict([(v.var_i, DDQN(v.state_size, v.action_space, self.model_params, device=self.device)) for v in non_chance_vars])
		
		elif self.model_type == SolverModelType.sac:
			self.models = dict(
				[(
					v.var_i, 
					SAC(v.state_size, v.action_space, self.model_params, device=self.device)
					if v.is_continuous else 
					SACDiscrete(v.state_size, v.action_space, self.model_params, device=self.device)
				) for v in non_chance_vars])
		else:
			raise ValueError(f"Unknown model type: {self.model_type}")

		# training loop
		for e in range(episodes):
			total_rewards = dict((v.var_i, 0) for v in non_chance_vars)
			for step in count():
				any_done = False
				tmp_mem = dict([(v.var_i, []) for v in self.all_vars])
				# perform actions and move state
				for v in self.env.all_vars:
					var_state = self.env.get_var_state(v.var_i)
					group_state = self.env.get_group_state(v.util_group)
					global_state = self.env.get_global_state()

					logprob, value = None, None
					if not v.is_chance and self.model_type == SolverModelType.ppo:
						var_action, logprob, value = self.models[v.var_i].select_action(var_state)
					elif not v.is_chance and self.model_type == SolverModelType.ppo_centralized_group:
						var_action, logprob, value = self.models[v.var_i].select_action(var_state, group_state)
					elif not v.is_chance:
						var_action = self.models[v.var_i].select_action(var_state)
					else:
						var_action = v.sample_action_space(var_state)
					
					_, terminated, truncated, info = self.env.step(v.var_i, var_action)
					done = terminated or truncated

					if not v.is_chance:
						if self.model_type == SolverModelType.ppo:
							t = TempTransition(state=var_state, action=var_action, group_state=None, global_state=None, logprob=logprob, value=value, done=done)
							tmp_mem[v.var_i].append(t)
						elif self.model_type == SolverModelType.ppo_centralized_group:
							t = TempTransition(state=var_state, action=var_action, group_state=group_state, global_state=None, logprob=logprob, value=value, done=done)
							tmp_mem[v.var_i].append(t)
						elif self.model_type == SolverModelType.ddpg_centralized_group:
							t = TempTransition(state=var_state, action=var_action, group_state=group_state, global_state=None, logprob=None, value=None, done=done)
							tmp_mem[v.var_i].append(t)
						elif self.model_type == SolverModelType.ddpg_centralized_global:
							t = TempTransition(state=var_state, action=var_action, group_state=None, global_state=global_state, logprob=None, value=None, done=done)
							tmp_mem[v.var_i].append(t)
						else:
							t = TempTransition(state=var_state, action=var_action, group_state=None, global_state=None, logprob=None, value=None, done=done)
							tmp_mem[v.var_i].append(t)
					else: # chance variable -> recorded for stats
						t = TempTransition(state=var_state, action=var_action, group_state=None, global_state=None, logprob=None, value=None, done=None)
						tmp_mem[v.var_i].append(t)
					
					if done:
						any_done = True
						break

				# compute group rewards
				rewards = dict([(u, 0) for u in util_groups])
				objectives = dict([(u, None) for u in util_groups])
				for u in util_groups:
					rewards[u], objectives[u] = self.env.group_reward(u)
				indiv_rewards = dict([(v.var_i, rewards[v.util_group]) for v in non_chance_vars])

				# update vars
				for v in non_chance_vars:
					var_next_state = self.env.get_var_state(v.var_i)
					group_next_state = self.env.get_group_state(v.util_group)
					reward = indiv_rewards[v.var_i]
					t : TempTransition

					if self.model_type == SolverModelType.ppo:
						for t in tmp_mem[v.var_i]:
							self.models[v.var_i].step(state=t.state, action=t.action, next_state=var_next_state, log_prob=t.logprob, reward=reward, done=t.done, value=t.value)
							total_rewards[v.var_i] += reward
							# print(f"{v.name} -> state={var_state}, action={var_action}, next_state={var_next_state}, reward={reward}")

					elif self.model_type == SolverModelType.ppo_centralized_group:
						for t in tmp_mem[v.var_i]:
							self.models[v.var_i].step(state=t.state, state_critic=t.group_state, action=t.action,
													  next_state_critic=group_next_state, log_prob=t.logprob, reward=reward, done=t.done, value=value)
							total_rewards[v.var_i] += reward
							# print(f"{v.name} -> state={var_state}, action={var_action}, next_state={var_next_state}, reward={reward}")
							# print(f"group_state={group_state}")

					elif self.model_type == SolverModelType.ddpg_centralized_group:
						for t in tmp_mem[v.var_i]:
							next_group_state = self.env.get_group_state(v.util_group)
							self.models[v.var_i].step(state_actor=t.state, state_critic=t.group_state, action=t.action,
													  next_state_actor=var_next_state, next_state_critic=next_group_state, reward=reward, done=t.done)
							total_rewards[v.var_i] += reward

					elif self.model_type == SolverModelType.ddpg_centralized_global:
						for t in tmp_mem[v.var_i]:
							next_global_state = self.env.get_global_state()
							self.models[v.var_i].step(state_actor=t.state, state_critic=t.global_state, action=t.action,
													  next_state_actor=var_next_state, next_state_critic=next_global_state, reward=reward, done=t.done)
							total_rewards[v.var_i] += reward
							# print(f"{v.name} -> state={var_state}, action={var_action}, next_state={var_next_state}, reward={reward}")
							# print(f"global_state={global_state}, next_global_state={next_global_state}")
					else:
						for t in tmp_mem[v.var_i]:
							self.models[v.var_i].step(state=t.state, action=t.action, next_state=var_next_state, reward=reward, done=t.done)
							total_rewards[v.var_i] += reward
							#print(f"{v.name} -> state={var_state}, action={var_action}, next_state={var_next_state}, reward={reward}")

				# metrics
				for v in self.all_vars:
					_, actions, _, _, _, _, _ = zip(*tmp_mem[v.var_i])
					#### WARNING !!!! this assumes a single action per agent per episode					
					self.env.stats[v.var_i]["actions"].append(actions[0])
					if not v.is_chance:
						self.env.stats[v.var_i]["rewards"].append(total_rewards[v.var_i])
						self.env.stats[v.var_i]["objectives"].append(objectives[v.util_group])
				
				if any_done:
					if plot_stats:
						self.plot_stats(show_rewards=False, show_mean=True)
					break
			
			# if (e+1) % reset_freq == 0:
			# 	if agents_not_reset > 0:
			# 		for v in non_chance_vars[-agents_not_reset:]:
			# 			print(f"resetting agent {v.name}...")
			# 			self.reset_agent(v.var_i)
			# 		agents_not_reset -= 1

			if (e+1) % log_freq == 0 and verbose:
				print(self.env.sol_summary(e))

	def plot_stats(self, show_rewards=True, show_mean=False):
		plt.figure(1)
		plt.clf()
		plt.title('Training...')
		plt.xlabel('Episode')
		plt.ylabel('Reward')

		# plot rewards & durations foreach var
		for v in self.env.all_vars:
			if show_rewards:
				plt.plot(self.env.stats[v.var_i]
						 ["rewards"], label=f"{v.name} rewards")
				# plt.plot(self.stats[v.var_i]["durations"], label=f"{v.name} durations")

			if show_mean:
				# Take 100 episode averages and plot them too
				if len(self.env.stats[v.var_i]["rewards"]) >= 100:
					r_t = torch.tensor(self.env.stats[v.var_i]["rewards"])
					means = r_t.unfold(0, 100, 1).mean(1).view(-1)
					means = torch.cat((torch.zeros(99), means))
					plt.plot(means.numpy(), label=f"{v.name} rewards mean")
		
		if plt.gca().has_data():
			plt.legend()
		plt.pause(0.001)  # pause a bit so that plots are updated