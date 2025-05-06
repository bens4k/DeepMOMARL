import json
import gymnasium as gym
import numpy as np
from madsol import Utility, UtilityGroup, MultiAgentSolver, VarFactory
from madsol import SolverModelType, UtilityMethod, Optimization
import argparse
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
import os

class InfluenceProgram:
	def __init__(self, 
			  model : SolverModelType = None,
			  model_params = None, 
			  utility_groups: list = [], 
			  vars: list = [], 
			  adj_matrix: np.ndarray = None, 
			  device = "cpu"):
		self.vars = vars
		self.adj_matrix = adj_matrix
		self.utility_groups = utility_groups
		self.device = device
		self.model = SolverModelType(model) if model is not None else None
		self.model_params = model_params
		self.factory = VarFactory()
		self.solver = None

	def load(self, file_path: str):
		with open(file_path, 'r') as f:
			data = json.load(f)
			# load adjacency matrix, device and model
			self.model = SolverModelType(data["model"])
			self.model_params = data["model_params"]
			self.adj_matrix = np.array(data["adj_matrix"])
			self.device = data["device"]
			# Create utility groups
			self.utility_groups = {}
			for ug in data["utility_groups"]:
				utils = []
				for u in ug["utils"]:
					utils.append(Utility(
						func=u["func"],
						weight=u["weight"],
						utopian_point=u["utopian_point"],
						optim=Optimization(u["optim"]),
						name=u["name"]
					))
				self.utility_groups[ug["group_name"]] = UtilityGroup(
					utils=utils,
					name=ug["group_name"],
					method=UtilityMethod[ug["utility_method"]]
				)
			# load var data into the factory
			for v in data["variables"]:
				if v["is_continuous"]:
					action_space = gym.spaces.Box(
						low=np.array(v["action_space_low"]),
						high=np.array(v["action_space_high"]),
						shape=tuple(v["action_space_shape"]),
						dtype=np.float64
					)
				else:
					action_space = gym.spaces.Discrete(v["number_of_actions"])
				
				self.factory.add(
					action_space=action_space,
					is_chance=v["is_chance"],
					util_group=self.utility_groups[v["util_group"]] if v["util_group"] is not None else None,
					sampling_space=np.array(literal_eval(v["sampling_space"])) if v["sampling_space"] is not None else None,
					name=v["name"],
					action_labels=v["action_labels"],
					action_values= np.array(v["action_values"]) if v["action_values"] is not None else None,
				)

	def reset(self, verbose = False):
		self.vars = self.factory.create_vars(self.adj_matrix)
		self.solver = MultiAgentSolver(self.vars, self.adj_matrix, model=self.model, params=self.model_params, device=self.device)
		if verbose:
			print(self.solver)

	def start(self, episodes: int, log_freq: int, plot_stats: bool, verbose : bool):	
		assert len(self.vars) > 0 and self.solver != None, \
			"Influence program not initialized, call reset() to initialize"
		self.solver.train(episodes=episodes, log_freq=log_freq, plot_stats=plot_stats, verbose=verbose)

def compute_metrics(vars: list, run_stats: list):
	non_chance_vars = list(filter(lambda v: not v.is_chance, vars))
	metrics = {
		"num_episodes": len(run_stats[0][0]["actions"]),
		"num_runs": len(run_stats)
	}
	utopian_objectives = [[u.utopian_point for u in v.util_group.utils] if not v.is_chance else None for v in vars]
	
	# converged solution
	converged_actions = {}
	converged_objectives = {}
	
	stochastic_program = len([var for var in vars if var.is_chance]) > 0
	if stochastic_program:
		# calculate converged solution by finding the mean of the last 1000 episodes
		for var in vars:
			if not var.is_chance:
				actions = []
				objectives = []
				for stats in run_stats:
					actions.extend(stats[var.var_i]["actions"][-1000:])
					objectives.extend(stats[var.var_i]["objectives"][-1000:])
				converged_actions[var.var_i] = np.mean(actions, axis=0).tolist()
				converged_objectives[var.var_i] = np.mean(objectives, axis=0).tolist()
	else:
		# construct a list of (joint_actions, objectives, rewards) for the last 1000 episodes for each episode for each run
		joint_data = []
		for stats in run_stats:
			count = max(-1000, -len(stats[0]["actions"]))
			for ep in range(count, 0):
				joint_actions = [stats[var.var_i]["actions"][ep] for var in vars if not var.is_chance]
				objectives = [stats[var.var_i]["objectives"][ep] for var in vars if not var.is_chance]
				rewards = [stats[var.var_i]["rewards"][ep] for var in vars if not var.is_chance]
				joint_data.append((joint_actions, objectives, rewards))
		
		# find the top 100 elements of the list by sorting by rewards of the first var
		top_data = sorted(joint_data, key=lambda x: x[2][0], reverse=True)[:100]
		
		# finally, find the mean for these actions and objectives over all runs
		for var in vars:
			if not var.is_chance:
				var_actions = [data[0][var.var_i] for data in top_data]
				var_objectives = [data[1][var.var_i] for data in top_data]
				converged_actions[var.var_i] = np.mean(var_actions, axis=0).tolist()
				converged_objectives[var.var_i] = np.mean(var_objectives, axis=0).tolist()
	
	metrics["converged_actions"] = converged_actions
	metrics["converged_objectives"] = converged_objectives
	
	# Solution Quality Metrics
	# optimality gap
	opt_gap_mean = {}
	opt_gap_std = {}
	for var in non_chance_vars:
		var_diffs = []
		for stats in run_stats:
			objectives = np.array(stats[var.var_i]["objectives"])
			run_diffs = objectives - np.array(utopian_objectives[var.var_i])
			var_diffs.append(run_diffs)
		opt_gap_mean[var.var_i] = np.mean(var_diffs, axis=0).tolist()
		opt_gap_std[var.var_i] = np.std(var_diffs, axis=0).tolist()
	metrics["opt_gaps_mean"] = opt_gap_mean
	metrics["opt_gaps_std"] = opt_gap_std

	# Convergence Metrics
	# rewards
	rewards_mean = {}
	rewards_std = {}
	for v in non_chance_vars:
		var_avg_rewards = np.mean([stats[v.var_i]["rewards"] for stats in run_stats], axis=0)
		var_std_rewards = np.std([stats[v.var_i]["rewards"] for stats in run_stats], axis=0)
		rewards_mean[v.var_i] = var_avg_rewards.tolist()
		rewards_std[v.var_i] = var_std_rewards.tolist()
	metrics["rewards_mean"] = rewards_mean
	metrics["rewards_std"] = rewards_std
	
	# Smoothed rewards mean
	rewards_mean_smooth = {}
	for v in non_chance_vars:
		var_rewards = [stats[v.var_i]["rewards"] for stats in run_stats]
		mean_rewards = np.mean(var_rewards, axis=0)
		smoothed_rewards = np.convolve(mean_rewards, np.ones(100)/100, mode='valid')
		rewards_mean_smooth[v.var_i] = smoothed_rewards.tolist()
	metrics["rewards_mean_smooth"] = rewards_mean_smooth

	# objectives regrets
	cum_regrets_mean = {}
	cum_regrets_std = {}
	regrets_mean = {}
	regrets_std = {}
	for var in non_chance_vars:
		var_cum_regrets = []
		var_regrets = []
		# if the task is to maximize or minimize objectives,
		# regret is the difference between the best objective and the current objective
		# depending on the optimization direction Regret_t​={O_t​−O∗ ​​(minimization); O* -O_t (maximization)​}
		if var.util_group.utility_method == UtilityMethod.weighted_sum:
			optim_signs = np.array([-1 if utility.optim == Optimization.minimize else 1 for utility in var.util_group.utils])
			inverse_optim_signs = -optim_signs
			for stat in run_stats:
				cum_regret = np.zeros(len(stat[var.var_i]["objectives"][0]))
				episode_regrets = []
				episode_cum_regrets = []
				for ep in range(len(stat[var.var_i]["objectives"])):
					regret = np.array(inverse_optim_signs*stat[var.var_i]["objectives"][ep] + optim_signs*utopian_objectives[var.var_i])
					cum_regret += regret
					episode_regrets.append(regret.tolist())
					episode_cum_regrets.append(cum_regret.tolist())
				var_regrets.append(episode_regrets)
				var_cum_regrets.append(episode_cum_regrets)
		# if the task is to minimize the distance to the utopian point,
		# we use distance-based regret Regret_t​ = ∥ O_t​ − U ∥ ;; Euclidean norm
		else:
			for stats in run_stats:
				cum_regret = 0
				episode_cum_regrets = []
				episode_regrets = []
				objectives = np.array(stats[var.var_i]["objectives"])
				best_objectives_array = np.array(utopian_objectives[var.var_i])
				min_values = np.min(np.vstack([objectives, best_objectives_array]), axis=0)
				max_values = np.max(np.vstack([objectives, best_objectives_array]), axis=0)
				normalized_objectives = (objectives - min_values) / (max_values - min_values + 1e-8)
				normalized_best_objectives = (best_objectives_array - min_values) / (max_values - min_values + 1e-8)
				#print(f"min_values: {min_values}, max_values: {max_values}")
				#print(f"normalized_objectives: {normalized_objectives}, normalized_best_objectives: {normalized_best_objectives}")
				for ep in range(len(stats[var.var_i]["objectives"])):
					regret = np.linalg.norm(normalized_objectives[ep] - normalized_best_objectives)
					cum_regret += regret
					episode_regrets.append(regret.tolist())
					episode_cum_regrets.append(cum_regret.tolist())
				var_cum_regrets.append(episode_cum_regrets)
				var_regrets.append(episode_regrets)

		regrets_mean[var.var_i] = np.mean(var_regrets, axis=0).tolist()
		regrets_std[var.var_i] = np.std(var_regrets, axis=0).tolist()
		cum_regrets_mean[var.var_i] = np.mean(var_cum_regrets, axis=0).tolist()
		cum_regrets_std[var.var_i] = np.std(var_cum_regrets, axis=0).tolist()
	
	metrics["inst_regrets_mean"] = regrets_mean
	metrics["inst_regrets_std"] = regrets_std
	metrics["cum_regrets_mean"] = cum_regrets_mean
	metrics["cum_regrets_std"] = cum_regrets_std

	return metrics

def plot_metrics(vars, metrics, out_path):
	matplotlib.use('Agg')
	# mean reward
	plt.figure()
	for var_i, rewards in metrics["rewards_mean"].items():
		plt.plot(rewards, label=f"{vars[var_i].name}")
	plt.title("Rewards mean")
	plt.xlabel("Episodes")
	plt.ylabel("Reward")
	plt.legend()
	plt.savefig(os.path.join(out_path, "rewards_mean.png"))
	
	# smoothed mean reward
	plt.figure()
	for var_i, rewards in metrics["rewards_mean_smooth"].items():
		plt.plot(rewards, label=f"{vars[var_i].name}")
	plt.title("Rolling average reward over 100 episodes")
	plt.xlabel("Episodes")
	plt.ylabel("Reward")
	plt.legend()
	plt.savefig(os.path.join(out_path, "rewards_mean_smooth.png"))

	# std reward
	plt.figure()
	for var_i, std_rewards in metrics["rewards_std"].items():
		plt.plot(std_rewards, label=f"{vars[var_i].name}")
	plt.title("Rewards std deviation")
	plt.xlabel("Episodes")
	plt.ylabel("Std deviation")
	plt.legend()
	plt.savefig(os.path.join(out_path, "rewards_std.png"))

	# mean optimality gap
	plt.figure()
	for var_i, mean_opt_gaps in metrics["opt_gaps_mean"].items():
		for obj_idx, obj_gaps in enumerate(zip(*mean_opt_gaps)):
			obj_name = vars[var_i].util_group.utils[obj_idx].name
			plt.plot(obj_gaps, label=f"{vars[var_i].name} - {obj_name}")
	plt.title("Optimality gaps mean")
	plt.xlabel("Episodes")
	plt.ylabel("Optimality Gap")
	plt.legend()
	plt.savefig(os.path.join(out_path, "opt_gaps_mean.png"))
	
	# std optimality gap
	plt.figure()
	for var_i, std_opt_gaps in metrics["opt_gaps_std"].items():
		for obj_idx, obj_gaps in enumerate(zip(*std_opt_gaps)):
			obj_name = vars[var_i].util_group.utils[obj_idx].name
			plt.plot(obj_gaps, label=f"{vars[var_i].name} - {obj_name}")
	plt.title("Optimality gaps std deviation")
	plt.xlabel("Episodes")
	plt.ylabel("Std deviation")
	plt.legend()
	plt.savefig(os.path.join(out_path, "opt_gaps_std.png"))

	# Mean instantaneous regret
	plt.figure()
	for var_i, regrets in metrics["inst_regrets_mean"].items():
		plt.plot(regrets, label=f"{vars[var_i].name}")
	plt.title("Instantaneous regrets mean")
	plt.xlabel("Episodes")
	plt.ylabel("Regret")
	plt.legend()
	plt.savefig(os.path.join(out_path, "inst_regrets_mean.png"))

	# std instantaneous regrets
	plt.figure()
	for var_i, std_regrets in metrics["inst_regrets_std"].items():
		plt.plot(std_regrets, label=f"{vars[var_i].name}")
	plt.title("Instantaneous regrets std deviation")
	plt.xlabel("Episodes")
	plt.ylabel("Std deviation")
	plt.legend()
	plt.savefig(os.path.join(out_path, "inst_regrets_std.png"))

	# Mean cumulative regret
	plt.figure()
	for var_i, regrets in metrics["cum_regrets_mean"].items():
		plt.plot(regrets, label=f"{vars[var_i].name}")
	plt.title("Cumulative regrets mean")
	plt.xlabel("Episodes")
	plt.ylabel("Regret")
	plt.legend()
	plt.savefig(os.path.join(out_path, "cum_regrets_mean.png"))

	# std cumulative regrets
	plt.figure()
	for var_i, regrets in metrics["cum_regrets_std"].items():
		plt.plot(regrets, label=f"{vars[var_i].name}")
	plt.title("Cumulative regrets std deviation")
	plt.xlabel("Episodes")
	plt.ylabel("Std deviation")
	plt.legend()
	plt.savefig(os.path.join(out_path, "cum_regrets_std.png"))

def plot_opt_gaps_metrics(vars, metrics, output_image_path):
	non_chance_vars = [v for v in vars if not v.is_chance]

	# # Read the JSON file into a dictionary
	# with open(json_file_path, 'r') as file:
	# 	metrics = json.load(file)
	
	opt_gaps_mean = metrics.get("opt_gaps_mean", [])
	opt_gaps_std = metrics.get("opt_gaps_std", [])

	# smoothed optimality gaps
	opt_gaps_mean_smooth = {}
	for v in non_chance_vars:
		opt_gaps_mean_np = np.array(opt_gaps_mean[v.var_i]).T
		objectives = []
		for i in range(len(opt_gaps_mean_np)):
			smoothed_opt_gaps_mean = np.convolve(opt_gaps_mean_np[i], np.ones(100)/100, mode='valid')
			objectives.append(smoothed_opt_gaps_mean)
		opt_gaps_mean_smooth[v.var_i] = objectives

	# smoothed optimality gaps std deviation
	opt_gaps_std_smooth = {}
	for v in non_chance_vars:
		opt_gaps_std_np = np.array(opt_gaps_std[v.var_i]).T
		objectives = []
		for i in range(len(opt_gaps_std_np)):
			smoothed_opt_gaps_std = np.convolve(opt_gaps_std_np[i], np.ones(100)/100, mode='valid')
			objectives.append(smoothed_opt_gaps_std)
		opt_gaps_std_smooth[v.var_i] = objectives
	
	# Create opt gaps plot
	plt.figure(figsize=(10, 6))
	for var_i, var_opt_gaps_mean in opt_gaps_mean_smooth.items():
		for i in range(len(var_opt_gaps_mean)):
			steps = range(len(var_opt_gaps_mean[i]))
			plt.plot(steps, var_opt_gaps_mean[i], label=f"{vars[var_i].name} {vars[var_i].util_group.utils[i].name}")
			plt.fill_between(
				steps,
				[m - s for m, s in zip(var_opt_gaps_mean[i], opt_gaps_std_smooth[var_i][i])],
				[m + s for m, s in zip(var_opt_gaps_mean[i], opt_gaps_std_smooth[var_i][i])],
				alpha=0.2
			)

	plt.xlabel("Episodes")
	plt.ylabel("Optimality gaps")
	plt.title("Mean optimality gaps with standard deviation")
	plt.legend()
	plt.grid(True)
	
	# Save the plot to an image file
	plt.savefig(output_image_path)
	plt.close()

def plot_rewards_metrics(vars, metrics, output_image_path):
	non_chance_vars = [v for v in vars if not v.is_chance]
	
	# Extract data for plotting
	rewards_mean_smooth = metrics.get("rewards_mean_smooth", [])
	rewards_std = metrics.get("rewards_std", [])
	
	# Smoothed rewards std deviation
	rewards_std_smooth = {}
	for v in non_chance_vars:
		smoothed_std_rewards = np.convolve(rewards_std[v.var_i], np.ones(100)/100, mode='valid')
		rewards_std_smooth[v.var_i] = smoothed_std_rewards.tolist()

	# Create rewards plot
	plt.figure(figsize=(10, 6))
	
	for var_i, rewards_mean_smooth in metrics.get("rewards_mean_smooth", {}).items():
		steps = range(len(rewards_mean_smooth))
		plt.plot(steps, rewards_mean_smooth, label=f"{vars[var_i].name}")
		plt.fill_between(
			steps,
			[m - s for m, s in zip(rewards_mean_smooth, rewards_std_smooth[var_i])],
			[m + s for m, s in zip(rewards_mean_smooth, rewards_std_smooth[var_i])],
			#color="blue",
			alpha=0.2,
			label=f"{vars[var_i].name} standard deviation"
		)
	
	plt.xlabel("Episodes")
	plt.ylabel("Rewards")
	plt.title("Mean rewards with standard deviation")
	plt.legend()
	plt.grid(True)
	
	# Save the plot to an image file
	plt.savefig(output_image_path)
	plt.close()

def plot_metrics_v1(vars, metrics, out_path):
	matplotlib.use('Agg')
	plot_rewards_metrics(vars, metrics, os.path.join(out_path, "rewards_mean.png"))
	plot_opt_gaps_metrics(vars, metrics, os.path.join(out_path, "opt_gaps_mean.png"))

def save_results(vars: list, run_stats: list, out_path: str):
	metrics = compute_metrics(vars, run_stats)
	with open(out_path, "w") as f:
		json.dump(metrics, f, indent=4, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else x)
	plot_metrics_v1(vars, metrics, os.path.dirname(out_path))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Influence Program Solver")
	parser.add_argument("path", type=str, help="Path to the input JSON file")
	parser.add_argument("-e", "--episodes", type=int, help="Number of episodes")
	parser.add_argument("-n", "--logfreq", type=int, help="Log solution every n episodes")
	parser.add_argument("-p", "--plot", action="store_true", help="Plot stats")
	parser.add_argument("-o", "--out", required=False, type=str, help="Path to a file where stats will be saved")
	parser.add_argument("-r", "--runs", default=1, required=False, type=int, help="number of runs")

	args = parser.parse_args()
	ip = InfluenceProgram()
	ip.load(args.path) # load from json
	ip.reset(verbose=True) # create vars and solver

	run_stats = []
	for i in range(args.runs):
		print(f'>>>>>> run {i+1} <<<<<<')
		ip.start(
			episodes= int(args.episodes), 
			log_freq=int(args.logfreq),
			plot_stats = args.plot,
			verbose = True
		)
		run_stats.append(ip.solver.env.stats)
		ip.reset()
	
	if args.out is not None:
		save_results(ip.vars, run_stats, args.out)