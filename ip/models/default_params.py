from dataclasses import dataclass
import json

@dataclass
class DDPGParams:
	seed: int = 42
	discount: float = 0.99
	actor_lr: float = 1e-4
	critic_lr: float = 1e-3
	tau: float = 0.005
	batch_size: int = 64
	replay_buffer_size: int = int(1e6)
	noise_std_dev: float = 0.1
	hidden_dim: int = 256
	ou_noise_mu : float = 0.0
	ou_noise_theta : float = 0.15
	ou_noise_sigma : float = 0.2

@dataclass
class PPOParams:
	hidden_dim: int = 256
	lr: float = 1e-3
	gamma: float = 0.995
	gae_lambda: float = 0.9
	clip_ratio: float = 0.3
	target_kl: float = 0.02
	value_coef: float = 0.5
	entropy_coef: float = 0.02
	update_epochs: int = 15
	horizon: int = 4096
	minibatch_size: int = 64
	max_grad_norm: float = 0.5
	reward_scale: float = 1.0

@dataclass
class SACParams:
	hidden_size: int = 256
	gamma: float = 0.995
	tau: float = 0.005
	alpha: float = 0.1
	lr: float = 1e-3
	policy_type: str = "Gaussian"
	target_update_interval: int = 1
	automatic_entropy_tuning: bool = True
	batch_size: int = 256
	memory_capacity: int = 1e6

@dataclass
class DDQNParams:
	hidden_size : int  = 64
	batch_size: int  = 64
	gamma : float = 0.99  # Discount factor
	learning_rate : float = 1e-4
	tau : float = 0.005
	target_update : int = 500
	eps_start : float = 0.9  # Exploration rate
	eps_end : float = 0.05
	eps_decay : float = 1000
	memory_capacity : int = 10000

defualt_params = {
	"ddpg": DDPGParams(),
	"ddpg_centralized_global" : DDPGParams(),
	"ddpg_centralized_group" : DDPGParams(),
	"ppo_centralized_group" : PPOParams(),
	"ppo" : PPOParams(),
	"sac" : SACParams(),
	"ddqn" : DDQNParams()
}

if __name__ == "__main__":
    # Convert the default_params dictionary to a serializable format
    serializable_params = {key: vars(value) for key, value in defualt_params.items()}
    # Write the dictionary to a JSON file
    with open("default_params.json", "w") as json_file:
        json.dump(serializable_params, json_file, indent=4)