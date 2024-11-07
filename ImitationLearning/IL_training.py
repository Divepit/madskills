import os
import logging
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback,BaseCallback
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

from IL_env import ILEnv
from planning_sandbox.environment_class import Environment

dir_path = os.path.dirname(os.path.realpath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_env(rank, final_num_agents, final_num_goals, final_size, num_skills, seed=0):
    logging.info(f"Creating environment {rank+1}")
    def _init():
        env = ILEnv(final_num_agents=final_num_agents, final_num_goals=final_num_goals, final_size=final_size, num_skills=num_skills)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        self.logger.dump(self.num_timesteps)
        for info in self.locals['infos']:
            if 'episode' in info:
                logging.debug(info['episode'])
                self.logger.record("env/ep_av_reward", info['episode']['r'])
                self.logger.record("env/ep_av_distributed_goals", info['episode']['distributed_goals'])
                self.logger.record("env/ep_av_unclaimed_goals", info['episode']['unclaimed_goals'])
                self.logger.record("env/ep_av_cost", info['episode']['cost'])
                self.logger.record("env/ep_av_length", info['episode']['episode_attempts'])
                self.logger.record("env/ep_av_deadlocks", info['episode']['deadlocks'])
                self.logger.record("env/ep_av_steps_waited", info['episode']['steps_waited'])
                self.logger.record("env/ep_av_steps_walked", info['episode']['steps_walked'])
                self.logger.record("env/num_goals", info['episode']['num_goals'])
                self.logger.record("env/num_agents", info['episode']['num_agents'])
                self.logger.record("env/size", info['episode']['size'])
                self.episode_rewards.append(info['episode']['r'])
            if 'terminal_observation' in info:
                logging.debug(info['terminal_observation'])
        return True
    
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=dir_path+"/model_logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

def main():

    final_num_agents = 3
    final_num_goals = 5
    num_skills = 2
    final_size = 32

    evalEnv = ILEnv(final_num_agents=final_num_agents, final_num_goals=final_num_goals, final_size=final_size, num_skills=num_skills)
    logging.info("Checking environment...")
    check_env(evalEnv, warn=True)

    n_envs = 12
    n_timesteps = 500000
    logging.info(f"Creating {n_envs} environments...")


    subproc_env = SubprocVecEnv([make_env(rank=i, final_num_agents=final_num_agents, final_num_goals=final_num_goals, final_size=final_size, num_skills=num_skills ) for i in range(n_envs)])
    # norm_env = VecNormalize(subproc_env)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[128,128])

    model = PPO(
        "MultiInputPolicy",
        subproc_env,
        learning_rate=0.0001,
        n_steps=15,  # Increase this if your environment needs longer trajectories for good learning
        batch_size=64,
        clip_range=0.2,  # Increase clip range slightly for more exploration
        ent_coef=0.0001,  # Add entropy to encourage exploration
        verbose=1,
        tensorboard_log=dir_path+"/tensorboard_logs/",
        device=device,
        policy_kwargs=policy_kwargs
    )

    logging.info(f"Training model for {n_timesteps} timesteps...")
    model.learn(
        total_timesteps=n_timesteps,
        progress_bar=True,
        callback=[checkpoint_callback, TensorboardCallback()],
        )


if __name__ == "__main__":
    main()