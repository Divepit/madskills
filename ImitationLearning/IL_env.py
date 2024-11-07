import gymnasium as gym
import copy
import numpy as np
import logging

from gymnasium.spaces import Dict, MultiDiscrete, Discrete, Box, MultiBinary

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ILEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, final_num_agents, final_num_goals, num_skills, final_size, render_mode="human"):
        super(ILEnv, self).__init__()
        self.final_num_agents = final_num_agents
        self.final_num_goals = final_num_goals
        self.final_size = final_size
        self.num_skills = num_skills

        self.current_num_agents = 3
        self.current_num_goals = 5
        self.current_size = 32
        self.assume_lander = False

        self.sandboxEnv: Environment = Environment(num_agents=self.current_num_agents, num_goals=self.current_num_goals, size=self.current_size, num_skills=self.num_skills, use_geo_data=True, assume_lander=self.assume_lander)

        self.render_mode = render_mode
        self.max_episode_attempts = 5

        self.step_count = 0
        self.episode_steps_walked = 0
        self.episode_steps_waited = 0
        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_attempts = 0
        self.episode_distributed_goals = 0
        self.episode_cost = 0
        self.episode_deadlocks = 0
        self.episode_unclaimed_goals = 0
        self.previous_reward = -np.inf
        self.reward_count = 0

        self.advance = False
        self.stage = 0



        self.action_space = gym.spaces.MultiDiscrete(
            nvec=[self.final_num_goals+1]*self.final_num_agents*self.final_num_goals,
            start=[0]*self.final_num_agents*self.final_num_goals,
            dtype=np.int64
        )

        
        self.observation_space = Dict(
            {   
                "claimed_goals": MultiDiscrete(nvec=[3]*self.final_num_goals, start=[0]*self.final_num_goals, dtype=np.int16),
                # "map_elevations": Box(low=-1, high=2, shape=(self.final_size * self.final_size,), dtype=np.float32),
                "goal_positions": Box(low=0, high=2, shape=(self.final_num_goals*2,), dtype=np.float32),
                "agent_positions": Box(low=0, high=2, shape=(self.final_num_agents*2,), dtype=np.float32),
                "goal_required_skills": MultiDiscrete(
                nvec=[3]*self.sandboxEnv.num_skills*self.final_num_goals,
                start=[0]*self.sandboxEnv.num_skills*self.final_num_goals,
                dtype=np.int16
                ),
                "agent_skills": MultiDiscrete(
                nvec=[3]*self.sandboxEnv.num_skills*self.final_num_agents,
                start=[0]*self.sandboxEnv.num_skills*self.final_num_agents,
                dtype=np.int16
                ),
                # "steps_walked": MultiDiscrete(nvec=[self.final_size**2]*self.final_num_agents, start=[0]*self.final_num_agents, dtype=np.int16),
                # "steps_waited": MultiDiscrete(nvec=[self.final_size**2]*self.final_num_agents, start=[0]*self.final_num_agents, dtype=np.int16),
                # "costs": MultiDiscrete(nvec=[self.final_size**2]*self.final_num_agents, start=[0]*self.final_num_agents, dtype=np.int16),
            }
        )

    def step(self, action):
        done = False
        self.episode_attempts += 1
        distributed_goals = 0
        reward = 0

        corrected_action = action-1

        agent_actions = [
            corrected_action[i * self.current_num_agents:(i + 1) * self.current_num_goals] 
            for i in range(self.current_num_agents)
        ]
        
        for agent_index, actions in enumerate(agent_actions):
            agent = self.sandboxEnv.agents[agent_index]
            valid_goals = [goal for goal in actions if goal != -1 and goal < len(self.sandboxEnv.goals)]
            self.sandboxEnv.full_solution[agent] = [self.sandboxEnv.goals[goal] for goal in valid_goals]
            distributed_goals += len(valid_goals)
        

        prev_amount_of_unclaimed_goals = len(self.sandboxEnv.scheduler.unclaimed_goals)
        prev_steps, prev_waited, prev_cost = self.sandboxEnv.get_agent_benchmarks()
        self.sandboxEnv.solve_full_solution(fast=False, soft_reset=False)
        total_steps, steps_waited, total_cost = self.sandboxEnv.get_agent_benchmarks()
        amount_of_unclaimed_goals = len(self.sandboxEnv.scheduler.unclaimed_goals)
        
        newly_claimed_goals = prev_amount_of_unclaimed_goals - amount_of_unclaimed_goals

        # reward -= amount_of_unclaimed_goals*(total_cost-prev_cost)/self.sandboxEnv.size*2

        # reward -= (len(self.sandboxEnv.goals) - amount_of_unclaimed_goals*5)
        # reward -= (total_cost-prev_cost)
        # reward -= (steps_waited-prev_waited)*0.01
        reward -= (total_steps-prev_steps)*0.025
        # reward -= (distributed_goals*0.5)**2
        reward += newly_claimed_goals

        self.episode_reward += reward
        self.episode_cost += total_cost
        self.episode_steps_waited += steps_waited
        self.episode_steps_walked += total_steps
        self.episode_distributed_goals += distributed_goals
        self.episode_deadlocks += int(self.sandboxEnv.deadlocked)
        self.episode_unclaimed_goals += amount_of_unclaimed_goals

        logging.debug("Reward: {}".format(reward))
        logging.debug("Unclaimed Goals: {}".format(amount_of_unclaimed_goals))

        done = self.sandboxEnv.scheduler.all_goals_claimed()

        truncated = self.episode_attempts >= self.max_episode_attempts

        if done or truncated:

            # if done:
            #     reward += 5
            # else:
            #     reward -= amount_of_unclaimed_goals/self.current_num_goals

            # self.previous_reward = self.episode_reward
            # if self.episode_attempts == 1:
            #     self.reward_count += 1
            #     logging.debug("Reward counter: {}".format(self.reward_count))
            # else:
            #     if self.reward_count > 0:
            #         logging.debug("Reward counter reset")
            #         self.reward_count = 0

            # if self.reward_count >= 25:
            #     self.advance = True

            info = {"episode": {"r": self.episode_reward/self.episode_attempts, "l": self.episode_attempts,"distributed_goals": self.episode_distributed_goals/self.episode_attempts, "cost": self.episode_cost/self.episode_attempts, "unclaimed_goals": self.episode_unclaimed_goals/self.episode_attempts, "episode_attempts": self.episode_attempts, "deadlocks": self.episode_deadlocks/self.episode_attempts, "steps_walked": self.episode_steps_walked/self.episode_attempts, "steps_waited": self.episode_steps_waited/self.episode_attempts, "num_goals": self.current_num_goals, "num_agents": self.current_num_agents, "size": self.current_size, "reward_count": self.reward_count}}

        else:
            info = {}

        # logging.debug("Observation: {}".format(self.sandboxEnv.get_observation_vector()))
        return self.sandboxEnv.get_observation_vector(pad_agents=self.final_num_agents-self.current_num_agents, pad_goals=self.final_num_goals-self.current_num_goals, pad_map=self.final_size**2-self.sandboxEnv.grid_map.downscaled_data.shape[0]**2), float(reward), done, truncated, info

    def reset(self, seed=None, options=None, reset = True):
        logging.debug("Resetting environment")
        super().reset(seed=seed)
        stages = [
            {"num_agents": 3, "num_goals": 5, "size": 32},
            # {"num_agents": 3, "num_goals": 7, "size": 64},
            # {"num_agents": 3, "num_goals": 8, "size": 64},
            # {"num_agents": 3, "num_goals": 9, "size": 64},
            # {"num_agents": 3, "num_goals": 10, "size": 64},

        ]

        if self.advance:
            self.stage += 1
            if self.stage >= len(stages):
                self.stage = len(stages)-1
            logging.debug("Advancing")
            self.current_num_agents = stages[self.stage]["num_agents"]
            self.current_num_goals = stages[self.stage]["num_goals"]
            self.current_size = stages[self.stage]["size"]
            self.advance = False
            self.reward_count = 0
            self.sandboxEnv: Environment = Environment(num_agents=self.current_num_agents, num_goals=self.current_num_goals, size=self.current_size, num_skills=self.num_skills, use_geo_data=True, assume_lander=self.assume_lander)
        else:
            if reset:
                self.sandboxEnv.reset()

        
        self.episode_steps_walked = 0
        self.episode_steps_waited = 0
        self.episode_attempts = 0
        self.episode_reward = 0
        self.episode_claimed_goals = 0
        self.episode_distributed_goals = 0
        self.episode_cost = 0
        self.step_count = 0
        self.episode_deadlocks = 0
        self.episode_unclaimed_goals = 0
        
        return self.sandboxEnv.get_observation_vector(pad_agents=self.final_num_agents-self.current_num_agents, pad_goals=self.final_num_goals-self.current_num_goals, pad_map=self.final_size**2-self.sandboxEnv.grid_map.downscaled_data.shape[0]**2), {}
    
    def render(self, soft_reset=False):
        vis = Visualizer(env=self.sandboxEnv, speed=20)
        vis.visualise_full_solution(soft_reset=soft_reset)
        del vis
        