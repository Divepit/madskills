from typing import List, Dict
import logging
import networkx as nx
import planning_sandbox.utils as utils

from itertools import permutations, product, combinations

from planning_sandbox.grid_map_class import GridMap
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal
from planning_sandbox.scheduler_class import Scheduler
from planning_sandbox.benchmark_class import Benchmark

import numpy as np

class Environment:
    def __init__(self, size, num_skills, num_agents = 1, num_goals = 1, use_geo_data=True, solve_type="optimal",replan_on_goal_claim=False, custom_agents: List[Agent] = None, custom_goals: List[Goal] = None, assume_lander=True, random_map=False):
        self.size = size
        self.solve_type = solve_type
        self.num_skills = num_skills
        self.replan_on_goal_claim = replan_on_goal_claim
        self.deadlocked = False

        self.custom_goals = custom_goals
        self.custom_agents = custom_agents
        self.goals: List[Goal] = [] if self.custom_goals is None else self.custom_goals
        self.agents: List[Agent] = [] if self.custom_agents is None else self.custom_agents
        self.new_goal_added = False if self.custom_goals is None else True
        self.scheduler = Scheduler(agents=self.agents, goals=self.goals)
        self.full_solution = {}
        

        self._initial_num_agents = num_agents if self.custom_agents is None else len(self.custom_agents)
        self._initial_num_goals = num_goals if self.custom_goals is None else len(self.custom_goals)
        
        self.grid_map = GridMap(self.size, use_geo_data=use_geo_data, random_map=random_map)
        self._starting_position = self.grid_map.random_valid_position()
        self._assume_lander = assume_lander

        self.initialised = False
        self.agents_goals_connected = False
        
        self._init()
        self._log_environment_info()

    def _init(self):
        assert not self.initialised, "Environment already initialised"
        if self.custom_goals is None:
            self._initialize_goals()
        if self.custom_agents is None:
            self._initialize_agents()

        while self.custom_agents is None and self.custom_goals is None and not self._all_skills_represented():
            self._reset_skills()
            self._initialize_skills()
        
        self.scheduler.reset()
        
        self.initialised = True

    def _init_agent_goal_connections(self):
        assert self.initialised, "Environment not initialised"
        assert not self.agents_goals_connected, "Agents and goals already connected"
        self._connect_agents_and_goals()
        self._inform_goals_of_costs_to_other_goals()
        self.agents_goals_connected = True

    def _log_environment_info(self):
        logging.debug(f"=== Environment settings ===")
        logging.debug(f"Num agents: {len(self.agents)}")
        logging.debug(f"Num goals: {len(self.goals)}")
        logging.debug(f"Num skills: {self.num_skills}")
        logging.debug(f"Map size: {self.size}")
        logging.debug(f"=== === === === === === ===")
        
    def _initialize_agents(self):
        if self._starting_position is not None:
            start_pos = self._starting_position
        else:
            start_pos = self.grid_map.random_valid_position()
        for _ in range(self._initial_num_agents):
            if not self._assume_lander:
                start_pos = self.grid_map.random_valid_position()
            agent = Agent(start_pos)
            self.agents.append(agent)

    def _initialize_goals(self):
        for _ in range(self._initial_num_goals):
            random_position = self.grid_map.random_valid_position()
            self._add_goal(random_position)
        self.new_goal_added = False

    def _add_goal(self, position):
        goal: Goal = Goal(position)
        self.goals.append(goal)
        self.new_goal_added = True
        return goal

    def _initialize_skills(self):
        if self.num_skills == 1:
            if self.custom_goals is None:
                for goal in self.goals:
                    if goal.required_skills == []:
                        goal.required_skills.append(0)
            if self.custom_agents is None:
                for agent in self.agents:
                    if agent.skills == []:
                        agent.skills.append(0)
            return

        if self.custom_goals is None:
            for goal in self.goals:
                amount_of_skills = np.random.randint(1, self.num_skills+1)
                skills = []
                for _ in range(amount_of_skills):
                    skill = np.random.randint(0, self.num_skills)
                    while skill in skills:
                        skill = np.random.randint(0, self.num_skills)
                    skills.append(skill)
                goal.required_skills = skills
                
        if self.custom_agents is None:
            for agent in self.agents:
                amount_of_skills = np.random.randint(1,self.num_skills+1)
                skills = []
                for _ in range(amount_of_skills):
                    skill = np.random.randint(0, self.num_skills)
                    while skill in skills:
                        skill = np.random.randint(0, self.num_skills)
                    skills.append(skill)
                agent.skills = skills

    def _all_skills_represented(self):
        all_skills = [0]*self.num_skills
        for agent in self.agents:
            for skill in agent.skills:
                all_skills[skill] = 1
        return all(all_skills)
    
    def _reset_skills(self):
        for agent in self.agents:
            agent.skills = []
        for goal in self.goals:
            goal.required_skills = []
        self._initialize_skills()  

    def _connect_agents_and_goals(self):
        logging.debug("Connecting agents and goals")
        inform_goals_of_agents_bench = Benchmark("inform_goals_of_agents", start_now=True, silent=True)
        for goal in self.scheduler.unclaimed_goals:
            for agent in self.agents:
                if any(skill in agent.skills for skill in goal.required_skills):
                    goal.add_agent_which_has_required_skills(agent)
        inform_goals_of_agents_bench.stop()

        inform_agents_of_costs_bench = Benchmark("inform_agents_of_costs_to_goals", start_now=True, silent=True)
        for goal in self.scheduler.unclaimed_goals:
            for agent in goal.agents_which_have_required_skills:
                    path = self.grid_map.generate_shortest_path_for_agent(agent, goal)
                    cost = self.grid_map.calculate_path_cost(path)
                    agent.add_path_to_goal(goal, path, cost)
        inform_agents_of_costs_bench.stop()

        for goal in self.scheduler.unclaimed_goals:
            goal.generate_agent_combinations_which_solve_goal()

    def _inform_goals_of_costs_to_other_goals(self):
        logging.debug("Informing goals of costs to other goals")
        inform_goals_of_costs_bench = Benchmark("_inform_goals_of_costs_to_other_goals", start_now=True, silent=True)
        for goal in self.scheduler.unclaimed_goals:
            for other_goal in self.goals:
                if goal == other_goal:
                    continue

                if not other_goal in goal.paths_and_costs_to_other_goals:
                    path = self.grid_map.shortest_path(goal.position, other_goal.position)
                    cost = self.grid_map.calculate_path_cost(path)
                    goal.add_path_to_other_goal(other_goal, path, cost)

                if not goal in other_goal.paths_and_costs_to_other_goals:
                    path = self.grid_map.shortest_path(other_goal.position, goal.position)
                    cost = self.grid_map.calculate_path_cost(path)
                    other_goal.add_path_to_other_goal(goal, path, cost)


        inform_goals_of_costs_bench.stop()    

    def _replan(self):
        logging.debug("Replanning")
        self._connect_agents_and_goals()
        new_goal = False
        for goal in self.goals:
            if  len(goal.paths_and_costs_to_other_goals) == 0:
                new_goal = True
        if new_goal:
            self._inform_goals_of_costs_to_other_goals()
        self.find_numerical_solution()
        self.new_goal_added = False

    def _calculate_cost_of_chain(self, agent: Agent, chain: List[Goal]):
        cost = 0
        length = 0
        if not chain:
            return cost, length
        
        first_goal = chain[0]

        if self.agents_goals_connected and first_goal in agent.paths_and_costs_to_goals:
            cost = agent.paths_and_costs_to_goals[first_goal][1]
            length = len(agent.paths_and_costs_to_goals[first_goal][0])
        else:
            path = self.grid_map.generate_shortest_path_for_agent(agent, first_goal)
            cost = self.grid_map.calculate_path_cost(path)
            length = len(path)

        for i in range(1,len(chain)):
            previous_goal = chain[i-1]
            current_goal = chain[i]
            if previous_goal == current_goal:
                continue
            if self.agents_goals_connected:
                cost += previous_goal.paths_and_costs_to_other_goals[current_goal][1]
                length += len(previous_goal.paths_and_costs_to_other_goals[current_goal][0])
            else:
                path = self.grid_map.shortest_path(previous_goal.position, current_goal.position)
                cost += self.grid_map.calculate_path_cost(path)
                length += len(path)
        return cost, length

    def soft_reset(self):
        logging.debug("Soft resetting environment")
        self.deadlocked = False
        self.grid_map.reset()
        for goal in self.goals:
            goal.soft_reset()

        for agent in self.agents:
            agent.soft_reset()

        self.scheduler.reset()

    def reset(self):
        self.initialised = False
        self.deadlocked = False
        self.agents_goals_connected = False
        self.grid_map.reset()
        self.full_solution = {}
        self._starting_position = self.grid_map.random_valid_position()
        self.goals.clear()
        self.agents.clear()
        self._init()  

    def find_numerical_solution(self, solve_type=None):
        if not self.agents_goals_connected:
            self._init_agent_goal_connections()
        if solve_type is not None:
            self.solve_type = solve_type
        if self.solve_type == "optimal":
            cost = 0
            self.replan_on_goal_claim = False
            self.full_solution, cost = self.find_optimal_solution()
        elif self.solve_type == "fast":
            self.replan_on_goal_claim = True
            cost = 0
            if self.full_solution is None:
                self.full_solution = {}
            intermediate_solution = self.scheduler.find_fast_solution()
            cost = np.inf
            for agent in intermediate_solution:
                if agent not in self.full_solution:
                    self.full_solution[agent] = []
                self.full_solution[agent].extend(intermediate_solution[agent])
        return self.full_solution, cost

    def step_environment(self, fast=False):
        logging.debug("Stepping environment")
        for agent, goal_list in self.full_solution.items():
            for i, goal in enumerate(goal_list):
                if goal.claimed:
                    continue
                self.scheduler.goal_assignments[agent] = goal
                break # OH MY GOD DO NOT REMOVE THIS BREAK
        not_deadlocked = self.update(fast=fast)
        return not_deadlocked

    def solve_full_solution(self, fast=False, soft_reset=True):
        if soft_reset:
            self.soft_reset()
        else:
            self.deadlocked = False
        
        solving_bench: Benchmark = Benchmark("solve_full_solution", start_now=True, silent=True)
        while not (self.deadlocked or self.scheduler.all_goals_claimed()):
            self.step_environment(fast=fast)
        
        if self.deadlocked:
            logging.debug("Deadlocked")

        solve_time = solving_bench.stop()

        total_steps, steps_waited, total_cost = self.get_agent_benchmarks()
        amount_of_claimed_goals = len(self.goals) - len(self.scheduler.unclaimed_goals)

        return total_steps, steps_waited, total_cost, solve_time, amount_of_claimed_goals
    
    def get_observation_vector(self, pad_agents: int = 0, pad_goals: int = 0, pad_map: int = 0):
        goals_map = []
        # goals_map = np.zeros((self.size, self.size), dtype=np.int16)
        for goal in self.goals:
            goals_map.append(goal.position[0]/self.size)
            goals_map.append(goal.position[1]/self.size)
        for _ in range(pad_goals):
            goals_map.append(2)
            goals_map.append(2)
        goals_map = np.array(goals_map, dtype=np.float32)
            # goals_map[goal.position[0], goal.position[1]] = 1
        # agents_map = np.zeros((self.size, self.size), dtype=np.int16)
        agents_map = []
        for agent in self.agents:
            agents_map.append(agent.position[0]/self.size)
            agents_map.append(agent.position[1]/self.size)
            # agents_map[agent.position[0], agent.position[1]] = 1
        for _ in range(pad_agents):
            agents_map.append(2)
            agents_map.append(2)
        agents_map = np.array(agents_map, dtype=np.float32)

        flattened_map = self.grid_map.downscaled_data.flatten()
        for _ in range(pad_map):
            flattened_map = np.append(flattened_map, 2)

        min_value = np.min(flattened_map)
        max_value = np.max(flattened_map)
        normalized_map = 2*((flattened_map - min_value) / (max_value - min_value))-1


        agent_skills = [[(1 if skill in agent.skills else 0) for skill in range(self.num_skills)] 
            for agent in self.agents]
        agent_skills.extend([[2]*self.num_skills for _ in range(pad_agents)])
        agent_skills = np.array(agent_skills, dtype=np.int16).flatten()

        goal_required_skills = [[(1 if skill in goal.required_skills else 0) for skill in range(self.num_skills)] 
            for goal in self.goals]
        goal_required_skills.extend([[2]*self.num_skills for _ in range(pad_goals)])
        goal_required_skills = np.array(goal_required_skills, dtype=np.int16).flatten()

        claimed_goals = [1 if goal.claimed else 0 for goal in self.goals]
        claimed_goals.extend([2]*pad_goals)
        claimed_goals = np.array(claimed_goals, dtype=np.int16)

        steps_walked = [agent.steps_moved for agent in self.agents]
        steps_walked.extend([0]*pad_agents)

        steps_waited = [agent.steps_waited for agent in self.agents]
        steps_waited.extend([0]*pad_agents)

        total_cost = [agent.accumulated_cost for agent in self.agents]
        total_cost.extend([0]*pad_agents)
        
        
        observation_vector = {
            
            "claimed_goals": claimed_goals,
            # "map_elevations": normalized_map.astype(np.float32),
            "goal_positions": goals_map,
            "agent_positions": agents_map,
            "goal_required_skills": goal_required_skills,
            "agent_skills": agent_skills,
            # "steps_walked": np.array(steps_walked, dtype=np.int16),
            # "steps_waited": np.array(steps_waited, dtype=np.int16),
            # "costs": np.array(total_cost, dtype=np.int16)
        }
        return observation_vector
    
    # has to start at 0, not at -1
    def get_action_vector(self):
        action_vector = []
        for agent in self.agents:
            if agent in self.full_solution:
                goal_list = self.full_solution[agent]
                # Pad with -1 (or use `num_goals` as a dummy) if fewer than max_goals_per_agent
                padded_goal_list = [node_id_map[goal] for goal in goal_list] + [-1] * (len(self.goals) - len(goal_list))
                action_vector.append(padded_goal_list)
            else:
                # No goals, append `-1` for all slots
                action_vector.append([-1] * len(self.goals))
        
        # Flatten the list for the MultiDiscrete action space
        flattened_action_vector = [goal+1 for sublist in action_vector for goal in sublist]
        
        return flattened_action_vector

    # has to start at 0, not at -1
    def get_full_solution_from_action_vector(self, action_vector):
        full_solution = {}
        action_vector = action_vector
        for flat_index, selected_goal in enumerate(action_vector):
            if selected_goal-1 != -1:  # Only process if action is valid
                try:
                    # Compute agent and goal indices from flat_index
                    agent_index = flat_index // len(self.goals)
                    
                    agent = self.agents[agent_index]
                    goal = self.goals[selected_goal-1]

                    # Add goal to the agent's full solution
                    if agent not in full_solution:
                        full_solution[agent] = []
                    full_solution[agent].append(goal)
                except IndexError:
                    pass
                    # logging.error(f"IndexError: flat_index={flat_index}, selected_goal={selected_goal}")
        return full_solution


    def update(self, fast=False):
        logging.debug("Updating environment")
        agent_positions_start = [agent.position for agent in self.agents]
        for agent,goal in self.scheduler.goal_assignments.items():
            if (not self.replan_on_goal_claim) and fast:
                logging.debug("Fast update")
                agent.move_to_position(goal.position)
            else:
                self.grid_map.assign_shortest_path_for_goal_to_agent(agent=agent, goal=goal)
                path = self.grid_map.paths[agent]
                cost = self.grid_map.calculate_path_cost(path)
                action, action_cost = self.grid_map.get_move_and_cost_to_reach_next_position(agent)
                agent.apply_action(action, action_cost)

        claimed_a_goal = self.scheduler.update_goal_statuses()
        if (claimed_a_goal and self.replan_on_goal_claim) or self.new_goal_added:
            self._replan()
        
        agent_positions_end = [agent.position for agent in self.agents]
        if not self.scheduler.all_goals_claimed():
            if agent_positions_start == agent_positions_end:
                self.deadlocked = True
            else:
                self.deadlocked = False
        logging.debug("Environment updated")
        return not self.deadlocked

    
    def get_agent_benchmarks(self):
        total_steps = 0
        steps_waited = 0
        total_cost = 0
        for agent in self.agents:
            total_steps += agent.steps_moved
            steps_waited += agent.steps_waited
            total_cost += agent.accumulated_cost
        return total_steps, steps_waited, total_cost

    
    def calculate_cost_of_closed_solution(self, solution: Dict[Agent, List[Goal]], max_cost=np.inf) -> int:
        solution_cost = 0
        for agent, goals in solution.items():
            cost, _ = self._calculate_cost_of_chain(agent, goals)
            solution_cost += cost
            if solution_cost > max_cost:
                return np.inf
        return solution_cost
    
    
    def find_optimal_solution(self):
        assert self.agents_goals_connected, "Agents and goals not connected"
        logging.debug("Finding optimal solution")
        full_solution = None
        cheapest_cost = np.inf
        all_goal_orders = iter(permutations(self.scheduler.unclaimed_goals, len(self.scheduler.unclaimed_goals)))
        for goal_order in all_goal_orders:
            candidate_permutations =iter(product(*[goal.agent_combinations_which_solve_goal.keys() for goal in goal_order]))
            for candidate_permutation in candidate_permutations:
                proposed_solution = {}
                proposed_solution_cost = np.inf
                for i, agent_list in enumerate(candidate_permutation):
                    goal = goal_order[i]
                    for agent in agent_list:
                        if agent not in proposed_solution:
                            proposed_solution[agent] = []
                        proposed_solution[agent].append(goal)
                    proposed_solution_cost = self.calculate_cost_of_closed_solution(solution=proposed_solution, max_cost=cheapest_cost)
                if full_solution is None or proposed_solution_cost < cheapest_cost:
                    full_solution = proposed_solution
                    cheapest_cost = proposed_solution_cost
        return full_solution, cheapest_cost
    
    def get_observation_graph(self):
        node_id_map = {}
        i = 0
        for agent in self.agents:
            node_id_map[agent] = i
            i += 1
        for goal in self.goals:
            node_id_map[goal] = i
            i += 1

        g = nx.Graph()
        types = {
            'goal': 0,
            'agent': 1
        }

        for agent in self.agents:
            g.add_node(node_id_map[agent], 
            type=types["agent"], 
            x=float(agent.position[0])/self.size, 
            y=float(agent.position[1])/self.size,
            skills=[1 if skill in agent.skills else 0 for skill in range(self.num_skills)]
            )
        for goal in self.goals:
            g.add_node(node_id_map[goal], 
            type=types["goal"], 
            x=float(goal.position[0])/self.size, 
            y=float(goal.position[1])/self.size,
            skills=[1 if skill in goal.required_skills else 0 for skill in range(self.num_skills)]
            )

        for goal in self.goals:
            for other_goal in self.goals:
                if goal == other_goal:
                    continue
                g.add_edge(
                    node_id_map[goal], 
                    node_id_map[other_goal], 
                    manhattan_distance=utils.manhattan_distance(start=goal.position, goal=other_goal.position),
                    idx=0
                    )
            for agent in self.agents:
                g.add_edge(
                node_id_map[goal], 
                node_id_map[agent], 
                manhattan_distance=utils.manhattan_distance(start=goal.position, goal=agent.position), 
                idx=0
                )
        for agent in self.agents:
            for other_agent in self.agents:
                if agent == other_agent:
                    continue
                g.add_edge(
                    node_id_map[agent], 
                    node_id_map[other_agent], 
                    manhattan_distance=utils.manhattan_distance(start=agent.position, goal=other_agent.position),
                    idx=0
                    )
        return g

    def get_solution_graph(self):
        node_id_map = {}

        i = 0
        for agent in self.agents:
            node_id_map[agent] = i
            i += 1
        for goal in self.goals:
            node_id_map[goal] = i
            i += 1

        g = nx.Graph()
        types = {
            'goal': 0,
            'agent': 1
        }

        for agent in self.agents:
            g.add_node(node_id_map[agent], 
            type=types["agent"], 
            x=float(agent.position[0])/self.size, 
            y=float(agent.position[1])/self.size,
            skills=[1 if skill in agent.skills else 0 for skill in range(self.num_skills)]
            )
        for goal in self.goals:
            g.add_node(node_id_map[goal], 
            type=types["goal"], 
            x=float(goal.position[0])/self.size, 
            y=float(goal.position[1])/self.size,
            skills=[1 if skill in goal.required_skills else 0 for skill in range(self.num_skills)]
            )

        for agent, goals in self.full_solution.items():
            assert self.full_solution is not None, "No solution found"
            for i, goal in enumerate(goals):
                g.add_edge(
                    node_id_map[agent],
                    node_id_map[goal], 
                    manhattan_distance=utils.manhattan_distance(start=agent.position, goal=goal.position),
                    idx=i
                    )
        return g