import logging

from typing import List, Dict
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal

class Scheduler:
    def __init__(self, agents, goals):
        self.agents: List[Agent] = agents
        self.goals: List[Goal] = goals
        self.unclaimed_goals: List[Goal] = [goal for goal in goals if not goal.claimed]
        self.goal_assignments: Dict[Agent, Goal] = {}
        self.last_visited_goals = {agent: None for agent in self.agents}

    def _get_agents_present_at_goal(self, goal: Goal):
        return [agent for agent in self.agents if agent.position == goal.position]
    
    def _get_skills_of_agents_present_at_goal(self, goal: Goal):
        agents = self._get_agents_present_at_goal(goal)
        if not agents:
            return []
        skills = []
        for agent in agents:
            skills.extend(agent.skills)
        return skills
    
    def _goal_can_be_claimed(self, goal: Goal):
        skills_of_agents_present = self._get_skills_of_agents_present_at_goal(goal)
        if not skills_of_agents_present:
            return False
        skills_required = goal.required_skills
        if set(skills_required).issubset(set(skills_of_agents_present)):
            return True
    
    def reset(self):
        self.goal_assignments = {}
        self.unclaimed_goals = [goal for goal in self.goals if not goal.claimed]
        self.last_visited_goals = {agent: None for agent in self.agents}
    
    def all_goals_claimed(self):
        return len(self.unclaimed_goals) == 0
    
    def update_goal_statuses(self):
        logging.debug("Updating goal statuses")
        claimed_a_goal = False
        for goal in self.goals:
            if goal.claimed:
                continue
            if self._goal_can_be_claimed(goal):
                goal.claimed = True
                claimed_a_goal = True
        self.unclaimed_goals = [goal for goal in self.goals if not goal.claimed]
        logging.debug("Goal statuses updated")
        return claimed_a_goal

    # SOLVERS            
    
    def find_fast_solution(self):
        logging.debug("Using fast solver")
        cheapest_combinations = {} # goal: (combination, cost)
        full_solution: Dict[Agent, List[Goal]] = {} # agent: [goal]
        unaccounted_for_goals = set(self.unclaimed_goals)
        while len(full_solution) != len(self.agents):
            for goal in unaccounted_for_goals:
                sorted_combinations = iter(sorted(goal.agent_combinations_which_solve_goal.items(), key=lambda combo_and_cost: combo_and_cost[1]))
                looking_for_goal_solution = True
                while looking_for_goal_solution:
                    try:
                        (cheapest_goal_combination,cost) = next(sorted_combinations)
                        looking_for_goal_solution = False
                    except StopIteration:
                        break
                    if any([agent in full_solution for agent in cheapest_goal_combination]):
                        looking_for_goal_solution = True
                        continue
                if not looking_for_goal_solution:
                    cheapest_combinations[goal] = (cheapest_goal_combination,cost)
            
            cheapest_goals_sorted = iter(sorted(cheapest_combinations.items(), key=lambda goal_and_combo_and_cost: goal_and_combo_and_cost[1][1]))

            goal_available = True
            while True:
                try:
                    cheapest_goal, (cheapest_combination, cost) = next(cheapest_goals_sorted)
                except StopIteration:
                    goal_available = False
                    break

                if any([agent in full_solution for agent in cheapest_combination]):
                    continue
                else:
                    break
            
            if not goal_available:
                cheapest_combinations.clear()
                break

            for agent in cheapest_combination:
                full_solution[agent] = [cheapest_goal]

            unaccounted_for_goals.remove(cheapest_goal)
            cheapest_combinations.clear()

        return full_solution, cost
    