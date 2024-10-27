from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal

num_agents = 2
num_goals = 3
num_skills = 2
size = 32
visualisation_speed = 20 # Max 200
solve_type = 'optimal' # 'optimal' or 'fast'
use_map = False

agent_1 = Agent(initial_position=(0, 0), skills=[0])
agent_2 = Agent(initial_position=(31, 31), skills=[1])
goal_1 = Goal(position=(30, 1), skills=[0])
goal_2 = Goal(position=(1, 30), skills=[0])
goal_3 = Goal(position=(15, 15), skills=[0,1])


my_environment = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=use_map, custom_agents=[agent_1, agent_2], custom_goals=[goal_1, goal_2, goal_3])
my_visualiser = Visualizer(my_environment, speed=visualisation_speed)

my_environment.find_numerical_solution(solve_type=solve_type)
my_solution = my_environment.full_solution

total_steps, steps_waited, total_cost, solve_time, amount_of_claimed_goals = my_environment.solve_full_solution()

my_visualiser.visualise_full_solution()

print(f"Total steps taken: {total_steps}")
print(f"Total steps waited: {steps_waited}")
print(f"Total cost: {int(total_cost)}")
print(f"Solve time: {int(solve_time*1000)} ms")
print(f"Amount of claimed goals: {amount_of_claimed_goals}")
