import logging

from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer
from planning_sandbox.agent_class import Agent
from planning_sandbox.goal_class import Goal

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

while True:
    num_agents = 1
    num_goals = 1
    num_skills = 1
    size = 64
    visualisation_speed = 50 # Max 200
    solve_type = 'optimal' # 'optimal' or 'fast'
    use_map = True
    random_map = True
    assume_lander = False

    # custom_agents = []
    # custom_goals = []

    # custom_agents.append(Agent(initial_position=(0, 0), skills=[0]))
    # custom_agents.append(Agent(initial_position=(31, 31), skills=[1]))
    # custom_goals.append(Goal(position=(30, 1), skills=[0]))
    # custom_goals.append(Goal(position=(1, 30), skills=[0]))
    # custom_goals.append(Goal(position=(15, 15), skills=[0,1]))

    # For custom goals/agents
    # my_environment = Environment(size=size, num_skills=num_skills, use_geo_data=use_map, custom_agents=custom_agents, custom_goals=custom_goals)

    # For random goals/agents
    my_environment = Environment(size=size, num_agents=num_agents, num_goals=num_goals, num_skills=num_skills, use_geo_data=use_map, random_map=random_map, assume_lander=assume_lander)

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
