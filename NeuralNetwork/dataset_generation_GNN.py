import os
import logging
import pickle

import planning_sandbox as ps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_directory = os.path.dirname(os.path.abspath(__file__))

def load_pickled_objects(filename):
    objects = []
    with open(filename, 'rb') as f:
        while True:
            try:
                obj = pickle.load(f)
                objects.append(obj)
            except EOFError:
                break
    return objects

# check if graph and solution files exist
if not os.path.exists(current_directory+'/GNN_graphs.pkl') or not os.path.exists(current_directory+'/GNN_solutions.pkl'):
    graphs = []
    solutions = []
else:
    graphs = load_pickled_objects(current_directory+'/GNN_graphs.pkl')[0]
    solutions = load_pickled_objects(current_directory+'/GNN_solutions.pkl')[0]
    print(f"Loaded {len(graphs)} graphs and {len(solutions)} solutions.")

num_agents = 2
num_goals = 5
num_skills = 3
size = 64
use_geo_data = True
loops = 1000000
save_interval = 100

print("Creating environment")
env = ps.Environment(size=size,num_skills=num_skills, num_agents=num_agents, num_goals=num_goals,use_geo_data=True,assume_lander=False)
for _ in range(loops):
    print(f"Amount of samples: {len(graphs)}", end="\r")
    env.find_numerical_solution()
    solution = env.get_action_vector()
    graph = env.grid_map.graph.copy()

    agent_nodes = []
    for idx,agent in enumerate(env.agents):
        agent_node = {
            "x": agent.position[0],
            "y": agent.position[1],
            "skills": [1 if skill in agent.skills else 0 for skill in range(num_skills)]
        }
        graph.add_node('agent'+str(idx), **agent_node)
        graph.add_edge((agent.position[0],agent.position[1]),'agent'+str(idx),weight=0)
        agent_nodes.append(agent_node)

    goal_nodes = []
    for idx,goal in enumerate(env.goals):
        goal_node = {
            "x": goal.position[0],
            "y": goal.position[1],
            "skills": [1 if skill in goal.required_skills else 0 for skill in range(num_skills)]
        }
        graph.add_node('goal'+str(idx), **goal_node)
        graph.add_edge((goal.position[0],goal.position[1]),'goal'+str(idx),weight=0)
        goal_nodes.append(goal_node)

    for idx,goal in enumerate(env.goals):
        for agent in goal.agents_which_have_required_skills:
            agent_node = graph.nodes['agent'+str(env.agents.index(agent))]
            goal_node = graph.nodes['goal'+str(idx)]
            graph.add_edge('goal'+str(idx),'agent'+str(env.agents.index(agent)),weight=0)
    
    graphs.append(graph)
    solutions.append(solution)

    if len(graphs) % save_interval == 0:
        # After the loop, ensure all data is saved
        print("Saving data to:" + current_directory)
        with open(current_directory+'/GNN_graphs.pkl', 'wb') as f_graphs:
            pickle.dump(graphs, f_graphs)

        with open(current_directory+'/GNN_solutions.pkl', 'wb') as f_solutions:
            pickle.dump(solutions, f_solutions)

    env.reset()

# After the loop, ensure all data is saved
print("Final save of data to:" + current_directory)
with open(current_directory+'/GNN_graphs.pkl', 'wb') as f_graphs:
    pickle.dump(graphs, f_graphs)

with open(current_directory+'/GNN_solutions.pkl', 'wb') as f_solutions:
    pickle.dump(solutions, f_solutions)

print("All graphs and solutions have been saved.")

    # # show graph
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,10))
    # pos = nx.spring_layout(graph)
    # nx.draw(graph,pos, with_labels=True)
    # plt.show()
