from torch_geometric.data import InMemoryDataset
class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super(MyDataset,self).__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])

def compare_observation_and_solution_graph(observation_graph, solution_graph):
    import networkx as nx
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Observation Graph")
    colors = ['blue' if observation_graph.nodes[node]['type'] == 1 else 'red' for node in observation_graph.nodes]
    nx.draw(observation_graph, with_labels=True, node_color=colors)
    plt.subplot(122)
    plt.title("Solution Graph")
    colors = ['blue' if solution_graph.nodes[node]['type'] == 1 else 'red' for node in solution_graph.nodes]
    nx.draw(solution_graph, with_labels=True, node_color=colors)
    plt.show(block=False)