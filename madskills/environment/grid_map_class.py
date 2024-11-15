import networkx as nx
import numpy as np
from noise import pnoise2
import logging
import os
from PIL import Image
from skimage.transform import resize

from madskills.environment.agent_class import Agent
from madskills.environment.goal_class import Goal

current_directory = os.path.dirname(os.path.abspath(__file__))


TIF = current_directory+'/maps/shoemaker_ele_5mpp.tif'
MPP = 5
WINDOW_SIZE = 4000
X_OFFSET = 0
Y_OFFSET = 0
ORIGINAL_TOP_LEFT = (X_OFFSET, Y_OFFSET)
ORIGINAL_BOTTOM_RIGHT = (X_OFFSET + WINDOW_SIZE, Y_OFFSET + WINDOW_SIZE)

class GridMap:
    def __init__(self, size, use_geo_data=False, downhill_slope_max=np.inf, uphill_slope_max=np.inf, uphill_factor=1, random_map=False):
    
        self.use_geo_data = use_geo_data
        self.use_random_map = random_map
        
        self.downhill_slope_max = downhill_slope_max
        self.uphill_slope_max = uphill_slope_max
        self.uphill_factor = uphill_factor

        self.seed = 0

        self.size = size
        self.graph = None

        self.path_lookup_table = {}

        self.paths = {}
        
        if self.use_geo_data:
            if self.use_random_map:
                self.data = self._generate_random_map()
            else:
                self.data = self._extract_data_from_tif()
            self.downscaled_data, self.pixel_size = self._downscale_data()
        else:
            self.data = np.zeros((size, size))
            self.downscaled_data = self.data
            self.pixel_size = MPP
        
        self._create_directed_graph(data=self.downscaled_data, pixel_size=self.pixel_size, uphill_factor=uphill_factor, downhill_slope_max=downhill_slope_max, uphill_slope_max=uphill_slope_max)

    def _random_position(self):
        return (np.random.randint(0, self.size), np.random.randint(0, self.size))
    
    # map generator function partially created by chatGPT (https://chatgpt.com/share/672ce6f8-bc90-8008-b221-ea49ba881e74)
    def _generate_random_map(self):
            map_size = self.size
            scale = 100
            octaves = 6
            persistence = 0.5
            lacunarity = 2.0

            # Increment the seed slightly each time for new but consistent patterns
            self.seed = np.random.randint(1, 2000)

            x_offset = np.random.randint(0, 1000)
            y_offset = np.random.randint(0, 1000)

            terrain_map = np.zeros((map_size, map_size))
            for i in range(map_size):
                for j in range(map_size):
                    height = pnoise2(
                        (i + x_offset) / scale,
                        (j + y_offset) / scale,
                        octaves=octaves, 
                        persistence=persistence, 
                        lacunarity=lacunarity, 
                        repeatx=map_size, 
                        repeaty=map_size, 
                        base=self.seed
                    )
                    terrain_map[i][j] = height

            return terrain_map
    
    def _print_tif_info(self,file_path):
        logging.debug("TIFF File Information:")
        logging.debug("-----------------------")
        
        with Image.open(file_path) as img:
            width, height = img.size
            logging.debug(f"Dimensions: {width} x {height}")
            data = np.array(img)
            min_val = np.min(data)
            max_val = np.max(data)
            logging.debug(f"Min value: {min_val:.2f}")
            logging.debug(f"Max value: {max_val:.2f}")
            
        logging.debug("-----------------------")

    
    def _extract_data_from_tif(self):
        self._print_tif_info(TIF)
        with Image.open(TIF) as img:
            cropped = img.crop((ORIGINAL_TOP_LEFT[0], ORIGINAL_TOP_LEFT[1], ORIGINAL_BOTTOM_RIGHT[0], ORIGINAL_BOTTOM_RIGHT[1]))
            data = np.array(cropped)
        return data
    
    def _downscale_data(self):

        current_height, current_width = self.data.shape
        current_pixel_size = MPP * max(current_height, current_width) / max(self.size, self.size)

        scale_factor = current_pixel_size / MPP

        new_height = int(current_height / scale_factor)
        new_width = int(current_width / scale_factor)

        downscaled_data = resize(self.data, (new_height, new_width), order=1, mode='reflect', anti_aliasing=True)

        new_pixel_size = MPP * max(current_height, current_width) / max(new_height, new_width)

        return downscaled_data, new_pixel_size
    
    def _get_current_index_on_path(self, agent: Agent):
        return self.paths[agent].index(agent.position)
    
    def _get_move_to_reach_position(self, agent: Agent, next_position):
        current_position = agent.position
        next_position = next_position
        if current_position[0] == next_position[0] and current_position[1] == next_position[1] - 1:
            return 'down'
        elif current_position[0] == next_position[0] and current_position[1] == next_position[1] + 1:
            return 'up'
        elif current_position[0] == next_position[0] - 1 and current_position[1] == next_position[1]:
            return 'right'
        elif current_position[0] == next_position[0] + 1 and current_position[1] == next_position[1]:
            return 'left'
        elif current_position[0] == next_position[0] and current_position[1] == next_position[1]:
            return 'stay'
        else:
            return 'moveto'
        
    def _is_valid_position(self, pos):
        return (0 <= pos[0] < self.size and 
                0 <= pos[1] < self.size)
    
    def _get_cost_for_move(self, start, goal):
        if start == goal:
            return 0
        if self.use_geo_data:
            return self.graph.get_edge_data(start, goal)['weight']
        else:
            return MPP
        
    def _create_directed_graph(self, data, pixel_size, uphill_factor, downhill_slope_max, uphill_slope_max):
        np.set_printoptions(linewidth=100000)
        height, width = data.shape
        G = nx.DiGraph()
        for i in range(height):
            for j in range(width):
                node = (i, j)
                node_elevation = data[i, j]
                
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for ni, nj in neighbors:
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_node = (ni, nj)
                        neighbor_elevation = data[ni, nj]
                        elevation_diff = neighbor_elevation - node_elevation
                        slope = elevation_diff / pixel_size
                        weight = self._calculate_weight(slope)
                        
                        G.add_edge(node, neighbor_node, weight=weight)
        for i in range(height):
            for j in range(width):
                node = (i, j)
                node_elevation = data[i, j]
                G.nodes[node]["elevation"] = node_elevation
                G.nodes[node]["x"] = i
                G.nodes[node]["y"] = j

        self.graph = G

    def _calculate_weight(self, slope):
        if slope > 0:
            if slope > self.uphill_slope_max:
                weight = np.inf
            else:
                weight = slope * self.uphill_factor
        else:
            if abs(slope) > self.downhill_slope_max:
                weight = np.inf
            else:
                weight = abs(slope)
        
        weight += 0.1 * MPP
        return weight
    
    def _get_next_index_on_path(self, agent: Agent):
        if self._get_current_index_on_path(agent) == len(self.paths[agent]) - 1:
            return self._get_current_index_on_path(agent)
        return self._get_current_index_on_path(agent) + 1
    
    def _get_next_position_on_path(self, agent: Agent):
        if agent in self.paths:
            return self.paths[agent][self._get_next_index_on_path(agent)]
        return agent.position
        
    def calculate_path_cost(self, path):
        start = path[0]
        goal = path[-1]
        if (start,goal) in self.path_lookup_table:
            return self.path_lookup_table[(start,goal)][1]
        elif self.use_geo_data:
            return nx.path_weight(self.graph, path, weight="weight")
        else:
            return nx.path_weight(self.graph, path, weight="weight")
    
    def soft_reset(self):
        self.paths.clear()

    def reset(self):
        self.paths.clear()
        if self.use_geo_data and self.use_random_map:
            self.data = self._generate_random_map()
            self.downscaled_data, self.pixel_size = self._downscale_data()
            self._create_directed_graph(data=self.downscaled_data, pixel_size=self.pixel_size, uphill_factor=self.uphill_factor, downhill_slope_max=self.downhill_slope_max, uphill_slope_max=self.uphill_slope_max)
    
    def random_valid_position(self):
        pos = self._random_position()
        while not self._is_valid_position(pos):
            pos = self._random_position()
        return pos
    
    def shortest_path(self, start, goal):
        if (start, goal) in self.path_lookup_table:
            path, cost = self.path_lookup_table[(start, goal)]
        if self.use_geo_data:
            path = nx.astar_path(G=self.graph, source=start, target=goal, weight="weight")
        else:
            path = nx.astar_path(G=self.graph, source=start, target=goal)
        
        cost = self.calculate_path_cost(path)
        self.path_lookup_table[(start, goal)] = (path, cost)

        return path

    def generate_shortest_path_for_agent(self, agent: Agent, goal: Goal):
        if agent in self.paths:
            current_agent_goal = self.paths[agent][-1]
        else:
            current_agent_goal = None

        if current_agent_goal is not None and current_agent_goal == goal.position:
            path_index = self._get_current_index_on_path(agent)
            return self.paths[agent][path_index:]
        
        start = agent.position
        goal = goal.position
        path = self.shortest_path(start, goal)

        return path
        
    
    def get_move_and_cost_to_reach_next_position(self, agent: Agent):
        current_position = agent.position
        next_position = self._get_next_position_on_path(agent)
        move_cost = self._get_cost_for_move(current_position, next_position)
        return self._get_move_to_reach_position(agent, next_position), move_cost
    
    def assign_shortest_path_for_goal_to_agent(self, agent: Agent, goal: Goal):
        path = self.generate_shortest_path_for_agent(agent, goal) 
        self.paths[agent] = path