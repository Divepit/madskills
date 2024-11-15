import os
from madskills.environment.map_generator_class import MapGenerator

current_directory = os.path.dirname(os.path.abspath(__file__))

gen = MapGenerator(amount_of_maps=100, save_path=current_directory+'/maps/random_maps.npy', map_size=64)
gen.generate_maps()