import numpy as np
import os
from madskills.environment.grid_map_class import GridMap

class MapGenerator():
    def __init__(self, amount_of_maps, save_path, save_frequency=100, map_size=64):
        # Parameters
        self.amount_of_maps = amount_of_maps
        self.save_path = save_path
        self.save_frequency = save_frequency 
        self.map_size = map_size
        self.existing_maps = []
        self._init()

    def _init(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Check if file exists to load previous maps, otherwise start fresh
        if os.path.exists(self.save_path):
            self.existing_maps = list(np.load(self.save_path, allow_pickle=True))
            print(f"Loaded {len(self.existing_maps)} maps from {self.save_path}")
        else:
            self.existing_maps = []
            print(f"No existing file found at {self.save_path}. Starting fresh.")

    def generate_maps(self):
        # Generate and save maps in chunks
        for i in range(self.amount_of_maps):
            print(f"Generating Map {i+1}/{self.amount_of_maps}", end="\r")
            random_map = GridMap(size=self.map_size, use_geo_data=True, random_map=True).downscaled_data
            self.existing_maps.append(random_map*100)

            # Periodic saving
            if (i + 1) % self.save_frequency == 0 or i == self.amount_of_maps - 1:
                self.save_maps()

    def save_maps(self):
        # Final save after all maps are generated
        np.save(self.save_path, np.array(self.existing_maps, dtype=object))
        print(f"\nSave completed with a total of {len(self.existing_maps)} maps.")