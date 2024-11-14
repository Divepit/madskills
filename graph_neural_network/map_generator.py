import numpy as np
import os
from planning_sandbox.environment.grid_map_class import GridMap

# Parameters
amount_of_maps = 10000
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'maps', 'random_maps_64.npy')  # Path to save the map data
save_frequency = 100  # Save progress every 100 maps
map_size = 64

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Check if file exists to load previous maps, otherwise start fresh
if os.path.exists(file_path):
    existing_maps = list(np.load(file_path, allow_pickle=True))
    print(f"Loaded {len(existing_maps)} maps from {file_path}")
else:
    existing_maps = []
    print(f"No existing file found at {file_path}. Starting fresh.")

# Generate and save maps in chunks
for i in range(amount_of_maps):
    print(f"Generating Map {i+1}/{amount_of_maps}", end="\r")
    random_map = GridMap(size=map_size, use_geo_data=True, random_map=True).downscaled_data
    existing_maps.append(random_map*100)

    # Periodic saving
    if (i + 1) % save_frequency == 0:
        np.save(file_path, np.array(existing_maps, dtype=object))
        print(f"\nProgress saved at {len(existing_maps)} maps.")

# Final save after all maps are generated
np.save(file_path, np.array(existing_maps, dtype=object))
print(f"\nFinal save completed with a total of {len(existing_maps)} maps.")

print("done")