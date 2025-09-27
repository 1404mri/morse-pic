# Auto-generated visualization code
import trimesh
import pyvista as pv
import numpy as np
import random
import os

# --- Difficulty Control ---
difficulty_level = 3 # Corresponds to 'intermediate' from difficulty_parameters

# --- Parameters based on difficulty_parameters ---
num_data_points = 25
decimal_places = 3 # Not directly used for object positions, but for consistency if needed
num_categories = 6
visual_elements = 6 # Number of distinct object types

output_path = "synthetic_scene_visualization.png"

# --- Generate Data ---

# Define available shapes and colors to create num_categories distinct visual elements
available_shapes = ['sphere', 'cube', 'cylinder']
available_colors = ['red', 'blue'] # Using two colors to combine with 3 shapes for 6 categories

# Create a list of all possible category combinations
all_categories = []
for shape in available_shapes:
    for color in available_colors:
        all_categories.append(f"{color}_{shape}")

# Select the exact number of categories required
selected_categories = all_categories[:num_categories]

scene_objects = []
for i in range(num_data_points):
    obj_category = random.choice(selected_categories)
    color, shape = obj_category.split('_')
    
    # Random position within a reasonable range
    x = round(random.uniform(-5, 5), decimal_places)
    y = round(random.uniform(-5, 5), decimal_places)
    z = round(random.uniform(-2, 2), decimal_places) # Keep z somewhat flat for easier viewing
    
    scene_objects.append({
        'id': i,
        'category': obj_category,
        'shape': shape,
        'color': color,
        'position': [x, y, z]
    })

# --- Create Visualization ---
plotter = pv.Plotter(off_screen=True)
plotter.set_background('white')

# Map color names to PyVista/Trimesh compatible colors
color_map = {
    'red': 'red',
    'blue': 'blue',
    'green': 'green',
    'yellow': 'yellow',
    'purple': 'purple',
    'orange': 'orange'
}

for obj_data in scene_objects:
    shape_type = obj_data['shape']
    color_name = obj_data['color']
    position = obj_data['position']
    
    mesh = None
    if shape_type == 'sphere':
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    elif shape_type == 'cube':
        mesh = trimesh.creation.box(extents=[1, 1, 1])
    elif shape_type == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=0.5, height=1)
    
    if mesh:
        # Translate the mesh to its position
        mesh.apply_translation(position)
        
        # Add to plotter
        plotter.add_mesh(mesh, color=color_map.get(color_name, 'gray'), smooth_shading=True)

# Set camera position for a good view of the scene
plotter.camera_position = [(10, 10, 10), (0, 0, 0), (0, 0, 1)] # (position, focal_point, view_up)
plotter.show(screenshot=output_path, auto_close=True)

final_image_path = output_path

# --- Calculate Ground Truth ---
# For the adapted question, let's pick two categories: 'red_sphere' and 'blue_cube'
target_category_1 = 'red_sphere'
target_category_2 = 'blue_cube'

count_cat1 = sum(1 for obj in scene_objects if obj['category'] == target_category_1)
count_cat2 = sum(1 for obj in scene_objects if obj['category'] == target_category_2)

absolute_difference = abs(count_cat1 - count_cat2)

ground_truth = {
    "count_of_red_spheres": count_cat1,
    "count_of_blue_cubes": count_cat2,
    "absolute_difference": absolute_difference
}