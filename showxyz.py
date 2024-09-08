import numpy as np
import pyvista as pv
import argparse

# Function to read the .xyz file
def read_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)

# Set up argument parser to accept file path as a command-line argument
parser = argparse.ArgumentParser(description='Display a 3D point cloud from an .xyz file.')
parser.add_argument('file', type=str, help='Path to the .xyz file')

# Parse the arguments
args = parser.parse_args()

# Read the .xyz file provided via the command line
file_path = args.file
points = read_xyz_file(file_path)

# Create a PyVista point cloud
point_cloud = pv.PolyData(points)

# Create a plotter object
plotter = pv.Plotter()

# Add the point cloud to the plotter
plotter.add_mesh(point_cloud, color='blue', point_size=5.0)

# Enable interaction with the 3D point cloud
plotter.show_grid()

# Display the 3D point cloud interactively
plotter.show()
