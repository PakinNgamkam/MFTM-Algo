# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import open3d as o3d

# def read_xyz(file_path):
#     """Read XYZ data from a file."""
#     points = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             x, y, z = map(float, line.strip().split())
#             points.append((x, y, z))
#     return np.array(points)

# def filter_duplicate_with_lowest_y(points):
#     """Filter out points that have the same (x, z), appear exactly twice, 
#     and one of them has the lowest y-value in the entire point cloud."""
    
#     # Group points by their (x, z) coordinates
#     grouped_points = defaultdict(list)

#     for point in points:
#         x, y, z = point
#         grouped_points[(x, z)].append((x, y, z))  # Store the entire point, not just y

#     # Find the globally lowest y-value across the entire point cloud
#     lowest_y_global = np.min(points[:, 1])

#     # Filter out groups with exactly two points where one of them has the lowest y-value globally
#     filtered_points = []
#     for (x, z), point_group in grouped_points.items():
#         if len(point_group) == 2:  # Exactly two points
#             # Check if one of the points has the globally lowest y-value
#             if any(y == lowest_y_global for _, y, _ in point_group):
#                 continue  # Eliminate both points if one has the globally lowest y-value
#         # Keep all other points
#         filtered_points.extend(point_group)

#     return np.array(filtered_points)

# def visualize_with_open3d(points, colors):
#     """Visualize point cloud using Open3D with the same colors as the 2D plot."""
#     # Create an Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()

#     # Set the points to the Open3D point cloud
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # Set colors for Open3D point cloud (from matplotlib colormap)
#     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Open3D expects RGB values in [0, 1]

#     # Visualize the point cloud
#     o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")
#     ####################################################################

# def xyz_to_image(xyz_file_path, output_image_path, padding_pixels=50, image_size=(500, 500), meter_interval=1):
#     # Read XYZ data
#     points = read_xyz(xyz_file_path)

#     # Filter out duplicate points with negative y-values
#     points = filter_duplicate_with_lowest_y(points)

#     # Invert the x-axis for left-handed coordinate system
#     points[:, 0] = -points[:, 0]  # Invert the x-axis

#     # Sort points by y-values so that higher points are plotted last (on top)
#     points = points[np.argsort(points[:, 1])]

#     # Get x, z for 2D plotting, and y for the gradient color
#     x_vals = points[:, 0]
#     z_vals = points[:, 2]
#     y_vals = points[:, 1]  # Use y for color gradient

#     ################################################################
#     # Create a discrete color map where the color changes every 1 meter
#     min_y = np.min(y_vals)
#     max_y = np.max(y_vals)

#     # Define color steps, one color per meter interval
#     meter_bins = np.arange(min_y, max_y + meter_interval, meter_interval)
    
#     # Use a fixed colormap (gist_rainbow or any other colormap) to assign colors for each interval
#     cmap = plt.cm.gist_rainbow
#     norm = plt.Normalize(vmin=min_y, vmax=max_y)
    
#     # Find which bin each point's y value falls into (rounded down to the nearest meter)
#     y_bin_indices = np.digitize(y_vals, meter_bins) - 1  # Subtract 1 to get the index for bins

#     # Assign colors based on the bins
#     colors = cmap(norm(meter_bins[y_bin_indices]))  # Map bin indices to colors

#     # Visualize the filtered point cloud with Open3D using the same colors
#     visualize_with_open3d(points, colors)
#     ####################################################################

#     # Calculate the bounding box of the point cloud
#     min_x, max_x = np.min(x_vals), np.max(x_vals)
#     min_z, max_z = np.min(z_vals), np.max(z_vals)

#     # Calculate the full range for both x and z
#     range_x = max_x - min_x
#     range_z = max_z - min_z

#     # Determine the maximum range for symmetric limits around (0, 0)
#     max_range = max(abs(min_x), abs(max_x), abs(min_z), abs(max_z))

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)

#     # Plot the points in the xz plane, applying the color based on the meter intervals
#     scatter = ax.scatter(x_vals, z_vals, c=colors, s=1) 

#     # Convert the padding in pixels to data units (assuming square image for simplicity)
#     fig_width_inch = image_size[0] / 100
#     fig_height_inch = image_size[1] / 100
#     data_padding_x = (padding_pixels / 100) * (range_x / fig_width_inch)
#     data_padding_z = (padding_pixels / 100) * (range_z / fig_height_inch)

#     # Set limits with (0, 0) in the center of the plot and padding
#     ax.set_xlim(-max_range - data_padding_x, max_range + data_padding_x)
#     ax.set_ylim(-max_range - data_padding_z, max_range + data_padding_z)

#     # Set equal aspect ratio to prevent distortion
#     ax.set_aspect('equal', 'box')

#     # Remove axes for a cleaner image
#     ax.axis('off')

#     # Save the figure as an image
#     plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # Example usage
# xyz_file_path = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/firstFloorSouth_even.xyz'
# output_image_path = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/firstFloorSouth_even.png'
# xyz_to_image(xyz_file_path, output_image_path, padding_pixels=10)
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import open3d as o3d
from PIL import Image

def read_xyz(file_path):
    """Read XYZ data from a file."""
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            points.append((x, y, z))
    return np.array(points)

def filter_duplicate_with_lowest_y(points, error_range=0.05):
    """Filter out points that have the same (x, z), appear exactly twice, 
    and one of them has the lowest y-value in the entire point cloud within an error range."""
    
    # Group points by their (x, z) coordinates
    grouped_points = defaultdict(list)

    for point in points:
        x, y, z = point
        grouped_points[(x, z)].append((x, y, z))  # Store the entire point, not just y

    # Find the globally lowest y-value across the entire point cloud
    lowest_y_global = np.min(points[:, 1])

    # Filter out groups with exactly two points where one of them has the lowest y-value globally within the error range
    filtered_points = []
    for (x, z), point_group in grouped_points.items():
        if len(point_group) == 2:  # Exactly two points
            # Check if one of the points has a y-value in the range [lowest_y_global, lowest_y_global + error_range]
            if any(lowest_y_global <= y <= lowest_y_global + error_range for _, y, _ in point_group):
                continue  # Eliminate both points if one falls within the error range
        # Keep all other points
        filtered_points.extend(point_group)

    return np.array(filtered_points)

def visualize_with_open3d(points, colors):
    """Visualize point cloud using Open3D with the same colors as the 2D plot."""
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points to the Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points)

    # Set colors for Open3D point cloud (from matplotlib colormap)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Open3D expects RGB values in [0, 1]

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")
    ####################################################################

def xyz_to_image(xyz_file_path, output_image_path, boundary, padding_pixels=50, image_size=(500, 500), meter_interval=1, is_blueprint=False):
    """
    Converts an XYZ file to an image and optionally calculates pixels per unit if it's a blueprint.
    """
    # Read XYZ data
    points = read_xyz(xyz_file_path)

    # Filter out duplicate points with negative y-values
    points = filter_duplicate_with_lowest_y(points)

    # Invert the x-axis for left-handed coordinate system
    points[:, 0] = -points[:, 0]  # Invert the x-axis

    # Sort points by y-values so that higher points are plotted last (on top)
    points = points[np.argsort(points[:, 1])]

    # Get x, z for 2D plotting, and y for the gradient color
    x_vals = points[:, 0]
    z_vals = points[:, 2]
    y_vals = points[:, 1]  # Use y for color gradient

    ################################################################
    # Create a discrete color map where the color changes every 1 meter
    min_y = np.min(y_vals)
    max_y = np.max(y_vals)

    # Define color steps, one color per meter interval
    meter_bins = np.arange(min_y, max_y + meter_interval, meter_interval)
    
    # Use a fixed colormap (gist_rainbow or any other colormap) to assign colors for each interval
    cmap = plt.cm.gist_rainbow
    norm = plt.Normalize(vmin=min_y, vmax=max_y)
    
    # Find which bin each point's y value falls into (rounded down to the nearest meter)
    y_bin_indices = np.digitize(y_vals, meter_bins) - 1  # Subtract 1 to get the index for bins

    # Assign colors based on the bins
    colors = cmap(norm(meter_bins[y_bin_indices]))  # Map bin indices to colors

    # Visualize the filtered point cloud with Open3D using the same colors
    visualize_with_open3d(points, colors)
    ####################################################################

    # Calculate the bounding box using the blueprint's boundary
    min_x, max_x, min_z, max_z = boundary

    # Calculate the full range for both x and z
    range_x = max_x - min_x
    range_z = max_z - min_z

    # Determine the maximum range for symmetric limits around (0, 0)
    max_range = max(abs(min_x), abs(max_x), abs(min_z), abs(max_z))

    # Create the plot
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)

    # Plot the points in the xz plane, applying the color based on the meter intervals
    scatter = ax.scatter(x_vals, z_vals, c=colors, s=1) 

    # Convert the padding in pixels to data units (assuming square image for simplicity)
    fig_width_inch = image_size[0] / 100
    fig_height_inch = image_size[1] / 100
    data_padding_x = (padding_pixels / 100) * (range_x / fig_width_inch)
    data_padding_z = (padding_pixels / 100) * (range_z / fig_height_inch)

    # Set limits with (0, 0) in the center of the plot and padding
    ax.set_xlim(-max_range - data_padding_x, max_range + data_padding_x)
    ax.set_ylim(-max_range - data_padding_z, max_range + data_padding_z)

    # Set equal aspect ratio to prevent distortion
    ax.set_aspect('equal', 'box')

    # Remove axes for a cleaner image
    ax.axis('off')

    # Save the figure as an image
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Calculate pixels per unit only if this image is marked as a blueprint
    if is_blueprint:
        calculate_pixels_per_unit_from_image(output_image_path, range_x, range_z)

def calculate_pixels_per_unit_from_image(image_path, range_x, range_z):
    """Calculate pixels per unit using the image and XYZ range (range_x and range_z)."""
    # Open the saved image to get its actual dimensions
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert the image to grayscale for easier analysis
    gray_img = np.mean(img_array, axis=2)

    # Find the first and last non-white pixels along the X and Z axes
    x_non_empty = np.where(np.any(gray_img != 255, axis=0))[0]
    z_non_empty = np.where(np.any(gray_img != 255, axis=1))[0]

    first_x_pixel, last_x_pixel = x_non_empty[0], x_non_empty[-1]
    first_z_pixel, last_z_pixel = z_non_empty[0], z_non_empty[-1]

    # Calculate the actual width and height of the plotted area in pixels
    effective_width = last_x_pixel - first_x_pixel
    effective_height = last_z_pixel - first_z_pixel

    # Calculate pixels per unit using the effective width and height
    pixels_per_unit_x = effective_width / range_x
    pixels_per_unit_z = effective_height / range_z

    # Print the results
    print(f"Image dimensions (width x height): {img.size[0]} x {img.size[1]}")
    print(f"Effective plotted width (pixels): {effective_width}")
    print(f"Effective plotted height (pixels): {effective_height}")
    print(f"Pixels per unit (X): {pixels_per_unit_x}")
    print(f"Pixels per unit (Z): {pixels_per_unit_z}")

    # Take the minimum of the two to avoid distortion
    final_pixels_per_unit = min(pixels_per_unit_x, pixels_per_unit_z)
    print(f"Final pixels per unit (min of X and Z): {final_pixels_per_unit}")


# Example usage
xyz_file_path_blueprint = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/room1_test_even.xyz'
xyz_file_path_second = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/room1_testcase1_even.xyz'
output_image_path_blueprint = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/room1_test_even_output.png'
output_image_path_second = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/room1_testcase1_even_output.png'

# Read the blueprint data and get the bounding box
blueprint_points = read_xyz(xyz_file_path_blueprint)

# Calculate the boundary of the blueprint
min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

# Generate an image for the blueprint and calculate pixels per unit
xyz_to_image(xyz_file_path_blueprint, output_image_path_blueprint, boundary, padding_pixels=50, image_size=(500, 500), is_blueprint=True)

# Generate an image for a non-blueprint file (no pixels per unit calculation)
xyz_to_image(xyz_file_path_second, output_image_path_second, boundary, padding_pixels=50, image_size=(500, 500), is_blueprint=False)
