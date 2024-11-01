import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
# import open3d as o3d
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
    and one of them has the lowest y-value in the entire point cloud within an error range 
    while the other point's y-value is below zero."""
    
    # Group points by their (x, z) coordinates
    grouped_points = defaultdict(list)

    for point in points:
        x, y, z = point
        grouped_points[(x, z)].append((x, y, z))  # Store the entire point, not just y

    # Find the globally lowest y-value across the entire point cloud
    lowest_y_global = np.min(points[:, 1])

    # Filter out groups with exactly two points where one of them has the lowest y-value globally within the error range,
    # and the second point has a y-value lower than zero.
    filtered_points = []
    for (x, z), point_group in grouped_points.items():
        if len(point_group) == 2:  # Exactly two points
            # Check if one of the points has a y-value in the range [lowest_y_global, lowest_y_global + error_range]
            # and the other point has a y-value lower than zero
            y_values = [y for _, y, _ in point_group]
            if (lowest_y_global <= y_values[0] <= lowest_y_global + error_range and y_values[1] < 0) or \
               (lowest_y_global <= y_values[1] <= lowest_y_global + error_range and y_values[0] < 0):
                continue  # Eliminate both points if the condition is met
        # Keep all other points
        filtered_points.extend(point_group)

    return np.array(filtered_points)

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
    ####################################################################

def xyz_to_image(xyz_file_path, output_image_path, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), meter_interval=1, is_blueprint=False):
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
    # visualize_with_open3d(points, colors)
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
        pixels_per_unit = calculate_pixels_per_unit_from_image(output_image_path, range_x, range_z, error_pixels)
        return pixels_per_unit

    # Return None if not a blueprint
    return None

def error_pixels_from_image(xyz_file_path, output_image_path, boundary, padding_pixels=50, image_size=(500, 500)):
    # Read XYZ data
    points = read_xyz(xyz_file_path)

    # Find the point with the maximum y-value
    max_y_index = np.argmax(points[:, 1])
    max_y_point = points[max_y_index]  # Extract the single point with the max y-value

    # Get x, z for 2D plotting (max_y_point contains a single point)
    x_vals = [max_y_point[0]]
    z_vals = [max_y_point[2]]

    # Calculate the bounding box using the blueprint's boundary
    min_x, max_x, min_z, max_z = boundary

    # Calculate the full range for both x and z
    range_x = max_x - min_x
    range_z = max_z - min_z

    # Determine the maximum range for symmetric limits around (0, 0)
    max_range = max(abs(min_x), abs(max_x), abs(min_z), abs(max_z))

    # Create the plot
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)

    colors = 'blue'

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

    # Open the saved image to get its actual dimensions
    img = Image.open(output_image_path)
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

    # Print the results
    # print(f"Image dimensions (width x height): {img.size[0]} x {img.size[1]}")
    # print(f"Error plotted width (pixels): {effective_width}")
    # print(f"Error plotted height (pixels): {effective_height}")

    # Calculate the error pixels
    error_pixels = max(effective_width, effective_height)

    return error_pixels

def calculate_pixels_per_unit_from_image(image_path, range_x, range_z, error_pixels):
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
    effective_width = last_x_pixel - first_x_pixel - error_pixels - 1
    effective_height = last_z_pixel - first_z_pixel - error_pixels - 1

    # Calculate pixels per unit using the effective width and height
    pixels_per_unit_x = effective_width / range_x
    pixels_per_unit_z = effective_height / range_z

    # Print the results
    # print(f"Image dimensions (width x height): {img.size[0]} x {img.size[1]}")
    # print(f"Effective plotted width (pixels): {effective_width}")
    # print(f"Effective plotted height (pixels): {effective_height}")
    # print(f"Range X: {range_x}")
    # print(f"Range Z: {range_z}")
    # print(f"Pixels per unit (X): {pixels_per_unit_x}")
    # print(f"Pixels per unit (Z): {pixels_per_unit_z}")

    # Take the minimum of the two to avoid distortion
    final_pixels_per_unit = min(pixels_per_unit_x, pixels_per_unit_z)
    print(f"Final pixels per unit (min of X and Z): {final_pixels_per_unit}")

    return final_pixels_per_unit

xyz_file_path_blueprint = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/firstFloorSouth_even.xyz' # secondFloor_even.xyz' # firstFloorSouth_even.xyz' # room1(2)_even.xyz'
xyz_file_path_scan = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/KitchenNorth_even.xyz' # SameSpotWest_even.xyz' # KitchenNorth_even.xyz' # RoomSecondFloor_even.xyz' # room1_testcase5_even.xyz'
output_image_path_blueprint = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/firstFloorSouth_even_output.png' # firstFloorSouth_even_output.png' # secondFloor_even_output.png' # room1(2)_even_output.png'
output_image_path_scan = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/KitchenNorth_even_output.png' # SameSpotWest_even_output.png' # KitchenNorth_even_output.png' # RoomSecondFloor_even_output.png' # room1_testcase5_even_output.png'
output_image_path_error_pixels = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/error_pixels.png'
aligned_image_path = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/aligned.png'

# Read the blueprint data and get the bounding box
blueprint_points = read_xyz(xyz_file_path_blueprint)

# Calculate the boundary of the blueprint
min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

# Generate an image for the blueprint and calculate error pixels
error_pixels = error_pixels_from_image(xyz_file_path_blueprint, output_image_path_error_pixels, boundary, padding_pixels=50, image_size=(500, 500))

# Generate an image for the blueprint and calculate pixels per unit
scale_factor = xyz_to_image(xyz_file_path_blueprint, output_image_path_blueprint, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=True)

# Generate an image for a non-blueprint file (no pixels per unit calculation)
xyz_to_image(xyz_file_path_scan, output_image_path_scan, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=False)

################################ 2nd Part ################################

def align_images_and_calculate_vector(output_image_path_scan, output_image_path_blueprint, transformation_matrix):
    """
    Align img1 with img2 using the given transformation matrix and calculate the vector difference between the centers.

    Parameters:
    - output_image_path_scan: Path to the first image (to be transformed).
    - output_image_path_blueprint: Path to the second image (reference image).
    - transformation_matrix: 2x3 transformation matrix for affine transformation.

    Returns:
    - aligned_img1: The transformed version of img1 aligned with img2.
    - img2: The reference image.
    - center_vector: The vector difference (dx, dy) between the center of img1 and img2 after alignment.
    """
    # Load the images
    img1 = cv2.imread(output_image_path_scan)
    img2 = cv2.imread(output_image_path_blueprint)

    # Ensure the images are loaded properly
    if img1 is None or img2 is None:
        raise ValueError("One or both of the image paths are invalid!")

    # Get the dimensions of img2 for warping (we want to align img1 to img2's size)
    rows, cols = img2.shape[:2]

    # Calculate the center of img1 and img2
    center_img1 = np.array([img1.shape[1] // 2, img1.shape[0] // 2, 1])  # (x, y, 1)
    center_img2 = np.array([img2.shape[1] // 2, img2.shape[0] // 2])      # (x, y)

    # Transform the center of img1 using the transformation matrix
    transformed_center_img1 = transformation_matrix @ center_img1

    # Calculate the vector difference between transformed center of img1 and center of img2
    dx = transformed_center_img1[0] - center_img2[0]
    dy = transformed_center_img1[1] - center_img2[1]
    center_vector = (dx, dy)

    # Apply the affine transformation to align img1 to img2
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (cols, rows))

    return aligned_img1, img2, center_vector, transformed_center_img1, center_img2

def feature_matching_with_geometric_constraints(img1, img2):
    """
    Perform feature matching between two images while applying geometric constraints for rotation and translation.
    Assumes that the scale does not change.
    """
    # Step 1: Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Step 2: Match descriptors using BFMatcher with cross-check to ensure mutual nearest matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Step 3: Sort the matches based on distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 4: Extract matched keypoints in both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Step 5: Estimate a transformation matrix using RANSAC
    # We use cv2.estimateAffinePartial2D with RANSAC to get a rotation and translation transformation
    transformation_matrix, inliers = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

    # Step 6: Apply the transformation to img1 to align it with img2
    rows, cols = img2.shape[:2]
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (cols, rows))

    return aligned_img1, transformation_matrix, matches, keypoints1, keypoints2, points1, points2

# Load the two images to be matched
img1 = cv2.imread(output_image_path_scan)
img2 = cv2.imread(output_image_path_blueprint)

# Perform feature matching with geometric constraints (translation and rotation only)
aligned_img1, transformation_matrix, matches, keypoints1, keypoints2, points1, points2 = feature_matching_with_geometric_constraints(img1, img2)

# Align img1 using the transformation matrix to align with img2 and calculate the vector difference
aligned_img1_with_center, img2_with_center, center_vector, transformed_center_img1, center_img2 = align_images_and_calculate_vector(
    output_image_path_scan, output_image_path_blueprint, transformation_matrix
)

# print("Original Transformation Matrix:")
# print(f"[{transformation_matrix[0, 0]}, {transformation_matrix[0, 1]}, {transformation_matrix[0, 2]}],")
# print(f"[{transformation_matrix[1, 0]}, {transformation_matrix[1, 1]}, {transformation_matrix[1, 2]}]")

# # Print the vector difference between centers
# print(f"Vector difference between centers (dx, dy): {center_vector}")

# Create the transformation matrix for XYZ file with adjusted tx and ty
xyz_transformation_matrix = transformation_matrix.copy()
xyz_transformation_matrix[0, 2] = -center_vector[0]
xyz_transformation_matrix[1, 2] = -center_vector[1]

# print("Transformation Matrix for XYZ file:")
# print(f"[{xyz_transformation_matrix[0, 0]}, {xyz_transformation_matrix[0, 1]}, {xyz_transformation_matrix[0, 2]}],")
# print(f"[{xyz_transformation_matrix[1, 0]}, {xyz_transformation_matrix[1, 1]}, {xyz_transformation_matrix[1, 2]}]")

# # Blend the aligned image with the reference image for visualization
# alpha = 0.5  # Adjust the opacity level as needed
# blended_image = cv2.addWeighted(aligned_img1_with_center, alpha, img2_with_center, 1 - alpha, 0)

# # Display the blended result with centers
# plt.figure(figsize=(6, 6))

# # Plot blended image with marked centers
# plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
# plt.scatter(transformed_center_img1[0], transformed_center_img1[1], color='red', s=100, marker='x', label='Aligned Center')
# plt.title("Blended Image with Centers After Alignment")
# plt.legend()
# plt.axis('off')
# plt.savefig('aligned_image.png', bbox_inches='tight', pad_inches=0)
# plt.show()

########################### 3rd part ########################################
# import pyvista as pv

# # Function has been defined
# # def read_xyz(file_path):
# #     """Read XYZ data from a file and return a NumPy array."""
# #     points = []
# #     with open(file_path, 'r') as f:
# #         for line in f:
# #             x, y, z = map(float, line.strip().split())
# #             points.append((x, y, z))
# #     return np.array(points)

# def transform_point_cloud_without_x_inversion(points, transformation_matrix_3d):
#     """
#     Applies a 3D transformation matrix to a point cloud without X-axis inversion.
#     :param points: An Nx3 NumPy array of (X, Y, Z) coordinates.
#     :param transformation_matrix_3d: A 3x4 transformation matrix for 3D transformation.
#     :return: Transformed Nx3 NumPy array of (X, Y, Z) coordinates.
#     """
#     # Step 1: Add a column of ones to the points to enable matrix multiplication with the transformation matrix
#     num_points = points.shape[0]
#     homogeneous_points = np.hstack((points, np.ones((num_points, 1))))

#     # Step 2: Apply the transformation matrix to the points
#     transformed_points = homogeneous_points @ transformation_matrix_3d.T

#     # Return only the first three columns (X, Y, Z) from the transformed points
#     return transformed_points[:, :3]

# def visualize_point_clouds_with_grid(pcd1, pcd2, original_scan, transformed_origin):
#     """Visualize two point clouds, a transformed origin point, and a grid in PyVista."""
#     plotter = pv.Plotter()

#     # Add the first point cloud (blueprint)
#     plotter.add_points(pcd1, color='blue', point_size=5.0, render_points_as_spheres=True, label='Blueprint')

#     # Add the second point cloud (transformed scan)
#     plotter.add_points(pcd2, color='red', point_size=5.0, render_points_as_spheres=True, label='Transformed Scan')

#     # Add the original scan point cloud (for reference)
#     # plotter.add_points(original_scan, color='green', point_size=5.0, render_points_as_spheres=True, label='Original Scan')

#     # Add the transformed origin point (in magenta)
#     plotter.add_points(np.array([transformed_origin]), color='magenta', point_size=15.0, render_points_as_spheres=True, label='Transformed Origin')

#     # Display the axes
#     plotter.show_grid()

#     # Add a legend
#     plotter.add_legend()

#     # Show the plot interactively
#     plotter.show()

# def apply_transformation_and_visualize(blueprint_file, scan_file, scale_factor, transformation_matrix_2d):
#     """Apply a 3D transformation to the scan point cloud and visualize both point clouds."""
#     # Step 1: Read both XYZ files
#     blueprint_points = read_xyz(blueprint_file)
#     scan_points = read_xyz(scan_file)

#     # Step 2: Create the 3D transformation matrix based on the 2D matrix and scale factor
#     # Map the transformation from XY to XZ (i.e., switch Y with Z)
#     transformation_matrix_3d = np.array([
#         [transformation_matrix_2d[0][0], 0, transformation_matrix_2d[0][1], transformation_matrix_2d[0][2] / scale_factor],
#         [0, 1, 0, 0],  # No change in the Y (height) axis
#         [transformation_matrix_2d[1][0], 0, transformation_matrix_2d[1][1], transformation_matrix_2d[1][2] / scale_factor]
#     ])

#     # Print the resulting 3D transformation matrix
#     print("Resulting 3D Transformation Matrix:")
#     print(transformation_matrix_3d)

#     # Step 3: Apply the transformation to the scan point cloud without X-axis inversion
#     transformed_scan_points = transform_point_cloud_without_x_inversion(scan_points, transformation_matrix_3d)

#     # Step 4: Calculate the transformed origin point (0, 0, 0)
#     origin = np.array([0, 0, 0, 1])  # Homogeneous coordinates for the origin
#     transformed_origin = transformation_matrix_3d @ origin.T

#     # Print the transformed origin point
#     print(f"Transformed Origin Point: {transformed_origin[:3]}")

#     # Step 5: Visualize the point clouds, grid, and transformed origin point using PyVista
#     visualize_point_clouds_with_grid(blueprint_points, transformed_scan_points, scan_points, transformed_origin[:3])

# # Apply the transformation and visualize
# apply_transformation_and_visualize(xyz_file_path_blueprint, xyz_file_path_scan, scale_factor, xyz_transformation_matrix)

###############################################################################################################################
                   
def transform_point_cloud_without_x_inversion(points, transformation_matrix_3d):
    """
    Applies a 3D transformation matrix to a point cloud without X-axis inversion.
    :param points: An Nx3 NumPy array of (X, Y, Z) coordinates.
    :param transformation_matrix_3d: A 3x4 transformation matrix for 3D transformation.
    :return: Transformed Nx3 NumPy array of (X, Y, Z) coordinates.
    """
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    transformed_points = homogeneous_points @ transformation_matrix_3d.T
    return transformed_points[:, :3]

def visualize_point_clouds_with_grid(pcd1, pcd2, transformed_origin, direction_vector, output_path):
    """Visualize two point clouds and a rotated arrow at the transformed origin point in a bird's-eye (XZ) view using Matplotlib."""
    plt.figure(figsize=(10, 10))
    
    # Plot the first point cloud (blueprint) in blue
    plt.scatter(pcd1[:, 0], pcd1[:, 2], color='blue', s=5, label='Blueprint')
    
    # Plot the transformed scan point cloud in red
    plt.scatter(pcd2[:, 0], pcd2[:, 2], color='red', s=5, label='Transformed Scan')
    
    # Define arrowhead parameters for a minimal line segment
    arrow_length = 0.005 * (pcd1[:, 0].max() - pcd1[:, 0].min())  # Minimal arrow length just to display the head
    head_width = arrow_length * 5  # Large enough width to make the head noticeable
    head_length = arrow_length * 5  # Large enough length to make the head noticeable

    # Format the transformed origin coordinates to two decimal places
    transformed_origin_coords = f"({transformed_origin[0]:.2f}, {transformed_origin[2]:.2f})"

    # Plot only the arrowhead at the transformed origin with minimal extension
    plt.arrow(
        transformed_origin[0],  # Starting X position (transformed origin)
        transformed_origin[2],  # Starting Z position (transformed origin)
        direction_vector[0] * arrow_length,  # Minimal X component for the arrow direction
        direction_vector[2] * arrow_length,  # Minimal Z component for the arrow direction
        head_width=head_width, 
        head_length=head_length, 
        fc='magenta', 
        ec='magenta', 
        label=f'Current Location {transformed_origin_coords}'  # Display coordinates in the label
    )

    # Flip the X-axis so negative is on the right
    plt.gca().invert_xaxis()
    
    # Set equal scaling for X and Z axes and maintain a square aspect ratio
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add labels, grid, and title
    plt.xlabel("X")  # Flipped
    plt.ylabel("Z")
    plt.grid(True)

    # Move legend to the top-right corner, outside the plot area
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

    # Save the plot as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

    # Display the plot
    plt.show()

def apply_transformation_and_visualize(blueprint_file, scan_file, aligned_image_file, scale_factor, transformation_matrix_2d):
    """Apply a 3D transformation to the scan point cloud and visualize both point clouds."""
    # Read both XYZ files
    blueprint_points = read_xyz(blueprint_file)
    scan_points = read_xyz(scan_file)

    blueprint_points = filter_duplicate_with_lowest_y(blueprint_points)
    scan_points = filter_duplicate_with_lowest_y(scan_points)

    # Create the 3D transformation matrix based on the 2D matrix and scale factor
    transformation_matrix_3d = np.array([
        [transformation_matrix_2d[0][0], 0, transformation_matrix_2d[0][1], transformation_matrix_2d[0][2] / scale_factor],
        [0, 1, 0, 0],
        [transformation_matrix_2d[1][0], 0, transformation_matrix_2d[1][1], transformation_matrix_2d[1][2] / scale_factor]
    ])

    # Print the resulting 3D transformation matrix
    print("Resulting 3D Transformation Matrix:")
    print(transformation_matrix_3d)

    # Apply the transformation to the scan point cloud without X-axis inversion
    transformed_scan_points = transform_point_cloud_without_x_inversion(scan_points, transformation_matrix_3d)

    # Calculate the transformed origin point (0, 0, 0)
    origin = np.array([0, 0, 0, 1])
    transformed_origin = transformation_matrix_3d @ origin.T

    # Print the transformed origin point
    print(f"Transformed Origin Point: {transformed_origin[:3]}")

    # Extract the direction vector for the arrow from the transformation matrix
    direction_vector = transformation_matrix_3d[:3, 0]  # X direction component of the rotation matrix

    # Visualize the point clouds in the XZ plane using Matplotlib with the rotated arrow
    visualize_point_clouds_with_grid(blueprint_points, transformed_scan_points, transformed_origin[:3], direction_vector, aligned_image_file)

# Apply the transformation and visualize
apply_transformation_and_visualize(xyz_file_path_blueprint, xyz_file_path_scan, aligned_image_path, scale_factor, xyz_transformation_matrix)

