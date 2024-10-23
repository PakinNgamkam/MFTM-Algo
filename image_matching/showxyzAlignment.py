import numpy as np
import pyvista as pv

def read_xyz(file_path):
    """Read XYZ data from a file and return a NumPy array."""
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            points.append((x, y, z))
    return np.array(points)

def transform_point_cloud_without_x_inversion(points, transformation_matrix_3d):
    """
    Applies a 3D transformation matrix to a point cloud without X-axis inversion.
    :param points: An Nx3 NumPy array of (X, Y, Z) coordinates.
    :param transformation_matrix_3d: A 3x4 transformation matrix for 3D transformation.
    :return: Transformed Nx3 NumPy array of (X, Y, Z) coordinates.
    """
    # Step 1: Add a column of ones to the points to enable matrix multiplication with the transformation matrix
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))

    # Step 2: Apply the transformation matrix to the points
    transformed_points = homogeneous_points @ transformation_matrix_3d.T

    # Return only the first three columns (X, Y, Z) from the transformed points
    return transformed_points[:, :3]

def visualize_point_clouds_with_grid(pcd1, pcd2, original_scan):
    """Visualize two point clouds and a grid in PyVista."""
    plotter = pv.Plotter()

    # Add the first point cloud (blueprint)
    plotter.add_points(pcd1, color='blue', point_size=5.0, render_points_as_spheres=True, label='Blueprint')

    # Add the second point cloud (transformed scan)
    plotter.add_points(pcd2, color='red', point_size=5.0, render_points_as_spheres=True, label='Transformed Scan')

    # Add the original scan point cloud (for reference)
    plotter.add_points(original_scan, color='green', point_size=5.0, render_points_as_spheres=True, label='Original Scan')

    # Display the axes
    plotter.show_grid()

    # Add a legend
    plotter.add_legend()

    # Show the plot interactively
    plotter.show()

def apply_transformation_and_visualize(blueprint_file, scan_file, scale_factor, transformation_matrix_2d):
    """Apply a 3D transformation to the scan point cloud and visualize both point clouds."""
    # Step 1: Read both XYZ files
    blueprint_points = read_xyz(blueprint_file)
    scan_points = read_xyz(scan_file)

    # Step 2: Create the 3D transformation matrix based on the 2D matrix and scale factor
    # Map the transformation from XY to XZ (i.e., switch Y with Z)
    transformation_matrix_3d = np.array([
        [transformation_matrix_2d[0][0], 0, transformation_matrix_2d[0][1], transformation_matrix_2d[0][2] / scale_factor],
        [0, 1, 0, 0],  # No change in the Y (height) axis
        [transformation_matrix_2d[1][0], 0, transformation_matrix_2d[1][1], transformation_matrix_2d[1][2] / scale_factor]
    ])

    # Print the resulting 3D transformation matrix
    print("Resulting 3D Transformation Matrix:")
    print(transformation_matrix_3d)

    # Step 3: Apply the transformation to the scan point cloud without X-axis inversion
    transformed_scan_points = transform_point_cloud_without_x_inversion(scan_points, transformation_matrix_3d)

    # Step 4: Visualize the point clouds and grid using PyVista
    visualize_point_clouds_with_grid(blueprint_points, transformed_scan_points, scan_points)

# Example usage
blueprint_file = "firstFloorSouth_even.xyz"
scan_file = "KitchenNorth_even.xyz"
scale_factor =  19.90257585559306 # secondFloor = 24.420789383589398, firstFloorSouth = 19.90257585559306
transformation_matrix_2d = np.array([
    # [1.005140606795668, 0.09542898822209142, 4.345628751981366],
    # [-0.09542898822209142, 1.005140606795668, 7.825432616593076]
    # [1.00514061, 0.09542899, 0],
    # [-0.09542899, 1.00514061, 0]
[0.9972916375419565, 0.05723934816917654, 134.52590221273334],
[-0.05723934816917654, 0.9972916375419565, -15.012756303836511]
])

apply_transformation_and_visualize(blueprint_file, scan_file, scale_factor, transformation_matrix_2d)

#--------------------------------------------------------------------------------------------
# import numpy as np
# import pyvista as pv

# def read_xyz(file_path):
#     """Read XYZ data from a file and return a NumPy array."""
#     points = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             x, y, z = map(float, line.strip().split())
#             points.append((x, y, z))
#     return np.array(points)

# def transform_point_cloud_without_x_inversion(points, transformation_matrix_3d):
#     """Applies a 3D transformation matrix to a point cloud without X-axis inversion."""
#     num_points = points.shape[0]
#     homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
#     transformed_points = homogeneous_points @ transformation_matrix_3d.T
#     return transformed_points[:, :3]

# def visualize_point_clouds_with_grid(pcd1, pcd2, original_scan, scale_factor):
#     """Visualize two point clouds and a grid in PyVista."""
#     plotter = pv.Plotter()
#     plotter.add_points(pcd1, color='blue', point_size=5.0, render_points_as_spheres=True, label='Blueprint')
#     plotter.add_points(pcd2, color='red', point_size=5.0, render_points_as_spheres=True, label='Transformed Scan')
#     plotter.add_points(original_scan, color='green', point_size=5.0, render_points_as_spheres=True, label='Original Scan')
#     plotter.show_grid()
#     plotter.add_legend()
#     plotter.add_text(f"Scale Factor: {scale_factor}", font_size=14)
#     plotter.show()

# def apply_transformation_and_visualize(blueprint_file, scan_file, scale_factors, transformation_matrix_2d):
#     """Apply a 3D transformation to the scan point cloud and visualize both point clouds for each scale factor."""
#     blueprint_points = read_xyz(blueprint_file)
#     scan_points = read_xyz(scan_file)

#     for scale_factor in scale_factors:
#         if scale_factor == 0:
#             print(f"Skipping scale factor 0 to avoid division by zero.")
#             continue

#         # Create the 3D transformation matrix based on the 2D matrix and scale factor
#         transformation_matrix_3d = np.array([
#             [transformation_matrix_2d[0][0], 0, transformation_matrix_2d[0][1], transformation_matrix_2d[0][2] / scale_factor],
#             [0, 1, 0, 0],
#             [transformation_matrix_2d[1][0], 0, transformation_matrix_2d[1][1], transformation_matrix_2d[1][2] / scale_factor]
#         ])

#         print(f"Resulting 3D Transformation Matrix for Scale Factor {scale_factor}:")
#         print(transformation_matrix_3d)

#         # Apply the transformation to the scan point cloud without X-axis inversion
#         transformed_scan_points = transform_point_cloud_without_x_inversion(scan_points, transformation_matrix_3d)

#         # Visualize the point clouds with the current scale factor
#         visualize_point_clouds_with_grid(blueprint_points, transformed_scan_points, scan_points, scale_factor)

# # Define the image paths and transformation matrix
# blueprint_file = "secondFloor_even.xyz"
# scan_file = "RoomSecondFloor_even.xyz"
# transformation_matrix_2d = np.array([
#     [1.00514061, 0.09542899, -23.08755767],
#     [-0.09542899, 1.00514061, 9.63620925]
# ])

# # Define the range of scale factors to visualize (0 to 24)
# scale_factors = range(500, 502)

# # Apply the transformations and visualize each scale factor
# apply_transformation_and_visualize(blueprint_file, scan_file, scale_factors, transformation_matrix_2d)

#--------------------------------------------------------------------------------------------

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def align_images(img1_path, img2_path, transformation_matrix):
#     """
#     Align img1 with img2 using the given transformation matrix.

#     Parameters:
#     - img1_path: Path to the first image (to be transformed).
#     - img2_path: Path to the second image (reference image).
#     - transformation_matrix: 2x3 transformation matrix for affine transformation.

#     Returns:
#     - aligned_img1: The transformed version of img1 aligned with img2.
#     - img2: The reference image.
#     """
#     # Load the images
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)

#     # Ensure the images are loaded properly
#     if img1 is None or img2 is None:
#         raise ValueError("One or both of the image paths are invalid!")

#     # Convert the transformation matrix into a NumPy array if it's not already
#     transformation_matrix = np.array(transformation_matrix)

#     # Get the dimensions of img2 for warping (we want to align img1 to img2's size)
#     rows, cols = img2.shape[:2]

#     # Apply the affine transformation to align img1 to img2
#     aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (cols, rows))

#     return aligned_img1, img2

# # Define the image paths
# img1_path = 'RoomSecondFloor_even_output.png'
# img2_path = 'secondFloor_even_output.png'

# # Define the transformation matrix (to be applied to img1)
# transformation_matrix = [
#     [1.00514061, 0.09542899, -23.08755767],
#     [-0.09542899, 1.00514061, 9.63620925]
# ]

# # Align img1 using the transformation matrix to align with img2
# aligned_img1, img2 = align_images(img1_path, img2_path, transformation_matrix)

# # Plot the original and aligned images for comparison
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))
# plt.title("Transformed Image (img1 aligned)")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# plt.title("Reference Image (img2)")
# plt.axis('off')

# plt.show()

# # Blend the aligned image with the reference image for visualization
# alpha = 0.5  # Adjust the opacity level as needed
# blended_image = cv2.addWeighted(aligned_img1, alpha, img2, 1 - alpha, 0)

# # Display the blended result
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
# plt.title("Blended Image After Alignment")
# plt.axis('off')
# plt.show()

#-----------------------------Prove that png aligned using the matrix----------------------------------
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def align_images_and_calculate_vector(img1_path, img2_path, transformation_matrix):
#     """
#     Align img1 with img2 using the given transformation matrix and calculate the vector difference between the centers.

#     Parameters:
#     - img1_path: Path to the first image (to be transformed).
#     - img2_path: Path to the second image (reference image).
#     - transformation_matrix: 2x3 transformation matrix for affine transformation.

#     Returns:
#     - aligned_img1: The transformed version of img1 aligned with img2.
#     - img2: The reference image.
#     - center_vector: The vector difference (dx, dy) between the center of img1 and img2 after alignment.
#     """
#     # Load the images
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)

#     # Ensure the images are loaded properly
#     if img1 is None or img2 is None:
#         raise ValueError("One or both of the image paths are invalid!")

#     # Convert the transformation matrix into a NumPy array if it's not already
#     transformation_matrix = np.array(transformation_matrix)

#     # Get the dimensions of img2 for warping (we want to align img1 to img2's size)
#     rows, cols = img2.shape[:2]

#     # Calculate the center of img1 and img2
#     center_img1 = np.array([img1.shape[1] // 2, img1.shape[0] // 2, 1])  # (x, y, 1)
#     center_img2 = np.array([img2.shape[1] // 2, img2.shape[0] // 2])      # (x, y)

#     # Transform the center of img1 using the transformation matrix
#     transformed_center_img1 = transformation_matrix @ center_img1

#     # Calculate the vector difference between transformed center of img1 and center of img2
#     dx = transformed_center_img1[0] - center_img2[0]
#     dy = transformed_center_img1[1] - center_img2[1]
#     center_vector = (dx, dy)

#     # Apply the affine transformation to align img1 to img2
#     aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (cols, rows))

#     # Mark the centers on the aligned image and the reference image
#     aligned_img1_with_center = aligned_img1.copy()
#     img2_with_center = img2.copy()

#     cv2.circle(aligned_img1_with_center, (int(transformed_center_img1[0]), int(transformed_center_img1[1])), 5, (0, 0, 255), -1)
#     cv2.circle(img2_with_center, (center_img2[0], center_img2[1]), 5, (0, 255, 0), -1)

#     return aligned_img1_with_center, img2_with_center, center_vector

# # Define the image paths
# img1_path = 'RoomSecondFloor_even_output.png'
# img2_path = 'secondFloor_even_output.png'

# # Define the transformation matrix (to be applied to img1)
# transformation_matrix = [
#     [1.00514061, 0.09542899, -23.08755767],
#     [-0.09542899, 1.00514061, 9.63620925]
# ]

# # Align img1 using the transformation matrix to align with img2 and calculate the vector difference
# aligned_img1_with_center, img2_with_center, center_vector = align_images_and_calculate_vector(
#     img1_path, img2_path, transformation_matrix
# )

# # Plot the original and aligned images for comparison
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(aligned_img1_with_center, cv2.COLOR_BGR2RGB))
# plt.title("Transformed Image (img1 aligned) with Center")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(img2_with_center, cv2.COLOR_BGR2RGB))
# plt.title("Reference Image (img2) with Center")
# plt.axis('off')

# plt.show()

# # Print the vector difference between centers
# print(f"Vector difference between centers (dx, dy): {center_vector}")

# # Blend the aligned image with the reference image for visualization
# alpha = 0.5  # Adjust the opacity level as needed
# blended_image = cv2.addWeighted(aligned_img1_with_center, alpha, img2_with_center, 1 - alpha, 0)

# # Display the blended result
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
# plt.title("Blended Image After Alignment")
# plt.axis('off')
# plt.show()
