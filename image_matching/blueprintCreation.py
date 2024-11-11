import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, PngImagePlugin

def read_xyz(file_path):
    """Read XYZ data from a file."""
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            points.append((x, y, z))
    return np.array(points)

def xyz_to_image(points, output_image_path, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), meter_interval=1):
    """
    Converts an XYZ file to an image and optionally calculates pixels per unit if it's a blueprint.
    """
    # Create a copy of the points array to prevent modification of the original points
    points = points.copy()

    # Invert the x-axis for left-handed coordinate system
    points[:, 0] = -points[:, 0]  # Invert the x-axis

    # Sort points by y-values so that higher points are plotted last (on top)
    points = points[np.argsort(points[:, 1])]

    # Get x, z for 2D plotting, and y for the gradient color
    x_vals = points[:, 0]
    z_vals = points[:, 2]
    y_vals = points[:, 1]  # Use y for color gradient

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

    pixels_per_unit = calculate_pixels_per_unit_from_image(output_image_path, range_x, range_z, error_pixels)
    return pixels_per_unit


def error_pixels_from_image(points, output_image_path, boundary, padding_pixels=50, image_size=(500, 500)):
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

    # Take the minimum of the two to avoid distortion
    final_pixels_per_unit = min(pixels_per_unit_x, pixels_per_unit_z)
    print(f"Final pixels per unit (min of X and Z): {final_pixels_per_unit}")

    return final_pixels_per_unit

def save_with_metadata(image_path, boundary, error_pixels, scale_factor):
    # Open the image to add metadata
    img = Image.open(image_path)
    
    # Convert boundary and scale_factor to strings for metadata storage
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("boundary", str(boundary))
    metadata.add_text("error_pixels", str(error_pixels))
    metadata.add_text("scale_factor", str(scale_factor))
    
    # Save the image with the metadata
    img.save(image_path, "PNG", pnginfo=metadata)

xyz_file_path_blueprint = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/secondFloor_even.xyz' # secondFloor_even.xyz' # firstFloorSouth_even.xyz' # room1(2)_even.xyz'
output_image_path_blueprint = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/secondFloor_even_output.png' # firstFloorSouth_even_output.png' # secondFloor_even_output.png' # room1(2)_even_output.png'
output_image_path_error_pixels = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/error_pixels.png'

# Read the blueprint data and get the bounding box
blueprint_points = read_xyz(xyz_file_path_blueprint)

# Calculate the boundary of the blueprint
min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

# Generate an image for the blueprint and calculate error pixels
error_pixels = error_pixels_from_image(blueprint_points, output_image_path_error_pixels, boundary, padding_pixels=50, image_size=(500, 500))

# Generate an image for the blueprint and calculate pixels per unit
scale_factor = xyz_to_image(blueprint_points, output_image_path_blueprint, boundary, error_pixels, padding_pixels=50, image_size=(500, 500))

# Save the output_image_path_blueprint with metadata
save_with_metadata(output_image_path_blueprint, boundary, error_pixels, scale_factor)