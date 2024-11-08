import os
from pxr import Usd, UsdGeom, Gf
import math
from collections import defaultdict
import numpy as np

def apply_transform(prim, point):
    """Apply the cumulative transformation from the prim to the point."""
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return world_transform.Transform(point)

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)

def distribute_points(start, end, step_size):
    """Distribute points along the edge at intervals of step_size."""
    dist = distance(start, end)
    
    # If the distance is smaller than step_size, return only the start and end
    if dist < step_size:
        return [start, end]
    
    num_steps = int(dist // step_size)
    points = [Gf.Lerp(float(i) / num_steps, start, end) for i in range(num_steps + 1)]
    return points

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
    lowest_y_global = np.min(np.array(points)[:, 1])

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

    return filtered_points

def usdz_to_xyz(usdz_file_path, output_xyz_file_path, step_size=0.1, error_range=0.05):
    # Initialize USD stage
    stage = Usd.Stage.Open(usdz_file_path)
    
    # Collect points in a list to filter them before writing to file
    collected_points = []

    # Iterate over all prims looking for Mesh prims
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            # Get the Mesh geometry
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            points = points_attr.Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()

            # Track visited vertices to avoid duplications
            visited_vertices = set()
            index_offset = 0

            for count in counts:
                face_points = [apply_transform(prim, points[indices[i + index_offset]]) for i in range(count)]

                # Sample more points along the edges using step_size
                for i in range(len(face_points)):
                    start = face_points[i]
                    end = face_points[(i + 1) % len(face_points)]

                    # Distribute points along the edge at intervals of step_size
                    for sample_point in distribute_points(start, end, step_size):
                        key = tuple(sample_point)
                        if key not in visited_vertices:
                            collected_points.append(sample_point)
                            visited_vertices.add(key)

                index_offset += count

    # Filter duplicates based on the lowest y-value criterion
    filtered_points = filter_duplicate_with_lowest_y(collected_points, error_range=error_range)

    # Write the filtered points to the output XYZ file
    with open(output_xyz_file_path, 'w') as xyz_file:
        for point in filtered_points:
            xyz_file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Conversion completed. Output saved to {output_xyz_file_path}")

# Example usage
usdz_file_path = 'C:/Users/pakin/OneDrive/Desktop/test/room1_testcase5.usdz'
output_xyz_file_path = 'C:/Users/pakin/OneDrive/Desktop/test/image_matching/room1_testcase5_even.xyz'
usdz_to_xyz(usdz_file_path, output_xyz_file_path, step_size=0.25, error_range=0.05)
