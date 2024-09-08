import open3d as o3d
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    print(f":: Downsample with a voxel size {voxel_size:.3f}.")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(f":: Estimate normal with search radius {radius_normal:.3f}.")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 2
    print(f":: Compute FPFH feature with search radius {radius_feature:.3f}.")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, initial_transformation=None):
    distance_threshold = voxel_size * 1.5
    # If an initial transformation is provided, use it; otherwise, default to identity matrix
    if initial_transformation is None:
        initial_transformation = np.identity(4)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=2000000, confidence=1))
    return result
def get_floor_y(pcd):
    """Return the minimum y-coordinate, assuming the floor is flat and on the x-z plane."""
    points = np.asarray(pcd.points)
    return np.min(points[:, 1])

def apply_y_axis_constraints(transformation, y_offset):
    """Constrain translation to fix the floor alignment and allow only y-axis rotation."""
    constrained_transformation = np.copy(transformation)

    # Fix the translation on the y-axis by applying the y-offset
    constrained_transformation[1, 3] = y_offset

    # Extract the y-axis rotation (rotation around the y-axis only)
    R = constrained_transformation[:3, :3]
    
    # Compute the angle of rotation around the y-axis using the rotation matrix
    yaw = np.arctan2(R[0, 2], R[2, 2])

    # Construct a new rotation matrix that only rotates around the y-axis
    R_y_only = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                         [0, 1, 0],
                         [-np.sin(yaw), 0, np.cos(yaw)]])

    # Replace the rotation matrix with the y-axis only rotation matrix
    constrained_transformation[:3, :3] = R_y_only

    return constrained_transformation

def execute_icp_with_y_constraint(source, target, voxel_size, y_offset, initial_transformation=None):
    if initial_transformation is None:
        initial_transformation = np.identity(4)
    
    # Perform standard ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # Apply the constraint to the resulting transformation, fixing the floor alignment
    constrained_transformation = apply_y_axis_constraints(result_icp.transformation, y_offset)

    # Update the result transformation to constrained one
    result_icp.transformation = constrained_transformation

    return result_icp

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # Red for source
    target_temp.paint_uniform_color([0, 1, 0])  # Green for target

    # Apply the transformation to the source point cloud
    source_temp.transform(transformation)

    # Create a sphere at the transformed origin (0, 0, 0)
    # Define the original origin point
    origin_point = np.array([0, 0, 0, 1])  # Homogeneous coordinates [x, y, z, 1]

    # Apply the transformation to the origin point
    transformed_origin = np.dot(transformation, origin_point)

    # Create a larger sphere at the transformed origin point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)  # Increase the radius to 0.15
    sphere.translate(transformed_origin[:3])  # Sphere at the transformed origin (x, y, z)
    sphere.paint_uniform_color([0, 0, 1])  # Blue for the origin marker

    # Draw both the source, target, and the origin marker
    o3d.visualization.draw_geometries([source_temp, target_temp, sphere])


# Main function where the registration pipeline runs
def main():
    voxel_size_coarse = 1
    voxel_size_medium = 0.5
    voxel_size_fine = 0.25

    source = o3d.io.read_point_cloud("sameSpotNorth_usd2xyzV2.1.xyz")
    # source = o3d.io.read_point_cloud("firstFloorSouthNoSofa.xyz")
    target = o3d.io.read_point_cloud("firstFloorSouth_usd2xyzV2.1.xyz")
    # target = o3d.io.read_point_cloud("xz_firstFloorSouthNoFurniture.xyz")

    # Coarse registration
    source_down_coarse, source_fpfh_coarse = preprocess_point_cloud(source, voxel_size_coarse)
    target_down_coarse, target_fpfh_coarse = preprocess_point_cloud(target, voxel_size_coarse)
    global_result_coarse = execute_global_registration(source_down_coarse, target_down_coarse, source_fpfh_coarse, target_fpfh_coarse, voxel_size_coarse)

    # Medium registration
    source_down_medium, source_fpfh_medium = preprocess_point_cloud(source, voxel_size_medium)
    target_down_medium, target_fpfh_medium = preprocess_point_cloud(target, voxel_size_medium)
    global_result_medium = execute_global_registration(source_down_medium, target_down_medium, source_fpfh_medium, target_fpfh_medium, voxel_size_medium, global_result_coarse.transformation)

    # Ensure the floors are aligned by fixing y-translation
    source_floor_y = get_floor_y(source)
    target_floor_y = get_floor_y(target)
    y_offset = target_floor_y - source_floor_y

    # Fine ICP refinement with y-axis constraint, fixing the floor alignment
    icp_result = execute_icp_with_y_constraint(source, target, voxel_size_fine, y_offset, global_result_medium.transformation)

    print("Final ICP result:")
    print(icp_result)
    
    # Draw the final result with the constrained transformation
    draw_registration_result(source, target, icp_result.transformation)
    # draw_registration_result(source, target, global_result_medium.transformation)

if __name__ == "__main__":
    main()