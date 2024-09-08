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
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=4000000, confidence=1))
    return result



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.translate((0, 0, 0))
    sphere.paint_uniform_color([0, 0, 1])
    sphere_cloud = o3d.geometry.PointCloud()
    sphere_cloud.points = sphere.vertices
    sphere_cloud.colors = sphere.vertex_colors

    o3d.visualization.draw_geometries([source_temp, target_temp, sphere_cloud])

def main():
    voxel_size_coarse = 1
    voxel_size_medium = 0.5
    voxel_size_fine = 0.25

    # source = o3d.io.read_point_cloud("firstFloorSouth_usd2xyzV6.xyz")
    source = o3d.io.read_point_cloud("sameSpotNorth_usd2xyzV2.1.xyz")
    target = o3d.io.read_point_cloud("firstFloorSouth_usd2xyzV2.1.xyz")


    # Coarse registration
    source_down_coarse, source_fpfh_coarse = preprocess_point_cloud(source, voxel_size_coarse)
    target_down_coarse, target_fpfh_coarse = preprocess_point_cloud(target, voxel_size_coarse)
    global_result_coarse = execute_global_registration(source_down_coarse, target_down_coarse, source_fpfh_coarse, target_fpfh_coarse, voxel_size_coarse)

    # Medium registration
    source_down_medium, source_fpfh_medium = preprocess_point_cloud(source, voxel_size_medium)
    target_down_medium, target_fpfh_medium = preprocess_point_cloud(target, voxel_size_medium)
    global_result_medium = execute_global_registration(source_down_medium, target_down_medium, source_fpfh_medium, target_fpfh_medium, voxel_size_medium, global_result_coarse.transformation)

    # Fine ICP refinement
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size_fine, global_result_medium.transformation,  # Use the medium result's transformation
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("Final ICP result:")
    print(icp_result)  # Use `icp_result`
    draw_registration_result(source, target, icp_result.transformation)  # Use `icp_result`


if __name__ == "__main__":
    main()
