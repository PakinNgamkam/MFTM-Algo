import os
from pxr import Usd, UsdGeom, Gf

def apply_transform(prim, point):
    """Apply the cumulative transformation from the prim to the point."""
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return world_transform.Transform(point)

def interpolate_points(start, end, num_points):
    """Linearly interpolate between two points."""
    return [Gf.Lerp(float(i) / (num_points - 1), start, end) for i in range(num_points)]

def usdz_to_xyz(usdz_file_path, output_xyz_file_path, edge_samples=10):
    # Initialize USD stage
    stage = Usd.Stage.Open(usdz_file_path)
    
    # Open the output XYZ file
    with open(output_xyz_file_path, 'w') as xyz_file:
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

                    # Sample more points along the edges
                    for i in range(len(face_points)):
                        start = face_points[i]
                        end = face_points[(i + 1) % len(face_points)]

                        # Increase the number of samples specifically along the edges
                        for sample_point in interpolate_points(start, end, edge_samples):
                            key = tuple(sample_point)
                            if key not in visited_vertices:
                                xyz_file.write(f"{sample_point[0]} {sample_point[1]} {sample_point[2]}\n")
                                visited_vertices.add(key)

                    index_offset += count

    print(f"Conversion completed. Output saved to {output_xyz_file_path}")

# Example usage
usdz_file_path = 'C:/Users/pakin/OneDrive/Desktop/test/sameSpotNorth.usdz'
output_xyz_file_path = 'C:/Users/pakin/OneDrive/Desktop/test/sameSpotNorth_usd2xyzV2.1.xyz'
usdz_to_xyz(usdz_file_path, output_xyz_file_path, edge_samples=10)  # Increase edge_samples for denser points along edges
