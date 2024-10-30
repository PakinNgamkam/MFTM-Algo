# MFTM-Algo
Alignment Algorithm for MFTM

## Current best ICP
icp_with _y_constraint.py  

// increase max_iteration for more accuracy  
o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=2000000, confidence=1)  

## New Image Alignment using feature_matching_with_geometric_constraints
combinedImageAlignment.py

## Current Best xyz
usd2xyzEvenEdge.py (change file path before running this)

## See xyz
python showxyz.py XYZfile.xyz
