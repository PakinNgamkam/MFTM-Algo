# import cv2
# import numpy as np

# # Load images
# img1 = cv2.imread('SameSpotWest_even_output.png')
# img2 = cv2.imread('firstFloorSouth_even_output.png')

# # Convert images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Threshold images to get non-white regions (you can adjust the threshold as needed)
# _, mask1 = cv2.threshold(gray1, 250, 255, cv2.THRESH_BINARY_INV)
# _, mask2 = cv2.threshold(gray2, 250, 255, cv2.THRESH_BINARY_INV)

# # Feature detection using ORB
# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(mask1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(mask2, None)

# # Feature matching using BFMatcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)

# # Sort matches by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

# # Extract matched keypoints
# points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# # Estimate affine transformation matrix using RANSAC
# matrix, mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

# # Apply transformation to img1
# aligned_img1 = cv2.warpAffine(img1, matrix, (img2.shape[1], img2.shape[0]))

# # Blend the aligned image and img2 (blueprint)
# alpha = 0.5  # Adjust opacity
# blended = cv2.addWeighted(aligned_img1, alpha, img2, 1 - alpha, 0)

# # Display the blended result
# cv2.imshow('Blended Image', blended)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#######ORB Feature Matching########
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load images
# img1_path = 'KitchenNorth_even_output.png'
# img2_path = 'firstFloorSouth_even_output.png'
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# # Convert images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Threshold images to get non-white regions (you can adjust the threshold as needed)
# _, mask1 = cv2.threshold(gray1, 250, 255, cv2.THRESH_BINARY_INV)
# _, mask2 = cv2.threshold(gray2, 250, 255, cv2.THRESH_BINARY_INV)

# # Feature detection using ORB
# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(mask1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(mask2, None)

# # Draw keypoints
# img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0))
# img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0))

# # Show the keypoints detected
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(cv2.cvtColor(img1_with_keypoints, cv2.COLOR_BGR2RGB))
# ax[0].set_title("Keypoints in img1 (scan)")
# ax[0].axis('off')
# ax[1].imshow(cv2.cvtColor(img2_with_keypoints, cv2.COLOR_BGR2RGB))
# ax[1].set_title("Keypoints in img2 (blueprint)")
# ax[1].axis('off')
# plt.show()

# # Feature matching using BFMatcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)

# # Sort matches by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

# # Draw first 50 matches
# img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)

# # Show the matched keypoints
# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
# plt.title("Top 50 Keypoint Matches")
# plt.axis('off')
# plt.show()

# # Extract matched keypoints (points before transformation)
# points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
# points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# # Estimate affine transformation matrix using RANSAC
# matrix, mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

# # Apply the affine transformation matrix to the points1
# transformed_points1 = cv2.transform(np.array([points1]), matrix).reshape(-1, 2)

# # Plotting the original matched points and the transformed points
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # Plot original matched points1 (from img1) and points2 (from img2)
# ax[0].scatter(points1[:, 0], points1[:, 1], color='red', label='Points1 (img1)', alpha=0.7)
# ax[0].scatter(points2[:, 0], points2[:, 1], color='blue', label='Points2 (img2)', alpha=0.7)
# ax[0].set_title("Original Matched Keypoints")
# ax[0].legend()
# ax[0].axis('equal')

# # Plot transformed points1 after applying the transformation matrix and points2
# ax[1].scatter(transformed_points1[:, 0], transformed_points1[:, 1], color='green', label='Transformed Points1 (img1)', alpha=0.7)
# ax[1].scatter(points2[:, 0], points2[:, 1], color='blue', label='Points2 (img2)', alpha=0.7)
# ax[1].set_title("Transformed Keypoints After Applying Matrix")
# ax[1].legend()
# ax[1].axis('equal')

# plt.show()

# # Apply transformation to img1
# aligned_img1 = cv2.warpAffine(img1, matrix, (img2.shape[1], img2.shape[0]))

# # Blend the aligned image and img2 (blueprint)
# alpha = 0.5  # Adjust opacity
# blended = cv2.addWeighted(aligned_img1, alpha, img2, 1 - alpha, 0)

# # Show the blended result
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
# plt.title("Blended Image After Alignment")
# plt.axis('off')
# plt.show()

# # Print the transformation matrix
# print("Transformation Matrix:")
# print(matrix)

######HISTOGRAM MATCHING######
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load images
# img1_path = 'KitchenNorth_even_output.png'
# img2_path = 'firstFloorSouth_even_output.png'
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# # Convert images to HSV for better color comparison
# img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
# img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# # Calculate histograms for each channel in HSV
# hist_img1_hue = cv2.calcHist([img1_hsv], [0], None, [180], [0, 180])
# hist_img2_hue = cv2.calcHist([img2_hsv], [0], None, [180], [0, 180])

# # Normalize the histograms
# hist_img1_hue = cv2.normalize(hist_img1_hue, hist_img1_hue).flatten()
# hist_img2_hue = cv2.normalize(hist_img2_hue, hist_img2_hue).flatten()

# # Plot the histograms to visualize the color distribution
# plt.figure(figsize=(12, 6))
# plt.plot(hist_img1_hue, color='r', label='Image 1 Hue Histogram')
# plt.plot(hist_img2_hue, color='b', label='Image 2 Hue Histogram')
# plt.title("Hue Channel Histograms")
# plt.legend()
# plt.show()

# # Compare histograms using correlation
# score = cv2.compareHist(hist_img1_hue, hist_img2_hue, cv2.HISTCMP_CORREL)
# print(f"Histogram Correlation Score (Hue): {score}")

# # Compute affine transformation matrix for alignment
# # Since histograms alone won't give us the transformation matrix, you may align based on structural similarity
# # For this, we assume alignment based on image registration techniques like ECC (Enhanced Correlation Coefficient)

# # Convert to grayscale for ECC
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Find ECC-based affine transformation matrix
# warp_mode = cv2.MOTION_AFFINE
# warp_matrix = np.eye(2, 3, dtype=np.float32)

# # Specify the number of iterations and the threshold
# number_of_iterations = 5000
# termination_eps = 1e-10
# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

# # Run the ECC algorithm
# (cc, warp_matrix) = cv2.findTransformECC(gray1, gray2, warp_matrix, warp_mode, criteria)

# # Apply affine transformation to img1
# aligned_img1 = cv2.warpAffine(img1, warp_matrix, (img2.shape[1], img2.shape[0]))

# # Blend the aligned image and img2 (blueprint)
# alpha = 0.5  # Adjust opacity
# blended = cv2.addWeighted(aligned_img1, alpha, img2, 1 - alpha, 0)

# # Show the blended result
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
# plt.title("Blended Image After Alignment")
# plt.axis('off')
# plt.show()

# # Print the transformation matrix
# print("Transformation Matrix (from ECC-based Alignment):")
# print(warp_matrix)


######################### REAL SHIT #####################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load images
# img1_path = 'RoomSecondFloor_even_output.png'
# img2_path = 'secondFloor_even_output.png'
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)

# # Split into color channels (R, G, B)
# b1, g1, r1 = cv2.split(img1)
# b2, g2, r2 = cv2.split(img2)

# # Feature detection using ORB on each color channel
# orb = cv2.ORB_create()

# # Detect keypoints in each channel for img1
# kp_b1, des_b1 = orb.detectAndCompute(b1, None)
# kp_g1, des_g1 = orb.detectAndCompute(g1, None)
# kp_r1, des_r1 = orb.detectAndCompute(r1, None)

# # Detect keypoints in each channel for img2
# kp_b2, des_b2 = orb.detectAndCompute(b2, None)
# kp_g2, des_g2 = orb.detectAndCompute(g2, None)
# kp_r2, des_r2 = orb.detectAndCompute(r2, None)

# # Concatenate descriptors from all channels
# descriptors1 = np.vstack([des_b1, des_g1, des_r1])
# descriptors2 = np.vstack([des_b2, des_g2, des_r2])

# # Concatenate keypoints from all channels
# keypoints1 = kp_b1 + kp_g1 + kp_r1
# keypoints2 = kp_b2 + kp_g2 + kp_r2

# # Feature matching using BFMatcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)

# # Sort matches by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

# # Draw first 50 matches
# img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)

# # Show the matched keypoints
# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
# plt.title("Top 50 Keypoint Matches (Color-Based)")
# plt.axis('off')
# plt.show()

# # Extract matched keypoints (points before transformation)
# points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
# points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# # Estimate affine transformation matrix using RANSAC
# matrix, mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

# # Apply the affine transformation matrix to the points1
# transformed_points1 = cv2.transform(np.array([points1]), matrix).reshape(-1, 2)

# # Plotting the original matched points and the transformed points
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # Plot original matched points1 (from img1) and points2 (from img2)
# ax[0].scatter(points1[:, 0], points1[:, 1], color='red', label='Points1 (img1)', alpha=0.7)
# ax[0].scatter(points2[:, 0], points2[:, 1], color='blue', label='Points2 (img2)', alpha=0.7)
# ax[0].set_title("Original Matched Keypoints")
# ax[0].legend()
# ax[0].axis('equal')

# # Plot transformed points1 after applying the transformation matrix and points2
# ax[1].scatter(transformed_points1[:, 0], transformed_points1[:, 1], color='green', label='Transformed Points1 (img1)', alpha=0.7)
# ax[1].scatter(points2[:, 0], points2[:, 1], color='blue', label='Points2 (img2)', alpha=0.7)
# ax[1].set_title("Transformed Keypoints After Applying Matrix")
# ax[1].legend()
# ax[1].axis('equal')

# plt.show()

# # Apply transformation to img1
# aligned_img1 = cv2.warpAffine(img1, matrix, (img2.shape[1], img2.shape[0]))

# # Blend the aligned image and img2 (blueprint)
# alpha = 0.5  # Adjust opacity
# blended = cv2.addWeighted(aligned_img1, alpha, img2, 1 - alpha, 0)

# # Show the blended result
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
# plt.title("Blended Image After Alignment")
# plt.axis('off')
# plt.show()

# # Print the transformation matrix
# print("Transformation Matrix:")
# print(matrix)
#########################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images_and_calculate_vector(img1_path, img2_path, transformation_matrix):
    """
    Align img1 with img2 using the given transformation matrix and calculate the vector difference between the centers.

    Parameters:
    - img1_path: Path to the first image (to be transformed).
    - img2_path: Path to the second image (reference image).
    - transformation_matrix: 2x3 transformation matrix for affine transformation.

    Returns:
    - aligned_img1: The transformed version of img1 aligned with img2.
    - img2: The reference image.
    - center_vector: The vector difference (dx, dy) between the center of img1 and img2 after alignment.
    """
    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

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

    # Mark the centers on the aligned image and the reference image
    aligned_img1_with_center = aligned_img1.copy()
    img2_with_center = img2.copy()

    cv2.circle(aligned_img1_with_center, (int(transformed_center_img1[0]), int(transformed_center_img1[1])), 5, (0, 0, 255), -1)
    cv2.circle(img2_with_center, (center_img2[0], center_img2[1]), 5, (0, 255, 0), -1)

    return aligned_img1_with_center, img2_with_center, center_vector

# Define the image paths
img1_path = 'KitchenNorth_even_output.png'      # Scanned image
img2_path = 'firstFloorSouth_even_output.png'   # Blueprint image

# Calculate the transformation matrix using the previous algorithm
# Load images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Split into color channels (R, G, B)
b1, g1, r1 = cv2.split(img1)
b2, g2, r2 = cv2.split(img2)

# Feature detection using ORB on each color channel
orb = cv2.ORB_create()

# Detect keypoints in each channel for img1
kp_b1, des_b1 = orb.detectAndCompute(b1, None)
kp_g1, des_g1 = orb.detectAndCompute(g1, None)
kp_r1, des_r1 = orb.detectAndCompute(r1, None)

# Detect keypoints in each channel for img2
kp_b2, des_b2 = orb.detectAndCompute(b2, None)
kp_g2, des_g2 = orb.detectAndCompute(g2, None)
kp_r2, des_r2 = orb.detectAndCompute(r2, None)

# Concatenate descriptors from all channels
descriptors1 = np.vstack([des_b1, des_g1, des_r1])
descriptors2 = np.vstack([des_b2, des_g2, des_r2])

# Concatenate keypoints from all channels
keypoints1 = kp_b1 + kp_g1 + kp_r1
keypoints2 = kp_b2 + kp_g2 + kp_r2

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints (points before transformation)
points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# Estimate affine transformation matrix using RANSAC
transformation_matrix, mask = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

# Align img1 using the transformation matrix to align with img2 and calculate the vector difference
aligned_img1_with_center, img2_with_center, center_vector = align_images_and_calculate_vector(
    img1_path, img2_path, transformation_matrix
)

# Plot the original and aligned images for comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(aligned_img1_with_center, cv2.COLOR_BGR2RGB))
plt.title("Transformed Image (img1 aligned) with Center")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_with_center, cv2.COLOR_BGR2RGB))
plt.title("Reference Image (img2) with Center")
plt.axis('off')

plt.show()

# Print the vector difference between centers
print(f"Vector difference between centers (dx, dy): {center_vector}")

# Create the transformation matrix for XYZ file with adjusted tx and ty
new_transformation_matrix = transformation_matrix.copy()
new_transformation_matrix[0, 2] = -center_vector[0]
new_transformation_matrix[1, 2] = -center_vector[1]

print("Transformation Matrix for XYZ file:")
print(f"[{new_transformation_matrix[0, 0]}, {new_transformation_matrix[0, 1]}, {new_transformation_matrix[0, 2]}],")
print(f"[{new_transformation_matrix[1, 0]}, {new_transformation_matrix[1, 1]}, {new_transformation_matrix[1, 2]}]")
# Blend the aligned image with the reference image for visualization
alpha = 0.5  # Adjust the opacity level as needed
blended_image = cv2.addWeighted(aligned_img1_with_center, alpha, img2_with_center, 1 - alpha, 0)

# Display the blended result
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
plt.title("Blended Image After Alignment")
plt.axis('off')
plt.show()
