#!/usr/bin/env python3
"""
6DoF Camera Pose Estimation using Depth Pro and LightGlue

This script calculates the rotation and translation between two images
using metric depth estimation and sparse feature matching.

Installation (using uv with .venv):
    # Activate the virtual environment
    source .venv/bin/activate

    # Install dependencies
    uv pip install torch torchvision
    uv pip install numpy matplotlib scipy pillow
    uv pip install git+https://github.com/apple/ml-depth-pro.git
    uv pip install git+https://github.com/cvg/LightGlue.git

Usage:
    python camera_pose_estimation.py <image1_path> <image2_path>

Example:
    python camera_pose_estimation.py img1.jpg img2.jpg

Hardware:
    Optimized for Apple M4 Pro (uses MPS device when available).
    Falls back to CPU if MPS is not supported.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from scipy.spatial.transform import Rotation as R


def get_device():
    """Get the best available device (MPS for Mac M4, fallback to CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device")
    return device


def load_depth_pro_model(device):
    """Load the Depth Pro model for metric depth estimation."""
    import depth_pro
    from huggingface_hub import hf_hub_download

    # Download checkpoint from Hugging Face if not present locally
    checkpoint_path = Path("./checkpoints/depth_pro.pt")
    if not checkpoint_path.exists():
        print("  Downloading Depth Pro checkpoint from Hugging Face...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir="./checkpoints"
        )
        print(f"  Checkpoint saved to: {downloaded_path}")

    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    return model, transform


def estimate_depth_and_focal(model, transform, image_path, device):
    """
    Estimate metric depth map and focal length using Depth Pro.

    Returns:
        depth: numpy array of shape (H, W) with metric depth in meters
        focal_length_px: estimated focal length in pixels
        image: PIL Image object
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    # Transform image for the model
    image_tensor = transform(image)

    # Move to device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model.infer(image_tensor)

    # Extract depth and focal length
    depth = prediction["depth"].squeeze().cpu().numpy()

    # Depth Pro returns focal length in pixels for the processed image size
    # We need to scale it to the original image size
    focal_length_px = prediction["focallength_px"].item()

    # The depth map might be at a different resolution than the original image
    # Resize depth to match original image size if needed
    if depth.shape[0] != original_size[1] or depth.shape[1] != original_size[0]:
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(
            depth_tensor,
            size=(original_size[1], original_size[0]),
            mode='bilinear',
            align_corners=False
        )
        depth = depth_resized.squeeze().numpy()

        # Scale focal length proportionally
        scale_factor = original_size[0] / prediction["depth"].shape[-1]
        focal_length_px *= scale_factor

    return depth, focal_length_px, image


def load_lightglue_models(device):
    """Load SuperPoint extractor and LightGlue matcher."""
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image

    # SuperPoint feature extractor
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # LightGlue matcher
    matcher = LightGlue(features='superpoint').eval().to(device)

    return extractor, matcher


def extract_and_match_features(extractor, matcher, image1_path, image2_path, device):
    """
    Extract SuperPoint features and match using LightGlue.

    Returns:
        keypoints1: matched keypoints in image 1, shape (N, 2)
        keypoints2: matched keypoints in image 2, shape (N, 2)
        confidence: match confidence scores, shape (N,)
    """
    from lightglue.utils import load_image

    # Load images for LightGlue (normalized tensor format)
    image1 = load_image(image1_path).to(device)
    image2 = load_image(image2_path).to(device)

    with torch.no_grad():
        # Extract features
        feats1 = extractor.extract(image1)
        feats2 = extractor.extract(image2)

        # Match features
        matches_dict = matcher({"image0": feats1, "image1": feats2})

    # Get matched keypoint indices
    matches = matches_dict["matches"][0].cpu().numpy()  # Shape: (M, 2)

    # Filter valid matches (matches with -1 indicate no match)
    valid_mask = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
    valid_matches = matches[valid_mask]

    if len(valid_matches) == 0:
        raise ValueError("No valid matches found between images!")

    # Get keypoint coordinates
    kpts1 = feats1["keypoints"][0].cpu().numpy()  # Shape: (K1, 2)
    kpts2 = feats2["keypoints"][0].cpu().numpy()  # Shape: (K2, 2)

    # Extract matched keypoints
    matched_kpts1 = kpts1[valid_matches[:, 0]]  # Shape: (N, 2)
    matched_kpts2 = kpts2[valid_matches[:, 1]]  # Shape: (N, 2)

    # Get match scores if available
    if "matching_scores" in matches_dict and matches_dict["matching_scores"] is not None:
        scores = matches_dict["matching_scores"][0].cpu().numpy()
        confidence = scores[valid_mask]
    else:
        confidence = np.ones(len(matched_kpts1))

    print(f"Found {len(matched_kpts1)} matches between images")

    return matched_kpts1, matched_kpts2, confidence


def unproject_to_3d(keypoints, depth_map, focal_length, principal_point=None):
    """
    Unproject 2D keypoints to 3D using depth and camera intrinsics.

    Args:
        keypoints: 2D pixel coordinates, shape (N, 2) as (u, v)
        depth_map: depth map, shape (H, W) with metric depth in meters
        focal_length: focal length in pixels (assuming fx = fy)
        principal_point: (cx, cy), defaults to image center

    Returns:
        points_3d: 3D coordinates, shape (N, 3) as (X, Y, Z)
    """
    H, W = depth_map.shape

    if principal_point is None:
        cx, cy = W / 2, H / 2
    else:
        cx, cy = principal_point

    points_3d = []
    valid_indices = []

    for i, (u, v) in enumerate(keypoints):
        # Ensure coordinates are within bounds
        u_int = int(np.clip(np.round(u), 0, W - 1))
        v_int = int(np.clip(np.round(v), 0, H - 1))

        # Get depth value at this pixel
        z = depth_map[v_int, u_int]

        # Skip invalid depth values
        if z <= 0 or np.isnan(z) or np.isinf(z):
            continue

        # Unproject using pinhole camera model:
        # X = (u - cx) * Z / f
        # Y = (v - cy) * Z / f
        # Z = Z
        x = (u - cx) * z / focal_length
        y = (v - cy) * z / focal_length

        points_3d.append([x, y, z])
        valid_indices.append(i)

    return np.array(points_3d), np.array(valid_indices)


def compute_rigid_transform_svd(points_a, points_b):
    """
    Compute rigid body transformation (R, t) from points_a to points_b using SVD.

    This implements the Procrustes analysis / Kabsch algorithm.

    Args:
        points_a: source 3D points, shape (N, 3)
        points_b: target 3D points, shape (N, 3)

    Returns:
        R: rotation matrix, shape (3, 3)
        t: translation vector, shape (3,)

    The transformation satisfies: points_b ≈ (R @ points_a.T).T + t
    """
    assert points_a.shape == points_b.shape
    assert points_a.shape[0] >= 3, "Need at least 3 points for rigid transformation"

    # Compute centroids
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    # Center the point clouds
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b

    # Compute cross-covariance matrix
    H = centered_a.T @ centered_b

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case (ensure proper rotation with det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_b - R @ centroid_a

    return R, t


def compute_reprojection_error(points_a, points_b, R, t):
    """Compute the mean reprojection error after transformation."""
    transformed = (R @ points_a.T).T + t
    errors = np.linalg.norm(transformed - points_b, axis=1)
    return np.mean(errors), np.std(errors)


def rotation_matrix_to_euler(R_matrix):
    """Convert rotation matrix to Euler angles (in degrees)."""
    r = R.from_matrix(R_matrix)
    euler = r.as_euler('xyz', degrees=True)
    return euler


def create_camera_frustum(R_cam, t_cam, focal_length, img_size, scale=0.3):
    """
    Create vertices for a camera frustum visualization.

    Args:
        R_cam: camera rotation matrix (world to camera)
        t_cam: camera translation (camera position in world)
        focal_length: focal length for frustum aspect
        img_size: (width, height) of image
        scale: size scale for visualization

    Returns:
        vertices: list of 3D points representing the frustum
    """
    w, h = img_size
    aspect = w / h

    # Camera origin (in world coordinates)
    origin = t_cam

    # Frustum corners in camera coordinates (looking down -Z axis)
    half_w = scale * aspect / 2
    half_h = scale / 2
    depth = scale

    corners_cam = np.array([
        [-half_w, -half_h, depth],  # bottom-left
        [half_w, -half_h, depth],   # bottom-right
        [half_w, half_h, depth],    # top-right
        [-half_w, half_h, depth],   # top-left
    ])

    # Transform to world coordinates
    R_world = R_cam.T  # Inverse rotation
    corners_world = (R_world @ corners_cam.T).T + origin

    return origin, corners_world


def visualize_3d(points_a, points_b, R, t, focal_a, focal_b, img_size):
    """
    Visualize the 3D point clouds and camera poses.

    Camera A is at origin, Camera B is transformed by (R, t).
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot point clouds
    ax.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2],
               c='blue', s=20, alpha=0.6, label='Points from Image A')

    # Transform points from A to B's coordinate system for visualization
    points_a_transformed = (R @ points_a.T).T + t
    ax.scatter(points_a_transformed[:, 0], points_a_transformed[:, 1], points_a_transformed[:, 2],
               c='cyan', s=10, alpha=0.3, label='Points A (transformed)')

    ax.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2],
               c='red', s=20, alpha=0.6, label='Points from Image B')

    # Camera A at origin (identity pose)
    R_a = np.eye(3)
    t_a = np.zeros(3)
    origin_a, corners_a = create_camera_frustum(R_a, t_a, focal_a, img_size, scale=0.5)

    # Camera B pose (the transformation takes points from A to B,
    # so camera B is at -R^T @ t relative to A)
    R_b = R
    t_b = t
    origin_b, corners_b = create_camera_frustum(R_b, t_b, focal_b, img_size, scale=0.5)

    # Draw Camera A frustum (blue)
    ax.scatter(*origin_a, c='blue', s=200, marker='^', label='Camera A')
    for i in range(4):
        ax.plot([origin_a[0], corners_a[i, 0]],
                [origin_a[1], corners_a[i, 1]],
                [origin_a[2], corners_a[i, 2]], 'b-', linewidth=2)
    # Connect corners
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners_a[i, 0], corners_a[j, 0]],
                [corners_a[i, 1], corners_a[j, 1]],
                [corners_a[i, 2], corners_a[j, 2]], 'b-', linewidth=2)

    # Draw Camera B frustum (red)
    ax.scatter(*origin_b, c='red', s=200, marker='^', label='Camera B')
    for i in range(4):
        ax.plot([origin_b[0], corners_b[i, 0]],
                [origin_b[1], corners_b[i, 1]],
                [origin_b[2], corners_b[i, 2]], 'r-', linewidth=2)
    # Connect corners
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners_b[i, 0], corners_b[j, 0]],
                [corners_b[i, 1], corners_b[j, 1]],
                [corners_b[i, 2], corners_b[j, 2]], 'r-', linewidth=2)

    # Draw motion vector
    ax.quiver(origin_a[0], origin_a[1], origin_a[2],
              t[0], t[1], t[2],
              color='green', linewidth=3, arrow_length_ratio=0.1,
              label=f'Translation ({np.linalg.norm(t):.3f}m)')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('6DoF Camera Motion Estimation\nCamera A (blue) → Camera B (red)')
    ax.legend(loc='upper left')

    # Set equal aspect ratio
    all_points = np.vstack([points_a, points_b, [origin_a], [origin_b]])
    max_range = np.max(np.ptp(all_points, axis=0)) / 2
    mid = np.mean(all_points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    return fig


def filter_outliers_ransac(points_a, points_b, n_iterations=1000, threshold=0.1):
    """
    Use RANSAC to filter outlier correspondences.

    Args:
        points_a: source 3D points, shape (N, 3)
        points_b: target 3D points, shape (N, 3)
        n_iterations: number of RANSAC iterations
        threshold: inlier threshold in meters

    Returns:
        inlier_mask: boolean mask of inliers
    """
    n_points = len(points_a)
    if n_points < 4:
        return np.ones(n_points, dtype=bool)

    best_inliers = np.zeros(n_points, dtype=bool)
    best_count = 0

    for _ in range(n_iterations):
        # Random sample of 3 points
        indices = np.random.choice(n_points, 3, replace=False)

        try:
            # Compute transformation from sample
            R_sample, t_sample = compute_rigid_transform_svd(
                points_a[indices], points_b[indices]
            )

            # Transform all points
            transformed = (R_sample @ points_a.T).T + t_sample

            # Compute errors
            errors = np.linalg.norm(transformed - points_b, axis=1)

            # Count inliers
            inliers = errors < threshold
            inlier_count = np.sum(inliers)

            if inlier_count > best_count:
                best_count = inlier_count
                best_inliers = inliers

        except Exception:
            continue

    print(f"RANSAC: {best_count}/{n_points} inliers ({100*best_count/n_points:.1f}%)")
    return best_inliers


def main():
    parser = argparse.ArgumentParser(
        description="6DoF Camera Pose Estimation using Depth Pro and LightGlue"
    )
    parser.add_argument("image1", type=str, help="Path to first image")
    parser.add_argument("image2", type=str, help="Path to second image")
    parser.add_argument("--ransac-threshold", type=float, default=0.15,
                        help="RANSAC inlier threshold in meters (default: 0.15)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization")
    args = parser.parse_args()

    # Validate input files
    img1_path = Path(args.image1)
    img2_path = Path(args.image2)

    if not img1_path.exists():
        print(f"Error: Image 1 not found: {img1_path}")
        sys.exit(1)
    if not img2_path.exists():
        print(f"Error: Image 2 not found: {img2_path}")
        sys.exit(1)

    print("=" * 60)
    print("6DoF Camera Pose Estimation")
    print("=" * 60)

    # Get device
    device = get_device()

    # Step 1: Load Depth Pro model
    print("\n[1/5] Loading Depth Pro model...")
    depth_model, depth_transform = load_depth_pro_model(device)

    # Step 2: Estimate depth and focal length for both images
    print("\n[2/5] Estimating metric depth and focal length...")

    print(f"  Processing: {img1_path.name}")
    depth1, focal1, pil_img1 = estimate_depth_and_focal(
        depth_model, depth_transform, str(img1_path), device
    )
    print(f"    Depth range: {depth1.min():.2f}m - {depth1.max():.2f}m")
    print(f"    Focal length: {focal1:.2f} pixels")

    print(f"  Processing: {img2_path.name}")
    depth2, focal2, pil_img2 = estimate_depth_and_focal(
        depth_model, depth_transform, str(img2_path), device
    )
    print(f"    Depth range: {depth2.min():.2f}m - {depth2.max():.2f}m")
    print(f"    Focal length: {focal2:.2f} pixels")

    # Use average focal length
    focal_avg = (focal1 + focal2) / 2
    print(f"  Average focal length: {focal_avg:.2f} pixels")

    # Free up memory
    del depth_model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Step 3: Feature matching with LightGlue
    print("\n[3/5] Extracting and matching features with SuperPoint + LightGlue...")
    extractor, matcher = load_lightglue_models(device)

    keypoints1, keypoints2, confidence = extract_and_match_features(
        extractor, matcher, str(img1_path), str(img2_path), device
    )

    # Free up memory
    del extractor, matcher
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Step 4: Unproject to 3D
    print("\n[4/5] Unprojecting 2D keypoints to 3D coordinates...")

    points_3d_1, valid_idx_1 = unproject_to_3d(keypoints1, depth1, focal1)
    points_3d_2, valid_idx_2 = unproject_to_3d(keypoints2, depth2, focal2)

    # Find common valid indices
    common_mask_1 = np.isin(np.arange(len(keypoints1)), valid_idx_1)
    common_mask_2 = np.isin(np.arange(len(keypoints2)), valid_idx_2)
    common_mask = common_mask_1 & common_mask_2

    # Get corresponding 3D points
    common_indices = np.where(common_mask)[0]

    # Rebuild 3D points for common matches
    points_a = []
    points_b = []
    for idx in common_indices:
        u1, v1 = keypoints1[idx]
        u2, v2 = keypoints2[idx]

        v1_int = int(np.clip(np.round(v1), 0, depth1.shape[0] - 1))
        u1_int = int(np.clip(np.round(u1), 0, depth1.shape[1] - 1))
        v2_int = int(np.clip(np.round(v2), 0, depth2.shape[0] - 1))
        u2_int = int(np.clip(np.round(u2), 0, depth2.shape[1] - 1))

        z1 = depth1[v1_int, u1_int]
        z2 = depth2[v2_int, u2_int]

        if z1 > 0 and z2 > 0 and not np.isnan(z1) and not np.isnan(z2):
            cx1, cy1 = depth1.shape[1] / 2, depth1.shape[0] / 2
            cx2, cy2 = depth2.shape[1] / 2, depth2.shape[0] / 2

            x1 = (u1 - cx1) * z1 / focal1
            y1 = (v1 - cy1) * z1 / focal1

            x2 = (u2 - cx2) * z2 / focal2
            y2 = (v2 - cy2) * z2 / focal2

            points_a.append([x1, y1, z1])
            points_b.append([x2, y2, z2])

    points_a = np.array(points_a)
    points_b = np.array(points_b)

    print(f"  Valid 3D point correspondences: {len(points_a)}")

    if len(points_a) < 4:
        print("Error: Not enough valid 3D correspondences for pose estimation!")
        sys.exit(1)

    # Apply RANSAC to filter outliers
    print("  Applying RANSAC outlier rejection...")
    inlier_mask = filter_outliers_ransac(
        points_a, points_b,
        n_iterations=2000,
        threshold=args.ransac_threshold
    )

    points_a_filtered = points_a[inlier_mask]
    points_b_filtered = points_b[inlier_mask]

    if len(points_a_filtered) < 3:
        print("Warning: Too few inliers after RANSAC, using all points")
        points_a_filtered = points_a
        points_b_filtered = points_b

    # Step 5: Compute rigid transformation using SVD
    print("\n[5/5] Computing rigid body transformation (SVD / Procrustes)...")

    R_matrix, t_vector = compute_rigid_transform_svd(points_a_filtered, points_b_filtered)

    # Compute error metrics
    mean_error, std_error = compute_reprojection_error(
        points_a_filtered, points_b_filtered, R_matrix, t_vector
    )

    # Convert rotation to Euler angles
    euler_angles = rotation_matrix_to_euler(R_matrix)

    # Compute translation magnitude
    translation_magnitude = np.linalg.norm(t_vector)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nEstimated Focal Lengths (pixels):")
    print(f"  Image A: {focal1:.2f}")
    print(f"  Image B: {focal2:.2f}")
    print(f"  Average: {focal_avg:.2f}")

    print(f"\nRotation Matrix R:")
    print(f"  [{R_matrix[0, 0]:+.6f}  {R_matrix[0, 1]:+.6f}  {R_matrix[0, 2]:+.6f}]")
    print(f"  [{R_matrix[1, 0]:+.6f}  {R_matrix[1, 1]:+.6f}  {R_matrix[1, 2]:+.6f}]")
    print(f"  [{R_matrix[2, 0]:+.6f}  {R_matrix[2, 1]:+.6f}  {R_matrix[2, 2]:+.6f}]")

    print(f"\nRotation (Euler angles XYZ, degrees):")
    print(f"  Roll  (X): {euler_angles[0]:+.3f}°")
    print(f"  Pitch (Y): {euler_angles[1]:+.3f}°")
    print(f"  Yaw   (Z): {euler_angles[2]:+.3f}°")

    print(f"\nTranslation Vector t (METERS):")
    print(f"  X: {t_vector[0]:+.4f} m")
    print(f"  Y: {t_vector[1]:+.4f} m")
    print(f"  Z: {t_vector[2]:+.4f} m")

    print(f"\nEuclidean Distance Moved: {translation_magnitude:.4f} METERS")

    print(f"\nTransformation Quality:")
    print(f"  Mean reprojection error: {mean_error:.4f} m")
    print(f"  Std reprojection error:  {std_error:.4f} m")
    print(f"  Inlier correspondences:  {len(points_a_filtered)}")

    print("=" * 60)

    # Visualization
    if not args.no_viz:
        print("\nOpening 3D visualization...")
        img_size = pil_img1.size  # (W, H)
        fig = visualize_3d(
            points_a_filtered, points_b_filtered,
            R_matrix, t_vector,
            focal1, focal2, img_size
        )
        plt.show()

    return R_matrix, t_vector, focal_avg


if __name__ == "__main__":
    main()
