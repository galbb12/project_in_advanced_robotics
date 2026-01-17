#!/usr/bin/env python3
"""
FPV Drone Trajectory Estimator

Estimates the full 3D trajectory from a sequence of FPV video frames
using Depth Pro and LightGlue for visual odometry.

Usage:
    python trajectory_estimator.py <frames_directory>
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=FutureWarning)


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_models(device):
    """Load Depth Pro and LightGlue models."""
    import depth_pro
    from huggingface_hub import hf_hub_download
    from lightglue import LightGlue, SuperPoint

    # Download Depth Pro checkpoint if needed
    checkpoint_path = Path("./checkpoints/depth_pro.pt")
    if not checkpoint_path.exists():
        print("  Downloading Depth Pro checkpoint...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir="./checkpoints"
        )

    depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
    depth_model.eval()

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    return depth_model, depth_transform, extractor, matcher


def estimate_depth_and_focal(model, transform, image_path, device):
    """Estimate metric depth and focal length for an image."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image_tensor = transform(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model.infer(image_tensor)

    depth = prediction["depth"].squeeze().cpu().numpy()
    focal_length_px = prediction["focallength_px"].item()

    if depth.shape[0] != original_size[1] or depth.shape[1] != original_size[0]:
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_resized = F.interpolate(
            depth_tensor,
            size=(original_size[1], original_size[0]),
            mode='bilinear',
            align_corners=False
        )
        depth = depth_resized.squeeze().numpy()
        scale_factor = original_size[0] / prediction["depth"].shape[-1]
        focal_length_px *= scale_factor

    return depth, focal_length_px


def match_features(extractor, matcher, img1_path, img2_path, device):
    """Extract and match features between two images."""
    from lightglue.utils import load_image

    image1 = load_image(img1_path).to(device)
    image2 = load_image(img2_path).to(device)

    with torch.no_grad():
        feats1 = extractor.extract(image1)
        feats2 = extractor.extract(image2)
        matches_dict = matcher({"image0": feats1, "image1": feats2})

    matches = matches_dict["matches"][0].cpu().numpy()
    valid_mask = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
    valid_matches = matches[valid_mask]

    if len(valid_matches) == 0:
        return None, None

    kpts1 = feats1["keypoints"][0].cpu().numpy()
    kpts2 = feats2["keypoints"][0].cpu().numpy()

    return kpts1[valid_matches[:, 0]], kpts2[valid_matches[:, 1]]


def unproject_points(keypoints, depth_map, focal_length):
    """Unproject 2D keypoints to 3D."""
    H, W = depth_map.shape
    cx, cy = W / 2, H / 2

    points_3d = []
    valid_indices = []

    for i, (u, v) in enumerate(keypoints):
        u_int = int(np.clip(np.round(u), 0, W - 1))
        v_int = int(np.clip(np.round(v), 0, H - 1))
        z = depth_map[v_int, u_int]

        if z > 0 and not np.isnan(z) and not np.isinf(z):
            x = (u - cx) * z / focal_length
            y = (v - cy) * z / focal_length
            points_3d.append([x, y, z])
            valid_indices.append(i)

    return np.array(points_3d) if points_3d else None, np.array(valid_indices)


def compute_transform_ransac(points_a, points_b, n_iterations=1000, threshold=0.1):
    """Compute rigid transform with RANSAC."""
    n_points = len(points_a)
    if n_points < 4:
        return None, None, 0

    best_R, best_t = None, None
    best_inliers = 0

    for _ in range(n_iterations):
        indices = np.random.choice(n_points, 3, replace=False)
        try:
            # SVD-based rigid transform
            centroid_a = np.mean(points_a[indices], axis=0)
            centroid_b = np.mean(points_b[indices], axis=0)
            centered_a = points_a[indices] - centroid_a
            centered_b = points_b[indices] - centroid_b
            H = centered_a.T @ centered_b
            U, _, Vt = np.linalg.svd(H)
            R_sample = Vt.T @ U.T
            if np.linalg.det(R_sample) < 0:
                Vt[-1, :] *= -1
                R_sample = Vt.T @ U.T
            t_sample = centroid_b - R_sample @ centroid_a

            # Count inliers
            transformed = (R_sample @ points_a.T).T + t_sample
            errors = np.linalg.norm(transformed - points_b, axis=1)
            inliers = np.sum(errors < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_R, best_t = R_sample, t_sample
        except Exception:
            continue

    # Refine with all inliers
    if best_R is not None and best_inliers > 3:
        transformed = (best_R @ points_a.T).T + best_t
        errors = np.linalg.norm(transformed - points_b, axis=1)
        inlier_mask = errors < threshold

        centroid_a = np.mean(points_a[inlier_mask], axis=0)
        centroid_b = np.mean(points_b[inlier_mask], axis=0)
        centered_a = points_a[inlier_mask] - centroid_a
        centered_b = points_b[inlier_mask] - centroid_b
        H = centered_a.T @ centered_b
        U, _, Vt = np.linalg.svd(H)
        best_R = Vt.T @ U.T
        if np.linalg.det(best_R) < 0:
            Vt[-1, :] *= -1
            best_R = Vt.T @ U.T
        best_t = centroid_b - best_R @ centroid_a

    return best_R, best_t, best_inliers


def estimate_pairwise_pose(depth_model, depth_transform, extractor, matcher,
                           img1_path, img2_path, device):
    """Estimate relative pose between two images."""
    # Get depth and focal length
    depth1, focal1 = estimate_depth_and_focal(depth_model, depth_transform, str(img1_path), device)
    depth2, focal2 = estimate_depth_and_focal(depth_model, depth_transform, str(img2_path), device)

    # Match features
    kpts1, kpts2 = match_features(extractor, matcher, str(img1_path), str(img2_path), device)
    if kpts1 is None or len(kpts1) < 10:
        return None, None, 0

    # Unproject to 3D
    points_a = []
    points_b = []
    H1, W1 = depth1.shape
    H2, W2 = depth2.shape
    cx1, cy1 = W1 / 2, H1 / 2
    cx2, cy2 = W2 / 2, H2 / 2

    for (u1, v1), (u2, v2) in zip(kpts1, kpts2):
        u1_int = int(np.clip(np.round(u1), 0, W1 - 1))
        v1_int = int(np.clip(np.round(v1), 0, H1 - 1))
        u2_int = int(np.clip(np.round(u2), 0, W2 - 1))
        v2_int = int(np.clip(np.round(v2), 0, H2 - 1))

        z1 = depth1[v1_int, u1_int]
        z2 = depth2[v2_int, u2_int]

        if z1 > 0 and z2 > 0 and not np.isnan(z1) and not np.isnan(z2):
            x1 = (u1 - cx1) * z1 / focal1
            y1 = (v1 - cy1) * z1 / focal1
            x2 = (u2 - cx2) * z2 / focal2
            y2 = (v2 - cy2) * z2 / focal2
            points_a.append([x1, y1, z1])
            points_b.append([x2, y2, z2])

    if len(points_a) < 10:
        return None, None, 0

    points_a = np.array(points_a)
    points_b = np.array(points_b)

    # Compute transform with RANSAC
    R_mat, t_vec, n_inliers = compute_transform_ransac(points_a, points_b)

    return R_mat, t_vec, n_inliers


def accumulate_trajectory(rotations: List[np.ndarray], translations: List[np.ndarray]):
    """Accumulate relative poses into global trajectory."""
    positions = [np.zeros(3)]  # Start at origin
    orientations = [np.eye(3)]  # Start with identity rotation

    global_R = np.eye(3)
    global_t = np.zeros(3)

    for R_rel, t_rel in zip(rotations, translations):
        # Update global pose: new_pose = old_pose * relative_pose
        # t_global = t_global + R_global @ t_rel
        # R_global = R_global @ R_rel
        global_t = global_t + global_R @ t_rel
        global_R = global_R @ R_rel

        positions.append(global_t.copy())
        orientations.append(global_R.copy())

    return np.array(positions), orientations


def plot_trajectory_3d(positions, orientations, save_path=None):
    """Plot the 3D camera trajectory with orientation indicators."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Camera Path')

    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=np.arange(len(positions)), cmap='viridis', s=50)

    # Mark start and end
    ax.scatter(*positions[0], c='green', s=200, marker='^', label='Start')
    ax.scatter(*positions[-1], c='red', s=200, marker='v', label='End')

    # Draw camera frustums at regular intervals
    n_frustums = min(10, len(positions))
    indices = np.linspace(0, len(positions) - 1, n_frustums, dtype=int)

    for i in indices:
        pos = positions[i]
        R_cam = orientations[i]

        # Camera axes (scaled for visibility)
        scale = 0.3
        # Z-axis (forward) - blue
        z_axis = R_cam @ np.array([0, 0, scale])
        ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2],
                  color='blue', alpha=0.5, arrow_length_ratio=0.2)
        # X-axis (right) - red
        x_axis = R_cam @ np.array([scale * 0.5, 0, 0])
        ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2],
                  color='red', alpha=0.5, arrow_length_ratio=0.2)
        # Y-axis (down) - green
        y_axis = R_cam @ np.array([0, scale * 0.5, 0])
        ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2],
                  color='green', alpha=0.5, arrow_length_ratio=0.2)

    # Compute total distance
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'FPV Drone Trajectory\nTotal Distance: {total_distance:.2f} meters')
    ax.legend()

    # Equal aspect ratio
    max_range = np.max(np.ptp(positions, axis=0)) / 2 + 0.5
    mid = np.mean(positions, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="FPV Drone Trajectory Estimator")
    parser.add_argument("frames_dir", type=str, help="Directory containing video frames")
    parser.add_argument("--output", type=str, default="trajectory_3d.png",
                        help="Output image path")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every N-th frame (default: 1)")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"Error: Directory not found: {frames_dir}")
        sys.exit(1)

    # Get sorted list of frames
    frames = sorted(frames_dir.glob("*.png"))
    if len(frames) < 2:
        frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) < 2:
        print("Error: Need at least 2 frames")
        sys.exit(1)

    # Apply skip
    frames = frames[::args.skip]
    print(f"Processing {len(frames)} frames...")

    # Setup device and models
    device = get_device()
    print(f"Using device: {device}")

    print("Loading models...")
    depth_model, depth_transform, extractor, matcher = load_models(device)

    # Process frame pairs
    rotations = []
    translations = []
    successful_pairs = 0

    print("\nEstimating pairwise poses:")
    for i in range(len(frames) - 1):
        img1 = frames[i]
        img2 = frames[i + 1]

        R_rel, t_rel, n_inliers = estimate_pairwise_pose(
            depth_model, depth_transform, extractor, matcher,
            img1, img2, device
        )

        if R_rel is not None and n_inliers >= 10:
            rotations.append(R_rel)
            translations.append(t_rel)
            successful_pairs += 1

            # Print progress
            euler = R.from_matrix(R_rel).as_euler('xyz', degrees=True)
            dist = np.linalg.norm(t_rel)
            print(f"  [{i+1:3d}/{len(frames)-1}] {img1.name} → {img2.name}: "
                  f"t={dist:.3f}m, rot=({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°), "
                  f"inliers={n_inliers}")
        else:
            print(f"  [{i+1:3d}/{len(frames)-1}] {img1.name} → {img2.name}: FAILED (inliers={n_inliers})")
            # Use identity transform for failed pairs
            rotations.append(np.eye(3))
            translations.append(np.zeros(3))

    print(f"\nSuccessfully estimated {successful_pairs}/{len(frames)-1} frame pairs")

    # Accumulate trajectory
    positions, orientations = accumulate_trajectory(rotations, translations)

    # Statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    displacement = np.linalg.norm(positions[-1] - positions[0])

    print(f"\n{'='*60}")
    print("TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    print(f"Total path length:  {total_distance:.3f} meters")
    print(f"Net displacement:   {displacement:.3f} meters")
    print(f"Start position:     ({positions[0][0]:.3f}, {positions[0][1]:.3f}, {positions[0][2]:.3f})")
    print(f"End position:       ({positions[-1][0]:.3f}, {positions[-1][1]:.3f}, {positions[-1][2]:.3f})")
    print(f"{'='*60}")

    # Plot and save
    output_path = Path(args.output)
    fig = plot_trajectory_3d(positions, orientations, save_path=str(output_path))

    print("\nOpening 3D visualization...")
    plt.show()

    return positions, orientations


if __name__ == "__main__":
    main()
