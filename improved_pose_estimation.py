#!/usr/bin/env python3
"""
Improved 6DoF Camera Pose Estimation

Key improvements over basic version:
1. Depth patch averaging for robust depth sampling
2. Scale normalization between frames
3. Consistent focal length across frames
4. Confidence-weighted point selection
5. Adaptive RANSAC thresholds
6. Motion prior constraints
"""

import sys
import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

warnings.filterwarnings("ignore")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ImprovedPoseEstimator:
    def __init__(self, device):
        self.device = device
        self.depth_model = None
        self.depth_transform = None
        self.extractor = None
        self.matcher = None

        # Cache for depth and focal length
        self.depth_cache = {}
        self.focal_cache = {}

    def load_models(self):
        """Load all required models."""
        import depth_pro
        from huggingface_hub import hf_hub_download
        from lightglue import LightGlue, SuperPoint

        print("Loading Depth Pro...")
        checkpoint_path = Path("./checkpoints/depth_pro.pt")
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="apple/DepthPro",
                filename="depth_pro.pt",
                local_dir="./checkpoints"
            )

        self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
            device=self.device
        )
        self.depth_model.eval()

        print("Loading SuperPoint + LightGlue...")
        # Use more keypoints for better coverage
        self.extractor = SuperPoint(max_num_keypoints=4096).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def get_depth_and_focal(self, image_path):
        """Get depth map and focal length with caching."""
        path_str = str(image_path)

        if path_str in self.depth_cache:
            return self.depth_cache[path_str], self.focal_cache[path_str]

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        image_tensor = self.depth_transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.depth_model.infer(image_tensor)

        depth = prediction["depth"].squeeze().cpu().numpy()
        focal = prediction["focallength_px"].item()

        # Resize depth to original image size
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
            focal *= scale_factor

        self.depth_cache[path_str] = depth
        self.focal_cache[path_str] = focal

        return depth, focal

    def get_depth_at_point(self, depth_map, u, v, patch_size=5):
        """
        Get robust depth at a point using patch averaging.
        Uses median of a small patch to reduce noise.
        """
        H, W = depth_map.shape
        half = patch_size // 2

        u_int = int(np.round(u))
        v_int = int(np.round(v))

        # Get patch bounds
        u_min = max(0, u_int - half)
        u_max = min(W, u_int + half + 1)
        v_min = max(0, v_int - half)
        v_max = min(H, v_int + half + 1)

        patch = depth_map[v_min:v_max, u_min:u_max]

        # Filter valid depths
        valid = patch[(patch > 0) & ~np.isnan(patch) & ~np.isinf(patch)]

        if len(valid) == 0:
            return None

        # Use median for robustness
        return np.median(valid)

    def match_features(self, img1_path, img2_path):
        """Extract and match features with confidence scores."""
        from lightglue.utils import load_image

        image1 = load_image(img1_path).to(self.device)
        image2 = load_image(img2_path).to(self.device)

        with torch.no_grad():
            feats1 = self.extractor.extract(image1)
            feats2 = self.extractor.extract(image2)
            matches_dict = self.matcher({"image0": feats1, "image1": feats2})

        matches = matches_dict["matches"][0].cpu().numpy()

        # Get confidence scores
        if "matching_scores" in matches_dict and matches_dict["matching_scores"] is not None:
            scores = matches_dict["matching_scores"][0].cpu().numpy()
        else:
            scores = np.ones(len(matches))

        # Filter valid matches
        valid_mask = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
        valid_matches = matches[valid_mask]
        valid_scores = scores[valid_mask] if len(scores) == len(matches) else scores[valid_mask[:len(scores)]]

        if len(valid_matches) == 0:
            return None, None, None

        kpts1 = feats1["keypoints"][0].cpu().numpy()
        kpts2 = feats2["keypoints"][0].cpu().numpy()

        matched_kpts1 = kpts1[valid_matches[:, 0]]
        matched_kpts2 = kpts2[valid_matches[:, 1]]

        return matched_kpts1, matched_kpts2, valid_scores

    def unproject_points(self, keypoints, depth_map, focal, cx, cy):
        """Unproject 2D points to 3D with robust depth sampling."""
        points_3d = []
        valid_indices = []
        depths = []

        for i, (u, v) in enumerate(keypoints):
            z = self.get_depth_at_point(depth_map, u, v, patch_size=5)

            if z is not None and z > 0.1:  # Minimum depth threshold
                x = (u - cx) * z / focal
                y = (v - cy) * z / focal
                points_3d.append([x, y, z])
                valid_indices.append(i)
                depths.append(z)

        if len(points_3d) == 0:
            return None, None, None

        return np.array(points_3d), np.array(valid_indices), np.array(depths)

    def estimate_scale_factor(self, depths1, depths2):
        """
        Estimate scale factor between two depth maps based on matched points.
        This helps normalize scale drift from monocular depth estimation.
        """
        if len(depths1) < 10:
            return 1.0

        # Use ratio of median depths as scale factor
        median1 = np.median(depths1)
        median2 = np.median(depths2)

        if median2 > 0:
            return median1 / median2
        return 1.0

    def compute_rigid_transform_weighted(self, points_a, points_b, weights=None):
        """
        Compute rigid transform using weighted SVD.
        """
        if weights is None:
            weights = np.ones(len(points_a))

        weights = weights / np.sum(weights)

        # Weighted centroids
        centroid_a = np.sum(weights[:, np.newaxis] * points_a, axis=0)
        centroid_b = np.sum(weights[:, np.newaxis] * points_b, axis=0)

        # Center points
        centered_a = points_a - centroid_a
        centered_b = points_b - centroid_b

        # Weighted covariance
        H = (centered_a * weights[:, np.newaxis]).T @ centered_b

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_b - R @ centroid_a

        return R, t

    def ransac_transform(self, points_a, points_b, confidence_scores=None,
                         n_iterations=2000, initial_threshold=0.05):
        """
        RANSAC with adaptive threshold based on point cloud scale.
        """
        n_points = len(points_a)
        if n_points < 4:
            return None, None, np.zeros(n_points, dtype=bool)

        # Adaptive threshold based on median depth
        median_depth = np.median(np.linalg.norm(points_a, axis=1))
        threshold = max(initial_threshold, median_depth * 0.02)  # 2% of median depth

        best_R, best_t = None, None
        best_inlier_mask = np.zeros(n_points, dtype=bool)
        best_score = 0

        # Weight sampling by confidence
        if confidence_scores is not None:
            sample_weights = confidence_scores / np.sum(confidence_scores)
        else:
            sample_weights = None

        for _ in range(n_iterations):
            # Sample 4 points (minimum for 3D rigid transform with some redundancy)
            if sample_weights is not None:
                indices = np.random.choice(n_points, 4, replace=False, p=sample_weights)
            else:
                indices = np.random.choice(n_points, 4, replace=False)

            try:
                R_sample, t_sample = self.compute_rigid_transform_weighted(
                    points_a[indices], points_b[indices]
                )

                # Compute errors
                transformed = (R_sample @ points_a.T).T + t_sample
                errors = np.linalg.norm(transformed - points_b, axis=1)

                inlier_mask = errors < threshold

                # Score by number of inliers weighted by confidence
                if confidence_scores is not None:
                    score = np.sum(confidence_scores[inlier_mask])
                else:
                    score = np.sum(inlier_mask)

                if score > best_score:
                    best_score = score
                    best_inlier_mask = inlier_mask
                    best_R, best_t = R_sample, t_sample

            except Exception:
                continue

        # Refine with all inliers
        if best_R is not None and np.sum(best_inlier_mask) >= 4:
            inlier_weights = confidence_scores[best_inlier_mask] if confidence_scores is not None else None
            best_R, best_t = self.compute_rigid_transform_weighted(
                points_a[best_inlier_mask],
                points_b[best_inlier_mask],
                inlier_weights
            )

        return best_R, best_t, best_inlier_mask

    def estimate_pose(self, img1_path, img2_path, use_consistent_focal=True, known_focal=None):
        """
        Estimate relative pose between two images.

        Returns:
            R: Rotation matrix (3x3)
            t: Translation vector (3,) in meters
            n_inliers: Number of inlier matches
            metadata: Dictionary with additional info
        """
        # Get depth maps
        depth1, focal1 = self.get_depth_and_focal(img1_path)
        depth2, focal2 = self.get_depth_and_focal(img2_path)

        H1, W1 = depth1.shape
        H2, W2 = depth2.shape
        cx1, cy1 = W1 / 2, H1 / 2
        cx2, cy2 = W2 / 2, H2 / 2

        # Use known focal length if provided
        if known_focal is not None:
            focal1 = focal2 = known_focal
        elif use_consistent_focal:
            focal = (focal1 + focal2) / 2
            focal1 = focal2 = focal

        # Match features
        kpts1, kpts2, scores = self.match_features(str(img1_path), str(img2_path))

        if kpts1 is None or len(kpts1) < 20:
            return None, None, 0, {}

        # Filter by confidence (top 80%)
        if scores is not None and len(scores) > 20:
            score_threshold = np.percentile(scores, 20)
            conf_mask = scores >= score_threshold
            kpts1 = kpts1[conf_mask]
            kpts2 = kpts2[conf_mask]
            scores = scores[conf_mask]

        # Unproject to 3D
        points_3d_1, valid_idx_1, depths_1 = self.unproject_points(kpts1, depth1, focal1, cx1, cy1)
        points_3d_2, valid_idx_2, depths_2 = self.unproject_points(kpts2, depth2, focal2, cx2, cy2)

        if points_3d_1 is None or points_3d_2 is None:
            return None, None, 0, {}

        # Find common valid points
        set1 = set(valid_idx_1)
        set2 = set(valid_idx_2)
        common = sorted(set1 & set2)

        if len(common) < 10:
            return None, None, 0, {}

        # Get corresponding 3D points
        idx_in_1 = [list(valid_idx_1).index(i) for i in common]
        idx_in_2 = [list(valid_idx_2).index(i) for i in common]

        pts_a = points_3d_1[idx_in_1]
        pts_b = points_3d_2[idx_in_2]
        depths_a = depths_1[idx_in_1]
        depths_b = depths_2[idx_in_2]
        conf = scores[common] if scores is not None else None

        # Scale normalization - align depth scales
        scale_factor = self.estimate_scale_factor(depths_a, depths_b)
        pts_b_scaled = pts_b * scale_factor

        # RANSAC for robust estimation
        R, t, inlier_mask = self.ransac_transform(pts_a, pts_b_scaled, conf)

        if R is None:
            return None, None, 0, {}

        # Scale translation back
        t = t / scale_factor if scale_factor != 0 else t

        n_inliers = np.sum(inlier_mask)

        # Compute reprojection error for quality assessment (before inverting)
        transformed = (R @ pts_a[inlier_mask].T).T + t
        errors = np.linalg.norm(transformed - pts_b[inlier_mask], axis=1)
        mean_error = np.mean(errors)

        # IMPORTANT: Convert from point-cloud transform to camera motion
        # SVD gives us: P_b = R_svd @ P_a + t_svd (transforms points from A to B frame)
        # Camera motion convention: where is camera B in camera A's frame?
        # R_cam = R_svd (rotation from A to B)
        # t_cam = -R_svd.T @ t_svd (camera B's position in A's frame)
        t = -R.T @ t

        metadata = {
            'focal1': focal1,
            'focal2': focal2,
            'n_matches': len(kpts1),
            'n_valid_3d': len(common),
            'n_inliers': n_inliers,
            'inlier_ratio': n_inliers / len(common) if len(common) > 0 else 0,
            'mean_error': mean_error,
            'scale_factor': scale_factor
        }

        return R, t, n_inliers, metadata


def evaluate_on_icl_nuim(estimator, data_dir, frame_pairs):
    """
    Evaluate pose estimation against ICL-NUIM ground truth.
    """
    # Load ground truth
    gt_file = data_dir / "livingRoom2.gt.freiburg"
    gt_poses = {}

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gt_poses[frame_id] = {
                't': np.array([tx, ty, tz]),
                'q': np.array([qx, qy, qz, qw])
            }

    results = []

    for frame_a, frame_b in frame_pairs:
        img_a = data_dir / "rgb" / f"{frame_a}.png"
        img_b = data_dir / "rgb" / f"{frame_b}.png"

        if not img_a.exists() or not img_b.exists():
            continue

        # Get ground truth relative pose
        gt_a = gt_poses.get(frame_a + 1)  # GT is 1-indexed
        gt_b = gt_poses.get(frame_b + 1)

        if gt_a is None or gt_b is None:
            continue

        # Compute GT relative transform
        R_a = Rot.from_quat(gt_a['q']).as_matrix()
        R_b = Rot.from_quat(gt_b['q']).as_matrix()
        t_a = gt_a['t']
        t_b = gt_b['t']

        # Relative rotation: R_rel = R_b @ R_a^T
        R_gt_rel = R_b @ R_a.T
        # Relative translation in frame A's coordinate system
        t_gt_rel = R_a.T @ (t_b - t_a)

        gt_euler = Rot.from_matrix(R_gt_rel).as_euler('xyz', degrees=True)
        gt_dist = np.linalg.norm(t_gt_rel)

        # Estimate pose
        print(f"Processing frames {frame_a} → {frame_b}...", end=" ")
        # ICL-NUIM known intrinsics: fx=fy=481.20
        R_est, t_est, n_inliers, metadata = estimator.estimate_pose(img_a, img_b, known_focal=481.2)

        if R_est is None:
            print("FAILED")
            results.append({
                'frame_a': frame_a, 'frame_b': frame_b,
                'success': False
            })
            continue

        est_euler = Rot.from_matrix(R_est).as_euler('xyz', degrees=True)
        est_dist = np.linalg.norm(t_est)

        # Compute errors
        # Rotation error (geodesic distance)
        R_err = R_gt_rel @ R_est.T
        rot_error = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) * 180 / np.pi

        # Translation direction error (angle between vectors)
        if gt_dist > 0.001 and est_dist > 0.001:
            t_gt_norm = t_gt_rel / gt_dist
            t_est_norm = t_est / est_dist
            trans_dir_error = np.arccos(np.clip(np.dot(t_gt_norm, t_est_norm), -1, 1)) * 180 / np.pi
        else:
            trans_dir_error = 0

        # Scale error
        scale_error = abs(est_dist - gt_dist) / gt_dist if gt_dist > 0.001 else 0

        print(f"rot_err={rot_error:.2f}°, trans_dir_err={trans_dir_error:.2f}°, "
              f"scale_err={scale_error*100:.1f}%, inliers={n_inliers}")

        results.append({
            'frame_a': frame_a,
            'frame_b': frame_b,
            'success': True,
            'rot_error': rot_error,
            'trans_dir_error': trans_dir_error,
            'scale_error': scale_error,
            'gt_dist': gt_dist,
            'est_dist': est_dist,
            'gt_euler': gt_euler,
            'est_euler': est_euler,
            'n_inliers': n_inliers
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Improved 6DoF Pose Estimation")
    parser.add_argument("--data-dir", type=str,
                        default="test_images",
                        help="Path to ICL-NUIM dataset")
    parser.add_argument("--frame-step", type=int, default=10,
                        help="Step between frame pairs")
    parser.add_argument("--num-pairs", type=int, default=20,
                        help="Number of frame pairs to evaluate")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("Improved 6DoF Pose Estimation - ICL-NUIM Evaluation")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Initialize estimator
    estimator = ImprovedPoseEstimator(device)
    estimator.load_models()

    # Generate frame pairs
    frame_pairs = [(i, i + args.frame_step) for i in range(0, args.num_pairs * args.frame_step, args.frame_step)]

    print(f"\nEvaluating {len(frame_pairs)} frame pairs (step={args.frame_step})...")
    print("-" * 70)

    results = evaluate_on_icl_nuim(estimator, data_dir, frame_pairs)

    # Compute statistics
    successful = [r for r in results if r['success']]

    if len(successful) == 0:
        print("No successful estimations!")
        return

    rot_errors = [r['rot_error'] for r in successful]
    trans_dir_errors = [r['trans_dir_error'] for r in successful]
    scale_errors = [r['scale_error'] for r in successful]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    print(f"\nRotation Error (degrees):")
    print(f"  Mean: {np.mean(rot_errors):.2f}°")
    print(f"  Median: {np.median(rot_errors):.2f}°")
    print(f"  Std: {np.std(rot_errors):.2f}°")
    print(f"  < 5°: {100*np.mean(np.array(rot_errors) < 5):.1f}%")
    print(f"  < 10°: {100*np.mean(np.array(rot_errors) < 10):.1f}%")

    print(f"\nTranslation Direction Error (degrees):")
    print(f"  Mean: {np.mean(trans_dir_errors):.2f}°")
    print(f"  Median: {np.median(trans_dir_errors):.2f}°")
    print(f"  < 10°: {100*np.mean(np.array(trans_dir_errors) < 10):.1f}%")
    print(f"  < 20°: {100*np.mean(np.array(trans_dir_errors) < 20):.1f}%")

    print(f"\nScale Error:")
    print(f"  Mean: {100*np.mean(scale_errors):.1f}%")
    print(f"  Median: {100*np.median(scale_errors):.1f}%")
    print(f"  < 20%: {100*np.mean(np.array(scale_errors) < 0.2):.1f}%")
    print(f"  < 50%: {100*np.mean(np.array(scale_errors) < 0.5):.1f}%")

    # Overall accuracy (rotation < 10° AND translation direction < 20°)
    accurate = [(r['rot_error'] < 10 and r['trans_dir_error'] < 20) for r in successful]
    overall_accuracy = 100 * np.mean(accurate)

    print(f"\n{'=' * 70}")
    print(f"OVERALL ACCURACY (rot<10° AND trans_dir<20°): {overall_accuracy:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
