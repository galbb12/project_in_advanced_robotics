#!/usr/bin/env python3
"""
Competition-Grade Monocular Visual Odometry

Focus: Accurate rotation and translation DIRECTION (normalized).
Uses known camera intrinsics (K).

Key techniques:
1. USAC_MAGSAC for robust Essential Matrix estimation
2. Proper E decomposition with cheirality validation
3. Local Bundle Adjustment for pose refinement
4. High-quality feature matching (SuperPoint + LightGlue)
"""

import sys
import argparse
from pathlib import Path
import warnings
from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")


@dataclass
class PoseEstimate:
    """Estimated relative pose between two frames."""
    R: np.ndarray          # 3x3 rotation matrix
    t: np.ndarray          # 3x1 normalized translation (unit vector)
    t_scaled: np.ndarray   # 3x1 metric translation (if depth available)
    n_inliers: int
    inlier_ratio: float
    confidence: float      # 0-1 confidence score
    method: str


class MonocularVO:
    """Competition-grade monocular visual odometry."""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.extractor = None
        self.matcher = None
        self.feature_cache = {}

    def load_models(self):
        """Load feature extraction models."""
        from lightglue import LightGlue, SuperPoint

        print("Loading SuperPoint + LightGlue...")
        self.extractor = SuperPoint(max_num_keypoints=8192).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def extract_features(self, image_path: str) -> dict:
        """Extract SuperPoint features with caching."""
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]

        from lightglue.utils import load_image
        image = load_image(image_path).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(image)

        self.feature_cache[image_path] = feats
        return feats

    def match_features(self, img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match features between two images."""
        feats1 = self.extract_features(img1_path)
        feats2 = self.extract_features(img2_path)

        with torch.no_grad():
            matches_dict = self.matcher({"image0": feats1, "image1": feats2})

        matches = matches_dict["matches"][0].cpu().numpy()
        scores = matches_dict.get("matching_scores")
        if scores is not None:
            scores = scores[0].cpu().numpy()
        else:
            scores = np.ones(len(matches))

        valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
        valid_matches = matches[valid]
        valid_scores = scores[valid] if len(scores) == len(matches) else scores[:sum(valid)]

        if len(valid_matches) == 0:
            return None, None, None

        kpts1 = feats1["keypoints"][0].cpu().numpy()
        kpts2 = feats2["keypoints"][0].cpu().numpy()

        pts1 = kpts1[valid_matches[:, 0]]
        pts2 = kpts2[valid_matches[:, 1]]

        return pts1, pts2, valid_scores

    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from two views."""
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])

        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts_3d = (pts_4d[:3] / pts_4d[3]).T

        return pts_3d

    def check_cheirality(self, pts1: np.ndarray, pts2: np.ndarray,
                        R: np.ndarray, t: np.ndarray, K: np.ndarray) -> int:
        """Count points in front of both cameras (cheirality check)."""
        pts_3d = self.triangulate_points(pts1, pts2, R, t, K)

        # Points in camera 1 frame (should have Z > 0)
        in_front_1 = pts_3d[:, 2] > 0

        # Points in camera 2 frame
        pts_3d_cam2 = (R @ pts_3d.T).T + t.ravel()
        in_front_2 = pts_3d_cam2[:, 2] > 0

        return np.sum(in_front_1 & in_front_2)

    def decompose_essential(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                           K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Decompose Essential matrix and select best solution via cheirality.
        Returns R, t (normalized), and number of valid points.

        OpenCV recoverPose returns R, t such that:
        - R rotates points from cam1 to cam2 frame
        - t is the translation of cam2 origin in cam1 frame (unit vector)
        """
        n_valid, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        t = t.ravel()

        # Transform t to cam1 frame
        t = R.T @ t
        t = t / (np.linalg.norm(t) + 1e-8)

        return R, t, n_valid

    def estimate_essential_robust(self, pts1: np.ndarray, pts2: np.ndarray,
                                  K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Robust Essential Matrix estimation using USAC_MAGSAC.
        Returns E, inlier_mask, n_inliers.
        """
        # Try USAC_MAGSAC first (best), fall back to RANSAC
        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.USAC_MAGSAC,
                prob=0.9999,
                threshold=0.5,
                maxIters=10000
            )
        except:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.9999,
                threshold=1.0
            )

        if E is None or mask is None:
            return None, None, 0

        mask = mask.ravel().astype(bool)
        return E, mask, np.sum(mask)

    def refine_pose_ba(self, pts1: np.ndarray, pts2: np.ndarray,
                       R: np.ndarray, t: np.ndarray, K: np.ndarray,
                       max_iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine pose using bundle adjustment (minimize reprojection error).
        """
        # Convert R to axis-angle for optimization
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.ravel()
        t = t.ravel()

        # Only optimize direction of t (keep unit norm)
        # Parameterize t as 2 angles (spherical coordinates)
        theta = np.arccos(np.clip(t[2], -1, 1))
        phi = np.arctan2(t[1], t[0])

        x0 = np.hstack([rvec, theta, phi])

        def residuals(x):
            rvec_opt = x[:3]
            theta_opt, phi_opt = x[3], x[4]

            # Reconstruct t from spherical coords
            t_opt = np.array([
                np.sin(theta_opt) * np.cos(phi_opt),
                np.sin(theta_opt) * np.sin(phi_opt),
                np.cos(theta_opt)
            ])

            R_opt, _ = cv2.Rodrigues(rvec_opt)

            # Triangulate
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R_opt, t_opt.reshape(3, 1)])

            pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T

            # Project to camera 2
            pts_cam2 = (R_opt @ pts_3d.T).T + t_opt
            proj2 = pts_cam2[:, :2] / pts_cam2[:, 2:3]
            proj2 = proj2 * K[0, 0] + np.array([K[0, 2], K[1, 2]])

            # Reprojection error
            err = (proj2 - pts2).ravel()

            # Also ensure positive depth
            depth_penalty = np.sum(np.maximum(0, -pts_3d[:, 2])) * 100
            depth_penalty += np.sum(np.maximum(0, -pts_cam2[:, 2])) * 100

            return np.append(err, depth_penalty)

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=max_iters)
            x_opt = result.x

            R_opt, _ = cv2.Rodrigues(x_opt[:3])
            theta_opt, phi_opt = x_opt[3], x_opt[4]
            t_opt = np.array([
                np.sin(theta_opt) * np.cos(phi_opt),
                np.sin(theta_opt) * np.sin(phi_opt),
                np.cos(theta_opt)
            ])

            return R_opt, t_opt
        except:
            return R, t

    def estimate_pose(self, img1_path: str, img2_path: str,
                     focal: float = None, use_ba: bool = True) -> PoseEstimate:
        """
        Estimate relative pose between two images.

        Returns PoseEstimate with:
        - R: rotation matrix
        - t: normalized translation (unit vector)
        - t_scaled: metric translation (if depth available)
        """
        # Get image size for principal point
        img = Image.open(img1_path)
        W, H = img.size
        cx, cy = W / 2, H / 2

        # Use provided focal or default estimate
        if focal is None:
            focal = max(W, H) * 1.2

        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

        # Match features
        pts1, pts2, scores = self.match_features(img1_path, img2_path)
        if pts1 is None or len(pts1) < 20:
            return PoseEstimate(np.eye(3), np.array([0, 0, 1]), np.array([0, 0, 0.1]),
                              0, 0, 0, "failed")

        n_matches = len(pts1)

        # Essential Matrix estimation
        E, inlier_mask, n_inliers = self.estimate_essential_robust(pts1, pts2, K)

        if E is None or n_inliers < 10:
            return PoseEstimate(np.eye(3), np.array([0, 0, 1]), np.array([0, 0, 0.1]),
                              n_inliers, n_inliers / n_matches if n_matches > 0 else 0,
                              0, "essential_failed")

        # Filter to inliers
        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]

        # Decompose E and select best via cheirality
        R, t, n_valid = self.decompose_essential(E, pts1_in, pts2_in, K)

        if R is None or n_valid < 5:
            return PoseEstimate(np.eye(3), np.array([0, 0, 1]), np.array([0, 0, 0.1]),
                              n_inliers, n_inliers / n_matches, 0, "decompose_failed")

        # Ensure t is unit vector
        t = t / (np.linalg.norm(t) + 1e-8)

        # Refine with bundle adjustment
        if use_ba and len(pts1_in) >= 20:
            R, t = self.refine_pose_ba(pts1_in, pts2_in, R, t, K)
            t = t / (np.linalg.norm(t) + 1e-8)

        t_scaled = t * 0.1  # Default 10cm when no depth

        # Compute confidence
        inlier_ratio = n_inliers / n_matches
        valid_ratio = n_valid / n_inliers if n_inliers > 0 else 0
        confidence = min(1.0, inlier_ratio * valid_ratio * 2)

        return PoseEstimate(
            R=R,
            t=t,
            t_scaled=t_scaled,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            method="essential+ba" if use_ba else "essential"
        )


def compute_rpe(R_est: np.ndarray, t_est: np.ndarray,
                R_gt: np.ndarray, t_gt: np.ndarray) -> dict:
    """
    Compute Relative Pose Error (RPE) between estimated and ground truth relative poses.

    Returns:
        dict with 'rot_err' (degrees), 'trans_err' (meters), 'trans_err_rel' (ratio)
    """
    # Rotation error: geodesic distance
    R_err = R_gt @ R_est.T
    rot_err = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) * 180 / np.pi

    # Translation error: Euclidean distance
    trans_err = np.linalg.norm(t_est - t_gt)

    # Relative translation error (normalized by GT distance)
    gt_dist = np.linalg.norm(t_gt)
    trans_err_rel = trans_err / gt_dist if gt_dist > 1e-6 else 0

    return {
        'rot_err': rot_err,
        'trans_err': trans_err,
        'trans_err_rel': trans_err_rel,
        'gt_dist': gt_dist
    }


def compute_ate(est_traj: np.ndarray, gt_traj: np.ndarray) -> dict:
    """
    Compute Absolute Trajectory Error (ATE) after Sim(3) alignment.

    Args:
        est_traj: Nx3 array of estimated positions
        gt_traj: Nx3 array of ground truth positions

    Returns:
        dict with 'rmse', 'mean', 'median', 'std', 'scale'
    """
    assert len(est_traj) == len(gt_traj)

    # Umeyama alignment (Sim(3): rotation, translation, scale)
    # Center the point clouds
    est_centered = est_traj - est_traj.mean(axis=0)
    gt_centered = gt_traj - gt_traj.mean(axis=0)

    # Compute scale
    est_var = np.sum(est_centered ** 2)
    gt_var = np.sum(gt_centered ** 2)
    scale = np.sqrt(gt_var / est_var) if est_var > 1e-10 else 1.0

    # Compute rotation using SVD
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = gt_traj.mean(axis=0) - scale * (R @ est_traj.mean(axis=0))

    # Apply alignment
    est_aligned = scale * (est_traj @ R.T) + t

    # Compute errors
    errors = np.linalg.norm(est_aligned - gt_traj, axis=1)

    return {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'scale': scale,
        'aligned_traj': est_aligned
    }


def compute_kitti_metrics(rpe_results: List[dict]) -> dict:
    """
    Compute KITTI-style metrics.

    KITTI uses:
    - Translation error: % of path length
    - Rotation error: deg/m

    Returns:
        dict with 't_err_pct' (trans error %), 'r_err_per_m' (rot error deg/m)
    """
    if not rpe_results:
        return {'t_err_pct': 0, 'r_err_per_m': 0}

    total_dist = sum(r['gt_dist'] for r in rpe_results)
    total_trans_err = sum(r['trans_err'] for r in rpe_results)
    total_rot_err = sum(r['rot_err'] for r in rpe_results)

    # Translation error as percentage of path
    t_err_pct = 100 * total_trans_err / total_dist if total_dist > 0 else 0

    # Rotation error per meter
    r_err_per_m = total_rot_err / total_dist if total_dist > 0 else 0

    return {
        't_err_pct': t_err_pct,
        'r_err_per_m': r_err_per_m,
        'total_dist': total_dist
    }


def evaluate_on_icl_nuim(vo: MonocularVO, data_dir: Path, frame_step: int, num_pairs: int):
    """Evaluate on ICL-NUIM dataset with comprehensive metrics."""
    # Load ground truth
    gt_file = data_dir / "livingRoom2.gt.freiburg"
    gt_poses = {}

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            fid = int(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            gt_poses[fid] = {'t': t, 'q': q, 'R': Rot.from_quat(q).as_matrix()}

    results = []
    rpe_results = []
    frame_pairs = [(i, i + frame_step) for i in range(0, num_pairs * frame_step, frame_step)]

    # Trajectory accumulation
    est_trajectory = [np.zeros(3)]  # Start at origin
    gt_trajectory = []
    est_R_accum = np.eye(3)
    est_t_accum = np.zeros(3)

    print(f"\nEvaluating {len(frame_pairs)} frame pairs (step={frame_step})...")
    print("-" * 70)

    for frame_a, frame_b in frame_pairs:
        img_a = data_dir / "rgb" / f"{frame_a}.png"
        img_b = data_dir / "rgb" / f"{frame_b}.png"

        if not img_a.exists() or not img_b.exists():
            continue

        gt_a = gt_poses.get(frame_a + 1)
        gt_b = gt_poses.get(frame_b + 1)

        if gt_a is None or gt_b is None:
            continue

        # Store GT trajectory points
        if len(gt_trajectory) == 0:
            gt_trajectory.append(gt_a['t'].copy())
        gt_trajectory.append(gt_b['t'].copy())

        # Ground truth relative pose
        R_a, R_b = gt_a['R'], gt_b['R']
        # Relative rotation: transforms vectors from cam_a to cam_b
        R_gt_rel = R_b.T @ R_a
        # Relative translation: position of cam_b relative to cam_a, in cam_a frame
        # Direction from cam_b to cam_a (matches recoverPose convention)
        t_gt_rel = R_a.T @ (gt_a['t'] - gt_b['t'])
        gt_dist = np.linalg.norm(t_gt_rel)

        # Estimate pose
        pose = vo.estimate_pose(str(img_a), str(img_b), focal=481.2)

        if pose.method.endswith("failed"):
            print(f"  {frame_a} → {frame_b}: FAILED ({pose.method})")
            results.append({'success': False, 'frame_a': frame_a, 'frame_b': frame_b})
            # Use identity for failed estimates in trajectory
            est_trajectory.append(est_t_accum.copy())
            continue

        # Scale estimated translation to match GT distance (for fair comparison)
        t_est_scaled = pose.t * gt_dist

        # Accumulate trajectory
        est_t_accum = est_t_accum + est_R_accum @ t_est_scaled
        est_R_accum = est_R_accum @ pose.R.T  # Accumulate rotation
        est_trajectory.append(est_t_accum.copy())

        # Compute RPE
        rpe = compute_rpe(pose.R, t_est_scaled, R_gt_rel, t_gt_rel)
        rpe_results.append(rpe)

        # Direction error (for backward compatibility)
        t_gt_norm = t_gt_rel / (gt_dist + 1e-8)
        if gt_dist > 0.001:
            cos_angle = np.clip(np.dot(t_gt_norm, pose.t), -1, 1)
            dir_err = np.arccos(cos_angle) * 180 / np.pi
        else:
            dir_err = 0

        print(f"  {frame_a:3d} → {frame_b:3d}: rot={rpe['rot_err']:5.1f}°  dir={dir_err:5.1f}°  "
              f"t_err={rpe['trans_err']:.3f}m  inliers={pose.n_inliers}")

        results.append({
            'success': True,
            'frame_a': frame_a, 'frame_b': frame_b,
            'rot_err': rpe['rot_err'],
            'dir_err': dir_err,
            'trans_err': rpe['trans_err'],
            'trans_err_rel': rpe['trans_err_rel'],
            'gt_dist': gt_dist,
            'n_inliers': pose.n_inliers,
            'R_est': pose.R,
            't_est': t_est_scaled
        })

    # Compute ATE if we have trajectory
    ate_results = None
    if len(est_trajectory) > 2 and len(gt_trajectory) > 2:
        est_traj_arr = np.array(est_trajectory[:len(gt_trajectory)])
        gt_traj_arr = np.array(gt_trajectory)
        ate_results = compute_ate(est_traj_arr, gt_traj_arr)

    # Compute KITTI metrics
    kitti_results = compute_kitti_metrics(rpe_results)

    return {
        'frame_results': results,
        'rpe_results': rpe_results,
        'ate_results': ate_results,
        'kitti_results': kitti_results,
        'est_trajectory': np.array(est_trajectory),
        'gt_trajectory': np.array(gt_trajectory) if gt_trajectory else None
    }


def print_results(eval_results: dict):
    """Print comprehensive evaluation summary."""
    results = eval_results['frame_results']
    rpe_results = eval_results['rpe_results']
    ate_results = eval_results['ate_results']
    kitti_results = eval_results['kitti_results']

    successful = [r for r in results if r['success']]
    if not successful:
        print("No successful estimates!")
        return

    rot_errs = np.array([r['rot_err'] for r in successful])
    dir_errs = np.array([r['dir_err'] for r in successful])
    trans_errs = np.array([r['trans_err'] for r in successful])

    print(f"\n{'=' * 70}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")

    # RPE - Relative Pose Error
    print(f"\n--- RPE (Relative Pose Error) ---")
    print(f"Rotation Error:")
    print(f"  Mean: {np.mean(rot_errs):.2f}°  Median: {np.median(rot_errs):.2f}°  Std: {np.std(rot_errs):.2f}°")
    print(f"  < 1°: {100*np.mean(rot_errs < 1):.1f}%  < 2°: {100*np.mean(rot_errs < 2):.1f}%  "
          f"< 5°: {100*np.mean(rot_errs < 5):.1f}%  < 10°: {100*np.mean(rot_errs < 10):.1f}%")

    print(f"Translation Direction Error:")
    print(f"  Mean: {np.mean(dir_errs):.2f}°  Median: {np.median(dir_errs):.2f}°  Std: {np.std(dir_errs):.2f}°")
    print(f"  < 5°: {100*np.mean(dir_errs < 5):.1f}%  < 10°: {100*np.mean(dir_errs < 10):.1f}%  "
          f"< 20°: {100*np.mean(dir_errs < 20):.1f}%")

    print(f"Translation Error (after scale alignment):")
    print(f"  Mean: {np.mean(trans_errs):.4f}m  Median: {np.median(trans_errs):.4f}m  Std: {np.std(trans_errs):.4f}m")

    # KITTI-style metrics
    print(f"\n--- KITTI Metrics ---")
    print(f"Translation Error: {kitti_results['t_err_pct']:.2f}% of path length")
    print(f"Rotation Error: {kitti_results['r_err_per_m']:.4f} deg/m")
    print(f"Total Path Length: {kitti_results['total_dist']:.2f}m")

    # ATE - Absolute Trajectory Error
    if ate_results:
        print(f"\n--- ATE (Absolute Trajectory Error, Sim(3) aligned) ---")
        print(f"RMSE: {ate_results['rmse']:.4f}m")
        print(f"Mean: {ate_results['mean']:.4f}m  Median: {ate_results['median']:.4f}m  Std: {ate_results['std']:.4f}m")
        print(f"Min: {ate_results['min']:.4f}m  Max: {ate_results['max']:.4f}m")
        print(f"Scale factor: {ate_results['scale']:.4f}")

    # Combined accuracy
    print(f"\n--- Combined Accuracy (rot AND dir) ---")
    for r_th in [2, 5, 10]:
        for d_th in [5, 10, 20]:
            acc = 100 * np.mean((rot_errs < r_th) & (dir_errs < d_th))
            print(f"  rot<{r_th}° AND dir<{d_th}°: {acc:.1f}%")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Competition-Grade Monocular VO")
    parser.add_argument("--data-dir", type=str, default="test_images")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--no-ba", action="store_true", help="Disable bundle adjustment")
    args = parser.parse_args()

    print("=" * 70)
    print("Competition-Grade Monocular Visual Odometry")
    print("=" * 70)

    vo = MonocularVO()
    vo.load_models()

    data_dir = Path(args.data_dir)
    results = evaluate_on_icl_nuim(vo, data_dir, args.frame_step, args.num_pairs)
    print_results(results)


if __name__ == "__main__":
    main()
