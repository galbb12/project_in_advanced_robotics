#!/usr/bin/env python3
"""
Visual Odometry Benchmark Script

Evaluates pose estimation on various datasets (KITTI, TUM, ICL-NUIM, UZH-FPV).
Uses the core PoseEstimator from estimate_pose.py.

Metrics computed:
- RPE (Relative Pose Error): rotation and translation per frame
- ATE (Absolute Trajectory Error): after Sim(3) alignment
- KITTI metrics: translation %, rotation deg/m
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from estimate_pose import PoseEstimator


# ============================================================================
# Metric computation functions
# ============================================================================

def compute_rpe(R_est: np.ndarray, t_est: np.ndarray,
                R_gt: np.ndarray, t_gt: np.ndarray) -> dict:
    """Compute Relative Pose Error between estimated and ground truth poses."""
    # Rotation error (geodesic distance)
    R_err = R_gt @ R_est.T
    rot_err = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) * 180 / np.pi

    # Translation direction error
    t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-8)
    t_est_norm = t_est / (np.linalg.norm(t_est) + 1e-8)
    cos_angle = np.clip(np.dot(t_gt_norm, t_est_norm), -1, 1)
    dir_err = np.arccos(cos_angle) * 180 / np.pi

    return {
        'rot_err': rot_err,
        'dir_err': dir_err,
        'gt_dist': np.linalg.norm(t_gt)
    }


def compute_ate(est_traj: np.ndarray, gt_traj: np.ndarray) -> dict:
    """Compute Absolute Trajectory Error after Sim(3) alignment."""
    assert len(est_traj) == len(gt_traj)

    # Umeyama alignment
    est_centered = est_traj - est_traj.mean(axis=0)
    gt_centered = gt_traj - gt_traj.mean(axis=0)

    est_var = np.sum(est_centered ** 2)
    gt_var = np.sum(gt_centered ** 2)
    scale = np.sqrt(gt_var / est_var) if est_var > 1e-10 else 1.0

    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = gt_traj.mean(axis=0) - scale * (R @ est_traj.mean(axis=0))
    est_aligned = scale * (est_traj @ R.T) + t
    errors = np.linalg.norm(est_aligned - gt_traj, axis=1)

    return {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'scale': scale
    }


def compute_kitti_metrics(rpe_results: List[dict]) -> dict:
    """Compute KITTI-style metrics (translation %, rotation deg/m)."""
    if not rpe_results:
        return {'t_err_pct': 0, 'r_err_per_m': 0, 'total_dist': 0}

    total_dist = sum(r['gt_dist'] for r in rpe_results)
    total_rot_err = sum(r['rot_err'] for r in rpe_results)
    total_dir_err = sum(r['dir_err'] for r in rpe_results)

    return {
        'r_err_per_m': total_rot_err / total_dist if total_dist > 0 else 0,
        'dir_err_mean': total_dir_err / len(rpe_results),
        'total_dist': total_dist
    }


# ============================================================================
# Dataset loaders
# ============================================================================

def load_kitti(data_dir: Path, sequence: str) -> Tuple[List[str], List[dict], np.ndarray]:
    """Load KITTI sequence. Returns (image_paths, poses, K)."""
    seq_dir = data_dir / "sequences" / sequence
    pose_file = data_dir / "poses" / f"{sequence}.txt"
    calib_file = seq_dir / "calib.txt"

    # Load calibration
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array([float(x) for x in value.strip().split()])

    P_key = 'P2' if 'P2' in calib else 'P0'
    P = calib[P_key].reshape(3, 4)
    K = P[:3, :3]

    # Load poses
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            T = np.array(values).reshape(3, 4)
            poses.append({'R': T[:3, :3], 't': T[:3, 3]})

    # Get image paths
    img_dir = seq_dir / "image_2" if (seq_dir / "image_2").exists() else seq_dir / "image_0"
    image_paths = sorted(img_dir.glob("*.png"))

    return [str(p) for p in image_paths], poses, K, None, False


def load_tum(data_dir: Path) -> Tuple[List[str], List[dict], np.ndarray, None, bool]:
    """Load TUM RGB-D sequence. Returns (image_paths, poses, K, dist_coeffs, fisheye)."""
    # TUM freiburg1 camera intrinsics
    K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], dtype=np.float64)

    # Load groundtruth
    gt_file = data_dir / "groundtruth.txt"
    gt_data = {}
    with open(gt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            ts = float(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            gt_data[ts] = {'t': t, 'R': Rot.from_quat(q).as_matrix()}

    # Load rgb.txt for image timestamps
    rgb_file = data_dir / "rgb.txt"
    images = []
    with open(rgb_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            ts = float(parts[0])
            img_path = data_dir / parts[1]
            images.append((ts, str(img_path)))

    # Associate images with groundtruth (nearest timestamp)
    gt_times = sorted(gt_data.keys())
    image_paths = []
    poses = []

    for ts, img_path in images:
        # Find nearest groundtruth
        idx = np.searchsorted(gt_times, ts)
        if idx == 0:
            nearest_ts = gt_times[0]
        elif idx == len(gt_times):
            nearest_ts = gt_times[-1]
        else:
            if abs(gt_times[idx] - ts) < abs(gt_times[idx-1] - ts):
                nearest_ts = gt_times[idx]
            else:
                nearest_ts = gt_times[idx-1]

        if abs(nearest_ts - ts) < 0.1:  # Within 100ms
            image_paths.append(img_path)
            poses.append(gt_data[nearest_ts])

    return image_paths, poses, K, None, False


def load_icl_nuim(data_dir: Path) -> Tuple[List[str], List[dict], np.ndarray, None, bool]:
    """Load ICL-NUIM sequence. Returns (image_paths, poses, K, dist_coeffs, fisheye)."""
    K = np.array([[481.2, 0, 319.5], [0, 481.2, 239.5], [0, 0, 1]], dtype=np.float64)

    gt_file = data_dir / "livingRoom2.gt.freiburg"
    gt_data = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            fid = int(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            gt_data[fid] = {'t': t, 'R': Rot.from_quat(q).as_matrix()}

    rgb_dir = data_dir / "rgb"
    image_paths = []
    poses = []

    for i in sorted(gt_data.keys()):
        img_path = rgb_dir / f"{i-1}.png"
        if img_path.exists():
            image_paths.append(str(img_path))
            poses.append(gt_data[i])

    return image_paths, poses, K, None, False


def load_uzh(data_dir: Path) -> Tuple[List[str], List[dict], np.ndarray, np.ndarray, bool]:
    """Load UZH-FPV dataset. Returns (image_paths, poses, K, dist_coeffs, is_fisheye)."""
    # DAVIS 346 camera intrinsics (from indoor_forward_calib_davis)
    K = np.array([[172.98992850734132, 0, 163.33639726024606],
                  [0, 172.98303181090185, 134.99537889030861],
                  [0, 0, 1]], dtype=np.float64)
    # Equidistant (fisheye) distortion coefficients
    dist_coeffs = np.array([-0.027576733308582076, -0.006593578674675004,
                            0.0008566938165177085, -0.00030899587045247486])

    # Load ground truth poses (timestamp tx ty tz qx qy qz qw)
    gt_file = data_dir / "groundtruth.txt"
    gt_data = []
    with open(gt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            ts = float(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
            gt_data.append({'ts': ts, 't': t, 'R': Rot.from_quat(q).as_matrix()})

    gt_times = np.array([g['ts'] for g in gt_data])

    # Load image list (id timestamp image_name)
    images_file = data_dir / "images.txt"
    images = []
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            ts = float(parts[1])
            img_path = data_dir / parts[2]
            images.append((ts, str(img_path)))

    # Associate images with ground truth (interpolate for better accuracy)
    image_paths = []
    poses = []
    gt_start, gt_end = gt_times[0], gt_times[-1]

    for ts, img_path in images:
        # Skip images outside GT time range
        if ts < gt_start or ts > gt_end:
            continue

        idx = np.searchsorted(gt_times, ts)
        if idx == 0 or idx >= len(gt_times):
            continue

        # Linear interpolation
        t0, t1 = gt_times[idx - 1], gt_times[idx]
        alpha = (ts - t0) / (t1 - t0 + 1e-10)

        # Interpolate translation
        t_interp = (1 - alpha) * gt_data[idx - 1]['t'] + alpha * gt_data[idx]['t']

        # Interpolate rotation using SLERP
        R0, R1 = gt_data[idx - 1]['R'], gt_data[idx]['R']
        r0, r1 = Rot.from_matrix(R0), Rot.from_matrix(R1)
        slerp = Rot.from_rotvec((1 - alpha) * r0.as_rotvec() + alpha * r1.as_rotvec())
        R_interp = slerp.as_matrix()

        if Path(img_path).exists():
            image_paths.append(img_path)
            poses.append({'t': t_interp, 'R': R_interp})

    return image_paths, poses, K, dist_coeffs, True  # True = fisheye


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(estimator: PoseEstimator, image_paths: List[str], poses: List[dict],
             K: np.ndarray, frame_step: int, num_pairs: int,
             dist_coeffs: np.ndarray = None, fisheye: bool = False) -> dict:
    """Run evaluation on a sequence."""
    results = []
    rpe_results = []

    max_pairs = min(num_pairs, (len(image_paths) - 1) // frame_step)
    frame_pairs = [(i, i + frame_step) for i in range(0, max_pairs * frame_step, frame_step)]

    # Trajectory accumulation
    est_trajectory = [np.zeros(3)]
    gt_trajectory = []
    est_R_accum = np.eye(3)
    est_t_accum = np.zeros(3)

    print(f"\nEvaluating {len(frame_pairs)} frame pairs (step={frame_step})...")
    print("-" * 70)

    for frame_a, frame_b in frame_pairs:
        if frame_b >= len(poses):
            continue

        gt_a, gt_b = poses[frame_a], poses[frame_b]

        if len(gt_trajectory) == 0:
            gt_trajectory.append(gt_a['t'].copy())
        gt_trajectory.append(gt_b['t'].copy())

        # Ground truth relative pose
        R_a, R_b = gt_a['R'], gt_b['R']
        R_gt_rel = R_b.T @ R_a
        t_gt_rel = R_a.T @ (gt_b['t'] - gt_a['t'])  # Direction from a to b
        gt_dist = np.linalg.norm(gt_a['t'] - gt_b['t'])

        # Skip pairs with insufficient motion for reliable direction estimation
        MIN_MOTION_M = 0.02  # 2cm minimum
        if gt_dist < MIN_MOTION_M:
            print(f"  {frame_a:4d} → {frame_b:4d}: SKIP (motion={gt_dist*100:.1f}cm < {MIN_MOTION_M*100:.0f}cm)")
            est_trajectory.append(est_t_accum.copy())
            continue

        # Estimate pose
        result = estimator.estimate(image_paths[frame_a], image_paths[frame_b], K,
                                      dist_coeffs=dist_coeffs, fisheye=fisheye)

        if not result['success']:
            print(f"  {frame_a:4d} → {frame_b:4d}: FAILED ({result['error']})")
            results.append({'success': False, 'frame_a': frame_a, 'frame_b': frame_b})
            est_trajectory.append(est_t_accum.copy())
            continue

        R_est, t_est = result['R'], result['t']

        # Accumulate trajectory (scale to GT distance for fair comparison)
        t_scaled = t_est * gt_dist
        est_t_accum = est_t_accum + est_R_accum @ t_scaled
        est_R_accum = est_R_accum @ R_est.T
        est_trajectory.append(est_t_accum.copy())

        # Compute RPE
        rpe = compute_rpe(R_est, t_est, R_gt_rel, t_gt_rel / (gt_dist + 1e-8))
        rpe['gt_dist'] = gt_dist
        rpe_results.append(rpe)

        print(f"  {frame_a:4d} → {frame_b:4d}: rot={rpe['rot_err']:5.1f}°  "
              f"dir={rpe['dir_err']:5.1f}°  inliers={result['n_inliers']}")

        results.append({
            'success': True,
            'frame_a': frame_a, 'frame_b': frame_b,
            'rot_err': rpe['rot_err'],
            'dir_err': rpe['dir_err'],
            'gt_dist': gt_dist,
            'n_inliers': result['n_inliers']
        })

    # Compute ATE
    ate_results = None
    if len(est_trajectory) > 2 and len(gt_trajectory) > 2:
        est_traj_arr = np.array(est_trajectory[:len(gt_trajectory)])
        gt_traj_arr = np.array(gt_trajectory)
        ate_results = compute_ate(est_traj_arr, gt_traj_arr)

    kitti_results = compute_kitti_metrics(rpe_results)

    return {
        'frame_results': results,
        'rpe_results': rpe_results,
        'ate_results': ate_results,
        'kitti_results': kitti_results
    }


def print_results(eval_results: dict):
    """Print evaluation summary."""
    results = eval_results['frame_results']
    ate_results = eval_results['ate_results']
    kitti_results = eval_results['kitti_results']

    successful = [r for r in results if r['success']]
    if not successful:
        print("No successful estimates!")
        return

    rot_errs = np.array([r['rot_err'] for r in successful])
    dir_errs = np.array([r['dir_err'] for r in successful])

    print(f"\n{'=' * 70}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")

    print(f"\n--- RPE (Relative Pose Error) ---")
    print(f"Rotation Error:")
    print(f"  Mean: {np.mean(rot_errs):.2f}°  Median: {np.median(rot_errs):.2f}°")
    print(f"  < 1°: {100*np.mean(rot_errs < 1):.1f}%  < 2°: {100*np.mean(rot_errs < 2):.1f}%  "
          f"< 5°: {100*np.mean(rot_errs < 5):.1f}%  < 10°: {100*np.mean(rot_errs < 10):.1f}%")

    print(f"Translation Direction Error:")
    print(f"  Mean: {np.mean(dir_errs):.2f}°  Median: {np.median(dir_errs):.2f}°")
    print(f"  < 5°: {100*np.mean(dir_errs < 5):.1f}%  < 10°: {100*np.mean(dir_errs < 10):.1f}%  "
          f"< 20°: {100*np.mean(dir_errs < 20):.1f}%")

    print(f"\n--- KITTI Metrics ---")
    print(f"Rotation Error: {kitti_results['r_err_per_m']:.4f} deg/m")
    print(f"Direction Error: {kitti_results['dir_err_mean']:.2f}° mean")
    print(f"Total Path Length: {kitti_results['total_dist']:.2f}m")

    if ate_results:
        print(f"\n--- ATE (Sim(3) aligned) ---")
        print(f"RMSE: {ate_results['rmse']:.4f}m  Mean: {ate_results['mean']:.4f}m")
        print(f"Scale factor: {ate_results['scale']:.4f}")

    print(f"\n--- Combined Accuracy ---")
    for r_th in [2, 5, 10]:
        for d_th in [5, 10, 20]:
            acc = 100 * np.mean((rot_errs < r_th) & (dir_errs < d_th))
            print(f"  rot<{r_th}° AND dir<{d_th}°: {acc:.1f}%")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="VO Benchmark")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "kitti", "tum", "icl", "uzh"],
                        help="Dataset type")
    parser.add_argument("--sequence", type=str, default="00", help="KITTI sequence")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--num-pairs", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Auto-detect dataset
    dataset_type = args.dataset
    if dataset_type == "auto":
        if (data_dir / "sequences").exists():
            dataset_type = "kitti"
        elif (data_dir / "images.txt").exists() and (data_dir / "img").exists():
            dataset_type = "uzh"
        elif (data_dir / "groundtruth.txt").exists():
            dataset_type = "tum"
        elif (data_dir / "livingRoom2.gt.freiburg").exists():
            dataset_type = "icl"
        else:
            print("Could not detect dataset type. Use --dataset flag.")
            return

    print("=" * 70)
    print(f"VO Benchmark - {dataset_type.upper()}")
    print("=" * 70)

    # Load dataset
    if dataset_type == "kitti":
        image_paths, poses, K, dist_coeffs, fisheye = load_kitti(data_dir, args.sequence)
    elif dataset_type == "tum":
        image_paths, poses, K, dist_coeffs, fisheye = load_tum(data_dir)
    elif dataset_type == "uzh":
        image_paths, poses, K, dist_coeffs, fisheye = load_uzh(data_dir)
    else:
        image_paths, poses, K, dist_coeffs, fisheye = load_icl_nuim(data_dir)

    print(f"Loaded {len(image_paths)} images, {len(poses)} poses")
    print(f"K:\n{K}")
    if dist_coeffs is not None:
        print(f"Distortion: {dist_coeffs} (fisheye={fisheye})")

    # Initialize estimator
    estimator = PoseEstimator()
    estimator.load_models()

    # Run evaluation
    results = evaluate(estimator, image_paths, poses, K, args.frame_step, args.num_pairs,
                       dist_coeffs=dist_coeffs, fisheye=fisheye)
    print_results(results)


if __name__ == "__main__":
    main()
