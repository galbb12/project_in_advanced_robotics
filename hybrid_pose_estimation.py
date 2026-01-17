#!/usr/bin/env python3
"""
Hybrid 6DoF Pose Estimation using Essential Matrix + Depth Scale

This approach:
1. Uses Essential Matrix decomposition for rotation and translation direction
2. Uses Depth Pro only to recover the metric scale of translation
3. This avoids depth scale inconsistency issues with pure 3D alignment

The Essential Matrix approach is the gold standard for relative pose estimation
in visual odometry.
"""

import sys
import argparse
from pathlib import Path
import warnings

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import cv2

warnings.filterwarnings("ignore")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HybridPoseEstimator:
    def __init__(self, device):
        self.device = device
        self.depth_model = None
        self.depth_transform = None
        self.extractor = None
        self.matcher = None
        self.depth_cache = {}
        self.focal_cache = {}

    def load_models(self):
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
        self.extractor = SuperPoint(max_num_keypoints=4096).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

    def get_depth_and_focal(self, image_path):
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
        H, W = depth_map.shape
        half = patch_size // 2
        u_int, v_int = int(round(u)), int(round(v))
        u_min, u_max = max(0, u_int - half), min(W, u_int + half + 1)
        v_min, v_max = max(0, v_int - half), min(H, v_int + half + 1)
        patch = depth_map[v_min:v_max, u_min:u_max]
        valid = patch[(patch > 0) & ~np.isnan(patch) & ~np.isinf(patch)]
        return np.median(valid) if len(valid) > 0 else None

    def match_features(self, img1_path, img2_path):
        from lightglue.utils import load_image

        image1 = load_image(img1_path).to(self.device)
        image2 = load_image(img2_path).to(self.device)

        with torch.no_grad():
            feats1 = self.extractor.extract(image1)
            feats2 = self.extractor.extract(image2)
            matches_dict = self.matcher({"image0": feats1, "image1": feats2})

        matches = matches_dict["matches"][0].cpu().numpy()
        scores = matches_dict.get("matching_scores")
        if scores is not None:
            scores = scores[0].cpu().numpy()

        valid_mask = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
        valid_matches = matches[valid_mask]

        if len(valid_matches) == 0:
            return None, None, None

        kpts1 = feats1["keypoints"][0].cpu().numpy()
        kpts2 = feats2["keypoints"][0].cpu().numpy()

        matched_kpts1 = kpts1[valid_matches[:, 0]]
        matched_kpts2 = kpts2[valid_matches[:, 1]]

        valid_scores = None
        if scores is not None and len(scores) == len(matches):
            valid_scores = scores[valid_mask]

        return matched_kpts1, matched_kpts2, valid_scores

    def estimate_pose(self, img1_path, img2_path, known_focal=None):
        """
        Estimate relative pose using Essential Matrix + depth scale recovery.
        """
       # 1. Get Data
        depth1, focal1 = self.get_depth_and_focal(img1_path)
        
        # OPTIONAL: Use average focal length if available, otherwise use f1
        focal = known_focal if known_focal else focal1
        
        H, W = depth1.shape
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

        # 2. Get Matches
        kpts1, kpts2, scores = self.match_features(str(img1_path), str(img2_path))
        if kpts1 is None or len(kpts1) < 10:
            return None, None, 0, {}

        # 3. Lift Frame 1 Keypoints to 3D (Object Points)
        # We only need depth for the points in Image 1!
        object_points = []
        valid_indices = []
        
        for i, (u, v) in enumerate(kpts1):
            z = self.get_depth_at_point(depth1, u, v)
            if z is not None and 0.1 < z < 100: # Sanity check depth
                x = (u - cx) * z / focal
                y = (v - cy) * z / focal
                object_points.append([x, y, z])
                valid_indices.append(i)

        object_points = np.array(object_points, dtype=np.float64)
        image_points = kpts2[valid_indices].astype(np.float64) # Frame 2 2D points

        if len(object_points) < 6: # PnP needs at least 4-6 points
            return None, None, 0, {}

        # 4. Solve PnP (3D in Frame A -> 2D in Frame B)
        # Result is R, t that transforms Frame A points to Frame B
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            distCoeffs=None,
            iterationsCount=5000,
            reprojectionError=1.5, # tighter threshold
            confidence=0.9999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, 0, {}

        # Convert rvec to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        # 5. Coordinate System Convention
        # solvePnP gives R, t such that: P_cam2 = R @ P_cam1 + t
        # This means t is expressed in camera 2's frame.
        # To get camera 2's position in camera 1's frame (what GT uses):
        # t_in_cam1_frame = -R.T @ t
        t = -R.T @ t

        metadata = {
            'n_matches': len(kpts1),
            'n_inliers_pnp': len(inliers) if inliers is not None else 0,
            'focal': focal
        }
        
        return R, t, len(inliers), metadata


def evaluate_on_icl_nuim(estimator, data_dir, frame_pairs):
    """Evaluate against ground truth."""
    gt_file = data_dir / "livingRoom2.gt.freiburg"
    gt_poses = {}

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            gt_poses[frame_id] = {'t': np.array([tx, ty, tz]), 'q': np.array([qx, qy, qz, qw])}

    results = []

    for frame_a, frame_b in frame_pairs:
        img_a = data_dir / "rgb" / f"{frame_a}.png"
        img_b = data_dir / "rgb" / f"{frame_b}.png"

        if not img_a.exists() or not img_b.exists():
            continue

        gt_a = gt_poses.get(frame_a + 1)
        gt_b = gt_poses.get(frame_b + 1)

        if gt_a is None or gt_b is None:
            continue

        # Compute GT relative transform
        R_a = Rot.from_quat(gt_a['q']).as_matrix()
        R_b = Rot.from_quat(gt_b['q']).as_matrix()
        t_a, t_b = gt_a['t'], gt_b['t']

        R_gt_rel = R_a.T @ R_b
        t_gt_rel = R_a.T @ (t_b - t_a)

        gt_dist = np.linalg.norm(t_gt_rel)

        print(f"Processing {frame_a} → {frame_b} (GT: {gt_dist*100:.2f}cm)...", end=" ")

        # ICL-NUIM known intrinsics: fx=fy=481.20, cx=319.5, cy=239.5
        R_est, t_est, n_inliers, metadata = estimator.estimate_pose(img_a, img_b, known_focal=481.2)

        if R_est is None:
            print("FAILED")
            results.append({'frame_a': frame_a, 'frame_b': frame_b, 'success': False})
            continue

        est_dist = np.linalg.norm(t_est)

        # Rotation error
        R_err = R_gt_rel @ R_est.T
        rot_error = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) * 180 / np.pi

        # Translation direction error
        if gt_dist > 0.001 and est_dist > 0.001:
            cos_angle = np.clip(np.dot(t_gt_rel / gt_dist, t_est / est_dist), -1, 1)
            trans_dir_error = np.arccos(cos_angle) * 180 / np.pi
        else:
            trans_dir_error = 0 if gt_dist < 0.001 else 180

        # Scale error
        scale_error = abs(est_dist - gt_dist) / gt_dist if gt_dist > 0.001 else 0

        print(f"rot={rot_error:.1f}°, dir={trans_dir_error:.1f}°, "
              f"scale={scale_error*100:.0f}% (est:{est_dist*100:.1f}cm)")

        results.append({
            'frame_a': frame_a, 'frame_b': frame_b, 'success': True,
            'rot_error': rot_error, 'trans_dir_error': trans_dir_error,
            'scale_error': scale_error, 'gt_dist': gt_dist, 'est_dist': est_dist,
            'n_inliers': n_inliers
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="test_images")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--num-pairs", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("Hybrid Pose Estimation (Essential Matrix + Depth Scale)")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    estimator = HybridPoseEstimator(device)
    estimator.load_models()

    frame_pairs = [(i, i + args.frame_step)
                   for i in range(0, args.num_pairs * args.frame_step, args.frame_step)]

    print(f"\nEvaluating {len(frame_pairs)} frame pairs...")
    print("-" * 70)

    results = evaluate_on_icl_nuim(estimator, data_dir, frame_pairs)

    successful = [r for r in results if r['success']]
    if not successful:
        print("No successful estimations!")
        return

    rot_errors = [r['rot_error'] for r in successful]
    trans_dir_errors = [r['trans_dir_error'] for r in successful]
    scale_errors = [r['scale_error'] for r in successful]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Success rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")

    print(f"\nRotation Error:")
    print(f"  Mean: {np.mean(rot_errors):.2f}°, Median: {np.median(rot_errors):.2f}°")
    print(f"  < 5°: {100*np.mean(np.array(rot_errors) < 5):.1f}%")
    print(f"  < 10°: {100*np.mean(np.array(rot_errors) < 10):.1f}%")

    print(f"\nTranslation Direction Error:")
    print(f"  Mean: {np.mean(trans_dir_errors):.2f}°, Median: {np.median(trans_dir_errors):.2f}°")
    print(f"  < 10°: {100*np.mean(np.array(trans_dir_errors) < 10):.1f}%")
    print(f"  < 20°: {100*np.mean(np.array(trans_dir_errors) < 20):.1f}%")
    print(f"  < 30°: {100*np.mean(np.array(trans_dir_errors) < 30):.1f}%")

    print(f"\nScale Error:")
    print(f"  Mean: {100*np.mean(scale_errors):.1f}%")
    print(f"  < 50%: {100*np.mean(np.array(scale_errors) < 0.5):.1f}%")

    # Overall accuracy with different thresholds
    acc_strict = [(r['rot_error'] < 10 and r['trans_dir_error'] < 20) for r in successful]
    acc_medium = [(r['rot_error'] < 12 and r['trans_dir_error'] < 25) for r in successful]
    acc_relaxed = [(r['rot_error'] < 15 and r['trans_dir_error'] < 25) for r in successful]

    print(f"\n{'=' * 70}")
    print("OVERALL ACCURACY:")
    print(f"  Strict   (rot<10° AND dir<20°): {100*np.mean(acc_strict):.1f}%")
    print(f"  Medium   (rot<12° AND dir<25°): {100*np.mean(acc_medium):.1f}%")
    print(f"  Standard (rot<15° AND dir<25°): {100*np.mean(acc_relaxed):.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
