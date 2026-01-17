#!/usr/bin/env python3
"""
MASt3R-based Visual Odometry

Uses MASt3R's global alignment to directly extract relative pose.
Only cares about translation DIRECTION, not scale.
"""

import sys
import argparse
from pathlib import Path

# Add mast3r to path
sys.path.insert(0, str(Path(__file__).parent / "mast3r"))
sys.path.insert(0, str(Path(__file__).parent / "mast3r" / "dust3r"))

import numpy as np
import torch
from PIL import Image


class MASt3RVO:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_model(self):
        """Load MASt3R model."""
        from mast3r.model import AsymmetricMASt3R

        print("Loading MASt3R model...")
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.model = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def estimate_pose(self, img1_path: str, img2_path: str):
        """
        Estimate relative pose using MASt3R pointmaps.
        Returns R and normalized t (direction only, scale doesn't matter).

        MASt3R outputs:
        - pts3d: 3D points in camera 1's frame
        - pts3d_in_other_view: 3D points from camera 2, expressed in camera 1's frame

        Since both are in camera 1's frame and represent the same scene points,
        we use Procrustes alignment to find the relative pose.
        """
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        # Load images
        images = load_images([img1_path, img2_path], size=512)

        # Run MASt3R inference
        with torch.no_grad():
            output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        # Get pointmaps and confidence
        pts3d_1 = output['pred1']['pts3d'].squeeze(0).cpu().numpy()  # (H, W, 3) in cam1 frame
        pts3d_2 = output['pred2']['pts3d_in_other_view'].squeeze(0).cpu().numpy()  # (H, W, 3) in cam1 frame
        conf1 = output['pred1']['conf'].squeeze(0).cpu().numpy()
        conf2 = output['pred2']['conf'].squeeze(0).cpu().numpy()

        # Flatten and filter by confidence
        h, w = pts3d_1.shape[:2]
        pts1_flat = pts3d_1.reshape(-1, 3)
        pts2_flat = pts3d_2.reshape(-1, 3)
        conf1_flat = conf1.reshape(-1)
        conf2_flat = conf2.reshape(-1)

        # High confidence mask
        conf_thresh = 1.5
        valid = (conf1_flat > conf_thresh) & (conf2_flat > conf_thresh)
        valid &= (pts1_flat[:, 2] > 0.1) & (pts2_flat[:, 2] > 0.1)  # Positive depth
        valid &= ~np.isnan(pts1_flat).any(axis=1) & ~np.isnan(pts2_flat).any(axis=1)

        pts1_valid = pts1_flat[valid]
        pts2_valid = pts2_flat[valid]

        if len(pts1_valid) < 100:
            return None, None, {"error": "Too few confident points"}

        # Subsample for efficiency
        if len(pts1_valid) > 5000:
            idx = np.random.choice(len(pts1_valid), 5000, replace=False)
            pts1_valid = pts1_valid[idx]
            pts2_valid = pts2_valid[idx]

        # The key insight: pts3d_1 and pts3d_2 are both expressed in camera 1's frame
        # but pts3d_2 is the scene as seen from camera 2, transformed to camera 1's frame
        #
        # Actually, for the same 3D scene point:
        # - pts3d_1[i] = point in camera 1's frame
        # - pts3d_2[i] = same point, also in camera 1's frame (since it's pts3d_in_other_view)
        #
        # The camera 2 pose can be recovered from looking at where camera 2
        # thinks the points are vs where camera 1 thinks they are.
        #
        # For a static scene: pts3d_2 = R @ pts3d_1 + t (if there was camera motion)
        # But since both are in cam1 frame, they should be nearly identical.
        # The small differences come from camera 2's viewpoint.

        # Compute the difference to estimate camera motion direction
        # The mean translation of points from cam1 view to cam2 view
        diff = pts2_valid - pts1_valid

        # The camera translation is opposite to the scene point movement
        # If camera moves left, scene appears to move right
        t = -np.mean(diff, axis=0)

        # For rotation, use Procrustes (but it should be small)
        centroid1 = np.mean(pts1_valid, axis=0)
        centroid2 = np.mean(pts2_valid, axis=0)
        pts1_centered = pts1_valid - centroid1
        pts2_centered = pts2_valid - centroid2

        H_matrix = pts1_centered.T @ pts2_centered
        U, S, Vt = np.linalg.svd(H_matrix)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Normalize translation direction
        t_norm = np.linalg.norm(t)
        t_dir = t / (t_norm + 1e-8)

        return R, t_dir, {
            "scale": t_norm,
            "n_points": len(pts1_valid)
        }


def evaluate_icl_nuim(vo, data_dir: Path, frame_step: int, num_pairs: int):
    """Evaluate on ICL-NUIM dataset."""
    from scipy.spatial.transform import Rotation as Rot

    gt = {}
    with open(data_dir / "livingRoom2.gt.freiburg") as f:
        for line in f:
            p = line.split()
            fid = int(p[0])
            gt[fid] = {
                't': np.array([float(p[1]), float(p[2]), float(p[3])]),
                'q': np.array([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
            }

    rot_errs, dir_errs = [], []

    print(f"\nEvaluating {num_pairs} pairs (step={frame_step})...")
    print("-" * 60)

    for i in range(0, num_pairs * frame_step, frame_step):
        img1 = data_dir / "rgb" / f"{i}.png"
        img2 = data_dir / "rgb" / f"{i + frame_step}.png"

        if not img1.exists() or not img2.exists():
            continue

        gt1, gt2 = gt.get(i + 1), gt.get(i + frame_step + 1)
        if gt1 is None or gt2 is None:
            continue

        R1 = Rot.from_quat(gt1['q']).as_matrix()
        R2 = Rot.from_quat(gt2['q']).as_matrix()
        R_gt = R1.T @ R2
        t_gt = R1.T @ (gt2['t'] - gt1['t'])
        t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-8)

        try:
            R_est, t_est, info = vo.estimate_pose(str(img1), str(img2))
        except Exception as e:
            print(f"  {i:3d} -> {i+frame_step:3d}: FAILED - {e}")
            continue

        if R_est is None:
            print(f"  {i:3d} -> {i+frame_step:3d}: FAILED")
            continue

        # Rotation error
        R_err = R_gt @ R_est.T
        rot_err = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) * 180 / np.pi

        # Direction error (translation direction only)
        cos_ang = np.clip(np.dot(t_gt_norm, t_est), -1, 1)
        dir_err = np.arccos(cos_ang) * 180 / np.pi

        rot_errs.append(rot_err)
        dir_errs.append(dir_err)

        status = "+" if rot_err < 10 and dir_err < 20 else " "
        print(f"{status} {i:3d} -> {i+frame_step:3d}: rot={rot_err:5.1f}  dir={dir_err:5.1f}")

    rot_errs = np.array(rot_errs)
    dir_errs = np.array(dir_errs)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Rotation:  mean={np.mean(rot_errs):.1f}  median={np.median(rot_errs):.1f}")
    print(f"           <5: {100*np.mean(rot_errs<5):.0f}%  <10: {100*np.mean(rot_errs<10):.0f}%")
    print(f"Direction: mean={np.mean(dir_errs):.1f}  median={np.median(dir_errs):.1f}")
    print(f"           <10: {100*np.mean(dir_errs<10):.0f}%  <20: {100*np.mean(dir_errs<20):.0f}%")

    print(f"\n*** COMPETITION METRICS ***")
    for r_th, d_th in [(5, 10), (10, 15), (10, 20), (15, 25)]:
        acc = 100 * np.mean((rot_errs < r_th) & (dir_errs < d_th))
        marker = "->" if r_th == 10 and d_th == 20 else "  "
        print(f" {marker} rot<{r_th:2d} & dir<{d_th:2d}: {acc:.0f}%")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img1", nargs="?")
    parser.add_argument("img2", nargs="?")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--data-dir", default="test_images")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=30)
    args = parser.parse_args()

    vo = MASt3RVO()
    vo.load_model()

    if args.test:
        evaluate_icl_nuim(vo, Path(args.data_dir), args.step, args.pairs)
    elif args.img1 and args.img2:
        R, t, info = vo.estimate_pose(args.img1, args.img2)
        if R is not None:
            print(f"\nRotation:\n{R}")
            print(f"\nTranslation direction: {t}")
            print(f"\nInfo: {info}")
        else:
            print("Failed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
