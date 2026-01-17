#!/usr/bin/env python3
"""
Competition-Ready Visual Odometry with PnP + Depth Pro

Uses Depth Pro for depth estimation and PnP for pose recovery.
This avoids the sign ambiguity issues with Essential Matrix decomposition,
especially for small-baseline degenerate cases.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


class CompetitionVO:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.extractor = None
        self.matcher = None
        self.depth_model = None
        self.depth_transform = None
        self.depth_cache = {}
        self.focal_cache = {}

    def load_models(self):
        """Load all models."""
        from lightglue import LightGlue, SuperPoint
        import depth_pro
        from huggingface_hub import hf_hub_download

        print("Loading SuperPoint + LightGlue...")
        self.extractor = SuperPoint(max_num_keypoints=8192).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)

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

    def get_depth_and_focal(self, image_path):
        """Get depth map and focal length from Depth Pro."""
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
        """Get robust depth at a point using median of local patch."""
        H, W = depth_map.shape
        half = patch_size // 2
        u_int, v_int = int(round(u)), int(round(v))
        u_min, u_max = max(0, u_int - half), min(W, u_int + half + 1)
        v_min, v_max = max(0, v_int - half), min(H, v_int + half + 1)
        patch = depth_map[v_min:v_max, u_min:u_max]
        valid = patch[(patch > 0) & ~np.isnan(patch) & ~np.isinf(patch)]
        return np.median(valid) if len(valid) > 0 else None

    def match_features(self, img1_path, img2_path):
        """Match features using SuperPoint + LightGlue."""
        from lightglue.utils import load_image

        image1 = load_image(img1_path).to(self.device)
        image2 = load_image(img2_path).to(self.device)

        with torch.no_grad():
            feats1 = self.extractor.extract(image1)
            feats2 = self.extractor.extract(image2)
            matches_dict = self.matcher({"image0": feats1, "image1": feats2})

        matches = matches_dict["matches"][0].cpu().numpy()
        valid_mask = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
        valid_matches = matches[valid_mask]

        if len(valid_matches) == 0:
            return None, None

        kpts1 = feats1["keypoints"][0].cpu().numpy()
        kpts2 = feats2["keypoints"][0].cpu().numpy()

        matched_kpts1 = kpts1[valid_matches[:, 0]]
        matched_kpts2 = kpts2[valid_matches[:, 1]]

        return matched_kpts1, matched_kpts2

    def estimate_pose(self, img1_path: str, img2_path: str, known_focal: float = None):
        """
        Estimate R and t using PnP with depth.
        Returns normalized translation direction.
        """
        # Get depth from both frames
        depth1, focal1 = self.get_depth_and_focal(img1_path)
        depth2, focal2 = self.get_depth_and_focal(img2_path)

        focal = known_focal if known_focal else focal1

        H, W = depth1.shape
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

        # Get feature matches
        kpts1, kpts2 = self.match_features(str(img1_path), str(img2_path))
        if kpts1 is None or len(kpts1) < 10:
            return None, None, {"error": "Too few matches"}

        # Lift Frame 1 keypoints to 3D using depth
        # Also get depth at corresponding points in frame 2 for consistency check
        object_points = []
        image_points_list = []
        depth_ratios = []

        for i, ((u1, v1), (u2, v2)) in enumerate(zip(kpts1, kpts2)):
            z1 = self.get_depth_at_point(depth1, u1, v1)
            z2 = self.get_depth_at_point(depth2, u2, v2)

            if z1 is not None and z2 is not None and 0.1 < z1 < 100 and 0.1 < z2 < 100:
                x = (u1 - cx) * z1 / focal
                y = (v1 - cy) * z1 / focal
                object_points.append([x, y, z1])
                image_points_list.append([u2, v2])
                depth_ratios.append(z2 / z1)

        if len(object_points) < 6:
            return None, None, {"error": "Too few 3D points"}

        # Filter outliers based on depth ratio consistency
        depth_ratios = np.array(depth_ratios)
        median_ratio = np.median(depth_ratios)
        ratio_dev = np.abs(depth_ratios - median_ratio)
        ratio_thresh = np.percentile(ratio_dev, 85)  # Keep 85% most consistent
        consistent_mask = ratio_dev <= max(ratio_thresh, 0.15)

        object_points = np.array(object_points)[consistent_mask].astype(np.float64)
        image_points = np.array(image_points_list)[consistent_mask].astype(np.float64)

        if len(object_points) < 6:
            return None, None, {"error": "Too few consistent 3D points"}

        # Try multiple PnP methods and pick the best
        methods = [
            (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE", 2.0),
            (cv2.SOLVEPNP_EPNP, "EPNP", 2.5),
            (cv2.SOLVEPNP_SQPNP, "SQPNP", 2.0),
        ]

        best_result = None
        best_inliers = 0

        for method, name, reproj_thresh in methods:
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    K,
                    distCoeffs=None,
                    iterationsCount=5000,
                    reprojectionError=reproj_thresh,
                    confidence=0.9999,
                    flags=method
                )

                if success and inliers is not None and len(inliers) > best_inliers:
                    best_result = (rvec, tvec, inliers, name)
                    best_inliers = len(inliers)
            except:
                continue

        if best_result is None:
            return None, None, {"error": "PnP failed"}

        rvec, tvec, inliers, method_name = best_result

        # Convert rvec to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        # solvePnP gives R, t such that: P_cam2 = R @ P_cam1 + t
        # This means t is in camera 2's frame
        # To get camera 2's position in camera 1's frame (ground truth convention):
        t = -R.T @ t

        # Normalize translation to unit vector
        t_norm = np.linalg.norm(t)
        t = t / (t_norm + 1e-8)

        return R, t, {
            "n_matches": len(kpts1),
            "n_3d_points": len(object_points),
            "n_inliers": len(inliers),
            "scale": t_norm,
            "focal": focal
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

        R_est, t_est, info = vo.estimate_pose(str(img1), str(img2), known_focal=481.2)

        if R_est is None:
            print(f"  {i:3d} -> {i+frame_step:3d}: FAILED - {info.get('error', 'Unknown')}")
            continue

        R_err = R_gt @ R_est.T
        rot_err = np.abs(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))) * 180 / np.pi

        cos_ang = np.clip(np.dot(t_gt_norm, t_est), -1, 1)
        dir_err = np.arccos(cos_ang) * 180 / np.pi

        rot_errs.append(rot_err)
        dir_errs.append(dir_err)

        status = "+" if rot_err < 10 and dir_err < 20 else " "
        print(f"{status} {i:3d} -> {i+frame_step:3d}: rot={rot_err:5.1f}  dir={dir_err:5.1f}  "
              f"inliers={info['n_inliers']}")

    rot_errs = np.array(rot_errs)
    dir_errs = np.array(dir_errs)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Rotation:  mean={np.mean(rot_errs):.1f}  median={np.median(rot_errs):.1f}")
    print(f"           <2: {100*np.mean(rot_errs<2):.0f}%  <5: {100*np.mean(rot_errs<5):.0f}%  <10: {100*np.mean(rot_errs<10):.0f}%")
    print(f"Direction: mean={np.mean(dir_errs):.1f}  median={np.median(dir_errs):.1f}")
    print(f"           <5: {100*np.mean(dir_errs<5):.0f}%  <10: {100*np.mean(dir_errs<10):.0f}%  <20: {100*np.mean(dir_errs<20):.0f}%")

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
    parser.add_argument("--focal", type=float)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--data-dir", default="test_images")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=30)
    args = parser.parse_args()

    vo = CompetitionVO()
    vo.load_models()

    if args.test:
        evaluate_icl_nuim(vo, Path(args.data_dir), args.step, args.pairs)
    elif args.img1 and args.img2:
        R, t, info = vo.estimate_pose(args.img1, args.img2, args.focal)
        if R is not None:
            print(f"\nRotation:\n{R}")
            print(f"\nTranslation (normalized): {t}")
            print(f"\nInfo: {info}")
        else:
            print(f"Failed: {info}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
