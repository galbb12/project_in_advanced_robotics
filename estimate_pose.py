#!/usr/bin/env python3
"""
Core Monocular Visual Odometry - Relative Pose Estimation

Estimates relative pose (R, t) between two images given camera intrinsics.
Returns rotation matrix and normalized translation direction (scale-free).

Usage:
    python estimate_pose.py --img1 path/to/img1.png --img2 path/to/img2.png \
                            --fx 500 --fy 500 --cx 320 --cy 240

    Or with K matrix file:
    python estimate_pose.py --img1 img1.png --img2 img2.png --K calib.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2
import torch


def get_device(preferred: str = 'auto') -> torch.device:
    """
    Get the best available device for PyTorch.

    Args:
        preferred: 'auto', 'cuda', 'mps', or 'cpu'
                  'auto' will detect the best available device

    Returns:
        torch.device for the selected backend
    """
    if preferred == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    elif preferred == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("Warning: CUDA not available, falling back to CPU")
        return torch.device('cpu')
    elif preferred == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        print("Warning: MPS not available, falling back to CPU")
        return torch.device('cpu')
    else:
        return torch.device('cpu')


class PoseEstimator:
    """Estimates relative pose between two images using feature matching + Essential matrix."""

    def __init__(self, device: str = 'auto'):
        """
        Initialize the pose estimator.

        Args:
            device: 'auto' (detect best), 'cuda', 'mps' (Apple Silicon), or 'cpu'
        """
        self.device = get_device(device)
        self.extractor = None
        self.matcher = None
        self.feature_cache = {}

    def load_models(self):
        """Load feature extraction models (SuperPoint + LightGlue)."""
        from lightglue import LightGlue, SuperPoint

        self.extractor = SuperPoint(
            max_num_keypoints=8192,
            detection_threshold=0.0005,  # Lower = more keypoints
            nms_radius=3,  # Smaller = denser keypoints (default 4)
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features='superpoint',
            depth_confidence=0.9,  # Early stopping (default -1 = disabled)
            width_confidence=0.95,  # Pruning confidence
            filter_threshold=0.1,  # Match confidence threshold
        ).eval().to(self.device)

    def extract_features(self, image_path: str) -> dict:
        """Extract SuperPoint features."""
        if image_path in self.feature_cache:
            return self.feature_cache[image_path]

        from lightglue.utils import load_image
        image = load_image(image_path).to(self.device)

        with torch.no_grad():
            feats = self.extractor.extract(image)

        self.feature_cache[image_path] = feats
        return feats

    def match_features(self, img1_path: str, img2_path: str,
                       confidence_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match features between two images. Returns matched points (Nx2 arrays) and scores."""
        feats0 = self.extract_features(img1_path)
        feats1 = self.extract_features(img2_path)

        with torch.no_grad():
            matches = self.matcher({'image0': feats0, 'image1': feats1})

        matches_idx = matches['matches'][0].cpu().numpy()
        valid = matches_idx[:, 0] >= 0

        # Get scores for valid matches only
        if 'matching_scores0' in matches:
            scores = matches['matching_scores0'][0].cpu().numpy()
            # Scores correspond to keypoints in image0, get scores for matched ones
            match_scores = scores[matches_idx[valid, 0]]
            if confidence_threshold > 0:
                conf_mask = match_scores >= confidence_threshold
                valid_idx = np.where(valid)[0][conf_mask]
                valid = np.zeros_like(valid)
                valid[valid_idx] = True
                match_scores = match_scores[conf_mask]
        else:
            match_scores = np.ones(np.sum(valid))

        pts1 = feats0['keypoints'][0].cpu().numpy()[matches_idx[valid, 0]]
        pts2 = feats1['keypoints'][0].cpu().numpy()[matches_idx[valid, 1]]

        return pts1, pts2, match_scores

    def estimate_essential(self, pts1: np.ndarray, pts2: np.ndarray,
                           K: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Estimate Essential matrix using RANSAC.
        Returns E, inlier_mask, n_inliers.
        """
        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.USAC_MAGSAC,
                prob=0.99999,
                threshold=threshold,
                maxIters=50000
            )
        except:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.99999,
                threshold=threshold * 2
            )

        if E is None or mask is None:
            return None, None, 0

        mask = mask.ravel().astype(bool)
        return E, mask, np.sum(mask)

    def decompose_essential(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                            K: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Decompose Essential matrix into R, t using cheirality check.
        Returns R, t (unit vector), n_valid_points.
        """
        n_valid, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        t = t.ravel()

        # Position of cam2 origin in cam1 frame
        t = -(R.T @ t)
        t = t / (np.linalg.norm(t) + 1e-8)

        return R, t, n_valid

    def estimate(self, img1_path: str, img2_path: str,
                 K: np.ndarray, dist_coeffs: np.ndarray = None,
                 fisheye: bool = False) -> dict:
        """
        Estimate relative pose between two images.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            K: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (optional)
            fisheye: If True, use fisheye undistortion model

        Returns:
            dict with:
                - success: bool
                - R: 3x3 rotation matrix (cam1 to cam2)
                - t: 3x1 normalized translation direction
                - n_inliers: number of inlier matches
                - n_matches: total matches
                - error: error message if failed
        """
        # Match features
        pts1, pts2, scores = self.match_features(img1_path, img2_path, confidence_threshold=0.0)

        if len(pts1) < 20:
            return {
                'success': False,
                'error': f'Insufficient matches: {len(pts1)}',
                'n_matches': len(pts1)
            }

        n_matches = len(pts1)

        # Undistort points if distortion coefficients provided
        if dist_coeffs is not None:
            if fisheye:
                pts1 = cv2.fisheye.undistortPoints(
                    pts1.reshape(-1, 1, 2).astype(np.float64), K, dist_coeffs, P=K
                ).reshape(-1, 2)
                pts2 = cv2.fisheye.undistortPoints(
                    pts2.reshape(-1, 1, 2).astype(np.float64), K, dist_coeffs, P=K
                ).reshape(-1, 2)
            else:
                pts1 = cv2.undistortPoints(
                    pts1.reshape(-1, 1, 2).astype(np.float64), K, dist_coeffs, P=K
                ).reshape(-1, 2)
                pts2 = cv2.undistortPoints(
                    pts2.reshape(-1, 1, 2).astype(np.float64), K, dist_coeffs, P=K
                ).reshape(-1, 2)

        # Estimate Essential matrix (tight threshold for better precision)
        E, inlier_mask, n_inliers = self.estimate_essential(pts1, pts2, K, threshold=0.2)

        if E is None or n_inliers < 10:
            return {
                'success': False,
                'error': 'Essential matrix estimation failed',
                'n_matches': n_matches,
                'n_inliers': n_inliers if n_inliers else 0
            }

        # Filter to inliers
        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]

        # Decompose E
        R, t, n_valid = self.decompose_essential(E, pts1_in, pts2_in, K)

        if R is None or n_valid < 5:
            return {
                'success': False,
                'error': 'Essential matrix decomposition failed',
                'n_matches': n_matches,
                'n_inliers': n_inliers
            }

        return {
            'success': True,
            'R': R,
            't': t,
            'n_inliers': n_inliers,
            'n_matches': n_matches
        }


def load_K_from_file(filepath: str) -> np.ndarray:
    """Load camera intrinsic matrix from file (supports various formats)."""
    with open(filepath, 'r') as f:
        content = f.read().strip()

    # Try JSON format
    if content.startswith('{'):
        data = json.loads(content)
        if 'K' in data:
            return np.array(data['K']).reshape(3, 3)
        elif 'fx' in data:
            fx, fy = data['fx'], data['fy']
            cx, cy = data['cx'], data['cy']
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Try KITTI format (P0: ...)
    for line in content.split('\n'):
        if line.startswith('P0:') or line.startswith('P2:'):
            values = [float(x) for x in line.split(':')[1].strip().split()]
            P = np.array(values).reshape(3, 4)
            return P[:3, :3]

    # Try plain matrix (9 numbers)
    values = [float(x) for x in content.split()]
    if len(values) == 9:
        return np.array(values).reshape(3, 3)

    raise ValueError(f"Could not parse K matrix from {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Estimate relative pose between two images")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image")

    # Camera intrinsics (either K file or individual params)
    parser.add_argument("--K", type=str, help="Path to camera intrinsics file")
    parser.add_argument("--fx", type=float, help="Focal length x")
    parser.add_argument("--fy", type=float, help="Focal length y (defaults to fx)")
    parser.add_argument("--cx", type=float, help="Principal point x")
    parser.add_argument("--cy", type=float, help="Principal point y")

    # Distortion correction
    parser.add_argument("--dist", type=str, help="Distortion coefficients (comma-separated: k1,k2,k3,k4)")
    parser.add_argument("--fisheye", action="store_true", help="Use fisheye distortion model")

    parser.add_argument("--output", type=str, help="Output file (JSON format)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    # Build K matrix
    if args.K:
        K = load_K_from_file(args.K)
    elif args.fx and args.cx and args.cy:
        fx = args.fx
        fy = args.fy if args.fy else fx
        K = np.array([[fx, 0, args.cx], [0, fy, args.cy], [0, 0, 1]], dtype=np.float64)
    else:
        print("Error: Must provide either --K file or --fx, --cx, --cy", file=sys.stderr)
        sys.exit(1)

    # Parse distortion coefficients
    dist_coeffs = None
    if args.dist:
        dist_coeffs = np.array([float(x) for x in args.dist.split(',')])

    # Estimate pose
    estimator = PoseEstimator()
    estimator.load_models()

    result = estimator.estimate(args.img1, args.img2, K,
                                dist_coeffs=dist_coeffs, fisheye=args.fisheye)

    # Output
    if result['success']:
        output = {
            'success': True,
            'R': result['R'].tolist(),
            't': result['t'].tolist(),
            'n_inliers': int(result['n_inliers']),
            'n_matches': int(result['n_matches'])
        }

        if not args.quiet:
            print(f"Success: {result['n_inliers']} inliers / {result['n_matches']} matches")
            print(f"R:\n{result['R']}")
            print(f"t: {result['t']}")
    else:
        output = {
            'success': False,
            'error': result['error']
        }
        if not args.quiet:
            print(f"Failed: {result['error']}", file=sys.stderr)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
