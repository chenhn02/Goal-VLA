"""Kabsch algorithm for rigid transformation estimation from matched 3D points."""

import glob
from pathlib import Path

import numpy as np
from PIL import Image

from goalvla.config import Config, CameraIntrinsics


def estimate_depth_scale(depth_relative: np.ndarray, depth_real: np.ndarray) -> tuple[float, float]:
    """Linear regression to align relative depth to metric depth.

    Returns (scale, bias) such that depth_real ≈ scale * depth_relative + bias.
    """
    rel = depth_relative.flatten()
    real = depth_real.flatten()
    mask = (rel > 0.5) & (real > 0.5)
    rel_v, real_v = rel[mask], real[mask]

    if len(rel_v) < 10:
        raise ValueError(f"Too few valid pixels for depth scale estimation: {len(rel_v)}")

    s = np.sum((real_v - real_v.mean()) * (rel_v - rel_v.mean())) / np.sum((rel_v - rel_v.mean()) ** 2)
    b = real_v.mean() - s * rel_v.mean()
    return s, b


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch alignment: find R, t such that Q ≈ R @ P + t.

    Args:
        P: (N, 3) source points.
        Q: (N, 3) target points.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        err: mean alignment error.
    """
    assert P.shape == Q.shape
    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)
    X = P - muP
    Y = Q - muQ

    U, D, Vt = np.linalg.svd((X.T @ Y) / len(P))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = muQ - R @ muP
    Q_hat = (R @ P.T).T + t
    err = np.linalg.norm(Q - Q_hat, axis=1).mean()
    return R, t, err


def _lift_to_3d(matches: dict, depth1: np.ndarray, depth2: np.ndarray,
                K: np.ndarray, seg1: np.ndarray, seg2: np.ndarray,
                img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Back-project 2D matches to 3D using depth and intrinsics.

    Also returns the (row, col) pixel coordinate of each kept correspondence
    (px1 in init, px2 in goal), aligned with pts1/pts2, so callers can recover
    the original image color of every matched point for visualization.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    min_disp = img_size[0] * 0.05

    pts1, pts2, px1, px2 = [], [], [], []
    for (x1, y1), (x2, y2) in matches.items():
        if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) <= min_disp:
            continue
        if seg1[x1, y1] < 0.5 or seg2[x2, y2] < 0.5:
            continue
        d1, d2 = depth1[x1, y1], depth2[x2, y2]
        u1 = np.array([(y1 + cx) / fx, (x1 + cy) / fy, 1.0])
        u2 = np.array([(y2 + cx) / fx, (x2 + cy) / fy, 1.0])
        pts1.append(d1 * u1)
        pts2.append(d2 * u2)
        px1.append((x1, y1))
        px2.append((x2, y2))

    return np.stack(pts1), np.stack(pts2), np.array(px1), np.array(px2)


def _ransac_kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Robustly fit R, t (Q ≈ R@P + t) by RANSAC over minimal triplets.

    A single contaminated correspondence can swing the least-squares Kabsch
    rotation by tens of degrees. RANSAC samples 3-point minimal sets, scores each
    candidate transform by the number of 3D inliers (residual < RANSAC_THRESH),
    and refits Kabsch on the largest consensus set. Returns (R, t, err, inlier
    mask), where err is the mean residual over inliers.

    Sampling uses a fixed-seed RNG so the result is reproducible across processes.
    """
    n = len(P)
    rng = np.random.default_rng(Config.MATCHING_SEED)
    thr = Config.RANSAC_THRESH

    best_inliers = None
    best_count = -1
    for _ in range(Config.RANSAC_ITERS):
        sel = rng.choice(n, 3, replace=False)
        try:
            R, t, _ = kabsch(P[sel], Q[sel])
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        inliers = resid < thr
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers

    if best_inliers is None or best_inliers.sum() < 3:
        # RANSAC found no consensus; fall back to a plain fit over everything.
        R, t, err = kabsch(P, Q)
        return R, t, err, np.ones(n, dtype=bool)

    # Refit on the consensus set for the final estimate.
    R, t, _ = kabsch(P[best_inliers], Q[best_inliers])
    resid = np.linalg.norm((R @ P[best_inliers].T).T + t - Q[best_inliers], axis=1)
    return R, t, float(resid.mean()), best_inliers


def _filter_outliers(pts: np.ndarray) -> np.ndarray:
    """Remove outlier points, robust to small match counts.

    The original KMeans(2)-keep-majority scheme was designed for the hundreds
    of (collapsed) matches the old matcher produced. With mutual-consistency
    matching the correspondence set is small (often 3-10), where splitting into
    two clusters and discarding one can decimate the already-scarce inliers and
    leave < 3 points for Kabsch. So: skip filtering when points are few, and
    otherwise reject by distance from the centroid (median + MAD) rather than
    forcing a 2-way split.
    """
    n = len(pts)
    if n < 10:
        return np.ones(n, dtype=bool)

    centroid = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centroid, axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) + 1e-9
    keep = dist <= med + 3.0 * 1.4826 * mad
    if keep.sum() < 3:  # never strip below the Kabsch minimum
        return np.ones(n, dtype=bool)
    return keep


def estimate_transformation(
    matches: dict[tuple[int, int], tuple[int, int]],
    dpt_init: np.ndarray,
    dpt_goal: np.ndarray,
    depth_real: np.ndarray,
    seg_init: np.ndarray,
    seg_goal: np.ndarray,
    mode: str = "real",
    img_size: tuple[int, int] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate rigid transformation from pixel matches + depth.

    Args:
        matches: Dict mapping (row, col) in init to (row, col) in goal.
        dpt_init: DepthAnything relative depth for init image.
        dpt_goal: DepthAnything relative depth for goal image.
        depth_real: Real metric depth for init image (from sensor).
        seg_init: Binary mask for init image.
        seg_goal: Binary mask for goal image.
        mode: 'sim' or 'real' for camera intrinsics.
        img_size: Target image size, default Config.IMG_SIZE.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        info: Dict with debug info (P, Q, Q_hat, scale, bias, error).
    """
    if img_size is None:
        img_size = Config.IMG_SIZE

    K = CameraIntrinsics.get_matrix(mode, img_size)
    scale, bias = estimate_depth_scale(dpt_init, depth_real)

    dpt_init_scaled = dpt_init * scale + bias
    dpt_goal_scaled = dpt_goal * scale + bias

    pts1, pts2, px1, px2 = _lift_to_3d(matches, dpt_init_scaled, dpt_goal_scaled, K,
                                       seg_init, seg_goal, img_size)

    if len(pts1) < 3:
        raise ValueError(f"Need >= 3 valid matches for Kabsch, got {len(pts1)}.")

    if Config.RANSAC_ENABLE and len(pts1) >= Config.RANSAC_MIN_INLIERS:
        R, t, err, inliers = _ransac_kabsch(pts1, pts2)
        pts1, pts2 = pts1[inliers], pts2[inliers]
        px1, px2 = px1[inliers], px2[inliers]
    else:
        mask = _filter_outliers(pts1)
        pts1, pts2 = pts1[mask], pts2[mask]
        px1, px2 = px1[mask], px2[mask]
        if len(pts1) < 3:
            raise ValueError(f"After outlier filtering, only {len(pts1)} points remain.")
        R, t, err = kabsch(pts1, pts2)

    # Translation must be the Kabsch translation t = muQ - R @ muP so that
    # Q ≈ R @ P + t is a consistent rigid transform. Using the plain centroid
    # difference (muQ - muP) ignores the rotation and displaces the predicted
    # goal by (R - I) @ muP — ~1 m of error whenever R is not near identity.
    info = {
        "P": pts1,
        "Q": pts2,
        "px1": px1,
        "px2": px2,
        "Q_hat": (R @ pts1.T).T + t,
        "scale": scale,
        "bias": bias,
        "error": err,
        "K": K,
        "dpt_init_scaled": dpt_init_scaled,
        "dpt_goal_scaled": dpt_goal_scaled,
    }
    return R, t, info


def load_and_estimate(
    img_dir: str | Path,
    mode: str = "real",
    img_size: tuple[int, int] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load data from a directory and estimate transformation.

    Expects: dpt_init.png, dpt_goal.png, mask_init.png, mask_goal.png,
             depth_*.npy (real depth), point_wise_matching.pkl.
    """
    import pickle

    if img_size is None:
        img_size = Config.IMG_SIZE
    img_dir = Path(img_dir)

    dpt_init = np.array(Image.open(img_dir / "dpt_init.png").convert("L").resize(img_size))
    dpt_goal = np.array(Image.open(img_dir / "dpt_goal.png").convert("L").resize(img_size))
    seg_init = np.array(Image.open(img_dir / "mask_init.png").convert("1").resize(img_size))
    seg_goal = np.array(Image.open(img_dir / "mask_goal.png").convert("1").resize(img_size))

    depth_files = glob.glob(str(img_dir / "depth_*.npy"))
    if not depth_files:
        raise FileNotFoundError(f"No depth_*.npy found in {img_dir}")
    depth_real = np.load(depth_files[0])

    if depth_real.shape == (1080, 1920):
        depth_real = depth_real[:, :1440]
    depth_real = np.array(Image.fromarray(depth_real).resize(img_size))

    with open(img_dir / "point_wise_matching.pkl", "rb") as f:
        matches = pickle.load(f)

    return estimate_transformation(
        matches, dpt_init, dpt_goal, depth_real,
        seg_init, seg_goal, mode=mode, img_size=img_size,
    )
