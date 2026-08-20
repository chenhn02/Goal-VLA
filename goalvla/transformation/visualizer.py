"""Viser-based 3D visualization of transformation results."""

import time

import numpy as np
import viser
from PIL import Image


def visualize_transformation(
    img_init: np.ndarray,
    depth_real: np.ndarray,
    dpt_init_scaled: np.ndarray,
    dpt_goal_scaled: np.ndarray,
    seg_init: np.ndarray,
    seg_goal: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    img_goal: np.ndarray = None,
    P: np.ndarray = None,
    Q: np.ndarray = None,
    port: int = 8081,
):
    """Launch a viser server showing scene point clouds and predicted transformation.

    Visualizes:
      - init_scene: Init scene from real depth (RGB colored)
      - init_scene_depthanything: Init scene from DepthAnything depth
      - predicted_goal: Object points transformed by (R, t) to predicted goal
      - goal_scene: Goal scene from DepthAnything depth
      - P, Q: Matched source/target 3D points (if provided)

    Args:
        img_init: (H, W, 3) uint8 init RGB image.
        depth_real: (H, W) real metric depth.
        dpt_init_scaled: (H, W) scaled DepthAnything depth for init.
        dpt_goal_scaled: (H, W) scaled DepthAnything depth for goal.
        seg_init: (H, W) binary mask for init.
        seg_goal: (H, W) binary mask for goal.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
        K: (3, 3) camera intrinsic matrix.
        img_goal: (H, W, 3) optional goal RGB image.
        P: (N, 3) optional matched source points.
        Q: (N, 3) optional matched target points.
        port: Viser server port.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth_real.shape

    def _backproject(depth, colors_rgb):
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        us, vs = us.flatten(), vs.flatten()
        ds = depth.flatten()
        valid = ds >= 0
        us, vs, ds = us[valid], vs[valid], ds[valid]
        xs = (us + cx) * ds / fx
        ys = (vs + cy) * ds / fy
        pts = np.stack([xs, ys, ds], axis=1)
        colors = colors_rgb.reshape(-1, 3)[valid] / 255.0
        return pts, colors

    colors_init = img_init if img_init.max() > 1 else (img_init * 255).astype(np.uint8)
    pts_real, colors_real = _backproject(depth_real, colors_init)
    pts_dpt, colors_dpt = _backproject(dpt_init_scaled, colors_init)

    # Mask by the same validity used to build pts_dpt (dpt_init_scaled), not
    # depth_real: the real sensor depth has invalid pixels DepthAnything lacks,
    # so filtering by depth_real would misalign obj_mask against pts_dpt.
    seg_flat = seg_init.flatten()[dpt_init_scaled.flatten() >= 0]
    obj_mask = seg_flat > 0.5
    obj_pts = pts_dpt[obj_mask]
    obj_colors = colors_dpt[obj_mask]
    goal_pts = (R @ obj_pts.T + t[:, None]).T

    goal_scene_pts, goal_scene_colors = None, None
    if img_goal is not None:
        colors_goal = img_goal if img_goal.max() > 1 else (img_goal * 255).astype(np.uint8)
        goal_scene_pts, goal_scene_colors = _backproject(dpt_goal_scaled, colors_goal)

    server = viser.ViserServer(port=port)
    print(f"Viser server started at http://localhost:{port}")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.scene.set_up_direction("+z")

            client.scene.add_point_cloud(
                name="init_scene",
                points=pts_real,
                colors=colors_real,
                point_size=0.005,
                point_shape="circle",
            )
            client.scene.add_point_cloud(
                name="init_scene_depthanything",
                points=pts_dpt,
                colors=colors_dpt,
                point_size=0.005,
                point_shape="circle",
            )
            client.scene.add_point_cloud(
                name="predicted_goal",
                points=goal_pts,
                colors=obj_colors,
                point_size=0.005,
                point_shape="circle",
            )

            if P is not None:
                client.scene.add_point_cloud(
                    name="matched_source_P",
                    points=P,
                    colors=np.full((len(P), 3), [1.0, 0.0, 0.0]),
                    point_size=0.008,
                    point_shape="circle",
                )
            if Q is not None:
                client.scene.add_point_cloud(
                    name="matched_target_Q",
                    points=Q,
                    colors=np.full((len(Q), 3), [0.0, 0.0, 1.0]),
                    point_size=0.008,
                    point_shape="circle",
                )

            if goal_scene_pts is not None:
                client.scene.add_point_cloud(
                    name="goal_scene",
                    points=goal_scene_pts,
                    colors=goal_scene_colors,
                    point_size=0.01,
                    point_shape="circle",
                )

            client.camera.position = (0.0, 0.0, 0.0)
            client.camera.look_at = (0.0, 0.0, 1.0)
            client.camera.up = (1.0, 0.0, 0.0)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viser server...")
        server.close()
