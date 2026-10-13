import logging
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from camera import Camera
from aruco import Aruco

logger = logging.getLogger("VisionOrchestrator")

@dataclass
class VisionPayload:
    target_acquired: bool
    z_distance_m: float
    x_offset_m: float
    y_offset_m: float
    active_cameras: int


class VisionOrchestrator:
    def __init__(
        self,
        cam0_idx: int,
        cam1_idx: int,
        marker_size_m: float,
        cam0_matrix: np.ndarray,
        cam0_dist: np.ndarray,
        cam1_matrix: np.ndarray,
        cam1_dist: np.ndarray,
        target_id: int = 26,
        cam0_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        cam1_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        camera_settings: Optional[dict] = None
    ):
        logger.info("[ORCHESTRATOR] Initializing multi-camera sensor fusion pipeline...")

        cam_settings = camera_settings or {}
        
        # Initialize camera hardware abstraction threads
        self.cam0 = Camera(
            source=cam0_idx,
            resolution=resolution,
            fps=fps,
            AeEnable=cam_settings.get("AeEnable", False),
            ExposureTime=cam_settings.get("ExposureTime", 50000),
            AnalogueGain=cam_settings.get("AnalogueGain", 2.0),
            Brightness=cam_settings.get("Brightness", 0.1)
        )
        self.cam1 = Camera(
            source=cam1_idx,
            resolution=resolution,
            fps=fps,
            AeEnable=cam_settings.get("AeEnable", False),
            ExposureTime=cam_settings.get("ExposureTime", 50000),
            AnalogueGain=cam_settings.get("AnalogueGain", 2.0),
            Brightness=cam_settings.get("Brightness", 0.1)
        )

        # Initialize subpixel ArUco detection modules
        self.pipe0 = Aruco(marker_size_m, cam0_matrix, cam0_dist, target_id=target_id)
        self.pipe1 = Aruco(marker_size_m, cam1_matrix, cam1_dist, target_id=target_id)

        # Build 4x4 Extrinsic Matrices from camera lenses to the chassis / Center of Gravity
        self.T_chassis_cam0 = np.eye(4, dtype=np.float32)
        self.T_chassis_cam0[0:3, 3] = np.array(cam0_offset_m, dtype=np.float32)

        self.T_chassis_cam1 = np.eye(4, dtype=np.float32)
        self.T_chassis_cam1[0:3, 3] = np.array(cam1_offset_m, dtype=np.float32)

    @staticmethod
    def _transform_pose_to_chassis(
        rvec: np.ndarray,
        tvec: np.ndarray,
        T_chassis_cam: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms marker pose from camera optical frame to drone body/chassis frame:
        T_chassis_marker = T_chassis_cam * T_cam_marker
        """
        R_cam_marker, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4, dtype=np.float32)
        T_cam_marker[0:3, 0:3] = R_cam_marker
        T_cam_marker[0:3, 3] = tvec.reshape(3)

        T_chassis_marker = T_chassis_cam @ T_cam_marker

        tvec_chassis = T_chassis_marker[0:3, 3]
        rvec_chassis, _ = cv2.Rodrigues(T_chassis_marker[0:3, 0:3])
        return rvec_chassis, tvec_chassis

    def get_vision_data(self, do_draw: bool = True) -> Tuple[VisionPayload, Optional[np.ndarray]]:
        """
        Runs single-pass marker detection and fusion.
        Returns the structured VisionPayload and the pre-composited preview frame.
        """
        h0, frame0 = self.cam0.get_frame()
        h1, frame1 = self.cam1.get_frame()

        s0, r0, t0 = False, None, None
        s1, r1, t1 = False, None, None

        if h0 and frame0 is not None:
            s0, r0, t0, frame0 = self.pipe0.process_frame(frame0, draw=do_draw)

        if h1 and frame1 is not None:
            s1, r1, t1, frame1 = self.pipe1.process_frame(frame1, draw=do_draw)

        # Single-pass frame compositing for telemetry preview
        preview_frame = None
        if do_draw:
            if frame0 is not None and frame1 is not None:
                preview_frame = cv2.hconcat([frame0, frame1])
            elif frame0 is not None:
                preview_frame = frame0
            elif frame1 is not None:
                preview_frame = frame1

        if s0 and s1:
            _, t0_cg = self._transform_pose_to_chassis(r0, t0, self.T_chassis_cam0)
            _, t1_cg = self._transform_pose_to_chassis(r1, t1, self.T_chassis_cam1)

            t_fused = (t0_cg + t1_cg) / 2.0
            return (
                VisionPayload(
                    target_acquired=True,
                    z_distance_m=float(t_fused[2]),
                    x_offset_m=float(t_fused[0]),
                    y_offset_m=float(t_fused[1]),
                    active_cameras=2
                ),
                preview_frame
            )

        elif s0 and not s1:
            _, t0_cg = self._transform_pose_to_chassis(r0, t0, self.T_chassis_cam0)
            return (
                VisionPayload(
                    target_acquired=True,
                    z_distance_m=float(t0_cg[2]),
                    x_offset_m=float(t0_cg[0]),
                    y_offset_m=float(t0_cg[1]),
                    active_cameras=1
                ),
                preview_frame
            )

        elif not s0 and s1:
            _, t1_cg = self._transform_pose_to_chassis(r1, t1, self.T_chassis_cam1)
            return (
                VisionPayload(
                    target_acquired=True,
                    z_distance_m=float(t1_cg[2]),
                    x_offset_m=float(t1_cg[0]),
                    y_offset_m=float(t1_cg[1]),
                    active_cameras=1
                ),
                preview_frame
            )

        else:
            return (
                VisionPayload(
                    target_acquired=False,
                    z_distance_m=0.0,
                    x_offset_m=0.0,
                    y_offset_m=0.0,
                    active_cameras=0
                ),
                preview_frame
            )

    def stop(self):
        logger.info("[ORCHESTRATOR] Shutting down camera feeds...")
        self.cam0.stop()
        self.cam1.stop()