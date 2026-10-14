import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Union
import cv2
import numpy as np

try:
    from .camera import Camera
    from .aruco import Aruco
except (ImportError, ValueError):
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
    drift_ms: Optional[float] = None


class VisionOrchestrator:
    def __init__(
        self,
        cam0_idx: Union[int, str],
        cam1_idx: Union[int, str],
        marker_size_m: float,
        cam0_matrix: np.ndarray,
        cam0_dist: np.ndarray,
        cam1_matrix: np.ndarray,
        cam1_dist: np.ndarray,
        target_id: int = 26,
        aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
        cam0_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        cam1_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        cam0_settings: Optional[dict] = None,
        cam1_settings: Optional[dict] = None,
        do_gazebo: bool = False
    ):
        logger.info("[ORCHESTRATOR] Initializing sensor fusion perception pipeline...")

        s0 = cam0_settings or {}
        s1 = cam1_settings or {}
        self.do_gazebo = bool(do_gazebo)

        # Safe Camera 0 initialization
        self.cam0: Optional[Camera] = None
        try:
            self.cam0 = Camera(
                source=cam0_idx,
                resolution=resolution,
                fps=fps,
                AeEnable=s0.get("AeEnable", False),
                ExposureTime=s0.get("ExposureTime", 5000),  # Default 5ms for flight
                AnalogueGain=s0.get("AnalogueGain", 2.0),
                Brightness=s0.get("Brightness", 0.1),
                do_gazebo=self.do_gazebo
            )
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Primary camera 0 (idx {cam0_idx}) offline: {e}")

        # Safe Camera 1 initialization (Skipped in single-cam Gazebo SITL mode)
        self.cam1: Optional[Camera] = None
        if not self.do_gazebo:
            try:
                self.cam1 = Camera(
                    source=cam1_idx,
                    resolution=resolution,
                    fps=fps,
                    AeEnable=s1.get("AeEnable", False),
                    ExposureTime=s1.get("ExposureTime", 5000),
                    AnalogueGain=s1.get("AnalogueGain", 2.0),
                    Brightness=s1.get("Brightness", 0.1),
                    do_gazebo=False
                )
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Secondary camera 1 (idx {cam1_idx}) offline: {e}")
        else:
            logger.info("[ORCHESTRATOR] Gazebo SITL mode active: Secondary stereo camera bypassed.")

        if self.cam0 is None and self.cam1 is None:
            raise RuntimeError("[ORCHESTRATOR] CRITICAL: All camera sources failed to initialize.")

        self.pipe0 = Aruco(marker_size_m, cam0_matrix, cam0_dist, target_id=target_id, dict_id=aruco_dict_id)
        self.pipe1 = Aruco(marker_size_m, cam1_matrix, cam1_dist, target_id=target_id, dict_id=aruco_dict_id)

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
        R_cam_marker, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4, dtype=np.float32)
        T_cam_marker[0:3, 0:3] = R_cam_marker
        T_cam_marker[0:3, 3] = tvec.reshape(3)

        T_chassis_marker = T_chassis_cam @ T_cam_marker
        tvec_chassis = T_chassis_marker[0:3, 3]
        rvec_chassis, _ = cv2.Rodrigues(T_chassis_marker[0:3, 0:3])
        return rvec_chassis, tvec_chassis

    def get_vision_data(self, do_draw: bool = True) -> Tuple[VisionPayload, Optional[np.ndarray]]:
        # Only copy buffer if drawing overlays in-place to protect raw evidence frames
        h0, frame0, ts0 = self.cam0.get_frame(copy=do_draw) if self.cam0 else (False, None, 0.0)
        h1, frame1, ts1 = self.cam1.get_frame(copy=do_draw) if self.cam1 else (False, None, 0.0)

        drift_ms: Optional[float] = None
        if h0 and h1:
            drift_ms = abs(ts0 - ts1) * 1000.0

        s0, r0, t0 = False, None, None
        s1, r1, t1 = False, None, None

        if h0 and frame0 is not None:
            s0, r0, t0, frame0 = self.pipe0.process_frame(frame0, draw=do_draw)

        if h1 and frame1 is not None:
            s1, r1, t1, frame1 = self.pipe1.process_frame(frame1, draw=do_draw)

        # Telemetry compositing
        preview_frame = None
        if do_draw:
            if frame0 is not None and frame1 is not None:
                preview_frame = cv2.hconcat([frame0, frame1])
            elif frame0 is not None:
                preview_frame = frame0
            elif frame1 is not None:
                preview_frame = frame1

        # Sensor Fusion & Automatic Failover
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
                    active_cameras=2,
                    drift_ms=drift_ms
                ),
                preview_frame
            )

        if s0 and not s1:
            _, t0_cg = self._transform_pose_to_chassis(r0, t0, self.T_chassis_cam0)
            return (
                VisionPayload(
                    target_acquired=True,
                    z_distance_m=float(t0_cg[2]),
                    x_offset_m=float(t0_cg[0]),
                    y_offset_m=float(t0_cg[1]),
                    active_cameras=1,
                    drift_ms=drift_ms
                ),
                preview_frame
            )

        if not s0 and s1:
            _, t1_cg = self._transform_pose_to_chassis(r1, t1, self.T_chassis_cam1)
            return (
                VisionPayload(
                    target_acquired=True,
                    z_distance_m=float(t1_cg[2]),
                    x_offset_m=float(t1_cg[0]),
                    y_offset_m=float(t1_cg[1]),
                    active_cameras=1,
                    drift_ms=drift_ms
                ),
                preview_frame
            )

        return (
            VisionPayload(
                target_acquired=False,
                z_distance_m=0.0,
                x_offset_m=0.0,
                y_offset_m=0.0,
                active_cameras=0,
                drift_ms=drift_ms
            ),
            preview_frame
        )

    def get_evidence_frame(self) -> Optional[np.ndarray]:
        """Safely retrieves a clean copy of a frame from whichever camera is healthy."""
        if self.cam0:
            h, f, _ = self.cam0.get_frame(copy=True)
            if h and f is not None:
                return f
        if self.cam1:
            h, f, _ = self.cam1.get_frame(copy=True)
            if h and f is not None:
                return f
        return None

    def stop(self):
        logger.info("[ORCHESTRATOR] Shutting down perception streams...")
        if self.cam0:
            self.cam0.stop()
        if self.cam1:
            self.cam1.stop()