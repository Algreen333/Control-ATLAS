import logging
from dataclasses import dataclass
from camera import Camera
from aruco import ArucoPipeline

# 1. THE DATA CONTRACT
# A dataclass strictly formats the output. If your MAVLink pilot script 
# tries to read 'target_aquired' instead of 'target_acquired', it will crash here, 
# preventing a silent failure in the air.
@dataclass
class VisionPayload:
    target_acquired: bool
    z_distance_m: float
    x_offset_m: float
    y_offset_m: float
    active_cameras: int 

class VisionOrchestrator:
    def __init__(self, cam0_idx, cam1_idx, marker_size_m, 
                 cam0_matrix, cam0_dist, cam1_matrix, cam1_dist,
                 cam0_offset_m=(0.0, 0.0, 0.0), cam1_offset_m=(0.0, 0.0, 0.0)):
        
        logging.info("[ORCHESTRATOR] Initializing Vision System...")
        
        # Boot up the background threads
        self.cam0 = Camera(source=cam0_idx)
        self.cam1 = Camera(source=cam1_idx)
        
        # Inject calibration profiles into the math pipelines
        self.pipe0 = ArucoPipeline(marker_size_m, cam0_matrix, cam0_dist)
        self.pipe1 = ArucoPipeline(marker_size_m, cam1_matrix, cam1_dist)

        # Store Extrinsic Offsets (Distance from Camera lens to Drone Center of Gravity)
        self.cam0_offset = cam0_offset_m
        self.cam1_offset = cam1_offset_m

    def get_vision_data(self) -> VisionPayload:
        # Pull latest frames from HAL (instantaneous due to background threads)
        h0, frame0 = self.cam0.get_frame()
        h1, frame1 = self.cam1.get_frame()

        s0, x0, y0, z0 = False, 0.0, 0.0, 0.0
        s1, x1, y1, z1 = False, 0.0, 0.0, 0.0

        # 2. COORDINATE TRANSLATION
        # Calculate camera 0 math, then shift the coordinates from the lens to the drone's CG.
        if h0 and frame0 is not None:
            s0, x0_raw, y0_raw, z0_raw = self.pipe0.process_frame(frame0)
            if s0:
                x0 = x0_raw + self.cam0_offset[0]
                y0 = y0_raw + self.cam0_offset[1]
                z0 = z0_raw + self.cam0_offset[2]
            
        # Calculate camera 1 math, then shift the coordinates to the drone's CG.
        if h1 and frame1 is not None:
            s1, x1_raw, y1_raw, z1_raw = self.pipe1.process_frame(frame1)
            if s1:
                x1 = x1_raw + self.cam1_offset[0]
                y1 = y1_raw + self.cam1_offset[1]
                z1 = z1_raw + self.cam1_offset[2]

        # 3. FUSION LOGIC / FALLBACK STATE MACHINE
        # DUAL MODE: Both cameras are locked on. Average the CG-corrected coordinates.
        if s0 and s1:
            return VisionPayload(
                target_acquired=True,
                z_distance_m=(z0 + z1) / 2.0,
                x_offset_m=(x0 + x1) / 2.0,
                y_offset_m=(y0 + y1) / 2.0,
                active_cameras=2
            )

        # DEGRADED MODE (Cam 0 Failed): Silently drop Cam 0 and use Cam 1 exclusively.
        elif not s0 and s1:
            return VisionPayload(True, z1, x1, y1, 1)

        # DEGRADED MODE (Cam 1 Failed): Silently drop Cam 1 and use Cam 0 exclusively.
        elif s0 and not s1:
            return VisionPayload(True, z0, x0, y0, 1)

        # BLIND MODE: Both cameras failed or lost sight of the marker. 
        # The MAVLink pilot must react to target_acquired=False (e.g., hover in place).
        else:
            return VisionPayload(False, 0.0, 0.0, 0.0, 0)

    def stop(self):
        logging.info("[ORCHESTRATOR] Shutting down vision nodes...")
        self.cam0.stop()
        self.cam1.stop()