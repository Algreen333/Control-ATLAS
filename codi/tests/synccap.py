from picamera2 import Picamera2
from libcamera import controls
import cv2
import time
import numpy as np

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

WIDE_SZ = (1536,864)
NARROW_SZ = (1640,1232)

def resize_with_ratio(img, w_target):
        h_orig, w_orig = img.shape[:2]
        # Calculate aspect ratio
        aspect_ratio = h_orig / w_orig
        h_target = int(w_target * aspect_ratio)
        # Resize using inter_area for better downscaling quality
        return cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_AREA)

def dual_image_display(frame1, frame2):
    wide_res = resize_with_ratio(frame1, int(DISPLAY_WIDTH/2))
    narrow_res = resize_with_ratio(frame2, int(DISPLAY_WIDTH/2))

    h1, w1 = wide_res.shape[:2]
    h2, w2 = narrow_res.shape[:2]

    
    canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

    # Centra imatges
    y_off1 = int((DISPLAY_HEIGHT - h1) / 2)
    y_off2 = int((DISPLAY_HEIGHT - h2) / 2)

    canvas[y_off1 : y_off1 + h1, 0:w1] = wide_res
    canvas[y_off2 : y_off2 + h2, int(DISPLAY_WIDTH/2) : int(DISPLAY_WIDTH/2 + w2)] = narrow_res
    
    return canvas

def setup_video_sync_node(cam_index, sync_mode):
    """Initializes a camera for continuous video capture with sync controls."""
    picam = Picamera2(cam_index)
    
    # Create a continuous video configuration instead of still
    config = picam.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    
    # Queue=False ensures we pull the absolute freshest frame from the buffer [cite: 67]
    config["main"]["queue"] = False
    
    # Increase buffer count to provide a larger cushion for the CPU to handle data influx [cite: 68]
    config["main"]["buffer_count"] = 6
    
    picam.configure(config)
    
    # Inject the SyncMode control into the IPA layer
    if sync_mode == "server":
        picam.set_controls({"SyncMode": controls.rpi.SyncModeEnum.Server})
        print(f"Camera {cam_index} initialized as VIDEO SYNC SERVER.")
    elif sync_mode == "client":
        picam.set_controls({"SyncMode": controls.rpi.SyncModeEnum.Client})
        print(f"Camera {cam_index} initialized as VIDEO SYNC CLIENT.")
        
    picam.start()
    return picam

def run_realtime_feed(cam_server, cam_client):
    """Pulls continuous synchronized frames and monitors temporal drift."""
    print("Waiting 2 seconds for IPA feedback loop to converge...")
    time.sleep(2)
    
    print("Starting real-time synchronized feed. Press 'q' to exit.")
    
    try:
        while True:
            # Capture continuous requests from the hardware [cite: 60]
            req_server = cam_server.capture_request()
            req_client = cam_client.capture_request()
            
            # Extract the precise nanosecond the sensor began reading out the frame [cite: 63]
            ts_server = req_server.get_metadata().get('SensorTimestamp', 0)
            ts_client = req_client.get_metadata().get('SensorTimestamp', 0)
            
            drift_ns = abs(ts_server - ts_client)
            drift_ms = drift_ns / 1_000_000.0
            
            # Extract image buffers
            img_server = req_server.make_array("main")
            img_client = req_client.make_array("main")
            
            # Release requests to prevent memory exhaustion
            req_server.release()
            req_client.release()

            # Visual overlay for telemetry
            cv2.putText(img_server, f"SERVER | Drift: {drift_ms:.3f} ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_client, f"CLIENT | Drift: {drift_ms:.3f} ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the real-time feeds
            cv2.imshow("Server Feed", cv2.cvtColor(img_server, cv2.COLOR_RGB2BGR))
            cv2.imshow("Client Feed", cv2.cvtColor(img_client, cv2.COLOR_RGB2BGR))

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure ptp4l and phc2sys are running on your OS before executing this script
    cam_server = setup_video_sync_node(0, "server")
    cam_client = setup_video_sync_node(1, "client")
    
    try:
        run_realtime_feed(cam_server, cam_client)
    finally:
        cam_server.stop()
        cam_client.stop()