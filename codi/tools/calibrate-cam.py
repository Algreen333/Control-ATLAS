import cv2
from picamera2 import Picamera2

# 1. Initialize PiCamera2
picam2 = Picamera2()

# Configure the camera for a fast video stream
# You can change the size to (1280, 720) or higher, but 640x480 is very responsive for tuning.
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# 2. Setup OpenCV Window and Trackbars
window_name = "PiCamera2 Calibration"
cv2.namedWindow(window_name)

# Dummy callback function required by OpenCV trackbars
def on_trackbar(val):
    pass

# OpenCV trackbars only support positive integers (0 to max).
# We set ranges here and map them to the float limits libcamera expects later.
cv2.createTrackbar("Brightness", window_name, 100, 200, on_trackbar)  # Maps to: -1.0 to 1.0 (Default 0.0 = 100)
cv2.createTrackbar("Contrast", window_name, 10, 50, on_trackbar)      # Maps to: 0.0 to 5.0 (Default 1.0 = 10)
cv2.createTrackbar("Saturation", window_name, 10, 50, on_trackbar)    # Maps to: 0.0 to 5.0 (Default 1.0 = 10)
cv2.createTrackbar("Sharpness", window_name, 10, 50, on_trackbar)     # Maps to: 0.0 to 5.0 (Default 1.0 = 10)
cv2.createTrackbar("Exposure(EV)", window_name, 80, 160, on_trackbar) # Maps to: -8.0 to 8.0 (Default 0.0 = 80)

print("Starting live preview. Press 'q' or 'ESC' on the video window to quit.")

try:
    while True:
        # Capture the current frame from the camera stream
        frame = picam2.capture_array()
        
        # Picamera2 returns RGB by default; OpenCV requires BGR for display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Read the current values from the sliders
        tb_brightness = cv2.getTrackbarPos("Brightness", window_name)
        tb_contrast = cv2.getTrackbarPos("Contrast", window_name)
        tb_saturation = cv2.getTrackbarPos("Saturation", window_name)
        tb_sharpness = cv2.getTrackbarPos("Sharpness", window_name)
        tb_ev = cv2.getTrackbarPos("Exposure(EV)", window_name)
        
        # Convert integers back to the correct float values for libcamera
        brightness_val = (tb_brightness - 100) / 100.0
        contrast_val = tb_contrast / 10.0
        saturation_val = tb_saturation / 10.0
        sharpness_val = tb_sharpness / 10.0
        ev_val = (tb_ev - 80) / 10.0
        
        # Apply the parameters to the hardware in real-time
        controls = {
            "Brightness": brightness_val,
            "Contrast": contrast_val,
            "Saturation": saturation_val,
            "Sharpness": sharpness_val,
            "ExposureValue": ev_val
        }
        picam2.set_controls(controls)
        
        # Overlay the parameter values directly onto the video feed
        text = f"Brt:{brightness_val:.2f} Cnt:{contrast_val:.1f} Sat:{saturation_val:.1f} Shp:{sharpness_val:.1f} EV:{ev_val:.1f}"
        cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the live window
        cv2.imshow(window_name, frame_bgr)
        
        # Check for 'q' or 'ESC' key press to exit safely
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Always clean up camera and window resources when done
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely shut down.")