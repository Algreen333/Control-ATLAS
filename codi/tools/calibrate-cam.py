import io
from flask import Flask, Response, request, render_template_string
from picamera2 import Picamera2
from PIL import Image

app = Flask(__name__)

# Initialize Picamera2 and configure it for preview
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# HTML/JS UI updated for Picamera2 controls
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Picamera2 Calibration</title>
    <style>
        body { font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; justify-content: center; }
        .controls { display: flex; flex-direction: column; gap: 20px; background: #2d2d2d; padding: 25px; border-radius: 12px; min-width: 350px; }
        img { border: 2px solid #444; border-radius: 12px; background: #000; }
        label { display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
        input[type="range"] { width: 150px; }
        select { width: 150px; padding: 4px; background: #444; color: white; border: none; border-radius: 4px; }
        span.val { display: inline-block; width: 30px; text-align: right; color: #888; font-size: 12px; margin-right: 10px;}
    </style>
</head>
<body>
    <h2>Live Picamera2 Calibration</h2>
    <div class="container">
        <div>
            <img src="/video_feed" width="640" height="480" alt="Live Stream" />
        </div>
        <div class="controls">
            <label>Brightness: 
                <div>
                    <span class="val" id="val-Brightness">0</span>
                    <input type="range" min="-1" max="1" step="0.1" value="0" oninput="update('Brightness', this.value)">
                </div>
            </label>
            <label>Contrast: 
                <div>
                    <span class="val" id="val-Contrast">1.0</span>
                    <input type="range" min="0" max="2" step="0.1" value="1.0" oninput="update('Contrast', this.value)">
                </div>
            </label>
            <label>Saturation: 
                <div>
                    <span class="val" id="val-Saturation">1.0</span>
                    <input type="range" min="0" max="2" step="0.1" value="1.0" oninput="update('Saturation', this.value)">
                </div>
            </label>
            <label>Exposure Comp (EV): 
                <div>
                    <span class="val" id="val-ExposureValue">0</span>
                    <input type="range" min="-8" max="8" step="0.5" value="0" oninput="update('ExposureValue', this.value)">
                </div>
            </label>
            <label>White Balance:
                <select onchange="update('AwbMode', this.value)">
                    <option value="0">Auto</option>
                    <option value="1">Incandescent</option>
                    <option value="2">Tungsten</option>
                    <option value="3">Fluorescent</option>
                    <option value="5">Daylight</option>
                    <option value="6">Cloudy</option>
                </select>
            </label>
        </div>
    </div>

    <script>
        function update(param, value) {
            // Update the display number next to sliders
            let valSpan = document.getElementById('val-' + param);
            if(valSpan) valSpan.innerText = value;

            // Send to Flask backend
            fetch('/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [param]: value })
            }).catch(err => console.error("Update failed:", err));
        }
    </script>
</body>
</html>
"""

def generate_frames():
    """Continuously pull frames from Picamera2, encode to JPEG, and yield to stream."""
    while True:
        try:
            # Capture raw array from camera (this automatically throttles to camera FPS)
            frame = picam2.capture_array()
            
            # Convert numpy array to JPEG using Pillow
            img = Image.fromarray(frame)
            stream = io.BytesIO()
            img.save(stream, format='JPEG', quality=85)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream.getvalue() + b'\r\n')
        except Exception as e:
            print(f"Frame capture error: {e}")
            break

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update', methods=['POST'])
def update_params():
    """Receive JSON and set the new libcamera controls."""
    data = request.json
    controls_to_update = {}
    
    for key, value in data.items():
        try:
            val = float(value)
            # Enums like AwbMode require integers
            if key in ['AwbMode']:
                val = int(val)
                
            controls_to_update[key] = val
        except ValueError:
            print(f"Invalid value for {key}: {value}")
            
    if controls_to_update:
        try:
            picam2.set_controls(controls_to_update)
            print(f"Updated camera controls: {controls_to_update}")
        except Exception as e:
            print(f"Failed to update controls: {e}")
            
    return {"status": "ok"}

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        picam2.stop()