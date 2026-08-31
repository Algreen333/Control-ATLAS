import io
import time
from flask import Flask, Response, request, render_template_string
import picamera

app = Flask(__name__)

# Initialize PiCamera with default settings
camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24
time.sleep(2) # Allow sensor to warm up

# HTML/JS UI embedded as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PiCamera Calibration</title>
    <style>
        body { font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
        .container { display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; justify-content: center; }
        .controls { display: flex; flex-direction: column; gap: 20px; background: #2d2d2d; padding: 25px; border-radius: 12px; min-width: 300px; }
        img { border: 2px solid #444; border-radius: 12px; background: #000; }
        label { display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
        input[type="range"] { width: 150px; }
        select { width: 150px; padding: 4px; background: #444; color: white; border: none; border-radius: 4px; }
    </style>
</head>
<body>
    <h2>Live PiCamera Calibration</h2>
    <div class="container">
        <div>
            <img src="/video_feed" width="640" height="480" alt="Live Stream" />
        </div>
        <div class="controls">
            <label>Brightness (0-100): 
                <input type="range" min="0" max="100" value="50" oninput="update('brightness', this.value)">
            </label>
            <label>Contrast (-100-100): 
                <input type="range" min="-100" max="100" value="0" oninput="update('contrast', this.value)">
            </label>
            <label>Exposure Compensation (-25-25): 
                <input type="range" min="-25" max="25" value="0" oninput="update('exposure_compensation', this.value)">
            </label>
            <label>ISO:
                <select onchange="update('iso', this.value)">
                    <option value="0">Auto</option>
                    <option value="100">100</option>
                    <option value="200">200</option>
                    <option value="400">400</option>
                    <option value="800">800</option>
                </select>
            </label>
            <label>Exposure Mode:
                <select onchange="update('exposure_mode', this.value)">
                    <option value="auto">Auto</option>
                    <option value="night">Night</option>
                    <option value="sports">Sports</option>
                    <option value="snow">Snow</option>
                </select>
            </label>
            <label>AWB (White Balance):
                <select onchange="update('awb_mode', this.value)">
                    <option value="auto">Auto</option>
                    <option value="sunlight">Sunlight</option>
                    <option value="cloudy">Cloudy</option>
                    <option value="tungsten">Tungsten</option>
                    <option value="fluorescent">Fluorescent</option>
                </select>
            </label>
        </div>
    </div>

    <script>
        function update(param, value) {
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
    """Generator that continuously captures frames from the camera into a byte stream."""
    stream = io.BytesIO()
    for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
        stream.seek(0)
        stream.truncate()

@app.route('/')
def index():
    """Serve the main calibration UI."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Serve the MJPEG video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update', methods=['POST'])
def update_params():
    """Receive JSON from the UI and update camera hardware parameters instantly."""
    data = request.json
    for key, value in data.items():
        try:
            # Numeric values need to be cast to integers
            if key in ['brightness', 'contrast', 'exposure_compensation', 'iso']:
                setattr(camera, key, int(value))
            else:
                setattr(camera, key, value)
            print(f"Updated {key} to {value}")
        except Exception as e:
            print(f"Failed to update {key}: {e}")
            
    return {"status": "ok"}

if __name__ == '__main__':
    # Listen on all network interfaces so you can access it from another computer
    app.run(host='0.0.0.0', port=5000, threaded=True)