import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (main)
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lib.camera_lib import *
from lib.aruco_lib import *
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


cap = VideoCapture(0, resolution=(1920, 1080))
config = CalibrationConfig.from_path("../configs/iphone.conf")


# Initialize ORB detector
orb = cv2.ORB.create(nfeatures=2000)

# Matcher using Hamming distance (standard for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS |   cv2.TERM_CRITERIA_COUNT, 10, 0.03))


prev_img = None
trajectory_data = [[0, 0, 0]]
global_pose = np.eye(4)        # The "World" position of the camera


def get_transformation_matrix(R, t):
    """Combines R and t into a 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def get_trajectory_plot(trajectory_data):
    """
    Converts trajectory points into a NumPy image array.
    """
    # 1. Create a figure (Agg backend for headless servers)
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    traj = np.array(trajectory_data)
    
    if len(traj) > 1:
        # Plot the path
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='b', lw=2)
        
        # Plot the most recent point as a red dot
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='r', s=50)
        
        # Calculate and show 'direction' arrow
        # Vector from second-to-last to last point
        direction = traj[-1] - traj[-2]
        if np.linalg.norm(direction) > 0:
            ax.quiver(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                      direction[0], direction[1], direction[2], 
                      color='r', length=0.5, normalize=True)

    ax.set_title("3D Camera Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 2. Convert Matplotlib figure to a NumPy array (RGB)
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Get the RGBA buffer and convert to RGB

    rgba_buffer = canvas.buffer_rgba()
    img_array = np.asarray(rgba_buffer)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    
    # Clean up memory immediately
    plt.close(fig)
    
    return img_bgr
    
cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
def VIO(img1, img2):
    # Detect and Compute
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 2. Estimate Essential Matrix
    # E is the geometric constraint between two views
    E, mask = cv2.findEssentialMat(pts1, pts2, config.mtx, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # 3. Recover Pose
    # Decompose E into Rotation (R) and Translation (t)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, config.mtx)

    return R, t


def processor(frame):
    global prev_img, trajectory_data, global_pose
    
    try:
        if prev_img is not None:
            R, t = VIO(frame, prev_img)
            #print(f"Rotation: \n{R} \nTranslation: \n{t}")

            relative_T = np.eye(4)
            relative_T[:3, :3] = R
            relative_T[:3, 3] = t.flatten()

            global_pose = global_pose @ relative_T
            
            # 4. Extract the new [x, y, z] position
            pos = global_pose[:3, 3]
            trajectory_data.append(pos.tolist())

            print(f"Current Position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

            plot_img = get_trajectory_plot(trajectory_data)
            prev_img = frame

            cv2.imshow("im", plot_img)
            cv2.waitKey(30)
            return plot_img

        prev_img = frame
    except Exception as e:
        print("'VIO processor error':", e)
    return frame



if __name__ == "__main__":
    cap.start()

    server = VideoServer(cap, port=8008)
    server.create_route("/vio", "vio", processor)
    server.start()

    while input("input q to stop...") != "q":
        pass