import cv2
import numpy as np

K = np.array([
    [ 2.22862598e+03,            0.0, 9.60541960e+02 ],
    [            0.0, 2.23376484e+03, 5.26993274e+02 ],
    [            0.0,            0.0,            1.0 ] ])

dist = [-1.41154450e-01, 1.70984495e+00, -3.07334375e-03, -2.07539999e-04, -5.76548608e+00]


orb = cv2.ORB.create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def detect(fr1, fr2):
    kp1, des1 = orb.detectAndCompute(fr1, None)
    kp2, des2 = orb.detectAndCompute(fr2, None)

    return kp1, des1, kp2, des2

def match(des1, des2):
    # Match descriptors between frame I and frame I+1
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

def calcOpticalFlow(kp1, kp2, matches, img1, img2):
    points1 = np.array([kp.pt for kp in kp1], dtype=np.float32).reshape(-1, 1, 2)
    points2 = np.array([kp.pt for kp in kp2], dtype=np.float32).reshape(-1, 1, 2)

    lk_params = dict(
        winSize=(15, 15), 
        maxLevel=2, 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    #points2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, kp1, None, **lk_params)
    pointsnew, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, points2, **lk_params)
    return pointsnew, status, err

def poseEstimation(pts1, pts2):
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    inlier_mask = inliers.ravel().astype(bool)
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]

    E = K.T @ F @ K

    _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

    return R, t, mask


if __name__ == "__main__":
    video = cv2.VideoCapture("video/sample.MOV")

    frame_old = None
    ret, frame = video.read()
    while ret:
        kp1, des1, kp2, des2 = detect(frame, frame_old)
        matches = match(des1, des2)
        
        
        frame_old = frame
        ret, frame = video.read()

    print("Finished!q")
