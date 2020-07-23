import cv2
from scipy.spatial.transform import Rotation
import numpy as np

# Declaration of common variables -------------------------
foc = 3.04e-3
Rbc = Rotation.from_euler('ZYX', np.array([-90,0,0]), degrees=False).as_dcm()
Ritip = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # rotates around X axis for 180 degrees
Ripi = Rotation.from_euler('ZYX',np.array([150,0,0]), degrees=True).as_dcm()
# ----------------------------------------------------------
im0 = cv2.imread('test.jpg')
im1 = im0.copy()
# Further processing of detection results
ypr = np.array([89.535, 26.004, -4.9973])
Rib = Rotation.from_euler('ZYX', ypr, degrees=True).as_dcm().T
Ric = Rbc @ Rib
# >>> Infer data from ARUCO tag >>>

#Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters_create()

# Load the camera matrix and distortion from file
cam_mat = np.load('cam_mat.pca.npy')
dist = np.load('dist.pca.npy')

# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(im0, dictionary, parameters=parameters)
rvecs, tvecs, *other = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.18, cam_mat, dist)
if rvecs is None:
    print('Invalid ARUCO tag detected in the image. Skipping this image.')
else:
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = rvec[0]
        tvec = tvec[0]
        Ritc = Rotation.from_rotvec(rvec).as_dcm()
        Riti = Ripi @ Ritip
        heli_pos = Riti @ np.array([0.245,0,0]).T - Ric.T @ tvec.T # position with respect to the helipad
        print('ARUCO detected at ', heli_pos)
        cv2.putText(im1, 'ARUCO: ' + str(heli_pos), (5, 70), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(im1, 'TCYPR: ' + str(Rotation.from_dcm(Ritc).as_euler('ZYX', degrees=True)), (5, 80), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
# <<< Infer data from ARUCO tag <<<

cv2.imshow('Tag detection', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()