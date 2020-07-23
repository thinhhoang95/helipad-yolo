import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import cv2

def bprj(x, mRic, xb1, yb1, xb2, yb2, mfoc, R):
    s = 0.00000575
    xb1 = (xb1-320) * s
    yb1 = (yb1-240) * s
    xb2 = (xb2-320) * s
    yb2 = (yb2-240) * s
    a1, b1, c1 = mRic[0,0], mRic[0,1], mRic[0,2]
    a2, b2, c2 = mRic[1,0], mRic[1,1], mRic[1,2]
    a3, b3, c3 = mRic[2,0], mRic[2,1], mRic[2,2]
    depth1 = a3 * x[0] + b3 * x[1] + c3 * x[4]
    print('Depth of point 1: ', depth1)
    depth2 = a3 * x[2] + b3 * x[3] + c3 * x[4]
    print('Depth of point 2: ', depth2)
    # print((a1 - a3*xb2/mfoc))
    # print((b1 - b3*xb2/mfoc))
    # print((-c3*xb2/mfoc))
    # print((a2 - a3*yb2/mfoc))
    # print((b2 - b3*yb2/mfoc))
    # print((-c3*yb2/mfoc))
    return np.array([
        ((a1 - a3*xb1/mfoc)*x[0] + (b1 - b3*xb1/mfoc)*x[1] + (c1 -c3*xb1/mfoc)*x[4]),
        ((a2 - a3*yb1/mfoc)*x[0] + (b2 - b3*yb1/mfoc)*x[1] + (c2 -c3*yb1/mfoc)*x[4]),
        ((a1 - a3*xb2/mfoc)*x[2] + (b1 - b3*xb2/mfoc)*x[3] + (c1 -c3*xb2/mfoc)*x[4]),
        ((a2 - a3*yb2/mfoc)*x[2] + (b2 - b3*yb2/mfoc)*x[3] + (c2 -c3*yb2/mfoc)*x[4]),
        2*((x[2]-x[0])**2 + (x[3]-x[1])**2 - R**2),
        2*((x[2]-x[0])**2 / (x[3]-x[1])**2 - 1),
        # x[4] - 1.1418
    ])

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

if __name__ == '__main__':
    Rbc = Rotation.from_euler('ZYX', np.array([-90,0,0]), degrees=True).as_dcm()
    Rib = Rotation.from_euler('ZYX', np.array([89.535, 26.004, -4.9973]), degrees=True).as_dcm().T
    # print(Rib)
    Ric = Rbc @ Rib # multiply with vector in I->B->C 
    # Ric = Rotation.from_euler('ZYX', np.array([40.13, -7.34, 25.05]), degrees=True).as_matrix().T
    # Ric = np.eye(3)
    pose_sol = np.array([0,0,1.5,1.5,2.5])
    foc = 3.04e-3
    
    # Estimation of AI
    #print('Preoptimization cost')
    #print(bprj(pose_sol, Ric, 191.0, 152.0, 277.0, 217.0, foc, 0.22))
    res = least_squares(bprj, pose_sol, args=(Ric, 191.0, 152.0, 277.0, 217.0, foc, 0.22))
    print('Result: ', res.x)
    #print('Post optimization cost')
    #print(bprj(res.x, Ric, 191.0, 152.0, 277.0, 217.0, foc, 0.22))
    print('Post optimization residual ', res.cost)

    # Redraw the result on the image
    im0 = cv2.imread('test.jpg')
    im1 = im0.copy()
    print('Ric from IMU: ', Rotation.from_dcm(Ric).as_euler('ZYX', degrees=True))
    Ritip = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) # rotates around X axis for 180 degrees
    Ripi = Rotation.from_euler('ZYX',np.array([150,0,0]), degrees=True).as_dcm()
    #Riti = Ripi @ Ritip
    #Ritc = Riti @ Ric
    K = np.load('cam_mat.pca.npy')
    dist = np.load('dist.pca.npy')
    xi = np.array([(res.x[0] + res.x[2]) / 2.0, (res.x[1] + res.x[3]) / 2.0, res.x[4]])
    
    cv2.aruco.drawAxis(im1, K, dist, Ric, -Ric @ xi.T, 0.05)
    
    # Estimation of ARUCO tag

    #Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters_create()

    # Load the camera matrix and distortion from file
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(im0, dictionary, parameters=parameters)
    rvecs, tvecs, *other = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.18, K, dist)
    if rvecs is None:
        print('Invalid ARUCO tag detected in the image. Skipping this image.')
    else:
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = rvec[0]
            tvec = tvec[0]
            Ritc = Rotation.from_rotvec(rvec).as_dcm()
            Riti = Ripi @ Ritip
            Ric2 = Ritc @ Riti.T
            print('Ric from ArucoTag: ', Rotation.from_dcm(Ric2).as_euler('ZYX', degrees=True))
            heli_pos = Riti @ np.array([0.245,0,0]).T - Ric.T @ tvec.T # position with respect to the helipad
            print('ARUCO detected at ', heli_pos)
            # Put the axes to indicate the pose on image
            cv2.aruco.drawAxis(im1, K, dist, Ric2, - Ric2 @ heli_pos, 0.05)
            cv2.putText(im1, 'ARUCO: ' + str(heli_pos), (5, 10), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(im1, 'AI: ' + str(xi), (5, 20), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            # cv2.putText(im1, 'TCYPR: ' + str(Rotation.from_dcm(Ritc).as_euler('ZYX', degrees=True)), (5, 80), 0, 0.3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    # <<< Infer data from ARUCO tag <<<
cv2.imshow('Demo', im1)
cv2.waitKey(0)
cv2.destroyAllWindows()
