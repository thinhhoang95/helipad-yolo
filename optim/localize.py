import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

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
    # print('Depth of point 1: ', depth1)
    depth2 = a3 * x[2] + b3 * x[3] + c3 * x[4]
    # print('Depth of point 2: ', depth2)
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
        10*((x[2]-x[0])**2 + (x[3]-x[1])**2 - R**2),
        10*((x[2]-x[0])**2 / (x[3]-x[1])**2 - 1)
        #x[4] - 0.33966331
    ])

if __name__ == '__main__':
    Rbc = Rotation.from_euler('ZYX', np.array([90,0,0]), degrees=True).as_dcm()
    Rib = Rotation.from_euler('ZYX', np.array([40.13, -7.34, 25.05]), degrees=True).as_dcm().T
    # print(Rib)
    Ric = Rbc @ Rib # multiply with vector in I->B->C 
    # Ric = Rotation.from_euler('ZYX', np.array([40.13, -7.34, 25.05]), degrees=True).as_matrix().T
    # Ric = np.eye(3)
    pose_sol = np.array([0,0,1.5,1.5,2.5])
    foc = 3.04e-3
    print('Preoptimization cost')
    print(bprj(pose_sol, Ric, 296.0, 285.0, 456.0, 393.0, foc, 0.181))
    res = least_squares(bprj, pose_sol, args=(Ric, 177.0, 248.0, 416.0, 357.0, foc, 0.181))
    print('Result: ', res.x)
    print('Post optimization cost')
    print(bprj(res.x, Ric, 177.0, 248.0, 416.0, 357.0, foc, 0.181))
    print('Post optimization residual ', res.cost)