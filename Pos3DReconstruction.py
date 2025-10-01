import numpy as np
import cv2

if __name__ == "__main__":   
    # Camera Object Initialisation
    K_left = np.array([[1811.910046453570, 0.0, 588.5594517681759],
                       [0.0, 1809.640734154330, 477.3975900383616],
                       [0.0, 0.0, 1.0]])
    K_right = np.array([[1801.712669735144, 0.0, 791.7629609322958],
                        [0.0, 1796.928461921111, 437.4978636860555],
                        [0.0, 0.0, 1.0]])
    D_left = np.array([-0.251655177510111, 0.503352413478258, -0.002139555248137, -0.004349153536928, -0.246027939563351])
    D_right = np.array([-0.257090786509171, 0.101341249569555, 0.0007793893931081916, 0.0007405068673525044, 2.505085264989695])

    R_0 = 0.999940944984011
    R_1 = 0.010580130002784
    R_2 = -0.002483423767351
    R_3 = -0.010563627896603
    R_4 = 0.999922642208192
    R_5 = 0.006566533716574
    R_6 = 0.002552706435562
    R_7 = -0.006539911965232
    R_8 = 0.999975356317015
    T_0 = -5.921169311997873
    T_1 = -0.076797190621161
    T_2 = -0.796790606608169

    R = np.array([[R_0,R_1,R_2],[R_3,R_4,R_5],[R_6,R_7,R_8]])
    T = np.array([T_0, T_1, T_2])

    imgL = cv2.imread("/home/zc519/Downloads/SurgPoseDataSet/000000/regular/LeftImages/frame70.png", cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread("/home/zc519/Downloads/SurgPoseDataSet/000000/regular/RightImages/frame70.png", cv2.IMREAD_GRAYSCALE)
    img_size = imgL.shape[::-1]
    # Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, img_size, R, T)
    
    map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, img_size, cv2.CV_32FC1)

    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*8,   # must be multiple of 16
        blockSize=15,
        P1=8*3*5**2,
        P2=32*3*5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=16*8,   # must be multiple of 16
    #     blockSize=5
    # )

    disparity = stereo.compute(rectL, rectR).astype(np.float32) / 16.0
    cv2.imshow("disparity", disparity)
    cv2.waitKey(0)
    # disparity[688,965] = abs(965-1093)
    # Reproject disparity to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0  # filter valid points
    points = points_3D[mask]
    pass
