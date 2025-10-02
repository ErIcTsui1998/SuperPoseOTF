import numpy as np
import cv2
import os
import yaml
from scipy.optimize import least_squares

# Camera Object Initialisation
K_left = np.array([[1811.910046453570, 0.0, 588.5594517681759],
                    [0.0, 1809.640734154330, 477.3975900383616],
                    [0.0, 0.0, 1.0]])
K_right = np.array([[1801.712669735144, 0.0, 791.7629609322958],
                    [0.0, 1796.928461921111, 437.4978636860555],
                    [0.0, 0.0, 1.0]])
D_left = np.array([-0.251655177510111, 0.503352413478258, -0.002139555248137, -0.004349153536928, -0.246027939563351])
D_right = np.array([-0.257090786509171, 0.101341249569555, 0.0007793893931081916, 0.0007405068673525044, 2.505085264989695])

# D_left = np.zeros(5)
# D_right = np.zeros(5)

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
T = np.array([[T_0, T_1, T_2]],dtype=np.float64).T

img_size = (1400,986)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, img_size, R, T)

if __name__ == "__main__":   
    BaseFolder = "/home/zc519/Downloads/SurgPoseDataSet"
    os.chdir(BaseFolder)
    Subdir = os.listdir(BaseFolder)
    Subdir.sort()
    n_subdir = len(Subdir)
    for i in range(n_subdir):
        current_subdir = os.path.join(BaseFolder, Subdir[i]) 
        print(f"Current directory is {current_subdir}")
        os.chdir(current_subdir)
        KP_Left_file = "keypoints_left.yaml"
        KP_right_file = "keypoints_right.yaml"

        KP_labelled_left = {}
        KP_labelled_right = {}

        with open(KP_Left_file) as stream:
            try:
                KP_labelled_left = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        with open(KP_right_file) as stream:
            try:
                KP_labelled_right = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        assert len(KP_labelled_left) == len(KP_labelled_right), "number of labelled keypoints mismatch"
        n_frames = len(KP_labelled_left)
        KP_pos_3d_left_dic = {}

        for index in range(n_frames):
            keys_L = [key for key, val in KP_labelled_left[index].items() if val != None]
            keys_R = [key for key, val in KP_labelled_right[index].items() if val != None]
            keys_common = list(set(keys_L) & set(keys_R))

            ptsL = np.array([KP_labelled_left[index][name] for name in keys_common], dtype=np.float64)
            ptsR = np.array([KP_labelled_right[index][name] for name in keys_common], dtype=np.float64)

            ptsL = ptsL.reshape(-1, 1, 2)
            ptsR = ptsR.reshape(-1, 1, 2)
            
            ptsL_ud = cv2.undistortPoints(ptsL, K_left, D_left, R=R1, P=P1)
            ptsR_ud = cv2.undistortPoints(ptsR, K_right, D_right, R=R2, P=P2)

            pts4D = cv2.triangulatePoints(P1, P2, ptsL_ud, ptsR_ud)
            pts3D_init = (pts4D[:3] / pts4D[3]).T  # Nx3 array

            assert len(keys_common) == len(pts3D_init), "number of detected keys mismatch"
            n_keys = len(keys_common)
            PosDic = {}
            for j in range(n_keys):
               kp_key = keys_common[j]
               x, y, z = R1.T @ pts3D_init[j] * 1e-3 # unit (m)
               PosDic[kp_key] = [float(x), float(y), float(z)]
            KP_pos_3d_left_dic[index] = PosDic

        # Time to record 3d keypoints (left frame reference)    
        output_name = os.path.join(current_subdir, "keypoints_left_3d.yaml")
        with open(output_name, "w") as f:
            yaml.dump(KP_pos_3d_left_dic, f, default_flow_style=False, sort_keys=False)
        print(f"{Subdir[i]} done!")
        