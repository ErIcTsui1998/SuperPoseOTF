import numpy as np
import pickle
import os
import random
import cv2
from utils import get_rigid_transform, GetPositionInBaseFrame, GetPositionInCameraFrame, QuaternionToRot, dvrk_DH_transformation
from dvrk_camera import dvrk_camera
from KeyPointsDetector import KeyPointsDetector
from Jacobian import JacobianCalculatorImage, RotX, RotY, RotZ
from SuperPose_Kinematics import dvrk_arm
from JCBB import JCBB
from EKF_Knownpairs import EKF_SuperDataSet
from AEKF_Knownpairs import AEKF_SuperDataSet
from EKF_MC_Knownpairs import EKF_MC_dVRKDataSet

if __name__ == "__main__":    
    BaseFolder = "/home/zc519/Downloads/SurgPoseDataSet"
    dir_id = "000000"
    ArmNameList = ['PSM1','PSM3']
    PSM1 = dvrk_arm()
    PSM3 = dvrk_arm()
    SubDataSet = os.path.join(BaseFolder, dir_id)
    LeftImagesFolder = os.path.join(SubDataSet, "regular/LeftImages")
    LeftImages = os.listdir(LeftImagesFolder)
    RightImagesFolder = os.path.join(SubDataSet, "regular/RightImages")
    RightImages = os.listdir(RightImagesFolder)
    assert len(LeftImages) == len(RightImages), "left and right images do not match in number"
    n_images = len(LeftImages)

    # Read Kinematics data for PSM1 and PSM3
    for name in ArmNameList:
        KinFolder = os.path.join(SubDataSet, "Kinematics", name)
        os.chdir(KinFolder)
        js_data = np.loadtxt(name+"_jp_his.txt")
        cp_t_data = np.loadtxt(name+"_cp_t_his.txt")
        cp_R_data = np.loadtxt(name+"_cp_R_his.txt")
        if name == "PSM1":
            PSM1.ReadInputs(js_data, cp_t_data, cp_R_data)
        elif name == "PSM3":
            PSM3.ReadInputs(js_data, cp_t_data, cp_R_data)
    
    # Camera Object Initialisation
    K_left = np.array([[1811.910046453570, 0.0, 588.5594517681759],
                       [0.0, 1809.640734154330, 477.3975900383616],
                       [0.0, 0.0, 1.0]])
    K_right = np.array([[1801.712669735144, 0.0, 791.7629609322958],
                        [0.0, 1796.928461921111, 437.4978636860555],
                        [0.0, 0.0, 1.0]])

    LEFT_CAM_PSM1 = dvrk_camera(K_left, PSM1.cp_R_his[0])
    LEFT_CAM_PSM3 = dvrk_camera(K_left, PSM3.cp_R_his[0])
    T_cr1 = PSM1.T_cr_his[0]
    T_cr2 = PSM3.T_cr_his[0]

    print(0)