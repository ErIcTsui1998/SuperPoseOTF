import numpy as np
import pickle
import os
import random
import cv2
from utils import get_rigid_transform, GetPositionInBaseFrame, GetPositionInCameraFrame, QuaternionToRot, dvrk_DH_transformation
from dvrk_camera import dvrk_camera
from Jacobian import JacobianCalculatorImage, RotX, RotY, RotZ
from SuperPoseArmKinematics import dvrk_arm
from JCBB import JCBB
from EKF_Knownpairs import EKF_SuperDataSet
from AEKF_Knownpairs import AEKF_SuperDataSet
from EKF_MC_Knownpairs import EKF_MC_dVRKDataSet

if __name__ == "__main__":    
    BaseFolder = "/home/zc519/Downloads/SurgPoseDataSet"
    dir_id = "000007"
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

    LEFT_CAM_PSM1 = dvrk_camera(K_left, PSM1.T_cr_his[0])
    LEFT_CAM_PSM3 = dvrk_camera(K_left, PSM3.T_cr_his[0])
    RIGHT_CAM_PSM1 = dvrk_camera(K_right, PSM1.T_cr_his[0])
    RIGHT_CAM_PSM3 = dvrk_camera(K_right, PSM3.T_cr_his[0])

    T_cr1 = PSM1.T_cr_his[0]
    T_cr2 = PSM3.T_cr_his[0]

    ##################### Key points Initialisation ################################
    KeyPointsName = ["rf","rb","rr","rl","pf","pb","pr","pl","ef","eb","gr","gl"]
    JointIndex = [1,2,3,4,5,6,7]
    KeyPointsRelDic = {"rf":np.array([-0.004, 0,-0.00625]), "rb":np.array([0.004, 0,-0.00625]), "rr":np.array([0, 0.004,0]), "rl":np.array([0, -0.004,0]),
                    "pf":np.array([0.00275, -0.00275, -0.00025]), "pb":np.array([0.00275, 0.00275, 0.00025]), "pr":np.array([0.0035, -0.0015, 0.003]), "pl":np.array([0.0035, 0.0015, -0.003]),
                    "ef":np.array([0.0, 0.0 ,-0.00275]), "eb":np.array([0.0, 0.0 ,0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsJointDic = {"rf":4, "rb":4, "rr":4, "rl":4, "pf":5, "pb":5, "pr":5, "pl":5, "ef":6, "eb":6, "gr":6, "gl":6}

    # JCBB Initialisation
    LandmarkName = ["rf","rb","rr","rl","pf","pb", "pr","pl","ef","eb"]
    # LandmarkValue = [1,2,3,4,5,6,7,8,9,10] # Outliers are denoted 0
    LandmarkValue = [0,1,2,3,4,5,6,7,8,9] # Outliers are denoted 0

    LandmarkDic = dict(zip(LandmarkName, LandmarkValue))
    LandmarkDicInv = dict(zip(LandmarkValue, LandmarkName))
    JCBB_obj1 = JCBB(LandmarkDicInv)


    for index in range(n_images):
        img_name = "frame" + str(index) + ".png"
        os.chdir(LeftImagesFolder)
        img_left = cv2.imread(img_name)
        os.chdir(RightImagesFolder)
        img_right = cv2.imread(img_name)

        ###############  start from PSM1 only ###########
        PSM1_js = PSM1.js_his[index]
        alpha = 0.0
        KeyPointsRelDic["gr"] = np.array([(9e-3)*np.sin(alpha/2) + (5e-4)*np.cos(alpha/2), (9e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        KeyPointsRelDic["gl"] = np.array([-(6.5e-3)*np.sin(alpha/2) - (5e-4)*np.cos(alpha/2), (6.5e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        KeyPointsPosPSM = [GetPositionInBaseFrame(PSM1_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsName]
        KeyPointsPosPSMDic = dict(zip(KeyPointsName, KeyPointsPosPSM))
        JointPosPSM = [GetPositionInBaseFrame(PSM1_js, np.array([0,0,0]), i) for i in range(1,7)]
        KeyPointsPosCameraRight = RIGHT_CAM_PSM1.GetPositionInCameraFrameList(KeyPointsPosPSM)
        JointPosCameraRight = RIGHT_CAM_PSM1.GetPositionInCameraFrameList(JointPosPSM)
        KeyPointsPixel = RIGHT_CAM_PSM1.PixelProjectionList(KeyPointsPosCameraRight)
        gr_pixel, gl_pixel = KeyPointsPixel[-2:]

        gm_rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        gm_PosPSM = GetPositionInBaseFrame(PSM1_js, gm_rel, 6)
        gm_CameraRight = RIGHT_CAM_PSM1.GetPositionInCameraFrame(gm_PosPSM)
        gm_pixel = RIGHT_CAM_PSM1.PixelProjection(gm_CameraRight)

        j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = RIGHT_CAM_PSM1.PixelProjectionList(JointPosCameraRight)
        Edges = RIGHT_CAM_PSM1.GetEdgeProjectionCylinder(4e-3, PSM1_js)
        SkeletonPt_list = [j1_pixel, j4_pixel, j5_pixel, j6_pixel, gr_pixel, gl_pixel]

        overlay = RIGHT_CAM_PSM1.DrawToolSkeleton(img_right, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gr_pixel), (j6_pixel, gl_pixel)])
        overlay = RIGHT_CAM_PSM1.DrawLines(overlay, Edges)

        ##################### For PSM3 now
        PSM3_js = PSM3.js_his[index]
        KeyPointsPosPSM3 = [GetPositionInBaseFrame(PSM3_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsName]
        KeyPointsPosPSM3Dic = dict(zip(KeyPointsName, KeyPointsPosPSM3))
        JointPosPSM3 = [GetPositionInBaseFrame(PSM3_js, np.array([0,0,0]), i) for i in range(1,7)]
        KeyPointsPosCameraRight_PSM3 = RIGHT_CAM_PSM3.GetPositionInCameraFrameList(KeyPointsPosPSM3)
        JointPosCameraRight_PSM3 = RIGHT_CAM_PSM3.GetPositionInCameraFrameList(JointPosPSM3)
        KeyPointsPixel_PSM3 = RIGHT_CAM_PSM3.PixelProjectionList(KeyPointsPosCameraRight_PSM3)
        gr_pixel, gl_pixel = KeyPointsPixel_PSM3[-2:]

        gm_rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        gm_PosPSM = GetPositionInBaseFrame(PSM3_js, gm_rel, 6)
        gm_CameraRight = RIGHT_CAM_PSM3.GetPositionInCameraFrame(gm_PosPSM)
        gm_pixel = RIGHT_CAM_PSM3.PixelProjection(gm_CameraRight)

        j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = RIGHT_CAM_PSM3.PixelProjectionList(JointPosCameraRight_PSM3)
        Edges_PSM3 = RIGHT_CAM_PSM3.GetEdgeProjectionCylinder(4e-3, PSM3_js)
        SkeletonPt_list = [j1_pixel, j4_pixel, j5_pixel, j6_pixel, gr_pixel, gl_pixel]

        overlay = RIGHT_CAM_PSM3.DrawToolSkeleton(overlay, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gr_pixel), (j6_pixel, gl_pixel)], color=(255,0,0))
        overlay = RIGHT_CAM_PSM3.DrawLines(overlay, Edges_PSM3, color=(255,0,0))

        cv2.imshow("overlay", overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


