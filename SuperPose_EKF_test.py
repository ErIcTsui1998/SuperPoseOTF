import numpy as np
import pickle
import os
import random
import yaml
import cv2
from utils import get_rigid_transform, GetPositionInBaseFrame, GetPositionInCameraFrame, QuaternionToRot, dvrk_DH_transformation
from utils_vision import PnPEstimation, GetROIFromKinematics, GetListOfLineEquationFromPointSet, IsPointAbovelineList, GetPointToLineDistanceList, VisibilityEvaluation, IsPointVisibleList
from dvrk_camera import dvrk_camera
from Jacobian import JacobianCalculatorImage, RotX, RotY, RotZ
from SuperPoseArmKinematics import dvrk_arm
from JCBB import JCBB
from EKF_Knownpairs import EKF_SuperDataSet
from AEKF_Knownpairs import AEKF_SuperDataSet
from EKF_MC_Knownpairs import EKF_MC_dVRKDataSet
from Illustration import DynamicDrawThreeLines

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
    
    # Read Keypoints manually labelled
    KP_Left_file = os.path.join(SubDataSet, "keypoints_left.yaml")
    KP_right_file = os.path.join(SubDataSet, "keypoints_right.yaml")
    KP_left_3d_ground_file = os.path.join(SubDataSet, "keypoints_left_3d.yaml")
    KP_labelled_left = {}
    KP_labelled_right = {}
    KP_left_3d_ground = {}
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

    with open(KP_left_3d_ground_file) as stream:
        try:
            KP_left_3d_ground = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # Camera Object Initialisation
    K_left = np.array([[1811.910046453570, 0.0, 588.5594517681759],
                       [0.0, 1809.640734154330, 477.3975900383616],
                       [0.0, 0.0, 1.0]])
    K_right = np.array([[1801.712669735144, 0.0, 791.7629609322958],
                        [0.0, 1796.928461921111, 437.4978636860555],
                        [0.0, 0.0, 1.0]])
    D_left = np.array([-0.251655177510111, 0.503352413478258, -0.002139555248137, -0.004349153536928, -0.246027939563351])
    D_right = np.array([-0.257090786509171, 0.101341249569555, 0.0007793893931081916, 0.0007405068673525044, 2.505085264989695])
    LEFT_CAM_PSM1 = dvrk_camera(K_left, PSM1.T_cr_his[0])
    LEFT_CAM_PSM3 = dvrk_camera(K_left, PSM3.T_cr_his[0])
    RIGHT_CAM_PSM1 = dvrk_camera(K_right, PSM1.T_cr_his[0])
    RIGHT_CAM_PSM3 = dvrk_camera(K_right, PSM3.T_cr_his[0])

    T_cr1 = PSM1.T_cr_his[0]
    T_cr3 = PSM3.T_cr_his[0]
    if "Tcr_psm1_1000.txt" in os.listdir(SubDataSet+"/HandEye"):
        os.chdir(SubDataSet+"/HandEye")
        T_cr1 = np.loadtxt("Tcr_psm1_1000.txt")
    if "Tcr_psm3_100.txt" in os.listdir(SubDataSet+"/HandEye"):
        os.chdir(SubDataSet+"/HandEye")
        T_cr3 = np.loadtxt("Tcr_psm3_100.txt")
    
    ##################### Key points Initialisation ################################
    KeyPointsName = ["rf","rb","rr","rl","pf","pb","pr","pl","ef","eb","gr","gl"]
    JointIndex = [1,2,3,4,5,6,7]
    KeyPointsRelDic = {"rf":np.array([-0.004, 0,-0.00625]), "rb":np.array([0.004, 0,-0.00625]), "rr":np.array([0, 0.004,0]), "rl":np.array([0, -0.004,0]),
                    "pf":np.array([0.00275, -0.00275, -0.00025]), "pb":np.array([0.00275, 0.00275, 0.00025]), "pr":np.array([0.0035, -0.0015, 0.003]), "pl":np.array([0.0035, 0.0015, -0.003]),
                    "ef":np.array([0.0, 0.0 ,-0.00275]), "eb":np.array([0.0, 0.0 ,0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsRelDic = {"rf":np.array([-0.004, 0.0, -0.0091]), "rb":np.array([0.004, 0.0, -0.0091]), "rr":np.array([0, 0.004,0]), "rl":np.array([0, -0.004,0]),
                    "pf":np.array([0.00275, -0.00275, -0.00025]), "pb":np.array([0.00275, 0.00275, 0.00025]), "pr":np.array([0.0035, -0.0015, 0.003]), "pl":np.array([0.0035, 0.0015, -0.003]),
                    "ef":np.array([0.0, 0.0 ,-0.00275]), "eb":np.array([0.0, 0.0 ,0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsJointDic = {"rf":4, "rb":4, "rr":4, "rl":4, "pf":5, "pb":5, "pr":5, "pl":5, "ef":6, "eb":6, "gr":6, "gl":6}
    LabelDic = {1:"rf", 2:"pf", 3:"ef", 4:"gl", 5:"gr", 6:"pl", 7:"rl", 8:"rf", 9:"pf", 10:"ef", 11:"gl", 12:"gr", 13:"pr", 14:"rr"}
    
    KeyPointsNamePSM1 = [item for item in KeyPointsName if "g" not in item] # everything excluding gripper tips, PSM1
    KeyPointsNamePSM3 = [item for item in KeyPointsName if "g" not in item] # everything excluding gripper tips, PSM3

    # JCBB Initialisation
    # LandmarkPSM1Name = ["rf","pf","ef","pl","rl"]
    # LandmarkPSM1Name = ["rf","rb","rr","rl","pf","pb", "pr","pl","ef","eb"]
    LandmarkPSM1Name = [item for item in KeyPointsName if "g" not in item]
    LandmarkPSM3Name = [item for item in KeyPointsName if "g" not in item]
    LandmarkPSM1Value = list(range(len(LandmarkPSM1Name)))
    LandmarkPSM3Value = list(range(len(LandmarkPSM3Name)))

    LandmarkPSM1Dic = dict(zip(LandmarkPSM1Name, LandmarkPSM1Value))
    LandmarkPSM3Dic = dict(zip(LandmarkPSM3Name, LandmarkPSM3Value))
    LandmarkPSM1DicInv = dict(zip(LandmarkPSM1Value, LandmarkPSM1Name))
    LandmarkPSM3DicInv = dict(zip(LandmarkPSM3Value, LandmarkPSM3Name))
    JCBB_obj1 = JCBB(LandmarkPSM1DicInv)
    JCBB_obj1_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    JCBB_obj1_cov_measure = np.array([50,50])

    JCBB_obj3 = JCBB(LandmarkPSM3DicInv)
    JCBB_obj3_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    JCBB_obj3_cov_measure = np.array([50,50])

    # EKF Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*1e-3
    cov_measure = np.array([50,50])
    EKF_obj1 = EKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr1)
    EKF_obj3 = EKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr3)

    # AEKF Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*7e-4
    cov_measure = np.array([10,10])
    forget_factor = 0.3
    AEKF_obj1 = AEKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr1, LandmarkPSM1Name, forget_factor)
    AEKF_obj3 = AEKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr3, LandmarkPSM3Name, forget_factor)

    # EKF MC Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.007, 0.007, 0.007, 0.25e-3, 0.25e-3, 0.25e-3])*6e-4
    cov_measure = np.array([20,20])
    EKF_MC_OBJ1 = EKF_MC_dVRKDataSet(mean_state, cov_state, cov_measure, T_cr1, LandmarkPSM1Name, bandwidth=15)
    EKF_MC_OBJ3 = EKF_MC_dVRKDataSet(mean_state, cov_state, cov_measure, T_cr3, LandmarkPSM3Name, bandwidth=15)

    color_pink = (255,141,161)
    color_blue = (255,0,0)
    color_yellow = (0,255,255)

    for index in range(n_images):
        img_name = "frame" + str(index) + ".png"
        os.chdir(LeftImagesFolder)
        img_left = cv2.imread(img_name)
        os.chdir(RightImagesFolder)
        img_right = cv2.imread(img_name)
        # KP_UV_LABELLED_NOW = KP_labelled_right[index]
        KP_UV_LABELLED_NOW = KP_labelled_left[index]
        KP_UV_3D_NOW = KP_left_3d_ground[index]
        ###############  start from PSM1 only ###########
        PSM1_js = PSM1.js_his[index]
        alpha = 0.0
        KeyPointsRelDic["gr"] = np.array([(9e-3)*np.sin(alpha/2) + (5e-4)*np.cos(alpha/2), (9e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        KeyPointsRelDic["gl"] = np.array([-(6.5e-3)*np.sin(alpha/2) - (5e-4)*np.cos(alpha/2), (6.5e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        
        KeyPointsPSM1Pos = [GetPositionInBaseFrame(PSM1_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsNamePSM1]
        KeyPointsPSM1PosDic = dict(zip(KeyPointsNamePSM1, KeyPointsPSM1Pos))
        JointPSM1Pos = [GetPositionInBaseFrame(PSM1_js, np.array([0,0,0]), i) for i in range(1,7)]
        KeyPointsPSM1PosCameraRight = LEFT_CAM_PSM1.GetPositionInCameraFrameList(KeyPointsPSM1Pos)
        JointPSM1PosCameraRight = LEFT_CAM_PSM1.GetPositionInCameraFrameList(JointPSM1Pos)
        KeyPointsPSM1Pixel = LEFT_CAM_PSM1.PixelProjectionList(KeyPointsPSM1PosCameraRight)
        gr_PSM1_pixel, gl_PSM1_pixel = KeyPointsPSM1Pixel[-2:]

        gm_PSM1rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        gm_PSM1Pos = GetPositionInBaseFrame(PSM1_js, gm_PSM1rel, 6)
        gm_PSM1CameraRight = LEFT_CAM_PSM1.GetPositionInCameraFrame(gm_PSM1Pos)
        gm_PSM1pixel = LEFT_CAM_PSM1.PixelProjection(gm_PSM1CameraRight)

        j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = LEFT_CAM_PSM1.PixelProjectionList(JointPSM1PosCameraRight)
        Edges_PSM1 = LEFT_CAM_PSM1.GetEdgeProjectionCylinder(4e-3, PSM1_js)
        SkeletonPt_PSM1_list = [j1_pixel, j4_pixel, j5_pixel, j6_pixel, gr_PSM1_pixel, gl_PSM1_pixel]
        SkeletonLine_PSM1_list = GetListOfLineEquationFromPointSet([(j1_pixel,j4_pixel),(j5_pixel,j6_pixel), (j6_pixel,gm_PSM1pixel)])

        overlay = LEFT_CAM_PSM1.DrawToolSkeleton(img_left, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel)])
        overlay = LEFT_CAM_PSM1.DrawLines(overlay, Edges_PSM1)

        ########################### Visibility Test & Scores ############################################
        roll_part_evaluation = IsPointAbovelineList(KeyPointsPSM1Pixel[:4], SkeletonLine_PSM1_list[0])
        roll_part_central_distance = GetPointToLineDistanceList(KeyPointsPSM1Pixel[:4], SkeletonLine_PSM1_list[0])
        
        pe_part_evaluation = IsPointAbovelineList(KeyPointsPSM1Pixel[4:10], SkeletonLine_PSM1_list[1])
        pe_part_central_distance = GetPointToLineDistanceList(KeyPointsPSM1Pixel[4:10], SkeletonLine_PSM1_list[1])

        overall_evaluation = roll_part_evaluation + pe_part_evaluation
        overall_central_distance = roll_part_central_distance + pe_part_central_distance

        # grip_part_evaluation = IsPointAbovelineList([gr_pixel, gl_pixel], SkeletonLine_list[2])
        # grip_part_central_distance = GetPointToLineDistanceList([gr_pixel, gl_pixel], SkeletonLine_list[2])        
        # overall_evaluation = roll_part_evaluation + pe_part_evaluation + grip_part_evaluation
        # overall_central_distance = roll_part_central_distance + pe_part_central_distance + grip_part_central_distance

        Top_distance_list = GetPointToLineDistanceList(KeyPointsPSM1Pixel, Edges_PSM1[1])
        Down_distance_list = GetPointToLineDistanceList(KeyPointsPSM1Pixel, Edges_PSM1[0])
        Side_distance_list = [min(i,j) for i, j in zip(Top_distance_list, Down_distance_list)]

        visibility_score_dic = {}
        visibility_score = VisibilityEvaluation(overall_evaluation, overall_central_distance, Side_distance_list, KeyPointsNamePSM1)
        if len(visibility_score) > 0:
            visibility_score = [round(visibility_score[i],2) for i in range(len(KeyPointsPSM1Pixel))]
            visibility_score_dic = dict(zip(LandmarkPSM1Value, visibility_score))
        #########################################################################################################################

        # Draw keypoints and labels
        # KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key < 7 and key != 4 and key !=5}
        KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key <= 7 and key != 4 and key !=5}
        UV_KP_NAMES_PSM1 = list(KP_UV_LABELLED_PSM1.keys())
        UV_KP_NAMES_PSM1 = [str(item) for item in UV_KP_NAMES_PSM1]
        UV_KP_PIXELS_PSM1 = list(KP_UV_LABELLED_PSM1.values())
        UV_KP_PIXELS_PSM1 = [tuple(item) for item in UV_KP_PIXELS_PSM1]

        KP_UV_3D_PSM1 = {LabelDic[key]: value for key, value in KP_UV_3D_NOW.items() if value != None and key <= 7 and key != 4 and key !=5}
        UV_KP_3D_PSM1 = list(KP_UV_3D_PSM1.values())

        overlay = LEFT_CAM_PSM1.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM1, text_list=UV_KP_NAMES_PSM1, color = color_blue)
        overlay = LEFT_CAM_PSM1.DrawKeyPointsList(overlay, KeyPointsPSM1Pixel, text_list=KeyPointsNamePSM1)

        # JCBB data association using Jacobian 
        JacobianValues_PSM1 = [JacobianCalculatorImage(K_left, np.zeros(3), np.zeros(3), T_cr1, KeyPointsPSM1PosDic[name]) for name in KeyPointsNamePSM1]
        JacobiansInput_PSM1 = dict(zip(LandmarkPSM1Value, JacobianValues_PSM1))
        JCBB_obj1.ReadPredictedFeatureValues(KeyPointsPSM1Pixel, JacobiansInput_PSM1, JCBB_obj1_cov_state)
        # JCBB_obj1.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM1, JCBB_obj1_cov_measure)
        JCBB_obj1.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM1, JCBB_obj1_cov_measure, visibility_score_dic)
        OutputMatchedKeys_PSM1 = JCBB_obj1.ReturnMatchingKeys()
        # OutputMatchedKeys = [LandmarkPSM1Dic[name] for name in UV_KP_NAMES_PSM1 if name in LandmarkPSM1Dic.keys()]
        JCBB_obj1.Clear()
        overlay = LEFT_CAM_PSM1.DrawKeyPointsAssociation(overlay, UV_KP_PIXELS_PSM1, KeyPointsPSM1Pixel, OutputMatchedKeys_PSM1, color_blue)

        ##################################### EKF/PF with known data associations block ################################################
        KeyPointsPSM1PixelDic = dict(zip(KeyPointsNamePSM1, KeyPointsPSM1Pixel))
        IndexMatched_PSM1 = [i for i,j in enumerate(OutputMatchedKeys_PSM1) if j is not None]
        MatchedNamePSM1List = [LandmarkPSM1DicInv[key] for key in OutputMatchedKeys_PSM1 if key is not None]
        MatchedValuePSM1List = [UV_KP_PIXELS_PSM1[index] for index in IndexMatched_PSM1]
        MatchedMeasurementPSM1Dict = dict(zip(MatchedNamePSM1List, MatchedValuePSM1List))
        JacobiansPSM1Dict = dict(zip(KeyPointsNamePSM1, JacobianValues_PSM1))
        
        EKF_obj1.EKFReadMeasurement(KeyPointsPSM1PixelDic, MatchedMeasurementPSM1Dict, JacobiansPSM1Dict)
        T_cr1_new = EKF_obj1.ReturnTcrEstimation()
        LEFT_CAM_PSM1.UpdateTcr(T_cr1_new)

        # AEKF_obj1.AEKFReadMeasurement(KeyPointsPSM1PosDic, MatchedMeasurementPSM1Dict, JacobiansPSM1Dict, K_left)
        # T_cr1_new = AEKF_obj1.ReturnTcrEstimation()
        # mean_state_PSM1, cov_state_PSM1 = AEKF_obj1.ReturnStateEstimation()
        # LEFT_CAM_PSM1.UpdateTcr(T_cr1_new)

        # EKF_MC_OBJ1.EKFReadMeasurement(KeyPointsPSM1PixelDic, MatchedMeasurementPSM1Dict, JacobiansPSM1Dict)
        # T_cr1_new = EKF_MC_OBJ1.ReturnTcrEstimation()
        # LEFT_CAM_PSM1.UpdateTcr(T_cr1_new)

        ######################### Error Analysis 3d ############################################
        KP_3D_CALIBRATED_LIST = LEFT_CAM_PSM1.GetPositionInCameraFrameList(KeyPointsPSM1Pos)
        KP_3D_CALIBRATED_SELECTED = [KP_3D_CALIBRATED_LIST[key] for key in OutputMatchedKeys_PSM1 if key != None]
        KP_3D_RECONSTRUCTED_SELECTED = [UV_KP_3D_PSM1[id_match] for id_match in IndexMatched_PSM1 if id_match != None]
        KP_3D_ERROR_LIST = [1e3*np.linalg.norm(KP_3D_CALIBRATED_SELECTED[i] - KP_3D_RECONSTRUCTED_SELECTED[i]) for i in range(len(KP_3D_CALIBRATED_SELECTED))]
        KP_3D_ERROR_DIC = dict(zip(MatchedNamePSM1List, KP_3D_ERROR_LIST))

        # ##################### For PSM3 now ###########################################################################################
        # PSM3_js = PSM3.js_his[index]
        # KeyPointsPSM3Pos = [GetPositionInBaseFrame(PSM3_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsNamePSM3]
        # KeyPointsPSM3PosDic = dict(zip(KeyPointsNamePSM3, KeyPointsPSM3Pos))
        # JointPSM3Pos = [GetPositionInBaseFrame(PSM3_js, np.array([0,0,0]), i) for i in range(1,7)]
        # KeyPointsPSM3PosCameraRight = LEFT_CAM_PSM3.GetPositionInCameraFrameList(KeyPointsPSM3Pos)
        # JointPosPSM3CameraRight = LEFT_CAM_PSM3.GetPositionInCameraFrameList(JointPSM3Pos)
        # KeyPointsPSM3Pixel = LEFT_CAM_PSM3.PixelProjectionList(KeyPointsPSM3PosCameraRight, D_left)
        # gr_PSM3_pixel, gl_PSM3_pixel = KeyPointsPSM3Pixel[-2:]

        # gm_PSM3rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        # gm_PSM3Pos = GetPositionInBaseFrame(PSM3_js, gm_PSM3rel, 6)
        # gm_PSM3CameraRight = LEFT_CAM_PSM3.GetPositionInCameraFrame(gm_PSM3Pos)
        # gm_PSM3pixel = LEFT_CAM_PSM3.PixelProjection(gm_PSM3CameraRight, D_left)

        # j1_PSM3_pixel, j2_PSM3_pixel, j3_PSM3_pixel, j4_PSM3_pixel, j5_PSM3_pixel, j6_PSM3_pixel = LEFT_CAM_PSM3.PixelProjectionList(JointPosPSM3CameraRight, D_left)
        # Edges_PSM3 = LEFT_CAM_PSM3.GetEdgeProjectionCylinder(4e-3, PSM3_js)
        # SkeletonPt_PSM3_list = [j3_PSM3_pixel, j4_PSM3_pixel, j5_PSM3_pixel, j6_PSM3_pixel, gr_PSM3_pixel, gl_PSM3_pixel]
        # SkeletonLine_PSM3_list = GetListOfLineEquationFromPointSet([(j3_PSM3_pixel,j4_PSM3_pixel),(j5_PSM3_pixel,j6_PSM3_pixel), (j6_PSM3_pixel,gm_PSM3pixel)])

        # overlay = LEFT_CAM_PSM3.DrawToolSkeleton(img_left, [(j3_PSM3_pixel,j4_PSM3_pixel), (j5_PSM3_pixel, j6_PSM3_pixel)], color=color_yellow)
        # overlay = LEFT_CAM_PSM3.DrawLines(overlay, Edges_PSM3, color=color_yellow)

        # ########################### Visibility Test & Scores ############################################
        # roll_part_evaluation = IsPointAbovelineList(KeyPointsPSM3Pixel[:4], SkeletonLine_PSM3_list[0])
        # roll_part_central_distance = GetPointToLineDistanceList(KeyPointsPSM3Pixel[:4], SkeletonLine_PSM3_list[0])
        
        # pe_part_evaluation = IsPointAbovelineList(KeyPointsPSM3Pixel[4:10], SkeletonLine_PSM3_list[1])
        # pe_part_central_distance = GetPointToLineDistanceList(KeyPointsPSM3Pixel[4:10], SkeletonLine_PSM3_list[1])

        # overall_evaluation = roll_part_evaluation + pe_part_evaluation
        # overall_central_distance = roll_part_central_distance + pe_part_central_distance

        # # grip_part_evaluation = IsPointAbovelineList([gr_pixel, gl_pixel], SkeletonLine_list[2])
        # # grip_part_central_distance = GetPointToLineDistanceList([gr_pixel, gl_pixel], SkeletonLine_list[2])        
        # # overall_evaluation = roll_part_evaluation + pe_part_evaluation + grip_part_evaluation
        # # overall_central_distance = roll_part_central_distance + pe_part_central_distance + grip_part_central_distance

        # Top_distance_list = GetPointToLineDistanceList(KeyPointsPSM3Pixel, Edges_PSM3[1])
        # Down_distance_list = GetPointToLineDistanceList(KeyPointsPSM3Pixel, Edges_PSM3[0])
        # Side_distance_list = [min(i,j) for i, j in zip(Top_distance_list, Down_distance_list)]

        # visibility_score_dic = {}
        # visibility_score = VisibilityEvaluation(overall_evaluation, overall_central_distance, Side_distance_list, KeyPointsNamePSM3)
        # if len(visibility_score) > 0:
        #     visibility_score = [round(visibility_score[i],2) for i in range(len(KeyPointsPSM3Pixel))]
        #     visibility_score_dic = dict(zip(LandmarkPSM3Value, visibility_score))
        # #########################################################################################################################

        # # Draw keypoints and labels
        # KP_UV_LABELLED_PSM3 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key > 7 and key != 11 and key !=12}
        # UV_KP_NAMES_PSM3 = list(KP_UV_LABELLED_PSM3.keys())
        # UV_KP_NAMES_PSM3 = [str(item) for item in UV_KP_NAMES_PSM3]
        # UV_KP_PIXELS_PSM3 = list(KP_UV_LABELLED_PSM3.values())
        # UV_KP_PIXELS_PSM3 = [tuple(item) for item in UV_KP_PIXELS_PSM3]
        
        # KP_UV_3D_PSM3 = {LabelDic[key]: value for key, value in KP_UV_3D_NOW.items() if value != None and key > 7 and key != 11 and key !=12}
        # UV_KP_3D_PSM3 = list(KP_UV_3D_PSM3.values())

        # overlay = LEFT_CAM_PSM3.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM3, text_list=UV_KP_NAMES_PSM3, color = color_blue)
        # overlay = LEFT_CAM_PSM3.DrawKeyPointsList(overlay, KeyPointsPSM3Pixel, text_list=KeyPointsNamePSM3)

        # # JCBB data association using Jacobian 
        # JacobianValues_PSM3 = [JacobianCalculatorImage(K_left, np.zeros(3), np.zeros(3), T_cr3, KeyPointsPSM3PosDic[name]) for name in KeyPointsNamePSM3]
        # JacobiansInput_PSM3 = dict(zip(LandmarkPSM3Value, JacobianValues_PSM3))
        # JCBB_obj3.ReadPredictedFeatureValues(KeyPointsPSM3Pixel, JacobiansInput_PSM3, JCBB_obj3_cov_state)
        # # JCBB_obj3.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM3, JCBB_obj3_cov_measure, visibility_score_dic)
        # JCBB_obj3.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM3, JCBB_obj3_cov_measure)
        # OutputMatchedKeys_PSM3 = JCBB_obj3.ReturnMatchingKeys()
        # # OutputMatchedKeys_PSM3 = [LandmarkPSM3Dic[name] for name in UV_KP_NAMES_PSM3 if name in LandmarkPSM3Dic.keys()]
        # JCBB_obj3.Clear()
        # overlay = LEFT_CAM_PSM3.DrawKeyPointsAssociation(overlay, UV_KP_PIXELS_PSM3, KeyPointsPSM3Pixel, OutputMatchedKeys_PSM3, color_blue)

        # ##################################### EKF/PF with known data associations block ################################################
        # KeyPointsPSM3PixelDic = dict(zip(KeyPointsNamePSM3, KeyPointsPSM3Pixel))
        # IndexMatched_PSM3 = [i for i,j in enumerate(OutputMatchedKeys_PSM3) if j is not None]
        # MatchedNamePSM3List = [LandmarkPSM3DicInv[key] for key in OutputMatchedKeys_PSM3 if key is not None]
        # MatchedValuePSM3List = [UV_KP_PIXELS_PSM3[index] for index in IndexMatched_PSM3]
        # MatchedMeasurementPSM3Dict = dict(zip(MatchedNamePSM3List, MatchedValuePSM3List))
        # JacobiansPSM3Dict = dict(zip(KeyPointsNamePSM3, JacobianValues_PSM3))
        
        # EKF_obj3.EKFReadMeasurement(KeyPointsPSM3PixelDic, MatchedMeasurementPSM3Dict, JacobiansPSM3Dict)
        # T_cr3_new = EKF_obj3.ReturnTcrEstimation()
        # LEFT_CAM_PSM3.UpdateTcr(T_cr3_new)

        # # AEKF_obj3.AEKFReadMeasurement(KeyPointsPSM3PosDic, MatchedMeasurementPSM3Dict, JacobiansPSM3Dict, K_left)
        # # T_cr3_new = AEKF_obj3.ReturnTcrEstimation()
        # # mean_state_PSM3, cov_state_PSM3 = AEKF_obj3.ReturnStateEstimation()
        # # LEFT_CAM_PSM3.UpdateTcr(T_cr3_new)

        # # EKF_MC_OBJ3.EKFReadMeasurement(KeyPointsPSM3PixelDic, MatchedMeasurementPSM3Dict, JacobiansPSM3Dict)
        # # T_cr3_new = EKF_MC_OBJ3.ReturnTcrEstimation()
        # # LEFT_CAM_PSM3.UpdateTcr(T_cr3_new)

        # ######################### Error Analysis 3d ############################################
        # KP_3D_CALIBRATED_LIST = LEFT_CAM_PSM3.GetPositionInCameraFrameList(KeyPointsPSM3Pos)
        # KP_3D_CALIBRATED_SELECTED = [KP_3D_CALIBRATED_LIST[key] for key in OutputMatchedKeys_PSM3 if key != None]
        # KP_3D_RECONSTRUCTED_SELECTED = [UV_KP_3D_PSM3[id_match] for id_match in IndexMatched_PSM3 if id_match != None]
        # KP_3D_ERROR_LIST = [1e3*np.linalg.norm(KP_3D_CALIBRATED_SELECTED[i] - KP_3D_RECONSTRUCTED_SELECTED[i]) for i in range(len(KP_3D_CALIBRATED_SELECTED))]
        # KP_3D_ERROR_DIC = dict(zip(MatchedNamePSM3List, KP_3D_ERROR_LIST))

        cv2.imshow("overlay", overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


