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
    dir_id = "000031"

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
    if "Tcr_psm1_800.txt" in os.listdir(SubDataSet+"/HandEye"):
        os.chdir(SubDataSet+"/HandEye")
        T_cr1 = np.loadtxt("Tcr_psm1_800.txt")

    ##################### Key points Initialisation ################################
    KeyPointsName = ["rf","rb","rr","rl","pf","pb","pr","pl","ef","eb","gr","gl"]
    JointIndex = [1,2,3,4,5,6,7]
    KeyPointsRelDic = {"rf":np.array([-0.004, 0,-0.00625]), "rb":np.array([0.004, 0,-0.00625]), "rr":np.array([0, 0.004,0]), "rl":np.array([0, -0.004,0]),
                    "pf":np.array([0.00275, -0.00275, -0.00025]), "pb":np.array([0.00275, 0.00275, 0.00025]), "pr":np.array([0.0035, -0.0015, 0.003]), "pl":np.array([0.0035, 0.0015, -0.003]),
                    "ef":np.array([0.0, 0.0 ,-0.00275]), "eb":np.array([0.0, 0.0 ,0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsJointDic = {"rf":4, "rb":4, "rr":4, "rl":4, "pf":5, "pb":5, "pr":5, "pl":5, "ef":6, "eb":6, "gr":6, "gl":6}
    LabelDic = {1:"rf", 2:"pf", 3:"ef", 4:"gl", 5:"gr", 6:"pl", 7:"rl", 8:"rf", 9:"pf", 10:"ef", 11:"gl", 12:"gr", 13:"pr", 14:"rr"}
    
    KeyPointsNamePSM1 = [item for item in KeyPointsName] # everything excluding gripper tips, PSM1
    KeyPointsNamePSM3 = [item for item in KeyPointsName] # everything excluding gripper tips, PSM3

    # JCBB Initialisation
    # LandmarkPSM1Name = ["rf","pf","ef","pl","rl"]
    # LandmarkPSM1Name = ["rf","rb","rr","rl","pf","pb", "pr","pl","ef","eb"]
    LandmarkPSM1Name = KeyPointsName
    LandmarkPSM1Value = list(range(len(LandmarkPSM1Name)))

    LandmarkPSM1Dic = dict(zip(LandmarkPSM1Name, LandmarkPSM1Value))
    LandmarkPSM1DicInv = dict(zip(LandmarkPSM1Value, LandmarkPSM1Name))
    JCBB_obj1 = JCBB(LandmarkPSM1DicInv)
    JCBB_obj1_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    JCBB_obj1_cov_measure = np.array([50,50])

    # EKF Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*1e-3
    cov_measure = np.array([50,50])
    EKF_obj1 = EKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr1)

    # AEKF Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*15e-4
    cov_measure = np.array([5,5])
    forget_factor = 0.3
    AEKF_obj1 = AEKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr1, LandmarkPSM1Name, forget_factor)

    # EKF MC Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.007, 0.007, 0.007, 0.25e-3, 0.25e-3, 0.25e-3])*1e-4
    cov_measure = np.array([20,20])
    EKF_MC_OBJ1 = EKF_MC_dVRKDataSet(mean_state, cov_state, cov_measure, T_cr1, LandmarkPSM1Name, bandwidth=15)

    color_pink = (255,141,161)
    color_blue = (255,0,0)
    color_yellow = (0,255,255)
    PNP_PIXEL_LIST = []
    PNP_OBJ_PTS_LIST = []

    # State variables evolution
    r_vec_his = [cv2.Rodrigues(T_cr1[:3,:3])[0].reshape(1,-1)[0]]
    t_vec_his = [T_cr1[:3,-1]]

    for index in range(n_images):
        img_name = "frame" + str(index) + ".png"
        os.chdir(LeftImagesFolder)
        img_left = cv2.imread(img_name)
        os.chdir(RightImagesFolder)
        img_right = cv2.imread(img_name)
        KP_UV_LABELLED_NOW = KP_labelled_right[index]

        ###############  start from PSM1 only ###########
        PSM1_js = PSM1.js_his[index]
        alpha = 2*np.pi/180
        KeyPointsRelDic["gr"] = np.array([(9e-3)*np.sin(alpha/2) + (5e-4)*np.cos(alpha/2), (9e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        KeyPointsRelDic["gl"] = np.array([-(6.5e-3)*np.sin(alpha/2) - (5e-4)*np.cos(alpha/2), (6.5e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        
        KeyPointsPosPSM = [GetPositionInBaseFrame(PSM1_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsNamePSM1]
        KeyPointsPosPSMDic = dict(zip(KeyPointsNamePSM1, KeyPointsPosPSM))
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
        SkeletonLine_list = GetListOfLineEquationFromPointSet([(j1_pixel,j4_pixel),(j5_pixel,j6_pixel), (j6_pixel,gm_pixel)])

        overlay = RIGHT_CAM_PSM1.DrawToolSkeleton(img_right, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel)])
        overlay = RIGHT_CAM_PSM1.DrawLines(overlay, Edges)

        ########################### Visibility Test & Scores ############################################
        roll_part_evaluation = IsPointAbovelineList(KeyPointsPixel[:4], SkeletonLine_list[0])
        roll_part_central_distance = GetPointToLineDistanceList(KeyPointsPixel[:4], SkeletonLine_list[0])
        
        pe_part_evaluation = IsPointAbovelineList(KeyPointsPixel[4:10], SkeletonLine_list[1])
        pe_part_central_distance = GetPointToLineDistanceList(KeyPointsPixel[4:10], SkeletonLine_list[1])

        # overall_evaluation = roll_part_evaluation + pe_part_evaluation
        # overall_central_distance = roll_part_central_distance + pe_part_central_distance

        grip_part_evaluation = IsPointAbovelineList([gr_pixel, gl_pixel], SkeletonLine_list[2])
        grip_part_central_distance = GetPointToLineDistanceList([gr_pixel, gl_pixel], SkeletonLine_list[2])   

        overall_evaluation = roll_part_evaluation + pe_part_evaluation + grip_part_evaluation
        overall_central_distance = roll_part_central_distance + pe_part_central_distance + grip_part_central_distance

        Top_distance_list = GetPointToLineDistanceList(KeyPointsPixel, Edges[1])
        Down_distance_list = GetPointToLineDistanceList(KeyPointsPixel, Edges[0])
        Side_distance_list = [min(i,j) for i, j in zip(Top_distance_list, Down_distance_list)]

        visibility_score_dic = {}
        visibility_score = VisibilityEvaluation(overall_evaluation, overall_central_distance, Side_distance_list, KeyPointsNamePSM1)
        if len(visibility_score) > 0:
            visibility_score = [round(visibility_score[i],2) for i in range(len(KeyPointsPixel))]
            visibility_score_dic = dict(zip(LandmarkPSM1Value, visibility_score))
        #########################################################################################################################

        # Draw keypoints and labels
        # KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key < 7 and key != 4 and key !=5}
        KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key <= 7}
        UV_KP_NAMES_PSM1 = list(KP_UV_LABELLED_PSM1.keys())
        UV_KP_NAMES_PSM1 = [str(item) for item in UV_KP_NAMES_PSM1]
        UV_KP_PIXELS_PSM1 = list(KP_UV_LABELLED_PSM1.values())
        UV_KP_PIXELS_PSM1 = [tuple(item) for item in UV_KP_PIXELS_PSM1]
        overlay = RIGHT_CAM_PSM1.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM1, text_list=UV_KP_NAMES_PSM1, color = color_blue)
        overlay = RIGHT_CAM_PSM1.DrawKeyPointsList(overlay, KeyPointsPixel, text_list=KeyPointsNamePSM1)

        # JCBB data association using Jacobian 
        JacobianValues_PSM1 = [JacobianCalculatorImage(K_right, np.zeros(3), np.zeros(3), T_cr1, KeyPointsPosPSMDic[name]) for name in KeyPointsNamePSM1]
        JacobiansInput = dict(zip(LandmarkPSM1Value, JacobianValues_PSM1))
        JCBB_obj1.ReadPredictedFeatureValues(KeyPointsPixel, JacobiansInput, JCBB_obj1_cov_state)
        JCBB_obj1.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM1, JCBB_obj1_cov_measure, visibility_score_dic)
        OutputMatchedKeys = JCBB_obj1.ReturnMatchingKeys()
        # OutputMatchedKeys = [LandmarkPSM1Dic[name] for name in UV_KP_NAMES_PSM1 if name in LandmarkPSM1Dic.keys()]
        JCBB_obj1.Clear()
        overlay = RIGHT_CAM_PSM1.DrawKeyPointsAssociation(overlay, UV_KP_PIXELS_PSM1, KeyPointsPixel, OutputMatchedKeys, color_blue)

        ##################################### EKF/PF with known data associations block ################################################
        KeyPointsPixelDic = dict(zip(KeyPointsNamePSM1, KeyPointsPixel))
        IndexMatched = [i for i,j in enumerate(OutputMatchedKeys) if j is not None]
        MatchedNameList = [LandmarkPSM1DicInv[key] for key in OutputMatchedKeys if key is not None]
        MatchedValueList = [UV_KP_PIXELS_PSM1[index] for index in IndexMatched]
        MatchedMeasurementDict = dict(zip(MatchedNameList, MatchedValueList))
        JacobiansDict = dict(zip(KeyPointsNamePSM1, JacobianValues_PSM1))
        
        EKF_obj1.EKFReadMeasurement(KeyPointsPixelDic, MatchedMeasurementDict, JacobiansDict)
        T_cr1_new = EKF_obj1.ReturnTcrEstimation()
        RIGHT_CAM_PSM1.UpdateTcr(T_cr1_new)

        # AEKF_obj1.AEKFReadMeasurement(KeyPointsPosPSMDic, MatchedMeasurementDict, JacobiansDict, K_right)
        # T_cr1_new = AEKF_obj1.ReturnTcrEstimation()
        # mean_state, cov_state = AEKF_obj1.ReturnStateEstimation()
        # RIGHT_CAM_PSM1.UpdateTcr(T_cr1_new)

        # EKF_MC_OBJ1.EKFReadMeasurement(KeyPointsPixelDic, MatchedMeasurementDict, JacobiansDict)
        # T_cr1_new = EKF_MC_OBJ1.ReturnTcrEstimation()
        # RIGHT_CAM_PSM1.UpdateTcr(T_cr1_new)

        ##################### For PSM3 now ###########################################################################################
        # PSM3_js = PSM3.js_his[index]
        # KeyPointsPosPSM3 = [GetPositionInBaseFrame(PSM3_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsName]
        # KeyPointsPosPSM3Dic = dict(zip(KeyPointsName, KeyPointsPosPSM3))
        # JointPosPSM3 = [GetPositionInBaseFrame(PSM3_js, np.array([0,0,0]), i) for i in range(1,7)]
        # KeyPointsPosCameraRight_PSM3 = RIGHT_CAM_PSM3.GetPositionInCameraFrameList(KeyPointsPosPSM3)
        # JointPosCameraRight_PSM3 = RIGHT_CAM_PSM3.GetPositionInCameraFrameList(JointPosPSM3)
        # KeyPointsPixel_PSM3 = RIGHT_CAM_PSM3.PixelProjectionList(KeyPointsPosCameraRight_PSM3)
        # gr_pixel, gl_pixel = KeyPointsPixel_PSM3[-2:]

        # gm_rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        # gm_PosPSM = GetPositionInBaseFrame(PSM3_js, gm_rel, 6)
        # gm_CameraRight = RIGHT_CAM_PSM3.GetPositionInCameraFrame(gm_PosPSM)
        # gm_pixel = RIGHT_CAM_PSM3.PixelProjection(gm_CameraRight)

        # j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = RIGHT_CAM_PSM3.PixelProjectionList(JointPosCameraRight_PSM3)
        # Edges_PSM3 = RIGHT_CAM_PSM3.GetEdgeProjectionCylinder(4e-3, PSM3_js)
        # SkeletonPt_list = [j1_pixel, j4_pixel, j5_pixel, j6_pixel, gr_pixel, gl_pixel]

        # overlay = RIGHT_CAM_PSM3.DrawToolSkeleton(overlay, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gr_pixel), (j6_pixel, gl_pixel)], color=(255,0,0))
        # overlay = RIGHT_CAM_PSM3.DrawLines(overlay, Edges_PSM3, color=(255,0,0))

        # # Draw keypoints and labels
        # KP_UV_LABELLED_PSM3 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key > 7}
        # UV_KP_NAMES_PSM3 = list(KP_UV_LABELLED_PSM3.keys())
        # UV_KP_NAMES_PSM3 = [str(item) for item in UV_KP_NAMES_PSM3]
        # UV_KP_PIXELS_PSM3 = list(KP_UV_LABELLED_PSM3.values())
        # UV_KP_PIXELS_PSM3 = [tuple(item) for item in UV_KP_PIXELS_PSM3]
        # overlay = RIGHT_CAM_PSM3.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM3, text_list=UV_KP_NAMES_PSM3)
        # overlay = RIGHT_CAM_PSM3.DrawKeyPointsList(overlay, KeyPointsPixel_PSM3, text_list=KeyPointsName)
        
        cv2.imshow("overlay", overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


