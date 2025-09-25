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
import argparse

if __name__ == "__main__":    
    BaseFolder = "/home/zc519/Downloads/SurgPoseDataSet"
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--id', type=str, help='dir id')
    # args = parser.parse_args()
    # dir_id = args.id
    dir_id = "000024"

    ArmNameList = ['PSM1','PSM3']
    PSM1 = dvrk_arm()
    PSM3 = dvrk_arm()
    SubDataSet = os.path.join(BaseFolder, dir_id)
    # LeftImagesFolder = os.path.join(SubDataSet, "green/LeftImages")
    LeftImagesFolder = os.path.join(SubDataSet, "regular/LeftImages")
    LeftImages = os.listdir(LeftImagesFolder)
    # RightImagesFolder = os.path.join(SubDataSet, "green/RightImages")
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
    if "Tcr_psm1_1000.txt" in os.listdir(SubDataSet+"/HandEye"):
        os.chdir(SubDataSet+"/HandEye")
        T_cr1 = np.loadtxt("Tcr_psm1_1000.txt")
    if "Tcr_psm3_1000.txt" in os.listdir(SubDataSet+"/HandEye"):
        os.chdir(SubDataSet+"/HandEye")
        T_cr3 = np.loadtxt("Tcr_psm3_1000.txt")

    ##################### Key points Initialisation ################################
    KeyPointsName = ["rf","rr","rl","pf","pl","ef","gr","gl","rb","pb","eb"]
    JointIndex = [1,2,3,4,5,6,7]
    KeyPointsRelDic = {"rf":np.array([-0.004, 0,-0.00625]), "rb":np.array([0.004, 0,-0.00625]), "rr":np.array([0, 0.004,0]), "rl":np.array([0, -0.004,0]),
                    "pf":np.array([0.00275, -0.00275, -0.00025]), "pb":np.array([0.00275, 0.00275, 0.00025]), "pr":np.array([0.0035, -0.0015, 0.003]), "pl":np.array([0.0035, 0.0015, -0.003]),
                    "ef":np.array([0.0, 0.0 ,-0.00275]), "eb":np.array([0.0, 0.0 ,0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsJointDic = {"rf":4, "rb":4, "rr":4, "rl":4, "pf":5, "pb":5, "pr":5, "pl":5, "ef":6, "eb":6, "gr":6, "gl":6}
    
    LabelDic = {1:"rf", 2:"pf", 3:"ef", 4:"gl", 5:"gr", 6:"pl", 7:"rl", 8:"rb", 9:"pb", 10:"eb", 11:"gl", 12:"gr", 13:"pr", 14:"rr"}
    # LabelDic = {1:"rf", 2:"pf", 3:"ef", 4:"gl", 5:"gr", 6:"pl", 7:"rl", 8:"rf", 9:"pf", 10:"ef", 11:"gl", 12:"gr", 13:"pl", 14:"rl"}
    # LabelDic = {1:"rb", 2:"pb", 3:"eb", 4:"gr", 5:"gl", 6:"pr", 7:"rr", 8:"rb", 9:"pb", 10:"eb", 11:"gr", 12:"gl", 13:"pl", 14:"rl"}

    KeyPointsNamePSM1 = [item for item in KeyPointsName if "g" not in item] # everything excluding gripper tips, PSM1
    KeyPointsNamePSM3 = [item for item in KeyPointsName if "g" not in item] # everything excluding gripper tips, PSM3
    # JCBB Initialisation

    LandmarkPSM1Name = [item for item in KeyPointsName if "g" not in item]
    LandmarkPSM1Value = list(range(len(LandmarkPSM1Name)))
    LandmarkPSM1Dic = dict(zip(LandmarkPSM1Name, LandmarkPSM1Value))
    LandmarkPSM1DicInv = dict(zip(LandmarkPSM1Value, LandmarkPSM1Name))
    JCBB_obj1 = JCBB(LandmarkPSM1DicInv)
    JCBB_obj1_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    JCBB_obj1_cov_measure = np.array([15,15])

    LandmarkPSM3Name = [item for item in KeyPointsNamePSM3 if "g" not in item and "r" != item[-1] and "l" != item[-1]]
    LandmarkPSM3Value = list(range(len(LandmarkPSM3Name)))
    LandmarkPSM3Dic = dict(zip(LandmarkPSM3Name, LandmarkPSM3Value))
    LandmarkPSM3DicInv = dict(zip(LandmarkPSM3Value, LandmarkPSM3Name))
    JCBB_obj2 = JCBB(LandmarkPSM3DicInv)
    JCBB_obj2_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    JCBB_obj2_cov_measure = np.array([50,50])

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

        # ###############  start from PSM1 only ###########
        PSM1_js = PSM1.js_his[index]
        alpha = 0.0
        KeyPointsRelDic["gr"] = np.array([(9e-3)*np.sin(alpha/2) + (5e-4)*np.cos(alpha/2), (9e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        KeyPointsRelDic["gl"] = np.array([-(6.5e-3)*np.sin(alpha/2) - (5e-4)*np.cos(alpha/2), (6.5e-3)*np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        
        KeyPointsPSM1Pos = [GetPositionInBaseFrame(PSM1_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsNamePSM1]
        KeyPointsPSM1PosDic = dict(zip(KeyPointsNamePSM1, KeyPointsPSM1Pos))
        JointPSM1Pos = [GetPositionInBaseFrame(PSM1_js, np.array([0,0,0]), i) for i in range(1,7)]
        KeyPointsPSM1PosCameraRight = RIGHT_CAM_PSM1.GetPositionInCameraFrameList(KeyPointsPSM1Pos)
        JointPSM1PosCameraRight = RIGHT_CAM_PSM1.GetPositionInCameraFrameList(JointPSM1Pos)
        KeyPointsPSM1Pixel = RIGHT_CAM_PSM1.PixelProjectionList(KeyPointsPSM1PosCameraRight)
        gr_PSM1_pixel, gl_PSM1_pixel = KeyPointsPSM1Pixel[-2:]

        gm_PSM1rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        gm_PSM1Pos = GetPositionInBaseFrame(PSM1_js, gm_PSM1rel, 6)
        gm_PSM1CameraRight = RIGHT_CAM_PSM1.GetPositionInCameraFrame(gm_PSM1Pos)
        gm_PSM1pixel = RIGHT_CAM_PSM1.PixelProjection(gm_PSM1CameraRight)

        j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = RIGHT_CAM_PSM1.PixelProjectionList(JointPSM1PosCameraRight)
        Edges_PSM1 = RIGHT_CAM_PSM1.GetEdgeProjectionCylinder(4e-3, PSM1_js)
        SkeletonPt_PSM1_list = [j1_pixel, j4_pixel, j5_pixel, j6_pixel, gr_PSM1_pixel, gl_PSM1_pixel]
        SkeletonLine_PSM1_list = GetListOfLineEquationFromPointSet([(j1_pixel,j4_pixel),(j5_pixel,j6_pixel), (j6_pixel,gm_PSM1pixel)])

        overlay = RIGHT_CAM_PSM1.DrawToolSkeleton(img_right, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel)])
        overlay = RIGHT_CAM_PSM1.DrawLines(overlay, Edges_PSM1)

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
        KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key < 7 and key != 4 and key !=5}
        # KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key < 7}
        UV_KP_NAMES_PSM1 = list(KP_UV_LABELLED_PSM1.keys())
        UV_KP_NAMES_PSM1 = [str(item) for item in UV_KP_NAMES_PSM1]
        UV_KP_PIXELS_PSM1 = list(KP_UV_LABELLED_PSM1.values())
        UV_KP_PIXELS_PSM1 = [tuple(item) for item in UV_KP_PIXELS_PSM1]
        overlay = RIGHT_CAM_PSM1.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM1, text_list=UV_KP_NAMES_PSM1, color = color_blue)
        overlay = RIGHT_CAM_PSM1.DrawKeyPointsList(overlay, KeyPointsPSM1Pixel, text_list=LandmarkPSM1Name)

        # JCBB data association using Jacobian 
        JacobianValues_PSM1 = [JacobianCalculatorImage(K_right, np.zeros(3), np.zeros(3), T_cr1, KeyPointsPSM1PosDic[name]) for name in LandmarkPSM1Name]
        JacobiansInput = dict(zip(LandmarkPSM1Value, JacobianValues_PSM1))
        JCBB_obj1.ReadPredictedFeatureValues(KeyPointsPSM1Pixel, JacobiansInput, JCBB_obj1_cov_state)
        JCBB_obj1.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM1, JCBB_obj1_cov_measure, visibility_score_dic)
        OutputMatchedKeys = JCBB_obj1.ReturnMatchingKeys()
        # OutputMatchedKeys = [LandmarkPSM1Dic[name] if name in LandmarkPSM1Dic.keys() else None for name in UV_KP_NAMES_PSM1]
        JCBB_obj1.Clear()
        overlay = RIGHT_CAM_PSM1.DrawKeyPointsAssociation(overlay, UV_KP_PIXELS_PSM1, KeyPointsPSM1Pixel, OutputMatchedKeys, color_blue)

        # PnP initial calib
        for id_x, key_x in enumerate(OutputMatchedKeys):
            if key_x != None:
                name = LandmarkPSM1DicInv[key_x]
                if visibility_score != []:
                    if visibility_score[id_x] >0:
                        PNP_PIXEL_LIST.append(list(UV_KP_PIXELS_PSM1[id_x]))
                        PNP_OBJ_PTS_LIST.append(KeyPointsPSM1Pos[key_x])
                        continue              
                else:
                    PNP_PIXEL_LIST.append(list(UV_KP_PIXELS_PSM1[id_x]))
                    PNP_OBJ_PTS_LIST.append(KeyPointsPSM1Pos[key_x])
                continue

        if np.mod(index, 50) == 0 and index > 50:
        # if index == 100 or index == 200 or index == 300 or index == 400 or index ==500 or index == 600 or index == 700 or index ==800 or index==900 or index==1000:
            Tcr_new = PnPEstimation(PNP_OBJ_PTS_LIST, PNP_PIXEL_LIST, K_right, D_right)
            RIGHT_CAM_PSM1.UpdateTcr(Tcr_new)
            os.chdir(SubDataSet)
            os.makedirs("HandEyeRefined", exist_ok=True)
            os.chdir("HandEyeRefined")
            file_name = "Tcr_psm1_"+str(index)+".txt"
            np.savetxt(file_name, Tcr_new)

        ##################### For PSM3 now ###########################################################################################
        # PSM3_js = PSM3.js_his[index]
        # KeyPointsPosPSM3 = [GetPositionInBaseFrame(PSM3_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in LandmarkPSM3Name]
        # KeyPointsPosPSM3Dic = dict(zip(LandmarkPSM3Name, KeyPointsPosPSM3))
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

        # overlay = RIGHT_CAM_PSM3.DrawToolSkeleton(img_right, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gm_pixel)], color=(255,0,0))
        # overlay = RIGHT_CAM_PSM3.DrawLines(overlay, Edges_PSM3, color=(255,0,0))

        # # Draw keypoints and labels
        # KP_UV_LABELLED_PSM3 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key > 7 and key != 11 and key !=12}
        # UV_KP_NAMES_PSM3 = list(KP_UV_LABELLED_PSM3.keys())
        # UV_KP_NAMES_PSM3 = [str(item) for item in UV_KP_NAMES_PSM3]
        # UV_KP_PIXELS_PSM3 = list(KP_UV_LABELLED_PSM3.values())
        # UV_KP_PIXELS_PSM3 = [tuple(item) for item in UV_KP_PIXELS_PSM3]
        # overlay = RIGHT_CAM_PSM3.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM3, text_list=UV_KP_NAMES_PSM3, color=color_blue)
        # overlay = RIGHT_CAM_PSM3.DrawKeyPointsList(overlay, KeyPointsPixel_PSM3, text_list=LandmarkPSM3Name, color=(0,0,255))

        # # JCBB data association using Jacobian 
        # JacobianValues_PSM3 = [JacobianCalculatorImage(K_right, np.zeros(3), np.zeros(3), T_cr1, KeyPointsPosPSM3Dic[name]) for name in LandmarkPSM3Name]
        # JacobiansInput_PSM3 = dict(zip(LandmarkPSM3Value, JacobianValues_PSM3))
        # JCBB_obj2.ReadPredictedFeatureValues(KeyPointsPixel_PSM3, JacobiansInput_PSM3, JCBB_obj2_cov_state)
        # JCBB_obj2.ReadMeasurementFeatureValues(UV_KP_PIXELS_PSM3, JCBB_obj2_cov_measure)
        # # OutputMatchedKeys = JCBB_obj1.ReturnMatchingKeys()
        # OutputMatchedKeys_PSM3 = [LandmarkPSM3Dic[name] if name in LandmarkPSM3Dic.keys() else None for name in UV_KP_NAMES_PSM3]
        # JCBB_obj2.Clear()
        # overlay = RIGHT_CAM_PSM3.DrawKeyPointsAssociation(overlay, UV_KP_PIXELS_PSM3, KeyPointsPixel_PSM3, OutputMatchedKeys_PSM3, color_blue)

        # # PnP initial calib
        # for ii in OutputMatchedKeys_PSM3:
        #     if ii != None:
        #         name = LandmarkPSM3DicInv[ii]
        #         PNP_PIXEL_LIST.append(KP_UV_LABELLED_PSM3[name])
        #         PNP_OBJ_PTS_LIST.append(KeyPointsPosPSM3Dic[name])
        #     else:
        #         continue
        # # # PnP initial calib
        # # for name in KP_UV_LABELLED_PSM3.keys():
        # #     PNP_PIXEL_LIST.append(KP_UV_LABELLED_PSM3[name])
        # #     PNP_OBJ_PTS_LIST.append(KeyPointsPosPSM3Dic[name])

        # if index == 100 or index == 200 or index == 300 or index == 400 or index ==500 or index == 600 or index == 700 or index ==800 or index==900 or index==1000:
        #     Tcr_new = PnPEstimation(PNP_OBJ_PTS_LIST, PNP_PIXEL_LIST, K_right, D_right)
        #     RIGHT_CAM_PSM3.UpdateTcr(Tcr_new)
        #     os.chdir(SubDataSet)
        #     os.makedirs("HandEyeRefinded", exist_ok=True)
        #     os.chdir("HandEyeRefined")
        #     file_name = "Tcr_psm3_"+str(index)+".txt"
        #     np.savetxt(file_name, Tcr_new)

        cv2.imshow("overlay", overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


