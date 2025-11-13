import numpy as np
import pickle
import os
import random
import yaml
import cv2
from utils import get_rigid_transform, GetPositionInBaseFrame, GetPositionInCameraFrame, QuaternionToRot, dvrk_DH_transformation
from utils_vision import PnPEstimation
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
    dir_id = "000006"
    # DrawOn = True
    DrawOn = False
    LabelDic = {1:"rf", 2:"pf", 3:"ef", 4:"gr", 5:"gl", 6:"pr", 7:"rr", 8:"rb", 9:"pb", 10:"eb", 11:"gl", 12:"gr", 13:"pl", 14:"rl"} # to be adjusted on demand

    l_gripper_PSM1 = 9.0 * 1e-3 # m
    l_gripper_PSM3 = 9.0 * 1e-3 # m

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
    gripper_angle_file = os.path.join(SubDataSet, "gripper_angle.yaml")
    KP_labelled_left = {}
    KP_labelled_right = {}
    GripperAngle = {}
    GripperAnglePSM1 = [] # list with no keys
    GripperAnglePSM3 = [] # list with no keys

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
    with open(gripper_angle_file) as stream:
        try:
            GripperAngle = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    GripperAnglePSM1 = GripperAngle["PSM1"]
    GripperAnglePSM1.insert(0,0.0)
    assert len(GripperAnglePSM1) == n_images, "num of gripper angle for PSM1 mismatch"
    GripperAnglePSM3 = GripperAngle["PSM3"]
    GripperAnglePSM3.insert(0,0.0)
    assert len(GripperAnglePSM3) == n_images, "num of gripper angle for PSM3  mismatch"

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

    Tcr1_PNP = np.identity(4)
    Tcr3_PNP = np.identity(4)

    ##################### Key points Initialisation ################################
    KeyPointsName = ["rf","rr","rl","pf","pl","pr","ef","gr","gl","rb","pb","eb"]
    JointIndex = [1,2,3,4,5,6,7]
    KeyPointsRelDic = {"rf":np.array([-0.004, 0,-0.0091]), "rb":np.array([0.004, 0,-0.0091]), "rr":np.array([0, 0.004,0]), "rl":np.array([0, -0.004,0]),
                    "pf":np.array([0.00275, -0.00275, -0.00025]), "pb":np.array([0.00275, 0.00275, 0.00025]), "pr":np.array([0.0035, -0.0015, 0.003]), "pl":np.array([0.0035, 0.0015, -0.003]),
                    "ef":np.array([0.0, 0.0 ,-0.00275]), "eb":np.array([0.0, 0.0 ,0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsJointDic = {"rf":4, "rb":4, "rr":4, "rl":4, "pf":5, "pb":5, "pr":5, "pl":5, "ef":6, "eb":6, "gr":6, "gl":6}

    # KeyPointsNamePSM1 = list(LabelDic.values())[:7]
    KeyPointsNamePSM1 = [item for item in KeyPointsName if "g" not in item] # everything excluding gripper tips, PSM1
    KeyPointsNamePSM3 = list(LabelDic.values())[7:]
    # JCBB Initialisation

    # LandmarkPSM1Name = [item for item in KeyPointsNamePSM1 if "g" not in item]
    # LandmarkPSM1Value = list(range(len(LandmarkPSM1Name)))
    LandmarkPSM1Name = [item for item in KeyPointsName if "g" not in item]
    LandmarkPSM1Value = list(range(len(LandmarkPSM1Name)))

    LandmarkPSM1Dic = dict(zip(LandmarkPSM1Name, LandmarkPSM1Value))
    LandmarkPSM1DicInv = dict(zip(LandmarkPSM1Value, LandmarkPSM1Name))
    # JCBB_obj1 = JCBB(LandmarkPSM1DicInv)
    # JCBB_obj1_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    # JCBB_obj1_cov_measure = np.array([50,50])

    LandmarkPSM3Name = [item for item in KeyPointsNamePSM3 if "g" not in item and "r" != item[-1] and "l" != item[-1]]
    LandmarkPSM3Value = list(range(len(LandmarkPSM3Name)))
    LandmarkPSM3Dic = dict(zip(LandmarkPSM3Name, LandmarkPSM3Value))
    LandmarkPSM3DicInv = dict(zip(LandmarkPSM3Value, LandmarkPSM3Name))
    # JCBB_obj2 = JCBB(LandmarkPSM3DicInv)
    # JCBB_obj2_cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])*10
    # JCBB_obj2_cov_measure = np.array([50,50])

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
        KP_UV_LABELLED_NOW = KP_labelled_left[index]

        # ###############  start from PSM1 only ###########
        # PSM1_js = PSM1.js_his[index]
        # alpha = GripperAnglePSM1[index]
        
        # KeyPointsRelDic["gr"] = np.array([l_gripper_PSM1 * np.sin(alpha/2) + (5e-4)*np.cos(alpha/2), l_gripper_PSM1 *np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        # KeyPointsRelDic["gl"] = np.array([-l_gripper_PSM1 *np.sin(alpha/2) - (5e-4)*np.cos(alpha/2), l_gripper_PSM1 * np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])

        # KeyPointsPosPSM1 = [GetPositionInBaseFrame(PSM1_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsName]
        # KeyPointsPosPSM1Dic = dict(zip(KeyPointsName, KeyPointsPosPSM1))
        # JointPosPSM1 = [GetPositionInBaseFrame(PSM1_js, np.array([0,0,0]), i) for i in range(1,7)]
        # KeyPointsPosCameraLeft_PSM1 = LEFT_CAM_PSM1.GetPositionInCameraFrameList(KeyPointsPosPSM1)
        # JointPosCameraLeft_PSM1 = LEFT_CAM_PSM1.GetPositionInCameraFrameList(JointPosPSM1)
        # KeyPointsPixel_PSM1 = LEFT_CAM_PSM1.PixelProjectionList(KeyPointsPosCameraLeft_PSM1)
        # gr_pixel, gl_pixel = KeyPointsPixel_PSM1[-2:]

        # gm_rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        # gm_PosPSM = GetPositionInBaseFrame(PSM1_js, gm_rel, 6)
        # gm_CameraLeft = LEFT_CAM_PSM1.GetPositionInCameraFrame(gm_PosPSM)
        # gm_pixel = LEFT_CAM_PSM1.PixelProjection(gm_CameraLeft)

        # j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = LEFT_CAM_PSM1.PixelProjectionList(JointPosCameraLeft_PSM1)
        # Edges_PSM1 = LEFT_CAM_PSM1.GetEdgeProjectionCylinder(4e-3, PSM1_js)

        # overlay = img_left.copy()
        # overlay = LEFT_CAM_PSM1.DrawToolSkeleton(img_left, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gm_pixel)], color=(255,0,0))
        # overlay = LEFT_CAM_PSM1.DrawLines(overlay, Edges_PSM1, color=(255,0,0))

        # overlay_visible = img_left.copy()
        
        # if DrawOn:
        #     # Draw keypoints and labels
        #     for key, value in KP_UV_LABELLED_NOW.items():
        #         if key <=7 and value is not None:
        #             overlay_visible = LEFT_CAM_PSM1.DrawKeyPoint(overlay_visible, value, text=str(key))
        #             # overlay_visible = LEFT_CAM_PSM1.DrawKeyPoint(overlay_visible, value, text=str(key)+" "+LabelDic[int(key)])
        #             cv2.imshow("visible", overlay_visible)
        #             cv2.waitKey(0)
        # # cv2.imshow("overlay", overlay_visible)
        # # if cv2.waitKey(10) & 0xFF == ord('q'):
        # #     cv2.destroyAllWindows()


        # KP_UV_LABELLED_PSM1 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key <= 7}
        # UV_KP_NAMES_PSM1 = list(KP_UV_LABELLED_PSM1.keys())
        # UV_KP_NAMES_PSM1 = [str(item) for item in UV_KP_NAMES_PSM1]
        # UV_KP_PIXELS_PSM1 = list(KP_UV_LABELLED_PSM1.values())
        # UV_KP_PIXELS_PSM1 = [tuple(item) for item in UV_KP_PIXELS_PSM1]
        # overlay = LEFT_CAM_PSM1.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM1, text_list=UV_KP_NAMES_PSM1, color=color_blue)
        # overlay = LEFT_CAM_PSM1.DrawKeyPointsList(overlay, KeyPointsPixel_PSM1, text_list=KeyPointsName, color=(0,0,255))
        
        
        # # PnP initial calib
        # for name in KP_UV_LABELLED_PSM1.keys():
        #     PNP_PIXEL_LIST.append(KP_UV_LABELLED_PSM1[name])
        #     PNP_OBJ_PTS_LIST.append(KeyPointsPosPSM1Dic[name])

        # if len(PNP_PIXEL_LIST) >= 4:
        #     Tcr1_PNP = PnPEstimation(PNP_OBJ_PTS_LIST, PNP_PIXEL_LIST, K_left, D_left)
        #     if Tcr1_PNP is None:
        #         print("EPnP failednow")
        #     else:
        #         LEFT_CAM_PSM1.UpdateTcr(Tcr1_PNP)
        #         print("PnP update now")

        # if (np.mod(index, 50) == 0 and index > 10) or index==10:
        #     os.chdir(SubDataSet)
        #     os.makedirs("HandEye", exist_ok=True)
        #     os.chdir("HandEye")
        #     file_name = "Tcr_psm1_"+str(index)+".txt"
        #     np.savetxt(file_name, Tcr1_PNP)
        #     print("End of story, bye bye bye!!!")
        #     if index == 200:
        #         break

        ##################### For PSM3 now ###########################################################################################
        PSM3_js = PSM3.js_his[index]
        alpha = GripperAnglePSM3[index]
        
        KeyPointsRelDic["gr"] = np.array([l_gripper_PSM3 * np.sin(alpha/2) + (5e-4)*np.cos(alpha/2), l_gripper_PSM3 *np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])
        KeyPointsRelDic["gl"] = np.array([-l_gripper_PSM3 *np.sin(alpha/2) - (5e-4)*np.cos(alpha/2), l_gripper_PSM3 * np.cos(alpha/2) - (5e-4)*np.sin(alpha/2), 0.0 ])

        KeyPointsPosPSM3 = [GetPositionInBaseFrame(PSM3_js, KeyPointsRelDic[name], KeyPointsJointDic[name]) for name in KeyPointsName]
        KeyPointsPosPSM3Dic = dict(zip(KeyPointsName, KeyPointsPosPSM3))
        JointPosPSM3 = [GetPositionInBaseFrame(PSM3_js, np.array([0,0,0]), i) for i in range(1,7)]
        KeyPointsPosCameraLeft_PSM3 = LEFT_CAM_PSM3.GetPositionInCameraFrameList(KeyPointsPosPSM3)
        JointPosCameraLeft_PSM3 = LEFT_CAM_PSM3.GetPositionInCameraFrameList(JointPosPSM3)
        KeyPointsPixel_PSM3 = LEFT_CAM_PSM3.PixelProjectionList(KeyPointsPosCameraLeft_PSM3)
        gr_pixel, gl_pixel = KeyPointsPixel_PSM3[-2:]

        gm_rel = np.array([0.0, 0.0102, 0.0]) # gripper middle
        gm_PosPSM = GetPositionInBaseFrame(PSM3_js, gm_rel, 6)
        gm_CameraLeft = LEFT_CAM_PSM3.GetPositionInCameraFrame(gm_PosPSM)
        gm_pixel = LEFT_CAM_PSM3.PixelProjection(gm_CameraLeft)

        j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = LEFT_CAM_PSM3.PixelProjectionList(JointPosCameraLeft_PSM3)
        Edges_PSM3 = LEFT_CAM_PSM3.GetEdgeProjectionCylinder(4e-3, PSM3_js)

        overlay = img_left.copy()
        overlay = LEFT_CAM_PSM3.DrawToolSkeleton(img_left, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gm_pixel)], color=(255,0,0))
        overlay = LEFT_CAM_PSM3.DrawLines(overlay, Edges_PSM3, color=(255,0,0))

        overlay_visible = img_left.copy()

        if DrawOn:
            # Draw keypoints and labels
            for key, value in KP_UV_LABELLED_NOW.items():
                if key >7 and value is not None:
                    overlay_visible = RIGHT_CAM_PSM3.DrawKeyPoint(overlay_visible, value, text=str(key))
                    # overlay_visible = RIGHT_CAM_PSM3.DrawKeyPoint(overlay_visible, value, text=str(key)+" "+LabelDic[int(key)])
                    cv2.imshow("visible", overlay_visible)
                    cv2.waitKey(0)

        # cv2.imshow("overlay", overlay_visible)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()


        KP_UV_LABELLED_PSM3 = {LabelDic[key]: value for key, value in KP_UV_LABELLED_NOW.items() if value != None and key > 7}
        UV_KP_NAMES_PSM3 = list(KP_UV_LABELLED_PSM3.keys())
        UV_KP_NAMES_PSM3 = [str(item) for item in UV_KP_NAMES_PSM3]
        UV_KP_PIXELS_PSM3 = list(KP_UV_LABELLED_PSM3.values())
        UV_KP_PIXELS_PSM3 = [tuple(item) for item in UV_KP_PIXELS_PSM3]
        overlay = LEFT_CAM_PSM3.DrawKeyPointsList(overlay, UV_KP_PIXELS_PSM3, text_list=UV_KP_NAMES_PSM3, color=color_blue)
        overlay = LEFT_CAM_PSM3.DrawKeyPointsList(overlay, KeyPointsPixel_PSM3, text_list=KeyPointsName, color=(0,0,255))
        
        # PnP initial calib
        for name in KP_UV_LABELLED_PSM3.keys():
            PNP_PIXEL_LIST.append(KP_UV_LABELLED_PSM3[name])
            PNP_OBJ_PTS_LIST.append(KeyPointsPosPSM3Dic[name])

        if len(PNP_PIXEL_LIST) >= 4:
            Tcr3_PNP = PnPEstimation(PNP_OBJ_PTS_LIST, PNP_PIXEL_LIST, K_left, D_left)
            if Tcr3_PNP is None:
                print("EPnP failednow")
            else:
                LEFT_CAM_PSM3.UpdateTcr(Tcr3_PNP)
                print("PnP update now")

        if (np.mod(index, 50) == 0 and index > 10) or index==10:
            os.chdir(SubDataSet)
            os.makedirs("HandEye", exist_ok=True)
            os.chdir("HandEye")
            file_name = "Tcr_psm3_"+str(index)+".txt"
            np.savetxt(file_name, Tcr3_PNP)
            print("End of story, bye bye bye!!!")
            if index == 200:
                break


        cv2.imshow("overlay", overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


