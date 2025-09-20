import numpy as np
import pickle
import os
import random
import cv2
from utils import get_rigid_transform, GetPositionInBaseFrame, GetPositionInCameraFrame, QuaternionToRot, dvrk_DH_transformation
from CRCDArmKinematics import ArmKinematicsData
from dvrk_camera import dvrk_camera
from KeyPointsDetector import KeyPointsDetector
from Jacobian import JacobianCalculatorImage, RotX, RotY, RotZ
from JCBB import JCBB
from EKF_Knownpairs import EKF_SuperDataSet
from AEKF_Knownpairs import AEKF_SuperDataSet
from EKF_MC_Knownpairs import EKF_MC_dVRKDataSet

if __name__ == "__main__":
    # Read Camera parameters
    MiscFolder = "/home/zc519/Downloads/dVRK-si-dataset/misc/cam_calib/"
    ECM_STEREO_FILE = "ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl"
    CameraParamsFile = {}

    with open(MiscFolder+ECM_STEREO_FILE, 'rb') as f:
        CameraParamsFile = pickle.load(f)

    # Extrinsic matrix between the left and right camera
    T1 = np.identity(4)
    T1[:3,:3] = CameraParamsFile['ecm_R']
    T1[:3,-1] = np.ravel(CameraParamsFile['ecm_T'])

    K_left = CameraParamsFile['ecm_left_rect_K']
    K_right = CameraParamsFile['ecm_right_rect_K']
    
    # Read kinematics and image data
    BaseFolder = "/home/zc519/Downloads/dVRK-si-dataset/original_dataset/"
    Subfoler = "C_2/"
    LeftImgFolder = BaseFolder + Subfoler + "left/"
    RightImgFolder = BaseFolder + Subfoler + "right/"
    KinematicsFolder = BaseFolder + Subfoler + "dvrk_kinematics/"
    # ArmNameList = ["ECM","PSM1","PSM2", "MTML", "MTMR"]
    ArmNameList = ["PSM1","PSM2"]
    ECM_data = ArmKinematicsData()
    PSM1_data = ArmKinematicsData()
    PSM2_data = ArmKinematicsData()
    MTML_data = ArmKinematicsData()
    MTMR_data = ArmKinematicsData()
    LeftCamSelected = True
    # LeftCamSelected = False
    RightCamSelected = not LeftCamSelected

    # Read arm kinematics data
    for arm in ArmNameList:
        data_dir = KinematicsFolder + arm
        if LeftCamSelected:
            data_dir = os.path.join(data_dir, "AlignedWithleft")
        else:
            data_dir = os.path.join(data_dir, "AlignedWithright")
        os.chdir(data_dir)
        file_list = os.listdir()
        js_history = []
        jaw_history = []
        local_cp_history = []
        eye_cp_history = []
        local_rotation_history = []
        eye_rotation_history = []
        for file in file_list:
            data = np.loadtxt(file)
            if "position" in file and "js" in file:
                if "jaw" in file:
                    jaw_history = data
                    continue
                js_history = data
            elif "position" in file and "local" in file:
                local_cp_history = data
            elif "position" in file and "local" not in file:
                eye_cp_history = data
            elif "rotation" in file and "local" in file:
                local_rotation_history = data
            elif "rotation" in file and "local" not in file:
                eye_rotation_history = data
        if arm == "ECM":
            ECM_data.ReadJointHistory(js_history[:,2:])
            ECM_data.ReadPosLocalHistory(local_cp_history[:,2:])
            ECM_data.ReadPosInEyeHistory(eye_cp_history[:,2:])
            ECM_data.ReadTimeStamp(js_history[:,0] + 1e-9 * js_history[:,1])
            ECM_data.ReadRotLocalHistory(local_rotation_history[:,2:])
            ECM_data.ReadRotInEyeHistory(eye_rotation_history[:,2:])
            # ECM_data.UpdateT_rcm_aruco()
            # ECM_data.UpdateT_c_rcm_list()
        elif arm == "PSM1":
            PSM1_data.ReadJointHistory(np.hstack((js_history[:,2:], jaw_history[:,-1].reshape(-1,1))))
            PSM1_data.ReadPosLocalHistory(local_cp_history[:,2:])
            PSM1_data.ReadPosInEyeHistory(eye_cp_history[:,2:])
            PSM1_data.ReadTimeStamp(js_history[:,0] + 1e-9 * js_history[:,1])
            PSM1_data.ReadRotLocalHistory(local_rotation_history[:,2:])
            PSM1_data.ReadRotInEyeHistory(eye_rotation_history[:,2:])
            PSM1_data.UpdateT_rcm_aruco()
            PSM1_data.UpdateT_c_rcm_list()
        elif arm == "PSM2":
            PSM2_data.ReadJointHistory(np.hstack((js_history[:,2:], jaw_history[:,-1].reshape(-1,1))))
            PSM2_data.ReadPosLocalHistory(local_cp_history[:,2:])
            PSM2_data.ReadPosInEyeHistory(eye_cp_history[:,2:])
            PSM2_data.ReadTimeStamp(js_history[:,0] + 1e-9 * js_history[:,1])
            PSM2_data.ReadRotLocalHistory(local_rotation_history[:,2:])
            PSM2_data.ReadRotInEyeHistory(eye_rotation_history[:,2:])
            PSM2_data.UpdateT_rcm_aruco()
            PSM2_data.UpdateT_c_rcm_list()

    ##################### Image and camera objects initialisation ###################################################
    ImageDir = os.path.join(LeftImgFolder, "images") if LeftCamSelected else os.path.join(RightImgFolder, "images")
    ImageFiles = os.listdir(ImageDir)
    ImageFiles.sort()

    K = K_left if LeftCamSelected else K_right
    CAM_PSM1 = dvrk_camera(K, PSM1_data.T_c_rcm_list[0])
    CAM_PSM2 = dvrk_camera(K, PSM2_data.T_c_rcm_list[0])
    T_cr1 = PSM1_data.T_c_rcm_list[0]
    T_cr2 = PSM2_data.T_c_rcm_list[0]

    CAM_PSM1.UpdateTcr(T_cr1) # inaccurate init hand-eye (PSM1)
    CAM_PSM2.UpdateTcr(T_cr2) # inaccurate init hand-eye (PSM2)
    N_images = len(ImageFiles)
    # N = 150 # randomly pick 10 frames to conduct experiments
    N = N_images
    color_pink = (255,141,161)
    color_blue = (255,0,0)

    FBF_DETECTOR = KeyPointsDetector("FBF")
    PCH_DETECTOR = KeyPointsDetector("PCH")

    ########################################## FBF DETECTOR CLASS ##################################################################
    # KeyPointsName = ["rf","rb","rr","rl","pf","pb","pr","pl","ef","eb","gr","gl"]
    KeyPointsName_FBF = ["pf","pl","ef","gr","gl"]
    JointIndex = [1,2,3,4,5,6,7]
    KeyPointsRelDic_FBF = {"pf":np.array([0.00275, -0.00275, -0.00025]), "pl":np.array([0.0035, 0.0015, -0.003]),
                           "ef":np.array([0.0, 0.0 ,-0.00275]), "gr":np.array([0.0, 0.0102, 0.0]), "gl":np.array([0.0, 0.0102, 0.0])}
    KeyPointsJointDic_FBF = {"pf":5, "pl":5, "ef":6, "gr":6, "gl":6}

    # JCBB initialisation
    LandmarkName_FBF = ["pf","pl","ef","gr","gl"]
    LandmarkValue_FBF = [0,1,2,3,4] # Outliers are denoted 0
    LandmarkDic_FBF = dict(zip(LandmarkName_FBF, LandmarkValue_FBF))
    LandmarkDicInv_FBF = dict(zip(LandmarkValue_FBF, LandmarkName_FBF))
    JCBB_FBF = JCBB(LandmarkDicInv_FBF)

    # EKF Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3]) * 10
    cov_measure = np.array([25,25])
    EKF_FBF = EKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr2)

    # AEKF Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3]) * 10
    cov_measure = np.array([50,50])
    forget_factor = 0.3
    AEKF_FBF = AEKF_SuperDataSet(mean_state, cov_state, cov_measure, T_cr2, LandmarkName_FBF, forget_factor)

    # EKF MC Initialisation
    mean_state = np.zeros(6)
    cov_state = np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3]) * 100
    cov_measure = np.array([50,50])
    EKF_MC_FBF = EKF_MC_dVRKDataSet(mean_state, cov_state, cov_measure, T_cr2, LandmarkName_FBF, bandwidth=3)

    ########################################## PCH DETECTOR CLASS ##################################################################
    # MTML-PSM2 
    # MTMR-PSM1

    for id in range(N):
        # index = random.randint(0, N_images)
        index = id
        os.chdir(ImageDir)
        img = cv2.imread(ImageFiles[index])
        img = img[:720,:,:]
        PSM1_js = PSM1_data.JointHistory[index]
        PSM2_js = PSM2_data.JointHistory[index]

        FBF_observed = FBF_DETECTOR.ExtractKeyPointsFromImg(img)
        PCH_observed = PCH_DETECTOR.ExtractKeyPointsFromImg(img)
        FBF_observed = FBF_observed[2:]
        # continue

        alpha_PSM1 = PSM1_js[-1]
        alpha_PSM2 = PSM2_js[-1]
        # KeyPointsRelDic_FBF["gr"] = np.array([(9e-3)*np.sin(alpha_PSM1/2) + (5e-4)*np.cos(alpha_PSM1/2), (9e-3)*np.cos(alpha_PSM1/2) - (5e-4)*np.sin(alpha_PSM1/2), 0.0 ])
        # KeyPointsRelDic_FBF["gl"] = np.array([-(6.5e-3)*np.sin(alpha_PSM1/2) - (5e-4)*np.cos(alpha_PSM1/2), (6.5e-3)*np.cos(alpha_PSM1/2) - (5e-4)*np.sin(alpha_PSM1/2), 0.0 ])

        # # For PSM1 first
        # JointPosPSM1 = [GetPositionInBaseFrame(PSM1_js, np.array([0,0,0]), j) for j in range(1,7)]
        # JointPosPSM1Camera = CAM_PSM1.GetPositionInCameraFrameList(JointPosPSM1)
        # j1_pixel, j2_pixel, j3_pixel, j4_pixel, j5_pixel, j6_pixel = CAM_PSM1.PixelProjectionList(JointPosPSM1Camera)
        # PSM1KeyPointsPosRCM = [GetPositionInBaseFrame(PSM1_js, KeyPointsRelDic_FBF[name], KeyPointsJointDic_FBF[name]) for name in KeyPointsName_FBF]
        # PSM1KeyPointsPosCam = CAM_PSM1.GetPositionInCameraFrameList(PSM1KeyPointsPosRCM)
        # PSM1KeyPointsPixel =  CAM_PSM1.PixelProjectionList(PSM1KeyPointsPosCam)
        # gr_pixel, gl_pixel = PSM1KeyPointsPixel[-2:]
        # overlay = CAM_PSM1.DrawToolSkeleton(img, [(j1_pixel,j4_pixel), (j5_pixel, j6_pixel), (j6_pixel, gr_pixel), (j6_pixel, gl_pixel)])
        # overlay = CAM_PSM1.DrawKeyPointsList(overlay, PSM1KeyPointsPixel,text_list=KeyPointsName_FBF)
        # overlay = CAM_PSM1.DrawKeyPointsList(overlay, FBF_observed)
        # # overlay = CAM_PSM1.DrawKeyPointsList(overlay, PCH_observed)

        # Now let's move onto key points detection using a pre-trained network

        ############################# For PSM2 (where FBF is fixed) ######################################################################################################
        KeyPointsRelDic_FBF["gr"] = np.array([(9e-3)*np.sin(alpha_PSM2/2) + (5e-4)*np.cos(alpha_PSM2/2), (9e-3)*np.cos(alpha_PSM2/2) - (5e-4)*np.sin(alpha_PSM2/2), 0.0 ])
        KeyPointsRelDic_FBF["gl"] = np.array([-(6.5e-3)*np.sin(alpha_PSM2/2) - (5e-4)*np.cos(alpha_PSM2/2), (6.5e-3)*np.cos(alpha_PSM2/2) - (5e-4)*np.sin(alpha_PSM2/2), 0.0 ])
        KeyPointsPosPSM_FBF = [GetPositionInBaseFrame(PSM2_js, KeyPointsRelDic_FBF[name], KeyPointsJointDic_FBF[name]) for name in KeyPointsName_FBF]
        KeyPointsPosPSM_FBF_Dic = dict(zip(KeyPointsName_FBF, KeyPointsPosPSM_FBF))
        KeyPointsPosCam_FBF = CAM_PSM2.GetPositionInCameraFrameList(KeyPointsPosPSM_FBF)
        KeyPointsPixel_FBF = CAM_PSM2.PixelProjectionList(KeyPointsPosCam_FBF)
        KeyPointsPixelDic_FBF = dict(zip(KeyPointsName_FBF, KeyPointsPixel_FBF))
        gr_pixel, gl_pixel = KeyPointsPixel_FBF[-2:]

        JointPosPSM2 = [GetPositionInBaseFrame(PSM2_js, np.array([0,0,0]), k) for k in range(1,7)]
        JointPosPSM2Camera = CAM_PSM2.GetPositionInCameraFrameList(JointPosPSM2)
        j1_pixel_PSM2, j2_pixel_PSM2, j3_pixel_PSM2, j4_pixel_PSM2, j5_pixel_PSM2, j6_pixel_PSM2 = CAM_PSM2.PixelProjectionList(JointPosPSM2Camera)
        Edges_FBF = CAM_PSM2.GetEdgeProjectionCylinder(4e-3, PSM2_js)

        overlay = CAM_PSM2.DrawToolSkeleton(img, [(j1_pixel_PSM2,j4_pixel_PSM2), (j5_pixel_PSM2, j6_pixel_PSM2), (j6_pixel_PSM2, gr_pixel), (j6_pixel_PSM2, gl_pixel)])
        overlay = CAM_PSM2.DrawLines(overlay, Edges_FBF)
        overlay = CAM_PSM2.DrawKeyPointsList(overlay, KeyPointsPixel_FBF, text_list=KeyPointsName_FBF)
        overlay = CAM_PSM2.DrawKeyPointsList(overlay, FBF_observed, color=color_blue)

        # JCBB data association using Jacobian
        JacobianValues_FBF = [JacobianCalculatorImage(K, np.zeros(3), np.zeros(3), T_cr2, KeyPointsPosPSM_FBF_Dic[name]) for name in LandmarkName_FBF]
        JacobiansInput_FBF = dict(zip(LandmarkValue_FBF, JacobianValues_FBF))
        JacobiansDict_FBF = dict(zip(LandmarkName_FBF, JacobianValues_FBF))
        JCBB_FBF.ReadPredictedFeatureValues(KeyPointsPixel_FBF, JacobiansInput_FBF, cov_state)
        JCBB_FBF.ReadMeasurementFeatureValues(FBF_observed, cov_measure) # equal measurement covariances
        OutputMatchedKeys_FBF = JCBB_FBF.ReturnMatchingKeys()
        JCBB_FBF.Clear()
        overlay = CAM_PSM2.DrawKeyPointsAssociation(overlay, FBF_observed, KeyPointsPixel_FBF, OutputMatchedKeys_FBF, color_blue)
        # cv2.imshow("Association", overlay)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Association")

        # EKF/AEKF with known data associations
        IndexMatched_FBF = [i for i,j in enumerate(OutputMatchedKeys_FBF) if j is not None]
        MatchedNameList_FBF = [LandmarkDicInv_FBF[key] for key in OutputMatchedKeys_FBF if key is not None]
        MatchedValueList_FBF = [FBF_observed[index] for index in IndexMatched_FBF]
        MatchedMeasurementDict_FBF = dict(zip(MatchedNameList_FBF, MatchedValueList_FBF))
        
        # EKF_FBF.EKFReadMeasurement(KeyPointsPixelDic_FBF, MatchedMeasurementDict_FBF, JacobiansDict_FBF)
        # T_cr2_new = EKF_FBF.ReturnTcrEstimation()
        # CAM_PSM2.UpdateTcr(T_cr2_new)


        # AEKF_FBF.AEKFReadMeasurement(KeyPointsPosPSM_FBF_Dic, MatchedMeasurementDict_FBF, JacobiansDict_FBF, K)
        # T_cr2_new = AEKF_FBF.ReturnTcrEstimation()
        # # mean_state, cov_state = AEKF_obj1.ReturnStateEstimation()
        # CAM_PSM2.UpdateTcr(T_cr2_new)

        EKF_MC_FBF.EKFReadMeasurement(KeyPointsPixelDic_FBF, MatchedMeasurementDict_FBF, JacobiansDict_FBF)
        T_cr2_new = EKF_MC_FBF.ReturnTcrEstimation()
        CAM_PSM2.UpdateTcr(T_cr2_new)

        cv2.imshow("overlay", overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        continue






