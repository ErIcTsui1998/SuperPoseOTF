# Adaptive EKF to address sudden changes in system dynamics
# Paper link: https://arxiv.org/pdf/1702.00884 

import numpy as np
from Jacobian import RotX, RotY, RotZ, TFromThetaVecTVec, JacobianCalculatorImage
from Kinematics import GetPositionInBaseFrame, GetPositionInCameraFrame
from utils_vision import PixelProjection

class AEKF_SuperDataSet:
    def __init__(self, state_mean_init, state_cov_init, measure_cov_init, Tcr_init, KeyPointsNameList, alpha):
        self.__state_cov = state_cov_init if len(state_cov_init.shape) == 2 else np.diag(state_cov_init) # 6*6
        self.__state_mean = state_mean_init # 6*1
        self.__Tcr_init = Tcr_init
        self.__Tcr = Tcr_init
        self.__T00 = np.identity(4)
        measure_cov = measure_cov_init if len(measure_cov_init.shape) == 2 else np.diag(measure_cov_init) # 2*2 for pixels
        self.__measure_cov_dic = {name: measure_cov for name in KeyPointsNameList}
        self.__measure_cov = measure_cov
        self.__alpha = alpha # forget factor
        self.__R = np.zeros(self.__state_cov.shape)
        self.__kp_name_list = KeyPointsNameList

    def UpdateParameters(self, new_state_mean, new_state_cov, new_measure_cov):
        self.__state_cov = new_state_cov
        self.__state_mean = new_state_mean
        measure_cov = new_measure_cov if len(new_measure_cov.shape) == 2 else np.diag(new_measure_cov) # 2*2 for pixels
        self.__measure_cov_dic = {name: measure_cov for name in self.__kp_name_list}
        self.__R = np.zeros(self.__state_cov.shape)
        self.__measure_cov = measure_cov 

    def AEKFReadMeasurement(self, MeasurementDic, KeyPointsPSMDic, K_cam):
        n_measure = len(MeasurementDic)
        if n_measure == 0:
            print("No measurements availble in this instance")
            return
        
        # Innovation & Correction Stage
        Tcr_pre = self.__Tcr
        KeyPointsCamDic = {key: GetPositionInCameraFrame(Tcr_pre, KeyPointsPSMDic[key]) for key in MeasurementDic.keys()}
        PredictionDic = {key: PixelProjection(KeyPointsCamDic[key], K_cam) for key in MeasurementDic.keys()}
        InnovationDic = {key: np.array(MeasurementDic[key]) - np.array(PredictionDic[key]) for key in MeasurementDic.keys()}
        state_mean_EKF, state_cov_EKF, JacobianDic = self.__EKF_measurement(KeyPointsPSMDic, MeasurementDic, K_cam)

        # Get Residual
        T00_tmp = TFromThetaVecTVec(state_mean_EKF[:3], state_mean_EKF[3:])
        Tcr_tmp = self.__Tcr_init @ T00_tmp
        KeyPointsCamDic_tmp = {key: GetPositionInCameraFrame(Tcr_tmp, KeyPointsPSMDic[key]) for key in MeasurementDic.keys()}
        PredictionDic_tmp = {key: PixelProjection(KeyPointsCamDic_tmp[key], K_cam) for key in MeasurementDic.keys()}
        ResidualDic = {key: np.array(MeasurementDic[key]) - np.array(PredictionDic_tmp[key]) for key in MeasurementDic.keys()}
        
        # JacobianDic = {name: JacobianCalculatorImage(K_cam, np.zeros(3), np.zeros(3), Tcr_tmp, KeyPointsPSMDic[name]) for name in MeasurementDic.keys()}

        # state_mean_candidate = self.__state_mean.copy()
        # state_cov_candidate = self.__state_cov.copy() + self.__R.copy()
        # state_cov_candidate = self.__state_cov.copy()
        state_cov_candidate = state_cov_EKF.copy()
        measure_cov_candidate = self.__measure_cov.copy() * self.__alpha
        R_candidate = self.__R.copy() * self.__alpha

        # y_all = np.concatenate(list(InnovationDic.values()))
        # H_all = np.vstack(list(JacobianDic.values()))
        # measure_cov_block = np.kron(np.eye(n_measure), self.__measure_cov)
        # S = H_all @ state_cov_candidate @ H_all.T + measure_cov_block
        # K = state_cov_candidate @ H_all.T @ np.linalg.inv(S)

        # measure_cov_inc = (1-self.__alpha) * (y_all@y_all.T + H_all@state_cov_candidate@H_all.T)
        # print("new measure cov")

        for name in MeasurementDic.keys():
            innovation = InnovationDic[name]
            residual = ResidualDic[name]

        #     # AEKF update
            H = JacobianDic[name] # 2 by 6
            residual = residual.reshape(-1,1) # 2 by 1

            measure_cov_inc = (1-self.__alpha) * (residual@residual.T + H@state_cov_candidate@H.T) / n_measure
            measure_cov_candidate += measure_cov_inc

            # Kalman Gain update
            S = np.matmul(np.matmul(H, state_cov_candidate), H.T) + self.__measure_cov
            K = state_cov_candidate @ H.T @ np.linalg.inv(S)
            innovation = innovation.reshape(-1,1)

            R_candidate_inc = (1-self.__alpha)*K@innovation@innovation.T@K.T  / n_measure
            R_candidate += R_candidate_inc

        self.__R = R_candidate.copy()
        self.__measure_cov = measure_cov_candidate.copy()
        self.__state_mean = state_mean_EKF.copy()
        self.__state_cov = state_cov_EKF.copy()
        self.__T00 = TFromThetaVecTVec(self.__state_mean[:3], self.__state_mean[3:])
        self.__Tcr = self.__Tcr_init @ self.__T00
    
    # Traditional EKF update, which returns Residual Dic
    def __EKF_measurement(self, KeyPointsPSMDic, MeasurementDic, K_cam):
        # Within known correspondences between prediction and measurements
        state_mean_candidate = self.__state_mean.copy()
        # state_cov_candidate = self.__state_cov.copy()
        state_cov_candidate = self.__state_cov.copy() + self.__R.copy()
        if len(state_cov_candidate.shape) == 1:
            state_cov_candidate = np.diag(state_cov_candidate)

        # No associated measurements have been obtained
        if len(MeasurementDic) == 0:
            return
        Tcr_tmp = self.__Tcr
        JacobianOutputDic = {}
        for index, measurement in MeasurementDic.items():
            KeyPointsPSM = KeyPointsPSMDic[index]
            KeyPointsCam = GetPositionInCameraFrame(Tcr_tmp, KeyPointsPSM)
            prediction = PixelProjection(KeyPointsCam, K_cam)
            H = JacobianCalculatorImage(K_cam, np.zeros(3), np.zeros(3), Tcr_tmp, KeyPointsPSM)
            JacobianOutputDic[index] = H

            m1 = np.array(measurement)
            p1 = np.array(prediction)
            error = m1 - p1 # 2 by 1
            # measurement_cov = self.__measure_cov_dic[index] # measurment cov associated with each keypoint observation
            measurement_cov = self.__measure_cov.copy()
            S = np.matmul(np.matmul(H, state_cov_candidate), H.T) + measurement_cov
            K = state_cov_candidate @ H.T @ np.linalg.inv(S)
            state_mean_candidate = state_mean_candidate + K @ error
            KH = K @ H
            mat1 = np.identity(KH.shape[0]) - KH
            state_cov_candidate = mat1 @ state_cov_candidate

            T00_tmp = TFromThetaVecTVec(state_mean_candidate[:3], state_mean_candidate[3:])
            Tcr_tmp = self.__Tcr_init @ T00_tmp

        return state_mean_candidate, state_cov_candidate, JacobianOutputDic

    def ReturnStateEstimation(self):
        return self.__state_mean, self.__state_cov
    
    def ReturnCovMeasureEstimation(self):
        return self.__measure_cov

    def ReturnT00Estimation(self):
        return self.__T00
    
    def ReturnTcrEstimation(self):
        return self.__Tcr

    def ReturnTcrInit(self):
        return self.__Tcr_init