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
        # self.__R = np.zeros(self.__state_cov.shape)

    def AEKFReadMeasurement(self, KeyPointsPSMDic, MeasurementDic, JacobianDic, K):
        if len(MeasurementDic) == 0:
            print("No measurements availble in this instance")
            return
        # Innovation & Correction Stage
        Tcr_pre = self.__Tcr
        KeyPointsCamDic = {key: GetPositionInCameraFrame(Tcr_pre, KeyPointsPSMDic[key]) for key in MeasurementDic.keys()}
        PredictionDic = {key: PixelProjection(KeyPointsCamDic[key], K) for key in MeasurementDic.keys()}
        InnovationDic = {key: np.array(MeasurementDic[key]) - np.array(PredictionDic[key]) for key in MeasurementDic.keys()}
        state_mean_EKF, state_cov_EKF = self.__EKF_measurement(PredictionDic, MeasurementDic, JacobianDic)

        # Get Residual
        T00_tmp = TFromThetaVecTVec(state_mean_EKF[:3], state_mean_EKF[3:])
        Tcr_tmp = self.__Tcr_init @ T00_tmp
        KeyPointsCamDic_tmp = {key: GetPositionInCameraFrame(Tcr_tmp, KeyPointsPSMDic[key]) for key in MeasurementDic.keys()}
        PredictionDic_tmp = {key: PixelProjection(KeyPointsCamDic_tmp[key], K) for key in MeasurementDic.keys()}
        ResidualDic = {key: np.array(MeasurementDic[key]) - np.array(PredictionDic_tmp[key]) for key in MeasurementDic.keys()}
        
        JacobianDic = {name: JacobianCalculatorImage(K, np.zeros(3), np.zeros(3), Tcr_tmp, KeyPointsPSMDic[name]) for name in MeasurementDic.keys()}
        state_cov_prior = self.__state_cov + self.__R # 6 by 6
        state_mean_candidate = self.__state_mean
        R_candidate = self.__alpha * self.__R # 6*6
        measure_cov_candidate = self.__alpha * self.__measure_cov
        for name in MeasurementDic.keys():
            # Measurement Covariance Update
            # measure_cov_pre = self.__measure_cov_dic[name] # 2 by 2

            H = JacobianDic[name] # 2 by 6
            residual = ResidualDic[name].reshape(-1,1) # 2 by 1
            # measure_cov_new = self.__alpha * measure_cov_pre + (1-self.__alpha) * (residual@residual.T + H@state_cov_prior@H.T)
            measure_cov_new = self.__alpha * self.__measure_cov + (1-self.__alpha) * (residual@residual.T + H@state_cov_prior@H.T)
            measure_cov_candidate = measure_cov_candidate + (1-self.__alpha) * (residual@residual.T + H@state_cov_prior@H.T)

            self.__measure_cov_dic[name] = measure_cov_new # measurement cov update

            # Kalman Gain update
            S = np.matmul(np.matmul(H, state_cov_prior), H.T) + measure_cov_new
            K = state_cov_prior @ H.T @ np.linalg.inv(S)
            innovation = InnovationDic[name].reshape(-1,1)
            state_mean_candidate = state_mean_candidate + (K @ innovation).ravel()

            # State Covariance Update
            KH = K @ H
            mat1 = np.identity(KH.shape[0]) - KH
            # state_cov_candidate = mat1 @ state_cov_candidate
            state_cov_candidate = mat1 @ self.__state_cov
            R_candidate = R_candidate + (1-self.__alpha)*K@innovation@innovation.T@K.T

        self.__state_mean = state_mean_candidate.copy()
        self.__state_cov = state_cov_candidate.copy()
        self.__measure_cov = measure_cov_candidate.copy()
        self.__R = R_candidate.copy()
        self.__T00 = TFromThetaVecTVec(self.__state_mean[:3], self.__state_mean[3:])
        self.__Tcr = self.__Tcr_init @ self.__T00
    
    # Traditional EKF update, which returns Residual Dic
    def __EKF_measurement(self, PredictionDic, MeasurementDic, JacobianDic):
        # Within known correspondences between prediction and measurements
        state_mean_candidate = self.__state_mean.copy()
        state_cov_candidate = self.__state_cov.copy()
        if len(state_cov_candidate.shape) == 1:
            state_cov_candidate = np.diag(state_cov_candidate)

        # No associated measurements have been obtained
        if len(MeasurementDic) == 0:
            return
        DiscrepancyDic = {}
        for index, measurement in MeasurementDic.items():
            H = JacobianDic[index]
            prediction = PredictionDic[index]
            m1 = np.array(measurement)
            p1 = np.array(prediction)
            error = m1 - p1 # 2 by 1
            DiscrepancyDic[index] = error
            measurement_cov = self.__measure_cov_dic[index] # measurment cov associated with each keypoint observation
            S = np.matmul(np.matmul(H, state_cov_candidate), H.T) + measurement_cov
            # L_mat = np.linalg.cholesky(S)
            # K = state_cov_candidate @ H.T @ np.linalg.inv(L_mat.T) @ L_mat
            K = state_cov_candidate @ H.T @ np.linalg.inv(S)
            state_mean_candidate = state_mean_candidate + K @ error
            KH = K @ H
            mat1 = np.identity(KH.shape[0]) - KH
            state_cov_candidate = mat1 @ state_cov_candidate

        return state_mean_candidate, state_cov_candidate

    def ReturnStateEstimation(self):
        return self.__state_mean, self.__state_cov
    
    def ReturnT00Estimation(self):
        return self.__T00
    
    def ReturnTcrEstimation(self):
        return self.__Tcr
    
    def ReturnTcrInit(self):
        return self.__Tcr_init

        
