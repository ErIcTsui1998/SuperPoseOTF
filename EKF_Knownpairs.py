import numpy as np
from Jacobian import RotX, RotY, RotZ, JacobianCalculator
from Kinematics import GetPositionInBaseFrame

class EKF_SuperDataSet:
    def __init__(self, state_mean_init, state_cov_init, measure_cov_init, Tcr_init):
        self.__state_cov = state_cov_init # 6*6
        self.__state_mean = state_mean_init # 6*1
        self.__measure_cov = measure_cov_init # 2*2 for pixels
        self.__Tcr_init = Tcr_init
        self.__Tcr = Tcr_init
        self.__T00 = np.identity(4)

    def UpdateParameters(self, new_state_mean, new_state_cov, new_measure_cov):
        self.__state_cov = new_state_cov
        self.__state_mean = new_state_mean
        self.__measure_cov = new_measure_cov

    def EKFReadMeasurement(self, PredictionDic, MeasurementDic, JacobianDic):
        # Within known correspondences between prediction and measurements
        state_mean_candidate = self.__state_mean.copy()
        state_cov_candidate = self.__state_cov.copy()
        measurement_cov = self.__measure_cov.copy()
        if len(state_cov_candidate.shape) == 1:
            state_cov_candidate = np.diag(state_cov_candidate)
        if len(measurement_cov.shape) == 1:
            measurement_cov = np.diag(measurement_cov) # this must be a symmetric matrix

        # No associated measurements have been obtained
        if len(MeasurementDic) == 0:
            return
        
        for index, measurement in MeasurementDic.items():
            H = JacobianDic[index]
            prediction = PredictionDic[index]
            m1 = np.array(measurement)
            p1 = np.array(prediction)
            error = m1 - p1 # 2 by 1
            S = np.matmul(np.matmul(H, state_cov_candidate), H.T) + measurement_cov
            # L_mat = np.linalg.cholesky(S)
            # K = state_cov_candidate @ H.T @ np.linalg.inv(L_mat.T) @ L_mat
            K = state_cov_candidate @ H.T @ np.linalg.inv(S)
            state_mean_candidate = state_mean_candidate + K @ error
            KH = K @ H
            mat1 = np.identity(KH.shape[0]) - KH
            state_cov_candidate = mat1 @ state_cov_candidate
        
        self.__state_mean = state_mean_candidate.copy()
        self.__state_cov = state_cov_candidate.copy()

        theta_z, theta_y, theta_x, tx, ty, tz = self.__state_mean
        self.__T00 = np.identity(4)
        self.__T00[:3,:3] = np.matmul(np.matmul(RotZ(theta_z), RotY(theta_y)), RotX(theta_x))
        self.__T00[:3,-1] = np.array([tx,ty,tz])
        self.__Tcr = self.__Tcr_init @ self.__T00

    def ReturnStateEstimation(self):
        return self.__state_mean, self.__state_cov
    
    def ReturnT00Estimation(self):
        return self.__T00
    
    def ReturnTcrEstimation(self):
        return self.__Tcr
    
    def ReturnTcrInit(self):
        return self.__Tcr_init

        
