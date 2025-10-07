# Implementation of Extended Kalman Filter under Maximum Correntropy Criterion
# Author: Zejian Cui
# Relevant paper link: https://ieeexplore.ieee.org/abstract/document/8736038 
# Date: 18/09/2025
# Notes: All parameter notations in this script conform to what was presribed in the original paper
import numpy as np
from Jacobian import RotX, RotY, RotZ, JacobianCalculator
from utils import DiagMergeMats

class EKF_MC_dVRKDataSet:
    def __init__(self, state_mean_init, state_cov_init, measure_cov_init, Tcr_init, KeyPointsNameList, bandwidth=0.5, threshold = 1e-6):
        self.__state_cov = state_cov_init*1e-3 # 6*6
        self.__state_mean = state_mean_init # 6*1
        # self.__measure_cov = measure_cov_init # 2*2 for pixels
        measure_cov = measure_cov_init if len(measure_cov_init.shape) == 2 else np.diag(measure_cov_init) # 2*2 for pixels
        self.__measure_cov_dic = {name: measure_cov for name in KeyPointsNameList}
        self.__measure_cov = measure_cov
        self.__Tcr_init = Tcr_init
        self.__Tcr = Tcr_init
        self.__T00 = np.identity(4)
        self.__sigma = bandwidth # kernel bandwidth
        self.__eps = threshold # epsilon

    def EKFReadMeasurement(self, PredictionDic, MeasurementDic, JacobianDic):
        # Within known correspondences between prediction and measurements
        state_mean_candidate = self.__state_mean.copy()
        state_cov_candidate = self.__state_cov.copy()
        n_state_param = len(state_mean_candidate) # 6 by default

        if len(state_cov_candidate.shape) == 1:
            state_cov_candidate = np.diag(state_cov_candidate)

        # No associated measurements have been obtained
        if len(MeasurementDic) == 0:
            return
        
        for index, measurement in MeasurementDic.items():
            measurement_cov = self.__measure_cov
            n_measure_param = measurement_cov.shape[0] # 2 by default
            H = JacobianDic[index] # 2*6 matrix          
            prediction = PredictionDic[index]
            m1 = np.array(measurement)
            p1 = np.array(prediction)
            psi = m1 - p1 # 2 by 1
            eta = H @ state_cov_candidate @ H.T + measurement_cov
            beta = psi.T @ np.linalg.inv(eta) @ psi

            # Initalisation before iteration
            state_mean_candidate_itr = state_mean_candidate.copy()
            state_cov_candidate_itr = state_cov_candidate.copy()
            distance = np.inf

            # Iteration block
            while distance >= 1e-7:
                Mp = np.linalg.cholesky(state_cov_candidate)
                Mr = np.linalg.cholesky(measurement_cov)
                M = DiagMergeMats([Mp, Mr]) # 6+2 by 6+2
                mat1 = np.vstack((np.identity(n_state_param) ,H)) # 6+2 by 6, by default
                D = np.linalg.inv(M) @ mat1 # 6+2 by 6
                vec1 = np.append(state_mean_candidate, psi + H @ state_mean_candidate)
                z_vec = np.linalg.inv(M) @ vec1 # (6+2 by 1 vector)
                e_vec = z_vec - D @ state_mean_candidate_itr # 6+2 by 1 vector
                sigma_obs_list = [self.__GK(e_vec[ii]) for ii in range(n_state_param, n_state_param + n_measure_param)]
                sigma_state_list = [self.__GK(e_vec[ii]) for ii in range(0, n_state_param)]
                C_obs = np.diag(sigma_obs_list)
                C_state = np.diag(sigma_state_list)
                R = Mr @ np.linalg.inv(C_obs) @ Mr.T
                P = Mp @ np.linalg.inv(C_state) @ Mp.T
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                KH = K @ H
                mat1 = np.identity(KH.shape[0]) - KH
                distance = np.linalg.norm(state_mean_candidate + K @ psi - state_mean_candidate_itr) / np.linalg.norm(state_mean_candidate_itr)
                state_mean_candidate_itr = state_mean_candidate + K @ psi
                state_cov_candidate_itr = mat1 @ state_cov_candidate @ mat1.T + K @ R @ K.T
                # state_cov_candidate_itr = mat1 @ state_cov_candidate
                print(f"distance is {distance}")
            state_mean_candidate = state_mean_candidate_itr
            state_cov_candidate = state_cov_candidate_itr
        
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

    # Gaussian Kernel function
    def __GK(self, error):
        return np.exp(-error**2 / (2*self.__sigma**2))
    
    