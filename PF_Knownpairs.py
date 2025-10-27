# Particle Filter Implementation with known correspondences
# Author: Zejian Cui
# Date: 14/09/2025
import numpy as np
from Jacobian import TFromThetaVecTVec
from utils_vision import PixelProjection

class PF_SuperDataSet:
    def __init__(self, state_mean_init, state_cov_init, measure_cov_init, Tcr_init, num_particles):
        self.__state_cov = state_cov_init*2e-3 if len(state_cov_init.shape) == 2 else 1e-3*np.diag(state_cov_init) # 6*6
        self.__state_mean = state_mean_init # 6*1
        self.__measure_cov = measure_cov_init if len(measure_cov_init.shape) == 2 else np.diag(measure_cov_init) # 2*2 for pixels
        self.__Tcr_init = Tcr_init
        self.__Tcr = Tcr_init
        self.__T00 = np.identity(4)
        self.__Neff = 0
        self.__num_particles = num_particles
        self.__particles = np.repeat(np.reshape(state_mean_init,(1,-1)), num_particles, axis=0)
        self.__weights = np.ones(num_particles, dtype=float) * 1/num_particles

    # Read Measurements and update particle weights (assume that data association is known)
    def PFReadMeasurement(self, MeasurementDic, KeyPointsPosPSMDic, K):
        if len(MeasurementDic) == 0:
            print("No matching pairs, PF")
            return
        KP_name_list = list(MeasurementDic.keys())
        n_KP_measured = len(KP_name_list)
        cov_mat = np.kron(np.eye(n_KP_measured), self.__measure_cov) # 2n*2n
        
        for i in range(self.__num_particles):
            self.__particles[i] = self.__particles[i] + self.__state_cov @ np.random.randn(6)
            theta_vec = self.__particles[i][:3]
            t_vec = self.__particles[i][3:]
            T00 = TFromThetaVecTVec(theta_vec, t_vec)
            T0r = T00 @ self.__Tcr_init
            error = []
            for name in KP_name_list:
                pos_PSM = KeyPointsPosPSMDic[name]
                pos_cam_pt = T0r[:3,:3]@pos_PSM + T0r[:3,-1]
                pixel_pt = np.array(PixelProjection(pos_cam_pt, K))
                pixel_measure = np.array(MeasurementDic[name])
                error.append(pixel_measure - pixel_pt) # pixel projection discrepancy
            error = np.array(error).reshape(1,-1) #1* 2n vector
            weight = 1.0/np.linalg.norm(error) + 1e-7
            # weight = float(-n_KP_measured/2 * np.log(2*np.pi) -0.5 * np.log(np.linalg.det(cov_mat)) - 0.5 * error @ np.linalg.inv(cov_mat) @ error.T)  # n-d Gaussian log likelihood
            self.__weights[i] *= weight
        self.__weights = self.__weights / np.sum(self.__weights)
        weight_square_list = [self.__weights[j]**2 for j in range(self.__num_particles)]
        self.__Neff = 1.0 / np.sum(weight_square_list)
    
    def ParticleResampling(self, N_eff_threshold=500):
        if self.__Neff >= N_eff_threshold:
            print("Still enough number of particles, no need to resample")
            self.__state_mean = np.average(self.__particles, axis=0, weights=self.__weights)
            self.__T00 = TFromThetaVecTVec(self.__state_mean[:3], self.__state_mean[3:])
            self.__Tcr = self.__T00 @ self.__Tcr_init
        else:
            # Resampling is needed. Stratified resampling, for example.
            indexes = self.__stratified_resampling(self.__weights)
            self.__particles = self.__particles[indexes]
            self.__weights = self.__weights[indexes]
            self.__state_mean = np.average(self.__particles, axis=0, weights=self.__weights)
            self.__T00 = TFromThetaVecTVec(self.__state_mean[:3], self.__state_mean[3:])
            self.__Tcr = self.__T00 @ self.__Tcr_init
            self.__weights = np.ones(self.__num_particles, dtype=float) * 1/self.__num_particles

    def ReturnStateEstimation(self):
        return self.__state_mean, self.__state_cov
    
    def ReturnT00Estimation(self):
        return self.__T00
    
    def ReturnTcrEstimation(self):
        return self.__Tcr
    
    def ReturnTcrInit(self):
        return self.__Tcr_init

    def __stratified_resampling(self, weights):
        if len(np.where(weights==0.0)[0]) >= self.__num_particles * 0.8:
            print("particle degeneration has occured")
            self.__weights.fill(1.0/self.__num_particles)
            indexes = np.arange(self.__num_particles, dtype=int)
            return indexes

        N = len(weights)
        positions = (np.arange(N) + np.random.uniform(size=N)) / N

        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes