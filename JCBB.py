# Create a python class for implementing the Joint Compatibility Branch and Bound algorithm
# Paper source: https://ieeexplore.ieee.org/abstract/document/976019
# C++ code source: https://github.com/jinkunw/mhjcbb/blob/master/da.h#L387 
# Author: Zejian Cui
# Date: 21/08/2025
import numpy as np
from scipy.stats import chi2
from typing import NamedTuple
import math

# Create a data structure for Innovation
class Innovation(NamedTuple):
    key: int # landmark index
    error: np.array # 1d numpy array
    H: np.array # Jacobian w.r.t key point or w.r.t edge
    sigmas: np.array # measurement noise
    md: float # Mahalanobis distance

class JCBB:
    def __init__(self, LandmarkDic):
        self.state_cov = [] # size: 6*6 for r translations and 3 euler angle variables
        self.measure_cov_point = []
        self.LandmarkDic = LandmarkDic # 1:rf; 2:rl; 3:rr; 4:pf
        self.num_landmark = len(self.LandmarkDic)
        self.PredictedFeatureValues = {} # 
        self.JacobianValues = {} # 1:H1; 2:H2 ... 

        self.KeyVector = []
        self.__Innovations = [] # [[potential landmarks for measure 1], [potential landmarks for measure 2], ..., [potential landmarks for measure i]]
        self.__best_hypothesis = []
        self.__distance = math.inf

############################### Public functions ########################################
# Add predicted feature values (key point positions etc.)
    def ReadPredictedFeatureValues(self, valueDic, JacobianDic, cov_state):
        valueDic = [np.array(value) for value in valueDic]
        self.PredictedFeatureValues = valueDic
        self.JacobianValues = JacobianDic
        self.state_cov = cov_state
        return

    def Clear(self):
        self.__Innovations = []
        self.__best_hypothesis = []
        self.__distance = math.inf

        self.PredictedFeatureValues = {}
        self.JacobianValues = {}

# Add current measurement with noise model, with only diagonal noise model supported.
    def ReadMeasurementFeatureValues(self, valueList, cov_measure_point, visibility_score_dic = None):
        # Search landmark candidates that is likely to be independently compatible.
        # valueList is index-free
        # weight_list = {1:0.9, 2:1.03, 3:1.03, 4:0.9, 5:1.04, 6:1.04, 7:0.9}
        # weight_list = {1:1.1, 2:1.00, 3:1.3, 4:1.1, 5:1.00, 6:1.1, 7:1.01}
        n_measurements = len(valueList)
        valueList = [np.array(value) for value in valueList]
        
        InvisibleKeyList = []
        if visibility_score_dic is not None:
            InvisibleKeyList =[key for key, value in visibility_score_dic.items() if value <=1e-3]

        for i in range(n_measurements):
            self.__Innovations.append([])
            current_measure = valueList[i]
            for index in self.LandmarkDic.keys():
                if index in InvisibleKeyList:
                    continue
                inn_error = current_measure - self.PredictedFeatureValues[index]
                inn_md = math.inf
                jacobian = self.JacobianValues[index]
                inn_sigma = cov_measure_point

                # if visibility_score_dic is not None:
                #     vscore = visibility_score_dic[index]
                #     print(f"score list = {visibility_score_dic.values()}")
                #     scale = self.Logistic(vscore, max(visibility_score_dic.values()))
                #     inn_sigma = inn_sigma * scale * 0.1

                inn = Innovation(key=index, error=inn_error, H=jacobian, sigmas=inn_sigma, md=inn_md)
                Indie_result, inn = self.__jc_(inn)
                if Indie_result:
                    self.__Innovations[i].append(inn)
        return
# Perform JCBB data association and return landmark keys for measurements.
    def ReturnMatchingKeys(self):
        Innovations_sorted = []
        for inns in self.__Innovations:
            if len(inns) > 0:
                inns_sorted = sorted(inns, key=lambda x: x.md)
                Innovations_sorted.append(inns_sorted)
            else:
                Innovations_sorted.append([])
        self.__Innovations = Innovations_sorted

        # Recursive search in interpretation tree starting with an empty hypothesis
        self.__JCBB([])
        MatchedKeys = [self.LandmarkDic[hypothesis.key] if hypothesis is not None else None for hypothesis in self.__best_hypothesis]
        MatchedKeys = [hypothesis.key if hypothesis is not None else None for hypothesis in self.__best_hypothesis]
        return MatchedKeys

############################### Private functions ######################################
# Joint Compatibility Branch and Bound
# One hypothesis candidate = [None, 1, 2, 10, 5], where each entry corresponds to a potentially associated landmark
    def __JCBB(self, hypothesis):
        k = len(hypothesis)
        h = self.__pairings(hypothesis)
        # when the length of hypothesis has reached the number of observations/measurements
        if k == len(self.__Innovations):
            __, score, score2 = self.__JC(hypothesis)
            num_best_hypothesis_paring = self.__pairings(self.__best_hypothesis)
            # if self.__pairings(self.__best_hypothesis) == 0 or (h >= self.__pairings(self.__best_hypothesis) and score <= self.__distance): # either all measurements are spurius, or the current hypothesis is better than the previously selected best hypothesis
            if num_best_hypothesis_paring == 0 or h > num_best_hypothesis_paring:
                self.__best_hypothesis = hypothesis.copy()
                self.__distance = score
                print(f"h = {h}, ML = {score}, distance = {score2}")
                print([item.key if item is not None else None for item in hypothesis])
                return
            elif h == num_best_hypothesis_paring:
                if score <= self.__distance:
                    self.__best_hypothesis = hypothesis.copy()
                    self.__distance = score
                    print(f"h = {h}, ML = {score}, distance = {score2}")
                    print([item.key if item is not None else None for item in hypothesis])
                    return 
            else:
                return     

        else:
            existing = [item.key for item in hypothesis if item is not None]
            current_best_hypothesis = self.__best_hypothesis.copy()
            # Same with below but with a null pairing
            if self.__Innovations[k] == []:
                remaining = set({})
                for j in range(k+1, len(self.__Innovations)):
                    for item in self.__Innovations[j]:
                        if item.key not in existing:
                            remaining.add(item.key)
                max_remaining = min(len(remaining), len(self.__Innovations)-k)
                if len(current_best_hypothesis) == 0 or h+max_remaining >= self.__pairings(current_best_hypothesis):
                    extended = hypothesis + [None]
                    self.__JCBB(extended)
            
            for inn in self.__Innovations[k]:
                # make sure that keys are used only once
                if inn.key in existing:
                    continue
                remaining = set({}) # set
                # Get remaining keys if we associate the k-th measurement with inn.key
                for j in range(k+1, len(self.__Innovations)):
                    for item in self.__Innovations[j]:
                        if item.key != inn.key and (item.key not in existing):
                            remaining.add(item.key)
                # print(f"k = {k}, existing = {existing}, remaining = {remaining}")
                # Caluclate the max pairings (upper bound) we can achieve with this association
                max_remaining = min(len(remaining)+1, len(self.__Innovations)-k)
                # Stop searching if upper bound <= current lower bound
                if h + max_remaining < self.__pairings(current_best_hypothesis):
                    continue
                # Keep searching in interpretation tree if current hypothesis is JC
                extended = hypothesis + [inn]
                if self.__JC(extended)[0]:
                    self.__JCBB(extended)
            
            # Else append a Null hypothesis 
            non_extended = hypothesis + [None]
            if self.__JC(non_extended)[0]:
                self.__JCBB(non_extended) 
        
# Count non None items in a list
    def __pairings(self, vector):
        return sum(1 for item in vector if item is not None)

# Fast independent compatibility (IC) test. gated Mahalanobis test
    def __jc_(self, inn_obj, desired_confidence_level = 0.85):
        cov_measure = np.asarray(inn_obj.sigmas) # 3*3
        # Different covariances for different measurements
        if cov_measure.ndim == 1:
            cov_measure = np.diag(cov_measure)
        error_inn = inn_obj.error # 1d 
        assert len(error_inn) == np.shape(cov_measure)[0], "please check the dimension of error against cov_measure"
        degrees_of_freedom = len(error_inn)
        G_mat = inn_obj.H # 3*6
        P_mat = self.state_cov # 6*6
        S_mat = np.matmul(np.matmul(G_mat, P_mat), G_mat.T) + cov_measure
        md_inn = np.matmul(error_inn, np.matmul(np.linalg.inv(S_mat),error_inn))
        inn_obj = Innovation(inn_obj.key, inn_obj.error, inn_obj.H, inn_obj.sigmas, md_inn)
        cdf_at_x = chi2.ppf(desired_confidence_level, df=degrees_of_freedom)
        return (True, inn_obj) if md_inn<= cdf_at_x else (False, inn_obj)

# Joint compatibility (JC) test.
    def __JC(self, inn_obj_list, desired_confidence_level = 0.85):
        # If inn_obj_list is full of None objects
        if all(v is None for v in inn_obj_list):
            return (True, np.inf, np.inf)
        
        combined_error_vec = []
        combined_Jacobian = []
        combined_R = []
        for inn_obj in inn_obj_list:
            if inn_obj == None:
                continue
            error = inn_obj.error
            Jacobian = inn_obj.H
            cov_measure = np.asarray(inn_obj.sigmas)
            if cov_measure.ndim == 1:
                cov_measure = np.diag(cov_measure)
            combined_error_vec.append(error)
            combined_Jacobian.append(Jacobian)
            combined_R.append(cov_measure)
        combined_R = self.__BlockDiagMatrix(combined_R)
        combined_error_vec = np.concatenate(combined_error_vec)
        combined_Jacobian = np.concatenate(combined_Jacobian, 0) #3n*6
        P_mat = self.state_cov # 6*6
        S_mat = np.matmul(np.matmul(combined_Jacobian, P_mat), combined_Jacobian.T) + combined_R
        L_mat = np.linalg.cholesky(S_mat)
        w_vec = np.linalg.inv(L_mat) @ combined_error_vec
        J_value = w_vec@w_vec
        # n_DOF = combined_error_vec.ndim
        n_DOF = len(combined_error_vec)
        cdf_at_x = chi2.ppf(desired_confidence_level, df=n_DOF)

        # Negative logarithm of the matching likelihood 
        ML = n_DOF*np.log(2*np.pi) + J_value + np.log(np.linalg.det(S_mat))
        # part1 = 1/(np.sqrt(np.power(2*np.pi, n_DOF) * np.linalg.det(S_mat) ))
        # part2 = np.exp(-1/2 * J_value)
        # prob = part1 * part2
        return (True,ML,J_value) if J_value<= cdf_at_x else (False,ML,J_value)
        # return (True,J_value) if J_value<= cdf_at_x else (False,J_value)
    

    def test(self):
        self.__JCBB()

    def Logistic(self, x, L, A=1.0, B=1.0):
        return L/ (1+A*np.exp(B*x))
    
# Block tile a list of n*n diag matrices
    def __BlockDiagMatrix(self,BlockList):
        n_blocks = len(BlockList)
        row_size_list = [np.shape(block)[0] for block in BlockList]
        col_size_list = [np.shape(block)[1] for block in BlockList]
        BlockMat = np.zeros((sum(row_size_list), sum(col_size_list)))
        start_i=0; start_j=0
        for i in range(n_blocks):
            n_row = row_size_list[i]
            n_col = col_size_list[i]
            BlockMat[start_i:start_i+n_row, start_j:start_j+n_col] = BlockList[i]
            start_i += n_row
            start_j += n_col
        return np.array(BlockMat)

if __name__ == "__main__":    
    LandmarkName = ["rf","rl","rr","pf","pl","pr","ef"]
    LandmarkValue = [1,2,3,4,5,6,7] # Outliers are denoted 0
    LandmarkDic = dict(zip(LandmarkName, LandmarkValue))
    LandmarkDicInv = dict(zip(LandmarkValue, LandmarkName))
    
    # Initialisation
    JCBB_obj1 = JCBB(LandmarkDicInv)
    # Validations
    Measurements = {}
    Measurements["rf"] = np.array([0.02231985514292228, -0.028333477747449848, 0.06552147013106002])   #1
    Measurements["rl"] = np.array([0.006331648814454659, -0.024067209183876722, 0.07277191024468112]) #2
    Measurements["rr"] = np.array([0.013059595300100743, -0.029663358001791623, 0.06347553649900363]) #3
    Measurements["pf"] = np.array([0.00860155233878352, -0.022179786103851632, 0.07324409038413715])  #4
    Measurements["pl"] = np.array([0.015617459919255738, -0.012968629834999317, 0.06991965167818624]) #5
    Measurements["pr"] = np.array([0.005310962297901286, -0.0331625056303602, 0.07250464620642182]) #6
    Measurements["ef"] = np.array([0.004393115355991871, -0.014185651401463849, 0.08164936029858334]) #7

    IndexList = list(Measurements.keys())
    np.random.shuffle(IndexList)
    # MeasurementsInput = [Measurements[index] for index in IndexList]
    MeasurementsInput = [(Measurements["rf"]+Measurements["pf"])/2, Measurements["pf"], Measurements["ef"]]
    
    # GroundTruthID = [LandmarkDic[index] for index in IndexList]
    GroundTruthID = [None, 4, 7]
    print(f"Correct order is = {GroundTruthID}")

    # Jacobians
    Jacobians = {}
    Jacobians["rf"] = np.array([[-0.0334524,   0.04928013, -0.04788387, -0.72051883, -0.69336428,  0.00992907],
                                [ 0.04974189,  0.09170661,  0.00952744, -0.45394959,  0.46080703, -0.76261829],
                                [-0.05916488,  0.01460315, -0.05756663,  0.52419689, -0.55398814, -0.64677258]])
    Jacobians["rl"] = np.array([[-0.03036196,  0.05422878, -0.05260344, -0.72051883, -0.69336428,  0.00992907],
                                [ 0.0522052,   0.0952666,   0.00900295, -0.45394959,  0.46080703, -0.76261829],
                                [-0.06202196,  0.01137068, -0.06450798,  0.52419689, -0.55398814, -0.64677258]])
    Jacobians["rr"] = np.array([[-0.03258508,  0.04973598, -0.0483106,  -0.72051883, -0.69336428,  0.00992907],
                                [ 0.05515265,  0.09616032,  0.00495855, -0.45394959,  0.46080703, -0.76261829],
                                [-0.06553148,  0.0177372,  -0.0621098,   0.52419689, -0.55398814, -0.64677258]])
    Jacobians["pf"] = np.array([[-0.03464006,  0.05346976, -0.05193197, -0.72051883, -0.69336428,  0.00992907],
                                [ 0.05421004,  0.0988143,   0.00923617, -0.45394959,  0.46080703, -0.76261829],
                                [-0.06445158,  0.01527163, -0.06338307,  0.52419689, -0.55398814, -0.64677258]])
    Jacobians["pl"] = np.array([[-0.03011979,  0.05518105, -0.05351648, -0.72051883, -0.69336428,  0.00992907],
                                [ 0.05459257,  0.09778076,  0.00754438, -0.45394959,  0.46080703, -0.76261829],
                                [-0.06483323,  0.01227011, -0.06702607,  0.52419689, -0.55398814, -0.64677258]])
    Jacobians["pr"] = np.array([[-0.0345703,   0.05193173, -0.05045094, -0.72051883, -0.69336428,  0.00992907],
                                [ 0.05606772,  0.09939882,  0.0067087,  -0.45394959,  0.46080703, -0.76261829],
                                [-0.06664092,  0.01768282, -0.06353614,  0.52419689, -0.55398814, -0.64677258]])
    Jacobians["ef"] = np.array([[-0.03629073,  0.05618505, -0.05456768, -0.72051883, -0.69336428,  0.00992907],
                                [ 0.05729793,  0.10406071,  0.00937429, -0.45394959,  0.46080703, -0.76261829],
                                [-0.06811789,  0.01623717, -0.06688629,  0.52419689, -0.55398814, -0.64677258]])
    JacobiansInput = {LandmarkDic[id]:value for id, value in Jacobians.items()}

    # Predicted feature values
    PredictedFeatures = {}
    PredictedFeatures["rf"] = np.array([0.01582419, -0.02567796, 0.07071904])
    PredictedFeatures["rl"] = np.array([0.01198386, -0.01846367,  0.07278504])
    PredictedFeatures["rr"] = np.array([0.00746235, -0.02464077,  0.07046138])
    PredictedFeatures["pf"] = np.array([0.00884405, -0.0220031,   0.07539302])
    PredictedFeatures["pl"] = np.array([0.00828238, -0.01730089,  0.07345363])
    PredictedFeatures["pr"] = np.array([0.00599385, -0.02358638,  0.07395877])
    PredictedFeatures["ef"] = np.array([0.00401285, -0.02018484,  0.07909955])
    PredictedFeaturesInput = {LandmarkDic[id]:value for id, value in PredictedFeatures.items()}

    # Association based on distance
    l1 = MeasurementsInput.copy()
    l2 = list(PredictedFeatures.values())  
    distance_matrix = np.zeros(((len(l1)),len(l2)))
    for i in range(len(l1)):
        for j in range(len(l2)):
            distance_matrix[i,j] = np.linalg.norm(l1[i]-l2[j])
    indices = distance_matrix.argmin(axis=1)
    out = list(enumerate(indices+1))
    print(out)

    cov_state = 10 * np.diag([0.005, 0.005, 0.005, 0.25e-3, 0.25e-3, 0.25e-3])
    # cov_state = np.array([[0.05, 0.01, 0.005, 0.0, 0.0, 0.0],
    #                       [0.01, 0.05, 0.02, 0.0, 0.0, 0.0],
    #                       [0.005, 0.02, 0.05, 0.0, 0.0, 0.0],
    #                       [0.0, 0.0, 0.0, 0.25e-2, 0.3e-3, 0.2e-3 ],
    #                       [0.0, 0.0, 0.0, 0.3e-3, 0.25e-2, 0.1e-3],
    #                       [0.0, 0.0, 0.0, 0.2e-3, 0.1e-3, 0.25e-2]])

    # Let's assume that covariances for different measurements are different 
    # 1,4,7 low covariances
    # 2,3,5,6 higher variances
    cov_measure = np.diag([5e-3, 5e-3, 5e-3]) 
    JCBB_obj1.ReadPredictedFeatureValues(PredictedFeaturesInput, JacobiansInput, cov_state)
    JCBB_obj1.ReadMeasurementFeatureValues(MeasurementsInput, cov_measure)

    OutputMatchedKeys = JCBB_obj1.ReturnMatchingKeys()

    # PredictedValues = {"rf":[np.array([1,0,0]), np.array([[1,2],[3,4]])], "rl":[np.array([2,0,0]), np.array([[5,6],[7,8]])]}
    


    print("JCBB test")
    print(LandmarkDic)
