import numpy as np
from Kinematics import dvrk_DH_transformation

class dvrk_arm:
    def __init__(self):
        self.js_his = []
        self.cp_t_his = []
        self.cp_R_his = []
        self.T_cr_his = []
        pass
    
    # order, js, cp_t, cp_r
    def ReadInputs(self, js_input, cp_t_input, cp_R_input):
        self.js_his = js_input
        self.cp_t_his = cp_t_input
        self.cp_R_his = [data.reshape(3,3) for data in cp_R_input]
        T_cr_his = []
        for i in range(len(js_input)):
            Tr6 = dvrk_DH_transformation(js_input[i], end_joint=6)
            Tc6 = np.identity(4)
            Tc6[:3,:3] = self.cp_R_his[i]
            Tc6[:3,-1] = cp_t_input[i]
            Tcr = Tc6 @ np.linalg.inv(Tr6)
            T_cr_his.append(Tcr)
        self.T_cr_his = T_cr_his

if __name__ == "__main__":   
    print("test")