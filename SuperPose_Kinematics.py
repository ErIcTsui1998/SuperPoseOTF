import numpy as np
import pickle
import os
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, help='path to the input video file')
    args = parser.parse_args()
    dir_id = args.id

    BaseFolder = "/home/zc519/Downloads/SurgPoseDataSet"
    SubDataSet = os.path.join(BaseFolder, dir_id)
    LeftImagesFolder = os.path.join(SubDataSet, "regular/LeftImages")
    RightImagesFolder = os.path.join(SubDataSet, "regular/RightImages")
    os.chdir(SubDataSet)
    os.makedirs("Kinematics", exist_ok=True)
    cp_data_dic = {}
    jp_data_dic = {}
    with open("api_cp_data.yaml") as stream:
        try:
            cp_data_dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open("api_jp_data.yaml") as stream:
        try:
            jp_data_dic = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ArmNameList = list(jp_data_dic['0'].keys())
    assert len(cp_data_dic) == len(jp_data_dic), "number of recordings do not match"
    n_recording = len(cp_data_dic)
    os.chdir("Kinematics")
    for arm in ArmNameList:
        arm_jp_his = []
        arm_cp_R_his = []
        arm_cp_t_his = []
        print(f"Now processing arm {arm}")
        for i in range(n_recording):
            arm_jp = jp_data_dic[str(i)][arm]
            arm_cp_R = cp_data_dic[str(i)][arm]['R']
            arm_cp_t = cp_data_dic[str(i)][arm]['t']
            arm_jp_his.append(arm_jp)
            arm_cp_R_his.append(arm_cp_R)
            arm_cp_t_his.append(arm_cp_t)
        file_jp_name = arm+"_jp_his.txt"
        file_cp_R_name = arm+"_cp_R_his.txt"
        file_cp_t_name = arm+"_cp_t_his.txt"
        os.makedirs(arm, exist_ok=True)
        np.savetxt(os.path.join(arm, file_jp_name), arm_jp_his)
        np.savetxt(os.path.join(arm, file_cp_R_name), arm_cp_R_his)
        np.savetxt(os.path.join(arm, file_cp_t_name), arm_cp_t_his)
        print(f"Arm {arm} has been processed")
