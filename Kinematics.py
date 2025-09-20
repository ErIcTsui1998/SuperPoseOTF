import numpy as np
import cv2 as cv

def Modified_DH(alpha, a, theta, d):
    T = np.identity(4)
    T[0,:] = [np.cos(theta), -np.sin(theta), 0, a]
    T[1,:] = [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)]
    T[2,:] = [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)]
    T[3,:] = [0.0, 0.0, 0.0, 1.0]
    return T

# Input: current joint angles
# Output: DH transformation matrix from the base frame to the end frame, indicated by end_joint

def dvrk_DH_transformation(JointAngles, end_joint=6):
    assert np.size(JointAngles) >= 6, "Please input at least 6 joint angles"
    assert end_joint <= 7 and end_joint >= 1, "Please ensure 1 <= end_joint <= 7"
    L1 = 0.4318
    L2 = 0.4162
    L3 = 0.0091
    L4 = 0.0102
    q1,q2,q3,q4,q5,q6 = JointAngles[:6]
    T01 = Modified_DH(np.pi/2, 0.0, q1 + np.pi/2, 0.0)
    T12 = Modified_DH(-np.pi/2, 0.0, q2 - np.pi/2, 0.0)
    T23 = Modified_DH(np.pi/2, 0.0, 0.0, q3-L1)
    T34 = Modified_DH(0.0, 0.0, q4, L2)
    T45 = Modified_DH(-np.pi/2, 0.0, q5 - np.pi/2, 0.0)
    T56 = Modified_DH(-np.pi/2, L3, q6 - np.pi/2, 0.0)
    T67 = Modified_DH(-np.pi/2, 0.0, 0.0, L4)
    
    T02 = np.matmul(T01,T12)
    T03 = np.matmul(T02,T23)
    T04 = np.matmul(T03,T34)
    T05 = np.matmul(T04,T45)
    T06 = np.matmul(T05,T56)
    T07 = np.matmul(T06,T67)

    if end_joint == 1:
        return T01
    elif end_joint == 2:
        return T02
    elif end_joint == 3:
        return T03
    elif end_joint == 4:
        return T04
    elif end_joint == 5:
        return T05
    elif end_joint == 6:
        return T06
    elif end_joint == 7:
        return T07
    else:
        print("There is nothing to calculate, please monitor inputs")
        return

def GetPositionInBaseFrame(JointAngles, relative_position, relative_joint):
    T_0x = dvrk_DH_transformation(JointAngles, relative_joint)
    relative_position_homo = np.append(relative_position, 1.0)
    absolute_position_homo = np.matmul(T_0x, relative_position_homo)
    return absolute_position_homo[:3]

def GetPositionInCameraFrame(Tcr, position_base):
    return np.matmul(Tcr[:3,:3],position_base) + Tcr[:3,-1]

if __name__ == "__main__":
    # input_angles = np.load("./"+"inputs/q_dvrk_history.npy")
    input_angles = np.loadtxt("./"+"inputs/Grasp4_jp_sync.txt")
    assert np.size(input_angles) >= 0, "Cannot read input joint angles"
    JointAngle_t0 = input_angles[0,:]
    
    roll_front_rel = np.array([-0.004, 0.0, -0.00625]) # Link: 4
    roll_back_rel = np.array([0.004, 0.0, -0.00625]) # Link: 4
    roll_right_rel = np.array([0.0, 0.004, 0.0]) # Link: 4
    roll_left_rel = np.array([0.0, -0.004, 0.0]) # Link: 4
    pitch_front_rel = np.array([0.00275, -0.00275, -0.00025]) # Link: 5
    pitch_back_rel = np.array([0.00275, 0.00275, 0.00025]) # Link: 5
    pitch_right_rel = np.array([0.0035, -0.0015, 0.003]) # Link: 5
    pitch_left_rel = np.array([0.0035, 0.0015, -0.003]) # LInk: 5
    ee_front_rel = np.array([0.0, 0.0, -0.00275]) # Link: 6
    ee_back_rel = np.array([0.0, 0.0, 0.00275]) # Link: 6

    # Initial hand-eye calibration result
    T_cr_init = np.identity(4)
    hand_eye_rvec = np.array([9.7843378782272339e-01, -2.4118134975433350e+00, 1.1228071451187134e+00])
    hand_eye_tvec = np.array([9.3805999755859375e-02, -5.6443046569824219e-02, 7.6213455200195312e-04])
    T_cr_init[:3,:3] = cv.Rodrigues(hand_eye_rvec)[0]
    T_cr_init[:3,-1] = hand_eye_tvec

    # Coordinate transformation
    ee_front_base = GetPositionInBaseFrame(JointAngle_t0, ee_front_rel, 6)
    T04 = dvrk_DH_transformation(JointAngle_t0, 4)
    
    print(f"T04 = \n {T04}")