import numpy as np

def Modified_DH(alpha, a, theta, d):
    T = np.identity(4)
    T[0,:] = [np.cos(theta), -np.sin(theta), 0, a]
    T[1,:] = [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)]
    T[2,:] = [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)]
    T[3,:] = [0.0, 0.0, 0.0, 1.0]
    return T

# Input: current joint angles
# Output: DH transformation matrix from the base frame to the end frame, indicated by end_joint
# dVRK Si Forward Kinematics
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

# Get a rigid transformation matrix from pts1 to pts2
# To get Tcr, pts1: pos in robot frame; pts2: pos in camera frame
def get_rigid_transform(pts1, pts2):
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    mean1 = pts1.mean(axis=0)
    mean2 = pts2.mean(axis=0)
    pts1 = np.array([p - mean1 for p in pts1])
    pts2 = np.array([p - mean2 for p in pts2])
    # if option=='clouds':
    H = pts1.T.dot(pts2)   # covariance matrix
    U,S,V = np.linalg.svd(H)
    V = V.T
    R = V.dot(U.T)
    t = -R.dot(mean1.T) + mean2.T
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, -1] = t
    T[-1, -1] = 1
    return T

def QuaternionToRot(quat):
    x,y,z,w = quat
    Rot = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                    [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                    [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
    return Rot

# Merge matrices diagonally 
# inputs [A,B,C..], output diag(A,B,C)
def DiagMergeMats(mat_list):
    n_mat = len(mat_list)
    if n_mat == 0:
        return
    n_cols = 0
    n_rows = 0
    for mat in mat_list:
        n_cols += mat.shape[0]
        n_rows += mat.shape[1]
    output = np.zeros((n_cols, n_rows))
    col_pos = 0
    row_pos = 0
    for mat in mat_list:
        mat_col = mat.shape[0]
        mat_row = mat.shape[1]
        output[col_pos:col_pos+mat_col, row_pos:row_pos+mat_row] = mat
        col_pos += mat_col
        row_pos += mat_row
    return output

# Make a nested dictionary yaml writable 
def MakeNumDicWritable(dic_ref):
    for key, value in dic_ref.items():
        if type(value) != dict:
            value_writable = [float(item) if type(item)!=str else item for item in value]
            dic_ref[key] = value_writable
        else:
            MakeNumDicWritable(value)
    return dic_ref

def FindRCMAnalyticalUtils(lines_3d):
    n_lines = len(lines_3d)
    if n_lines < 2:
        raise("number of lines is smaller than 2, cannot run the RCM estimation module")
    LHS_mat = np.zeros((3,3))
    RHS_vec = np.zeros(3)
    for i in range(n_lines):
        pt1, pt2 = lines_3d[i]
        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)
        n_vec = ((pt2-pt1) / np.linalg.norm(pt2-pt1)).reshape(-1,1)
        n_mat = n_vec @ n_vec.T
        mat_inc = n_mat - np.identity(3)
        vec_inc = mat_inc @ pt1
        LHS_mat = LHS_mat + mat_inc
        RHS_vec = RHS_vec + vec_inc
    rcm_estimated = np.linalg.solve(LHS_mat, RHS_vec)
    return rcm_estimated