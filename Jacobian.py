import numpy as np
# from sympy import sin, cos, Matrix
# from sympy.abc import rho, phi
# import sympy as sp
from utils_vision import LineABCToHesseForm

def RotX(theta):
    T = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(theta), -np.sin(theta)],
                  [0.0, np.sin(theta), np.cos(theta)]])
    return T

def RotY(theta):
    T = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(theta), 0.0, np.cos(theta)]])
    return T

def RotZ(theta):
    T = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    return T

def RotFromThetaVec(theta_vec):
    theta_z, theta_y, theta_x = theta_vec
    return RotZ(theta_z) @ RotY(theta_y) @ RotX(theta_x)

def TFromThetaVecTVec(theta_vec, t_vec):
    T = np.identity(4)
    T[:3,:3] = RotFromThetaVec(theta_vec)
    T[:3,-1] = t_vec
    return T

# theta_vec = [theta_z, theta_y, theta_x], Z-Y-X Euler angles
# t_vec = [t_x, t_y, t_z]
# T0_{4*4}: initial 4*4 hand-eye calibration matrix
# t2_{3*1}: marker position in the PSM base frame
# Return Jacobian matrix (3*6 matrix, or 3n*6 matrix)
def JacobianCalculator(theta_vec, t_vec, T0, t2):
    theta_z, theta_y, theta_x = theta_vec
    t_x, t_y, t_z = t_vec
    R0 = T0[:3,:3]
    t0 = T0[:3,-1]
    Rx = RotX(theta_x)
    Ry = RotY(theta_y)
    Rz = RotZ(theta_z)

    # Jacobian 4-6th column, w.r.t tx, ty, tz
    H_Jacobian = np.zeros((3,6))
    H_Jacobian[:,3:] = R0

    # Jacobian 1st column, w.r.t theta_z
    R_left = R0
    b_right = np.matmul(np.matmul(Ry,Rx), t2)
    bx, by, bz = b_right
    b_diff = np.array([-bx*np.sin(theta_z)-by*np.cos(theta_z), bx*np.cos(theta_z)-by*np.sin(theta_z), 0.0])
    H_Jacobian[:,0] = np.matmul(R_left, b_diff)

    # Jacobian 2nd column, w.r.t theta_y
    R_left = np.matmul(R0,Rz)
    b_right = np.matmul(Rx, t2)
    bx, by, bz = b_right
    b_diff = np.array([-bx*np.sin(theta_y)+bz*np.cos(theta_y), 0.0, -bx*np.cos(theta_y)-bz*np.sin(theta_y)])
    H_Jacobian[:,1] = np.matmul(R_left, b_diff)

    # Jacobian 3rd column, w.r.t theta_x
    R_left = np.matmul(np.matmul(R0, Rz), Ry)
    b_right = t2
    bx, by, bz = b_right
    b_diff = np.array([0.0, -by*np.sin(theta_x)-bz*np.cos(theta_x), by*np.cos(theta_x)-bz*np.sin(theta_x)])
    H_Jacobian[:,2] = np.matmul(R_left, b_diff)
    
    return H_Jacobian # 3*6

# t2 is the position of key point in the robot frame
def JacobianCalculatorImage(K_mat, theta_vec, t_vec, T0, t2):
    theta_z, theta_y, theta_x = theta_vec
    Rx = RotX(theta_x)
    Ry = RotY(theta_y)
    Rz = RotZ(theta_z)
    fx = K_mat[0,0]
    fy = K_mat[1,1]
    cx = K_mat[0,2]
    cy = K_mat[1,2]
    T0r = np.identity(4) # 
    T0r[:3,:3] = Rz@Ry@Rx
    T0r[:3,-1] = t_vec
    T00r = T0@T0r
    x,y,z = T00r[:3,:3]@t2 + T00r[:3,-1] # position in the camera frame
    Jacobian2 = JacobianCalculator(theta_vec, t_vec, T0, t2) # 3*6 matrix
    Jacobian1 = np.array([[fx/z, 0.0, -fx*x/(z*z)],
                          [0.0, fy/z, -fy*y/(z*z)]]) # 2*3 matrix
    output = Jacobian1@Jacobian2
    return output # the final 2*6 Jacobian matrix

############### Jacobian Calculator for Cylindrical Edge Features ###################
#************** !!!! Caveats: This function remains to be tested ********************
############### Use numerical method to find the Jacobian between (rho, theta) and (theta_x, theta_y, theta_z, tx, ty, tz)
# Required input: 
# K_mat: camera intrinsic; 
# theta_vec, t_vec : current states; 
# radius: instrument radius (m); 
# j3_pos_cam and j4_pos_cam are expressed in the PSM frame
# T0 initial hand_eye transformation matrix
def JacobianCalculatorEdge(K_mat, Tcr, theta_vec, t_vec, radius, j3_pos_PSM, j4_pos_PSM, h=1e-6):
    n_state = len(theta_vec) + len(t_vec)
    n_measurement = 4 # rho, theta for edge_down, and rho, theta for edge_up
    J = np.zeros((n_measurement, n_state))
    # Jacobian theta_vec part, the first 3 columns of J
    for j in range(len(theta_vec)):
        theta_upper = theta_vec.copy()
        theta_lower = theta_vec.copy()
        theta_upper[j] += h 
        theta_lower[j] -= h
        Tcr_upper = Tcr @ TFromThetaVecTVec(theta_upper, t_vec)
        Tcr_lower = Tcr @ TFromThetaVecTVec(theta_lower, t_vec)
        j3_pos_cam_upper = Tcr_upper[:3,:3] @ j3_pos_PSM + Tcr_upper[:3,-1]
        j3_pos_cam_lower = Tcr_lower[:3,:3] @ j3_pos_PSM + Tcr_lower[:3,-1]
        j4_pos_cam_upper = Tcr_upper[:3,:3] @ j4_pos_PSM + Tcr_upper[:3,-1]
        j4_pos_cam_lower = Tcr_lower[:3,:3] @ j4_pos_PSM + Tcr_lower[:3,-1]
        edge_down_lower, edge_up_lower = __EdgeProjectionCylinder(K_mat, radius, j3_pos_cam_lower, j4_pos_cam_lower)
        edge_down_upper, edge_up_upper = __EdgeProjectionCylinder(K_mat, radius, j3_pos_cam_upper, j4_pos_cam_upper)
        J[:2,j] = (edge_down_upper - edge_down_lower) / 2*h
        J[2:,j] = (edge_up_upper - edge_up_lower) / 2*h
    
     # Jacobian t_vec part, the last 3 columns of J
    for j in range(len(t_vec)):
        t_upper = t_vec.copy()
        t_lower = t_vec.copy()
        t_upper[j] += h 
        t_lower[j] -= h
        Tcr_upper = Tcr @ TFromThetaVecTVec(theta_vec, t_upper)
        Tcr_lower = Tcr @ TFromThetaVecTVec(theta_vec, t_lower)
        j3_pos_cam_upper = Tcr_upper[:3,:3] @ j3_pos_PSM + Tcr_upper[:3,-1]
        j3_pos_cam_lower = Tcr_lower[:3,:3] @ j3_pos_PSM + Tcr_lower[:3,-1]
        j4_pos_cam_upper = Tcr_upper[:3,:3] @ j4_pos_PSM + Tcr_upper[:3,-1]
        j4_pos_cam_lower = Tcr_lower[:3,:3] @ j4_pos_PSM + Tcr_lower[:3,-1]
        edge_down_lower, edge_up_lower = __EdgeProjectionCylinder(K_mat, radius, j3_pos_cam_lower, j4_pos_cam_lower)
        edge_down_upper, edge_up_upper = __EdgeProjectionCylinder(K_mat, radius, j3_pos_cam_upper, j4_pos_cam_upper)
        J[:2,3+j] = (edge_down_upper - edge_down_lower) / 2*h
        J[2:,3+j] = (edge_up_upper - edge_up_lower) / 2*h

    return J

# pos1, pos2 are two points on the centre of a cylinder, expressed in the camera frame
def __EdgeProjectionCylinder(K_mat, radius, pos1, pos2):
    fx = K_mat[0,0]; fy = K_mat[1,1]; cx = K_mat[0,2]; cy = K_mat[1,2]
    shaft_axis_cam = (pos1-pos2) / np.linalg.norm(pos1 - pos2)
    a, b, c = shaft_axis_cam
    x0, y0, z0 = pos1
    nu = a*x0 + b*y0 + c*z0
    C = x0*x0 + y0*y0 + z0*z0 - nu*nu - radius*radius
    alpha = c*y0-b*z0
    beta = a*z0-c*x0
    kappa = b*x0-a*y0
    edge_down_raw = [radius*(x0-a*nu)/np.sqrt(C)-alpha, radius*(y0-b*nu)/np.sqrt(C)-beta, radius*(z0-c*nu)/np.sqrt(C)-kappa]
    edge_up_raw = [radius*(x0-a*nu)/np.sqrt(C)+alpha, radius*(y0-b*nu)/np.sqrt(C)+beta, radius*(z0-c*nu)/np.sqrt(C)+kappa]
    edge_down = [edge_down_raw[0]/fx, edge_down_raw[1]/fy, edge_down_raw[2]-edge_down_raw[0]/fx*cx - edge_down_raw[1]/fy*cy]
    edge_up = [edge_up_raw[0]/fx, edge_up_raw[1]/fy, edge_up_raw[2]-edge_up_raw[0]/fx*cx - edge_up_raw[1]/fy*cy]
    edge_down_hesse = LineABCToHesseForm(edge_down)
    edge_up_hesse = LineABCToHesseForm(edge_up)
    return np.array(edge_down_hesse), np.array(edge_up_hesse)

if __name__ == "__main__":
    theta_t0 = np.array([np.pi/50, -np.pi/60, np.pi/100])
    t_vec = np.array([1e-3,1e-3,1e-3])
    T0 = np.identity(4)
    t2 = np.array([1e-2,0.0,-1e-2])
    H_t0 = JacobianCalculator(theta_t0, t_vec, T0, t2)
    print(f"Jacobian at t0 = \n {H_t0}")

    #