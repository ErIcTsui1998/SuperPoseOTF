# Reconstruct 3D position of key points after depth-matching 3D reconstruction

import numpy as np
import os
import cv2
import yaml

K_left = np.array([[1811.910046453570, 0.0, 588.5594517681759],
                    [0.0, 1809.640734154330, 477.3975900383616],
                    [0.0, 0.0, 1.0]])
fx,fy,cx,cy = K_left[0,0], K_left[1,1], K_left[0,2], K_left[1,2]
kc_0 = -0.266550275745263
kc_1 = 0.376985422203455 
kc_2 = 0.0007700651358933569
kc_3 = 0.0008052256575104228
kc_4 = 1.168032043658402
dist_coeffs_left = [kc_0,kc_1,kc_2,kc_3,kc_4]
dist_coeffs_left = np.array(dist_coeffs_left, dtype=np.float32)
def backproject_pixel_to_3D(u_d, v_d, Z, K, dist_coeffs=np.array([0,0,0,0,0])):
    """
    Back-project a distorted pixel with known depth to 3D camera coordinates.

    Parameters:
        u_d, v_d : distorted pixel coordinates (image coords)
        Z : depth at that pixel
        K : 3x3 camera matrix [[fx, 0, cx],
                               [0, fy, cy],
                               [0,  0,  1]]
        dist_coeffs : distortion coefficients [k1, k2, p1, p2, k3, ...]

    Returns:
        (X, Y, Z) in camera coordinates
    """

    # Step 1: Convert to homogeneous distorted point
    pts_distorted = np.array([[[u_d, v_d]]], dtype=np.float32)

    # Step 2: Undistort to normalized image coordinates
    pts_undistorted = cv2.undistortPoints(pts_distorted, K, dist_coeffs)

    # undistortPoints() returns normalized coords relative to fx, fy, cx, cy
    x_norm, y_norm = pts_undistorted[0, 0]

    # Step 3: Back-project using depth
    X = x_norm * Z
    Y = y_norm * Z

    return (X, Y, Z)

if __name__ == "__main__":   
    BaseFolder = "/home/zc519/Downloads/SurgPoseDataSet"
    os.chdir(BaseFolder)
    Subdir = os.listdir(BaseFolder)
    Subdir.sort()
    n_subdir = len(Subdir)
    for i in range(n_subdir):
        current_subdir = os.path.join(BaseFolder, Subdir[i]) 
        os.chdir(current_subdir)
        if "stereo" in os.listdir(current_subdir + "/regular"):
            print("3D reconstruction has been done")
            # Read keypoints_left file
            KP_pixel_left = {}
            with open("keypoints_left.yaml") as stream:
                try:
                    KP_pixel_left = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            KP_pos_3d_left_dic = {}

            # Read depth files
            DepthFolder = os.path.join(current_subdir,"regular/stereo/depth")
            DepthFiles = os.listdir(DepthFolder)
            DepthFiles.sort()
            assert len(KP_pixel_left) == len(DepthFiles)
            for j in range(len(KP_pixel_left)):
                KP_pixel_left_current = KP_pixel_left[j]
                Depth_map = np.load(os.path.join(DepthFolder,DepthFiles[j]))
                n_keypoints = len(KP_pixel_left[j])
                PosDic = {}
                for key, pixel_value in KP_pixel_left_current.items():
                    if pixel_value == None:
                        continue
                    u, v = pixel_value
                    # z = Depth_map[int(v),int(u)] * 1e-3 # (m)
                    # # x = (u-cx) * z / fx
                    # # y = (v-cy) * z / fy
                    # x,y,z = backproject_pixel_to_3D(u, v, z, K_left, dist_coeffs_left)
                    # PosDic[key] = [float(x),float(y),float(z)]
                    height, width = Depth_map.shape
                    Neighbours_pixel = [(int(u+kk), int(v+tt)) for kk in range(-3,3) for tt in range(-3,3) if u+kk < width and v+tt < height]
                    Neighbours_3d = [backproject_pixel_to_3D(pixel[0], pixel[1], Depth_map[pixel[1], pixel[0]],K_left) for pixel in Neighbours_pixel]
                    mean_3d = np.mean(np.array(Neighbours_3d),axis=0)*1e-3
                    x, y, z = mean_3d
                    PosDic[key] = [float(x),float(y),float(z)]
                KP_pos_3d_left_dic[j] = PosDic
            
            # Write
            output_name = os.path.join(current_subdir, "keypoints_left_3d.yaml")
            with open(output_name, "w") as f:
                yaml.dump(KP_pos_3d_left_dic, f, default_flow_style=False, sort_keys=False)
            print(f"{Subdir[i]} done!")

        else:
            continue # 3D depth information has not been retrieved

    print("Done")