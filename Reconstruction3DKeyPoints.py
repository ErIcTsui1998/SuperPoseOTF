# Reconstruct 3D position of key points after depth-matching 3D reconstruction

import numpy as np
import os
import yaml

K_left = np.array([[1811.910046453570, 0.0, 588.5594517681759],
                    [0.0, 1809.640734154330, 477.3975900383616],
                    [0.0, 0.0, 1.0]])
fx,fy,cx,cy = K_left[0,0], K_left[1,1], K_left[0,2], K_left[1,2]

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
                    u, v = pixel_value
                    z = Depth_map[v,u] * 1e-3 # (m)
                    x = (u-cx) * z / fx
                    y = (v-cy) * z / fy
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