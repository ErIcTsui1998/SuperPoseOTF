Intial hand-eye calibrations have been conducted via SolvePnPRansac <br>
Both PSM1 and PSM3 instruments are present <br>
Landmarks are chosen as all possible key points except for gripper tips. <br>
What to test: <br>
1. Noisy data keypoint observations <br>
2. How good it can perform when static camera positions. <br>
3. In the face of kinematics labelling error. <br>

Date: 25/09/2025 <br>
Initial Hand-eye calibration conducted via PnP <br>
Aim: Evaluate whether the algorithm works out for both PSM1 and PSM3 calibration <br>
Evaluation criteria: 3D position for all labelled key points. <br>

Modifications: <br>
1. Assume that no labels are available, for double instruments detection. <br>
2. Date: 3rd October. Gripper angles are availble now, and hence more keypoints can be used for analysis. <br>

Notes: <br>
1. The EKF MC method is barely useful in that when sigma is chosen as a small value, singularity error is flagged. However, when large sigma values are selected, the correction is not effective, because error kernel values are close to 1 (saturation).  

What's new: <br>
Evaluate the performance of different filters in the face of sudden disturbances
