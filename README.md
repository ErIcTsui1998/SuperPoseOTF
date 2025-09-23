Apply EKF estimator based on Maximum Correntropy onto the Super Pose dataset to rectify initial hand-eye calibration errors
Different from traditional EKF, Correntropy EKF claimed to possess advantages of eliminating Gaussian error assumption
In another words, if the algorithm works out well, good alignment results can still be expected even though the initial hand-eye calibration error is large.

From previous testing, it suggests that when initial hand-eye calibration error is large, key point association is not enough to bring the estimation back on track.
Here both keypoints and edge feature correspondences are used

Relevant papers link:\\
https://arxiv.org/pdf/1509.04580 \\
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7727408 \\
https://github.com/zijianwu1231/SurgPose \\
https://arxiv.org/pdf/2502.11534

Date: 23/09/2025 <br>
Initial Hand-eye calibration conducted via PnP <br>
Aim: Evaluate whether the algorithm works out for PSM3 calibration
