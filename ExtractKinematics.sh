#!/bin/bash
# This script lists all directories in the current folder
target_dir="/home/zc519/Downloads/SurgPoseDataSet"
echo "Directories in $target_dir:"
for d in "$target_dir"/*; do
    if [ -d "$d" ]; then
        echo "$(basename "$d")"
        python3 SuperPose_Kinematics.py --id "$(basename "$d")"
    fi
done

