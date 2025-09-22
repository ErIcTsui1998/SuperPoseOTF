cd "/home/zc519/Downloads/SurgPoseDataSet"
echo "Directories in $(pwd):"
for d in */ ; do
    if [ -d "$d" ]; then
        echo "$d"
        if [ -d "$d/HandEye" ];then
		echo "HandEye folder exists"     
        else
        	echo "HandEye not found"
        	folder_name=$(basename "$d")
        	python3 /home/zc519/Projects/SuperPose_OTF/SuperPose_EKF_test.py --id "$folder_name"
        	echo "$folder_name"
        fi
    fi
done
