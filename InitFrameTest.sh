Init_Frame_num_list=(10 50 100 150 200)
# Filter_list=("EKF" "AEKF" "PF")
Filter_list=("EKF" "AEKF" "PF")
Dir_list=("000032")
# Loop through each item
for dir_id in "${Dir_list[@]}"; do
	for filter in "${Filter_list[@]}"; do
		for frame_num in "${Init_Frame_num_list[@]}"; do
		    echo "Dir: $dir_id , Processing: $filter , init frame = $frame_num"
		    python3 SuperPose_EKF_test.py --id $dir_id --filter $filter --InitFrame $frame_num
		done
	done
done

