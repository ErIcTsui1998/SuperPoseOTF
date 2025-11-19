Init_Frame_num_list=(100)
# Filter_list=("EKF" "AEKF" "PF")
Filter_list=("EKF" "AEKF" "PF")
Dir_list=("000006")
Level_list=("Low" "Medium" "High")
# Loop through each item
for dir_id in "${Dir_list[@]}"; do
	for filter in "${Filter_list[@]}"; do
		for frame_num in "${Init_Frame_num_list[@]}"; do
			for level in "${Level_list[@]}"; do
			    echo "Dir: $dir_id , Processing: $filter , init frame = $frame_num, level = $level"
			    if [ "$level" == "Low" ]; then
			    	python3 SuperPose_EKF_test.py --id $dir_id --filter $filter --InitFrame $frame_num --Level $level --transerror 1 --roterror 1
			    elif [ "$level" == "Medium" ]; then	
			    	python3 SuperPose_EKF_test.py --id $dir_id --filter $filter --InitFrame $frame_num --Level $level --transerror 3 --roterror 3
			    elif [ "$level" == "High" ]; then	
			    	python3 SuperPose_EKF_test.py --id $dir_id --filter $filter --InitFrame $frame_num --Level $level --transerror 5 --roterror 5
			    fi
			done
		done
	done
done

