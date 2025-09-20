import numpy as np
import cv2
import json
import os

KEYPOINTSTHRESHOLD = 0.05
BOXAREATHRESHOLD = 10

def OverlayKeyPoints(img, keypointslist, KeypointsNameList, color_default=(0,255,0)):
    overlay = img.copy()
    n_pts = np.shape(keypointslist)[0]
    for i in range(n_pts):
        x,y,score = keypointslist[i]
        name = KeypointsNameList[i]
        if score >= KEYPOINTSTHRESHOLD:
            overlay = cv2.circle(overlay, (int(x),int(y)), radius=5, color=color_default, thickness=-1)
            overlay = cv2.putText(overlay, name, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color_default, 2, cv2.LINE_AA)
    return overlay

def DrawBoundingBox(img, bbox, color_default=(0,255,0)):
    overlay = img.copy()
    x,y,dx,dy = bbox
    if dx*dy < BOXAREATHRESHOLD:
        return overlay
    else:
        overlay = cv2.rectangle(overlay, (int(x), int(y)), (int(x+dx), int(y+dy)), color_default, 2)
        return overlay

if __name__ == "__main__":
    dataset_path = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/"
    inst_ids = ["FBF_1", "FBF_2", "FBF_E_1"]
    # inst_ids = ["PCH_1", "PCH_2", "PCH_E_1"]
    id = -1
    train_file = dataset_path + inst_ids[id] + "/train.json"
    test_file = dataset_path + inst_ids[id] + "/test.json"
    
    # Open train.json
    train_data = {}
    with open(train_file) as f:
        train_data = json.load(f)
    
    n_training_data = len(train_data['images'])
    keypoints_name = train_data['categories'][0]['keypoints']
    image_data_list = train_data['images']
    image_data_list.sort(key=lambda item: item['id'])
    
    KeyPointsData_list = train_data['annotations']
    KeyPointsData_list.sort(key=lambda item: item['image_id'])

    for i in range(n_training_data):
        img_file = image_data_list[i]['file_name'].split("/")[-1]
        img_file = os.path.dirname(train_file) + "/images/" + img_file
        img_frame = cv2.imread(img_file)
        num_keypoints_frame = train_data['annotations'][i]['num_keypoints']
        keypoints_frame = KeyPointsData_list[i]['keypoints']
        keypoints_frame = np.array(keypoints_frame).reshape(-1,3)
        bbox_frame = KeyPointsData_list[i]['bbox']
        
        kp_overlay = OverlayKeyPoints(img_frame, keypoints_frame, keypoints_name)
        kp_overlay = DrawBoundingBox(kp_overlay, bbox_frame, color_default=(255,0,0))
        cv2.imshow("key points overlay", kp_overlay)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


    # Open test.json
    test_data = {}
    with open(train_file) as f:
        train_data = json.load(f)
        print("Hello, World!")