# import torch, detectron2

# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.version.cuda

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def OverlayKeyPoints(img, keypointslist, KeypointsNameList, color_default=(0,255,0)):
    overlay = img.copy()
    n_pts = np.shape(keypointslist)[0]
    for i in range(n_pts):
        x,y,score = keypointslist[i]
        name = KeypointsNameList[i]
        if score >= 5e-3:
            overlay = cv2.circle(overlay, (int(x),int(y)), radius=5, color=color_default, thickness=-1)
            overlay = cv2.putText(overlay, name, (int(x-30),int(y-30)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.65, color_default, 2, cv2.LINE_AA)
    return overlay

class KeyPointsDetector:
    def __init__(self, instrument_type):
        # Instrument type 1: PCH (Permanent Cautery Hook)
        # Instrument type 2: FBF (Fenestrated Bipolar Forceps)
        self.IS_PCH = False
        self.IS_FBF = False
        self.predictor = None
        if instrument_type == "PCH":
            self.IS_PCH = True
            self.__PCH_init()
        elif instrument_type == "FBF":
            self.IS_FBF = True
            self.__FBF_init()
        pass

    def __FBF_init(self):
        # keypoint_names = ['Head', 'Edge', 'Center', 'TipLeft', 'TipRight']
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/output/FBF_outputs/model_final.pth" 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
        # 68%: 0.81, 95%: 0.61, 99%: 0.32
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
        kpt_oks_sigmas = [0.32]*cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS       
        cfg.TEST.KEYPOINT_OKS_SIGMAS = kpt_oks_sigmas
        self.predictor = DefaultPredictor(cfg)
    
    def __PCH_init(self):
        # keypoint_names = ['LeftScrewBottom', 'LeftScrewTop', 'CentralScrew', 'StartHook', 'CentralHook', 'TipHook', 'RightScrewTop', 'RightScrewBottom']
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/sitl_dt2_kpt_pch.pth" 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
        # 68%: 0.81, 95%: 0.61, 99%: 0.32
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
        kpt_oks_sigmas = [0.32]*cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS       
        cfg.TEST.KEYPOINT_OKS_SIGMAS = kpt_oks_sigmas
        self.predictor = DefaultPredictor(cfg)    

    def GetPredictor(self):
        return self.predictor
    
    def ExtractKeyPointsFromImg(self, img, prob_thres = 0e-2):
        outputs = self.predictor(img)
        output_keypoints = outputs['instances'].pred_keypoints.to('cpu').numpy()
        output_boxes = outputs['instances'].pred_boxes.to('cpu').tensor.detach().numpy()
        num_instance = output_keypoints.shape[0]
        if num_instance == 0:
            print("no keypoints discovered")
            return []
        else:
            KP_list = []
            for i in range(num_instance):
                for kp in output_keypoints[i]:
                    if kp[2] >= prob_thres:
                        u, v = int(kp[0]), int(kp[1])
                        KP_list.append((u,v))
            return KP_list

if __name__ == "__main__":
    image_dir = '/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/FBF_E_1/./images/'
    # image_dir = "/home/zc519/Downloads/dVRK-si-dataset/original_dataset/B_2/left/images"
    image_files = os.listdir(image_dir)
    image_files.sort()
    n_images = len(image_files)
    # N = 50 # randomly select 10 images
    N = n_images

    FBF = KeyPointsDetector("FBF")
    FBF_detector = FBF.GetPredictor()

    PCH = KeyPointsDetector("PCH")
    PCH_detector = PCH.GetPredictor()

    for i in range(N):
        index = random.randint(0, n_images-1)
        index = i
        os.chdir(image_dir)
        img = cv2.imread(image_files[index])

        FBF_KEYPOINTS = FBF.ExtractKeyPointsFromImg(img)
        PCH_KEYPOINTS = PCH.ExtractKeyPointsFromImg(img)

        FBF_keypoint_names = ['Head', 'Edge', 'Center', 'TipLeft', 'TipRight']
        PCH_keypoint_names = ['LeftScrewBottom', 'LeftScrewTop', 'CentralScrew', 'StartHook', 'CentralHook', 'TipHook', 'RightScrewTop', 'RightScrewBottom']

        FBF_outputs = FBF_detector(img)
        PCH_outputs = PCH_detector(img)
        
        overlay = img.copy()

        # For FBF key poitns
        FBF_output_keypoints = FBF_outputs['instances'].pred_keypoints.to('cpu').numpy()
        FBF_output_boxes = FBF_outputs['instances'].pred_boxes.to('cpu').tensor.detach().numpy()
        if FBF_output_keypoints.shape[0] == 0:
            print("no FBF keypoints discovered")
            # continue
        else:
            keypoints = FBF_output_keypoints 
            areas = np.prod(FBF_output_boxes[:, 2:] - FBF_output_boxes[:, :2], axis=1)
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            FBF_output_boxes = FBF_output_boxes[sorted_idxs] if FBF_output_boxes is not None else None
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
            n_instance = keypoints.shape[0]
            for j in range(n_instance):
                overlay = OverlayKeyPoints(overlay, keypoints[j], FBF_keypoint_names)
        
        # For PCH key points
        PCH_output_keypoints = PCH_outputs['instances'].pred_keypoints.to('cpu').numpy()
        PCH_output_boxes = PCH_outputs['instances'].pred_boxes.to('cpu').tensor.detach().numpy()
        if PCH_output_keypoints.shape[0] == 0:
            print("no PCH keypoints discovered")
            # continue
        else:
            keypoints = PCH_output_keypoints 
            areas = np.prod(PCH_output_boxes[:, 2:] - PCH_output_boxes[:, :2], axis=1)
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            PCH_output_boxes = PCH_output_boxes[sorted_idxs] if PCH_output_boxes is not None else None
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
            n_instance = keypoints.shape[0]
            for j in range(n_instance):
                overlay = OverlayKeyPoints(overlay, keypoints[j], PCH_keypoint_names)

        cv2.imshow("overlay", overlay)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        pass
    print("Test")