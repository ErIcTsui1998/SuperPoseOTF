import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.version.cuda
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

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
        if score >= 0.05:
            overlay = cv2.circle(overlay, (int(x),int(y)), radius=5, color=color_default, thickness=-1)
            overlay = cv2.putText(overlay, name, (int(x-20),int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color_default, 2, cv2.LINE_AA)
    return overlay

# Adding the dataset for PCH (Permanent Cautery Hook)

# the path to the dataset

dataset_path = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/"

# Folders containing the annotations and images
# inst_ids = ["PCH_1", "PCH_2", "PCH_E_1"]

# for inst_id in inst_ids:
#     filepath = dataset_path + inst_id
#     register_coco_instances(inst_id+"_train", {}, filepath + '/train.json', filepath)
#     register_coco_instances(inst_id+"_test", {}, filepath + '/test.json', filepath)
#     cur_metadata = MetadataCatalog.get(inst_id+"_train")
#     cur_metadata.set(thing_classes = ["PCH"])
#     cur_metadata.set(evaluator_type = 'coco')
#     cur_metadata.set(
#         keypoint_names = [
#             'LeftScrewBottom', 'LeftScrewTop', 'CentralScrew', 'StartHook',
#             'CentralHook', 'TipHook', 'RightScrewTop', 'RightScrewBottom'
#         ]
#     )
#     cur_metadata.set(
#         keypoint_flip_map = []
#     )
#     cur_metadata.set(
#         keypoint_connection_rules = [
#             ('LeftScrewBottom', 'LeftScrewTop', (0, 255, 0)), ('LeftScrewTop', 'CentralScrew', (0, 255, 0)),
#             ('RightScrewBottom', 'RightScrewTop', (0, 255, 0)), ('RightScrewTop', 'CentralScrew', (0, 255, 0)),
#             ('CentralScrew', 'StartHook', (0, 255, 0)), ('StartHook','CentralHook',(0, 255, 0)), 
#             ('CentralHook','TipHook', (0, 255, 0))
#         ]
#     )
#     cur_metadata.set(thing_colors=[(238, 130, 238)])


# Adding the dataset for FBF (Fenestrated Bipolar Forceps)

# the path to the dataset
# dataset_path = '/path_to_dataset/'
# dataset_path = '/home/' + os.environ.get('USERNAME') + '/dt2_dataset/kpts/'

# Folders containing the annotations and images
inst_ids = ["FBF_1", "FBF_2", "FBF_E_1"]

for inst_id in inst_ids:
    filepath = dataset_path + inst_id
    register_coco_instances(inst_id+"_train", {}, filepath + '/train.json', filepath)
    register_coco_instances(inst_id+"_test", {}, filepath + '/test.json', filepath)
    cur_metadata = MetadataCatalog.get(inst_id+"_train")
    cur_metadata.set(thing_classes = ["FBF"])
    cur_metadata.set(evaluator_type = 'coco')
    cur_metadata.set(
        keypoint_names = [
            'Head', 'Edge', 'Center', 'TipLeft', 'TipRight'
        ]
    )
    cur_metadata.set(
        keypoint_flip_map = []
    )
    cur_metadata.set(
        keypoint_connection_rules = [
            ('Head', 'Edge', (0, 255, 0)), ('Edge', 'Center', (0, 255, 0)),
            ('Center', 'TipLeft', (0, 255, 0)), ('Center', 'TipRight', (0, 255, 0))
        ]
    )
    cur_metadata.set(thing_colors=[(238, 130, 238)])


import matplotlib.pyplot as plt

dataset_dicts = DatasetCatalog.get(inst_ids[-1]+"_train")

for d in random.sample(dataset_dicts, 5):
    print(d['file_name'])
    img = cv2.imread(d['file_name'])
    visualizer = Visualizer(img, metadata=cur_metadata, scale=1)
    out = visualizer.draw_dataset_dict(d)
    
    cv2.imshow("key points", out.get_image())
    cv2.waitKey(0)
    # if cv2.waitKey(1000) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

from detectron2.engine import DefaultTrainer

# The configuration used for the model described in the paper.
# Feel free to tune it!
# More details and options are illustrated in 
# https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = tuple([inst_id+"_train" for inst_id in inst_ids])
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = [*range(int(cfg.SOLVER.MAX_ITER/10), cfg.SOLVER.MAX_ITER, int(cfg.SOLVER.MAX_ITER/10))]
cfg.SOLVER.GAMMA = 0.5
cfg.SOLVER.WARMUP_FACTOR = 1.0 / int(cfg.SOLVER.MAX_ITER/10)
cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER/10)
cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
cfg.OUTPUT_DIR = dataset_path + '/output/FBF_outputs' # the path where the model will be saved. Make sure to modify accordingly!
# cfg.OUTPUT_DIR = dataset_path + '/output/PCH_outputs' # the path where the model will be saved. Make sure to modify accordingly!
# for KeyPoints Detection
cfg.MODEL.RETINANET.NUM_CLASSES = 1

# 8 for PCH, 5 for FBF (Please comment/uncomment accordingly!)
# cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
# trainer.train()


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.DATASETS.TEST = tuple([inst_id+"_test" for inst_id in inst_ids])
# cfg.MODEL.WEIGHTS = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/sitl_dt2_kpt_pch.pth" # path to the model we just trained
# cfg.MODEL.WEIGHTS = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/sitl_dt2_kpt_fbf.pth" # path to the model we just trained
cfg.MODEL.WEIGHTS = "/home/zc519/Downloads/dVRK-si-dataset/annotated_dataset/instrument_keypoint/output/FBF_outputs/model_final.pth" 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
# 68%: 0.81, 95%: 0.61, 99%: 0.32
kpt_oks_sigmas = [0.32]*cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
cfg.TEST.KEYPOINT_OKS_SIGMAS = kpt_oks_sigmas

predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

dataset_dicts = DatasetCatalog.get(inst_ids[-1]+"_test")

for d in random.sample(dataset_dicts, 50):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im,
        metadata=cur_metadata, 
        scale=1.0,
    )
    name_list = v.metadata.get("keypoint_names") # The name list of keypoints
    # keypoints = outputs['instances'].pred_keypoints.to('cpu').numpy()[0] 

    # overlay = OverlayKeyPoints(im, keypoints, name_list)
    # cv2.imshow("after training", overlay)
    # if cv2.waitKey(1000) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("after training", out.get_image())
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
