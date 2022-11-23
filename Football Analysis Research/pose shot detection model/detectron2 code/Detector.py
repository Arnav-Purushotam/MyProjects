import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.config import cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import cv2
import numpy
import time



class Detector:
    
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()
        
        #object detection
        if model_type == "OD":
            #load model config and pretrained model
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        #instance segmentation
        elif model_type == "IS":
            
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
         
        #keypoint detection
        elif model_type == "KP":
            
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
            
        
            
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
        
    
    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)
        
        
        for det_keypoints in predictions["instances"].pred_keypoints:
            person_keypoint={
                "date_time":time.ctime(),
                "nose":{'x':str(det_keypoints.cpu().numpy()[0][0]),'y': str(det_keypoints.cpu().numpy()[0][1]), 'conf': str(det_keypoints.cpu().numpy()[0][2])},
                "left_eye":{'x':str(det_keypoints.cpu().numpy()[1][0]),'y': str(det_keypoints.cpu().numpy()[1][1]), 'conf': str(det_keypoints.cpu().numpy()[1][2])},
                "right_eye":{'x':str(det_keypoints.cpu().numpy()[2][0]),'y': str(det_keypoints.cpu().numpy()[2][1]), 'conf': str(det_keypoints.cpu().numpy()[2][2])},
                "left_ear":{'x':str(det_keypoints.cpu().numpy()[3][0]),'y': str(det_keypoints.cpu().numpy()[3][1]), 'conf': str(det_keypoints.cpu().numpy()[3][2])},
                "right_ear":{'x':str(det_keypoints.cpu().numpy()[4][0]),'y': str(det_keypoints.cpu().numpy()[4][1]), 'conf': str(det_keypoints.cpu().numpy()[4][2])},
                "left_shoulder":{'x':str(det_keypoints.cpu().numpy()[5][0]),'y': str(det_keypoints.cpu().numpy()[5][1]), 'conf': str(det_keypoints.cpu().numpy()[5][2])},
                "right_shoulder":{'x':str(det_keypoints.cpu().numpy()[6][0]),'y': str(det_keypoints.cpu().numpy()[6][1]), 'conf': str(det_keypoints.cpu().numpy()[6][2])},
                "left_elbow":{'x':str(det_keypoints.cpu().numpy()[7][0]),'y': str(det_keypoints.cpu().numpy()[7][1]), 'conf': str(det_keypoints.cpu().numpy()[7][2])},
                "right_elbow":{'x':str(det_keypoints.cpu().numpy()[8][0]),'y': str(det_keypoints.cpu().numpy()[8][1]), 'conf': str(det_keypoints.cpu().numpy()[8][2])},
                "left_wrist":{'x':str(det_keypoints.cpu().numpy()[9][0]),'y': str(det_keypoints.cpu().numpy()[9][1]), 'conf': str(det_keypoints.cpu().numpy()[9][2])},
                "right_wrist":{'x':str(det_keypoints.cpu().numpy()[10][0]),'y': str(det_keypoints.cpu().numpy()[10][1]), 'conf': str(det_keypoints.cpu().numpy()[10][2])},
                "left_hip":{'x':str(det_keypoints.cpu().numpy()[11][0]),'y': str(det_keypoints.cpu().numpy()[11][1]), 'conf': str(det_keypoints.cpu().numpy()[11][2])},
                "right_hip":{'x':str(det_keypoints.cpu().numpy()[12][0]),'y': str(det_keypoints.cpu().numpy()[12][1]), 'conf': str(det_keypoints.cpu().numpy()[12][2])},
                "left_knee":{'x':str(det_keypoints.cpu().numpy()[13][0]),'y': str(det_keypoints.cpu().numpy()[13][1]), 'conf': str(det_keypoints.cpu().numpy()[13][2])},
                "right_knee":{'x':str(det_keypoints.cpu().numpy()[14][0]),'y': str(det_keypoints.cpu().numpy()[14][1]), 'conf': str(det_keypoints.cpu().numpy()[14][2])},
                "left_ankle":{'x':str(det_keypoints.cpu().numpy()[15][0]),'y': str(det_keypoints.cpu().numpy()[15][1]), 'conf': str(det_keypoints.cpu().numpy()[15][2])},
                "right_ankle":{'x':str(det_keypoints.cpu().numpy()[16][0]),'y': str(det_keypoints.cpu().numpy()[16][1]), 'conf': str(det_keypoints.cpu().numpy()[16][2])}
                }

        
        
        
        viz = Visualizer(image[:, :, ::-1], metadata= MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)
        
        
        
        
        

        
        
detector = Detector("KP")

detector.onImage(r"D:\dev\projects\football AR\testing\shots\shot6_frames\6.jpg")


# link for the lstm model:
# https://github.com/spmallick/learnopencv/tree/master/Human-Action-Recognition-Using-Detectron2-And-Lstm




