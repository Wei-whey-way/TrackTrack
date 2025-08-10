import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
from inference import get_model
import requests
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", message=".*CoreMLExecutionProvider.*")


class RFDETRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RFDETRBase()
        self.to_pil = transforms.ToPILImage() #Need image to be in PIL format
        self.resize = transforms.Resize((560,560))
        self.to_tensor = transforms.ToTensor() #See whether PIL can change to CHW (ch height width) dimensions instead

    def forward(self, images):
        results_batch = []
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print('Debugging images:', images.size())
         
        # images = [self.to_tensor(self.resize(self.to_pil(img.cpu()))) for img in images]
        pil_images = [
            self.resize(self.to_pil(img.cpu())).convert("RGB")
            for img in images
        ]

        detection_list = self.model.predict(pil_images,threshold=0.5)
        print('DEBUGGING: Detection list:', detection_list)
        print('xyxy', detection_list.xyxy)
        print('confi', detection_list.confidence)
        print('class_id', detection_list.class_id)


        #Get detections in YOLOX format
        for detections in detection_list:
            print('Check for type detections', type(detections), detections)
            
            # boxes = np.array(detections.xyxy, dtype=np.float32)
            # confidence = np.array(detections.confidence, dtype=np.float32)
            # class_id = np.array(detections.class_id, dtype=np.float32)
            
            # confi = detections.confidence
            # print('DEBUGGING: Confidence:', confi)

            # Convert to YOLOX format: [x1, y1, x2, y2, confidence, class_id]
            output = np.hstack([
                # boxes,
                # confidence[:,None],
                # class_id[:,None]
            ])

            print('DEBUGGING: Output:', output)

            results_batch.append(torch.tensor(output, dtype=torch.float32, device=device))

        return results_batch
