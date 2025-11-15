from ultralytics import YOLO
import pandas as pd
import numpy as np
class ImageInfo :
    def __init__(self, pathImage):
        self.model = YOLO('yolov8m-seg.pt')
        self.results = self.model(pathImage)[0]
        self.boxes = self.results.boxes
        self.allNames = self.results.names
    
    def boxesClass(self, class_name):
        cls = self.boxes.cls.cpu().numpy()
        indices = [i for i, c in enumerate(cls) if self.allNames[c] == class_name]
        return indices
    
    def boxInfo(self, index):
        xyxy = self.boxes.xyxy.cpu().numpy()[index]
        conf = self.boxes.conf.cpu().numpy()[index]
        cls = self.boxes.cls.cpu().numpy()[index]
        class_name = self.allNames[cls]
        return (*xyxy, conf, class_name)
    
    def dataFrame(self):
        xyxy = self.boxes.xyxy.cpu().numpy()
        cls = self.boxes.cls.cpu().numpy()
        conf = self.boxes.conf.cpu().numpy()
        names = [self.allNames[c] for c in cls]
        df = pd.DataFrame(xyxy, columns=["xmin", "ymin", "xmax", "ymax"])
        df["name"] = names
        df["confidence"] = conf
        return df
    
    def suitcaseHandbagPerson(self, threshold):
        xywhn = self.boxes.xywhn.cpu().numpy()
        cls = self.boxes.cls.cpu().numpy()
        sh_indices = [i for i, c in enumerate(cls) if self.allNames[c] in ['suitcase', 'handbag']]
        person_indices = [i for i, c in enumerate(cls) if self.allNames[c] == 'person']
        
        def find_nearest_person(sh_idx):
            sh_center = xywhn[sh_idx][:2]
            distances = [(p_idx, np.sqrt((sh_center[0] - xywhn[p_idx][0])**2 + (sh_center[1] - xywhn[p_idx][1])**2)) for p_idx in person_indices]
            if not distances:
                return None
            nearest_person_idx, min_distance = min(distances, key=lambda x: x[1])
            return (nearest_person_idx, min_distance) if min_distance <= threshold else None
        
        return {sh_idx: find_nearest_person(sh_idx) for sh_idx in sh_indices}