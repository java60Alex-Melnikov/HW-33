from ultralytics import YOLO
import pandas as pd
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