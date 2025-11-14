from ultralytics import YOLO
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