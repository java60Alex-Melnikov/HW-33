from ultralytics import YOLO
class ImageInfo :
    def __init__(self, pathImage):
        self.model = YOLO('yolov8m-seg.pt')
        self.results = self.model(pathImage)[0]
        self.boxes = self.results.boxes
        self.allNames = self.results.names