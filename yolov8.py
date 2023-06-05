from ultralytics import YOLO
import cv2
import torch
import os
from memory_profiler import profile

@profile
def func():
    model = YOLO("yolov8n.pt")

    src = "000000000138.jpg"
    results = model.predict(device="0", source=src)
    fname = src[:-4]+'.txt'
    open(fname, 'w').close()
    with open(fname, 'w') as f:
        for result in results:
            bbox = result.boxes.xywh.tolist()
            classid = result.boxes.cls.tolist()
            conf = result.boxes.conf.tolist()
            nm = result.names
        for i in range(len(bbox)):
            s = str(nm[int(classid[i])])
            f.write(s.replace(' ','_')+" ")
            f.write(str((int(conf[i]*100))/100)+" ")
            for j in range(len(bbox[i])):
                if j==(len(bbox[i])-1):
                    f.write(str((int(bbox[i][j]*100))/100))
                else:
                    f.write(str((int(bbox[i][j]*100))/100)+" ")
            if i!=(len(bbox)-1):
                f.write('\n')

    # when NVIDIA GPU is available (although it automatically uses GPU with above line when available):
    #results = model.predict(device="0",source="zidane.jpg", show=True, save=True)

if __name__ == "__main__":
    func()