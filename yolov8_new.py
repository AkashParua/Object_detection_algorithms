import os
from ultralytics import YOLO
import cv2
import torch

from memory_profiler import profile

@profile
def func():
    # Initialize the YOLO model
    model = YOLO('yolov8n.pt')
    for d in os.listdir('IDD_with_yolov8_subset/img'):
        # Set the path to the folder containing the images
        image_folder = os.path.join('IDD_with_yolov8_subset/img',d)

        # Set the path to the folder where the detection results will be saved
        output_folder = os.path.join('IDD_with_yolov8_subset/dt',d)

        # Ensure the output folder exists, create it if necessary
        os.makedirs(output_folder, exist_ok=True)

        # Get a list of image files in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

        infr = 0
        c = 0

        # Perform object detection on each image
        for image_file in image_files:
            # Load the image
            image_path = os.path.join(image_folder, image_file)
            image = image_path  # You can replace this with your own image loading code if necessary
            
            results = model.predict(source=image)

            for result in results:
                infr += result.speed['inference']
                c += 1

            fname = os.path.join(output_folder, image_file[:-4])+'.txt'
            open(fname, 'w').close()
            with open(fname, 'w') as f:
                for result in results:
                    bbox = result.boxes.xywh.tolist()
                    classid = result.boxes.cls.tolist()
                    conf = result.boxes.conf.tolist()
                    nm = result.names
                for i in range(len(bbox)):
                    s = str(nm[int(classid[i])])
                    if s in {'cat', 'dog', 'cow', 'elephant', 'horse', 'bear', 'sheep', 'bird'}:
                        f.write("animal ")
                    elif s in {"backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
                    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
                    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
                    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
                    "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
                    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
                    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"}:
                        f.write("other ")
                    else :
                        f.write(s.replace(' ','_') + ' ')
                
                    f.write(str((int(conf[i]*100))/100)+" ")
                    f.write(str((int(bbox[i][0]*100))/100 - (int(bbox[i][2]*100))/200)+' ')
                    f.write(str((int(bbox[i][1]*100))/100 - (int(bbox[i][3]*100))/200)+' ')
                    f.write(str((int(bbox[i][2]*100))/100)+' ')
                    f.write(str((int(bbox[i][3]*100))/100)) # print("%0.2f"%(0.174643))
                    if i!=(len(bbox)-1):
                        f.write('\n')
                    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Average Inference Time (for (640,640) on "+device+": "+str(infr/c)+"ms")

if __name__ == "__main__":
    func()