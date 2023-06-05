import torch
from PIL import Image
import os
from super_gradients.training import models
import time
from memory_profiler import profile
import numpy as np
yolo_nas_l = models.get("yolo_nas_s", pretrained_weights="coco")

model = yolo_nas_l.to("cuda" if torch.cuda.is_available() else "cpu")

input_size = [1, 3, 640, 640]  # [Batch x Channels x Width x Height]
dummy_input = torch.from_numpy(np.random.randn(*input_size)).float()

torch.onnx.export(model, dummy_input,  "yolo_nas_s.onnx")


def yolonas(src, t):       
    image = Image.open(src)
    resized_image = image.resize((640,640))
    start = time.time()
    out = model.predict(resized_image)
    tt = t + (time.time() - start)
    
    width, height = image.size
    resized_width, resized_height = resized_image.size
    scale_x = width / resized_width
    scale_y = height / resized_height

    prediction_objects = list(out._images_prediction_lst)[0]
    bbox = prediction_objects.prediction.bboxes_xyxy

    int_labels = prediction_objects.prediction.labels.astype(int)
    class_names = prediction_objects.class_names
    nm = [class_names[i] for i in int_labels]

    conf = prediction_objects.prediction.confidence.astype(float)
   
    fname = f'yolo_nas_idd/'+src[13:-4]+'.txt'
    open(fname, 'w').close()
    with open(fname, 'w') as f:
        for i in range(len(bbox)):
            s = str(nm[i])
            #f.write(s.replace(' ','_')+" ")
            if s in {'cat', 'dog', 'cow', 'elephant', 'horse', 'bear', 'sheep', 'bird'}:
                f.write("animal ")
            elif s in {"laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
	                "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
                    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
	                "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
	                "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
	                "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
	                "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv"}:
                f.write("other ")
            else :
                f.write(s.replace(' ','_')+" ")
            f.write(str((int(conf[i]*100))/100)+" ")
            f.write(str((int(bbox[i][0]*100))/100 * scale_x ) +" ")
            f.write(str((int(bbox[i][1]*100))/100 * scale_y )+" ")
            f.write(str((int((bbox[i][2]-bbox[i][0])*100))/100 * scale_x)+" ")
            f.write(str((int((bbox[i][3]-bbox[i][1])*100))/100 * scale_y))
            if i!=(len(bbox)-1):
                f.write('\n')
    
    return tt



'''
@profile
def func():
    directory = "val2017"
    t = 0
    # Loop over files in the directory
    for filename in os.listdir(directory):
        # Get the full path of the file
        file_path = os.path.join(directory, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Print the file path
            t = yolonas(file_path, t)
    print("Average Inference Time: "+str(t/5000)+"ms")
        
if __name__=="__main__":
    func()
'''

@profile
def func():
    root_directory = 'IDD_val_data'
    #loop over directories in the diectory
    for d in os.listdir(root_directory):
        t = 0
        directory = os.path.join(root_directory , d)
        l = len(os.listdir(directory))
        print(l)
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory,filename)
            if os.path.isfile(file_path):
                print(file_path)
                t = yolonas(file_path, t)
        print(f'avg time for {directory} is {t/l} ms')
        

if __name__=="__main__":
    func()
    
