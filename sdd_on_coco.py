import torch
import torchvision
from torchvision import transforms as T
import os
from PIL import Image
from memory_profiler import profile
import time

model = torchvision.models.detection.ssd300_vgg16(pretrained = True)

@profile
def func():
    infr = 0
    c = 0
    for d in os.listdir('val2017'):
        src = os.path.join('val2017' , d)
        model.eval()
        print(src)
        
        ig = Image.open(src)
        width, height = ig.size
        ig = ig.resize((640,640))
        resized_width, resized_height = ig.size
        scale_x = width / resized_width
        scale_y = height / resized_height

        
        transform = T.ToTensor()
        img = transform(ig)

        with torch.no_grad():
            start = time.time()
            pred = model([img])
            infr += time.time()-start
            c+=1

        bboxes, scores, labels = pred[0]["boxes"], pred[0]["scores"], pred[0]["labels"]

        num = torch.argwhere(scores>0.25).shape[0]

        coco_names = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        vehicle = {'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck'}
        
        traffic = {'traffic light', 'stop sign', 'parking meter'}

        animal = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear'}

        
        for i,st in enumerate(coco_names):
            if st in vehicle:
                coco_names[i] = "vehicle"
            elif st in animal:
                coco_names[i] = "animal"
            elif st == 'person':
                coco_names[i] = "person"
            elif st in traffic:
                coco_names[i] = "traffic"
            else:
                coco_names[i] = "other"

            
        name = d.replace('.jpg' , '.txt')
        fname = os.path.join('COCO_with_SSD2', name)
        open(fname, 'w').close()
        with open(fname, 'w') as f:
            for i in range(num):
                class_name = coco_names[labels.numpy()[i]]
                if class_name == 'N/A':
                    continue
                f.write(str(class_name).replace(' ','_')+" ")

                conf = scores.numpy()[i]
                f.write(str(conf)+" ")

                x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
                f.write(str(x1 * scale_x)+" "+str(y1 * scale_y)+" "+str((x2-x1)*scale_x)+" "+str((y2-y1)*scale_y))
                if i!=num-1:
                    f.write("\n")
    print("Average Inference Time: "+str(infr/c)+"s")
                    
if __name__ == '__main__':
    func()