import random
import glob 
import os
from PIL import Image
import xml.etree.ElementTree as ET
root_dir = 'IDD_Detection'
pattern = f'{root_dir}/*/*/*/*.jpg'
image_list = glob.glob(pattern)
count = 0

classes = ['train', 'car' , 'bus' , 'truck' , 'bicycle' , 'autorickshaw'  , 'motorcycle' , 'traffic light' , 'traffic sign' ,'rider' , 'caravan' , 'trailer' , 'animal' , 'person' , 'vehicle fallback']
for file_path in image_list :
    count += 1
    split = file_path.split('/')
    xml_file_path = file_path.replace('JPEGImages','Annotations').replace('.jpg','.xml')
    new_file_name = split[-2] + split[-1]
    new_xml_name = new_file_name.replace('.jpg','.txt')
    if os.path.isfile(xml_file_path) == False :
        print('No Annotation found')
        continue
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    rand_val = random.randint(1,101)
    train_path_image = 'YOLO_NAS_IDD_Train/train/images'
    train_path_label = 'YOLO_NAS_IDD_Train/train/labels'
    test_path_image = 'YOLO_NAS_IDD_Train/test/images'
    test_path_label = 'YOLO_NAS_IDD_Train/test/labels'
    val_path_image = 'YOLO_NAS_IDD_Train/val/images'
    val_path_label = 'YOLO_NAS_IDD_Train/val/labels'
    image = Image.open(file_path)
    width , height = image.size 
    if rand_val <= 70 :
        with open(os.path.join(train_path_label , new_xml_name) , 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                cid = classes.index(name) + 1
                f.write(f'{cid} {xmin / width} {ymin / height} {(xmax - xmin)/width} {(ymax - ymin)/height}\n')
        f.close()
        os.system(f'cp {file_path} {os.path.join(train_path_image , new_file_name)}')
        print(f'{count} allocated to train')
    elif rand_val > 70 and rand_val < 90 :
        with open(os.path.join(test_path_label , new_xml_name) , 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                cid = classes.index(name) + 1
                f.write(f'{cid} {xmin / width} {ymin / height} {(xmax - xmin) /width} {(ymax - ymin)/height}\n')
        f.close()
        os.system(f'cp {file_path} {os.path.join(test_path_image , new_file_name)}')
        print(f'{count} allocated to test')
    else :
        with open(os.path.join(val_path_label , new_xml_name) , 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                cid = classes.index(name) + 1
                f.write(f'{cid} {xmin / width} {ymin / height} {(xmax - xmin) / width} {(ymax- ymin) / height}\n')
        f.close()
        os.system(f'cp {file_path} {os.path.join(val_path_image , new_file_name)}')
        print(f'{count} allocated to val')
        
        
    