from PIL import Image
import math
import os
valid_classes = {'bear' , 'bicycle' , 'bus' ,  'car' ,  'cat',  'cow' ,  'dog' ,  'elephant' , 'fire_hydrant' , 'motorcycle' ,  'person' ,'sheep' , 'stop_sign' , 'traffic_light' , 'train'  ,'truck'}
def extract_bounding_box(image_path , annotation_path):
    lines = []
    with open(annotation_path ,'r') as f :
        lines = f.readlines()
        f.close()
    for line in lines :
        #print(line + '-')
        if line.split()[0] in valid_classes :
            save_file_name = image_path.replace('val2017/','').replace('.jpg','')+'_'+line.split()[0]+'_'+'_'.join(line.split()[1:])+'.jpg'
            print(save_file_name)
            save_path = os.path.join(line.split()[0],save_file_name)
            #print(save_path)
            x,y,w,h = list(map(lambda a : int(float(a)) , line.split()[1:]))  #geting xmin , ymin ,width ,height
            #print(x,y,w,h)
            image = Image.open(image_path)
            region_of_interest = image.crop((x, y, x + w, y + h))
            
            new_image = Image.new("RGB", (w, h))
            new_image.paste(region_of_interest, (0, 0))
            new_image.save(save_path)
            
            


annotation_root = 'val2017a'      #annotation path (.txt file with class_name <xmin> <ymin> <height> <width>)
image_root = 'val2017'            #image path (path to the image datset)
import glob
image_path_pattern = 'val2017/*.jpg'
image_paths = glob.glob(image_path_pattern)
annotation_paths = [p.replace('val2017','val2017a').replace('jpg','txt') for p in image_paths]

for image_path , annotation_path in zip(image_paths , annotation_paths):
    extract_bounding_box(image_path , annotation_path)
    






'''
# Load the image
image = Image.open("original_image.jpg")

# Define the bounding box coordinates
x, y, w, h = 100, 100, 200, 200

# Extract the region of interest
region_of_interest = image.crop((x, y, x + w, y + h))

# Create a new image from the region of interest
new_image = Image.new("RGB", (w, h))
new_image.paste(region_of_interest, (0, 0))

# Save the new image
new_image.save("extracted_image.jpg")

# Display the new image
new_image.show()
'''